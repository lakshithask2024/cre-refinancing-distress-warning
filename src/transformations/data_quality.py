"""
Data Quality Gate — Bronze → Silver Validation
=================================================

Implements automated data quality checks at the bronze-to-silver transition.
Pipeline halts (raises DataQualityError) on critical failures.

Checks:
1. Row count > 0
2. Null rate < 5% on critical fields
3. No duplicate loan_ids
4. Value ranges within expected bounds
5. Schema completeness (all required columns present)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data quality checks fail at a critical level."""

    def __init__(self, failures: list[str]):
        self.failures = failures
        msg = f"Data quality gate FAILED with {len(failures)} critical issue(s):\n"
        msg += "\n".join(f"  - {f}" for f in failures)
        super().__init__(msg)


class DataQualityReport:
    """Structured report from a data quality check run."""

    def __init__(self) -> None:
        self.checks_passed: list[str] = []
        self.checks_warned: list[str] = []
        self.checks_failed: list[str] = []

    @property
    def passed(self) -> bool:
        return len(self.checks_failed) == 0

    def add_pass(self, check: str) -> None:
        self.checks_passed.append(check)

    def add_warning(self, check: str) -> None:
        self.checks_warned.append(check)

    def add_failure(self, check: str) -> None:
        self.checks_failed.append(check)

    def summary(self) -> str:
        lines = [
            f"Data Quality Report: {len(self.checks_passed)} PASS, "
            f"{len(self.checks_warned)} WARN, {len(self.checks_failed)} FAIL"
        ]
        if self.checks_failed:
            lines.append("  FAILURES:")
            for f in self.checks_failed:
                lines.append(f"    ✗ {f}")
        if self.checks_warned:
            lines.append("  WARNINGS:")
            for w in self.checks_warned:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


# ─── Loan Data Quality Gate ───────────────────────────────────────────────────

LOAN_REQUIRED_COLUMNS = [
    "loan_id",
    "origination_date",
    "maturity_date",
    "original_balance",
    "current_balance",
    "note_rate",
    "property_type",
    "metro_area",
    "ltv_at_origination",
    "dscr_at_origination",
    "occupancy_pct",
    "noi_annual",
]

LOAN_CRITICAL_FIELDS = ["loan_id", "maturity_date", "original_balance"]


def validate_bronze_loans(
    records: list[dict[str, Any]],
    max_null_rate: float = 0.05,
    halt_on_failure: bool = True,
) -> DataQualityReport:
    """
    Validate bronze loan records before silver transformation.

    Args:
        records: List of loan record dicts
        max_null_rate: Maximum acceptable null rate for critical fields (0.05 = 5%)
        halt_on_failure: If True, raise DataQualityError on critical failures

    Returns:
        DataQualityReport

    Raises:
        DataQualityError: If halt_on_failure=True and critical checks fail
    """
    report = DataQualityReport()

    # Check 1: Non-empty dataset
    if len(records) == 0:
        report.add_failure("EMPTY_DATASET: Bronze loans table has 0 records")
        if halt_on_failure:
            raise DataQualityError(report.checks_failed)
        return report
    report.add_pass(f"ROW_COUNT: {len(records)} records (> 0)")

    # Check 2: Required columns present
    if records:
        sample = records[0]
        missing_cols = [c for c in LOAN_REQUIRED_COLUMNS if c not in sample]
        if missing_cols:
            report.add_failure(
                f"MISSING_COLUMNS: Required columns missing: {missing_cols}"
            )
        else:
            report.add_pass(f"SCHEMA: All {len(LOAN_REQUIRED_COLUMNS)} required columns present")

    # Check 3: Null rates on critical fields
    total = len(records)
    for field in LOAN_CRITICAL_FIELDS:
        null_count = sum(1 for r in records if not r.get(field))
        null_rate = null_count / total
        if null_rate > max_null_rate:
            report.add_failure(
                f"NULL_RATE: {field} has {null_rate:.1%} nulls "
                f"(threshold: {max_null_rate:.1%})"
            )
        elif null_count > 0:
            report.add_warning(
                f"NULL_RATE: {field} has {null_count} nulls ({null_rate:.2%})"
            )
        else:
            report.add_pass(f"NULL_RATE: {field} has 0 nulls")

    # Check 4: No duplicate loan_ids
    loan_ids = [r.get("loan_id") for r in records if r.get("loan_id")]
    unique_count = len(set(loan_ids))
    dupe_count = len(loan_ids) - unique_count
    if dupe_count > 0:
        dupe_rate = dupe_count / total
        if dupe_rate > 0.01:  # > 1% duplicates is critical
            report.add_failure(
                f"DUPLICATES: {dupe_count} duplicate loan_ids ({dupe_rate:.2%})"
            )
        else:
            report.add_warning(
                f"DUPLICATES: {dupe_count} duplicate loan_ids ({dupe_rate:.2%})"
            )
    else:
        report.add_pass(f"UNIQUENESS: All {len(loan_ids)} loan_ids are unique")

    # Check 5: Value range checks
    _check_range(records, "ltv_at_origination", 0.0, 2.0, report, critical=False)
    _check_range(records, "dscr_at_origination", 0.0, 10.0, report, critical=False)
    _check_range(records, "note_rate", 0.0, 0.20, report, critical=False)
    _check_range(records, "occupancy_pct", 0.0, 1.0, report, critical=False)
    _check_range(records, "original_balance", 0, 1e10, report, critical=False)

    # Check 6: Property type values valid
    valid_types = {"office", "retail", "industrial", "multifamily", "hotel"}
    actual_types = set(r.get("property_type", "").lower() for r in records)
    invalid_types = actual_types - valid_types - {""}
    if invalid_types:
        report.add_warning(f"PROPERTY_TYPE: Unexpected values: {invalid_types}")
    else:
        report.add_pass(f"PROPERTY_TYPE: All values in expected set")

    # Log and potentially halt
    logger.info(report.summary())

    if halt_on_failure and not report.passed:
        raise DataQualityError(report.checks_failed)

    return report


# ─── Market Data Quality Gate ─────────────────────────────────────────────────

def validate_bronze_market(
    records: list[dict[str, Any]],
    halt_on_failure: bool = True,
) -> DataQualityReport:
    """
    Validate bronze market records before silver transformation.

    Args:
        records: List of market record dicts
        halt_on_failure: If True, raise DataQualityError on critical failures

    Returns:
        DataQualityReport
    """
    report = DataQualityReport()

    # Check 1: Non-empty
    if len(records) == 0:
        report.add_failure("EMPTY_DATASET: Bronze market table has 0 records")
        if halt_on_failure:
            raise DataQualityError(report.checks_failed)
        return report
    report.add_pass(f"ROW_COUNT: {len(records)} records (> 0)")

    # Check 2: All expected data types present
    expected_types = {"treasury_10y", "treasury_5y", "cap_rate", "sofr", "cre_price_index"}
    actual_types = set(r.get("data_type", "") for r in records)
    missing_types = expected_types - actual_types
    if missing_types:
        report.add_failure(f"MISSING_TYPES: Expected data types missing: {missing_types}")
    else:
        report.add_pass(f"DATA_TYPES: All {len(expected_types)} expected types present")

    # Check 3: Values are numeric and non-null
    null_values = sum(1 for r in records if r.get("value") is None)
    if null_values > 0:
        null_rate = null_values / len(records)
        if null_rate > 0.05:
            report.add_failure(f"NULL_VALUES: {null_values} records have null value ({null_rate:.1%})")
        else:
            report.add_warning(f"NULL_VALUES: {null_values} records have null value ({null_rate:.2%})")
    else:
        report.add_pass("VALUES: All records have non-null value")

    # Check 4: Rate values in reasonable range (0-20%)
    rate_types = {"treasury_10y", "treasury_5y", "sofr", "cap_rate"}
    rate_records = [r for r in records if r.get("data_type") in rate_types]
    out_of_range = [
        r for r in rate_records
        if r.get("value") is not None and (float(r["value"]) < -1.0 or float(r["value"]) > 20.0)
    ]
    if out_of_range:
        report.add_warning(f"VALUE_RANGE: {len(out_of_range)} rate records outside [-1, 20]%")
    else:
        report.add_pass("VALUE_RANGE: All rate values in [-1, 20]% range")

    # Check 5: Sufficient history (at least 12 months of treasury data)
    treasury_records = [r for r in records if r.get("data_type") == "treasury_10y"]
    if len(treasury_records) < 12:
        report.add_warning(f"HISTORY: Only {len(treasury_records)} treasury_10y records (want >= 12)")
    else:
        report.add_pass(f"HISTORY: {len(treasury_records)} treasury_10y records (sufficient)")

    logger.info(report.summary())

    if halt_on_failure and not report.passed:
        raise DataQualityError(report.checks_failed)

    return report


# ─── Silver Output Validation ─────────────────────────────────────────────────

def validate_silver_features(
    records: list[dict[str, Any]],
    halt_on_failure: bool = True,
) -> DataQualityReport:
    """Validate silver_loan_features output."""
    report = DataQualityReport()

    if len(records) == 0:
        report.add_failure("EMPTY_OUTPUT: silver_loan_features has 0 records")
        if halt_on_failure:
            raise DataQualityError(report.checks_failed)
        return report
    report.add_pass(f"ROW_COUNT: {len(records)} enriched records")

    # Check derived features are present
    derived_fields = [
        "current_value", "current_ltv", "refinance_rate",
        "rate_gap", "new_dscr", "debt_yield", "months_to_maturity",
    ]
    if records:
        missing = [f for f in derived_fields if f not in records[0]]
        if missing:
            report.add_failure(f"MISSING_FEATURES: Derived fields missing: {missing}")
        else:
            report.add_pass(f"FEATURES: All {len(derived_fields)} derived features present")

    # Check no unexpected nulls in key derived features
    for field in ["current_ltv", "refinance_rate", "new_dscr", "debt_yield"]:
        null_count = sum(1 for r in records if r.get(field) is None)
        if null_count > 0:
            rate = null_count / len(records)
            if rate > 0.05:
                report.add_failure(f"DERIVED_NULLS: {field} has {null_count} nulls ({rate:.1%})")
            else:
                report.add_warning(f"DERIVED_NULLS: {field} has {null_count} nulls ({rate:.2%})")
        else:
            report.add_pass(f"DERIVED_OK: {field} — 0 nulls")

    # Check value reasonableness
    _check_range(records, "current_ltv", 0.0, 5.0, report, critical=False)
    _check_range(records, "new_dscr", 0.0, 50.0, report, critical=False)
    _check_range(records, "refinance_rate", 0.0, 0.20, report, critical=False)

    logger.info(report.summary())

    if halt_on_failure and not report.passed:
        raise DataQualityError(report.checks_failed)

    return report


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_range(
    records: list[dict[str, Any]],
    field: str,
    min_val: float,
    max_val: float,
    report: DataQualityReport,
    critical: bool = False,
) -> None:
    """Check that a field's values fall within an expected range."""
    values = [r[field] for r in records if r.get(field) is not None]
    if not values:
        return

    out_of_range = [v for v in values if v < min_val or v > max_val]
    if out_of_range:
        pct = len(out_of_range) / len(values) * 100
        msg = f"RANGE({field}): {len(out_of_range)} values ({pct:.1f}%) outside [{min_val}, {max_val}]"
        if critical:
            report.add_failure(msg)
        else:
            report.add_warning(msg)
    else:
        report.add_pass(f"RANGE({field}): all {len(values)} values in [{min_val}, {max_val}]")
