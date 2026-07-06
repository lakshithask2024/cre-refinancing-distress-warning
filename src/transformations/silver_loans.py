"""
Silver Layer — Loan Cleaning & Validation
============================================

Reads bronze_loans Delta table and produces silver_loans with:
- Null handling: drop rows missing critical fields (loan_id, maturity_date);
  impute occupancy with metro/property-type median
- Type casting: ensure correct types (dates as ISO strings, amounts as float)
- Outlier flagging: z-score flags for LTV > 1.0 or DSCR < 0.5
- Deduplication: remove duplicate loan_ids (keep first)
- Derived fields: months_to_maturity (from current date)

Input:  data/bronze/loans (Delta)
Output: data/silver/loans (Delta)
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Critical fields — rows missing these are dropped
CRITICAL_FIELDS = ["loan_id", "maturity_date", "origination_date", "original_balance"]

# Fields that should be numeric
NUMERIC_FIELDS = [
    "original_balance",
    "current_balance",
    "note_rate",
    "ltv_at_origination",
    "dscr_at_origination",
    "occupancy_pct",
    "noi_annual",
    "property_value_at_origination",
]


def transform_silver_loans(
    bronze_records: list[dict[str, Any]],
    reference_date: date | None = None,
) -> list[dict[str, Any]]:
    """
    Transform bronze loan records into silver loan records.

    Steps:
    1. Drop rows missing critical fields
    2. Deduplicate by loan_id
    3. Cast types (ensure numeric fields are float)
    4. Impute occupancy nulls with metro/property_type median
    5. Flag outliers (LTV > 1.0, DSCR < 0.5)
    6. Compute months_to_maturity

    Args:
        bronze_records: Raw loan records from bronze layer
        reference_date: Date for months_to_maturity calc (default: today)

    Returns:
        Cleaned and enriched silver loan records
    """
    if reference_date is None:
        reference_date = date.today()

    logger.info(f"Silver loans: processing {len(bronze_records)} bronze records")

    # Step 1: Drop rows missing critical fields
    records = _drop_missing_critical(bronze_records)
    logger.info(f"  After critical-field filter: {len(records)} records")

    # Step 2: Deduplicate by loan_id
    records = _deduplicate(records)
    logger.info(f"  After deduplication: {len(records)} records")

    # Step 3: Type casting
    records = [_cast_types(r) for r in records]

    # Step 4: Impute occupancy
    records = _impute_occupancy(records)

    # Step 5: Flag outliers
    records = [_flag_outliers(r) for r in records]

    # Step 6: Compute months_to_maturity
    records = [_compute_months_to_maturity(r, reference_date) for r in records]

    # Step 7: Add silver metadata
    silver_ts = datetime.utcnow().isoformat()
    for r in records:
        r["_silver_processed_at"] = silver_ts

    logger.info(f"Silver loans: output {len(records)} cleaned records")
    return records


def _drop_missing_critical(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop records missing critical fields."""
    valid = []
    dropped = 0
    for r in records:
        missing = [f for f in CRITICAL_FIELDS if not r.get(f)]
        if missing:
            dropped += 1
        else:
            valid.append(r)
    if dropped > 0:
        logger.warning(f"  Dropped {dropped} records with missing critical fields")
    return valid


def _deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate loan_ids, keeping first occurrence."""
    seen: set[str] = set()
    unique = []
    dupes = 0
    for r in records:
        lid = r["loan_id"]
        if lid in seen:
            dupes += 1
        else:
            seen.add(lid)
            unique.append(r)
    if dupes > 0:
        logger.warning(f"  Removed {dupes} duplicate loan_ids")
    return unique


def _cast_types(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure numeric fields are proper floats."""
    r = dict(record)
    for field in NUMERIC_FIELDS:
        val = r.get(field)
        if val is not None:
            try:
                r[field] = float(val)
            except (ValueError, TypeError):
                r[field] = None
    # Ensure boolean
    if "balloon_flag" in r:
        r["balloon_flag"] = bool(r["balloon_flag"])
    # Ensure loan_term_years is int
    if "loan_term_years" in r:
        try:
            r["loan_term_years"] = int(r["loan_term_years"])
        except (ValueError, TypeError):
            pass
    # Ensure origination_year is int
    if "origination_year" in r:
        try:
            r["origination_year"] = int(r["origination_year"])
        except (ValueError, TypeError):
            pass
    return r


def _impute_occupancy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Impute null/missing occupancy with metro+property_type median."""
    # Build medians by (metro, property_type)
    group_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        occ = r.get("occupancy_pct")
        if occ is not None and occ > 0:
            key = (r.get("metro_area", ""), r.get("property_type", ""))
            group_values[key].append(occ)

    # Compute medians
    group_medians: dict[tuple[str, str], float] = {}
    for key, values in group_values.items():
        group_medians[key] = statistics.median(values)

    # Global fallback median
    all_occ = [r["occupancy_pct"] for r in records if r.get("occupancy_pct") and r["occupancy_pct"] > 0]
    global_median = statistics.median(all_occ) if all_occ else 0.85

    # Impute
    imputed_count = 0
    for r in records:
        occ = r.get("occupancy_pct")
        if occ is None or occ <= 0:
            key = (r.get("metro_area", ""), r.get("property_type", ""))
            r["occupancy_pct"] = group_medians.get(key, global_median)
            r["_occupancy_imputed"] = True
            imputed_count += 1
        else:
            r["_occupancy_imputed"] = False

    if imputed_count > 0:
        logger.info(f"  Imputed occupancy for {imputed_count} records")

    return records


def _flag_outliers(record: dict[str, Any]) -> dict[str, Any]:
    """Flag outlier values for LTV and DSCR."""
    r = dict(record)

    ltv = r.get("ltv_at_origination")
    dscr = r.get("dscr_at_origination")

    # LTV > 1.0 is an outlier (over-leveraged)
    r["_ltv_outlier"] = bool(ltv is not None and ltv > 1.0)

    # DSCR < 0.5 is an outlier (severely under-covered)
    r["_dscr_outlier"] = bool(dscr is not None and dscr < 0.5)

    # Combined outlier flag
    r["_is_outlier"] = r["_ltv_outlier"] or r["_dscr_outlier"]

    return r


def _compute_months_to_maturity(
    record: dict[str, Any], reference_date: date
) -> dict[str, Any]:
    """Compute months remaining until maturity."""
    r = dict(record)
    mat_str = r.get("maturity_date", "")
    try:
        mat_date = date.fromisoformat(mat_str)
        delta_days = (mat_date - reference_date).days
        r["months_to_maturity"] = round(delta_days / 30.44, 1)  # avg days per month
        r["is_matured"] = delta_days <= 0
    except (ValueError, TypeError):
        r["months_to_maturity"] = None
        r["is_matured"] = None
    return r
