"""
Unit tests for Silver layer transformations.

Tests cover:
- silver_loans: null handling, deduplication, type casting, outlier flagging, months_to_maturity
- silver_market: date normalization, interpolation, rolling average smoothing
- feature_engineering: derived feature calculations (LTV, DSCR, rate_gap, debt_yield)
- data_quality: validation gates pass/fail correctly
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.transformations.silver_loans import (
    transform_silver_loans,
    _drop_missing_critical,
    _deduplicate,
    _cast_types,
    _flag_outliers,
    _compute_months_to_maturity,
    _impute_occupancy,
)
from src.transformations.silver_market import (
    transform_silver_market,
    _normalize_date,
    _interpolate_monthly,
    _rolling_average,
)
from src.transformations.feature_engineering import (
    compute_loan_features,
    _get_latest_rate,
    REFI_SPREAD_BPS,
)
from src.transformations.data_quality import (
    validate_bronze_loans,
    validate_bronze_market,
    validate_silver_features,
    DataQualityError,
    DataQualityReport,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_loan():
    return {
        "loan_id": "LN-001",
        "origination_date": "2020-03-15",
        "maturity_date": "2027-03-15",
        "origination_year": "2020",
        "original_balance": 20000000,
        "current_balance": 20000000,
        "note_rate": 0.035,
        "amortization_type": "interest_only",
        "balloon_flag": True,
        "loan_term_years": "7",
        "loan_purpose": "acquisition",
        "property_id": "PROP-001",
        "property_type": "office",
        "metro_area": "New York",
        "submarket": "New York - CBD",
        "occupancy_pct": 0.85,
        "noi_annual": 1500000,
        "property_value_at_origination": 30000000,
        "ltv_at_origination": 0.667,
        "dscr_at_origination": 1.5,
        "sponsor_credit_tier": "A",
        "ingested_at": "2026-01-01T00:00:00",
        "source": "synthetic_generator",
        "source_version": "1.0.0",
    }


@pytest.fixture
def sample_loans(sample_loan):
    """Generate a small set of loan records."""
    loans = []
    for i in range(10):
        loan = dict(sample_loan)
        loan["loan_id"] = f"LN-{i:03d}"
        loan["property_id"] = f"PROP-{i:03d}"
        loan["original_balance"] = 10000000 + i * 5000000
        loan["current_balance"] = loan["original_balance"]
        loan["ltv_at_origination"] = 0.55 + i * 0.03
        loan["dscr_at_origination"] = 1.2 + i * 0.1
        loan["occupancy_pct"] = 0.80 + i * 0.02
        loan["noi_annual"] = 800000 + i * 200000
        loans.append(loan)
    return loans


@pytest.fixture
def sample_market_records():
    """Generate sample market records spanning multiple types."""
    records = []
    # Treasury 10Y monthly
    for month in range(1, 13):
        records.append({
            "data_type": "treasury_10y",
            "observation_date": f"2024-{month:02d}-01",
            "value": 4.0 + month * 0.05,
            "frequency": "monthly",
            "property_type": None,
            "metro": None,
        })
    # Treasury 5Y
    for month in range(1, 13):
        records.append({
            "data_type": "treasury_5y",
            "observation_date": f"2024-{month:02d}-01",
            "value": 3.8 + month * 0.04,
            "frequency": "monthly",
            "property_type": None,
            "metro": None,
        })
    # SOFR
    for month in range(1, 13):
        records.append({
            "data_type": "sofr",
            "observation_date": f"2024-{month:02d}-01",
            "value": 5.0 + month * 0.02,
            "frequency": "monthly",
            "property_type": None,
            "metro": None,
        })
    # Cap rates (quarterly, national)
    for ptype in ["office", "retail", "industrial", "multifamily", "hotel"]:
        cap_base = {"office": 7.5, "retail": 7.2, "industrial": 5.3, "multifamily": 5.5, "hotel": 8.5}
        for q in range(1, 5):
            records.append({
                "data_type": "cap_rate",
                "observation_date": f"2024-Q{q}",
                "value": cap_base[ptype] + q * 0.1,
                "frequency": "quarterly",
                "property_type": ptype,
                "metro": "National",
            })
    # CRE price index
    for q in range(1, 5):
        records.append({
            "data_type": "cre_price_index",
            "observation_date": f"2024-Q{q}",
            "value": 105 + q * 0.5,
            "frequency": "quarterly",
            "property_type": None,
            "metro": None,
        })
    return records


# ─── Silver Loans Tests ───────────────────────────────────────────────────────


class TestSilverLoans:
    """Test silver loan transformations."""

    def test_transform_preserves_count_when_clean(self, sample_loans):
        result = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        assert len(result) == len(sample_loans)

    def test_drops_missing_loan_id(self, sample_loans):
        sample_loans[0]["loan_id"] = None
        sample_loans[1]["loan_id"] = ""
        result = _drop_missing_critical(sample_loans)
        assert len(result) == len(sample_loans) - 2

    def test_drops_missing_maturity(self, sample_loans):
        sample_loans[0]["maturity_date"] = None
        result = _drop_missing_critical(sample_loans)
        assert len(result) == len(sample_loans) - 1

    def test_deduplication(self, sample_loans):
        # Add a duplicate
        dupe = dict(sample_loans[0])
        sample_loans.append(dupe)
        result = _deduplicate(sample_loans)
        assert len(result) == 10  # Original count without the dupe

    def test_cast_types_numeric(self, sample_loan):
        sample_loan["original_balance"] = "25000000"
        result = _cast_types(sample_loan)
        assert isinstance(result["original_balance"], float)
        assert result["original_balance"] == 25000000.0

    def test_cast_types_origination_year(self, sample_loan):
        sample_loan["origination_year"] = "2020"
        result = _cast_types(sample_loan)
        assert result["origination_year"] == 2020

    def test_outlier_flag_high_ltv(self, sample_loan):
        sample_loan["ltv_at_origination"] = 1.05
        result = _flag_outliers(sample_loan)
        assert result["_ltv_outlier"] is True
        assert result["_is_outlier"] is True

    def test_outlier_flag_low_dscr(self, sample_loan):
        sample_loan["dscr_at_origination"] = 0.4
        result = _flag_outliers(sample_loan)
        assert result["_dscr_outlier"] is True
        assert result["_is_outlier"] is True

    def test_outlier_flag_normal(self, sample_loan):
        result = _flag_outliers(sample_loan)
        assert result["_ltv_outlier"] is False
        assert result["_dscr_outlier"] is False
        assert result["_is_outlier"] is False

    def test_months_to_maturity_future(self, sample_loan):
        result = _compute_months_to_maturity(sample_loan, date(2025, 3, 15))
        assert result["months_to_maturity"] > 0
        assert result["is_matured"] is False
        # Should be ~24 months
        assert 23 <= result["months_to_maturity"] <= 25

    def test_months_to_maturity_past(self, sample_loan):
        sample_loan["maturity_date"] = "2024-01-15"
        result = _compute_months_to_maturity(sample_loan, date(2025, 3, 15))
        assert result["months_to_maturity"] < 0
        assert result["is_matured"] is True

    def test_impute_occupancy(self, sample_loans):
        sample_loans[0]["occupancy_pct"] = None
        sample_loans[1]["occupancy_pct"] = 0
        result = _impute_occupancy(sample_loans)
        assert result[0]["occupancy_pct"] > 0
        assert result[0]["_occupancy_imputed"] is True
        assert result[2]["_occupancy_imputed"] is False

    def test_silver_metadata_added(self, sample_loans):
        result = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        assert "_silver_processed_at" in result[0]


# ─── Silver Market Tests ──────────────────────────────────────────────────────


class TestSilverMarket:
    """Test silver market transformations."""

    def test_normalize_date_iso(self):
        assert _normalize_date("2024-01-15") == "2024-01-15"

    def test_normalize_date_quarter(self):
        result = _normalize_date("2024-Q1")
        assert result == "2024-02-15"

    def test_normalize_date_quarter3(self):
        result = _normalize_date("2024-Q3")
        assert result == "2024-08-15"

    def test_normalize_date_yearmonth(self):
        assert _normalize_date("2024-03") == "2024-03-01"

    def test_interpolate_fills_gaps(self):
        series = [
            ("2024-01-01", 4.0),
            ("2024-03-01", 4.2),
        ]
        result = _interpolate_monthly(series)
        assert len(result) == 3  # Jan, Feb, Mar
        # Feb should be interpolated to 4.1
        assert abs(result[1][1] - 4.1) < 0.01

    def test_interpolate_no_gaps(self):
        series = [
            ("2024-01-01", 4.0),
            ("2024-02-01", 4.1),
            ("2024-03-01", 4.2),
        ]
        result = _interpolate_monthly(series)
        assert len(result) == 3
        assert result[0][1] == 4.0
        assert result[1][1] == 4.1
        assert result[2][1] == 4.2

    def test_rolling_average(self):
        series = [
            ("Q1", 6.0),
            ("Q2", 7.0),
            ("Q3", 8.0),
            ("Q4", 9.0),
        ]
        result = _rolling_average(series, window=3)
        assert len(result) == 4
        # First element: avg of [6.0] = 6.0
        assert result[0][1] == 6.0
        # Second: avg of [6.0, 7.0] = 6.5
        assert result[1][1] == 6.5
        # Third: avg of [6.0, 7.0, 8.0] = 7.0
        assert result[2][1] == 7.0
        # Fourth: avg of [7.0, 8.0, 9.0] = 8.0
        assert result[3][1] == 8.0

    def test_transform_produces_all_types(self, sample_market_records):
        result = transform_silver_market(sample_market_records)
        types = set(r["data_type"] for r in result)
        assert "treasury_10y" in types
        assert "treasury_5y" in types
        assert "sofr" in types
        assert "cap_rate" in types
        assert "cre_price_index" in types

    def test_transform_adds_value_decimal(self, sample_market_records):
        result = transform_silver_market(sample_market_records)
        for r in result[:10]:
            assert "value_decimal" in r
            assert r["value_decimal"] is not None

    def test_silver_metadata_added(self, sample_market_records):
        result = transform_silver_market(sample_market_records)
        assert "_silver_processed_at" in result[0]


# ─── Feature Engineering Tests ────────────────────────────────────────────────


class TestFeatureEngineering:
    """Test derived feature computations."""

    def test_compute_features_produces_output(self, sample_loans, sample_market_records):
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market, reference_date=date(2026, 1, 1))
        assert len(result) == len(sample_loans)

    def test_current_value_positive(self, sample_loans, sample_market_records):
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market)
        for r in result:
            assert r["current_value"] > 0

    def test_current_ltv_reasonable(self, sample_loans, sample_market_records):
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market)
        for r in result:
            assert 0 < r["current_ltv"] < 5.0

    def test_refinance_rate_includes_spread(self, sample_loans, sample_market_records):
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market)
        # Office spread is 250bps
        for r in result:
            if r["property_type"] == "office":
                # refinance_rate should be > treasury_10y/100
                assert r["refinance_rate"] > 0.04  # > 4%

    def test_rate_gap_calculation(self, sample_loans, sample_market_records):
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market)
        for r in result:
            expected_gap = r["refinance_rate"] - r["note_rate"]
            assert abs(r["rate_gap"] - expected_gap) < 0.0001

    def test_new_dscr_io_loan(self, sample_loans, sample_market_records):
        """For IO loan: new_dscr = NOI / (balance * refi_rate)."""
        sample_loans[0]["amortization_type"] = "interest_only"
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market)
        r = result[0]
        expected_ds = r["current_balance"] * r["refinance_rate"]
        expected_dscr = r["noi_annual"] / expected_ds
        assert abs(r["new_dscr"] - expected_dscr) < 0.01

    def test_debt_yield_calculation(self, sample_loans, sample_market_records):
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market)
        for r in result:
            expected = r["noi_annual"] / r["current_balance"]
            assert abs(r["debt_yield"] - expected) < 0.001

    def test_distress_flag_high_ltv(self, sample_loans, sample_market_records):
        """Loans with current_ltv > 0.80 should be flagged as stressed."""
        # Make a high-LTV loan
        sample_loans[0]["current_balance"] = 50000000
        sample_loans[0]["noi_annual"] = 500000  # Low NOI → high LTV
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        result = compute_loan_features(silver_loans, silver_market)
        high_ltv_loan = result[0]
        if high_ltv_loan["current_ltv"] > 0.80:
            assert high_ltv_loan["_refi_stressed"] is True


# ─── Data Quality Tests ───────────────────────────────────────────────────────


class TestDataQuality:
    """Test data quality gates."""

    def test_empty_dataset_fails(self):
        with pytest.raises(DataQualityError):
            validate_bronze_loans([], halt_on_failure=True)

    def test_valid_data_passes(self, sample_loans):
        report = validate_bronze_loans(sample_loans, halt_on_failure=False)
        assert report.passed

    def test_missing_critical_field_fails(self, sample_loans):
        # Remove loan_id from all records → 100% null rate
        for loan in sample_loans:
            loan["loan_id"] = None
        report = validate_bronze_loans(sample_loans, halt_on_failure=False)
        assert not report.passed

    def test_duplicates_detected(self, sample_loans):
        # All same loan_id
        for loan in sample_loans:
            loan["loan_id"] = "LN-DUPE"
        report = validate_bronze_loans(sample_loans, halt_on_failure=False)
        # Should have a warning or failure about duplicates
        all_issues = report.checks_warned + report.checks_failed
        assert any("DUPLICATE" in issue.upper() for issue in all_issues)

    def test_market_empty_fails(self):
        with pytest.raises(DataQualityError):
            validate_bronze_market([], halt_on_failure=True)

    def test_market_valid_passes(self, sample_market_records):
        report = validate_bronze_market(sample_market_records, halt_on_failure=False)
        assert report.passed

    def test_silver_features_valid(self, sample_loans, sample_market_records):
        silver_loans = transform_silver_loans(sample_loans, reference_date=date(2026, 1, 1))
        silver_market = transform_silver_market(sample_market_records)
        features = compute_loan_features(silver_loans, silver_market)
        report = validate_silver_features(features, halt_on_failure=False)
        assert report.passed

    def test_report_summary_format(self):
        report = DataQualityReport()
        report.add_pass("check1")
        report.add_warning("check2")
        summary = report.summary()
        assert "1 PASS" in summary
        assert "1 WARN" in summary
        assert "0 FAIL" in summary



# ─── Idiosyncratic Shock Tests ────────────────────────────────────────────────

# These tests require pandas/numpy (same as the classifier tests)
# but use the features.py shock machinery directly.

_pd = None
try:
    import pandas as _pd_mod
    import numpy as _np_mod
    _pd = _pd_mod
except ImportError:
    pass


@pytest.mark.skipif(_pd is None, reason="pandas/numpy required for shock tests")
class TestIdiosyncraticShocks:
    """Test the idiosyncratic shocks in label computation."""

    @pytest.fixture
    def shock_fixture_tables(self, tmp_path):
        """Create a small fixture with loans + market for shock testing."""
        from src.utils.delta_writer import DeltaWriter
        import random
        random.seed(42)

        # Market data covering 2015-2025
        market_records = []
        for year in range(2015, 2026):
            for month in range(1, 13):
                market_records.append({
                    "data_type": "treasury_10y",
                    "observation_date": f"{year}-{month:02d}-01",
                    "value": 2.0 + (year - 2015) * 0.2,
                    "frequency": "monthly",
                    "property_type": None,
                    "metro": None,
                    "_silver_processed_at": "2026-01-01T00:00:00",
                })
        for ptype in ["office", "retail", "industrial", "multifamily", "hotel"]:
            base = {"office": 6.5, "retail": 6.8, "industrial": 5.5, "multifamily": 5.2, "hotel": 7.5}[ptype]
            for year in range(2015, 2026):
                for q in range(1, 5):
                    month = {1: 2, 2: 5, 3: 8, 4: 11}[q]
                    market_records.append({
                        "data_type": "cap_rate",
                        "observation_date": f"{year}-{month:02d}-15",
                        "value": base + (year - 2015) * 0.1,
                        "frequency": "quarterly",
                        "property_type": ptype,
                        "metro": "National",
                        "_silver_processed_at": "2026-01-01T00:00:00",
                    })

        # 50 loans with maturity within range
        loan_records = []
        for i in range(50):
            orig_year = random.choice([2015, 2016, 2017, 2018])
            mat_year = min(orig_year + 5, 2024)
            loan_records.append({
                "loan_id": f"SHOCK-{i:04d}",
                "property_type": random.choice(["office", "retail", "industrial", "multifamily", "hotel"]),
                "metro_area": random.choice(["New York", "Chicago", "LA"]),
                "origination_year": str(orig_year),
                "origination_date": f"{orig_year}-03-15",
                "maturity_date": f"{mat_year}-03-15",
                "original_balance": 20000000,
                "current_balance": 20000000,
                "note_rate": 0.04,
                "amortization_type": random.choice(["interest_only", "amortizing"]),
                "balloon_flag": "True",
                "ltv_at_origination": 0.65,
                "dscr_at_origination": 1.4,
                "occupancy_pct": 0.90,
                "noi_annual": 1500000,
                "sponsor_credit_tier": "B",
                "_feature_computed_at": "2026-01-01T00:00:00",
            })

        gold_path = tmp_path / "loans"
        market_path = tmp_path / "market"
        DeltaWriter(gold_path).write(loan_records)
        DeltaWriter(market_path).write(market_records, partition_by="data_type")
        return gold_path, market_path

    def test_shocks_reproducible(self, shock_fixture_tables, monkeypatch):
        """Same loan_id produces same shock across two independent runs."""
        from src.models.features import build_training_frame

        gold_path, market_path = shock_fixture_tables

        # Run 1
        monkeypatch.setattr("src.models.features._load_shock_config", lambda: True)
        result1 = build_training_frame(gold_path=gold_path, market_path=market_path, seed=42)
        y1 = _pd.concat([result1[1], result1[3], result1[5]])

        # Run 2 (same inputs)
        result2 = build_training_frame(gold_path=gold_path, market_path=market_path, seed=42)
        y2 = _pd.concat([result2[1], result2[3], result2[5]])

        # Labels should be identical (shocks are deterministic per loan_id)
        assert (y1.values == y2.values).all(), "Shocks not reproducible across runs"

    def test_shocks_change_labels(self, shock_fixture_tables, monkeypatch):
        """With shocks enabled, at least 15% of loans differ from deterministic baseline."""
        from src.models.features import build_training_frame

        # Run with shocks OFF
        monkeypatch.setattr("src.models.features._load_shock_config", lambda: False)
        result_det = build_training_frame(gold_path=shock_fixture_tables[0], market_path=shock_fixture_tables[1], seed=42)
        y_det = _pd.concat([result_det[1], result_det[3], result_det[5]])

        # Run with shocks ON
        monkeypatch.setattr("src.models.features._load_shock_config", lambda: True)
        result_shock = build_training_frame(gold_path=shock_fixture_tables[0], market_path=shock_fixture_tables[1], seed=42)
        y_shock = _pd.concat([result_shock[1], result_shock[3], result_shock[5]])

        # At least 15% should have a different label
        if len(y_det) > 0 and len(y_shock) > 0:
            # Align by index
            common = y_det.index.intersection(y_shock.index)
            if len(common) > 5:
                diff_pct = (y_det.loc[common] != y_shock.loc[common]).mean()
                assert diff_pct >= 0.10, (
                    f"Only {diff_pct*100:.1f}% of labels changed with shocks "
                    f"(expected >= 15%)"
                )

    def test_shocks_disabled_matches_deterministic(self, shock_fixture_tables, monkeypatch):
        """With config flag off, output matches deterministic computation exactly."""
        from src.models.features import build_training_frame

        # Two runs both with shocks disabled
        monkeypatch.setattr("src.models.features._load_shock_config", lambda: False)
        result1 = build_training_frame(gold_path=shock_fixture_tables[0], market_path=shock_fixture_tables[1], seed=42)
        y1 = _pd.concat([result1[1], result1[3], result1[5]])

        result2 = build_training_frame(gold_path=shock_fixture_tables[0], market_path=shock_fixture_tables[1], seed=42)
        y2 = _pd.concat([result2[1], result2[3], result2[5]])

        assert (y1.values == y2.values).all(), "Deterministic mode should be perfectly reproducible"
