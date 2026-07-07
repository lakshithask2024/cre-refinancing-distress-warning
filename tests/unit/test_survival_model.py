"""
Unit tests for the Cox PH Survival Model.

Tests verify data preparation correctness without fitting a real model
(lifelines may not be installed in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pd = pytest.importorskip("pandas", reason="pandas required")
np = pytest.importorskip("numpy", reason="numpy required")
lifelines = pytest.importorskip("lifelines", reason="lifelines required")


@pytest.fixture
def survival_fixture_tables(tmp_path):
    """Create fixture data for survival model tests."""
    from src.utils.delta_writer import DeltaWriter
    import random

    random.seed(42)
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

    loan_records = []
    for i in range(80):
        orig_year = random.choice([2015, 2016, 2017, 2018, 2019])
        mat_year = min(orig_year + random.choice([5, 7]), 2025)
        tier = random.choice(["critical", "high", "medium", "low"])
        loan_records.append({
            "loan_id": f"SURV-{i:04d}",
            "property_type": random.choice(["office", "retail", "industrial"]),
            "metro_area": random.choice(["New York", "Chicago"]),
            "origination_year": str(orig_year),
            "origination_date": f"{orig_year}-06-15",
            "maturity_date": f"{mat_year}-06-15",
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
            "distress_tier": tier,
            "_feature_computed_at": "2026-01-01T00:00:00",
        })

    gold_path = tmp_path / "loans"
    market_path = tmp_path / "market"
    DeltaWriter(gold_path).write(loan_records)
    DeltaWriter(market_path).write(market_records, partition_by="data_type")
    return gold_path, market_path


class TestSurvivalDataPrep:
    """Test survival data preparation."""

    def test_duration_positive(self, survival_fixture_tables):
        from src.models.survival_model import build_survival_frame

        df_train, df_valid, df_test, _ = build_survival_frame(
            gold_path=survival_fixture_tables[0],
            market_path=survival_fixture_tables[1],
        )
        for df in [df_train, df_valid, df_test]:
            if len(df) > 0:
                assert (df["duration_months"] > 0).all(), "All durations must be > 0"

    def test_event_indicator_binary(self, survival_fixture_tables):
        from src.models.survival_model import build_survival_frame

        df_train, df_valid, df_test, _ = build_survival_frame(
            gold_path=survival_fixture_tables[0],
            market_path=survival_fixture_tables[1],
        )
        for df in [df_train, df_valid, df_test]:
            if len(df) > 0:
                assert set(df["event_observed"].unique()).issubset({0, 1})

    def test_survival_probs_monotonic(self, survival_fixture_tables):
        """S(6) >= S(12) >= S(24) >= S(36) for any loan."""
        from src.models.survival_model import build_survival_frame
        from lifelines import CoxPHFitter

        df_train, _, _, feature_cols = build_survival_frame(
            gold_path=survival_fixture_tables[0],
            market_path=survival_fixture_tables[1],
        )
        if len(df_train) < 10:
            pytest.skip("Not enough training data")

        fit_cols = feature_cols + ["duration_months", "event_observed"]
        fit_df = df_train[fit_cols].astype(float).fillna(0.0)

        cph = CoxPHFitter(penalizer=0.5)
        cph.fit(fit_df, duration_col="duration_months", event_col="event_observed")

        X = fit_df[feature_cols].iloc[:5]
        sf = cph.predict_survival_function(X)

        for col in sf.columns:
            values = sf[col].values
            # Survival function must be monotonically non-increasing
            assert all(values[i] >= values[i + 1] - 1e-10 for i in range(len(values) - 1))

    @pytest.mark.slow
    def test_concordance_reasonable(self, survival_fixture_tables):
        """C-index should be > 0.50 (better than random)."""
        from src.models.survival_model import build_survival_frame
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

        df_train, df_valid, _, feature_cols = build_survival_frame(
            gold_path=survival_fixture_tables[0],
            market_path=survival_fixture_tables[1],
        )
        if len(df_train) < 10:
            pytest.skip("Not enough training data")

        fit_cols = feature_cols + ["duration_months", "event_observed"]
        fit_df = df_train[fit_cols].astype(float).fillna(0.0)

        cph = CoxPHFitter(penalizer=0.5)
        cph.fit(fit_df, duration_col="duration_months", event_col="event_observed")

        assert cph.concordance_index_ > 0.50
