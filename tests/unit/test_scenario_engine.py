"""
Unit tests for the Stress Testing Scenario Engine.

Tests verify shock application mechanics without requiring a trained model.
Integration tests (full 8-scenario runs) are marked @pytest.mark.slow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pd = pytest.importorskip("pandas", reason="pandas required")
np = pytest.importorskip("numpy", reason="numpy required")


@pytest.fixture
def sample_portfolio():
    """Create a minimal portfolio DataFrame for shock testing."""
    return pd.DataFrame([
        {
            "loan_id": "LN-001", "property_type": "office", "metro_area": "New York",
            "current_balance": 20000000.0, "noi_annual": 1500000.0,
            "current_cap_rate": 7.5, "current_ltv": 0.85,
            "refinance_rate": 0.065, "note_rate": 0.04,
            "rate_gap": 0.025, "rate_gap_bps": 250.0,
            "new_dscr": 1.15, "debt_yield": 0.075,
            "amortization_type": "interest_only",
            "occupancy_pct": 0.85, "ltv_at_origination": 0.65,
            "dscr_at_origination": 1.4, "original_balance": 20000000.0,
        },
        {
            "loan_id": "LN-002", "property_type": "industrial", "metro_area": "Chicago",
            "current_balance": 15000000.0, "noi_annual": 1200000.0,
            "current_cap_rate": 5.5, "current_ltv": 0.60,
            "refinance_rate": 0.060, "note_rate": 0.035,
            "rate_gap": 0.025, "rate_gap_bps": 250.0,
            "new_dscr": 1.33, "debt_yield": 0.08,
            "amortization_type": "amortizing",
            "occupancy_pct": 0.95, "ltv_at_origination": 0.55,
            "dscr_at_origination": 1.6, "original_balance": 15000000.0,
        },
        {
            "loan_id": "LN-003", "property_type": "office", "metro_area": "Dallas",
            "current_balance": 30000000.0, "noi_annual": 2000000.0,
            "current_cap_rate": 7.8, "current_ltv": 0.95,
            "refinance_rate": 0.065, "note_rate": 0.045,
            "rate_gap": 0.020, "rate_gap_bps": 200.0,
            "new_dscr": 0.90, "debt_yield": 0.067,
            "amortization_type": "interest_only",
            "occupancy_pct": 0.78, "ltv_at_origination": 0.70,
            "dscr_at_origination": 1.2, "original_balance": 30000000.0,
        },
    ])


class TestShockApplication:
    """Test that shocks are applied correctly to portfolio features."""

    def test_rate_shock_100_applied(self, sample_portfolio):
        """After rate_shock_100bps, refinance_rate is exactly 100 bps higher."""
        from src.stress_testing.scenario_engine import apply_scenario

        scenario = {"name": "rate_100", "rate_shock_bps": 100, "cap_rate_shock_bps": 0, "noi_shock_pct": 0.0}
        original_rates = sample_portfolio["refinance_rate"].copy()
        stressed = apply_scenario(sample_portfolio, scenario)

        for i in range(len(stressed)):
            expected = original_rates.iloc[i] + 0.01  # +100 bps = +0.01
            assert abs(stressed.iloc[i]["refinance_rate"] - expected) < 1e-10, (
                f"Loan {i}: expected refi rate {expected}, got {stressed.iloc[i]['refinance_rate']}"
            )

    def test_cap_rate_shock_updates_property_value(self, sample_portfolio):
        """After cap rate shock, property value decreases."""
        from src.stress_testing.scenario_engine import apply_scenario

        # Record original values
        orig_cap = sample_portfolio["current_cap_rate"].copy()
        orig_noi = sample_portfolio["noi_annual"].copy()
        orig_values = orig_noi / (orig_cap / 100.0)

        scenario = {"name": "cap_100", "rate_shock_bps": 0, "cap_rate_shock_bps": 100, "noi_shock_pct": 0.0}
        stressed = apply_scenario(sample_portfolio, scenario)

        # New values should be lower (higher cap rate → lower value)
        new_values = stressed["current_value"]
        for i in range(len(stressed)):
            assert new_values.iloc[i] < orig_values.iloc[i], (
                f"Loan {i}: value should decrease with cap rate increase"
            )

    def test_noi_shock_applied(self, sample_portfolio):
        """NOI shock reduces NOI by the specified percentage."""
        from src.stress_testing.scenario_engine import apply_scenario

        original_noi = sample_portfolio["noi_annual"].copy()
        scenario = {"name": "noi_down", "rate_shock_bps": 0, "cap_rate_shock_bps": 0, "noi_shock_pct": -0.10}
        stressed = apply_scenario(sample_portfolio, scenario)

        for i in range(len(stressed)):
            expected = original_noi.iloc[i] * 0.90
            assert abs(stressed.iloc[i]["noi_annual"] - expected) < 1.0, (
                f"Loan {i}: expected NOI {expected}, got {stressed.iloc[i]['noi_annual']}"
            )

    def test_property_type_filter(self, sample_portfolio):
        """Shocks with property_type_filter only affect matching loans."""
        from src.stress_testing.scenario_engine import apply_scenario

        original_caps = sample_portfolio["current_cap_rate"].copy()
        scenario = {
            "name": "office_only", "rate_shock_bps": 0,
            "cap_rate_shock_bps": 200, "noi_shock_pct": 0.0,
            "property_type_filter": "office",
        }
        stressed = apply_scenario(sample_portfolio, scenario)

        # Office loans should have higher cap rate
        office_mask = sample_portfolio["property_type"] == "office"
        for i in range(len(stressed)):
            if office_mask.iloc[i]:
                assert stressed.iloc[i]["current_cap_rate"] > original_caps.iloc[i]
            else:
                # Non-office should be unchanged
                assert abs(stressed.iloc[i]["current_cap_rate"] - original_caps.iloc[i]) < 1e-10

    def test_baseline_no_change(self, sample_portfolio):
        """Baseline scenario should not change any features."""
        from src.stress_testing.scenario_engine import apply_scenario

        original = sample_portfolio.copy()
        scenario = {"name": "baseline", "rate_shock_bps": 0, "cap_rate_shock_bps": 0, "noi_shock_pct": 0.0}
        stressed = apply_scenario(sample_portfolio, scenario)

        for col in ["refinance_rate", "current_cap_rate", "noi_annual"]:
            assert (stressed[col] == original[col]).all(), f"{col} changed under baseline"

    def test_combined_severe_increases_ltv(self, sample_portfolio):
        """Combined severe scenario should increase LTV (rate up + cap up + NOI down)."""
        from src.stress_testing.scenario_engine import apply_scenario

        original_ltv = sample_portfolio["current_ltv"].copy()
        scenario = {
            "name": "combined", "rate_shock_bps": 200,
            "cap_rate_shock_bps": 200, "noi_shock_pct": -0.10,
        }
        stressed = apply_scenario(sample_portfolio, scenario)

        # LTV should increase (higher cap rate → lower value → higher LTV)
        for i in range(len(stressed)):
            assert stressed.iloc[i]["current_ltv"] > original_ltv.iloc[i], (
                f"Loan {i}: LTV should increase under combined stress"
            )


class TestScenarioConfig:
    """Test scenario configuration loading."""

    def test_load_scenarios_returns_list(self):
        """load_scenarios should return a non-empty list."""
        from src.stress_testing.scenario_engine import load_scenarios

        scenarios = load_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) == 8

    def test_all_scenarios_have_name(self):
        """Every scenario must have a 'name' field."""
        from src.stress_testing.scenario_engine import load_scenarios

        scenarios = load_scenarios()
        for s in scenarios:
            assert "name" in s, f"Scenario missing 'name': {s}"

    def test_baseline_exists(self):
        """There must be a 'baseline' scenario with zero shocks."""
        from src.stress_testing.scenario_engine import load_scenarios

        scenarios = load_scenarios()
        baselines = [s for s in scenarios if s["name"] == "baseline"]
        assert len(baselines) == 1
        assert baselines[0]["rate_shock_bps"] == 0
        assert baselines[0]["cap_rate_shock_bps"] == 0



# ─── Smoke test: scoring with featurization ───────────────────────────────────

xgb_mod = pytest.importorskip("xgboost", reason="xgboost required for scoring smoke test")


class TestScoringSmoke:
    """Smoke test that the full featurize→score path works without errors."""

    @pytest.mark.slow
    def test_baseline_scores_run(self, tmp_path):
        """Load a fixture, featurize, and score with a dummy model — no KeyError."""
        import xgboost as xgb
        from src.stress_testing.scenario_engine import apply_scenario, score_portfolio
        from src.models.features import featurize_for_scoring, NUMERIC_FEATURES_AT_TOBS, ONEHOT_FEATURES
        from src.utils.delta_writer import DeltaWriter
        import random

        random.seed(42)

        # Create market fixture
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
                    "_silver_processed_at": "2026-01-01",
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
                        "_silver_processed_at": "2026-01-01",
                    })

        market_df = pd.DataFrame(market_records)

        # Create loan fixture (100 loans)
        loans = []
        for i in range(100):
            orig_year = random.choice([2015, 2016, 2017, 2018])
            mat_year = orig_year + 5
            loans.append({
                "loan_id": f"SMOKE-{i:04d}",
                "property_type": random.choice(["office", "retail", "industrial", "multifamily", "hotel"]),
                "metro_area": random.choice(["New York", "Chicago", "LA"]),
                "origination_year": str(orig_year),
                "origination_date": f"{orig_year}-06-15",
                "maturity_date": f"{mat_year}-06-15",
                "original_balance": 20000000.0,
                "current_balance": 20000000.0,
                "note_rate": 0.04,
                "noi_annual": 1500000.0,
                "current_cap_rate": 7.0,
                "current_ltv": 0.75,
                "refinance_rate": 0.065,
                "rate_gap": 0.025,
                "rate_gap_bps": 250.0,
                "new_dscr": 1.2,
                "debt_yield": 0.075,
                "amortization_type": random.choice(["interest_only", "amortizing"]),
                "balloon_flag": "True",
                "ltv_at_origination": 0.65,
                "dscr_at_origination": 1.4,
                "occupancy_pct": 0.90,
                "sponsor_credit_tier": "B",
            })
        portfolio = pd.DataFrame(loans)

        # Featurize to get expected feature names
        X = featurize_for_scoring(
            loans_df=portfolio,
            market_df=market_df,
            model_feature_names=NUMERIC_FEATURES_AT_TOBS + ["metro_encoded"],  # partial
        )

        # Build a dummy model with the right number of features
        # Use ALL features that featurize_for_scoring might produce
        full_feature_names = list(X.columns)
        # Add one-hot columns that would exist
        for col in ONEHOT_FEATURES:
            for val in portfolio[col].unique():
                fname = f"{col}_{val}"
                if fname not in full_feature_names:
                    full_feature_names.append(fname)

        # Re-featurize with full feature set
        X_full = featurize_for_scoring(
            loans_df=portfolio,
            market_df=market_df,
            model_feature_names=full_feature_names,
        )

        # Train a dummy model on this feature set
        y_dummy = (np.random.rand(len(X_full)) > 0.5).astype(int)
        dummy_model = xgb.XGBClassifier(
            n_estimators=5, max_depth=2, use_label_encoder=False,
            eval_metric="logloss", verbosity=0, random_state=42,
        )
        dummy_model.fit(X_full, y_dummy)

        # Now run the scenario engine's score_portfolio with this model
        baseline = {"name": "baseline", "rate_shock_bps": 0, "cap_rate_shock_bps": 0, "noi_shock_pct": 0.0}
        stressed = apply_scenario(portfolio, baseline)
        scored = score_portfolio(stressed, dummy_model, full_feature_names, market_df=market_df)

        # Assertions
        assert "stressed_pd" in scored.columns
        assert "stressed_distress_tier" in scored.columns
        assert len(scored) == 100
        assert scored["stressed_pd"].between(0, 1).all()
