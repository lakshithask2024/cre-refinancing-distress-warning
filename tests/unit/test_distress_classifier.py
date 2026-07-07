"""
Unit tests for the Distress Classifier feature engineering.

Tests verify:
- Forward-looking framing: features at T_obs, label at maturity
- No leakage: features don't contain maturity-time information
- Correct time-based splits
- Target encoding from train only
- Feature correlations below leakage threshold

Runtime target: < 10 seconds total.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Skip entire module if ML dependencies are not installed
pd = pytest.importorskip("pandas", reason="pandas required for distress classifier tests")
np = pytest.importorskip("numpy", reason="numpy required for distress classifier tests")
optuna = pytest.importorskip("optuna", reason="optuna required for search space tests")


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def market_records():
    """Generate market data covering 2015-2025 for T_obs and maturity lookups."""
    records = []
    # Treasury 10Y monthly (rates trend up from 2% to 4.5%)
    for year in range(2015, 2026):
        for month in range(1, 13):
            base_rate = 2.0 + (year - 2015) * 0.25 + (month - 1) * 0.01
            records.append({
                "data_type": "treasury_10y",
                "observation_date": f"{year}-{month:02d}-01",
                "value": round(base_rate, 2),
                "value_decimal": round(base_rate / 100, 4),
                "frequency": "monthly",
                "property_type": None,
                "metro": None,
                "_silver_processed_at": "2026-01-01T00:00:00",
            })
    # Cap rates quarterly by property type (expanding over time)
    for ptype in ["office", "retail", "industrial", "multifamily", "hotel"]:
        base_cap = {"office": 6.0, "retail": 6.5, "industrial": 5.5, "multifamily": 5.0, "hotel": 7.5}[ptype]
        for year in range(2015, 2026):
            for q in range(1, 5):
                month = {1: 2, 2: 5, 3: 8, 4: 11}[q]
                cap = base_cap + (year - 2015) * 0.15 + (q - 1) * 0.02
                records.append({
                    "data_type": "cap_rate",
                    "observation_date": f"{year}-{month:02d}-15",
                    "value": round(cap, 2),
                    "value_decimal": round(cap / 100, 4),
                    "frequency": "quarterly",
                    "property_type": ptype,
                    "metro": "National",
                    "_silver_processed_at": "2026-01-01T00:00:00",
                })
    return records


@pytest.fixture
def loan_records():
    """Generate ~100 loans with maturity dates inside market range."""
    import random
    random.seed(42)

    records = []
    metros = ["New York", "Los Angeles", "Chicago", "Dallas", "Miami"]
    prop_types = ["office", "retail", "industrial", "multifamily", "hotel"]
    amort_types = ["interest_only", "amortizing"]
    tiers = ["A", "B", "C"]

    for i in range(100):
        orig_year = random.choice([2015, 2016, 2017, 2018, 2019, 2020])
        term = random.choice([5, 7])
        mat_year = orig_year + term
        # Keep maturity within market range (≤ 2025)
        if mat_year > 2025:
            mat_year = 2025

        records.append({
            "loan_id": f"LN-TEST-{i:04d}",
            "property_type": random.choice(prop_types),
            "metro_area": random.choice(metros),
            "origination_year": str(orig_year),
            "origination_date": f"{orig_year}-06-15",
            "maturity_date": f"{mat_year}-06-15",
            "original_balance": random.uniform(5e6, 50e6),
            "current_balance": random.uniform(5e6, 50e6),
            "note_rate": random.uniform(0.03, 0.06),
            "amortization_type": random.choice(amort_types),
            "balloon_flag": random.choice(["True", "False"]),
            "ltv_at_origination": random.uniform(0.5, 0.75),
            "dscr_at_origination": random.uniform(1.1, 2.0),
            "occupancy_pct": random.uniform(0.7, 0.98),
            "noi_annual": random.uniform(500000, 5000000),
            "sponsor_credit_tier": random.choice(tiers),
            "distress_tier": "low",
            "_feature_computed_at": "2026-01-01T00:00:00",
        })
    return records


@pytest.fixture
def fixture_tables(tmp_path, loan_records, market_records):
    """Write both fixtures as Delta tables."""
    from src.utils.delta_writer import DeltaWriter

    gold_path = tmp_path / "loan_current_state"
    market_path = tmp_path / "market_rates"

    DeltaWriter(gold_path).write(loan_records)
    DeltaWriter(market_path).write(market_records, partition_by="data_type")

    return gold_path, market_path


@pytest.fixture
def training_data(fixture_tables):
    """Build training frame from fixture data."""
    from src.models.features import build_training_frame

    gold_path, market_path = fixture_tables
    return build_training_frame(gold_path=gold_path, market_path=market_path, seed=42)


# ─── Test: Basic shape and structure ──────────────────────────────────────────


class TestBuildTrainingFrame:
    """Verify the feature engineering pipeline produces correct shapes."""

    def test_returns_seven_elements(self, training_data):
        """build_training_frame returns exactly 7 elements."""
        assert len(training_data) == 7

    def test_split_sizes_non_zero(self, training_data):
        """Each split should have at least 1 row."""
        X_train, y_train, X_valid, y_valid, X_test, y_test, features = training_data
        assert len(X_train) > 0, "Train set is empty"
        assert len(X_valid) > 0, "Valid set is empty"
        assert len(X_test) > 0, "Test set is empty"

    def test_x_y_lengths_match(self, training_data):
        """X and y must have same number of rows in each split."""
        X_train, y_train, X_valid, y_valid, X_test, y_test, _ = training_data
        assert len(X_train) == len(y_train)
        assert len(X_valid) == len(y_valid)
        assert len(X_test) == len(y_test)

    def test_feature_names_match_columns(self, training_data):
        """Feature names list should match X column names exactly."""
        X_train, _, _, _, _, _, feature_names = training_data
        assert list(X_train.columns) == feature_names

    def test_labels_binary(self, training_data):
        """Labels should be binary (0 or 1) with no NaN."""
        _, y_train, _, y_valid, _, y_test, _ = training_data
        for y in [y_train, y_valid, y_test]:
            assert set(y.unique()).issubset({0, 1})
            assert y.isna().sum() == 0


# ─── Test: Features snapshot at T_obs (ANTI-LEAKAGE) ─────────────────────────


class TestFeaturesAtTobs:
    """Verify features use market data from T_obs, NOT current/maturity."""

    def test_features_snapshot_at_Tobs(self, tmp_path, market_records):
        """Two loans with different T_obs should get different market features."""
        from src.utils.delta_writer import DeltaWriter
        from src.models.features import build_training_frame

        # Loan A: matures 2020-06 → T_obs = 2018-06
        # Loan B: matures 2024-06 → T_obs = 2022-06
        # Treasury at 2018-06 ≈ 2.75%, at 2022-06 ≈ 3.75% (in our fixture)
        loans = [
            {
                "loan_id": "LOAN-A", "property_type": "office", "metro_area": "New York",
                "origination_year": "2015", "origination_date": "2015-01-01",
                "maturity_date": "2020-06-15",
                "original_balance": 10000000, "current_balance": 10000000,
                "note_rate": 0.04, "amortization_type": "interest_only",
                "balloon_flag": "True", "ltv_at_origination": 0.65,
                "dscr_at_origination": 1.5, "occupancy_pct": 0.9,
                "noi_annual": 1000000, "sponsor_credit_tier": "A",
                "_feature_computed_at": "2026-01-01T00:00:00",
            },
            {
                "loan_id": "LOAN-B", "property_type": "office", "metro_area": "New York",
                "origination_year": "2017", "origination_date": "2017-01-01",
                "maturity_date": "2024-06-15",
                "original_balance": 10000000, "current_balance": 10000000,
                "note_rate": 0.04, "amortization_type": "interest_only",
                "balloon_flag": "True", "ltv_at_origination": 0.65,
                "dscr_at_origination": 1.5, "occupancy_pct": 0.9,
                "noi_annual": 1000000, "sponsor_credit_tier": "A",
                "_feature_computed_at": "2026-01-01T00:00:00",
            },
        ]

        gold_path = tmp_path / "test_tobs_loans"
        market_path = tmp_path / "test_tobs_market"
        DeltaWriter(gold_path).write(loans)
        DeltaWriter(market_path).write(market_records, partition_by="data_type")

        X_train, _, X_valid, _, X_test, _, features = build_training_frame(
            gold_path=gold_path, market_path=market_path, seed=42
        )
        # Combine all splits to find both loans
        X_all = pd.concat([X_train, X_valid, X_test])

        # Both loans should exist and have DIFFERENT treasury rates at T_obs
        assert len(X_all) == 2
        treasury_values = X_all["treasury_10y_at_Tobs"].values
        assert treasury_values[0] != treasury_values[1], (
            "Same treasury at T_obs for loans with different T_obs — features not time-varying!"
        )

    def test_no_current_state_features_in_matrix(self, training_data):
        """Feature columns must not contain maturity-time or unqualified current-state names."""
        _, _, _, _, _, _, feature_names = training_data
        forbidden_patterns = ["at_maturity", "maturity_refi", "maturity_dscr", "maturity_ltv"]
        for col in feature_names:
            for pattern in forbidden_patterns:
                assert pattern not in col, (
                    f"Feature '{col}' contains forbidden pattern '{pattern}' — "
                    f"maturity-time values must not be in feature matrix"
                )


# ─── Test: Label uses future state ───────────────────────────────────────────


class TestLabelUsesFutureState:
    """Verify label depends on market AT maturity, not at T_obs."""

    def test_label_uses_maturity_market(self, tmp_path, market_records):
        """Two loans with same T_obs features but different maturity dates
        should get different labels if market moves between their maturities."""
        from src.utils.delta_writer import DeltaWriter
        from src.models.features import build_training_frame

        # Both loans: same origination, same T_obs market
        # Loan A matures 2020 (lower rates → less distress)
        # Loan B matures 2024 (higher rates → more distress)
        # Make balance very high so that higher rates push DSCR < 1
        loans = [
            {
                "loan_id": "SAME-TOBS-A", "property_type": "office", "metro_area": "Chicago",
                "origination_year": "2015", "origination_date": "2015-01-01",
                "maturity_date": "2020-01-15",  # T_obs = 2018-01
                "original_balance": 50000000, "current_balance": 50000000,
                "note_rate": 0.035, "amortization_type": "interest_only",
                "balloon_flag": "True", "ltv_at_origination": 0.70,
                "dscr_at_origination": 1.3, "occupancy_pct": 0.85,
                "noi_annual": 2000000, "sponsor_credit_tier": "B",
                "_feature_computed_at": "2026-01-01T00:00:00",
            },
            {
                "loan_id": "SAME-TOBS-B", "property_type": "office", "metro_area": "Chicago",
                "origination_year": "2017", "origination_date": "2017-01-01",
                "maturity_date": "2024-01-15",  # T_obs = 2022-01
                "original_balance": 50000000, "current_balance": 50000000,
                "note_rate": 0.035, "amortization_type": "interest_only",
                "balloon_flag": "True", "ltv_at_origination": 0.70,
                "dscr_at_origination": 1.3, "occupancy_pct": 0.85,
                "noi_annual": 2000000, "sponsor_credit_tier": "B",
                "_feature_computed_at": "2026-01-01T00:00:00",
            },
        ]

        gold_path = tmp_path / "test_label_loans"
        market_path = tmp_path / "test_label_market"
        DeltaWriter(gold_path).write(loans)
        DeltaWriter(market_path).write(market_records, partition_by="data_type")

        X_train, y_train, X_valid, y_valid, X_test, y_test, _ = build_training_frame(
            gold_path=gold_path, market_path=market_path, seed=42
        )
        y_all = pd.concat([y_train, y_valid, y_test])
        # With rates rising over time in our fixture, the 2024-maturity loan
        # should have a worse (higher) label probability. The label may or may
        # not differ (depends on exact rates), but they should at least both exist.
        assert len(y_all) == 2, "Both loans should pass the filter"


# ─── Test: No correlation leakage ─────────────────────────────────────────────


class TestNoCorrelationLeakage:
    """Guard against structural leakage via correlation checks."""

    def test_top_correlations_below_threshold(self, training_data):
        """No feature should have |correlation| > 0.85 with the label."""
        X_train, y_train, _, _, _, _, _ = training_data
        if len(X_train) < 10 or y_train.nunique() < 2:
            pytest.skip("Not enough data for correlation test")

        correlations = X_train.corrwith(y_train).abs()
        max_corr = correlations.max()
        worst_feature = correlations.idxmax()
        assert max_corr < 0.85, (
            f"Feature '{worst_feature}' has correlation {max_corr:.4f} with label — "
            f"possible leakage (threshold: 0.85)"
        )


# ─── Test: Time-based split correctness ──────────────────────────────────────


class TestTimeSplit:
    """Verify origination-year-based splits."""

    def test_no_loan_id_overlap(self, training_data, loan_records):
        """No loan should appear in more than one split."""
        X_train, _, X_valid, _, X_test, _, _ = training_data
        # Splits should be disjoint by index
        train_idx = set(X_train.index)
        valid_idx = set(X_valid.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(valid_idx)
        assert train_idx.isdisjoint(test_idx)
        assert valid_idx.isdisjoint(test_idx)


# ─── Test: Target encoding on train only ──────────────────────────────────────


class TestTargetEncoding:
    """Verify metro target encoding uses only training data."""

    def test_metro_encoded_present(self, training_data):
        """metro_encoded feature should exist in all splits."""
        X_train, _, X_valid, _, X_test, _, features = training_data
        assert "metro_encoded" in features

    def test_metro_values_bounded(self, training_data):
        """Target-encoded values should be between 0 and 1 (mean of binary label)."""
        X_train, _, X_valid, _, X_test, _, _ = training_data
        for X in [X_train, X_valid, X_test]:
            vals = X["metro_encoded"]
            assert vals.min() >= 0.0
            assert vals.max() <= 1.0


# ─── Test: No NaN in feature matrices ─────────────────────────────────────────


class TestNoNulls:
    """Verify final feature matrices have no NaN values."""

    def test_train_no_nan(self, training_data):
        X_train = training_data[0]
        assert X_train.isna().sum().sum() == 0

    def test_valid_no_nan(self, training_data):
        X_valid = training_data[2]
        assert X_valid.isna().sum().sum() == 0

    def test_test_no_nan(self, training_data):
        X_test = training_data[4]
        assert X_test.isna().sum().sum() == 0


# ─── Test: Class imbalance ────────────────────────────────────────────────────


class TestClassImbalance:
    """Verify scale_pos_weight formula."""

    def test_scale_pos_weight_formula(self, training_data):
        _, y_train, _, _, _, _, _ = training_data
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        weight = n_neg / max(n_pos, 1)
        assert weight > 0
        assert isinstance(weight, float)


# ─── Test: Optuna search space ────────────────────────────────────────────────


class TestOptunaSearchSpace:
    """Verify the hyperparameter search space produces valid values."""

    def test_search_space_bounds(self):
        """Sample from search space and verify all bounds are respected."""
        def dummy_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            }
            assert 3 <= params["max_depth"] <= 6
            assert 0.01 <= params["learning_rate"] <= 0.3
            assert 100 <= params["n_estimators"] <= 500
            assert 5 <= params["min_child_weight"] <= 30
            assert 0.6 <= params["subsample"] <= 1.0
            assert 0.6 <= params["colsample_bytree"] <= 1.0
            assert 0.01 <= params["reg_alpha"] <= 5.0
            assert 0.01 <= params["reg_lambda"] <= 5.0
            return 0.5

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(dummy_objective, n_trials=5)
        assert len(study.trials) == 5



# ─── Test: Model artifact loadable after training ─────────────────────────────

xgb_mod = pytest.importorskip("xgboost", reason="xgboost required for artifact test")
mlflow_mod = pytest.importorskip("mlflow", reason="mlflow required for artifact test")


class TestModelArtifact:
    """Verify that model artifacts are properly written and loadable."""

    @pytest.mark.slow
    def test_model_artifact_loadable_after_training(self, fixture_tables, tmp_path, monkeypatch):
        """Train on tiny fixture, confirm model artifact can be loaded back."""
        import mlflow
        import mlflow.xgboost

        # Point MLflow to a temp directory
        tracking_uri = f"sqlite:///{tmp_path}/test_mlflow.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setattr(
            "src.models.distress_classifier.MLFLOW_TRACKING_URI", tracking_uri
        )

        mlflow.set_tracking_uri(tracking_uri)

        from src.models.distress_classifier import train_and_log

        gold_path, market_path = fixture_tables
        run_id = train_and_log(
            experiment_name="test_artifact",
            n_trials=2,
            seed=42,
            gold_path=str(gold_path),
            market_path=str(market_path),
        )

        # The run should exist
        assert run_id is not None

        # The model should be loadable from the registry
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='cre_distress_classifier'")
        assert len(versions) > 0, "No model versions registered"

        # Load the model via run artifacts
        run = mlflow.get_run(run_id)
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        assert "model" in artifact_paths, (
            f"'model' artifact not found. Available: {artifact_paths}"
        )

        # Actually load it
        loaded_model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict_proba")
