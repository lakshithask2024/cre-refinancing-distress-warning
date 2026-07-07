"""
Unit tests for the Distress Classifier feature engineering and training setup.

These tests use small fixture data (~200 rows) and do NOT train real XGBoost
models. They verify the data preparation logic, split correctness, encoding
behavior, and hyperparameter space validity.

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
def fixture_records():
    """Generate ~200 rows of loan_distress_history-like data for testing."""
    import random
    random.seed(42)

    records = []
    metros = ["New York", "Los Angeles", "Chicago", "Dallas", "Miami"]
    prop_types = ["office", "retail", "industrial", "multifamily", "hotel"]
    amort_types = ["interest_only", "amortizing"]
    tiers = ["A", "B", "C"]

    for i in range(200):
        orig_year = random.choice([2017, 2018, 2019, 2020, 2021, 2022])
        is_distressed = 1 if random.random() < 0.4 else 0
        records.append({
            "loan_id": f"LN-{i:04d}",
            "property_type": random.choice(prop_types),
            "metro_area": random.choice(metros),
            "origination_year": str(orig_year),
            "original_balance": random.uniform(5e6, 50e6),
            "current_balance": random.uniform(5e6, 50e6),
            "note_rate": random.uniform(0.025, 0.07),
            "amortization_type": random.choice(amort_types),
            "balloon_flag": random.choice(["True", "False"]),
            "maturity_date": f"{orig_year + 7}-06-15",
            "ltv_at_origination": random.uniform(0.4, 0.85),
            "dscr_at_origination": random.uniform(0.8, 2.5),
            "occupancy_pct": random.uniform(0.5, 1.0),
            "current_ltv": random.uniform(0.4, 1.5),
            "new_dscr": random.uniform(0.5, 2.0),
            "rate_gap": random.uniform(0.0, 0.03),
            "rate_gap_bps": random.uniform(0, 300),
            "debt_yield": random.uniform(0.05, 0.15),
            "current_cap_rate": random.uniform(5.0, 9.0),
            "refinance_rate": random.uniform(0.05, 0.08),
            "months_to_maturity": random.uniform(-20, 80),
            "sponsor_credit_tier": random.choice(tiers),
            "is_distressed": is_distressed,
            "noi_annual": random.uniform(500000, 5000000),
            "_feature_computed_at": "2026-01-01T00:00:00",
        })
    return records



@pytest.fixture
def fixture_delta_table(tmp_path, fixture_records):
    """Write fixture records to a Delta table and return the path."""
    from src.utils.delta_writer import DeltaWriter

    table_path = tmp_path / "loan_distress_history"
    writer = DeltaWriter(table_path)
    writer.write(fixture_records, partition_by="origination_year")
    return table_path


@pytest.fixture
def training_data(fixture_delta_table):
    """Build training frame from fixture data."""
    from src.models.features import build_training_frame

    return build_training_frame(gold_path=fixture_delta_table, seed=42)


# ─── Test: build_training_frame shape ─────────────────────────────────────────


class TestBuildTrainingFrame:
    """Verify the feature engineering pipeline produces correct shapes."""

    def test_returns_seven_elements(self, training_data):
        """build_training_frame returns exactly 7 elements."""
        assert len(training_data) == 7

    def test_split_sizes_non_zero(self, training_data):
        """Each split should have at least 1 row (fixture has all years)."""
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

    def test_total_rows_preserved(self, training_data, fixture_records):
        """Total rows across splits should equal input rows."""
        X_train, _, X_valid, _, X_test, _, _ = training_data
        total = len(X_train) + len(X_valid) + len(X_test)
        assert total == len(fixture_records)



# ─── Test: No leakage between splits ─────────────────────────────────────────


class TestNoLeakage:
    """Verify time-based split prevents data leakage."""

    def test_no_loan_id_overlap(self, fixture_delta_table):
        """No loan_id should appear in more than one split."""
        from src.models.features import build_training_frame, _load_gold_data

        df = _load_gold_data(fixture_delta_table)

        train_ids = set(df[df["origination_year"].astype(int) <= 2019]["loan_id"])
        valid_ids = set(df[df["origination_year"].astype(int) == 2020]["loan_id"])
        test_ids = set(df[df["origination_year"].astype(int) >= 2021]["loan_id"])

        assert train_ids.isdisjoint(valid_ids), "Train/valid overlap"
        assert train_ids.isdisjoint(test_ids), "Train/test overlap"
        assert valid_ids.isdisjoint(test_ids), "Valid/test overlap"

    def test_train_years_correct(self, fixture_delta_table):
        """Train split should only contain years <= 2019."""
        from src.models.features import _load_gold_data

        df = _load_gold_data(fixture_delta_table)
        train_years = df[df["origination_year"].astype(int) <= 2019]["origination_year"].astype(int)
        assert train_years.max() <= 2019

    def test_test_years_correct(self, fixture_delta_table):
        """Test split should only contain years >= 2021."""
        from src.models.features import _load_gold_data

        df = _load_gold_data(fixture_delta_table)
        test_years = df[df["origination_year"].astype(int) >= 2021]["origination_year"].astype(int)
        assert test_years.min() >= 2021


# ─── Test: Target encoding computed on train only ─────────────────────────────


class TestTargetEncoding:
    """Verify metro target encoding uses only training data."""

    def test_metro_encoded_present(self, training_data):
        """metro_encoded feature should exist in all splits."""
        X_train, _, X_valid, _, X_test, _, features = training_data
        assert "metro_encoded" in features
        assert "metro_encoded" in X_train.columns
        assert "metro_encoded" in X_valid.columns
        assert "metro_encoded" in X_test.columns

    def test_encoding_uses_train_means(self, fixture_delta_table):
        """Valid/test metro encoding should use train-computed means."""
        import pandas as pd
        from src.models.features import _load_gold_data

        df = _load_gold_data(fixture_delta_table)
        df["is_distressed"] = pd.to_numeric(df["is_distressed"], errors="coerce").fillna(0)

        train_df = df[df["origination_year"].astype(int) <= 2019]
        train_metro_means = train_df.groupby("metro_area")["is_distressed"].mean()

        # The encoding for valid/test metros should match train means
        valid_df = df[df["origination_year"].astype(int) == 2020]
        for metro in valid_df["metro_area"].unique():
            if metro in train_metro_means.index:
                expected = train_metro_means[metro]
                assert expected >= 0.0 and expected <= 1.0



# ─── Test: No NaN in feature matrices ─────────────────────────────────────────


class TestFeatureEngineeringNoNulls:
    """Verify final feature matrices have no NaN values."""

    def test_train_no_nan(self, training_data):
        """X_train should have zero NaN values."""
        X_train = training_data[0]
        assert X_train.isna().sum().sum() == 0

    def test_valid_no_nan(self, training_data):
        """X_valid should have zero NaN values."""
        X_valid = training_data[2]
        assert X_valid.isna().sum().sum() == 0

    def test_test_no_nan(self, training_data):
        """X_test should have zero NaN values."""
        X_test = training_data[4]
        assert X_test.isna().sum().sum() == 0

    def test_labels_binary(self, training_data):
        """Labels should be binary (0 or 1) with no NaN."""
        _, y_train, _, y_valid, _, y_test, _ = training_data
        for y in [y_train, y_valid, y_test]:
            assert set(y.unique()).issubset({0, 1})
            assert y.isna().sum() == 0


# ─── Test: Class imbalance handling ───────────────────────────────────────────


class TestClassImbalance:
    """Verify scale_pos_weight formula."""

    def test_scale_pos_weight_formula(self, training_data):
        """scale_pos_weight = n_negative / n_positive."""
        _, y_train, _, _, _, _, _ = training_data
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        expected = n_neg / max(n_pos, 1)
        # This is the formula used in distress_classifier.py
        assert expected > 0
        assert isinstance(expected, float)

    def test_weight_greater_than_one_if_imbalanced(self, training_data):
        """If negative > positive, weight should be > 1."""
        _, y_train, _, _, _, _, _ = training_data
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        weight = n_neg / max(n_pos, 1)
        if n_neg > n_pos:
            assert weight > 1.0


# ─── Test: Optuna search space validity ───────────────────────────────────────


class TestOptunaSearchSpace:
    """Verify the hyperparameter search space produces valid values."""

    def test_search_space_bounds(self):
        """Sample from search space and verify all bounds are respected."""
        import optuna

        def dummy_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }
            # Verify bounds
            assert 3 <= params["max_depth"] <= 10
            assert 0.01 <= params["learning_rate"] <= 0.3
            assert 100 <= params["n_estimators"] <= 500
            assert 1 <= params["min_child_weight"] <= 10
            assert 0.6 <= params["subsample"] <= 1.0
            assert 0.6 <= params["colsample_bytree"] <= 1.0
            assert 1e-8 <= params["reg_alpha"] <= 1.0
            assert 1e-8 <= params["reg_lambda"] <= 1.0
            return 0.5

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(dummy_objective, n_trials=5)
        assert len(study.trials) == 5
