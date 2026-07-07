"""
Unit tests for the SHAP Explainability layer.

Tests verify SHAP computation correctness and API contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pd = pytest.importorskip("pandas", reason="pandas required")
np = pytest.importorskip("numpy", reason="numpy required")
xgb_mod = pytest.importorskip("xgboost", reason="xgboost required")
shap_mod = pytest.importorskip("shap", reason="shap required")


@pytest.fixture
def simple_model_and_data():
    """Train a tiny XGBoost model on synthetic data for SHAP testing."""
    import xgboost as xgb

    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feat_a": np.random.randn(n),
        "feat_b": np.random.randn(n),
        "feat_c": np.random.randn(n),
        "feat_d": np.random.randn(n),
        "feat_e": np.random.randn(n),
    })
    y = ((X["feat_a"] + X["feat_b"]) > 0).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )
    model.fit(X, y)

    feature_names = list(X.columns)
    loan_ids = pd.Series([f"LN-{i:04d}" for i in range(n)])
    return model, X, y, feature_names, loan_ids


class TestShapValues:
    """Test SHAP value computation properties."""

    def test_shap_values_sum_to_prediction(self, simple_model_and_data):
        """SHAP contributions + expected_value should ≈ predicted logit."""
        import shap

        model, X, y, feature_names, _ = simple_model_and_data
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[:10])
        expected = explainer.expected_value

        # For binary classification, check that shap sum ≈ margin output
        margins = model.predict(X.iloc[:10], output_margin=True)
        for i in range(10):
            shap_sum = shap_values[i].sum() + expected
            assert abs(shap_sum - margins[i]) < 1e-4, (
                f"Row {i}: SHAP sum {shap_sum:.6f} != margin {margins[i]:.6f}"
            )

    def test_get_loan_explanation_returns_top5(self, simple_model_and_data):
        """get_loan_explanation should return exactly 5 drivers."""
        from src.explainability.shap_explainer import get_loan_explanation

        model, X, y, feature_names, loan_ids = simple_model_and_data
        result = get_loan_explanation(
            loan_id="LN-0001",
            model=model,
            X_all=X,
            loan_ids=loan_ids,
            feature_names=feature_names,
        )
        assert "top_drivers" in result
        assert len(result["top_drivers"]) == 5
        assert "predicted_pd" in result
        assert 0.0 <= result["predicted_pd"] <= 1.0

    def test_explanation_drivers_have_required_fields(self, simple_model_and_data):
        """Each driver should have feature, value, shap, direction."""
        from src.explainability.shap_explainer import get_loan_explanation

        model, X, y, feature_names, loan_ids = simple_model_and_data
        result = get_loan_explanation(
            loan_id="LN-0005",
            model=model,
            X_all=X,
            loan_ids=loan_ids,
            feature_names=feature_names,
        )
        for driver in result["top_drivers"]:
            assert "feature" in driver
            assert "value" in driver
            assert "shap" in driver
            assert "direction" in driver
            assert driver["direction"] in ("increases_risk", "decreases_risk")

    def test_loan_not_found_returns_error(self, simple_model_and_data):
        """Non-existent loan_id should return error dict."""
        from src.explainability.shap_explainer import get_loan_explanation

        model, X, y, feature_names, loan_ids = simple_model_and_data
        result = get_loan_explanation(
            loan_id="NONEXISTENT",
            model=model,
            X_all=X,
            loan_ids=loan_ids,
            feature_names=feature_names,
        )
        assert "error" in result

    @pytest.mark.slow
    def test_shap_explanations_saved(self, simple_model_and_data, tmp_path):
        """compute_shap_explanations should save Delta table."""
        from src.explainability.shap_explainer import _save_loan_explanations

        model, X, y, feature_names, _ = simple_model_and_data
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[:10])
        pred_proba = model.predict_proba(X.iloc[:10])[:, 1]

        import src.explainability.shap_explainer as se
        original_output = se.EXPLANATIONS_OUTPUT
        se.EXPLANATIONS_OUTPUT = tmp_path / "shap_explanations"

        _save_loan_explanations(X.iloc[:10], shap_values, feature_names, pred_proba)

        # Verify Delta table written
        from src.utils.delta_writer import DeltaReader
        reader = DeltaReader(se.EXPLANATIONS_OUTPUT)
        records = reader.read()
        assert len(records) > 0
        assert "feature_name" in records[0]
        assert "shap_value" in records[0]
        assert "rank_in_loan" in records[0]

        se.EXPLANATIONS_OUTPUT = original_output
