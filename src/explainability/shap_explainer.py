"""
SHAP Explainability for the CRE Distress Classifier
=====================================================

Computes SHAP values (TreeExplainer for XGBoost) to explain individual loan
risk predictions. Produces global importance plots, beeswarm summaries,
dependence plots, and per-loan explanation tables.

Why SHAP over LIME:
  - Exact Shapley values for tree models (not approximations)
  - Additive: sum of SHAP values + expected = prediction (verifiable)
  - Both global (feature importance) and local (per-loan) explanations
  - Consistent with the SR 11-7 requirement for model interpretability
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.models.features import build_training_frame

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("docs/model_risk_management/figures")
EXPLANATIONS_OUTPUT = Path("data/gold/loan_shap_explanations")


def compute_shap_explanations(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    feature_names: list[str],
    n_sample: int = 1000,
    n_top_risk: int = 100,
    gold_path: str | Path = "data/gold/loan_current_state",
    market_path: str | Path = "data/silver/market_rates",
) -> dict[str, Any]:
    """
    Compute SHAP values and generate all explanation artifacts.

    Returns:
        Summary dict with paths to generated artifacts
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SHAP EXPLAINABILITY — Computing")
    logger.info("=" * 60)

    # TreeExplainer for exact SHAP values
    explainer = shap.TreeExplainer(model)

    # Sample for global analysis
    if len(X_test) > n_sample:
        sample_idx = np.random.choice(len(X_test), n_sample, replace=False)
        X_sample = X_test.iloc[sample_idx]
    else:
        X_sample = X_test

    logger.info(f"  Computing SHAP values for {len(X_sample)} loans...")
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value

    # Global importance plot
    _plot_global_importance(shap_values, X_sample, feature_names)

    # Beeswarm plot
    _plot_beeswarm(shap_values, X_sample, feature_names)

    # Dependence plots for top features
    _plot_dependence(shap_values, X_sample, feature_names, "current_ltv_at_Tobs")
    _plot_dependence(shap_values, X_sample, feature_names, "current_dscr_at_Tobs")

    # Top-risk loans: highest predicted probability
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    top_risk_idx = np.argsort(y_pred_proba)[::-1][:n_top_risk]
    X_top_risk = X_test.iloc[top_risk_idx]
    shap_top_risk = explainer.shap_values(X_top_risk)

    # Save per-loan explanations
    _save_loan_explanations(X_top_risk, shap_top_risk, feature_names, y_pred_proba[top_risk_idx])

    logger.info("  SHAP artifacts saved to docs/model_risk_management/figures/")
    return {
        "n_sample": len(X_sample),
        "n_top_risk": len(X_top_risk),
        "expected_value": float(expected_value) if np.isscalar(expected_value) else float(expected_value[0]),
        "top_feature": feature_names[np.argmax(np.abs(shap_values).mean(axis=0))],
    }


def get_loan_explanation(
    loan_id: str,
    model: xgb.XGBClassifier,
    X_all: pd.DataFrame,
    loan_ids: pd.Series,
    feature_names: list[str],
    survival_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Get a complete explanation for a single loan.

    Returns dict with predicted PD, median TtD, and top-5 SHAP drivers.
    Used by the Power BI drill-through page for loan-level narratives.
    """
    mask = loan_ids == loan_id
    if not mask.any():
        return {"error": f"loan_id '{loan_id}' not found"}

    X_loan = X_all[mask]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_loan)[0]

    pred_proba = float(model.predict_proba(X_loan)[:, 1][0])

    # Rank features by absolute SHAP contribution
    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[::-1][:5]

    drivers = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        feat_val = float(X_loan.iloc[0, idx])
        shap_val = float(shap_vals[idx])
        drivers.append({
            "feature": feat_name,
            "value": round(feat_val, 4),
            "shap": round(shap_val, 4),
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
        })

    result: dict[str, Any] = {
        "loan_id": loan_id,
        "predicted_pd": round(pred_proba, 4),
        "top_drivers": drivers,
    }

    # Add survival prediction if available
    if survival_predictions is not None:
        surv_row = survival_predictions[survival_predictions["loan_id"] == loan_id]
        if len(surv_row) > 0:
            result["predicted_median_months_to_distress"] = float(
                surv_row.iloc[0].get("predicted_median_months_to_distress", 0)
            )

    return result


# ─── Plotting helpers ─────────────────────────────────────────────────────────


def _plot_global_importance(
    shap_values: np.ndarray, X: pd.DataFrame, feature_names: list[str]
) -> None:
    """Bar chart of mean |SHAP| values (global feature importance)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:20]

    ax.barh(
        range(len(sorted_idx)),
        mean_abs[sorted_idx][::-1],
        align="center",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1])
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_beeswarm(
    shap_values: np.ndarray, X: pd.DataFrame, feature_names: list[str]
) -> None:
    """Beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def _plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    feature: str,
) -> None:
    """Dependence plot for a specific feature."""
    if feature not in feature_names:
        logger.warning(f"  Feature '{feature}' not in feature list, skipping dependence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    idx = feature_names.index(feature)
    shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, ax=ax, show=False)
    ax.set_title(f"SHAP Dependence: {feature}")
    plt.tight_layout()
    safe_name = feature.replace("/", "_")
    fig.savefig(FIGURES_DIR / f"shap_dependence_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_loan_explanations(
    X_top: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    pred_proba: np.ndarray,
) -> None:
    """Save per-loan SHAP breakdowns to Delta table."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.delta_writer import DeltaWriter

    records = []
    for i in range(len(X_top)):
        loan_shap = shap_values[i]
        sorted_idx = np.argsort(np.abs(loan_shap))[::-1]

        for rank, feat_idx in enumerate(sorted_idx[:10]):  # top 10 per loan
            records.append({
                "loan_id": str(X_top.index[i]) if hasattr(X_top.index, '__iter__') else f"loan_{i}",
                "predicted_pd": round(float(pred_proba[i]), 4),
                "feature_name": feature_names[feat_idx],
                "feature_value": round(float(X_top.iloc[i, feat_idx]), 4),
                "shap_value": round(float(loan_shap[feat_idx]), 4),
                "rank_in_loan": rank + 1,
            })

    EXPLANATIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = DeltaWriter(EXPLANATIONS_OUTPUT)
    writer.write(records, mode="overwrite")
    logger.info(f"  Saved {len(records)} SHAP explanations to {EXPLANATIONS_OUTPUT}")
