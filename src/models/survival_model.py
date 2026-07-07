"""
Cox Proportional Hazards Survival Model for Time-to-Distress
==============================================================

Estimates how long until a loan enters refinancing distress, not just whether
it will. This complements the XGBoost binary classifier by adding a temporal
dimension — enabling prioritization by urgency, not just probability.

Why Cox PH:
  - Semi-parametric: no distributional assumption on baseline hazard
  - Handles right-censored observations (loans that haven't matured yet)
  - Coefficients are interpretable as hazard ratios
  - Well-established in credit risk (analogous to EAD timing models)

Architecture:
  build_survival_frame() → fit CoxPHFitter → log to MLflow → predict median TtD
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from src.models.features import (
    _load_delta_as_df,
    _build_treasury_lookup,
    _build_cap_rate_lookup,
    _get_market_date_range,
    _compute_features_at_Tobs,
    _lookup_treasury,
    _lookup_cap_rate,
    _apply_onehot,
    REFI_SPREAD_BPS,
    NUMERIC_FEATURES_AT_TOBS,
    ONEHOT_FEATURES,
)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
METRICS_OUTPUT = Path("models/evaluation/survival_model_metrics.json")
PREDICTIONS_OUTPUT = Path("data/gold/loan_survival_predictions")


def build_survival_frame(
    gold_path: str | Path = "data/gold/loan_current_state",
    market_path: str | Path = "data/silver/market_rates",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare survival analysis data with duration and event indicator.

    Returns:
        (df_train, df_valid, df_test, feature_names)
        Each df contains feature columns + 'duration_months' + 'event_observed'
    """
    np.random.seed(seed)

    loans_df = _load_delta_as_df(gold_path)
    market_df = _load_delta_as_df(market_path)

    logger.info(f"Loaded {len(loans_df)} loans, {len(market_df)} market records")

    # Build market lookups
    treasury_lookup = _build_treasury_lookup(market_df)
    cap_rate_lookup = _build_cap_rate_lookup(market_df)
    market_start, market_end = _get_market_date_range(treasury_lookup)

    # Parse dates
    loans_df["maturity_date_parsed"] = pd.to_datetime(loans_df["maturity_date"], errors="coerce")
    loans_df["origination_date_parsed"] = pd.to_datetime(loans_df["origination_date"], errors="coerce")
    loans_df["T_obs"] = loans_df["maturity_date_parsed"] - pd.Timedelta(days=730)

    # Filter to loans with valid dates in market range
    mask = (
        (loans_df["T_obs"] >= pd.Timestamp(market_start))
        & (loans_df["maturity_date_parsed"].notna())
        & (loans_df["origination_date_parsed"].notna())
    )
    loans_df = loans_df[mask].copy()

    # Compute duration: months from origination to maturity (or censoring)
    loans_df["duration_months"] = (
        (loans_df["maturity_date_parsed"] - loans_df["origination_date_parsed"]).dt.days / 30.44
    ).clip(lower=1.0)

    # Event indicator: use the distress_tier from gold layer as proxy
    # 'critical' or 'high' = event observed; 'medium'/'low' = censored
    loans_df["event_observed"] = loans_df["distress_tier"].isin(["critical", "high"]).astype(int)

    # Compute features at T_obs (reuse classifier's feature engineering)
    loans_df = _compute_features_at_Tobs(loans_df, treasury_lookup, cap_rate_lookup)

    # Time-based split
    loans_df["origination_year"] = pd.to_numeric(loans_df["origination_year"], errors="coerce")
    df_train = loans_df[loans_df["origination_year"] <= 2018].copy()
    df_valid = loans_df[loans_df["origination_year"] == 2019].copy()
    df_test = loans_df[loans_df["origination_year"] >= 2020].copy()

    # One-hot encode
    df_train, df_valid, df_test, onehot_cols = _apply_onehot(df_train, df_valid, df_test)

    # Feature columns (same as classifier for consistency)
    feature_cols = [c for c in NUMERIC_FEATURES_AT_TOBS if c in df_train.columns] + onehot_cols
    # Remove any constant columns
    feature_cols = [c for c in feature_cols if df_train[c].nunique() > 1]

    logger.info(
        f"Survival frame: train={len(df_train)}, valid={len(df_valid)}, "
        f"test={len(df_test)}, features={len(feature_cols)}"
    )

    return df_train, df_valid, df_test, feature_cols


def train_survival_model(
    experiment_name: str = "cre_distress",
    seed: int = 42,
    gold_path: str | Path = "data/gold/loan_current_state",
    market_path: str | Path = "data/silver/market_rates",
    penalizer: float = 0.1,
) -> str:
    """
    Train Cox PH model and log to MLflow.

    Returns:
        MLflow run_id
    """
    start_time = time.time()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    logger.info("=" * 60)
    logger.info("SURVIVAL MODEL — Training Pipeline")
    logger.info("=" * 60)

    # Build data
    df_train, df_valid, df_test, feature_cols = build_survival_frame(
        gold_path=gold_path, market_path=market_path, seed=seed
    )

    # Prepare fitting DataFrame
    fit_cols = feature_cols + ["duration_months", "event_observed"]
    train_fit = df_train[fit_cols].astype(float).fillna(0.0)
    valid_fit = df_valid[fit_cols].astype(float).fillna(0.0)
    test_fit = df_test[fit_cols].astype(float).fillna(0.0)

    with mlflow.start_run(run_name=f"survival_cox_ph_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id

        # Fit Cox PH model with regularization
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
        cph.fit(
            train_fit,
            duration_col="duration_months",
            event_col="event_observed",
        )

        # Evaluate concordance
        c_train = cph.concordance_index_
        c_valid = concordance_index(
            valid_fit["duration_months"],
            -cph.predict_partial_hazard(valid_fit[feature_cols].astype(float)),
            valid_fit["event_observed"],
        )
        c_test = concordance_index(
            test_fit["duration_months"],
            -cph.predict_partial_hazard(test_fit[feature_cols].astype(float)),
            test_fit["event_observed"],
        )

        logger.info(f"  Concordance — train: {c_train:.4f}, valid: {c_valid:.4f}, test: {c_test:.4f}")

        # Log to MLflow
        mlflow.log_params({
            "model_type": "CoxPH",
            "penalizer": penalizer,
            "n_features": len(feature_cols),
            "train_size": len(train_fit),
            "valid_size": len(valid_fit),
            "test_size": len(test_fit),
            "event_rate_train": float(train_fit["event_observed"].mean()),
        })
        mlflow.log_metrics({
            "concordance_train": c_train,
            "concordance_valid": c_valid,
            "concordance_test": c_test,
            "log_likelihood": float(cph.log_likelihood_),
        })

        # Coefficient summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        cph.plot(ax=ax)
        ax.set_title("Cox PH Coefficient Forest Plot")
        plt.tight_layout()
        fig.savefig("/tmp/cox_ph_coefficients.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact("/tmp/cox_ph_coefficients.png", "plots")

        # Save metrics JSON
        metrics = {
            "model_name": "cre_distress_survival_cox_ph",
            "mlflow_run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "concordance_train": round(c_train, 4),
                "concordance_valid": round(c_valid, 4),
                "concordance_test": round(c_test, 4),
                "log_likelihood": float(cph.log_likelihood_),
            },
            "params": {
                "penalizer": penalizer,
                "n_features": len(feature_cols),
            },
        }
        METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_OUTPUT, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(METRICS_OUTPUT), "evaluation")

        # Predict survival for all loans
        _predict_and_save_survival(cph, df_train, df_valid, df_test, feature_cols)

    elapsed = time.time() - start_time
    logger.info(f"\nSurvival model complete in {elapsed:.1f}s (run_id: {run_id})")
    return run_id


def _predict_and_save_survival(
    model: CoxPHFitter,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Predict median time-to-distress and survival probabilities for all loans."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.delta_writer import DeltaWriter

    all_loans = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    X = all_loans[feature_cols].astype(float).fillna(0.0)

    # Predict survival function for each loan
    surv_funcs = model.predict_survival_function(X)

    predictions = []
    for i, loan_id in enumerate(all_loans["loan_id"].values):
        sf = surv_funcs.iloc[:, i]

        # Median time to distress (time at which S(t) = 0.5)
        below_50 = sf[sf <= 0.5]
        median_ttd = float(below_50.index[0]) if len(below_50) > 0 else 999.0

        # Survival probabilities at specific horizons
        s_12 = float(sf.iloc[(sf.index - 12).abs().argmin()]) if len(sf) > 0 else 1.0
        s_24 = float(sf.iloc[(sf.index - 24).abs().argmin()]) if len(sf) > 0 else 1.0
        s_36 = float(sf.iloc[(sf.index - 36).abs().argmin()]) if len(sf) > 0 else 1.0

        predictions.append({
            "loan_id": str(loan_id),
            "predicted_median_months_to_distress": round(median_ttd, 1),
            "predicted_survival_prob_12m": round(s_12, 4),
            "predicted_survival_prob_24m": round(s_24, 4),
            "predicted_survival_prob_36m": round(s_36, 4),
        })

    # Write to Delta
    writer = DeltaWriter(PREDICTIONS_OUTPUT)
    writer.write(predictions, mode="overwrite")
    logger.info(f"  Saved {len(predictions)} survival predictions to {PREDICTIONS_OUTPUT}")
