"""
Stress Testing Scenario Engine
================================

Applies macroeconomic stress scenarios to the loan portfolio, recomputes
derived features under shocked conditions, and re-scores each loan using
the registered XGBoost distress classifier.

Architecture:
  Load portfolio → Apply shock → Recompute features → Score with model → Write Delta
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Property-type credit spreads (same as feature engineering)
REFI_SPREAD_BPS: dict[str, float] = {
    "office": 250,
    "retail": 275,
    "industrial": 200,
    "multifamily": 180,
    "hotel": 325,
}

STRESS_OUTPUT = Path("data/gold/stress_test_results")


def load_scenarios(config_path: str | Path = "config/stress_scenarios.yaml") -> list[dict[str, Any]]:
    """Load scenario definitions from YAML config."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.yaml_compat import load_yaml_file

    config = load_yaml_file(str(config_path))
    return config.get("scenarios", [])


def load_portfolio(gold_path: str | Path = "data/gold/loan_current_state") -> pd.DataFrame:
    """Load the current portfolio from Gold Delta table."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.delta_writer import DeltaReader

    records = DeltaReader(gold_path).read()
    df = pd.DataFrame(records)

    # Ensure numeric columns
    numeric_cols = [
        "original_balance", "current_balance", "note_rate", "noi_annual",
        "current_cap_rate", "current_ltv", "new_dscr", "refinance_rate",
        "rate_gap", "rate_gap_bps", "debt_yield", "occupancy_pct",
        "ltv_at_origination", "dscr_at_origination",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_classifier():
    """Load the registered XGBoost classifier from MLflow."""
    import mlflow
    import mlflow.xgboost

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    # Strategy 1: Alias syntax (MLflow 3.x)
    try:
        model = mlflow.xgboost.load_model("models:/cre_distress_classifier@Staging")
        logger.info("✓ Loaded classifier via alias: @Staging")
        return model
    except Exception:
        pass

    # Strategy 2: Latest version
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='cre_distress_classifier'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            model = mlflow.xgboost.load_model(f"models:/cre_distress_classifier/{latest.version}")
            logger.info(f"✓ Loaded classifier via version: v{latest.version}")
            return model
    except Exception:
        pass

    # Strategy 3: Search runs
    try:
        runs = mlflow.search_runs(experiment_names=["cre_distress"], order_by=["start_time DESC"], max_results=5)
        client = mlflow.tracking.MlflowClient()
        for _, row in runs.iterrows():
            artifacts = client.list_artifacts(row["run_id"])
            if any(a.path == "model" for a in artifacts):
                model = mlflow.xgboost.load_model(f"runs:/{row['run_id']}/model")
                logger.info(f"✓ Loaded classifier from run {row['run_id'][:8]}...")
                return model
    except Exception:
        pass

    raise RuntimeError(
        "Could not load cre_distress_classifier from MLflow. "
        "Ensure the model is trained and registered. Run: python -m src.models.train_cli"
    )


def apply_scenario(
    df: pd.DataFrame,
    scenario: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply a stress scenario to the portfolio by shocking features.

    Returns a copy of the DataFrame with stressed feature values.
    """
    stressed = df.copy()
    name = scenario.get("name", "unknown")
    rate_shock = float(scenario.get("rate_shock_bps", 0))
    cap_shock = float(scenario.get("cap_rate_shock_bps", 0))
    noi_shock = float(scenario.get("noi_shock_pct", 0.0))
    ptype_filter = scenario.get("property_type_filter")

    # Determine which loans are affected
    if ptype_filter and ptype_filter != "null":
        mask = stressed["property_type"].astype(str).str.lower() == ptype_filter.lower()
    else:
        mask = pd.Series(True, index=stressed.index)

    # Apply rate shock: increases refinance rate
    if rate_shock != 0:
        stressed.loc[mask, "refinance_rate"] = (
            stressed.loc[mask, "refinance_rate"] + rate_shock / 10000.0
        )

    # Apply cap rate shock: increases cap rate → decreases value → increases LTV
    if cap_shock != 0:
        stressed.loc[mask, "current_cap_rate"] = (
            stressed.loc[mask, "current_cap_rate"] + cap_shock / 100.0
        )

    # Apply NOI shock: reduces NOI → reduces value and DSCR
    if noi_shock != 0:
        stressed.loc[mask, "noi_annual"] = (
            stressed.loc[mask, "noi_annual"] * (1.0 + noi_shock)
        )

    # Recompute downstream features
    stressed = _recompute_features(stressed, mask)

    stressed["scenario_name"] = name
    return stressed


def _recompute_features(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Recompute derived features after applying shocks."""
    # Property value = NOI / cap_rate (cap_rate in %)
    cap_decimal = df.loc[mask, "current_cap_rate"] / 100.0
    noi = df.loc[mask, "noi_annual"]
    balance = df.loc[mask, "current_balance"]

    # Recompute current value
    property_value = np.where(cap_decimal > 0, noi / cap_decimal, 0)
    df.loc[mask, "current_value"] = property_value

    # Recompute LTV
    df.loc[mask, "current_ltv"] = np.where(property_value > 0, balance / property_value, 99.0)

    # Recompute rate gap
    df.loc[mask, "rate_gap"] = df.loc[mask, "refinance_rate"] - df.loc[mask, "note_rate"]
    df.loc[mask, "rate_gap_bps"] = df.loc[mask, "rate_gap"] * 10000

    # Recompute DSCR (IO: balance * rate; amortizing: standard formula)
    refi_rate = df.loc[mask, "refinance_rate"].values
    is_io = (df.loc[mask, "amortization_type"].astype(str) == "interest_only").values
    bal = balance.values

    monthly_rate = refi_rate / 12.0
    safe_mr = np.where(monthly_rate > 0, monthly_rate, 1e-10)
    pmt_factor = (safe_mr * (1 + safe_mr) ** 360) / ((1 + safe_mr) ** 360 - 1)
    amort_ds = bal * pmt_factor * 12

    annual_ds = np.where(is_io, bal * refi_rate, amort_ds)
    noi_vals = noi.values
    new_dscr = np.where(annual_ds > 0, noi_vals / annual_ds, 0)
    df.loc[mask, "new_dscr"] = new_dscr

    # Recompute debt yield
    df.loc[mask, "debt_yield"] = np.where(bal > 0, noi_vals / bal, 0)

    return df


def score_portfolio(
    df: pd.DataFrame,
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Score loans using the XGBoost model under stressed features."""
    # Build feature matrix matching the model's expected input
    X = df[feature_names].astype(float).fillna(0.0)
    proba = model.predict_proba(X)[:, 1]

    df["stressed_pd"] = proba
    df["stressed_distress_tier"] = pd.cut(
        proba,
        bins=[-0.01, 0.40, 0.70, 1.01],
        labels=["low", "medium", "high"],
    )
    return df


def run_scenario(
    scenario: dict[str, Any],
    portfolio: pd.DataFrame,
    model: Any,
    feature_names: list[str],
    baseline_pds: pd.Series | None = None,
) -> pd.DataFrame:
    """Execute a single stress scenario end-to-end."""
    name = scenario.get("name", "unknown")

    # Apply shocks
    stressed = apply_scenario(portfolio, scenario)

    # Score
    stressed = score_portfolio(stressed, model, feature_names)

    # Add baseline comparison
    if baseline_pds is not None:
        stressed["baseline_pd"] = baseline_pds.values
        stressed["delta_pd"] = stressed["stressed_pd"] - stressed["baseline_pd"]
    else:
        stressed["baseline_pd"] = stressed["stressed_pd"]
        stressed["delta_pd"] = 0.0

    # Select output columns
    output_cols = [
        "loan_id", "scenario_name", "property_type", "metro_area",
        "stressed_pd", "stressed_distress_tier",
        "current_ltv", "new_dscr", "rate_gap_bps",
        "baseline_pd", "delta_pd",
        "current_balance", "noi_annual",
    ]
    output_cols = [c for c in output_cols if c in stressed.columns]

    result = stressed[output_cols].copy()
    result.rename(columns={
        "current_ltv": "stressed_ltv",
        "new_dscr": "stressed_dscr",
        "rate_gap_bps": "stressed_refinance_gap",
    }, inplace=True)

    return result


def run_all_scenarios(
    scenarios: list[dict[str, Any]],
    gold_path: str | Path = "data/gold/loan_current_state",
    output_path: str | Path = STRESS_OUTPUT,
    feature_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run all stress scenarios and save results.

    Returns dict of scenario_name → result DataFrame.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.delta_writer import DeltaWriter

    output_path = Path(output_path)

    logger.info("Loading portfolio...")
    portfolio = load_portfolio(gold_path)
    logger.info(f"  Portfolio: {len(portfolio)} loans")

    logger.info("Loading classifier...")
    model = load_classifier()

    # Determine feature names from model
    if feature_names is None:
        try:
            feature_names = model.get_booster().feature_names
        except Exception:
            # Fallback: use numeric columns from portfolio
            feature_names = [
                c for c in portfolio.select_dtypes(include=[np.number]).columns
                if c not in ("origination_year",)
            ]

    # Run baseline first
    baseline_scenario = {"name": "baseline", "rate_shock_bps": 0, "cap_rate_shock_bps": 0, "noi_shock_pct": 0.0}
    baseline_result = run_scenario(baseline_scenario, portfolio, model, feature_names)
    baseline_pds = baseline_result["stressed_pd"]

    results: dict[str, pd.DataFrame] = {}
    results["baseline"] = baseline_result

    for scenario in scenarios:
        name = scenario.get("name", "unknown")
        if name == "baseline":
            continue  # Already computed

        logger.info(f"  Running scenario: {name}...")
        result = run_scenario(scenario, portfolio, model, feature_names, baseline_pds)
        results[name] = result

        # Summary line
        pct_distressed = (result["stressed_pd"] > 0.5).mean() * 100
        baseline_pct = (baseline_pds > 0.5).mean() * 100
        delta = pct_distressed - baseline_pct
        logger.info(f"    {name}: {pct_distressed:.1f}% distressed vs {baseline_pct:.1f}% baseline ({delta:+.1f}pp)")

    # Save all results
    all_results = pd.concat(results.values(), ignore_index=True)
    writer = DeltaWriter(output_path)
    writer.write(all_results.to_dict("records"), partition_by="scenario_name", mode="overwrite")
    logger.info(f"\n  Saved {len(all_results)} stress test records to {output_path}")

    return results
