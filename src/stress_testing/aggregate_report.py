"""
Stress Test Aggregate Reporting
=================================

Computes portfolio-level summary metrics for each stress scenario
and exports results as both Delta table and human-readable CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUMMARY_OUTPUT = Path("data/gold/stress_test_summary")
CSV_OUTPUT = Path("reports/stress_summary.csv")


def compute_aggregate_report(
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute portfolio-level metrics for each scenario.

    Returns DataFrame with one row per scenario.
    """
    summaries = []

    for scenario_name, df in results.items():
        total_loans = len(df)
        if total_loans == 0:
            continue

        distressed_mask = df["stressed_pd"] > 0.5
        upb_col = "current_balance" if "current_balance" in df.columns else None

        summary: dict[str, Any] = {
            "scenario_name": scenario_name,
            "total_loans": total_loans,
            "pct_loans_distressed": round(distressed_mask.mean() * 100, 2),
            "avg_predicted_pd": round(df["stressed_pd"].mean(), 4),
            "median_predicted_pd": round(df["stressed_pd"].median(), 4),
            "avg_delta_pd": round(df["delta_pd"].mean(), 4) if "delta_pd" in df.columns else 0,
        }

        if upb_col and upb_col in df.columns:
            total_upb = pd.to_numeric(df[upb_col], errors="coerce").sum()
            distressed_upb = pd.to_numeric(df.loc[distressed_mask, upb_col], errors="coerce").sum()
            summary["total_upb"] = round(total_upb, 0)
            summary["total_upb_distressed"] = round(distressed_upb, 0)
            summary["pct_upb_distressed"] = round(distressed_upb / total_upb * 100, 2) if total_upb > 0 else 0

        # Worst-hit metros (top 5 by avg delta_pd)
        if "delta_pd" in df.columns and "metro_area" in df.columns:
            metro_impact = df.groupby("metro_area")["delta_pd"].mean().sort_values(ascending=False)
            summary["worst_metros"] = ", ".join(metro_impact.head(5).index.tolist())

        # Worst-hit property types
        if "delta_pd" in df.columns and "property_type" in df.columns:
            ptype_impact = df.groupby("property_type")["delta_pd"].mean().sort_values(ascending=False)
            summary["worst_property_types"] = ", ".join(ptype_impact.head(5).index.tolist())

        summaries.append(summary)

    return pd.DataFrame(summaries)


def save_report(
    summary_df: pd.DataFrame,
    output_delta: str | Path = SUMMARY_OUTPUT,
    output_csv: str | Path = CSV_OUTPUT,
) -> None:
    """Save aggregate report as Delta table and CSV."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.delta_writer import DeltaWriter

    # Delta
    output_delta = Path(output_delta)
    writer = DeltaWriter(output_delta)
    writer.write(summary_df.to_dict("records"), mode="overwrite")
    logger.info(f"  Summary Delta saved to {output_delta}")

    # CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    logger.info(f"  Summary CSV saved to {output_csv}")
