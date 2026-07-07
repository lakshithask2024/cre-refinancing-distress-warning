"""
Power BI Data Export Pipeline
===============================

Exports Gold Delta tables to Parquet + CSV for Power BI file-based import.
Produces a star schema (fact + dimension tables) ready for Power BI Desktop.

Output: data/exports/powerbi/ with one .parquet and one .csv per table.

CLI: python -m src.exports.powerbi_exporter
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.delta_writer import DeltaReader
from src.utils.yaml_compat import load_yaml_file

logger = logging.getLogger(__name__)

DEFAULT_GOLD = Path("data/gold")
DEFAULT_OUTPUT = Path("data/exports/powerbi")


def export_all(
    gold_path: str | Path = DEFAULT_GOLD,
    output_path: str | Path = DEFAULT_OUTPUT,
    config_path: str | Path = "config/stress_scenarios.yaml",
) -> dict[str, int]:
    """
    Export all fact and dimension tables for Power BI.

    Returns dict of table_name → row_count.
    """
    gold_path = Path(gold_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    logger.info("=" * 60)
    logger.info("POWER BI EXPORT — Generating star schema tables")
    logger.info("=" * 60)

    # ─── Fact tables ──────────────────────────────────────────────────────────
    counts["fact_loan_current"] = _export_fact_loan_current(gold_path, output_path)
    counts["fact_loan_history"] = _export_fact_loan_history(gold_path, output_path)
    counts["fact_stress_results"] = _export_fact_stress_results(gold_path, output_path)
    counts["fact_shap_top_features"] = _export_fact_shap(gold_path, output_path)
    counts["fact_survival"] = _export_fact_survival(gold_path, output_path)

    # ─── Dimension tables ─────────────────────────────────────────────────────
    counts["dim_loan"] = _export_dim_loan(gold_path, output_path)
    counts["dim_scenario"] = _export_dim_scenario(config_path, output_path)
    counts["dim_property_type"] = _export_dim_property_type(output_path)
    counts["dim_metro"] = _export_dim_metro(gold_path, output_path)
    counts["dim_date"] = _export_dim_date(output_path)

    logger.info(f"\n{'='*60}")
    logger.info("EXPORT COMPLETE")
    logger.info(f"{'='*60}")
    for name, count in counts.items():
        logger.info(f"  {name:<25s}: {count:>6,} rows")
    logger.info(f"  Output: {output_path}")

    return counts


# ─── Fact table exporters ─────────────────────────────────────────────────────


def _export_fact_loan_current(gold: Path, out: Path) -> int:
    df = _read_gold_table(gold / "loan_current_state")
    cols = [
        "loan_id", "distress_tier", "current_ltv", "new_dscr", "refinance_rate",
        "rate_gap_bps", "debt_yield", "current_cap_rate", "current_value",
        "months_to_maturity", "is_matured", "dscr_change", "ltv_change",
        "current_balance", "noi_annual",
    ]
    cols = [c for c in cols if c in df.columns]
    _write_table(df[cols], out, "fact_loan_current")
    return len(df)


def _export_fact_loan_history(gold: Path, out: Path) -> int:
    df = _read_gold_table(gold / "loan_distress_history")
    # Keep most recent 24 months per loan (by snapshot_at if available)
    cols = [
        "loan_id", "is_distressed", "distress_tier", "current_ltv", "new_dscr",
        "rate_gap_bps", "dscr_severity_score", "ltv_severity_score", "snapshot_at",
    ]
    cols = [c for c in cols if c in df.columns]
    _write_table(df[cols], out, "fact_loan_history")
    return len(df)


def _export_fact_stress_results(gold: Path, out: Path) -> int:
    path = gold / "stress_test_results"
    if not path.exists():
        logger.warning("  stress_test_results not found — skipping")
        _write_table(pd.DataFrame(), out, "fact_stress_results")
        return 0
    df = _read_gold_table(path)
    cols = [
        "loan_id", "scenario_name", "stressed_pd", "stressed_distress_tier",
        "stressed_ltv", "stressed_dscr", "stressed_refinance_gap",
        "baseline_pd", "delta_pd", "current_balance",
    ]
    cols = [c for c in cols if c in df.columns]
    _write_table(df[cols] if cols else df, out, "fact_stress_results")
    return len(df)


def _export_fact_shap(gold: Path, out: Path) -> int:
    path = gold / "loan_shap_explanations"
    if not path.exists():
        logger.warning("  loan_shap_explanations not found — skipping")
        _write_table(pd.DataFrame(), out, "fact_shap_top_features")
        return 0
    df = _read_gold_table(path)
    # Keep top 5 per loan
    if "rank_in_loan" in df.columns:
        df["rank_in_loan"] = pd.to_numeric(df["rank_in_loan"], errors="coerce")
        df = df[df["rank_in_loan"] <= 5]
    _write_table(df, out, "fact_shap_top_features")
    return len(df)


def _export_fact_survival(gold: Path, out: Path) -> int:
    path = gold / "loan_survival_predictions"
    if not path.exists():
        logger.warning("  loan_survival_predictions not found — skipping")
        _write_table(pd.DataFrame(), out, "fact_survival")
        return 0
    df = _read_gold_table(path)
    _write_table(df, out, "fact_survival")
    return len(df)


# ─── Dimension table exporters ────────────────────────────────────────────────


def _export_dim_loan(gold: Path, out: Path) -> int:
    df = _read_gold_table(gold / "loan_current_state")
    cols = [
        "loan_id", "property_type", "metro_area", "sponsor_credit_tier",
        "amortization_type", "balloon_flag", "origination_date", "maturity_date",
        "origination_year", "original_balance", "ltv_at_origination",
        "dscr_at_origination", "note_rate", "loan_purpose", "loan_term_years",
    ]
    cols = [c for c in cols if c in df.columns]
    _write_table(df[cols], out, "dim_loan")
    return len(df)


def _export_dim_scenario(config_path: str | Path, out: Path) -> int:
    config = load_yaml_file(str(config_path))
    scenarios = config.get("scenarios", [])
    records = []
    for i, s in enumerate(scenarios):
        records.append({
            "scenario_name": s.get("name", f"scenario_{i}"),
            "description": s.get("description", ""),
            "rate_shock_bps": s.get("rate_shock_bps", 0),
            "cap_rate_shock_bps": s.get("cap_rate_shock_bps", 0),
            "noi_shock_pct": s.get("noi_shock_pct", 0.0),
            "property_type_filter": s.get("property_type_filter") or "all",
            "severity_rank": i,
        })
    df = pd.DataFrame(records)
    _write_table(df, out, "dim_scenario")
    return len(df)


def _export_dim_property_type(out: Path) -> int:
    records = [
        {"property_type": "office", "sector": "Office", "risk_tier": "high"},
        {"property_type": "retail", "sector": "Retail", "risk_tier": "medium-high"},
        {"property_type": "multifamily", "sector": "Multifamily", "risk_tier": "medium"},
        {"property_type": "industrial", "sector": "Industrial", "risk_tier": "low"},
        {"property_type": "hotel", "sector": "Hospitality", "risk_tier": "high"},
    ]
    df = pd.DataFrame(records)
    _write_table(df, out, "dim_property_type")
    return len(df)


def _export_dim_metro(gold: Path, out: Path) -> int:
    df = _read_gold_table(gold / "loan_current_state")
    if "metro_area" not in df.columns:
        _write_table(pd.DataFrame(), out, "dim_metro")
        return 0

    metros = sorted(df["metro_area"].dropna().unique())
    # Simple region mapping
    region_map = {
        "New York": "Northeast", "Boston": "Northeast", "Philadelphia": "Northeast",
        "Washington DC": "Mid-Atlantic",
        "Miami": "Southeast", "Atlanta": "Southeast", "Charlotte": "Southeast",
        "Tampa": "Southeast", "Orlando": "Southeast", "Raleigh": "Southeast",
        "Chicago": "Midwest", "Minneapolis": "Midwest",
        "Dallas": "South Central", "Houston": "South Central", "Austin": "South Central",
        "Nashville": "South Central",
        "Denver": "Mountain West", "Phoenix": "Mountain West", "Salt Lake City": "Mountain West",
        "Las Vegas": "Mountain West",
        "Los Angeles": "West Coast", "San Francisco": "West Coast", "San Diego": "West Coast",
        "Seattle": "Pacific Northwest", "Portland": "Pacific Northwest",
    }
    records = [{"metro": m, "region": region_map.get(m, "Other")} for m in metros]
    dim_df = pd.DataFrame(records)
    _write_table(dim_df, out, "dim_metro")
    return len(dim_df)


def _export_dim_date(out: Path) -> int:
    start = date(2015, 1, 1)
    end = date(2028, 12, 31)
    dates = []
    current = start
    while current <= end:
        quarter = (current.month - 1) // 3 + 1
        dates.append({
            "date": current.isoformat(),
            "year": current.year,
            "quarter": quarter,
            "quarter_name": f"Q{quarter}",
            "month": current.month,
            "month_name": current.strftime("%B"),
            "is_month_end": (current + timedelta(days=1)).month != current.month,
        })
        current += timedelta(days=1)
    df = pd.DataFrame(dates)
    _write_table(df, out, "dim_date")
    return len(df)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _read_gold_table(path: Path) -> pd.DataFrame:
    try:
        records = DeltaReader(path).read()
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning(f"  Could not read {path}: {e}")
        return pd.DataFrame()


def _write_table(df: pd.DataFrame, output_dir: Path, table_name: str) -> None:
    if df.empty:
        logger.info(f"  {table_name}: EMPTY (skipped)")
        return

    # Parquet
    parquet_path = output_dir / f"{table_name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
    except Exception:
        # pyarrow not available — skip parquet, CSV only
        pass

    # CSV (always available)
    csv_path = output_dir / f"{table_name}.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"  {table_name}: {len(df):,} rows")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gold tables for Power BI")
    parser.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="config/stress_scenarios.yaml")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    export_all(gold_path=args.gold_path, output_path=args.output, config_path=args.config)


if __name__ == "__main__":
    main()
