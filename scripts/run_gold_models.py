#!/usr/bin/env python3
"""
Gold Layer Materializer (Sandbox Fallback)
============================================

Implements the same transformations as the dbt Gold models using Python + sqlite3.
Used when dbt-duckdb is not available in the sandbox environment.

Reads Silver Delta tables, executes Gold model SQL, writes output as Delta tables.

Models:
  1. loan_current_state     — one row per loan, latest distress assessment
  2. loan_distress_history  — full time series of distress events
  3. market_distress_index  — aggregated by metro × property_type × vintage
  4. maturity_wall          — loans by maturity period (2024-2028)

Usage:
    python scripts/run_gold_models.py
    python scripts/run_gold_models.py --silver-dir data/silver --gold-dir data/gold
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.delta_writer import DeltaReader, DeltaWriter

logger = logging.getLogger(__name__)

DEFAULT_SILVER = Path("data/silver")
DEFAULT_GOLD = Path("data/gold")


def create_in_memory_db(silver_dir: Path) -> sqlite3.Connection:
    """Load Silver data into an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Load loan_features
    logger.info("Loading silver.loan_features...")
    features = DeltaReader(silver_dir / "loan_features").read()
    if features:
        _create_table_from_records(conn, "loan_features", features)
    logger.info(f"  Loaded {len(features)} records")

    return conn


def _create_table_from_records(
    conn: sqlite3.Connection, table_name: str, records: list[dict]
) -> None:
    """Create a SQLite table from a list of dicts and insert all records."""
    if not records:
        return

    columns = list(records[0].keys())
    col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
    conn.execute(f"CREATE TABLE {table_name} ({col_defs})")

    placeholders = ", ".join("?" * len(columns))
    insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

    rows = []
    for r in records:
        row = tuple(str(r.get(c, "")) if r.get(c) is not None else None for c in columns)
        rows.append(row)
    conn.executemany(insert_sql, rows)
    conn.commit()


def run_loan_current_state(conn: sqlite3.Connection) -> list[dict]:
    """Execute loan_current_state model."""
    sql = """
    WITH ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY _feature_computed_at DESC) AS _row_rank
        FROM loan_features
    ),
    current_state AS (
        SELECT
            loan_id, deal_name, property_id,
            origination_date, maturity_date,
            CAST(origination_year AS INTEGER) AS origination_year,
            CAST(original_balance AS REAL) AS original_balance,
            CAST(current_balance AS REAL) AS current_balance,
            CAST(note_rate AS REAL) AS note_rate,
            amortization_type, balloon_flag,
            CAST(loan_term_years AS INTEGER) AS loan_term_years,
            loan_purpose, property_type, metro_area, submarket,
            CAST(occupancy_pct AS REAL) AS occupancy_pct,
            CAST(noi_annual AS REAL) AS noi_annual,
            CAST(property_value_at_origination AS REAL) AS property_value_at_origination,
            CAST(ltv_at_origination AS REAL) AS ltv_at_origination,
            CAST(dscr_at_origination AS REAL) AS dscr_at_origination,
            sponsor_credit_tier,
            CAST(current_cap_rate AS REAL) AS current_cap_rate,
            CAST(current_value AS REAL) AS current_value,
            CAST(current_ltv AS REAL) AS current_ltv,
            CAST(refinance_rate AS REAL) AS refinance_rate,
            CAST(refinance_rate_pct AS REAL) AS refinance_rate_pct,
            CAST(rate_gap AS REAL) AS rate_gap,
            CAST(rate_gap_bps AS REAL) AS rate_gap_bps,
            CAST(new_dscr AS REAL) AS new_dscr,
            CAST(annual_debt_service_refi AS REAL) AS annual_debt_service_refi,
            CAST(debt_yield AS REAL) AS debt_yield,
            CAST(debt_yield_pct AS REAL) AS debt_yield_pct,
            CAST(months_to_maturity AS REAL) AS months_to_maturity,
            is_matured,
            _refi_stressed,
            _rate_gap_positive,
            CASE
                WHEN CAST(new_dscr AS REAL) < 1.0 AND CAST(current_ltv AS REAL) > 0.80 THEN 'critical'
                WHEN CAST(new_dscr AS REAL) < 1.0 OR CAST(current_ltv AS REAL) > 0.80 THEN 'high'
                WHEN CAST(new_dscr AS REAL) < 1.25 OR CAST(current_ltv AS REAL) > 0.70 THEN 'medium'
                ELSE 'low'
            END AS distress_tier,
            CAST(new_dscr AS REAL) - CAST(dscr_at_origination AS REAL) AS dscr_change,
            CAST(current_ltv AS REAL) - CAST(ltv_at_origination AS REAL) AS ltv_change,
            _feature_computed_at
        FROM ranked
        WHERE _row_rank = 1
    )
    SELECT * FROM current_state
    """
    return _execute_query(conn, sql)


def run_loan_distress_history(conn: sqlite3.Connection) -> list[dict]:
    """Execute loan_distress_history model."""
    sql = """
    SELECT
        loan_id, property_type, metro_area,
        CAST(origination_year AS INTEGER) AS origination_year,
        CAST(original_balance AS REAL) AS original_balance,
        CAST(current_balance AS REAL) AS current_balance,
        CAST(note_rate AS REAL) AS note_rate,
        amortization_type, maturity_date,
        CAST(current_ltv AS REAL) AS current_ltv,
        CAST(new_dscr AS REAL) AS new_dscr,
        CAST(rate_gap AS REAL) AS rate_gap,
        CAST(rate_gap_bps AS REAL) AS rate_gap_bps,
        CAST(debt_yield AS REAL) AS debt_yield,
        CAST(current_cap_rate AS REAL) AS current_cap_rate,
        CAST(refinance_rate AS REAL) AS refinance_rate,
        CAST(months_to_maturity AS REAL) AS months_to_maturity,
        CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN 1 ELSE 0 END AS is_distressed,
        CASE WHEN CAST(new_dscr AS REAL) < 1.0 THEN 1 ELSE 0 END AS dscr_below_1,
        CASE WHEN CAST(current_ltv AS REAL) > 0.80 THEN 1 ELSE 0 END AS ltv_above_80,
        CASE WHEN CAST(new_dscr AS REAL) < 1.0 AND CAST(current_ltv AS REAL) > 0.80 THEN 1 ELSE 0 END AS dual_trigger,
        CASE
            WHEN CAST(new_dscr AS REAL) < 1.0 AND CAST(current_ltv AS REAL) > 0.80 THEN 'critical'
            WHEN CAST(new_dscr AS REAL) < 1.0 OR CAST(current_ltv AS REAL) > 0.80 THEN 'high'
            WHEN CAST(new_dscr AS REAL) < 1.25 OR CAST(current_ltv AS REAL) > 0.70 THEN 'medium'
            ELSE 'low'
        END AS distress_tier,
        CASE
            WHEN CAST(new_dscr AS REAL) >= 1.5 THEN 0.0
            WHEN CAST(new_dscr AS REAL) >= 1.0 THEN (1.5 - CAST(new_dscr AS REAL)) / 0.5
            ELSE 1.0 + (1.0 - CAST(new_dscr AS REAL))
        END AS dscr_severity_score,
        CASE
            WHEN CAST(current_ltv AS REAL) <= 0.60 THEN 0.0
            WHEN CAST(current_ltv AS REAL) <= 0.80 THEN (CAST(current_ltv AS REAL) - 0.60) / 0.20
            ELSE 1.0 + (CAST(current_ltv AS REAL) - 0.80) / 0.20
        END AS ltv_severity_score,
        _feature_computed_at AS snapshot_at
    FROM loan_features
    """
    return _execute_query(conn, sql)


def run_market_distress_index(conn: sqlite3.Connection) -> list[dict]:
    """Execute market_distress_index model."""
    sql = """
    SELECT
        metro_area, property_type,
        CAST(origination_year AS INTEGER) AS origination_year,
        COUNT(*) AS loan_count,
        SUM(CAST(current_balance AS REAL)) AS total_upb,
        SUM(CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN 1 ELSE 0 END) AS distressed_loan_count,
        CAST(SUM(CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN 1 ELSE 0 END) AS REAL)
            / COUNT(*) AS pct_distressed_loans,
        SUM(CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN CAST(current_balance AS REAL) ELSE 0 END) AS total_upb_at_risk,
        CAST(SUM(CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN CAST(current_balance AS REAL) ELSE 0 END) AS REAL)
            / NULLIF(SUM(CAST(current_balance AS REAL)), 0) AS pct_upb_at_risk,
        AVG(CAST(rate_gap_bps AS REAL)) AS avg_refinance_gap_bps,
        MAX(CAST(rate_gap_bps AS REAL)) AS max_refinance_gap_bps,
        SUM(CAST(new_dscr AS REAL) * CAST(current_balance AS REAL)) / NULLIF(SUM(CAST(current_balance AS REAL)), 0) AS weighted_avg_dscr,
        AVG(CAST(new_dscr AS REAL)) AS simple_avg_dscr,
        MIN(CAST(new_dscr AS REAL)) AS min_dscr,
        SUM(CAST(current_ltv AS REAL) * CAST(current_balance AS REAL)) / NULLIF(SUM(CAST(current_balance AS REAL)), 0) AS weighted_avg_ltv,
        AVG(CAST(current_ltv AS REAL)) AS simple_avg_ltv,
        MAX(CAST(current_ltv AS REAL)) AS max_ltv,
        AVG(CAST(debt_yield AS REAL)) AS avg_debt_yield,
        SUM(CASE WHEN CAST(new_dscr AS REAL) < 1.0 AND CAST(current_ltv AS REAL) > 0.80 THEN 1 ELSE 0 END) AS critical_count,
        SUM(CASE WHEN (CAST(new_dscr AS REAL) < 1.0 OR CAST(current_ltv AS REAL) > 0.80)
                  AND NOT (CAST(new_dscr AS REAL) < 1.0 AND CAST(current_ltv AS REAL) > 0.80) THEN 1 ELSE 0 END) AS high_count,
        SUM(CASE WHEN (CAST(new_dscr AS REAL) < 1.25 OR CAST(current_ltv AS REAL) > 0.70)
                  AND NOT (CAST(new_dscr AS REAL) < 1.0 OR CAST(current_ltv AS REAL) > 0.80) THEN 1 ELSE 0 END) AS medium_count
    FROM loan_features
    GROUP BY metro_area, property_type, origination_year
    ORDER BY pct_distressed_loans DESC, total_upb_at_risk DESC
    """
    return _execute_query(conn, sql)


def run_maturity_wall(conn: sqlite3.Connection) -> list[dict]:
    """Execute maturity_wall model."""
    sql = """
    WITH with_maturity_period AS (
        SELECT *,
            CAST(SUBSTR(maturity_date, 1, 4) AS INTEGER) AS maturity_year,
            CASE
                WHEN CAST(SUBSTR(maturity_date, 6, 2) AS INTEGER) <= 3 THEN 1
                WHEN CAST(SUBSTR(maturity_date, 6, 2) AS INTEGER) <= 6 THEN 2
                WHEN CAST(SUBSTR(maturity_date, 6, 2) AS INTEGER) <= 9 THEN 3
                ELSE 4
            END AS maturity_quarter
        FROM loan_features
        WHERE CAST(SUBSTR(maturity_date, 1, 4) AS INTEGER) >= 2024
          AND CAST(SUBSTR(maturity_date, 1, 4) AS INTEGER) <= 2028
    )
    SELECT
        maturity_year, maturity_quarter,
        maturity_year || '-Q' || maturity_quarter AS maturity_period,
        property_type,
        COUNT(*) AS count_loans_maturing,
        SUM(CAST(current_balance AS REAL)) AS total_upb_maturing,
        AVG(CAST(current_balance AS REAL)) AS avg_loan_size,
        SUM(CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN 1 ELSE 0 END) AS count_distressed,
        CAST(SUM(CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN 1 ELSE 0 END) AS REAL)
            / COUNT(*) AS pct_distressed,
        SUM(CASE WHEN _refi_stressed IN ('True', '1', 'true') THEN CAST(current_balance AS REAL) ELSE 0 END) AS upb_distressed,
        AVG(CAST(new_dscr AS REAL)) AS avg_dscr,
        AVG(CAST(current_ltv AS REAL)) AS avg_ltv,
        AVG(CAST(rate_gap_bps AS REAL)) AS avg_rate_gap_bps,
        SUM(CASE WHEN CAST(new_dscr AS REAL) < 1.0 AND CAST(current_ltv AS REAL) > 0.80 THEN 1 ELSE 0 END) AS critical_count,
        SUM(CASE WHEN (CAST(new_dscr AS REAL) < 1.0 OR CAST(current_ltv AS REAL) > 0.80)
                  AND NOT (CAST(new_dscr AS REAL) < 1.0 AND CAST(current_ltv AS REAL) > 0.80) THEN 1 ELSE 0 END) AS high_count
    FROM with_maturity_period
    GROUP BY maturity_year, maturity_quarter, maturity_period, property_type
    ORDER BY maturity_year, maturity_quarter, property_type
    """
    return _execute_query(conn, sql)


def _execute_query(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Execute SQL and return list of dicts."""
    cursor = conn.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Gold layer models (sandbox)")
    parser.add_argument("--silver-dir", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start = time.time()
    logger.info("=" * 60)
    logger.info("GOLD LAYER MATERIALIZER — Starting")
    logger.info("=" * 60)

    # Load silver data into SQLite
    conn = create_in_memory_db(args.silver_dir)

    # Run each model
    models = [
        ("loan_current_state", run_loan_current_state),
        ("loan_distress_history", run_loan_distress_history),
        ("market_distress_index", run_market_distress_index),
        ("maturity_wall", run_maturity_wall),
    ]

    for model_name, runner in models:
        logger.info(f"\n  Running model: {model_name}...")
        records = runner(conn)
        output_path = args.gold_dir / model_name
        writer = DeltaWriter(output_path)
        writer.write(records, mode="overwrite")
        logger.info(f"    → {len(records)} rows written to {output_path}")

    conn.close()

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"GOLD LAYER MATERIALIZER — Complete ({elapsed:.1f}s)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
