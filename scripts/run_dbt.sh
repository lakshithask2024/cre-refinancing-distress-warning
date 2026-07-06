#!/usr/bin/env bash
# ─── dbt Gold Layer Execution Script ──────────────────────────────────────────
#
# Attempts to run dbt (dbt-duckdb) for Gold layer materialization.
# Falls back to the Python-based Gold materializer if dbt is not installed.
#
# Usage:
#   ./scripts/run_dbt.sh
#   ./scripts/run_dbt.sh --target sandbox
#   ./scripts/run_dbt.sh --target production
#
# Prerequisites:
#   - dbt-duckdb installed (sandbox) OR dbt-spark (production)
#   - Silver tables materialized in data/silver/
#   - If dbt not available: Python 3.11+ with project dependencies
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DBT_DIR="$PROJECT_ROOT/dbt"
TARGET="${1:-sandbox}"

echo "═══════════════════════════════════════════════════════════════"
echo "  CRE Distress Warning — Gold Layer Build"
echo "  Target: $TARGET"
echo "═══════════════════════════════════════════════════════════════"
echo

# Check if dbt is available
if command -v dbt &> /dev/null; then
    echo "[dbt] dbt found: $(dbt --version | head -1)"
    echo "[dbt] Running dbt pipeline..."
    echo

    cd "$DBT_DIR"

    # Install packages
    echo "[1/3] Installing dbt packages..."
    dbt deps --target "$TARGET" || echo "  (deps failed — continuing without packages)"

    # Run models
    echo "[2/3] Running dbt models..."
    dbt run --target "$TARGET"

    # Run tests
    echo "[3/3] Running dbt tests..."
    dbt test --target "$TARGET"

    echo
    echo "[dbt] Gold layer build complete!"
else
    echo "[fallback] dbt not installed — using Python Gold materializer"
    echo "[fallback] (dbt project files at dbt/ are ready for deployment with dbt-duckdb or dbt-spark)"
    echo

    cd "$PROJECT_ROOT"
    python scripts/run_gold_models.py --silver-dir data/silver --gold-dir data/gold

    echo
    echo "[fallback] Gold layer materialized via Python."
    echo "           Production deployment should use: dbt run --target production"
fi

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Gold tables written to: data/gold/"
echo "═══════════════════════════════════════════════════════════════"
