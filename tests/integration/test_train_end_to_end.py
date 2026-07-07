"""
Integration test: end-to-end model training.

Marked @pytest.mark.slow — not run by default (use: pytest -m slow).
Trains a real XGBoost model on 500-row fixture data with 3 Optuna trials.

Asserts:
  - Training completes without error
  - MLflow run exists and contains expected metrics
  - Metrics JSON file is written to disk
  - Test AUC > 0.5 (better than random — sanity check)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Skip if ML dependencies not installed
pytest.importorskip("pandas", reason="pandas required")
pytest.importorskip("numpy", reason="numpy required")
pytest.importorskip("xgboost", reason="xgboost required for e2e training test")
pytest.importorskip("mlflow", reason="mlflow required for e2e training test")
pytest.importorskip("optuna", reason="optuna required for e2e training test")


@pytest.fixture
def e2e_fixture_table(tmp_path):
    """Create a 500-row fixture Delta table for end-to-end training."""
    from src.utils.delta_writer import DeltaWriter

    random.seed(99)
    records = []
    metros = ["New York", "LA", "Chicago", "Dallas", "Miami"]
    prop_types = ["office", "retail", "industrial", "multifamily", "hotel"]

    for i in range(500):
        orig_year = random.choice([2017, 2018, 2019, 2020, 2021, 2022])
        # Make distress correlated with features for AUC > 0.5
        ltv = random.uniform(0.4, 1.5)
        dscr = random.uniform(0.5, 2.0)
        is_distressed = 1 if (ltv > 0.8 and dscr < 1.0) else (1 if random.random() < 0.2 else 0)

        records.append({
            "loan_id": f"LN-E2E-{i:04d}",
            "property_type": random.choice(prop_types),
            "metro_area": random.choice(metros),
            "origination_year": str(orig_year),
            "original_balance": random.uniform(5e6, 50e6),
            "current_balance": random.uniform(5e6, 50e6),
            "note_rate": random.uniform(0.025, 0.07),
            "amortization_type": random.choice(["interest_only", "amortizing"]),
            "balloon_flag": random.choice(["True", "False"]),
            "maturity_date": f"{orig_year + 7}-06-15",
            "ltv_at_origination": random.uniform(0.4, 0.85),
            "dscr_at_origination": random.uniform(0.8, 2.5),
            "occupancy_pct": random.uniform(0.5, 1.0),
            "current_ltv": ltv,
            "new_dscr": dscr,
            "rate_gap": random.uniform(0.0, 0.03),
            "rate_gap_bps": random.uniform(0, 300),
            "debt_yield": random.uniform(0.05, 0.15),
            "current_cap_rate": random.uniform(5.0, 9.0),
            "refinance_rate": random.uniform(0.05, 0.08),
            "months_to_maturity": random.uniform(-20, 80),
            "sponsor_credit_tier": random.choice(["A", "B", "C"]),
            "is_distressed": is_distressed,
            "noi_annual": random.uniform(500000, 5000000),
            "_feature_computed_at": "2026-01-01T00:00:00",
        })

    table_path = tmp_path / "loan_distress_history"
    writer = DeltaWriter(table_path)
    writer.write(records, partition_by="origination_year")
    return table_path


@pytest.mark.slow
def test_train_end_to_end(e2e_fixture_table, tmp_path, monkeypatch):
    """Full training run: 500 rows, 3 trials, verify outputs."""
    # Point MLflow and metrics output to temp directory
    mlruns_dir = tmp_path / "mlruns"
    metrics_dir = tmp_path / "metrics"

    monkeypatch.setattr(
        "src.models.distress_classifier.MLFLOW_TRACKING_URI",
        f"file:{mlruns_dir}",
    )
    monkeypatch.setattr(
        "src.models.distress_classifier.METRICS_OUTPUT_DIR",
        metrics_dir,
    )

    from src.models.distress_classifier import train_and_log

    import mlflow
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    run_id = train_and_log(
        experiment_name="test_e2e",
        n_trials=3,
        seed=42,
        gold_path=str(e2e_fixture_table),
    )

    # Assert run completed
    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 0

    # Assert MLflow run exists with metrics
    run = mlflow.get_run(run_id)
    assert "test_auc" in run.data.metrics
    assert run.data.metrics["test_auc"] > 0.5, "AUC should be > 0.5 (better than random)"

    # Assert metrics JSON was written
    metrics_file = metrics_dir / "distress_classifier_metrics.json"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        metrics = json.load(f)
    assert "test_metrics" in metrics
    assert metrics["test_metrics"]["test_auc"] > 0.5
