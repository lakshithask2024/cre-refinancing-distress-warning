"""
Shared test fixtures for the CRE Distress Warning System test suite.

Provides:
- SparkSession fixture (shared across integration tests)
- Temporary Delta Lake path fixture
- Sample data fixtures for unit tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def spark():
    """Create a shared SparkSession for integration tests."""
    # Lazy import — only needed for integration tests
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("cre-distress-warning-tests")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    yield session
    session.stop()


@pytest.fixture
def tmp_delta_path(tmp_path: Path) -> Path:
    """Provide a temporary directory for Delta Lake tables during tests."""
    delta_dir = tmp_path / "delta"
    delta_dir.mkdir()
    return delta_dir


@pytest.fixture
def tmp_data_dir() -> Path:
    """Provide a temporary data directory that mimics the data/ structure."""
    with tempfile.TemporaryDirectory(prefix="cre_test_") as tmpdir:
        base = Path(tmpdir)
        (base / "bronze").mkdir()
        (base / "silver").mkdir()
        (base / "gold").mkdir()
        (base / "exports" / "powerbi").mkdir(parents=True)
        yield base


@pytest.fixture
def sample_loan_record() -> dict:
    """Provide a single sample loan record for unit tests."""
    return {
        "loan_id": "test-loan-001",
        "origination_date": "2020-01-15",
        "maturity_date": "2025-01-15",
        "loan_amount": 15_000_000.0,
        "interest_rate": 0.0425,
        "property_type": "Office",
        "metro": "New York",
        "ltv_at_origination": 0.68,
        "dscr_at_origination": 1.35,
        "occupancy_rate": 0.85,
        "noi": 1_200_000.0,
    }
