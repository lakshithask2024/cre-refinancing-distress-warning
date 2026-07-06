"""
Unit tests for the Delta Lake writer/reader (src/utils/delta_writer.py).

Tests cover:
- Pure-Python write + read round-trip (JSON-lines-backed Delta)
- Partitioned write/read with partition value reconstruction
- Fallback read from plain JSON-lines (no _delta_log/)
- Fallback read from Parquet (when pyarrow available)
- Spark Delta write + read round-trip (when PySpark available, else skipped)
- Schema inference and count()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.delta_writer import DeltaReader, DeltaWriter, DeltaSchema


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_records():
    """Simple records for round-trip testing."""
    return [
        {"id": "A001", "value": 100.5, "category": "alpha", "count": 10},
        {"id": "A002", "value": 200.0, "category": "alpha", "count": 20},
        {"id": "B001", "value": 300.75, "category": "beta", "count": 30},
        {"id": "B002", "value": 400.0, "category": "beta", "count": 40},
        {"id": "C001", "value": 500.25, "category": "gamma", "count": 50},
    ]


@pytest.fixture
def loan_like_records():
    """Records that resemble the loan schema (for Spark schema test)."""
    return [
        {
            "loan_id": "LN-001",
            "origination_year": 2020,
            "original_balance": 15000000.0,
            "note_rate": 0.0425,
            "property_type": "office",
            "ltv_at_origination": 0.65,
            "dscr_at_origination": 1.45,
            "amortization_type": "interest_only",
            "balloon_flag": True,
        },
        {
            "loan_id": "LN-002",
            "origination_year": 2021,
            "original_balance": 25000000.0,
            "note_rate": 0.035,
            "property_type": "industrial",
            "ltv_at_origination": 0.55,
            "dscr_at_origination": 1.72,
            "amortization_type": "amortizing",
            "balloon_flag": False,
        },
    ]


# ─── Pure-Python Round-Trip Tests ─────────────────────────────────────────────


class TestPurePythonRoundTrip:
    """Test write + read using the pure-Python (JSON-lines) path."""

    def test_write_creates_delta_log(self, tmp_path, sample_records):
        """Writing should create a _delta_log/ directory."""
        table_path = tmp_path / "test_table"
        writer = DeltaWriter(table_path)
        writer.write(sample_records)
        assert (table_path / "_delta_log").exists()

    def test_write_read_unpartitioned(self, tmp_path, sample_records):
        """Unpartitioned write + read returns same records."""
        table_path = tmp_path / "test_table"
        writer = DeltaWriter(table_path)
        writer.write(sample_records)

        reader = DeltaReader(table_path)
        result = reader.read()
        assert len(result) == len(sample_records)

        # Verify values (order may differ)
        result_ids = sorted(r["id"] for r in result)
        expected_ids = sorted(r["id"] for r in sample_records)
        assert result_ids == expected_ids

    def test_write_read_partitioned(self, tmp_path, sample_records):
        """Partitioned write + read returns records with partition values."""
        table_path = tmp_path / "test_table"
        writer = DeltaWriter(table_path)
        writer.write(sample_records, partition_by="category")

        reader = DeltaReader(table_path)
        result = reader.read()
        assert len(result) == len(sample_records)

        # Partition values should be reconstructed
        categories = {r["category"] for r in result}
        assert categories == {"alpha", "beta", "gamma"}

    def test_count_matches(self, tmp_path, sample_records):
        """count() should match actual record count."""
        table_path = tmp_path / "test_table"
        writer = DeltaWriter(table_path)
        writer.write(sample_records)

        reader = DeltaReader(table_path)
        assert reader.count() == 5

    def test_schema_inference(self, tmp_path, sample_records):
        """Schema should be inferred and stored in the commit log."""
        table_path = tmp_path / "test_table"
        writer = DeltaWriter(table_path)
        writer.write(sample_records)

        reader = DeltaReader(table_path)
        schema = reader.get_schema()
        assert schema is not None
        field_names = [f["name"] for f in schema["fields"]]
        assert "id" in field_names
        assert "value" in field_names
        assert "category" in field_names

    def test_overwrite_mode(self, tmp_path, sample_records):
        """Overwrite mode replaces previous data entirely."""
        table_path = tmp_path / "test_table"
        writer = DeltaWriter(table_path)
        writer.write(sample_records)
        # Overwrite with fewer records
        writer.write(sample_records[:2], mode="overwrite")

        reader = DeltaReader(table_path)
        result = reader.read()
        assert len(result) == 2

    def test_values_preserved_numeric(self, tmp_path, sample_records):
        """Numeric values should be preserved through write/read cycle."""
        table_path = tmp_path / "test_table"
        writer = DeltaWriter(table_path)
        writer.write(sample_records)

        reader = DeltaReader(table_path)
        result = reader.read()
        result_by_id = {r["id"]: r for r in result}

        assert result_by_id["A001"]["value"] == 100.5
        assert result_by_id["B001"]["value"] == 300.75
        assert result_by_id["C001"]["count"] == 50


# ─── Fallback Reader Tests ────────────────────────────────────────────────────


class TestFallbackReader:
    """Test reader fallback paths (no _delta_log/)."""

    def test_read_jsonl_without_delta_log(self, tmp_path, sample_records):
        """Reader should handle plain JSON-lines dirs without _delta_log/."""
        table_path = tmp_path / "plain_jsonl"
        table_path.mkdir()
        # Write plain JSONL (no _delta_log/)
        with open(table_path / "data.json", "w") as f:
            for r in sample_records:
                f.write(json.dumps(r) + "\n")

        reader = DeltaReader(table_path)
        result = reader.read()
        assert len(result) == len(sample_records)

    def test_read_partitioned_jsonl_without_delta_log(self, tmp_path, sample_records):
        """Reader should reconstruct partition values from directory names."""
        table_path = tmp_path / "partitioned_jsonl"
        table_path.mkdir()
        # Write partitioned without _delta_log/
        for category in ["alpha", "beta", "gamma"]:
            part_dir = table_path / f"category={category}"
            part_dir.mkdir()
            category_records = [r for r in sample_records if r["category"] == category]
            with open(part_dir / "data.json", "w") as f:
                for r in category_records:
                    f.write(json.dumps(r) + "\n")

        reader = DeltaReader(table_path)
        result = reader.read()
        assert len(result) == len(sample_records)
        # Partition values should be extracted
        categories = {r["category"] for r in result}
        assert "alpha" in categories

    def test_read_nonexistent_raises(self, tmp_path):
        """Reader should raise FileNotFoundError for empty directories."""
        table_path = tmp_path / "empty"
        table_path.mkdir()

        reader = DeltaReader(table_path)
        with pytest.raises(FileNotFoundError):
            reader.read()


# ─── Spark Delta Round-Trip Test ──────────────────────────────────────────────


class TestSparkDeltaRoundTrip:
    """Test write via Spark + read via reader (requires PySpark + delta-spark)."""

    @pytest.fixture
    def has_spark(self):
        """Check if PySpark and delta-spark are available."""
        try:
            from pyspark.sql import SparkSession
            from delta import configure_spark_with_delta_pip
            return True
        except ImportError:
            return False

    def test_spark_write_then_read(self, tmp_path, loan_like_records, has_spark):
        """Write Delta via Spark, read via DeltaReader — rows and values match."""
        if not has_spark:
            pytest.skip("PySpark + delta-spark not installed")

        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            StructType, StructField, StringType, DoubleType, IntegerType, BooleanType,
        )
        from delta import configure_spark_with_delta_pip

        table_path = tmp_path / "spark_delta"

        # Define schema
        schema = StructType([
            StructField("loan_id", StringType(), False),
            StructField("origination_year", IntegerType(), True),
            StructField("original_balance", DoubleType(), True),
            StructField("note_rate", DoubleType(), True),
            StructField("property_type", StringType(), True),
            StructField("ltv_at_origination", DoubleType(), True),
            StructField("dscr_at_origination", DoubleType(), True),
            StructField("amortization_type", StringType(), True),
            StructField("balloon_flag", BooleanType(), True),
        ])

        # Write via Spark Delta
        builder = (
            SparkSession.builder.master("local[2]")
            .appName("test-delta-roundtrip")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()

        df = spark.createDataFrame(loan_like_records, schema=schema)
        df.write.format("delta").mode("overwrite").save(str(table_path))

        # Verify _delta_log/ was created with Parquet data files
        assert (table_path / "_delta_log").exists()
        parquet_files = list(table_path.rglob("*.parquet"))
        assert len(parquet_files) > 0, "Spark should produce .parquet data files"

        # Read via DeltaReader
        reader = DeltaReader(table_path)
        result = reader.read()

        # Verify row count
        assert len(result) == len(loan_like_records)

        # Verify column values
        result_by_id = {r["loan_id"]: r for r in result}
        assert "LN-001" in result_by_id
        assert "LN-002" in result_by_id
        assert result_by_id["LN-001"]["original_balance"] == 15000000.0
        assert result_by_id["LN-001"]["note_rate"] == 0.0425
        assert result_by_id["LN-001"]["property_type"] == "office"
        assert result_by_id["LN-002"]["origination_year"] == 2021
        assert result_by_id["LN-002"]["dscr_at_origination"] == 1.72

        spark.stop()

    def test_spark_partitioned_write_then_read(self, tmp_path, loan_like_records, has_spark):
        """Partitioned Spark Delta write, read back with partition values."""
        if not has_spark:
            pytest.skip("PySpark + delta-spark not installed")

        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            StructType, StructField, StringType, DoubleType, IntegerType, BooleanType,
        )
        from delta import configure_spark_with_delta_pip

        table_path = tmp_path / "spark_delta_partitioned"

        schema = StructType([
            StructField("loan_id", StringType(), False),
            StructField("origination_year", IntegerType(), True),
            StructField("original_balance", DoubleType(), True),
            StructField("note_rate", DoubleType(), True),
            StructField("property_type", StringType(), True),
            StructField("ltv_at_origination", DoubleType(), True),
            StructField("dscr_at_origination", DoubleType(), True),
            StructField("amortization_type", StringType(), True),
            StructField("balloon_flag", BooleanType(), True),
        ])

        builder = (
            SparkSession.builder.master("local[2]")
            .appName("test-delta-partitioned")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()

        df = spark.createDataFrame(loan_like_records, schema=schema)
        df.write.format("delta").mode("overwrite").partitionBy("property_type").save(str(table_path))

        # Read back
        reader = DeltaReader(table_path)
        result = reader.read()

        assert len(result) == 2
        types = {r["property_type"] for r in result}
        assert types == {"office", "industrial"}

        spark.stop()
