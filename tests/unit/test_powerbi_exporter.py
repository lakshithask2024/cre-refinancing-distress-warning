"""
Unit tests for the Power BI data export pipeline.

Verifies all expected tables are produced with correct structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pd = pytest.importorskip("pandas", reason="pandas required")
np = pytest.importorskip("numpy", reason="numpy required")


@pytest.fixture
def gold_fixture(tmp_path):
    """Create minimal Gold tables for export testing."""
    from src.utils.delta_writer import DeltaWriter

    # loan_current_state
    loans = [
        {
            "loan_id": f"LN-{i:04d}", "property_type": "office",
            "metro_area": "New York", "sponsor_credit_tier": "B",
            "amortization_type": "interest_only", "balloon_flag": "True",
            "origination_date": "2017-06-15", "maturity_date": "2024-06-15",
            "origination_year": "2017", "original_balance": 20000000.0,
            "current_balance": 20000000.0, "ltv_at_origination": 0.65,
            "dscr_at_origination": 1.4, "note_rate": 0.04,
            "loan_purpose": "acquisition", "loan_term_years": 7,
            "distress_tier": "high", "current_ltv": 0.85,
            "new_dscr": 0.95, "refinance_rate": 0.065,
            "rate_gap_bps": 250.0, "debt_yield": 0.075,
            "current_cap_rate": 7.5, "current_value": 20000000.0,
            "months_to_maturity": -12.0, "is_matured": "True",
            "dscr_change": -0.45, "ltv_change": 0.20,
            "noi_annual": 1500000.0,
        }
        for i in range(50)
    ]
    DeltaWriter(tmp_path / "loan_current_state").write(loans)

    # loan_distress_history
    history = [
        {"loan_id": f"LN-{i:04d}", "is_distressed": 1, "distress_tier": "high",
         "current_ltv": 0.85, "new_dscr": 0.95, "rate_gap_bps": 250.0,
         "dscr_severity_score": 0.8, "ltv_severity_score": 0.5,
         "snapshot_at": "2026-01-01T00:00:00"}
        for i in range(50)
    ]
    DeltaWriter(tmp_path / "loan_distress_history").write(history)

    return tmp_path


class TestPowerBIExporter:
    """Test export pipeline produces all expected files."""

    def test_export_produces_csv_files(self, gold_fixture, tmp_path):
        """All expected CSV files should be created."""
        from src.exports.powerbi_exporter import export_all

        output = tmp_path / "exports"
        counts = export_all(
            gold_path=gold_fixture,
            output_path=output,
        )

        # Check key tables exist as CSV
        expected_tables = [
            "fact_loan_current", "fact_loan_history",
            "dim_loan", "dim_scenario", "dim_property_type",
            "dim_metro", "dim_date",
        ]
        for table in expected_tables:
            csv_file = output / f"{table}.csv"
            assert csv_file.exists(), f"Missing: {csv_file}"

    def test_dim_date_covers_2015_to_2028(self, gold_fixture, tmp_path):
        """dim_date should cover the full 2015-2028 range."""
        from src.exports.powerbi_exporter import export_all

        output = tmp_path / "exports"
        export_all(gold_path=gold_fixture, output_path=output)

        dim_date = pd.read_csv(output / "dim_date.csv")
        years = dim_date["year"].unique()
        assert 2015 in years
        assert 2028 in years
        assert len(dim_date) == (2028 - 2015 + 1) * 365 + 4  # approx (leap years)
        # More precise: just check > 5000 days
        assert len(dim_date) > 5000

    def test_row_counts_match_gold(self, gold_fixture, tmp_path):
        """Fact table row counts should match Gold source."""
        from src.exports.powerbi_exporter import export_all

        output = tmp_path / "exports"
        counts = export_all(gold_path=gold_fixture, output_path=output)

        assert counts["fact_loan_current"] == 50
        assert counts["fact_loan_history"] == 50
        assert counts["dim_loan"] == 50
        assert counts["dim_scenario"] == 8  # 8 scenarios in config
        assert counts["dim_property_type"] == 5

    def test_dim_scenario_has_all_scenarios(self, gold_fixture, tmp_path):
        """dim_scenario should have all 8 scenarios from config."""
        from src.exports.powerbi_exporter import export_all

        output = tmp_path / "exports"
        export_all(gold_path=gold_fixture, output_path=output)

        dim = pd.read_csv(output / "dim_scenario.csv")
        assert "baseline" in dim["scenario_name"].values
        assert "combined_severe" in dim["scenario_name"].values
        assert "office_specific" in dim["scenario_name"].values
        assert len(dim) == 8
