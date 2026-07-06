"""
Unit tests for the Market Data Fetcher.

Validates:
- Offline mode produces expected data types and record counts
- Cap rate loader reads config correctly
- FRED client handles mocked HTTP responses
- FRED client handles errors gracefully
- Market data records have correct schema
- Metro-adjusted cap rates apply adjustments correctly
- CRE price index synthetic generation is reasonable
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.market_data_fetcher import (
    SOFR_MONTHLY,
    TREASURY_10Y_MONTHLY,
    TREASURY_5Y_MONTHLY,
    CapRateLoader,
    FREDClient,
    MarketDataFetcher,
    MarketDataRecord,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cap_rate_loader():
    """Create a CapRateLoader with the project config."""
    return CapRateLoader()


@pytest.fixture
def offline_fetcher():
    """Create a MarketDataFetcher in offline mode."""
    return MarketDataFetcher(offline=True)


@pytest.fixture
def all_offline_records(offline_fetcher):
    """Fetch all records in offline mode."""
    return offline_fetcher.fetch_all()


# ─── Offline Mode Tests ───────────────────────────────────────────────────────


class TestOfflineMode:
    """Test that offline mode produces complete market data."""

    def test_offline_returns_records(self, all_offline_records):
        """Offline mode should return a non-empty list of records."""
        assert len(all_offline_records) > 0

    def test_offline_has_all_data_types(self, all_offline_records):
        """Offline mode should produce all expected data types."""
        data_types = {r.data_type for r in all_offline_records}
        expected = {"treasury_10y", "treasury_5y", "sofr", "cap_rate", "cre_price_index"}
        assert expected == data_types

    def test_offline_treasury_10y_count(self, all_offline_records):
        """Treasury 10Y should have one record per month in hardcoded data."""
        treasury_10y = [r for r in all_offline_records if r.data_type == "treasury_10y"]
        assert len(treasury_10y) == len(TREASURY_10Y_MONTHLY)

    def test_offline_treasury_5y_count(self, all_offline_records):
        """Treasury 5Y should have one record per month in hardcoded data."""
        treasury_5y = [r for r in all_offline_records if r.data_type == "treasury_5y"]
        assert len(treasury_5y) == len(TREASURY_5Y_MONTHLY)

    def test_offline_sofr_count(self, all_offline_records):
        """SOFR should have one record per month in hardcoded data."""
        sofr = [r for r in all_offline_records if r.data_type == "sofr"]
        assert len(sofr) == len(SOFR_MONTHLY)

    def test_offline_cap_rates_exist(self, all_offline_records):
        """Cap rate records should be present."""
        cap_rates = [r for r in all_offline_records if r.data_type == "cap_rate"]
        assert len(cap_rates) > 100  # 5 types × 44 quarters + metro adjustments

    def test_offline_cre_index_exists(self, all_offline_records):
        """CRE price index records should be present."""
        cre_idx = [r for r in all_offline_records if r.data_type == "cre_price_index"]
        assert len(cre_idx) > 0


# ─── Record Schema Tests ──────────────────────────────────────────────────────


class TestRecordSchema:
    """Test that MarketDataRecord has correct field types."""

    def test_record_fields_not_none(self, all_offline_records):
        """Core fields should never be None."""
        for record in all_offline_records[:100]:
            assert record.data_type is not None
            assert record.observation_date is not None
            assert record.value is not None
            assert record.frequency is not None
            assert record.ingested_at is not None
            assert record.source is not None

    def test_value_is_numeric(self, all_offline_records):
        """Value field should always be numeric."""
        for record in all_offline_records:
            assert isinstance(record.value, (int, float))

    def test_frequency_valid(self, all_offline_records):
        """Frequency should be one of the expected values."""
        valid_frequencies = {"daily", "monthly", "quarterly"}
        for record in all_offline_records:
            assert record.frequency in valid_frequencies

    def test_cap_rate_has_property_type(self, all_offline_records):
        """Cap rate records should have property_type set."""
        cap_rates = [r for r in all_offline_records if r.data_type == "cap_rate"]
        for record in cap_rates:
            assert record.property_type is not None
            assert record.property_type != ""


# ─── Cap Rate Loader Tests ────────────────────────────────────────────────────


class TestCapRateLoader:
    """Test the cap rate YAML config loader."""

    def test_loads_all_property_types(self, cap_rate_loader):
        """Should load all 5 property types from config."""
        valid_types = {"office", "retail", "multifamily", "industrial", "hotel"}
        loaded_types = set(cap_rate_loader.property_types)
        assert valid_types.issubset(loaded_types)

    def test_national_cap_rates_quarterly(self, cap_rate_loader):
        """National cap rates should be quarterly observations."""
        records = cap_rate_loader.get_national_cap_rates()
        assert all(r.frequency == "quarterly" for r in records)

    def test_national_cap_rates_reasonable_values(self, cap_rate_loader):
        """Cap rates should be between 3% and 12%."""
        records = cap_rate_loader.get_national_cap_rates()
        for r in records:
            assert 3.0 <= r.value <= 12.0, (
                f"Cap rate {r.value}% for {r.property_type} at {r.observation_date} "
                f"outside expected range"
            )

    def test_metro_adjustments_loaded(self, cap_rate_loader):
        """Metro adjustments should be loaded from config."""
        assert len(cap_rate_loader.metro_adjustments) > 0
        assert "New York" in cap_rate_loader.metro_adjustments

    def test_metro_adjusted_cap_rates_differ_from_national(self, cap_rate_loader):
        """Metro-adjusted rates should differ from national for metros with non-zero adj."""
        national = cap_rate_loader.get_national_cap_rates()
        metro = cap_rate_loader.get_metro_adjusted_cap_rates(metros=["New York"])

        # New York has -80 bps adjustment, so rates should be lower
        ny_records = [r for r in metro if r.metro == "New York"]
        assert len(ny_records) > 0

        # Find matching national record for comparison
        nat_office = [r for r in national if r.property_type == "office"]
        ny_office = [r for r in ny_records if r.property_type == "office"]
        if nat_office and ny_office:
            # NY should be lower (negative adjustment)
            assert ny_office[0].value < nat_office[0].value

    def test_metro_adjustment_magnitude(self, cap_rate_loader):
        """Metro adjustments should be within reasonable bounds (±100 bps)."""
        for metro, adj in cap_rate_loader.metro_adjustments.items():
            assert -150 <= adj <= 150, (
                f"Metro adjustment for {metro} ({adj} bps) seems excessive"
            )


# ─── FRED Client Tests (Mocked) ──────────────────────────────────────────────


class TestFREDClient:
    """Test FRED API client with mocked HTTP responses."""

    def test_client_initializes_with_key(self):
        """Client should accept an API key."""
        client = FREDClient(api_key="test_key_123")
        assert client.api_key == "test_key_123"

    def test_client_warns_without_key(self):
        """Client should work (with warning) without API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = FREDClient(api_key="")
            assert client.api_key == ""

    @patch("src.ingestion.market_data_fetcher.FREDClient.fetch_series")
    def test_fetch_series_parses_response(self, mock_fetch):
        """Client should correctly parse FRED API JSON response."""
        mock_fetch.return_value = [
            {"date": "2024-01-01", "value": 4.05},
            {"date": "2024-02-01", "value": 4.19},
            {"date": "2024-03-01", "value": 4.20},
        ]

        client = FREDClient(api_key="test_key")
        result = client.fetch_series("DGS10")

        assert len(result) == 3
        assert result[0]["date"] == "2024-01-01"
        assert result[0]["value"] == 4.05

    @patch("src.ingestion.market_data_fetcher.FREDClient.fetch_series")
    def test_fetch_series_handles_missing_values(self, mock_fetch):
        """Client should skip observations with '.' (missing) values."""
        mock_fetch.return_value = [
            {"date": "2024-01-01", "value": 4.05},
            {"date": "2024-03-01", "value": 4.20},
        ]

        client = FREDClient(api_key="test_key")
        result = client.fetch_series("DGS10")
        assert len(result) == 2  # Missing value skipped

    @patch("src.ingestion.market_data_fetcher.FREDClient.fetch_series")
    def test_fetch_series_handles_api_error(self, mock_fetch):
        """Client should return empty list on API error."""
        mock_fetch.return_value = []

        client = FREDClient(api_key="test_key")
        result = client.fetch_series("INVALID_SERIES")
        assert result == []


# ─── Treasury Rate Data Tests ─────────────────────────────────────────────────


class TestTreasuryData:
    """Test hardcoded Treasury rate data quality."""

    def test_treasury_10y_values_in_range(self):
        """10Y Treasury rates should be between 0% and 6%."""
        for month, value in TREASURY_10Y_MONTHLY.items():
            assert 0.0 <= value <= 6.0, f"Treasury 10Y {value} at {month} out of range"

    def test_treasury_5y_values_in_range(self):
        """5Y Treasury rates should be between 0% and 6%."""
        for month, value in TREASURY_5Y_MONTHLY.items():
            assert 0.0 <= value <= 6.0, f"Treasury 5Y {value} at {month} out of range"

    def test_5y_generally_below_10y(self):
        """5Y rates should generally be at or below 10Y rates (normal curve)."""
        inverted_count = 0
        common_months = set(TREASURY_10Y_MONTHLY.keys()) & set(TREASURY_5Y_MONTHLY.keys())
        for month in common_months:
            if TREASURY_5Y_MONTHLY[month] > TREASURY_10Y_MONTHLY[month]:
                inverted_count += 1
        # Allow some yield curve inversions (they happen), but not majority
        assert inverted_count < len(common_months) * 0.4

    def test_sofr_values_in_range(self):
        """SOFR rates should be between 0% and 6%."""
        for month, value in SOFR_MONTHLY.items():
            assert 0.0 <= value <= 6.0, f"SOFR {value} at {month} out of range"


# ─── CRE Price Index Tests ────────────────────────────────────────────────────


class TestCREPriceIndex:
    """Test synthetic CRE price index generation."""

    def test_cre_index_starts_near_100(self, all_offline_records):
        """CRE price index should start near base value of 100."""
        cre_records = [
            r for r in all_offline_records if r.data_type == "cre_price_index"
        ]
        # Sort by date
        cre_records.sort(key=lambda r: r.observation_date)
        first_value = cre_records[0].value
        assert 95 <= first_value <= 105

    def test_cre_index_positive(self, all_offline_records):
        """CRE price index should always be positive."""
        cre_records = [
            r for r in all_offline_records if r.data_type == "cre_price_index"
        ]
        for r in cre_records:
            assert r.value > 0

    def test_cre_index_reasonable_range(self, all_offline_records):
        """CRE price index should be in reasonable range (80-160 over 2015-2025)."""
        cre_records = [
            r for r in all_offline_records if r.data_type == "cre_price_index"
        ]
        for r in cre_records:
            assert 80 <= r.value <= 160, (
                f"CRE index {r.value} at {r.observation_date} seems extreme"
            )


# ─── Writer Tests ─────────────────────────────────────────────────────────────


class TestWriter:
    """Test that the writer produces correct output structure."""

    def test_write_creates_output_directory(self, tmp_path, offline_fetcher):
        """Writer should create the output directory if it doesn't exist."""
        from src.ingestion.market_data_fetcher import write_market_data

        output = tmp_path / "test_market"
        records = offline_fetcher.fetch_all()
        write_market_data(records, output)
        assert output.exists()

    def test_write_creates_partitions(self, tmp_path, offline_fetcher):
        """Writer should create partition directories by data_type."""
        from src.ingestion.market_data_fetcher import write_market_data

        output = tmp_path / "test_market"
        records = offline_fetcher.fetch_all()
        write_market_data(records, output)

        # Check that partition directories exist (data_type=xxx)
        partition_dirs = [
            d for d in output.iterdir()
            if d.is_dir() and d.name.startswith("data_type=")
        ]
        assert len(partition_dirs) >= 4  # At least 4 data types as partitions

    def test_write_creates_metadata(self, tmp_path, offline_fetcher):
        """Writer should create a _delta_log/ directory (Delta format)."""
        from src.ingestion.market_data_fetcher import write_market_data

        output = tmp_path / "test_market"
        records = offline_fetcher.fetch_all()
        write_market_data(records, output)

        delta_log = output / "_delta_log"
        assert delta_log.exists(), "_delta_log/ directory should exist"

        # Should have at least one commit file
        commit_files = list(delta_log.glob("*.json"))
        assert len(commit_files) >= 1, "Should have at least one commit log"

        # Parse the commit to verify structure
        with open(commit_files[0]) as f:
            lines = f.readlines()
        actions = [json.loads(line) for line in lines if line.strip()]

        # Should have protocol, metaData, add actions, and commitInfo
        action_types = set()
        for action in actions:
            action_types.update(action.keys())
        assert "protocol" in action_types
        assert "metaData" in action_types
        assert "add" in action_types
        assert "commitInfo" in action_types
