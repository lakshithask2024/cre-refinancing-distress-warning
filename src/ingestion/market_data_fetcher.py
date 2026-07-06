"""
Market Data Fetcher — Bronze Layer
====================================

Fetches and assembles market data for the CRE Distress Warning System:
  - Treasury yields (10Y = DGS10, 5Y = DGS5) from FRED API
  - SOFR rate from FRED API
  - Cap rates by property type from config/cap_rates_historical.yaml
  - CRE Price Index proxy from FRED (BOGZ1FL075035503Q)

Supports offline mode using hardcoded/config data when FRED API is unavailable.

Output: Delta Lake table (Parquet/JSONL fallback) at data/bronze/market/

CLI Usage:
    python -m src.ingestion.market_data_fetcher --output data/bronze/market/
    python -m src.ingestion.market_data_fetcher --offline --output data/bronze/market/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.yaml_compat import load_yaml_file

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "bronze" / "market"


# ─── FRED Series IDs ──────────────────────────────────────────────────────────

FRED_SERIES = {
    "treasury_10y": "DGS10",
    "treasury_5y": "DGS5",
    "sofr": "SOFR",
    "cre_price_index": "BOGZ1FL075035503Q",
}

# Hardcoded Treasury 10Y monthly data (2015-01 to 2025-06) for offline mode
# Approximate month-end values from FRED historical data
TREASURY_10Y_MONTHLY: dict[str, float] = {
    "2015-01": 1.88, "2015-02": 2.00, "2015-03": 1.93, "2015-04": 1.94,
    "2015-05": 2.12, "2015-06": 2.36, "2015-07": 2.32, "2015-08": 2.17,
    "2015-09": 2.17, "2015-10": 2.07, "2015-11": 2.26, "2015-12": 2.27,
    "2016-01": 2.09, "2016-02": 1.78, "2016-03": 1.77, "2016-04": 1.81,
    "2016-05": 1.81, "2016-06": 1.64, "2016-07": 1.50, "2016-08": 1.56,
    "2016-09": 1.60, "2016-10": 1.76, "2016-11": 2.14, "2016-12": 2.45,
    "2017-01": 2.43, "2017-02": 2.42, "2017-03": 2.48, "2017-04": 2.30,
    "2017-05": 2.30, "2017-06": 2.19, "2017-07": 2.29, "2017-08": 2.21,
    "2017-09": 2.20, "2017-10": 2.36, "2017-11": 2.35, "2017-12": 2.40,
    "2018-01": 2.58, "2018-02": 2.86, "2018-03": 2.84, "2018-04": 2.87,
    "2018-05": 2.98, "2018-06": 2.91, "2018-07": 2.89, "2018-08": 2.86,
    "2018-09": 3.05, "2018-10": 3.15, "2018-11": 3.01, "2018-12": 2.83,
    "2019-01": 2.72, "2019-02": 2.68, "2019-03": 2.57, "2019-04": 2.53,
    "2019-05": 2.40, "2019-06": 2.07, "2019-07": 2.06, "2019-08": 1.63,
    "2019-09": 1.68, "2019-10": 1.69, "2019-11": 1.77, "2019-12": 1.86,
    "2020-01": 1.76, "2020-02": 1.50, "2020-03": 0.70, "2020-04": 0.64,
    "2020-05": 0.65, "2020-06": 0.66, "2020-07": 0.55, "2020-08": 0.72,
    "2020-09": 0.68, "2020-10": 0.88, "2020-11": 0.84, "2020-12": 0.93,
    "2021-01": 1.07, "2021-02": 1.34, "2021-03": 1.62, "2021-04": 1.63,
    "2021-05": 1.58, "2021-06": 1.47, "2021-07": 1.24, "2021-08": 1.31,
    "2021-09": 1.37, "2021-10": 1.55, "2021-11": 1.55, "2021-12": 1.51,
    "2022-01": 1.78, "2022-02": 1.93, "2022-03": 2.14, "2022-04": 2.70,
    "2022-05": 2.85, "2022-06": 2.98, "2022-07": 2.90, "2022-08": 2.90,
    "2022-09": 3.52, "2022-10": 4.07, "2022-11": 3.68, "2022-12": 3.62,
    "2023-01": 3.53, "2023-02": 3.75, "2023-03": 3.66, "2023-04": 3.44,
    "2023-05": 3.57, "2023-06": 3.73, "2023-07": 3.86, "2023-08": 4.18,
    "2023-09": 4.44, "2023-10": 4.88, "2023-11": 4.47, "2023-12": 3.97,
    "2024-01": 4.05, "2024-02": 4.19, "2024-03": 4.20, "2024-04": 4.50,
    "2024-05": 4.51, "2024-06": 4.36, "2024-07": 4.25, "2024-08": 3.90,
    "2024-09": 3.78, "2024-10": 4.10, "2024-11": 4.25, "2024-12": 4.30,
    "2025-01": 4.35, "2025-02": 4.28, "2025-03": 4.20, "2025-04": 4.15,
    "2025-05": 4.10, "2025-06": 4.05,
}


TREASURY_5Y_MONTHLY: dict[str, float] = {
    "2015-01": 1.37, "2015-02": 1.50, "2015-03": 1.42, "2015-04": 1.39,
    "2015-05": 1.53, "2015-06": 1.72, "2015-07": 1.65, "2015-08": 1.55,
    "2015-09": 1.47, "2015-10": 1.40, "2015-11": 1.64, "2015-12": 1.72,
    "2016-01": 1.56, "2016-02": 1.25, "2016-03": 1.29, "2016-04": 1.30,
    "2016-05": 1.31, "2016-06": 1.14, "2016-07": 1.00, "2016-08": 1.09,
    "2016-09": 1.15, "2016-10": 1.29, "2016-11": 1.72, "2016-12": 1.93,
    "2017-01": 1.93, "2017-02": 1.92, "2017-03": 2.01, "2017-04": 1.80,
    "2017-05": 1.79, "2017-06": 1.76, "2017-07": 1.84, "2017-08": 1.73,
    "2017-09": 1.79, "2017-10": 1.96, "2017-11": 2.06, "2017-12": 2.21,
    "2018-01": 2.38, "2018-02": 2.64, "2018-03": 2.63, "2018-04": 2.67,
    "2018-05": 2.78, "2018-06": 2.73, "2018-07": 2.74, "2018-08": 2.74,
    "2018-09": 2.95, "2018-10": 3.07, "2018-11": 2.88, "2018-12": 2.69,
    "2019-01": 2.57, "2019-02": 2.50, "2019-03": 2.41, "2019-04": 2.31,
    "2019-05": 2.19, "2019-06": 1.85, "2019-07": 1.83, "2019-08": 1.42,
    "2019-09": 1.55, "2019-10": 1.52, "2019-11": 1.62, "2019-12": 1.69,
    "2020-01": 1.53, "2020-02": 1.27, "2020-03": 0.41, "2020-04": 0.36,
    "2020-05": 0.34, "2020-06": 0.30, "2020-07": 0.26, "2020-08": 0.31,
    "2020-09": 0.27, "2020-10": 0.39, "2020-11": 0.40, "2020-12": 0.44,
    "2021-01": 0.44, "2021-02": 0.60, "2021-03": 0.85, "2021-04": 0.82,
    "2021-05": 0.80, "2021-06": 0.87, "2021-07": 0.69, "2021-08": 0.77,
    "2021-09": 0.93, "2021-10": 1.11, "2021-11": 1.18, "2021-12": 1.26,
    "2022-01": 1.55, "2022-02": 1.71, "2022-03": 2.01, "2022-04": 2.56,
    "2022-05": 2.82, "2022-06": 3.04, "2022-07": 2.88, "2022-08": 3.01,
    "2022-09": 3.77, "2022-10": 4.14, "2022-11": 3.86, "2022-12": 3.80,
    "2023-01": 3.62, "2023-02": 3.94, "2023-03": 3.80, "2023-04": 3.50,
    "2023-05": 3.57, "2023-06": 3.92, "2023-07": 4.11, "2023-08": 4.34,
    "2023-09": 4.60, "2023-10": 4.83, "2023-11": 4.40, "2023-12": 3.92,
    "2024-01": 3.97, "2024-02": 4.12, "2024-03": 4.15, "2024-04": 4.40,
    "2024-05": 4.42, "2024-06": 4.30, "2024-07": 4.15, "2024-08": 3.75,
    "2024-09": 3.60, "2024-10": 3.95, "2024-11": 4.10, "2024-12": 4.15,
    "2025-01": 4.20, "2025-02": 4.12, "2025-03": 4.05, "2025-04": 4.00,
    "2025-05": 3.95, "2025-06": 3.90,
}

SOFR_MONTHLY: dict[str, float] = {
    "2018-05": 1.80, "2018-06": 1.91, "2018-07": 1.93, "2018-08": 1.92,
    "2018-09": 2.19, "2018-10": 2.19, "2018-11": 2.19, "2018-12": 2.40,
    "2019-01": 2.40, "2019-02": 2.40, "2019-03": 2.40, "2019-04": 2.45,
    "2019-05": 2.39, "2019-06": 2.38, "2019-07": 2.40, "2019-08": 2.13,
    "2019-09": 2.04, "2019-10": 1.82, "2019-11": 1.55, "2019-12": 1.55,
    "2020-01": 1.55, "2020-02": 1.58, "2020-03": 0.26, "2020-04": 0.01,
    "2020-05": 0.05, "2020-06": 0.08, "2020-07": 0.09, "2020-08": 0.09,
    "2020-09": 0.08, "2020-10": 0.08, "2020-11": 0.09, "2020-12": 0.07,
    "2021-01": 0.06, "2021-02": 0.03, "2021-03": 0.01, "2021-04": 0.01,
    "2021-05": 0.01, "2021-06": 0.05, "2021-07": 0.05, "2021-08": 0.05,
    "2021-09": 0.05, "2021-10": 0.05, "2021-11": 0.05, "2021-12": 0.05,
    "2022-01": 0.05, "2022-02": 0.05, "2022-03": 0.20, "2022-04": 0.30,
    "2022-05": 0.80, "2022-06": 1.21, "2022-07": 1.54, "2022-08": 2.29,
    "2022-09": 2.56, "2022-10": 3.05, "2022-11": 3.78, "2022-12": 4.05,
    "2023-01": 4.30, "2023-02": 4.55, "2023-03": 4.55, "2023-04": 4.80,
    "2023-05": 5.06, "2023-06": 5.08, "2023-07": 5.12, "2023-08": 5.30,
    "2023-09": 5.31, "2023-10": 5.31, "2023-11": 5.32, "2023-12": 5.33,
    "2024-01": 5.33, "2024-02": 5.31, "2024-03": 5.33, "2024-04": 5.33,
    "2024-05": 5.33, "2024-06": 5.33, "2024-07": 5.34, "2024-08": 5.35,
    "2024-09": 4.96, "2024-10": 4.83, "2024-11": 4.58, "2024-12": 4.40,
    "2025-01": 4.35, "2025-02": 4.33, "2025-03": 4.30, "2025-04": 4.30,
    "2025-05": 4.30, "2025-06": 4.28,
}



# ─── Data Record ──────────────────────────────────────────────────────────────


@dataclass
class MarketDataRecord:
    """A single market data observation."""

    data_type: str        # "treasury_10y", "treasury_5y", "sofr", "cap_rate", "cre_price_index"
    observation_date: str  # ISO date or quarter label
    value: float          # Rate in percent (e.g., 4.25 = 4.25%)
    frequency: str        # "daily", "monthly", "quarterly"
    property_type: str | None = None  # Only for cap_rate records
    metro: str | None = None          # Only for cap_rate records (metro-adjusted)
    series_id: str | None = None      # FRED series ID if applicable

    # Ingestion metadata
    ingested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "market_data_fetcher"
    source_version: str = "1.0.0"


# ─── FRED API Client ──────────────────────────────────────────────────────────


class FREDClient:
    """
    Client for FRED (Federal Reserve Economic Data) API.

    Requires FRED_API_KEY environment variable.
    See: https://fred.stlouisfed.org/docs/api/fred/
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            logger.warning("FRED_API_KEY not set — FRED fetches will fail. Use --offline mode.")

    def fetch_series(
        self,
        series_id: str,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
        frequency: str = "m",  # d=daily, m=monthly, q=quarterly
    ) -> list[dict[str, Any]]:
        """
        Fetch a FRED series.

        Returns list of {"date": "YYYY-MM-DD", "value": float} dicts.
        """
        try:
            import requests
        except ImportError:
            logger.error("requests library not available. Use --offline mode.")
            return []

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": frequency,
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")
            return []

        observations = []
        for obs in data.get("observations", []):
            if obs["value"] != ".":  # FRED uses "." for missing
                try:
                    observations.append({
                        "date": obs["date"],
                        "value": float(obs["value"]),
                    })
                except (ValueError, KeyError):
                    continue

        logger.info(f"Fetched {len(observations)} observations for {series_id}")
        return observations



# ─── Cap Rate Loader ──────────────────────────────────────────────────────────


class CapRateLoader:
    """
    Loads historical cap rate data from config/cap_rates_historical.yaml.

    Produces per-property-type quarterly cap rates, with optional metro adjustments.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = CONFIG_DIR / "cap_rates_historical.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Cap rates config not found: {config_path}")

        self.config = load_yaml_file(config_path)

        self.metro_adjustments = self.config.get("metro_adjustments_bps", {})
        self.property_types = [
            k for k in self.config.keys()
            if k not in ("metro_adjustments_bps",)
        ]

    def get_national_cap_rates(self) -> list[MarketDataRecord]:
        """Get national-level cap rates by property type and quarter."""
        records: list[MarketDataRecord] = []

        for prop_type in self.property_types:
            series = self.config[prop_type]
            if not isinstance(series, dict):
                continue
            for quarter, rate in series.items():
                records.append(MarketDataRecord(
                    data_type="cap_rate",
                    observation_date=str(quarter),
                    value=float(rate),
                    frequency="quarterly",
                    property_type=prop_type,
                    metro="National",
                    series_id=None,
                    source="cap_rates_historical.yaml",
                ))

        logger.info(f"Loaded {len(records)} national cap rate observations")
        return records

    def get_metro_adjusted_cap_rates(
        self, metros: list[str] | None = None
    ) -> list[MarketDataRecord]:
        """
        Get metro-adjusted cap rates.

        Applies metro_adjustments_bps from config to national averages.
        """
        national = self.get_national_cap_rates()
        if metros is None:
            metros = list(self.metro_adjustments.keys())

        metro_records: list[MarketDataRecord] = []
        for record in national:
            for metro in metros:
                adj_bps = self.metro_adjustments.get(metro, 0)
                adjusted_rate = record.value + adj_bps / 100.0
                metro_records.append(MarketDataRecord(
                    data_type="cap_rate",
                    observation_date=record.observation_date,
                    value=round(adjusted_rate, 2),
                    frequency="quarterly",
                    property_type=record.property_type,
                    metro=metro,
                    series_id=None,
                    source="cap_rates_historical.yaml",
                ))

        logger.info(
            f"Generated {len(metro_records)} metro-adjusted cap rate observations "
            f"({len(metros)} metros)"
        )
        return metro_records



# ─── Market Data Assembler ────────────────────────────────────────────────────


class MarketDataFetcher:
    """
    Orchestrates all market data fetching and assembly.

    Modes:
      - Online: Fetches from FRED API + loads cap rates from config
      - Offline: Uses hardcoded monthly series + cap rates from config
    """

    def __init__(self, offline: bool = False, api_key: str | None = None):
        self.offline = offline
        self.fred = FREDClient(api_key=api_key) if not offline else None
        self.cap_rate_loader = CapRateLoader()

    def fetch_all(self) -> list[MarketDataRecord]:
        """Fetch all market data series."""
        records: list[MarketDataRecord] = []

        # Treasury rates
        records.extend(self._fetch_treasury_rates())

        # SOFR
        records.extend(self._fetch_sofr())

        # Cap rates (always from config)
        records.extend(self.cap_rate_loader.get_national_cap_rates())
        records.extend(self.cap_rate_loader.get_metro_adjusted_cap_rates())

        # CRE Price Index
        records.extend(self._fetch_cre_price_index())

        logger.info(f"Total market data records assembled: {len(records)}")
        return records

    def _fetch_treasury_rates(self) -> list[MarketDataRecord]:
        """Fetch Treasury 10Y and 5Y rates."""
        records: list[MarketDataRecord] = []

        if self.offline or self.fred is None:
            # Use hardcoded monthly data
            for month_key, value in TREASURY_10Y_MONTHLY.items():
                records.append(MarketDataRecord(
                    data_type="treasury_10y",
                    observation_date=f"{month_key}-01",
                    value=value,
                    frequency="monthly",
                    series_id="DGS10",
                    source="hardcoded_historical",
                ))
            for month_key, value in TREASURY_5Y_MONTHLY.items():
                records.append(MarketDataRecord(
                    data_type="treasury_5y",
                    observation_date=f"{month_key}-01",
                    value=value,
                    frequency="monthly",
                    series_id="DGS5",
                    source="hardcoded_historical",
                ))
            logger.info(
                f"Loaded {len(records)} Treasury observations from hardcoded data (offline)"
            )
        else:
            # Fetch from FRED
            for data_type, series_id in [("treasury_10y", "DGS10"), ("treasury_5y", "DGS5")]:
                observations = self.fred.fetch_series(series_id, frequency="m")
                for obs in observations:
                    records.append(MarketDataRecord(
                        data_type=data_type,
                        observation_date=obs["date"],
                        value=obs["value"],
                        frequency="monthly",
                        series_id=series_id,
                        source="fred_api",
                    ))

        return records

    def _fetch_sofr(self) -> list[MarketDataRecord]:
        """Fetch SOFR rate."""
        records: list[MarketDataRecord] = []

        if self.offline or self.fred is None:
            for month_key, value in SOFR_MONTHLY.items():
                records.append(MarketDataRecord(
                    data_type="sofr",
                    observation_date=f"{month_key}-01",
                    value=value,
                    frequency="monthly",
                    series_id="SOFR",
                    source="hardcoded_historical",
                ))
            logger.info(f"Loaded {len(records)} SOFR observations from hardcoded data (offline)")
        else:
            observations = self.fred.fetch_series("SOFR", frequency="m")
            for obs in observations:
                records.append(MarketDataRecord(
                    data_type="sofr",
                    observation_date=obs["date"],
                    value=obs["value"],
                    frequency="monthly",
                    series_id="SOFR",
                    source="fred_api",
                ))

        return records

    def _fetch_cre_price_index(self) -> list[MarketDataRecord]:
        """Fetch CRE Price Index (or synthetic proxy)."""
        records: list[MarketDataRecord] = []

        if self.offline or self.fred is None:
            # Generate a synthetic CRE price index (base=100 in 2015-Q1)
            base_value = 100.0
            quarters = []
            for year in range(2015, 2026):
                for q in range(1, 5):
                    quarters.append(f"{year}-Q{q}")

            # CRE prices rose ~40% 2015-2021, then pulled back ~15% 2022-2024
            growth_rates = {
                2015: 0.02, 2016: 0.015, 2017: 0.02, 2018: 0.018,
                2019: 0.015, 2020: -0.01, 2021: 0.025, 2022: -0.02,
                2023: -0.03, 2024: -0.01, 2025: 0.005,
            }
            current_value = base_value
            for quarter in quarters:
                year = int(quarter[:4])
                quarterly_growth = growth_rates.get(year, 0.01) / 4.0
                current_value *= (1 + quarterly_growth)
                records.append(MarketDataRecord(
                    data_type="cre_price_index",
                    observation_date=quarter,
                    value=round(current_value, 2),
                    frequency="quarterly",
                    series_id="SYNTHETIC_CREPI",
                    source="synthetic_index",
                ))
            logger.info(f"Generated {len(records)} CRE price index observations (synthetic)")
        else:
            observations = self.fred.fetch_series("BOGZ1FL075035503Q", frequency="q")
            for obs in observations:
                records.append(MarketDataRecord(
                    data_type="cre_price_index",
                    observation_date=obs["date"],
                    value=obs["value"],
                    frequency="quarterly",
                    series_id="BOGZ1FL075035503Q",
                    source="fred_api",
                ))

        return records



# ─── Writers ──────────────────────────────────────────────────────────────────


def write_market_data(records: list[MarketDataRecord], output_path: Path) -> None:
    """
    Write market data to Delta Lake format.

    Falls back to Parquet (via pyarrow) or JSON-lines if Delta/Spark unavailable.
    Partitions by data_type.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(r) for r in records]

    # Attempt 1: Delta Lake via pyspark
    try:
        from pyspark.sql import SparkSession
        from delta import configure_spark_with_delta_pip

        builder = (
            SparkSession.builder.master("local[*]")
            .appName("cre-market-fetcher")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        df = spark.createDataFrame(dicts)
        (
            df.write.format("delta")
            .mode("overwrite")
            .partitionBy("data_type")
            .save(str(output_path))
        )
        logger.info(f"Written {len(records)} market records as Delta Lake to {output_path}")
        spark.stop()
        return
    except ImportError:
        logger.info("PySpark not available, trying pyarrow Parquet fallback...")
    except Exception as e:
        logger.warning(f"Delta write failed ({e}), trying Parquet fallback...")

    # Attempt 2: Parquet via pyarrow
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(dicts)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=["data_type"],
        )
        logger.info(f"Written {len(records)} market records as Parquet to {output_path}")
        return
    except ImportError:
        logger.info("pyarrow not available, using Delta writer fallback...")

    # Attempt 3: Pure-Python Delta writer (produces _delta_log/)
    from src.utils.delta_writer import DeltaWriter

    writer = DeltaWriter(output_path)
    writer.write(dicts, partition_by="data_type", mode="overwrite")
    logger.info(f"Written {len(records)} market records as Delta (pure-Python) to {output_path}")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch and assemble market data for the CRE bronze layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use hardcoded/config data instead of FRED API (no network required)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="FRED API key (overrides FRED_API_KEY env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the market data fetcher."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fetcher = MarketDataFetcher(offline=args.offline, api_key=args.api_key)
    records = fetcher.fetch_all()

    write_market_data(records, args.output)

    # Summary
    type_counts: dict[str, int] = {}
    for r in records:
        type_counts[r.data_type] = type_counts.get(r.data_type, 0) + 1

    logger.info("─── Market Data Summary ───")
    logger.info(f"  Total records: {len(records)}")
    for dtype, count in sorted(type_counts.items()):
        logger.info(f"  {dtype}: {count} observations")


if __name__ == "__main__":
    main()
