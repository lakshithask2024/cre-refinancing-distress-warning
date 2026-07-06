"""
Silver Layer — Market Data Standardization
=============================================

Reads bronze market data Delta table and produces silver_market_rates with:
- Treasury rates: monthly observations, interpolation for missing months
- Cap rates: quarterly, rolling-average smoothed
- SOFR: monthly observations
- CRE price index: quarterly observations
- Standardized output schema across all series

Input:  data/bronze/market (Delta)
Output: data/silver/market_rates (Delta)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


def transform_silver_market(
    bronze_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Transform bronze market records into standardized silver market rates.

    Steps:
    1. Separate by data_type
    2. Standardize observation_date to YYYY-MM-DD format
    3. Interpolate missing months for treasury/sofr (linear)
    4. Smooth cap rates with quarterly rolling average
    5. Output unified schema

    Returns:
        Standardized silver market rate records
    """
    logger.info(f"Silver market: processing {len(bronze_records)} bronze records")

    # Separate by data type
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in bronze_records:
        by_type[r.get("data_type", "unknown")].append(r)

    silver_records: list[dict[str, Any]] = []

    # Process Treasury rates (10Y and 5Y)
    for rate_type in ["treasury_10y", "treasury_5y"]:
        if rate_type in by_type:
            processed = _process_treasury(by_type[rate_type], rate_type)
            silver_records.extend(processed)
            logger.info(f"  {rate_type}: {len(processed)} records (interpolated)")

    # Process SOFR
    if "sofr" in by_type:
        processed = _process_sofr(by_type["sofr"])
        silver_records.extend(processed)
        logger.info(f"  sofr: {len(processed)} records")

    # Process cap rates
    if "cap_rate" in by_type:
        processed = _process_cap_rates(by_type["cap_rate"])
        silver_records.extend(processed)
        logger.info(f"  cap_rate: {len(processed)} records (smoothed)")

    # Process CRE price index
    if "cre_price_index" in by_type:
        processed = _process_cre_index(by_type["cre_price_index"])
        silver_records.extend(processed)
        logger.info(f"  cre_price_index: {len(processed)} records")

    # Add silver metadata
    silver_ts = datetime.utcnow().isoformat()
    for r in silver_records:
        r["_silver_processed_at"] = silver_ts

    logger.info(f"Silver market: output {len(silver_records)} standardized records")
    return silver_records


def _process_treasury(
    records: list[dict[str, Any]], rate_type: str
) -> list[dict[str, Any]]:
    """Process and interpolate treasury rate series."""
    # Parse and sort by date
    dated: list[tuple[str, float]] = []
    for r in records:
        obs_date = _normalize_date(r["observation_date"])
        if obs_date and r.get("value") is not None:
            dated.append((obs_date, float(r["value"])))

    dated.sort(key=lambda x: x[0])

    # Interpolate missing months
    if dated:
        interpolated = _interpolate_monthly(dated)
    else:
        interpolated = dated

    # Build output records
    output = []
    for obs_date, value in interpolated:
        output.append({
            "data_type": rate_type,
            "observation_date": obs_date,
            "value": round(value, 4),
            "value_decimal": round(value / 100.0, 6),  # Convert % to decimal
            "frequency": "monthly",
            "property_type": None,
            "metro": None,
        })
    return output


def _process_sofr(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process SOFR rate series."""
    output = []
    for r in records:
        obs_date = _normalize_date(r["observation_date"])
        if obs_date and r.get("value") is not None:
            output.append({
                "data_type": "sofr",
                "observation_date": obs_date,
                "value": round(float(r["value"]), 4),
                "value_decimal": round(float(r["value"]) / 100.0, 6),
                "frequency": "monthly",
                "property_type": None,
                "metro": None,
            })
    output.sort(key=lambda x: x["observation_date"])
    return output


def _process_cap_rates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process cap rates with quarterly rolling average smoothing."""
    # Group by (property_type, metro)
    groups: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for r in records:
        ptype = r.get("property_type") or "unknown"
        metro = r.get("metro") or "National"
        obs_date = r.get("observation_date", "")
        value = r.get("value")
        if value is not None:
            groups[(ptype, metro)].append((obs_date, float(value)))

    output = []
    for (ptype, metro), series in groups.items():
        series.sort(key=lambda x: x[0])

        # Apply 3-quarter rolling average for smoothing
        smoothed = _rolling_average(series, window=3)

        for obs_date, value in smoothed:
            output.append({
                "data_type": "cap_rate",
                "observation_date": obs_date,
                "value": round(value, 4),
                "value_decimal": round(value / 100.0, 6),
                "frequency": "quarterly",
                "property_type": ptype,
                "metro": metro,
            })

    return output


def _process_cre_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process CRE price index."""
    output = []
    for r in records:
        obs_date = r.get("observation_date", "")
        value = r.get("value")
        if value is not None:
            output.append({
                "data_type": "cre_price_index",
                "observation_date": obs_date,
                "value": round(float(value), 2),
                "value_decimal": round(float(value) / 100.0, 4),  # Index as ratio to base
                "frequency": "quarterly",
                "property_type": None,
                "metro": None,
            })
    output.sort(key=lambda x: x["observation_date"])
    return output


# ─── Helper Functions ─────────────────────────────────────────────────────────


def _normalize_date(date_str: str) -> str | None:
    """Normalize various date formats to YYYY-MM-DD."""
    if not date_str:
        return None

    # Already YYYY-MM-DD
    if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
        return date_str

    # Quarter format: YYYY-Qn → YYYY-MM-DD (mid-quarter)
    if "Q" in date_str.upper():
        parts = date_str.upper().replace("-Q", "-").split("-")
        if len(parts) == 2:
            year = int(parts[0])
            quarter = int(parts[1])
            # Map quarter to mid-month: Q1=02, Q2=05, Q3=08, Q4=11
            month = {1: 2, 2: 5, 3: 8, 4: 11}.get(quarter, 2)
            return f"{year:04d}-{month:02d}-15"

    # YYYY-MM format
    if len(date_str) == 7 and date_str[4] == "-":
        return f"{date_str}-01"

    return date_str


def _interpolate_monthly(
    series: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """
    Linear interpolation to fill missing months in a monthly series.

    Input: sorted list of (YYYY-MM-DD, value) tuples.
    Output: same format with gaps filled.
    """
    if len(series) < 2:
        return series

    # Parse to (year, month, value)
    parsed: list[tuple[int, int, float]] = []
    for date_str, value in series:
        try:
            y, m = int(date_str[:4]), int(date_str[5:7])
            parsed.append((y, m, value))
        except (ValueError, IndexError):
            continue

    if len(parsed) < 2:
        return series

    # Build complete monthly range
    start_y, start_m = parsed[0][0], parsed[0][1]
    end_y, end_m = parsed[-1][0], parsed[-1][1]

    # Index existing values
    existing: dict[tuple[int, int], float] = {}
    for y, m, v in parsed:
        existing[(y, m)] = v

    # Generate all months
    result: list[tuple[str, float]] = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        if (y, m) in existing:
            result.append((f"{y:04d}-{m:02d}-01", existing[(y, m)]))
        else:
            # Linear interpolation from nearest known points
            value = _lerp_value(y, m, parsed)
            if value is not None:
                result.append((f"{y:04d}-{m:02d}-01", value))

        # Advance month
        m += 1
        if m > 12:
            m = 1
            y += 1

    return result


def _lerp_value(
    target_y: int, target_m: int, series: list[tuple[int, int, float]]
) -> float | None:
    """Linear interpolation for a missing month."""
    target_months = target_y * 12 + target_m

    # Find bracketing points
    before: tuple[int, float] | None = None
    after: tuple[int, float] | None = None

    for y, m, v in series:
        month_idx = y * 12 + m
        if month_idx <= target_months:
            before = (month_idx, v)
        if month_idx >= target_months and after is None:
            after = (month_idx, v)
            break

    if before is None or after is None:
        return before[1] if before else (after[1] if after else None)

    if before[0] == after[0]:
        return before[1]

    # Linear interpolation
    fraction = (target_months - before[0]) / (after[0] - before[0])
    return before[1] + fraction * (after[1] - before[1])


def _rolling_average(
    series: list[tuple[str, float]], window: int = 3
) -> list[tuple[str, float]]:
    """Apply rolling average smoothing to a time series."""
    if len(series) <= window:
        return series

    values = [v for _, v in series]
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        avg = sum(window_values) / len(window_values)
        smoothed.append((series[i][0], avg))

    return smoothed
