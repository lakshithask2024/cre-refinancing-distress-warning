"""
Silver Layer — Feature Engineering
=====================================

Joins silver loans with silver market rates to compute derived features
for the distress model. Produces silver_loan_features table.

Derived features:
- current_value: noi / current_cap_rate
- current_ltv: current_balance / current_value
- refinance_rate: treasury_10y + property-type spread
- rate_gap: refinance_rate - note_rate
- new_dscr: noi / (annual_debt_service_at_refi_rate)
- debt_yield: noi / current_balance
- months_to_maturity: (already computed in silver_loans)

Input:  data/silver/loans (Delta), data/silver/market_rates (Delta)
Output: data/silver/loan_features (Delta)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Property-type credit spreads over Treasury 10Y (bps)
# These represent typical CMBS refinance spread by property type
REFI_SPREAD_BPS: dict[str, float] = {
    "office": 250,
    "retail": 275,
    "industrial": 200,
    "multifamily": 180,
    "hotel": 325,
}

# Assumed amortization period for debt service calculation (months)
AMORT_MONTHS = 360  # 30-year amortization schedule


def compute_loan_features(
    silver_loans: list[dict[str, Any]],
    silver_market: list[dict[str, Any]],
    reference_date: date | None = None,
) -> list[dict[str, Any]]:
    """
    Join silver loans with market data and compute derived features.

    Args:
        silver_loans: Cleaned loan records from silver_loans
        silver_market: Standardized market records from silver_market_rates
        reference_date: Date for "current" market lookups (default: today)

    Returns:
        Loan records enriched with derived features
    """
    if reference_date is None:
        reference_date = date.today()

    logger.info(
        f"Feature engineering: {len(silver_loans)} loans × "
        f"{len(silver_market)} market records"
    )

    # Build market data lookup structures
    treasury_10y = _build_rate_lookup(silver_market, "treasury_10y")
    cap_rates = _build_cap_rate_lookup(silver_market)

    # Get latest available rates as "current" for feature calculation
    latest_treasury = _get_latest_rate(treasury_10y)
    latest_cap_rates = _get_latest_cap_rates(cap_rates)

    logger.info(f"  Latest Treasury 10Y: {latest_treasury:.2f}%")
    logger.info(f"  Latest cap rates by type: { {k: f'{v:.2f}%' for k, v in latest_cap_rates.items()} }")

    # Compute features for each loan
    enriched = []
    join_failures = 0

    for loan in silver_loans:
        features = _compute_single_loan_features(
            loan, latest_treasury, latest_cap_rates, reference_date
        )
        if features is not None:
            enriched.append(features)
        else:
            join_failures += 1

    if join_failures > 0:
        logger.warning(f"  {join_failures} loans failed feature computation (missing market data)")

    # Add processing metadata
    proc_ts = datetime.utcnow().isoformat()
    for r in enriched:
        r["_feature_computed_at"] = proc_ts

    logger.info(f"Feature engineering: output {len(enriched)} enriched records")
    return enriched


def _compute_single_loan_features(
    loan: dict[str, Any],
    latest_treasury: float,
    latest_cap_rates: dict[str, float],
    reference_date: date,
) -> dict[str, Any] | None:
    """Compute derived features for a single loan."""
    # Start with all loan fields
    features = dict(loan)

    property_type = loan.get("property_type", "").lower()
    metro = loan.get("metro_area", "")

    # Get current cap rate for this property type
    current_cap_rate = latest_cap_rates.get(property_type)
    if current_cap_rate is None:
        # Fallback to overall average
        if latest_cap_rates:
            current_cap_rate = sum(latest_cap_rates.values()) / len(latest_cap_rates)
        else:
            return None  # Can't compute without cap rates

    features["current_cap_rate"] = round(current_cap_rate, 4)

    # Current value: NOI / cap_rate (cap rate in decimal)
    noi = loan.get("noi_annual", 0)
    if noi and current_cap_rate > 0:
        cap_decimal = current_cap_rate / 100.0
        current_value = noi / cap_decimal
    else:
        current_value = loan.get("property_value_at_origination", 0)

    features["current_value"] = round(current_value, 2)

    # Current LTV: current_balance / current_value
    current_balance = loan.get("current_balance", 0)
    if current_value > 0:
        current_ltv = current_balance / current_value
    else:
        current_ltv = loan.get("ltv_at_origination", 0)

    features["current_ltv"] = round(current_ltv, 4)

    # Refinance rate: Treasury 10Y + property-type spread
    spread_bps = REFI_SPREAD_BPS.get(property_type, 250)
    refinance_rate = (latest_treasury + spread_bps / 100.0) / 100.0  # as decimal
    features["refinance_rate"] = round(refinance_rate, 6)
    features["refinance_rate_pct"] = round(refinance_rate * 100, 4)

    # Rate gap: refinance_rate - note_rate (both as decimal)
    note_rate = loan.get("note_rate", 0)
    rate_gap = refinance_rate - note_rate
    features["rate_gap"] = round(rate_gap, 6)
    features["rate_gap_bps"] = round(rate_gap * 10000, 1)

    # New DSCR at refinance rate
    # Annual debt service = balance * (monthly_rate * (1+monthly_rate)^n) / ((1+monthly_rate)^n - 1) * 12
    # For IO loans: annual_ds = balance * annual_rate
    amort_type = loan.get("amortization_type", "amortizing")
    if amort_type == "interest_only":
        annual_debt_service = current_balance * refinance_rate
    else:
        monthly_rate = refinance_rate / 12.0
        if monthly_rate > 0:
            factor = (
                monthly_rate * (1 + monthly_rate) ** AMORT_MONTHS
            ) / ((1 + monthly_rate) ** AMORT_MONTHS - 1)
            annual_debt_service = current_balance * factor * 12
        else:
            annual_debt_service = current_balance / AMORT_MONTHS * 12

    if annual_debt_service > 0:
        new_dscr = noi / annual_debt_service
    else:
        new_dscr = 99.0  # effectively no debt service

    features["new_dscr"] = round(new_dscr, 4)
    features["annual_debt_service_refi"] = round(annual_debt_service, 2)

    # Debt yield: NOI / current_balance
    if current_balance > 0:
        debt_yield = noi / current_balance
    else:
        debt_yield = 0

    features["debt_yield"] = round(debt_yield, 4)
    features["debt_yield_pct"] = round(debt_yield * 100, 2)

    # Distress indicators (rule-based, pre-model)
    features["_refi_stressed"] = bool(new_dscr < 1.0 or current_ltv > 0.80)
    features["_rate_gap_positive"] = bool(rate_gap > 0)

    return features


# ─── Market Data Lookup Builders ──────────────────────────────────────────────


def _build_rate_lookup(
    market_records: list[dict[str, Any]], data_type: str
) -> dict[str, float]:
    """Build a date→value lookup for a specific rate type."""
    lookup: dict[str, float] = {}
    for r in market_records:
        if r.get("data_type") == data_type:
            obs_date = r.get("observation_date", "")
            value = r.get("value")
            if obs_date and value is not None:
                lookup[obs_date] = float(value)
    return lookup


def _build_cap_rate_lookup(
    market_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], float]:
    """Build a (date, property_type, metro) → cap_rate lookup."""
    lookup: dict[tuple[str, str, str], float] = {}
    for r in market_records:
        if r.get("data_type") == "cap_rate":
            obs_date = r.get("observation_date", "")
            ptype = (r.get("property_type") or "").lower()
            metro = r.get("metro") or "National"
            value = r.get("value")
            if obs_date and ptype and value is not None:
                lookup[(obs_date, ptype, metro)] = float(value)
    return lookup


def _get_latest_rate(rate_lookup: dict[str, float]) -> float:
    """Get the most recent rate value from a date-indexed lookup."""
    if not rate_lookup:
        return 4.0  # Fallback: approximate current 10Y

    latest_date = max(rate_lookup.keys())
    return rate_lookup[latest_date]


def _get_latest_cap_rates(
    cap_lookup: dict[tuple[str, str, str], float],
) -> dict[str, float]:
    """Get the most recent national-level cap rate per property type."""
    # Find latest date in the cap rate data
    all_dates = set(d for d, _, _ in cap_lookup.keys())
    if not all_dates:
        return {"office": 7.5, "retail": 7.2, "industrial": 5.2, "multifamily": 5.5, "hotel": 8.3}

    latest_date = max(all_dates)

    # Get national rates at that date (or any metro as fallback)
    result: dict[str, float] = {}
    for (d, ptype, metro), value in cap_lookup.items():
        if d == latest_date and metro == "National":
            result[ptype] = value

    # Fallback: if no "National" records, average across metros
    if not result:
        type_values: dict[str, list[float]] = defaultdict(list)
        for (d, ptype, metro), value in cap_lookup.items():
            if d == latest_date:
                type_values[ptype].append(value)
        for ptype, values in type_values.items():
            result[ptype] = sum(values) / len(values)

    return result
