"""
Feature Engineering for Distress Classifier — Forward-Looking Prediction
==========================================================================

TASK FRAMING (leakage-free):
  Predict whether a loan will be in refinancing distress AT MATURITY, using
  only information observable 24 months BEFORE maturity (T_obs).

  - T_obs = maturity_date - 24 months (the "early warning" observation point)
  - Features = loan attributes + market conditions snapshotted AT T_obs
  - Label = refinancing viability computed AT maturity_date using maturity-date
    market conditions

  This ensures NO future information leaks into features:
    Features use market data from T_obs (the past relative to maturity).
    Label uses market data from maturity_date (the future relative to T_obs).

FILTER:
  Only loans where BOTH T_obs and maturity_date fall within the available
  market data date range are included (no extrapolation).

SPLIT:
  Train: origination_year <= 2018
  Valid: origination_year == 2019
  Test:  origination_year >= 2020
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Feature column definitions ──────────────────────────────────────────────

# Features observed at T_obs (24mo before maturity)
NUMERIC_FEATURES_AT_TOBS = [
    "ltv_at_origination",
    "dscr_at_origination",
    "note_rate",
    "log_original_balance",
    "occupancy_pct",
    "months_since_origination_at_Tobs",
    "treasury_10y_at_Tobs",
    "cap_rate_at_Tobs",
    "rate_gap_bps_at_Tobs",
    "cap_rate_delta_since_origination_at_Tobs",
    "current_ltv_at_Tobs",
    "current_dscr_at_Tobs",
]

ONEHOT_FEATURES = [
    "property_type",
    "sponsor_credit_tier",
    "amortization_type",
    "balloon_flag",
]

TARGET_ENCODED_FEATURES = ["metro_area"]

LABEL_COL = "distress_at_maturity"

# Refinance spread over Treasury 10Y by property type (bps)
REFI_SPREAD_BPS: dict[str, float] = {
    "office": 250,
    "retail": 275,
    "industrial": 200,
    "multifamily": 180,
    "hotel": 325,
}


class FeatureLeakageError(Exception):
    """Raised when a sanity check detects potential label leakage."""
    pass


def build_training_frame(
    gold_path: str | Path = "data/gold/loan_current_state",
    market_path: str | Path = "data/silver/market_rates",
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    """
    Build leakage-free training data for distress prediction at maturity.

    The key insight: features are snapshotted 24 months before maturity (T_obs),
    while the label is determined by market conditions AT maturity. This creates
    a genuine prediction problem — can we identify distress risk 2 years early?

    Returns:
        (X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names)
    """
    np.random.seed(seed)

    # ─── Load data ────────────────────────────────────────────────────────────
    loans_df = _load_delta_as_df(gold_path)
    market_df = _load_delta_as_df(market_path)

    logger.info(f"Loaded {len(loans_df)} loans, {len(market_df)} market records")

    # ─── Build market lookup tables ───────────────────────────────────────────
    treasury_lookup = _build_treasury_lookup(market_df)
    cap_rate_lookup = _build_cap_rate_lookup(market_df)

    market_start, market_end = _get_market_date_range(treasury_lookup)
    logger.info(f"Market data range: {market_start} to {market_end}")

    # ─── Compute T_obs and filter loans ───────────────────────────────────────
    loans_df["maturity_date_parsed"] = pd.to_datetime(loans_df["maturity_date"], errors="coerce")
    loans_df["origination_date_parsed"] = pd.to_datetime(loans_df["origination_date"], errors="coerce")
    loans_df["T_obs"] = loans_df["maturity_date_parsed"] - pd.Timedelta(days=730)  # ~24 months

    # Filter: both T_obs and maturity must be within market data range
    mask_tobs = loans_df["T_obs"] >= pd.Timestamp(market_start)
    mask_maturity = loans_df["maturity_date_parsed"] <= pd.Timestamp(market_end)
    mask_valid_dates = (loans_df["maturity_date_parsed"].notna()) & (loans_df["T_obs"].notna())

    loans_filtered = loans_df[mask_tobs & mask_maturity & mask_valid_dates].copy()
    n_dropped = len(loans_df) - len(loans_filtered)
    logger.info(
        f"Filter: {len(loans_filtered)} loans pass (dropped {n_dropped} "
        f"where T_obs or maturity outside market range)"
    )

    if len(loans_filtered) == 0:
        raise ValueError(
            "No loans pass the date filter. Check that maturity dates fall "
            "within market data range and T_obs (maturity - 24mo) is after market start."
        )

    # ─── Snapshot features at T_obs ───────────────────────────────────────────
    loans_filtered = _compute_features_at_Tobs(loans_filtered, treasury_lookup, cap_rate_lookup)

    # ─── Load shock config ──────────────────────────────────────────────────
    enable_shocks = _load_shock_config()

    # ─── Compute label at maturity ────────────────────────────────────────────
    loans_filtered = _compute_label_at_maturity(
        loans_filtered, treasury_lookup, cap_rate_lookup,
        enable_shocks=enable_shocks,
        surprise_default_prob=0.05,
    )

    # ─── Diagnostic: label distribution before split ──────────────────────────
    total_labels = len(loans_filtered)
    positive_labels = int(loans_filtered[LABEL_COL].sum())
    logger.info(
        f"  Label distribution (post-shock): {positive_labels}/{total_labels} distressed "
        f"({positive_labels/total_labels*100:.1f}%)"
    )

    # ─── Time-based split ─────────────────────────────────────────────────────
    loans_filtered["origination_year"] = pd.to_numeric(
        loans_filtered["origination_year"], errors="coerce"
    )
    train_mask = loans_filtered["origination_year"] <= 2018
    valid_mask = loans_filtered["origination_year"] == 2019
    test_mask = loans_filtered["origination_year"] >= 2020

    df_train = loans_filtered[train_mask].copy()
    df_valid = loans_filtered[valid_mask].copy()
    df_test = loans_filtered[test_mask].copy()

    # Diagnostic: per-split label balance
    for name, split_df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        if len(split_df) == 0:
            logger.warning(f"WARNING: {name} split is empty! Check origination_year distribution.")
        else:
            split_pos = int(split_df[LABEL_COL].sum())
            logger.info(
                f"  {name:5s}: {len(split_df):5d} rows, "
                f"label_mean={split_df[LABEL_COL].mean():.3f} "
                f"({split_pos} pos / {len(split_df)-split_pos} neg)"
            )

    logger.info(
        f"Split sizes — train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}"
    )

    # ─── Target encoding for metro (train-only) ──────────────────────────────
    if len(df_train) > 0:
        metro_means = df_train.groupby("metro_area")[LABEL_COL].mean()
        global_mean = df_train[LABEL_COL].mean()
    else:
        metro_means = pd.Series(dtype=float)
        global_mean = 0.5

    df_train["metro_encoded"] = df_train["metro_area"].map(metro_means).fillna(global_mean)
    df_valid["metro_encoded"] = df_valid["metro_area"].map(metro_means).fillna(global_mean)
    df_test["metro_encoded"] = df_test["metro_area"].map(metro_means).fillna(global_mean)

    # ─── One-hot encoding ─────────────────────────────────────────────────────
    df_train, df_valid, df_test, onehot_cols = _apply_onehot(df_train, df_valid, df_test)

    # ─── Assemble feature matrices ────────────────────────────────────────────
    feature_cols = NUMERIC_FEATURES_AT_TOBS + ["metro_encoded"] + onehot_cols
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    X_train = df_train[feature_cols].astype(float).fillna(0.0)
    X_valid = df_valid[feature_cols].astype(float).fillna(0.0)
    X_test = df_test[feature_cols].astype(float).fillna(0.0)

    y_train = df_train[LABEL_COL].astype(int)
    y_valid = df_valid[LABEL_COL].astype(int)
    y_test = df_test[LABEL_COL].astype(int)

    # ─── Sanity checks ────────────────────────────────────────────────────────
    _run_sanity_checks(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


# ─── Feature computation at T_obs ────────────────────────────────────────────


def _compute_features_at_Tobs(
    df: pd.DataFrame,
    treasury_lookup: dict[str, float],
    cap_rate_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """
    Snapshot market-derived features at T_obs (24 months before maturity).

    Why T_obs features, not current: Using current market values would leak
    information about future market movements into the feature set. The model
    should only see what was knowable at the early-warning observation point.
    """
    # Static loan features (don't change with time)
    df["log_original_balance"] = np.log1p(
        pd.to_numeric(df["original_balance"], errors="coerce").fillna(0)
    )
    df["note_rate"] = pd.to_numeric(df["note_rate"], errors="coerce").fillna(0)
    df["ltv_at_origination"] = pd.to_numeric(df["ltv_at_origination"], errors="coerce").fillna(0)
    df["dscr_at_origination"] = pd.to_numeric(df["dscr_at_origination"], errors="coerce").fillna(0)
    df["occupancy_pct"] = pd.to_numeric(df["occupancy_pct"], errors="coerce").fillna(0)

    # Time-varying features snapshotted at T_obs
    df["months_since_origination_at_Tobs"] = (
        (df["T_obs"] - df["origination_date_parsed"]).dt.days / 30.44
    ).fillna(0)

    # Market lookups at T_obs
    df["treasury_10y_at_Tobs"] = df["T_obs"].apply(
        lambda d: _lookup_treasury(d, treasury_lookup)
    )
    df["cap_rate_at_Tobs"] = df.apply(
        lambda row: _lookup_cap_rate(row["T_obs"], row.get("property_type", "office"), cap_rate_lookup),
        axis=1,
    )

    # Rate gap at T_obs: projected refi rate at T_obs minus original note rate
    df["refi_rate_at_Tobs"] = df.apply(
        lambda row: (row["treasury_10y_at_Tobs"] + REFI_SPREAD_BPS.get(str(row.get("property_type", "office")), 250) / 100.0) / 100.0,
        axis=1,
    )
    df["rate_gap_bps_at_Tobs"] = (df["refi_rate_at_Tobs"] - df["note_rate"]) * 10000

    # Cap rate delta since origination (as of T_obs)
    df["cap_rate_at_origination"] = df.apply(
        lambda row: _lookup_cap_rate(row["origination_date_parsed"], row.get("property_type", "office"), cap_rate_lookup),
        axis=1,
    )
    df["cap_rate_delta_since_origination_at_Tobs"] = df["cap_rate_at_Tobs"] - df["cap_rate_at_origination"]

    # Current LTV at T_obs: balance / (NOI / cap_rate_at_Tobs)
    noi = pd.to_numeric(df["noi_annual"], errors="coerce").fillna(0)
    balance = pd.to_numeric(df["current_balance"], errors="coerce").fillna(
        pd.to_numeric(df["original_balance"], errors="coerce").fillna(0)
    )
    cap_decimal = df["cap_rate_at_Tobs"] / 100.0
    value_at_Tobs = np.where(cap_decimal > 0, noi / cap_decimal, 0)
    df["current_ltv_at_Tobs"] = np.where(value_at_Tobs > 0, balance / value_at_Tobs, 0)

    # Current DSCR at T_obs: NOI / debt_service at refi_rate_at_Tobs
    refi_rate = df["refi_rate_at_Tobs"].values
    is_io = (df["amortization_type"].astype(str) == "interest_only").values
    ds = np.where(
        is_io,
        balance * refi_rate,
        _amort_annual_ds_vectorized(balance.values, refi_rate),
    )
    df["current_dscr_at_Tobs"] = np.where(ds > 0, noi.values / ds, 0)

    return df


# ─── Deterministic per-loan RNG ───────────────────────────────────────────────


def _load_shock_config() -> bool:
    """
    Load the enable_idiosyncratic_shocks flag from config/loan_generator.yaml.

    Returns True by default if config cannot be read.
    """
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "loan_generator.yaml"
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.utils.yaml_compat import load_yaml_file
        config = load_yaml_file(str(config_path))
        return bool(config.get("enable_idiosyncratic_shocks", True))
    except Exception:
        return True  # Default: shocks enabled


def _loan_id_rng(loan_id: str) -> "random.Random":
    """
    Create a deterministic Random instance seeded from loan_id.

    Why: Shocks must be reproducible — same loan_id always gets same shock.
    Using hash(loan_id) as seed ensures this without requiring a global seed.
    """
    import hashlib
    import random as _random

    # Use MD5 hash for stable integer seed (not security-sensitive)
    seed_int = int(hashlib.md5(loan_id.encode()).hexdigest()[:8], 16)
    return _random.Random(seed_int)


# ─── Label computation at maturity ───────────────────────────────────────────


def _compute_label_at_maturity(
    df: pd.DataFrame,
    treasury_lookup: dict[str, float],
    cap_rate_lookup: dict[tuple[str, str], float],
    enable_shocks: bool = True,
    surprise_default_prob: float = 0.05,
) -> pd.DataFrame:
    """
    Compute the refinancing distress label using market conditions AT maturity,
    with optional idiosyncratic shocks that make the prediction genuinely
    probabilistic (prevents deterministic label-feature relationship).

    Label = 1 if loan would be in distress at maturity date:
      - maturity_dscr < 1.0 (can't cover debt service at maturity refi rate)
      OR
      - maturity_ltv > 0.90 (lender requires significant borrower cash-in to refi)

    Why 0.90 LTV (not 0.70):
      With cap rates expanding 150-200bps from origination, a 0.70 threshold
      mechanically flags ~65% of the portfolio as distressed just from the
      cap-rate-driven value decline — independent of any model features.
      The 0.90 threshold reflects the realistic point where a lender will
      actually reject a refinancing (10% equity remaining or less).
      This gives a ~44% base distress rate, leaving room for shocks to
      provide genuine stochasticity and for the model to learn meaningful
      discrimination.

    Shocks (applied ONLY to label, not features):
      1. NOI volatility: property-specific random shock to maturity NOI
      2. Occupancy shock: office/retail/hotel tenant loss events
      3. Submarket cap rate shock: 30% of loans get additional cap rate noise
      4. Surprise default: forced distress for unobservable risk factors
    """
    # Market conditions at maturity
    df["treasury_10y_at_maturity"] = df["maturity_date_parsed"].apply(
        lambda d: _lookup_treasury(d, treasury_lookup)
    )
    df["cap_rate_at_maturity"] = df.apply(
        lambda row: _lookup_cap_rate(row["maturity_date_parsed"], row.get("property_type", "office"), cap_rate_lookup),
        axis=1,
    )

    # Maturity refi rate
    df["maturity_refi_rate"] = df.apply(
        lambda row: (row["treasury_10y_at_maturity"] + REFI_SPREAD_BPS.get(str(row.get("property_type", "office")), 250) / 100.0) / 100.0,
        axis=1,
    )

    # Base NOI and balance
    noi_base = pd.to_numeric(df["noi_annual"], errors="coerce").fillna(0).values.copy()
    balance = pd.to_numeric(df["current_balance"], errors="coerce").fillna(
        pd.to_numeric(df["original_balance"], errors="coerce").fillna(0)
    ).values
    cap_rate_at_maturity = df["cap_rate_at_maturity"].values.copy()
    property_types = df["property_type"].astype(str).values
    loan_ids = df["loan_id"].astype(str).values

    # ─── Apply idiosyncratic shocks ──────────────────────────────────────────
    n_noi_shocks = 0
    n_tenant_loss = 0
    n_cap_shocks = 0
    noi_shock_values = []

    if enable_shocks:
        n_surprise_defaults = 0

        for i in range(len(df)):
            rng = _loan_id_rng(loan_ids[i])

            # Shock 1: Property-specific NOI volatility (WIDENED)
            noi_volatility = rng.uniform(0.10, 0.40)  # 10-40% annual std
            # 24-month horizon: std scales by sqrt(2)
            shock = rng.gauss(0, noi_volatility * (2 ** 0.5))
            shock = max(-0.60, min(0.60, shock))  # clip to [-60%, +60%]
            noi_base[i] = noi_base[i] * (1.0 + shock)
            noi_shock_values.append(shock)
            if abs(shock) > 0.01:
                n_noi_shocks += 1

            # Shock 2: Occupancy/tenant loss (office, retail, hotel)
            ptype = property_types[i].lower()
            if ptype in ("office", "retail"):
                if rng.random() < 0.25:  # 25% probability
                    tenant_loss = rng.uniform(0.15, 0.50)
                    noi_base[i] = noi_base[i] * (1.0 - tenant_loss)
                    n_tenant_loss += 1
            elif ptype == "hotel":
                if rng.random() < 0.10:  # 10% probability for hotel
                    tenant_loss = rng.uniform(0.15, 0.50)
                    noi_base[i] = noi_base[i] * (1.0 - tenant_loss)
                    n_tenant_loss += 1

            # Shock 3: Submarket cap rate shock (30% of all loans, 100bps std)
            if rng.random() < 0.30:
                cap_shock = rng.gauss(0, 1.00)  # 100 bps std, in percentage points
                cap_rate_at_maturity[i] = cap_rate_at_maturity[i] + cap_shock
                n_cap_shocks += 1

            # Shock 4: Surprise default (unobservable factors)
            if surprise_default_prob > 0 and rng.random() < surprise_default_prob:
                n_surprise_defaults += 1
                # Will force label=1 after metric computation (below)

        # Record which loans are surprise defaults (re-derive from same RNG)
        surprise_default_indices: list[int] = []
        for i in range(len(df)):
            rng2 = _loan_id_rng(loan_ids[i])
            # Consume draws in same order as above to reach Shock 4 draw
            rng2.uniform(0.10, 0.40)   # NOI volatility
            rng2.gauss(0, 1.0)         # NOI shock
            ptype = property_types[i].lower()
            if ptype in ("office", "retail"):
                rng2.random()           # tenant loss trigger
                rng2.uniform(0.15, 0.50)
            elif ptype == "hotel":
                rng2.random()
                rng2.uniform(0.15, 0.50)
            rng2.random()               # cap rate trigger
            rng2.gauss(0, 1.0)          # cap rate magnitude
            if rng2.random() < surprise_default_prob:    # surprise default trigger
                surprise_default_indices.append(i)

        logger.info(
            f"  Applied idiosyncratic shocks: "
            f"NOI (mean={np.mean(noi_shock_values):.4f}, std={np.std(noi_shock_values):.4f}), "
            f"{n_tenant_loss} tenant-loss events, "
            f"{n_cap_shocks} submarket cap rate shocks, "
            f"{n_surprise_defaults} surprise defaults"
        )
    else:
        n_surprise_defaults = 0
        surprise_default_indices = []
        logger.info("  Idiosyncratic shocks DISABLED (deterministic label computation)")

    # ─── Compute maturity metrics with (possibly shocked) values ─────────────
    cap_decimal_mat = cap_rate_at_maturity / 100.0
    maturity_value = np.where(cap_decimal_mat > 0, noi_base / cap_decimal_mat, 0)

    # Maturity LTV
    maturity_ltv = np.where(maturity_value > 0, balance / maturity_value, 99.0)

    # Maturity DSCR
    refi_rate_mat = df["maturity_refi_rate"].values
    is_io = (df["amortization_type"].astype(str) == "interest_only").values
    ds_mat = np.where(
        is_io,
        balance * refi_rate_mat,
        _amort_annual_ds_vectorized(balance, refi_rate_mat),
    )
    maturity_dscr = np.where(ds_mat > 0, noi_base / ds_mat, 0)

    # Label: distressed at maturity
    # Dual-trigger: DSCR below breakeven OR LTV above 90% (severe negative equity)
    df[LABEL_COL] = ((maturity_dscr < 1.0) | (maturity_ltv > 0.90)).astype(int)

    # Apply surprise defaults (Shock 4): force label=1 for ~5% of loans
    if enable_shocks and surprise_default_indices:
        label_values = df[LABEL_COL].values
        for idx in surprise_default_indices:
            label_values[idx] = 1
        df[LABEL_COL] = label_values

    # Drop the maturity-time columns (prevent leakage into features)
    df.drop(columns=[
        "treasury_10y_at_maturity", "cap_rate_at_maturity",
        "maturity_refi_rate",
    ], inplace=True, errors="ignore")

    return df


# ─── Market data lookup helpers ───────────────────────────────────────────────


def _build_treasury_lookup(market_df: pd.DataFrame) -> dict[str, float]:
    """Build date-string → rate lookup for Treasury 10Y."""
    t10 = market_df[market_df["data_type"] == "treasury_10y"]
    lookup: dict[str, float] = {}
    for _, row in t10.iterrows():
        obs = str(row["observation_date"])
        val = row.get("value")
        if val is not None:
            lookup[obs] = float(val)
    return lookup


def _build_cap_rate_lookup(market_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Build (date, property_type) → cap_rate lookup (national only)."""
    caps = market_df[
        (market_df["data_type"] == "cap_rate") &
        (market_df["metro"].isin(["National", None, ""]) | market_df["metro"].isna())
    ]
    lookup: dict[tuple[str, str], float] = {}
    for _, row in caps.iterrows():
        obs = str(row["observation_date"])
        ptype = str(row.get("property_type", "")).lower()
        val = row.get("value")
        if val is not None and ptype:
            lookup[(obs, ptype)] = float(val)
    return lookup


def _get_market_date_range(treasury_lookup: dict[str, float]) -> tuple[str, str]:
    """Get the earliest and latest dates in the treasury lookup."""
    if not treasury_lookup:
        return "2015-01-01", "2025-12-31"
    dates = sorted(treasury_lookup.keys())
    return dates[0], dates[-1]


def _lookup_treasury(dt: Any, lookup: dict[str, float]) -> float:
    """Look up Treasury 10Y rate for a given date (nearest available)."""
    if pd.isna(dt):
        return 4.0  # fallback
    target = pd.Timestamp(dt)
    target_str = target.strftime("%Y-%m-01")
    if target_str in lookup:
        return lookup[target_str]
    # Find nearest date
    dates = sorted(lookup.keys())
    for d in reversed(dates):
        if d <= target_str:
            return lookup[d]
    # If all dates are after target, use earliest
    return lookup[dates[0]] if dates else 4.0


def _lookup_cap_rate(dt: Any, property_type: str, lookup: dict[tuple[str, str], float]) -> float:
    """Look up cap rate for (date, property_type), matching nearest quarter."""
    if pd.isna(dt):
        return 6.5  # fallback
    target = pd.Timestamp(dt)
    ptype = str(property_type).lower()

    # Convert to quarter label
    quarter = (target.month - 1) // 3 + 1
    quarter_str = f"{target.year}-{target.month:02d}-15"  # normalized quarter date

    # Try exact quarter key format used in silver (YYYY-MM-15)
    key = (quarter_str, ptype)
    if key in lookup:
        return lookup[key]

    # Try YYYY-Qn format
    q_label = f"{target.year}-Q{quarter}"
    # Search through keys for matching quarter
    for (d, pt), val in lookup.items():
        if pt == ptype and q_label in d:
            return val

    # Fallback: find nearest date for this property type
    type_entries = [(d, v) for (d, pt), v in lookup.items() if pt == ptype]
    if type_entries:
        type_entries.sort(key=lambda x: x[0])
        target_approx = target.strftime("%Y-%m")
        for d, v in reversed(type_entries):
            if d[:7] <= target_approx:
                return v
        return type_entries[0][1]

    return 6.5  # absolute fallback


def _amort_annual_ds_vectorized(balance: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Compute annual debt service for amortizing loans (30yr schedule), vectorized."""
    monthly_rate = rate / 12.0
    # Avoid division by zero
    safe_mr = np.where(monthly_rate > 0, monthly_rate, 1e-10)
    pmt_factor = (safe_mr * (1 + safe_mr) ** 360) / ((1 + safe_mr) ** 360 - 1)
    annual_ds = balance * pmt_factor * 12
    # Zero out where rate was zero
    annual_ds = np.where(monthly_rate > 0, annual_ds, balance / 30.0)
    return annual_ds


# ─── One-hot encoding ─────────────────────────────────────────────────────────


def _apply_onehot(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply one-hot encoding consistently across all splits."""
    all_categories: dict[str, list[str]] = {}

    for col in ONEHOT_FEATURES:
        if col in train.columns:
            all_vals = set()
            for split_df in [train, valid, test]:
                if col in split_df.columns:
                    all_vals.update(split_df[col].dropna().unique())
            all_categories[col] = sorted(str(v) for v in all_vals)

    onehot_cols: list[str] = []
    for col, categories in all_categories.items():
        for cat in categories:
            new_col = f"{col}_{cat}"
            onehot_cols.append(new_col)
            train[new_col] = (train[col].astype(str) == cat).astype(int)
            valid[new_col] = (valid[col].astype(str) == cat).astype(int)
            test[new_col] = (test[col].astype(str) == cat).astype(int)

    return train, valid, test, onehot_cols


# ─── Data loading ─────────────────────────────────────────────────────────────


def _load_delta_as_df(path: str | Path) -> pd.DataFrame:
    """Load a Delta/Parquet/JSON table into a pandas DataFrame."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.delta_writer import DeltaReader

    reader = DeltaReader(path)
    records = reader.read()
    return pd.DataFrame(records)


# ─── Sanity checks ────────────────────────────────────────────────────────────


def _run_sanity_checks(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
) -> None:
    """
    Post-build sanity checks to catch residual leakage or data issues.

    Raises FeatureLeakageError if any check fails critically.
    """
    # Check 1: No split is empty
    for name, X, y in [("train", X_train, y_train), ("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        if len(X) == 0:
            logger.warning(f"Sanity check: {name} split is EMPTY")

    # Check 2: Label distribution is not degenerate (0% or 100%)
    for name, y in [("train", y_train), ("valid", y_valid), ("test", y_test)]:
        if len(y) > 0:
            mean = y.mean()
            if mean == 0.0 or mean == 1.0:
                logger.warning(
                    f"Sanity check: {name} label is degenerate "
                    f"(mean={mean:.3f}). Model cannot learn."
                )

    # Check 3: Top feature correlations with label (catch leakage)
    if len(X_train) > 10 and y_train.nunique() > 1:
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        top5 = correlations.head(5)
        logger.info("Top 5 feature correlations with label:")
        for feat, corr in top5.items():
            logger.info(f"  {feat}: {corr:.4f}")
            if corr > 0.90:
                raise FeatureLeakageError(
                    f"LEAKAGE DETECTED: feature '{feat}' has correlation {corr:.4f} "
                    f"with label. This suggests the feature contains future information."
                )

    # Check 4: No feature names suggest maturity-time or current-state leakage
    forbidden_patterns = ["at_maturity", "maturity_refi", "maturity_dscr", "maturity_ltv"]
    for col in feature_cols:
        for pattern in forbidden_patterns:
            if pattern in col:
                raise FeatureLeakageError(
                    f"LEAKAGE DETECTED: feature '{col}' contains forbidden pattern "
                    f"'{pattern}'. Maturity-time values must not be in feature matrix."
                )
