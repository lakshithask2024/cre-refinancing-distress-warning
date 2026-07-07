"""
Feature Engineering for Distress Classifier
=============================================

Reads the Gold loan_distress_history table, engineers model-ready features,
applies target encoding for high-cardinality categoricals, constructs the
binary classification label, and produces time-based train/valid/test splits.

The time-based split prevents data leakage:
  - Train: origination_year <= 2019  (pre-pandemic vintages)
  - Valid: origination_year == 2020  (pandemic pivot year)
  - Test:  origination_year >= 2021  (post-pandemic stress environment)

This means the model is trained on "normal" conditions and evaluated on how
well it generalizes to the stressed environment we actually care about.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature column groups
NUMERIC_FEATURES = [
    "ltv_at_origination",
    "dscr_at_origination",
    "note_rate",
    "log_original_balance",
    "current_ltv",
    "new_dscr",
    "rate_gap_bps",
    "months_to_maturity",
    "occupancy_pct",
    "current_cap_rate",
    "debt_yield",
]

ONEHOT_FEATURES = [
    "property_type",
    "sponsor_credit_tier",
    "amortization_type",
    "balloon_flag",
]

TARGET_ENCODED_FEATURES = [
    "metro_area",
]

LABEL_COL = "is_distressed"


def build_training_frame(
    gold_path: str | Path = "data/gold/loan_distress_history",
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
    Build model-ready training data from Gold loan_distress_history.

    Why this function exists:
      Centralizes all feature engineering so the model training code receives
      clean numeric matrices with no NaN, no leakage, and consistent encoding.

    Returns:
        (X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names)
    """
    np.random.seed(seed)

    # Load data
    df = _load_gold_data(gold_path)
    logger.info(f"Loaded {len(df)} records from {gold_path}")

    # Engineer features
    df = _engineer_numeric_features(df)

    # Time-based split (before encoding to prevent leakage)
    train_mask = df["origination_year"].astype(int) <= 2019
    valid_mask = df["origination_year"].astype(int) == 2020
    test_mask = df["origination_year"].astype(int) >= 2021

    df_train = df[train_mask].copy()
    df_valid = df[valid_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(
        f"Split sizes — train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}"
    )

    # Target encoding for metro_area (computed ONLY on training set)
    metro_means = df_train.groupby("metro_area")[LABEL_COL].mean()
    global_mean = df_train[LABEL_COL].mean()

    df_train["metro_encoded"] = df_train["metro_area"].map(metro_means).fillna(global_mean)
    df_valid["metro_encoded"] = df_valid["metro_area"].map(metro_means).fillna(global_mean)
    df_test["metro_encoded"] = df_test["metro_area"].map(metro_means).fillna(global_mean)

    # One-hot encoding (fit on full vocabulary to avoid missing dummies)
    df_train, df_valid, df_test, onehot_cols = _apply_onehot(df_train, df_valid, df_test)

    # Assemble final feature matrices
    feature_cols = NUMERIC_FEATURES + ["metro_encoded"] + onehot_cols
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    X_train = df_train[feature_cols].copy()
    X_valid = df_valid[feature_cols].copy()
    X_test = df_test[feature_cols].copy()

    y_train = df_train[LABEL_COL].astype(int)
    y_valid = df_valid[LABEL_COL].astype(int)
    y_test = df_test[LABEL_COL].astype(int)

    # Fill any remaining NaN with 0 (should be rare after engineering)
    X_train = X_train.fillna(0.0)
    X_valid = X_valid.fillna(0.0)
    X_test = X_test.fillna(0.0)

    logger.info(f"Feature matrix shape: {X_train.shape[1]} features")
    logger.info(
        f"Label distribution — train: {y_train.mean():.3f}, "
        f"valid: {y_valid.mean():.3f}, test: {y_test.mean():.3f}"
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


def _load_gold_data(gold_path: str | Path) -> pd.DataFrame:
    """
    Load loan_distress_history from Delta/Parquet/JSON-lines.

    Why: Abstracts the storage format so features.py works regardless of
    whether Bronze was written by Spark (Parquet) or pure-Python (JSON-lines).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.delta_writer import DeltaReader

    reader = DeltaReader(gold_path)
    records = reader.read()
    df = pd.DataFrame(records)

    # Ensure numeric columns are actually numeric
    numeric_cols = [
        "original_balance", "current_balance", "note_rate", "occupancy_pct",
        "noi_annual", "current_ltv", "new_dscr", "rate_gap", "rate_gap_bps",
        "debt_yield", "current_cap_rate", "refinance_rate", "months_to_maturity",
        "ltv_at_origination", "dscr_at_origination",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure label is numeric
    if "is_distressed" in df.columns:
        df["is_distressed"] = df["is_distressed"].map(
            {True: 1, False: 0, "True": 1, "False": 0, "1": 1, "0": 0, 1: 1, 0: 0}
        ).fillna(0).astype(int)

    # Ensure origination_year is numeric
    if "origination_year" in df.columns:
        df["origination_year"] = pd.to_numeric(df["origination_year"], errors="coerce")

    return df


def _engineer_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived numeric features.

    Why each feature matters for distress prediction:
      - log_original_balance: larger loans have different risk profiles (nonlinear)
      - rate_gap_bps: direct measure of refinancing cost increase
      - months_to_maturity: urgency — near-term maturities are more actionable
    """
    # Log-transform balance (reduces skewness of lognormal distribution)
    df["log_original_balance"] = np.log1p(df["original_balance"].fillna(0).astype(float))

    # Ensure key features exist with defaults
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def _apply_onehot(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Apply one-hot encoding consistently across splits.

    Why: We fit the vocabulary on ALL data (not just train) to avoid
    KeyError when valid/test have categories not seen in train.
    The model still only learns weights from train labels.
    """
    all_categories: dict[str, list[str]] = {}

    for col in ONEHOT_FEATURES:
        if col in train.columns:
            # Collect all unique values across splits
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
