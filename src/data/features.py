# src/data/features.py
import pandas as pd
from typing import List
from src.utils.config import config


def build_lag_features(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    """
    Add lag columns for a given variable.

    Args:
        df: Input DataFrame, date-indexed
        col: Column name to lag
        lags: List of lag periods (e.g. [1, 3, 6, 12])
    Returns:
        DataFrame with new lag columns appended
    """
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def build_rolling_features(df: pd.DataFrame, col: str,
                            windows: List[int]) -> pd.DataFrame:
    """
    Add rolling mean and std columns for a given variable.

    Args:
        df: Input DataFrame, date-indexed
        col: Column to compute rolling stats on
        windows: List of window sizes (e.g. [3, 6, 12])
    Returns:
        DataFrame with rolling mean/std columns appended
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_rollmean{w}"] = df[col].shift(1).rolling(w).mean()
        df[f"{col}_rollstd{w}"]  = df[col].shift(1).rolling(w).std()
        # NOTE: shift(1) before rolling prevents data leakage —
        # the rolling window must not include the current period's value
    return df


def build_growth_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add month-over-month and year-over-year growth rate columns."""
    df = df.copy()
    df[f"{col}_mom"] = df[col].pct_change(1)
    df[f"{col}_yoy"] = df[col].pct_change(12)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature builder — applies all transformations and returns the full
    feature matrix. panel.py selects named subsets or passes the full matrix
    to models that do their own selection.

    Column treatment:
      - Target columns (pce, mrts): excluded entirely — never lagged or rolled.
        Which target is used in a given experiment is set per-panel in config.
      - FSBI columns (fsbi_*): lags + rolling stats only. FSBI already contains
        native MoM/YoY % columns, so growth rates would be redundant.
      - All other columns (FRED macro, USCB): lags + rolling stats + growth rates.

    Lags and windows are read from config — no hardcoded values here.

    Args:
        df: Master DataFrame from store.read_master() — wide format, date-indexed.
    Returns:
        Full feature matrix with engineered columns appended; NaN rows dropped.
    """
    targets = config["data"]["targets"]   # list of all potential target column names
    lags    = config["features"]["lags"]
    windows = config["features"]["rolling_windows"]

    fsbi_cols = [c for c in df.columns if c.startswith("fsbi_")]
    # Non-target, non-FSBI columns (FRED macro, USCB, etc.)
    other_cols = [
        c for c in df.columns
        if c not in targets and not c.startswith("fsbi_")
    ]

    # FSBI features — lags and rolling only
    for col in fsbi_cols:
        df = build_lag_features(df, col=col, lags=lags)
        df = build_rolling_features(df, col=col, windows=windows)

    # Macro / other features — lags, rolling, and growth rates
    for col in other_cols:
        df = build_lag_features(df, col=col, lags=lags)
        df = build_rolling_features(df, col=col, windows=windows)
        df = build_growth_rate(df, col=col)

    df = df.dropna()
    return df


def build_named_subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Return the subset of the feature matrix defined by a named feature set
    in config.yaml under features.feature_sets.<name>.

    This is the data-layer hook for config-driven feature selection. Models
    that instead do algorithmic selection (LASSO, greedy, PCA, etc.) receive
    the full matrix from build_all_features() and perform selection internally
    within the experiment loop — that logic does NOT belong here.

    Args:
        df: Full feature matrix from build_all_features()
        name: Key in config["features"]["feature_sets"]
    Returns:
        DataFrame containing only the columns in the named set
    Raises:
        KeyError: if the name is not defined in config
        ValueError: if any configured column is absent from df
    """
    feature_sets = config["features"].get("feature_sets", {})
    if name not in feature_sets:
        raise KeyError(
            f"Feature set '{name}' not found in config. "
            f"Available sets: {list(feature_sets.keys())}"
        )
    cols = feature_sets[name]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature set '{name}' references columns not in the feature matrix: {missing}"
        )
    return df[cols].copy()
