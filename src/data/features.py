# src/data/features.py
import pandas as pd
import numpy as np
from typing import List, Optional
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
    # TODO: consider whether to drop NaN rows here or leave to caller
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


def build_handpicked_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    First handpicked feature set.
    TODO: define which columns to select based on EDA findings
    """
    cols = []  # TODO: populate with selected column names
    return df[cols].copy()


def build_handpicked_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second handpicked feature set.
    TODO: define alternative column selection
    """
    cols = []  # TODO: populate with selected column names
    return df[cols].copy()


def build_pca_features(df: pd.DataFrame,
                        n_components: int = 5) -> pd.DataFrame:
    """
    Reduce feature matrix to n_components via PCA.

    IMPORTANT: PCA must be fit only on training data in walk-forward
    evaluation to avoid lookahead bias. This function is for building
    the static feature matrix only — fit/transform happens in the
    experiment loop.
    TODO: implement fit/transform split for walk-forward compliance
    """
    # TODO: implement using sklearn.decomposition.PCA
    pass


def build_lasso_selected(df: pd.DataFrame, y: pd.Series,
                          alpha: float = 0.01) -> pd.DataFrame:
    """
    Return feature subset surviving LassoCV selection.
    Same lookahead caveat as PCA — fit only on training data.
    TODO: implement walk-forward compliant version
    """
    # TODO: implement using sklearn.linear_model.LassoCV
    pass


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature builder — applies all transformations and
    returns the full feature matrix. Individual feature sets
    are then selected as subsets of this.

    Args:
        df: Preprocessed DataFrame containing PCE, FSBI, and
            any other macroeconomic columns
    Returns:
        Full feature matrix, date-indexed, NaN rows dropped
    """
    target = config["data"]["target"]

    df = build_lag_features(df, col=target, lags=[1, 3, 6, 12])
    df = build_rolling_features(df, col=target, windows=[3, 6, 12])
    df = build_growth_rate(df, col=target)

    # TODO: apply same transformations to FSBI columns
    # TODO: add any interaction terms identified during EDA

    df = df.dropna()
    return df
