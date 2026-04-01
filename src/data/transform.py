# src/data/transform.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.config import config

_VALID_TRANSFORMS = {"mom", "yoy", None}


def transform_target(y: pd.Series, method: str) -> pd.Series:
    """
    Apply a growth-rate transformation to the target series.

    Args:
        y: Raw target level series (e.g. PCE in billions of dollars)
        method: "mom" (month-over-month %), "yoy" (year-over-year %),
                or None (return y unchanged)
    Returns:
        Transformed series with leading NaN rows from pct_change dropped:
          - None: unchanged, same length as input
          - "mom": 1 leading row dropped
          - "yoy": 12 leading rows dropped
        The caller (panel.py) is responsible for aligning X to match.
    """
    if method not in _VALID_TRANSFORMS:
        raise ValueError(
            f"Unknown target_transform '{method}'. "
            f"Valid options: {_VALID_TRANSFORMS}"
        )
    if method is None:
        return y
    periods = 1 if method == "mom" else 12
    return y.pct_change(periods).dropna()


def inverse_transform_target(
    y_pred: pd.Series,
    y_lag: pd.Series,
    method: str,
) -> pd.Series:
    """
    Reconstruct predicted levels from predicted growth rates.
    Use this in the evaluation stage to compute level-space error metrics.

    Args:
        y_pred: Predicted growth rates aligned to the forecast index
                (as decimals, e.g. 0.003 not 0.3%)
        y_lag:  Lagged actual levels aligned to y_pred's index.
                For "mom": the actual level shifted by 1 period.
                For "yoy": the actual level shifted by 12 periods.
                Available from the y_level return value of panel.build_panel().
        method: The same method passed to transform_target()
    Returns:
        Predicted levels on the same index as y_pred
    """
    if method is None:
        return y_pred
    if method not in {"mom", "yoy"}:
        raise ValueError(f"Unknown target_transform '{method}'")
    return y_lag * (1 + y_pred)


def difference_series(df: pd.DataFrame, order: int = 1) -> pd.DataFrame:
    """
    Apply differencing to all numeric columns.

    Args:
        df: Input DataFrame
        order: Differencing order (1 = first difference, 0 = no change)
    Returns:
        Differenced DataFrame with NaN rows from differencing dropped

    NOTE: differencing must be applied AFTER train/test split in
    walk-forward evaluation to avoid lookahead bias. This function
    is for standalone use; the walk-forward loop handles it internally.
    TODO: enforce this constraint or add a warning
    """
    if order == 0:
        return df
    return df.diff(order).dropna()


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   method: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit scaler on training data only, apply to both train and test.
    Always fit on train only — never on the full dataset.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        method: "standard", "minmax", or None (pass-through)
    Returns:
        Tuple of (scaled X_train, scaled X_test)
    """
    method = method or config["transforms"]["scaling"]

    if method is None:
        return X_train, X_test

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    return X_train_scaled, X_test_scaled
