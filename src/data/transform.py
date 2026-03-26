# src/data/transform.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.config import config


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
