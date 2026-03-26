# src/models/evaluate.py
import pandas as pd
import numpy as np
from typing import Dict

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute standard forecast error metrics.

    Args:
        y_true: Actual PCE values
        y_pred: Forecasted PCE values
    Returns:
        Dictionary of metric name -> value
    """
    # TODO: Add MAPE, but handle division-by-zero if PCE values near 0
    errors = y_true - y_pred
    return {
        "rmse": np.sqrt((errors ** 2).mean()),
        "mae": np.abs(errors).mean(),
        "me":  errors.mean(),  # Mean error — useful for bias detection
    }

def walk_forward_evaluate(
    model,
    y: pd.Series,
    X: pd.DataFrame = None,
    min_train_size: int = 36,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Run walk-forward (expanding window) evaluation for a BaseForecaster.

    Args:
        model: Instance of BaseForecaster
        y: Full target series, date-indexed
        X: Optional exogenous features, same index as y
        min_train_size: Minimum number of observations before first forecast
        horizon: Forecast horizon (steps ahead)
    Returns:
        DataFrame with columns [date, y_true, y_pred]
    """
    results = []

    # TODO: Add multi-step support — currently only handles horizon=1
    for t in range(min_train_size, len(y) - horizon + 1):
        y_train = y.iloc[:t]
        y_true  = y.iloc[t: t + horizon]

        X_train = X.iloc[:t] if X is not None else None
        X_test  = X.iloc[t: t + horizon] if X is not None else None

        y_pred = model.fit_predict(y_train, X_train, X_test)

        results.append({
            "date":   y_true.index[-1],
            "y_true": y_true.iloc[-1],
            "y_pred": float(y_pred.iloc[-1]),
        })

    return pd.DataFrame(results).set_index("date")
