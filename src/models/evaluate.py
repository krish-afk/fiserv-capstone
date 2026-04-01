# src/models/evaluate.py
import pandas as pd
import numpy as np
from typing import Dict

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute standard forecast error metrics.

    Args:
        y_true: Actual PCE values (MoM %, YoY %, or level depending on panel)
        y_pred: Forecasted PCE values
    Returns:
        Dictionary of metric name -> value
    """
    errors = y_true - y_pred

    # MAPE: clip denominator to avoid division by zero on near-zero MoM% values
    denom = y_true.abs().clip(lower=1e-8)
    mape = (errors.abs() / denom).mean()

    # Directional accuracy: fraction of periods where sign of y_pred matches sign of y_true
    # For MoM% targets this is "did we call the direction of PCE change correctly?"
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())

    # R²: negative values are valid (model worse than naive mean)
    ss_res = float((errors ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "rmse":    float(np.sqrt((errors ** 2).mean())),
        "mae":     float(np.abs(errors).mean()),
        "me":      float(errors.mean()),
        "mape":    float(mape),
        "dir_acc": dir_acc,
        "r2":      r2,
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
