# # src/models/evaluate.py
# import pandas as pd
# import numpy as np
# from typing import Dict

# def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
#     """
#     Compute standard forecast error metrics.

#     Args:
#         y_true: Actual PCE values (MoM %, YoY %, or level depending on panel)
#         y_pred: Forecasted PCE values
#     Returns:
#         Dictionary of metric name -> value
#     """
#     errors = y_true - y_pred

#     # MAPE: clip denominator to avoid division by zero on near-zero MoM% values
#     denom = y_true.abs().clip(lower=1e-8)
#     mape = (errors.abs() / denom).mean()

#     # Directional accuracy: fraction of periods where sign of y_pred matches sign of y_true
#     # For MoM% targets this is "did we call the direction of PCE change correctly?"
#     dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())

#     # R²: negative values are valid (model worse than naive mean)
#     ss_res = float((errors ** 2).sum())
#     ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
#     r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

#     return {
#         "rmse":    float(np.sqrt((errors ** 2).mean())),
#         "mae":     float(np.abs(errors).mean()),
#         "me":      float(errors.mean()),
#         "mape":    float(mape),
#         "dir_acc": dir_acc,
#         "r2":      r2,
#     }

# def walk_forward_evaluate(
#     model,
#     y: pd.Series,
#     X: pd.DataFrame = None,
#     min_train_size: int = 36,
#     horizon: int = 1,
# ) -> pd.DataFrame:
#     """
#     Run walk-forward (expanding window) evaluation for a BaseForecaster.

#     Args:
#         model: Instance of BaseForecaster
#         y: Full target series, date-indexed
#         X: Optional exogenous features, same index as y
#         min_train_size: Minimum number of observations before first forecast
#         horizon: Forecast horizon (steps ahead)
#     Returns:
#         DataFrame with columns [date, y_true, y_pred]
#     """
#     results = []

#     # TODO: Add multi-step support — currently only handles horizon=1
#     for t in range(min_train_size, len(y) - horizon + 1):
#         y_train = y.iloc[:t]
#         y_true  = y.iloc[t: t + horizon]

#         X_train = X.iloc[:t] if X is not None else None
#         X_test  = X.iloc[t: t + horizon] if X is not None else None

#         y_pred = model.fit_predict(y_train, X_train, X_test)

#         results.append({
#             "date":   y_true.index[-1],
#             "y_true": y_true.iloc[-1],
#             "y_pred": float(y_pred.iloc[-1]),
#         })

#     return pd.DataFrame(results).set_index("date")


# src/models/evaluate.py

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prev: Optional[pd.Series] = None,
    mape_eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute forecast metrics.

    Directional accuracy definition:
        Did the forecast correctly predict whether the target moves up or down
        from the previous known actual value?

        actual movement    = sign(y_true_t - y_prev_t)
        predicted movement = sign(y_pred_t - y_prev_t)

    MAPE convention:
        mape is stored as percentage points.
        Example: 0.774 means 0.774%, 77.4 means 77.4%.

    mape_ratio is also saved for debugging/backward compatibility.
    """
    df = pd.DataFrame({
        "y_true": pd.Series(y_true, dtype="float64"),
        "y_pred": pd.Series(y_pred, dtype="float64"),
    })

    if y_prev is not None:
        df["y_prev"] = pd.Series(y_prev, dtype="float64")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["y_true", "y_pred"])

    errors = df["y_true"] - df["y_pred"]

    # MAPE: drop near-zero actuals instead of clipping them.
    # Clipping can make MoM targets look absurd when y_true is close to 0.
    denom = df["y_true"].abs()
    mape_mask = denom > mape_eps

    if mape_mask.any():
        mape_ratio = float((errors.abs()[mape_mask] / denom[mape_mask]).mean())
        mape_pct = mape_ratio * 100.0
    else:
        mape_ratio = float("nan")
        mape_pct = float("nan")

    # Directional accuracy: up/down movement from previous known actual.
    if "y_prev" in df.columns:
        dir_df = df.dropna(subset=["y_prev"]).copy()

        actual_move = np.sign(dir_df["y_true"] - dir_df["y_prev"])
        pred_move = np.sign(dir_df["y_pred"] - dir_df["y_prev"])

    else:
        # Fallback for older callers: compare consecutive movements.
        # This is less ideal than using y_prev from the forecast origin.
        actual_move = np.sign(df["y_true"].diff())
        pred_move = np.sign(df["y_pred"].diff())

    dir_mask = actual_move.notna() & pred_move.notna()

    # Exclude true no-movement periods because they are neither up nor down.
    dir_mask = dir_mask & (actual_move != 0)

    if dir_mask.any():
        dir_acc = float((actual_move[dir_mask] == pred_move[dir_mask]).mean())
    else:
        dir_acc = float("nan")

    # R²: negative values are valid out-of-sample.
    ss_res = float((errors ** 2).sum())
    ss_tot = float(((df["y_true"] - df["y_true"].mean()) ** 2).sum())
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "mae": float(errors.abs().mean()),
        "me": float(errors.mean()),

        # Store MAPE as percentage points.
        "mape": float(mape_pct),

        # Keep raw ratio for audit/debugging.
        "mape_ratio": float(mape_ratio),

        # Keep dir_acc as ratio: 0.9565 means 95.65%.
        "dir_acc": dir_acc,

        "r2": r2,
    }


def walk_forward_evaluate(
    model,
    y: pd.Series,
    X: pd.DataFrame = None,
    min_train_size: int = 36,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Run walk-forward expanding-window evaluation.

    Returns:
        DataFrame with columns:
            date, y_true, y_pred, y_prev

    y_prev is the previous known actual at the forecast origin.
    It is needed to compute true directional accuracy:
        sign(y_true_t - y_prev_t) vs sign(y_pred_t - y_prev_t)
    """
    results = []

    for t in range(min_train_size, len(y) - horizon + 1):
        y_train = y.iloc[:t]
        y_true = y.iloc[t: t + horizon]

        X_train = X.iloc[:t] if X is not None else None
        X_test = X.iloc[t: t + horizon] if X is not None else None

        y_pred = model.fit_predict(y_train, X_train, X_test)

        # Previous actual known at the time the forecast is made.
        y_prev = y.iloc[t - 1] if t > 0 else np.nan

        results.append({
            "date": y_true.index[-1],
            "y_true": float(y_true.iloc[-1]),
            "y_pred": float(y_pred.iloc[-1]),
            "y_prev": float(y_prev) if pd.notna(y_prev) else np.nan,
        })

    return pd.DataFrame(results).set_index("date")