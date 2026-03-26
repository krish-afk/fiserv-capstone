# src/trading/strategy.py
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import config


def load_best_forecasts(experiments_dir: Path = None) -> pd.DataFrame:
    """
    Load forecast CSV from the most recent experiment run.

    Returns:
        DataFrame with columns [date, y_true, y_pred, model_name, feature_set]
    """
    if experiments_dir is None:
        experiments_dir = Path(config["paths"]["experiments"])

    # Find most recent run directory by timestamp prefix
    run_dirs = sorted(experiments_dir.glob("*_run"))
    if not run_dirs:
        raise FileNotFoundError(
            f"No experiment runs found in {experiments_dir}"
        )

    latest = run_dirs[-1]
    forecasts_path = latest / "forecasts.csv"
    print(f"[INFO] Loading forecasts from {latest.name}")
    return pd.read_csv(forecasts_path, parse_dates=["date"])


def select_best_model(forecasts_df: pd.DataFrame,
                      metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter forecast DataFrame to rows from the best-performing model.

    Args:
        forecasts_df: Full forecasts from all trials
        metrics_df: Summary metrics DataFrame from run_experiment()
    Returns:
        Forecasts from the single best model/feature_set combination
    """
    # TODO: replace with config-driven selection once team agrees on
    # final model (config["forecasting"]["selected_model"])
    best_row = metrics_df.dropna(subset=["rmse"]).iloc[0]
    best_model   = best_row["model_name"]
    best_fs      = best_row["feature_set"]

    print(f"[INFO] Selected model: {best_model} | features: {best_fs}")

    mask = (
        (forecasts_df["model_name"]  == best_model) &
        (forecasts_df["feature_set"] == best_fs)
    )
    return forecasts_df[mask].copy()


def generate_signals(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate directional trading signals from PCE forecasts.

    Signal logic (stub):
        +1 (long)  if forecast > last actual  → PCE expected to rise
        -1 (short) if forecast < last actual  → PCE expected to fall
         0          if forecast == last actual

    Args:
        forecasts_df: Forecasts from best model, date-indexed
    Returns:
        DataFrame with columns [date, y_true, y_pred, signal]
    """
    df = forecasts_df.copy().sort_values("date")
    df["signal"] = np.sign(df["y_pred"] - df["y_true"].shift(1))

    # TODO: replace stub signal logic with real strategy, e.g.:
    # - threshold-based signals (only trade if delta > X%)
    # - probabilistic signals using forecast confidence intervals
    # - sector rotation rules based on PCE component breakdowns

    return df


def save_signals(signals_df: pd.DataFrame, output_path: Path = None):
    """Write signals to the forecasts output directory."""
    if output_path is None:
        output_path = Path(config["paths"]["forecasts"]) / "signals.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_csv(output_path, index=False)
    print(f"[INFO] Signals written to {output_path}")
