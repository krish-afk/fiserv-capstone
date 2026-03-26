# src/models/experiment.py
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import List, Dict, Optional

from src.models.base import BaseForecaster
from src.models.evaluate import walk_forward_evaluate, compute_metrics
from src.utils.config import config


def _make_experiment_dir() -> Path:
    """Create a timestamped directory for this experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config["paths"]["experiments"]) / f"{timestamp}_run"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    return exp_dir


def run_experiment(
    models: List[BaseForecaster],
    feature_sets: Dict[str, Optional[pd.DataFrame]],
    y: pd.Series,
    min_train_size: int = 36,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Run walk-forward evaluation across all (model, feature_set) combinations,
    write results to a timestamped experiments/ directory, and return a
    summary DataFrame sorted by RMSE.

    Args:
        models: List of instantiated BaseForecaster models
        feature_sets: Dict of {name: DataFrame or None}
                      Use None for pure time-series models
        y: Target PCE series, date-indexed
        min_train_size: Minimum training observations before first forecast
        horizon: Forecast horizon
    Returns:
        Summary DataFrame with columns [model_name, feature_set, rmse, mae, me]
    """
    exp_dir = _make_experiment_dir()
    summary_rows = []
    all_forecasts = []

    for model, (fs_name, X) in product(models, feature_sets.items()):

        # Skip incompatible combinations cleanly
        # e.g. pure time-series models paired with a non-None feature set
        # TODO: implement a model.requires_X flag on BaseForecaster
        # to make this check systematic rather than relying on assertions

        try:
            preds = walk_forward_evaluate(
                model, y, X, min_train_size, horizon
            )
            metrics = compute_metrics(preds["y_true"], preds["y_pred"])

            summary_rows.append({
                "model_name":  model.name,
                "feature_set": fs_name,
                **metrics,
            })

            # Tag forecast rows for unified output file
            preds["model_name"]  = model.name
            preds["feature_set"] = fs_name
            all_forecasts.append(preds.reset_index())

        except Exception as e:
            # Log failures without aborting the full experiment
            print(f"[WARN] Trial failed — model={model.name}, "
                  f"features={fs_name}: {e}")
            summary_rows.append({
                "model_name":  model.name,
                "feature_set": fs_name,
                "rmse": None, "mae": None, "me": None,
                "error": str(e),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values("rmse")

    # --- Write outputs ---
    summary_df.to_csv(exp_dir / "metrics.csv", index=False)

    if all_forecasts:
        pd.concat(all_forecasts).to_csv(
            exp_dir / "forecasts.csv", index=False
        )

    # Write a human-readable experiment metadata file
    metadata = {
        "timestamp":      datetime.now().isoformat(),
        "horizon":        horizon,
        "min_train_size": min_train_size,
        "models":         [m.name for m in models],
        "feature_sets":   list(feature_sets.keys()),
        "n_trials":       len(summary_rows),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Experiment complete. Results written to {exp_dir}")
    print(summary_df[["model_name", "feature_set", "rmse", "mae"]].to_string())

    return summary_df
