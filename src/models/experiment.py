# src/models/experiment.py
import fnmatch
import json
import re
import pandas as pd
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass

from src.models.base import BaseForecaster
from src.models.baselines import NaiveForecaster, MeanForecaster
from src.models.timeseries import (
    ARIMAForecaster,
    ARIMAXForecaster,
    ETSForecaster,
    ThetaForecaster,
    ETSXForecaster,
)
from src.models.ml import (
    OLSForecaster,
    RidgeForecaster,
    LassoForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
    GradientBoostingForecaster,
)
from src.models.evaluate import walk_forward_evaluate, compute_metrics
from src.utils.config import config

# ---------------------------------------------------------------------------
# Model registry
# Map config class names → Python classes. To add a new model: implement it,
# import it above, add it here. No other code change needed.
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, type] = {
    "NaiveForecaster":            NaiveForecaster,
    "MeanForecaster":             MeanForecaster,
    "ARIMAForecaster":            ARIMAForecaster,
    "ARIMAXForecaster":           ARIMAXForecaster,
    "ETSForecaster":              ETSForecaster,
    "ThetaForecaster":            ThetaForecaster,
    "ETSXForecaster":             ETSXForecaster,
    "OLSForecaster":              OLSForecaster,
    "RidgeForecaster":            RidgeForecaster,
    "LassoForecaster":            LassoForecaster,
    "RandomForestForecaster":     RandomForestForecaster,
    "XGBoostForecaster":          XGBoostForecaster,
    "GradientBoostingForecaster": GradientBoostingForecaster,
}


class Trial(NamedTuple):
    """A single experiment trial: one model instance, one feature set, one panel."""
    model:            BaseForecaster
    model_label:      str                    # Name written to outputs
    feature_set_name: str
    X:                Optional[pd.DataFrame]
    y:                pd.Series
    panel_name:       str

@dataclass
class ExperimentRunResult:
    run_dir: Path
    metrics_df: pd.DataFrame
    forecasts_df: pd.DataFrame
    metadata: dict

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_feature_set(
    X: pd.DataFrame,
    patterns: List[str],
) -> Optional[pd.DataFrame]:
    """
    Slice X columns by a list of fnmatch glob patterns.

    Semantics:
      []     → None  (pure time-series: no feature matrix)
      ["*"]  → X unchanged
      other  → keep columns matching any pattern
    """
    if not patterns:
        return None
    if patterns == ["*"]:
        return X
    matched = [col for col in X.columns
               if any(fnmatch.fnmatch(col, p) for p in patterns)]
    if not matched:
        raise ValueError(
            f"Feature set patterns {patterns!r} matched no columns. "
            f"Available columns (first 10): {list(X.columns[:10])}"
        )
    return X[matched]


def _expand_param_grid(param_grid: Dict) -> List[Dict]:
    """Expand {kwarg: [values]} into a list of all (kwarg, value) combinations."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    combos = product(*[param_grid[k] for k in keys])
    return [dict(zip(keys, vals)) for vals in combos]


def _params_label(params: Dict) -> str:
    """Serialize {alpha: 0.01, max_depth: 3} → 'alpha0.01_max_depth3'."""
    def fmt(v):
        if isinstance(v, (list, tuple)):
            return "".join(str(x) for x in v)
        return "none" if v is None else str(v)
    return "_".join(f"{k}{fmt(v)}" for k, v in sorted(params.items()))


def _make_experiment_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config["paths"]["experiments"]) / f"{timestamp}_run"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "artifacts").mkdir(exist_ok=True)
    return exp_dir


def _artifact_filename(trial: "Trial") -> str:
    """Sanitized filename for a trial's artifact JSON."""
    raw = f"{trial.panel_name}__{trial.model_label}__{trial.feature_set_name}"
    return re.sub(r"[^\w\-]", "_", raw)[:180] + ".json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_trial_grid(
    cfg: dict,
    panels_data: Dict[str, Tuple[pd.Series, pd.DataFrame]],
) -> List[Trial]:
    """
    Construct the full experiment grid from config.

    For each panel × model variant × param combination × feature set, creates
    one Trial with a fresh model instance. No model instance is shared across
    trials.

    Args:
        cfg:          Loaded config dict (pass the global `config` object).
        panels_data:  {panel_name: (y, X)} built by the data stage.

    Returns:
        Ordered list of Trial objects, ready to pass to run_experiment().
    """
    exp_cfg  = cfg["experiment"]
    fs_defs  = cfg["features"]["feature_sets"]   # {name: [patterns]}
    horizon  = cfg["forecasting"]["horizons"][0]
    trials: List[Trial] = []

    for panel_name in exp_cfg["panels"]:
        if panel_name not in panels_data:
            print(f"[WARN] Panel '{panel_name}' not in panels_data — skipping")
            continue
        y, X_full = panels_data[panel_name]

        # Pre-build the column-filtered X for each named feature set
        X_by_fs: Dict[str, Optional[pd.DataFrame]] = {}
        for fs_name, patterns in fs_defs.items():
            try:
                X_by_fs[fs_name] = _apply_feature_set(X_full, patterns)
            except ValueError as e:
                print(f"[WARN] Panel '{panel_name}', feature set '{fs_name}': {e}")

        for family_cfg in exp_cfg["models"].values():
            if not family_cfg.get("enabled", True):
                continue

            for variant in family_cfg["variants"]:
                cls_name = variant["class"]
                if cls_name not in _MODEL_REGISTRY:
                    print(f"[WARN] Unknown model class '{cls_name}' — "
                          f"add it to _MODEL_REGISTRY in experiment.py")
                    continue
                cls = _MODEL_REGISTRY[cls_name]

                has_param_grid = "param_grid" in variant
                param_combos   = (_expand_param_grid(variant["param_grid"])
                                  if has_param_grid
                                  else [variant.get("params") or {}])

                variant_fs_names = variant.get("feature_sets", list(X_by_fs.keys()))

                for params in param_combos:
                    # Derive the display label for this (class, params) combo.
                    # Each model's .name property encodes its own params (e.g.
                    # "ridge_alpha=0.01", "xgboost_learning_rate=0.01_max_depth=2").
                    # Config 'name' overrides auto-generation for single-params variants.
                    if not has_param_grid and "name" in variant:
                        label = variant["name"]
                    else:
                        _tmp  = cls(horizon=horizon, **params)
                        label = _tmp.name

                    for fs_name in variant_fs_names:
                        if fs_name not in X_by_fs:
                            print(f"[WARN] Feature set '{fs_name}' not defined in "
                                  f"features.feature_sets — skipping")
                            continue
                        # Fresh model instance per trial — never share state
                        trials.append(Trial(
                            model=cls(horizon=horizon, **params),
                            model_label=label,
                            feature_set_name=fs_name,
                            X=X_by_fs[fs_name],
                            y=y,
                            panel_name=panel_name,
                        ))

    return trials


def _execute_experiment(
    trials: List[Trial],
    min_train_size: int = 36,
    horizon: int = 1,
) -> ExperimentRunResult:
    """
    Core experiment execution used by both:
      - run_experiment(...)               -> backward-compatible summary DataFrame
      - run_experiment_with_details(...)  -> rich structured result for dashboard/API
    """
    exp_dir = _make_experiment_dir()
    summary_rows = []
    all_forecasts = []

    for i, trial in enumerate(trials, start=1):
        print(f"[INFO] Running Experiment - Trial {i} / {len(trials)}")
        try:
            preds = walk_forward_evaluate(
                trial.model,
                trial.y,
                trial.X,
                min_train_size,
                horizon,
            )
            metrics = compute_metrics(preds["y_true"], preds["y_pred"])

            summary_rows.append({
                "panel_name": trial.panel_name,
                "model_name": trial.model_label,
                "feature_set": trial.feature_set_name,
                **metrics,
            })

            artifacts = trial.model.summarize_artifacts()
            if artifacts:
                artifact_path = exp_dir / "artifacts" / _artifact_filename(trial)
                artifact_path.write_text(json.dumps(artifacts, indent=2, default=float))

            preds = preds.reset_index()
            preds["model_name"] = trial.model_label
            preds["feature_set"] = trial.feature_set_name
            preds["panel_name"] = trial.panel_name
            all_forecasts.append(preds)

        except Exception as e:
            print(
                f"[WARN] Trial failed — panel={trial.panel_name}, "
                f"model={trial.model_label}, features={trial.feature_set_name}: {e}"
            )
            summary_rows.append({
                "panel_name": trial.panel_name,
                "model_name": trial.model_label,
                "feature_set": trial.feature_set_name,
                "mae": None,
                "rmse": None,
                "me": None,
                "mape": None,
                "dir_acc": None,
                "r2": None,
                "error": str(e),
            })

    metrics_df = pd.DataFrame(summary_rows).sort_values("mae", na_position="last")

    if all_forecasts:
        forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    else:
        forecasts_df = pd.DataFrame(
            columns=["date", "y_true", "y_pred", "model_name", "feature_set", "panel_name"]
        )

    metrics_df.to_csv(exp_dir / "metrics.csv", index=False)
    forecasts_df.to_csv(exp_dir / "forecasts.csv", index=False)

    n_artifacts = sum(
        1 for t in trials if (exp_dir / "artifacts" / _artifact_filename(t)).exists()
    )

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "horizon": horizon,
        "min_train_size": min_train_size,
        "panels": sorted({t.panel_name for t in trials}),
        "models": sorted({t.model_label for t in trials}),
        "feature_sets": sorted({t.feature_set_name for t in trials}),
        "n_trials": len(trials),
        "n_artifacts": n_artifacts,
        "run_dir": str(exp_dir),
    }

    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Experiment complete — {len(trials)} trials. Results written to {exp_dir}")

    return ExperimentRunResult(
        run_dir=exp_dir,
        metrics_df=metrics_df,
        forecasts_df=forecasts_df,
        metadata=metadata,
    )


def run_experiment(
    trials: List[Trial],
    min_train_size: int = 36,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Backward-compatible API.
    Returns the metrics summary DataFrame exactly like before.
    """
    result = _execute_experiment(
        trials=trials,
        min_train_size=min_train_size,
        horizon=horizon,
    )
    return result.metrics_df


def run_experiment_with_details(
    trials: List[Trial],
    min_train_size: int = 36,
    horizon: int = 1,
) -> ExperimentRunResult:
    """
    Dashboard/API-friendly API.
    Returns metrics + forecasts + metadata + run_dir.
    """
    return _execute_experiment(
        trials=trials,
        min_train_size=min_train_size,
        horizon=horizon,
    )