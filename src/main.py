# run_pipeline.py
import argparse
import pandas as pd
from pathlib import Path

from src.utils.config import config
from src.data.features import build_all_features, build_handpicked_v1
from src.data.ingest import run_ingestion
from src.data.load import load_all_raw
from src.data.clean import run_cleaning
from src.data.store import write_master
from src.data.features import build_all_features
from src.data.panel import build_panel
from src.models.baselines import NaiveForecaster, MeanForecaster
from src.models.timeseries import ARIMAForecaster, ETSForecaster
from src.models.ml import RidgeForecaster, XGBoostForecaster
from src.models.experiment import run_experiment
from src.trading.strategy import (
    load_best_forecasts, select_best_model,
    generate_signals, save_signals
)
from src.trading.simulation import run_simulation, compute_performance_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PCE Forecasting Pipeline")
    parser.add_argument(
        "--panel",
        default="national",
        help="Panel name from config.yaml (default: national)"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "experiment", "trading"],
        default="all",
        help="Run a specific pipeline stage or all stages (default: all)"
    )
    parser.add_argument(
        "--skip-trading",
        action="store_true",
        help="Run data + experiment stages only"
    )
    return parser.parse_args()


def run_data_stage(panel_name: str = "national"):
    """Load, process, and return a (y, X) panel."""
    print("\n[STAGE 1] Data loading & preprocessing")

    raw_dfs = load_all_raw()
    master  = run_cleaning(raw_dfs)
    master  = build_all_features(master)
    write_master(master)

    y, X = build_panel(panel_name)
    return y, X


def run_experiment_stage(y: pd.Series,
                         X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define models and feature sets, run all trials."""
    print("\n[STAGE 2] Model experimentation")

    models = [
        NaiveForecaster(horizon=1),
        MeanForecaster(horizon=1),
        ARIMAForecaster(order=(1, 1, 1), horizon=1),
        ETSForecaster(horizon=1),
        RidgeForecaster(alpha=1.0, horizon=1),
        XGBoostForecaster(horizon=1),
        # TODO: add remaining model variants here
    ]

    feature_sets = {
        "none":          None,          # For pure time-series models
        "handpicked_v1": build_handpicked_v1(X),
        "all_features":  X,
        # TODO: add pca_reduced, lasso_selected once implemented
    }

    horizon        = config["forecasting"]["horizons"][0]
    min_train_size = config["forecasting"]["walk_forward_min_train"]

    summary_df = run_experiment(
        models, feature_sets, y, min_train_size, horizon
    )
    return summary_df


def run_trading_stage(summary_df: pd.DataFrame):
    """Load best forecasts, generate signals, run simulation."""
    print("\n[STAGE 3] Trading strategy & simulation")

    forecasts_df = load_best_forecasts()
    best_forecasts = select_best_model(forecasts_df, summary_df)
    signals_df = generate_signals(best_forecasts)
    save_signals(signals_df)

    sim_results = run_simulation(signals_df)
    metrics = compute_performance_metrics(sim_results)

    if metrics:
        print("[INFO] Trading performance:", metrics)
    else:
        print("[INFO] Trading simulation stub complete — "
              "implement simulation logic to see metrics")


def main():
    args = parse_args()

    if args.stage in ("all", "data"):
        y, X = run_data_stage()

    if args.stage in ("all", "experiment"):
        summary_df = run_experiment_stage(y, X)

    if args.stage in ("all", "trading") and not args.skip_trading:
        run_trading_stage(summary_df)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
