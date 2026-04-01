# run_pipeline.py
import argparse
import pandas as pd
from pathlib import Path

from src.utils.config import config
from src.data.ingest import run_ingestion
from src.data.load import load_all_raw
from src.data.clean import run_cleaning
from src.data.store import write_master, write_fsbi
from src.data.panel import build_panel
from src.models.experiment import build_trial_grid, run_experiment
# from src.trading.strategy import (
#     load_best_forecasts, select_best_model,
#     generate_signals, save_signals
# )
# from src.trading.simulation import run_simulation, compute_performance_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PCE Forecasting Pipeline")
    parser.add_argument(
        "--panel",
        default="pce_national_mom",
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


def run_data_stage():
    """
    Ingest raw sources, clean, and write processed data to disk.
    Only call when data needs to be (re)built — i.e. --stage data or --stage all.
    """
    print("\n[STAGE 1] Data ingestion & processing")
    ingested          = run_ingestion()
    raw_dfs           = load_all_raw(ingested)
    master, fsbi_long = run_cleaning(raw_dfs)
    write_master(master)
    write_fsbi(fsbi_long)


def load_panel(panel_name: str) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Build (y, X, y_level) for the named panel by reading processed data from disk.
    Raises FileNotFoundError (via build_panel) if data/processed/ is empty —
    run --stage data first to populate it.

    Called unconditionally before the experiment stage, regardless of whether
    --stage data was also invoked in this run.
    """
    print(f"\n[PANEL] Building panel '{panel_name}' from disk")
    return build_panel(panel_name)


def run_experiment_stage(
    y: pd.Series,
    X: pd.DataFrame,
    panel_name: str,
) -> pd.DataFrame:
    """
    Config-driven experiment stage. Reads the full trial grid from
    config.yaml (experiment.models, features.feature_sets) and runs
    walk-forward evaluation across all combinations.

    To add a model variant, hyperparameter setting, feature set, or panel:
    edit config.yaml only — no code changes needed.

    To run multiple panels in one call, build a panels_data dict in main()
    and pass it directly to build_trial_grid() + run_experiment().
    """
    print("\n[STAGE 2] Model experimentation")

    horizon        = config["forecasting"]["horizons"][0]
    min_train_size = config["forecasting"]["walk_forward_min_train"]

    panels_data = {panel_name: (y, X)}
    trials      = build_trial_grid(config, panels_data)

    return run_experiment(trials, min_train_size, horizon)


# def run_trading_stage(summary_df: pd.DataFrame):
#     """Load best forecasts, generate signals, run simulation."""
#     print("\n[STAGE 3] Trading strategy & simulation")

#     forecasts_df = load_best_forecasts()
#     best_forecasts = select_best_model(forecasts_df, summary_df)
#     signals_df = generate_signals(best_forecasts)
#     save_signals(signals_df)

#     sim_results = run_simulation(signals_df)
#     metrics = compute_performance_metrics(sim_results)

#     if metrics:
#         print("[INFO] Trading performance:", metrics)
#     else:
#         print("[INFO] Trading simulation stub complete — "
#               "implement simulation logic to see metrics")


def main():
    args = parse_args()

    if args.stage in ("all", "data"):
        run_data_stage()

    if args.stage in ("all", "experiment"):
        y, X, _ = load_panel(args.panel)    # y_level unused until trading stage
        summary_df = run_experiment_stage(y, X, args.panel)

    # if args.stage in ("all", "trading") and not args.skip_trading:
    #     run_trading_stage(summary_df)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
