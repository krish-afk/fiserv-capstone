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
from src.trading.pipeline import run_configured_trading_pipeline


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
    parser.add_argument(
        "--refresh-market-data",
        action="store_true",
        help="Force refresh of ticker market data before running trading"
    )
    return parser.parse_args()


def run_data_stage():
    """
    Ingest raw sources, clean, and write processed data to disk.
    Only call when data needs to be (re)built — i.e. --stage data or --stage all.
    """
    print("\n[STAGE 1] Data ingestion & processing")
    ingested = run_ingestion()
    raw_dfs = load_all_raw(ingested)
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
    """
    print("\n[STAGE 2] Model experimentation")

    horizon = config["forecasting"]["horizons"][0]
    min_train_size = config["forecasting"]["walk_forward_min_train"]

    panels_data = {}
    for panel_name in config["experiment"]["panels"]:
        y, X, _ = load_panel(panel_name)
        panels_data[panel_name] = (y, X)

    trials = build_trial_grid(config, panels_data)
    return run_experiment(trials, min_train_size, horizon)


def run_trading_stage(refresh_market_data: bool = False):
    """
    Run the configured strategy from config.yaml against the latest
    experiment outputs already written to disk.

    This works after:
      - --stage all
      - --stage experiment
      - or directly with --stage trading, as long as a prior experiment run exists
    """
    print("\n[STAGE 3] Trading strategy & backtest")

    trading_result = run_configured_trading_pipeline(
        cfg=config,
        refresh_market_data=refresh_market_data,
    )

    print(
        "[INFO] Trading complete | "
        f"strategy={trading_result['strategy_name']} | "
        f"tickers={trading_result['tickers']} | "
        f"output_dir={trading_result['output_dir']}"
    )

    results = trading_result.get("results", {})
    if results:
        print(
            "[INFO] Backtest summary | "
            f"return_pct={results.get('return_pct')} | "
            f"sharpe_ratio={results.get('sharpe_ratio')} | "
            f"max_drawdown_pct={results.get('max_drawdown_pct')} | "
            f"final_value={results.get('final_value')}"
        )


def main():
    args = parse_args()

    if args.stage in ("all", "data"):
        run_data_stage()

    if args.stage in ("all", "experiment"):
        y, X, _ = load_panel(args.panel)    # y_level unused until trading stage
        run_experiment_stage(y, X, args.panel)

    if args.stage in ("all", "trading") and not args.skip_trading:
        run_trading_stage(refresh_market_data=args.refresh_market_data)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()