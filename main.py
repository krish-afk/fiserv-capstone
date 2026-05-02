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
        default=None,
        help="Panel name from config.yaml (default: national)"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "experiment", "trading", "plot"],
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


def run_experiment_stage(panel_name: str | None = None) -> pd.DataFrame:
    """
    Run model experimentation.

    If panel_name is provided, run only that panel.
    If panel_name is None, run all panels listed in config["experiment"]["panels"].
    """
    print("\n[STAGE 2] Model experimentation")

    horizon = config["forecasting"]["horizons"][0]
    min_train_size = config["forecasting"]["walk_forward_min_train"]

    if panel_name is None:
        panels_to_run = list(config["experiment"]["panels"])
    else:
        panels_to_run = [panel_name]

    print(f"[INFO] Experiment panels to run: {panels_to_run}")

    panels_data = {}
    for p in panels_to_run:
        y_p, X_p, _ = load_panel(p)
        panels_data[p] = (y_p, X_p)

    cfg = config.copy()
    cfg["experiment"] = config["experiment"].copy()
    cfg["experiment"]["panels"] = panels_to_run

    trials = build_trial_grid(cfg, panels_data)
    return run_experiment(trials, min_train_size, horizon)


def _safe_metric(results: dict, metric: str, stat: str):
    if not isinstance(results, dict):
        return None

    metrics = results.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    metric_values = metrics.get(metric, {})
    if not isinstance(metric_values, dict):
        return None

    return metric_values.get(stat)


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "NA"

    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_pct(value, digits: int = 2) -> str:
    formatted = _fmt(value, digits=digits)
    return "NA" if formatted == "NA" else f"{formatted}%"


def _fmt_rate(value, digits: int = 2) -> str:
    if value is None:
        return "NA"

    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass

    try:
        value = float(value)

        # win_rate is usually stored as 0.56, not 56.0.
        if abs(value) <= 1.0:
            value *= 100.0

        return f"{value:.{digits}f}%"
    except Exception:
        return str(value)


def _looks_like_single_result(results: dict) -> bool:
    if not isinstance(results, dict):
        return False

    nested_keys = {"backtest", "monte_carlo"}
    if any(k in results for k in nested_keys):
        return False

    return any(
        k in results
        for k in [
            "return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "final_value",
            "total_pnl",
            "final_cum_pnl",
        ]
    )


def _print_backtest_summary(r: dict) -> None:
    print(
        "[INFO] Backtest summary | "
        f"strategy={r.get('strategy', 'NA')} | "
        f"market_type={r.get('market_type', r.get('monthly_mode', 'NA'))} | "
        f"return_pct={_fmt_pct(r.get('return_pct'))} | "
        f"annualized_return_pct={_fmt_pct(r.get('annualized_return_pct'))} | "
        f"sharpe_ratio={_fmt(r.get('sharpe_ratio'))} | "
        f"max_drawdown_pct={_fmt_pct(r.get('max_drawdown_pct'))} | "
        f"win_rate={_fmt_rate(r.get('win_rate'))} | "
        f"num_trades={r.get('num_trades', 'NA')} | "
        f"final_value={_fmt(r.get('final_value'))} | "
        f"absolute_return={_fmt(r.get('absolute_return', r.get('total_pnl')))}"
    )


def _print_monte_carlo_summary(r: dict) -> None:
    initial_cash = r.get("initial_cash")

    final_value_mean = _safe_metric(r, "final_value", "mean")
    final_value_median = _safe_metric(r, "final_value", "median")
    final_value_p05 = _safe_metric(r, "final_value", "p05")
    final_value_p95 = _safe_metric(r, "final_value", "p95")
    final_value_min = _safe_metric(r, "final_value", "min")
    final_value_max = _safe_metric(r, "final_value", "max")

    absolute_return_mean = _safe_metric(r, "absolute_return", "mean")
    absolute_return_median = _safe_metric(r, "absolute_return", "median")
    absolute_return_p05 = _safe_metric(r, "absolute_return", "p05")
    absolute_return_p95 = _safe_metric(r, "absolute_return", "p95")

    print(
        "[INFO] Monte Carlo summary | "
        f"n_paths={r.get('n_paths')} | "
        f"model={r.get('simulation_model')} | "
        f"window={r.get('simulation_start')}->{r.get('simulation_end')} | "
        f"initial_cash={_fmt(initial_cash)}"
    )

    print(
        "[INFO] Monte Carlo returns | "
        f"mean={_fmt_pct(_safe_metric(r, 'return_pct', 'mean'))} | "
        f"median={_fmt_pct(_safe_metric(r, 'return_pct', 'median'))} | "
        f"p05={_fmt_pct(_safe_metric(r, 'return_pct', 'p05'))} | "
        f"p95={_fmt_pct(_safe_metric(r, 'return_pct', 'p95'))}"
    )

    print(
        "[INFO] Monte Carlo risk | "
        f"sharpe_mean={_fmt(_safe_metric(r, 'sharpe_ratio', 'mean'))} | "
        f"sharpe_median={_fmt(_safe_metric(r, 'sharpe_ratio', 'median'))} | "
        f"max_dd_mean={_fmt_pct(_safe_metric(r, 'max_drawdown_pct', 'mean'))} | "
        f"max_dd_median={_fmt_pct(_safe_metric(r, 'max_drawdown_pct', 'median'))}"
    )

    print(
        "[INFO] Monte Carlo final portfolio value | "
        f"mean={_fmt(final_value_mean)} | "
        f"median={_fmt(final_value_median)} | "
        f"p05={_fmt(final_value_p05)} | "
        f"p95={_fmt(final_value_p95)} | "
        f"min={_fmt(final_value_min)} | "
        f"max={_fmt(final_value_max)}"
    )

    print(
        "[INFO] Monte Carlo expected profit | "
        f"mean={_fmt(absolute_return_mean)} | "
        f"median={_fmt(absolute_return_median)} | "
        f"p05={_fmt(absolute_return_p05)} | "
        f"p95={_fmt(absolute_return_p95)}"
    )

    print(
        "[INFO] Monte Carlo trading stats | "
        f"win_rate_mean={_fmt(_safe_metric(r, 'win_rate', 'mean'))} | "
        f"win_rate_median={_fmt(_safe_metric(r, 'win_rate', 'median'))}"
    )


def _print_trading_summary(results: dict) -> None:
    if not isinstance(results, dict) or not results:
        print("[INFO] Trading summary | no results")
        return

    # Backwards compatibility: if a strategy returns a single flat result dict,
    # wrap it as one backtest result.
    if _looks_like_single_result(results):
        results = {str(results.get("mode", "backtest")): results}

    for mode, r in results.items():
        if not isinstance(r, dict):
            print(f"[INFO] Trading result | {mode}={r}")
            continue

        result_mode = str(r.get("mode", mode)).strip().lower()

        if result_mode == "monte_carlo":
            _print_monte_carlo_summary(r)

        elif result_mode == "backtest":
            _print_backtest_summary(r)

        else:
            print(
                "[INFO] Trading summary | "
                f"mode={result_mode} | "
                f"strategy={r.get('strategy', 'NA')} | "
                f"return_pct={_fmt_pct(r.get('return_pct'))} | "
                f"final_value={_fmt(r.get('final_value'))} | "
                f"num_trades={r.get('num_trades', 'NA')}"
            )


def run_trading_stage(refresh_market_data: bool = False):
    """
    Run the configured strategy from config.yaml against the latest
    experiment outputs already written to disk.
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
        _print_trading_summary(results)


def main():
    args = parse_args()

    if args.stage in ("all", "data"):
        run_data_stage()

    if args.stage in ("all", "experiment"):
        run_experiment_stage(args.panel)

    if args.stage in ("all", "trading") and not args.skip_trading:
        run_trading_stage(refresh_market_data=args.refresh_market_data)

    if args.stage in ("all", "plot"):
        from src.visualization.eval_plots import generate_all_dashboard_plots

        if args.panel is None:
            plot_panels = list(config["experiment"]["panels"])
        else:
            plot_panels = [args.panel]

        for p in plot_panels:
            print(f"\n[PLOT] Generating dashboard plots for panel '{p}'")
            generate_all_dashboard_plots(panel_name=p)
    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()