from __future__ import annotations

"""
src/trading/pipeline.py

End-to-end helpers for running whichever strategy is selected in config.yaml.
"""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.trading.backtest import BacktestEngine
from src.trading.strategy import (
    DEFAULT_STRATEGY_DIR,
    BaseStrategy,
    StrategyData,
    load_configured_strategy,
    load_latest_run_frames,
    select_best_model,
)
from src.utils.config import config

def _filter_frame_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    if date_col not in out.columns:
        return out

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    if start_date:
        start_ts = pd.to_datetime(start_date)
        out = out[out[date_col] >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date)
        out = out[out[date_col] <= end_ts]

    return out


def _filter_prices_by_date(
    prices_df: Optional[pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    if prices_df is None or prices_df.empty:
        return prices_df

    out = prices_df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])

        if start_date:
            out = out[out["date"] >= pd.to_datetime(start_date)]
        if end_date:
            out = out[out["date"] <= pd.to_datetime(end_date)]
        return out

    try:
        out.index = pd.to_datetime(out.index)
        if start_date:
            out = out[out.index >= pd.to_datetime(start_date)]
        if end_date:
            out = out[out.index <= pd.to_datetime(end_date)]
    except Exception:
        return prices_df

    return out

def _project_src_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _read_signals_artifact(output_dir: Path, strategy_name: str) -> pd.DataFrame:
    path = output_dir / f"signals_{strategy_name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def _load_data_load_module():
    path = _project_src_dir() / "data" / "load.py"
    return _load_module_from_file("src_data_load_direct", path)


def _market_raw_dir() -> Path:
    return Path(config["paths"]["raw_data"])


def _latest_market_file() -> Optional[Path]:
    matches = sorted(_market_raw_dir().glob("market_*.csv"))
    return matches[-1] if matches else None


def _make_market_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return _market_raw_dir() / f"market_{stamp}.csv"


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _load_market_data_local(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _fetch_market_data_local(
    tickers: list[str],
    start_date: str,
    end_date: str,
    source: str,
    out_path: Path,
) -> Path:
    if source != "yfinance":
        raise ValueError(f"Unsupported market data source '{source}'. Supported: ['yfinance']")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to refresh market data. Install it with: pip install yfinance"
        ) from exc

    frames = []
    for ticker in tickers:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{ticker.lower()}_{c[0].lower()}" for c in data.columns]
        else:
            data.columns = [f"{ticker.lower()}_{c.lower()}" for c in data.columns]

        data.index = pd.to_datetime(data.index)
        data.index.name = "date"
        frames.append(data)

    if not frames:
        raise ValueError(f"No market data fetched for tickers: {tickers}")

    merged = pd.concat(frames, axis=1).sort_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_csv(out_path, index=False)
    return out_path


def resolve_strategy_tickers(strategy: BaseStrategy, cfg: Optional[dict] = None) -> list[str]:
    cfg = cfg or config
    trading_cfg = cfg.get("trading", {})
    market_cfg = trading_cfg.get("market_data", {})

    configured = [str(t).upper() for t in market_cfg.get("tickers", [])]
    strategy_tickers = [str(t).upper() for t in getattr(strategy, "tickers", [])]

    fallback = []
    if not strategy_tickers and trading_cfg.get("portfolio", {}).get("ticker"):
        fallback = [str(trading_cfg["portfolio"]["ticker"]).upper()]

    return _dedupe_keep_order([*configured, *strategy_tickers, *fallback])


def _market_data_covers(df: pd.DataFrame, tickers: list[str], start_date: str, end_date: str) -> bool:
    if df is None or df.empty:
        return False

    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized.columns = [str(c).lower() for c in normalized.columns]

    required = [f"{ticker.lower()}_close" for ticker in tickers]
    if not all(col in normalized.columns for col in required):
        return False

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return normalized.index.min() <= start_ts and normalized.index.max() >= end_ts


def ensure_market_data_for_strategy(
    strategy: BaseStrategy,
    cfg: Optional[dict] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    cfg = cfg or config
    trading_cfg = cfg.get("trading", {})
    market_cfg = trading_cfg.get("market_data", {})

    tickers = resolve_strategy_tickers(strategy, cfg=cfg)
    if not tickers:
        return None

    start_date = str(market_cfg.get("start_date", cfg["data"]["start_date"]))
    end_date = str(market_cfg.get("end_date", cfg["data"]["end_date"]))
    source = str(market_cfg.get("source", "yfinance"))

    if not refresh:
        latest_market = _latest_market_file()
        if latest_market is not None:
            try:
                existing = _load_market_data_local(latest_market)
                if _market_data_covers(existing, tickers, start_date, end_date):
                    return existing
            except Exception:
                pass

    market_path = _fetch_market_data_local(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        source=source,
        out_path=_make_market_path(),
    )
    return _load_market_data_local(market_path)


def _forecast_panel_name(cfg: dict, key: str, default: str) -> str:
    return str(cfg.get("trading", {}).get("forecast_panels", {}).get(key, default))


def _load_macro_if_needed() -> pd.DataFrame:
    load_mod = _load_data_load_module()
    return load_mod.load_fred()


def build_strategy_data(
    strategy: BaseStrategy,
    cfg: Optional[dict] = None,
    experiments_dir: Optional[Path] = None,
    refresh_market_data: bool = False,
) -> StrategyData:
    cfg = cfg or config
    _, forecasts_df, metrics_df = load_latest_run_frames(experiments_dir=experiments_dir)

    primary_panel = _forecast_panel_name(cfg, "primary", "pce")
    primary_forecasts = select_best_model(
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        panel_name=primary_panel,
    )

    mrts_df = None
    if "mrts" in strategy.required_inputs:
        mrts_panel = _forecast_panel_name(cfg, "mrts", "mrts")
        mrts_df = select_best_model(
            forecasts_df=forecasts_df,
            metrics_df=metrics_df,
            panel_name=mrts_panel,
        )

    macro_df = None
    mc_cfg = cfg.get("trading", {}).get("monte_carlo", {}) or {}
    need_macro = (
        "macro" in strategy.required_inputs
        or str(mc_cfg.get("drift_mode", "historical")).lower() == "risk_free"
    )
    if need_macro:
        macro_df = _load_macro_if_needed()

    prices_df = None
    if resolve_strategy_tickers(strategy, cfg=cfg):
        prices_df = ensure_market_data_for_strategy(
            strategy=strategy,
            cfg=cfg,
            refresh=refresh_market_data,
        )

    return StrategyData(
        forecasts=primary_forecasts,
        prices=prices_df,
        macro=macro_df,
        mrts=mrts_df,
        cfg=cfg,
        context={
            "all_forecasts": forecasts_df,
            "metrics": metrics_df,
            "strategy_tickers": resolve_strategy_tickers(strategy, cfg=cfg),
        },
    )

def _equity_curve_frame(results: dict) -> pd.DataFrame:
    equity_map = results.get("equity_curve") or {}
    if not equity_map:
        return pd.DataFrame(columns=["date", "equity"])

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(equity_map.keys())),
            "equity": list(equity_map.values()),
        }
    ).sort_values("date")
    return df.reset_index(drop=True)


def _read_trades_artifact(output_dir: Path, strategy_name: str, test_mode: str) -> pd.DataFrame:
    mode = str(test_mode).strip().lower()
    if mode == "monte_carlo":
        path = output_dir / f"monte_carlo_trades_{strategy_name}.csv"
    else:
        path = output_dir / f"backtest_trades_{strategy_name}.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)

def _read_path_results_artifact(output_dir: Path, strategy_name: str, test_mode: str) -> pd.DataFrame:
    mode = str(test_mode).strip().lower()
    if mode != "monte_carlo":
        return pd.DataFrame()

    path = output_dir / f"monte_carlo_path_results_{strategy_name}.csv"
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def build_strategy_data_from_frames(
    strategy: BaseStrategy,
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    primary_panel: str,
    cfg: Optional[dict] = None,
    refresh_market_data: bool = False,
    mrts_panel: Optional[str] = None,
    trade_start_date: Optional[str] = None,
    trade_end_date: Optional[str] = None,
) -> StrategyData:
    """
    Build StrategyData directly from the current experiment outputs instead of
    loading the latest experiment folder from disk.
    """
    cfg = cfg or config

    primary_forecasts = select_best_model(
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        panel_name=primary_panel,
    )

    if primary_forecasts is None or primary_forecasts.empty:
        raise ValueError(f"No forecast rows available for panel '{primary_panel}'.")

    mrts_df = None
    if "mrts" in strategy.required_inputs:
        mrts_panel = mrts_panel or _forecast_panel_name(cfg, "mrts", "mrts_national")
        mrts_df = select_best_model(
            forecasts_df=forecasts_df,
            metrics_df=metrics_df,
            panel_name=mrts_panel,
        )

    macro_df = None
    mc_cfg = cfg.get("trading", {}).get("monte_carlo", {}) or {}
    need_macro = (
        "macro" in strategy.required_inputs
        or str(mc_cfg.get("drift_mode", "historical")).lower() == "risk_free"
    )
    if need_macro:
        macro_df = _load_macro_if_needed()

    prices_df = None
    if resolve_strategy_tickers(strategy, cfg=cfg):
        prices_df = ensure_market_data_for_strategy(
            strategy=strategy,
            cfg=cfg,
            refresh=refresh_market_data,
        )

    return StrategyData(
        forecasts=primary_forecasts,
        prices=prices_df,
        macro=macro_df,
        mrts=mrts_df,
        cfg=cfg,
        context={
            "all_forecasts": forecasts_df,
            "metrics": metrics_df,
            "trade_window_start": trade_start_date,
            "trade_window_end": trade_end_date,
            "strategy_tickers": resolve_strategy_tickers(strategy, cfg=cfg),
            "primary_panel": primary_panel,
            "mrts_panel": mrts_panel,
        },
    )


def run_trading_pipeline_from_frames(
    strategy: BaseStrategy,
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    primary_panel: str,
    cfg: Optional[dict] = None,
    refresh_market_data: bool = False,
    output_dir: Optional[Path] = None,
    mrts_panel: Optional[str] = None,
    test_mode: Optional[str] = None,
    trade_start_date: Optional[str] = None,
    trade_end_date: Optional[str] = None,
) -> dict:
    """
    Run trading from in-memory experiment outputs produced by the current API run.
    """
    cfg = cfg or config
    mode = str(test_mode or cfg.get("trading", {}).get("test_mode", "backtest")).lower()

    data = build_strategy_data_from_frames(
        strategy=strategy,
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        primary_panel=primary_panel,
        cfg=cfg,
        refresh_market_data=refresh_market_data,
        mrts_panel=mrts_panel,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
    )

    target_output_dir = Path(output_dir or (Path(cfg["paths"]["experiments"]) / "dashboard_trading"))
    engine = BacktestEngine()

    results = engine.run_portfolio(
        strategy=strategy,
        data=data,
        output_dir=target_output_dir,
        test_mode=mode,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
    )

    trades_df = _read_trades_artifact(
        output_dir=target_output_dir,
        strategy_name=strategy.name,
        test_mode=mode,
    )

    path_results_df = _read_path_results_artifact(
        output_dir=target_output_dir,
        strategy_name=strategy.name,
        test_mode=mode,
    )

    signals_df = _read_signals_artifact(
        output_dir=target_output_dir,
        strategy_name=strategy.name,
    )
    equity_curve_df = _equity_curve_frame(results)

    return {
        "strategy_name": strategy.name,
        "tickers": resolve_strategy_tickers(strategy, cfg=cfg),
        "output_dir": str(target_output_dir),
        "results": results,
        "data": data,
        "signals_df": signals_df,
        "trades_df": trades_df,
        "path_results_df": path_results_df,
        "equity_curve_df": equity_curve_df,
        "test_mode": mode,
        "trade_window_start": trade_start_date,
        "trade_window_end": trade_end_date,
    }

def run_configured_trading_pipeline(
    cfg: Optional[dict] = None,
    strategy_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
    refresh_market_data: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    cfg = cfg or config
    strategy = load_configured_strategy(
        cfg=cfg,
        strategy_dir=(strategy_dir or DEFAULT_STRATEGY_DIR),
    )

    latest_run, _, _ = load_latest_run_frames(experiments_dir=experiments_dir)
    data = build_strategy_data(
        strategy=strategy,
        cfg=cfg,
        experiments_dir=experiments_dir,
        refresh_market_data=refresh_market_data,
    )

    engine = BacktestEngine()
    target_output_dir = Path(output_dir or (latest_run / "trading"))
    results = engine.run_portfolio(strategy=strategy, data=data, output_dir=target_output_dir)

    return {
        "strategy_name": strategy.name,
        "tickers": resolve_strategy_tickers(strategy, cfg=cfg),
        "output_dir": str(target_output_dir),
        "results": results,
        "data": data,
    }