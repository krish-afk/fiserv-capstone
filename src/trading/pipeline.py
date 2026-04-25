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
    get_active_strategy_config,
    load_configured_strategy,
    load_latest_run_frames,
    select_best_model,
)
from src.utils.config import config


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
    strategy_tickers = [str(t).upper() for t in getattr(strategy, "tickers", [])]

    fallback = []
    if not strategy_tickers:
        try:
            ticker = get_active_strategy_config(cfg).get("params", {}).get("ticker")
            if ticker:
                fallback = [str(ticker).upper()]
        except KeyError:
            pass

    return _dedupe_keep_order([*strategy_tickers, *fallback])


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
    market_cfg = cfg.get("trading", {}).get("market_data", {})

    tickers = resolve_strategy_tickers(strategy, cfg=cfg)
    if not tickers:
        return None

    start_date = str(cfg["data"]["start_date"])
    end_date = str(cfg["data"]["end_date"])
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
    try:
        return str(get_active_strategy_config(cfg).get("forecast_panels", {}).get(key, default))
    except KeyError:
        return default


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

    if not strategy.tickers:
        all_forecasts = data.context.get("all_forecasts", data.forecasts)
        results = engine.run_forecastex(
            strategy=strategy,
            data=data,
            all_forecasts=all_forecasts,
            output_dir=target_output_dir,
        )
    else:
        results = engine.run_portfolio(strategy=strategy, data=data, output_dir=target_output_dir)

    return {
        "strategy_name": strategy.name,
        "tickers": resolve_strategy_tickers(strategy, cfg=cfg),
        "output_dir": str(target_output_dir),
        "results": results,
        "data": data,
    }