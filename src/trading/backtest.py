from __future__ import annotations

"""
src/trading/backtest.py

Portfolio backtesting for the strategy framework.

This version is intentionally aligned with the monthly workflow in krish-trade:
- strategy emits a decision/target allocation on each signal date
- engine enters on the next available trading day
- engine exits/rebalances on the next signal date
- supports multiple tickers and cash in one strategy

It still accepts the old single-column signal format, but the preferred output is:
    weight__SPY, weight__TLT, ..., cash_weight, confidence, metadata
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.trading.strategy import BaseStrategy, StrategyData
from src.utils.config import config

_MC_MODE_ALIASES = {
    "backtest": "backtest",
    "backtesting": "backtest",
    "historical": "backtest",
    "history": "backtest",
    "monte_carlo": "monte_carlo",
    "montecarlo": "monte_carlo",
    "mc": "monte_carlo",
    "simulation": "monte_carlo",
    "simulate": "monte_carlo",
}


def _coerce_test_mode(value: object) -> str:
    mode = str(value or "backtest").strip().lower()
    return _MC_MODE_ALIASES.get(mode, "backtest")


def _test_mode_from_config() -> str:
    trading_cfg = config.get("trading", {})
    return _coerce_test_mode(trading_cfg.get("test_mode", "backtest"))


def _monte_carlo_config() -> dict:
    trading_cfg = config.get("trading", {})
    mc_cfg = dict(trading_cfg.get("monte_carlo", {}) or {})
    mc_cfg.setdefault("model", "gbm")
    mc_cfg.setdefault("n_paths", 100)
    mc_cfg.setdefault("random_seed", 42)
    mc_cfg.setdefault("drift_mode", "historical")
    mc_cfg.setdefault("lookback_days", None)
    mc_cfg.setdefault("save_path_details", True)
    return mc_cfg


def _normalize_price_columns(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def _next_trading_day(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    candidates = index[index >= pd.Timestamp(dt)]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[0])


def _weight_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("weight__")]


def _legacy_signal_to_weights(signals_df: pd.DataFrame, default_ticker: str) -> pd.DataFrame:
    df = signals_df.copy()
    if "signal" not in df.columns:
        raise ValueError(
            "Strategy output must contain either 'signal' or one/more 'weight__TICKER' columns."
        )

    ticker = (default_ticker or "SPY").upper()
    weight_col = f"weight__{ticker}"
    signal = df["signal"].astype(float)

    if weight_col not in df.columns:
        df[weight_col] = signal
    if "cash_weight" not in df.columns:
        df["cash_weight"] = np.clip(1.0 - signal.abs(), 0.0, 1.0)
    if "confidence" not in df.columns:
        df["confidence"] = signal.abs()
    if "metadata" not in df.columns:
        df["metadata"] = "{}"

    return df


def _coerce_position_frame(signals_df: pd.DataFrame, default_ticker: str) -> pd.DataFrame:
    df = signals_df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.index.name = "date"

    if not _weight_columns(df):
        df = _legacy_signal_to_weights(df, default_ticker=default_ticker)

    if "cash_weight" not in df.columns:
        exposure = df[_weight_columns(df)].abs().sum(axis=1)
        df["cash_weight"] = np.clip(1.0 - exposure, 0.0, 1.0)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    if "metadata" not in df.columns:
        df["metadata"] = "{}"

    return df


def _extract_requested_tickers(positions_df: pd.DataFrame) -> list[str]:
    tickers = []
    for col in _weight_columns(positions_df):
        tickers.append(col.replace("weight__", "").upper())
    return tickers


def _close_price_frame(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    normalized = _normalize_price_columns(prices)
    cols = {}
    missing = []
    for ticker in tickers:
        col = f"{ticker.lower()}_close"
        if col not in normalized.columns:
            missing.append(col)
        else:
            cols[ticker] = normalized[col]

    if missing:
        raise ValueError(
            "Missing required close-price columns for requested tickers: "
            f"{missing}. Available columns (first 15): {list(normalized.columns[:15])}"
        )

    out = pd.DataFrame(cols).dropna(how="all").sort_index().ffill()
    if out.empty:
        raise ValueError("No close-price history available after aligning requested tickers.")
    return out


def _max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return float(drawdown.min())


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    if len(returns) < 2:
        return float("nan")
    vol = returns.std(ddof=1)
    if vol == 0 or pd.isna(vol):
        return float("nan")
    return float(np.sqrt(periods_per_year) * returns.mean() / vol)


def _run_weight_backtest(
    positions_df: pd.DataFrame,
    prices: pd.DataFrame,
    initial_cash: float,
    transaction_cost: float,
) -> tuple[pd.DataFrame, dict, pd.Series]:
    tickers = _extract_requested_tickers(positions_df)
    if not tickers:
        raise ValueError("No traded tickers found in strategy output.")

    close_prices = _close_price_frame(prices, tickers)
    equity = float(initial_cash)
    prev_weights = pd.Series(0.0, index=tickers, dtype=float)
    trade_rows: list[dict] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []

    for i in range(len(positions_df) - 1):
        row = positions_df.iloc[i]
        next_row = positions_df.iloc[i + 1]
        signal_date = pd.Timestamp(positions_df.index[i])
        next_signal_date = pd.Timestamp(positions_df.index[i + 1])

        entry_date = _next_trading_day(close_prices.index, signal_date)
        exit_date = _next_trading_day(close_prices.index, next_signal_date)
        if entry_date is None or exit_date is None or exit_date <= entry_date:
            continue

        weights = pd.Series(
            {
                ticker: float(row.get(f"weight__{ticker}", 0.0))
                for ticker in tickers
            },
            dtype=float,
        )

        ticker_returns = {}
        gross_return = 0.0
        for ticker in tickers:
            entry_px = float(close_prices.loc[entry_date, ticker])
            exit_px = float(close_prices.loc[exit_date, ticker])
            asset_return = (exit_px / entry_px) - 1.0
            ticker_returns[ticker] = asset_return
            gross_return += weights[ticker] * asset_return

        turnover = float((weights - prev_weights).abs().sum())
        cost = turnover * float(transaction_cost)
        net_return = gross_return - cost

        start_equity = equity
        equity = equity * (1.0 + net_return)
        prev_weights = weights
        equity_points.append((exit_date, equity))

        metadata = row.get("metadata", "{}")
        trade_rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "entry_date": entry_date.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "weights": json.dumps({k: round(float(v), 6) for k, v in weights.items()}),
                "cash_weight": float(row.get("cash_weight", 0.0)),
                "confidence": float(row.get("confidence", 1.0)),
                "gross_return": float(gross_return),
                "transaction_cost": float(cost),
                "net_return": float(net_return),
                "turnover": float(turnover),
                "ticker_returns": json.dumps({k: round(float(v), 6) for k, v in ticker_returns.items()}),
                "start_equity": float(start_equity),
                "end_equity": float(equity),
                "metadata": metadata,
            }
        )

    trades_df = pd.DataFrame(trade_rows)
    if trades_df.empty:
        equity_curve = pd.Series([initial_cash], index=[pd.Timestamp.today().normalize()], name="equity")
        results = {
            "sharpe_ratio": float("nan"),
            "max_drawdown_pct": 0.0,
            "return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "num_trades": 0,
            "won_trades": 0,
            "lost_trades": 0,
            "win_rate": 0.0,
            "avg_won_pnl": 0.0,
            "avg_lost_pnl": 0.0,
            "final_value": float(initial_cash),
            "absolute_return": 0.0,
            "monthly_mode": "multi_asset_weight",
            "tickers": tickers,
        }
        return trades_df, results, equity_curve

    equity_curve = pd.Series(
        [value for _, value in equity_points],
        index=pd.to_datetime([dt for dt, _ in equity_points]),
        name="equity",
    )
    period_returns = trades_df["net_return"].astype(float)
    win_mask = period_returns > 0
    years = max(len(trades_df) / 12.0, 1.0 / 12.0)
    total_return = (equity / float(initial_cash)) - 1.0
    annualized_return = (equity / float(initial_cash)) ** (1.0 / years) - 1.0

    results = {
        "sharpe_ratio": _annualized_sharpe(period_returns),
        "max_drawdown_pct": _max_drawdown(equity_curve) * 100.0,
        "return_pct": total_return * 100.0,
        "annualized_return_pct": annualized_return * 100.0,
        "num_trades": int(len(trades_df)),
        "won_trades": int(win_mask.sum()),
        "lost_trades": int((period_returns < 0).sum()),
        "win_rate": float(win_mask.mean()),
        "avg_won_pnl": float(period_returns[period_returns > 0].mean()) if (period_returns > 0).any() else 0.0,
        "avg_lost_pnl": float(period_returns[period_returns < 0].mean()) if (period_returns < 0).any() else 0.0,
        "final_value": float(equity),
        "absolute_return": float(equity - initial_cash),
        "monthly_mode": "multi_asset_weight",
        "tickers": tickers,
    }
    return trades_df, results, equity_curve

def _historical_risk_free_daily_from_macro(
    data: StrategyData,
    simulation_start: pd.Timestamp,
    lookback_days: Optional[int] = None,
    risk_free_column: Optional[str] = None,
) -> Optional[float]:
    macro = data.macro
    if macro is None or macro.empty:
        return None

    if risk_free_column and risk_free_column in macro.columns:
        series = pd.to_numeric(macro[risk_free_column], errors="coerce").dropna()
    else:
        candidates = ["DGS3MO", "DTB3", "TB3MS", "risk_free_rate", "rf"]
        found = next((c for c in candidates if c in macro.columns), None)
        if found is None:
            return None
        series = pd.to_numeric(macro[found], errors="coerce").dropna()

    series.index = pd.to_datetime(series.index)
    series = series.loc[series.index < simulation_start]

    if lookback_days is not None:
        lookback_days = int(lookback_days)
        if lookback_days > 0:
            series = series.tail(lookback_days)

    if series.empty:
        return None

    annual_rate = float(series.mean()) / 100.0
    daily_log_rf = np.log1p(annual_rate / 252.0)
    return float(daily_log_rf)

def _simulation_price_window(
    positions_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    tickers = _extract_requested_tickers(positions_df)
    if not tickers:
        raise ValueError("No traded tickers found in strategy output.")

    close_prices = _close_price_frame(prices, tickers)
    signal_dates = pd.DatetimeIndex(pd.to_datetime(positions_df.index)).sort_values().unique()

    start_dt = _next_trading_day(close_prices.index, signal_dates.min())
    if start_dt is None:
        raise ValueError("Could not find a tradable start date for Monte Carlo simulation.")

    end_dt = _next_trading_day(close_prices.index, signal_dates.max())
    if end_dt is None:
        end_dt = close_prices.index.max()

    simulation_window = close_prices.loc[(close_prices.index >= start_dt) & (close_prices.index <= end_dt)].copy()
    if simulation_window.empty:
        raise ValueError("Simulation window is empty after aligning signal dates to market prices.")

    return simulation_window, pd.Timestamp(start_dt), pd.Timestamp(end_dt)


def _calibration_returns(
    base_close_prices: pd.DataFrame,
    simulation_start: pd.Timestamp,
    lookback_days: Optional[int],
) -> pd.DataFrame:
    hist = base_close_prices.loc[base_close_prices.index < simulation_start].copy()
    if hist.empty:
        raise ValueError("No historical prices before simulation start; cannot calibrate Monte Carlo model.")

    if lookback_days is not None:
        lookback_days = int(lookback_days)
        if lookback_days > 0:
            hist = hist.tail(lookback_days + 1)

    log_returns = np.log(hist / hist.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("No calibration returns available for Monte Carlo simulation.")
    return log_returns


def _simulate_gbm_close_paths(
    base_close_prices: pd.DataFrame,
    simulation_window: pd.DataFrame,
    n_paths: int,
    random_seed: int,
    drift_mode: str = "historical",
    lookback_days: Optional[int] = None,
    risk_free_daily: Optional[float] = None,
) -> np.ndarray:
    simulation_start = pd.Timestamp(simulation_window.index.min())
    calibration_returns = _calibration_returns(
        base_close_prices=base_close_prices,
        simulation_start=simulation_start,
        lookback_days=lookback_days,
    )

    mu_hist = calibration_returns.mean().to_numpy(dtype=float)
    cov = calibration_returns.cov().to_numpy(dtype=float)

    n_steps = len(simulation_window.index)
    n_assets = len(simulation_window.columns)
    n_paths = int(n_paths)

    if n_steps == 0 or n_assets == 0:
        raise ValueError("Simulation window has no steps or no assets.")

    if str(drift_mode).lower() == "zero":
        mu_vec = np.zeros_like(mu_hist)
    elif str(drift_mode).lower() == "risk_free":
        if risk_free_daily is None:
            raise ValueError("drift_mode='risk_free' requires risk_free_daily.")
        mu_vec = np.full_like(mu_hist, float(risk_free_daily))
    else:
        mu_vec = mu_hist

    start_prices = simulation_window.iloc[0].to_numpy(dtype=float)
    rng = np.random.default_rng(int(random_seed))

    paths = np.zeros((n_paths, n_steps, n_assets), dtype=float)
    paths[:, 0, :] = start_prices

    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(n_assets))
    variances = np.diag(cov)

    for step in range(1, n_steps):
        z = rng.standard_normal(size=(n_paths, n_assets))
        correlated = z @ chol.T
        log_step = (mu_vec - 0.5 * variances) + correlated
        paths[:, step, :] = paths[:, step - 1, :] * np.exp(log_step)

    return paths

def _inject_simulated_close_prices(
    prices: pd.DataFrame,
    simulation_window: pd.DataFrame,
    simulated_path: np.ndarray,
) -> pd.DataFrame:
    updated = _normalize_price_columns(prices)
    for idx, ticker in enumerate(simulation_window.columns):
        close_col = f"{ticker.lower()}_close"
        updated.loc[simulation_window.index, close_col] = simulated_path[:, idx]
    return updated


def _aggregate_monte_carlo_results(
    path_results_df: pd.DataFrame,
    equity_curves: list[pd.Series],
    simulation_window: pd.DataFrame,
    mc_cfg: dict,
    strategy_name: str,
    initial_cash: float,
    ticker: str,
) -> dict:
    if path_results_df.empty:
        raise ValueError("Monte Carlo produced no path results.")

    summary = {
        "mode": "monte_carlo",
        "strategy": strategy_name,
        "ticker": ticker,
        "initial_cash": float(initial_cash),
        "simulation_model": str(mc_cfg.get("model", "gbm")),
        "n_paths": int(mc_cfg.get("n_paths", len(path_results_df))),
        "simulation_start": simulation_window.index.min().date().isoformat(),
        "simulation_end": simulation_window.index.max().date().isoformat(),
        "tickers": list(simulation_window.columns),
        "metrics": {},
    }

    metric_cols = [
        "return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate",
        "final_value",
        "absolute_return",
    ]
    for col in metric_cols:
        series = pd.to_numeric(path_results_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary["metrics"][col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "p05": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
        }

    if equity_curves:
        equity_df = pd.concat(equity_curves, axis=1).sort_index().ffill()
        median_equity = equity_df.median(axis=1)
        summary["equity_curve"] = {ts.isoformat(): float(val) for ts, val in median_equity.items()}

    return summary


def _run_weight_monte_carlo(
    strategy: BaseStrategy,
    data: StrategyData,
    initial_cash: float,
    transaction_cost: float,
    default_ticker: str,
    mc_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    base_signals_df = strategy.generate_signals(data)
    base_positions_df = _coerce_position_frame(base_signals_df, default_ticker=default_ticker)

    tickers = _extract_requested_tickers(base_positions_df)
    if not tickers:
        raise ValueError("No traded tickers found in strategy output.")

    simulation_window, _, _ = _simulation_price_window(
        positions_df=base_positions_df,
        prices=data.prices,
    )

    drift_mode = str(mc_cfg.get("drift_mode", "historical")).lower()
    risk_free_daily = None

    if drift_mode == "risk_free":
        risk_free_daily = _historical_risk_free_daily_from_macro(
            data=data,
            simulation_start=pd.Timestamp(simulation_window.index.min()),
            lookback_days=mc_cfg.get("lookback_days"),
            risk_free_column=mc_cfg.get("risk_free_column"),
        )
        if risk_free_daily is None:
            raise ValueError(
                "Monte Carlo drift_mode='risk_free' requested, but no usable risk-free series was found in data.macro."
            )

    simulated_paths = _simulate_gbm_close_paths(
        base_close_prices=_close_price_frame(data.prices, tickers),
        simulation_window=simulation_window,
        n_paths=int(mc_cfg.get("n_paths", 100)),
        random_seed=int(mc_cfg.get("random_seed", 42)),
        drift_mode=drift_mode,
        lookback_days=mc_cfg.get("lookback_days"),
        risk_free_daily=risk_free_daily,
    )

    save_path_details = bool(mc_cfg.get("save_path_details", True))
    path_rows = []
    trade_frames = []
    equity_curves = []

    for path_id, simulated_path in enumerate(simulated_paths):
        simulated_prices = _inject_simulated_close_prices(
            prices=data.prices,
            simulation_window=simulation_window,
            simulated_path=simulated_path,
        )
        simulated_data = data.copy_with(prices=simulated_prices)

        signals_df = strategy.generate_signals(simulated_data)
        positions_df = _coerce_position_frame(signals_df, default_ticker=default_ticker)
        trades_df, result, equity_curve = _run_weight_backtest(
            positions_df=positions_df,
            prices=simulated_prices,
            initial_cash=initial_cash,
            transaction_cost=transaction_cost,
        )

        equity_curves.append(equity_curve.rename(f"path_{path_id}"))
        path_rows.append(
            {
                "path_id": int(path_id),
                "return_pct": float(result.get("return_pct", 0.0)),
                "annualized_return_pct": float(result.get("annualized_return_pct", 0.0)),
                "sharpe_ratio": float(result.get("sharpe_ratio", 0.0)),
                "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0)),
                "num_trades": int(result.get("num_trades", 0)),
                "won_trades": int(result.get("won_trades", 0)),
                "lost_trades": int(result.get("lost_trades", 0)),
                "win_rate": float(result.get("win_rate", 0.0)),
                "final_value": float(result.get("final_value", initial_cash)),
                "absolute_return": float(result.get("absolute_return", 0.0)),
            }
        )

        if save_path_details:
            detailed = trades_df.copy()
            detailed.insert(0, "path_id", int(path_id))
            trade_frames.append(detailed)

    path_results_df = pd.DataFrame(path_rows)
    all_trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    results = _aggregate_monte_carlo_results(
        path_results_df=path_results_df,
        equity_curves=equity_curves,
        simulation_window=simulation_window,
        mc_cfg=mc_cfg,
        strategy_name=strategy.name,
        initial_cash=initial_cash,
        ticker=(default_ticker or "").upper(),
    )
    return base_signals_df, all_trades_df, path_results_df, results

def _run_forecastex_loop(
    signals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    contract_value: float = 100.0,
    bid_ask_spread: float = 0.02,
) -> pd.DataFrame:
    selected = forecasts_df.set_index("date")[["y_true", "y_pred"]].rename(
        columns={"y_pred": "our_pred"}
    )
    consensus = forecasts_df.groupby("date")["y_pred"].mean().rename("consensus")

    df = signals_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)

    if "signal" not in df.columns:
        weight_cols = _weight_columns(df)
        if not weight_cols:
            raise ValueError("FORECASTEX simulation requires a scalar signal or weight columns.")
        df["signal"] = df[weight_cols].sum(axis=1)

    df = df.join(selected, how="left")
    df = df.join(consensus, how="left")
    df["pnl"] = (
        df["signal"] * contract_value * (df["y_true"] - df["consensus"])
        - df["signal"].abs() * bid_ask_spread * contract_value
    )
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df


class BacktestEngine:
    def __init__(
        self,
        initial_cash: float = None,
        commission: float = None,
        ticker: str = None,
    ):
        cfg = config.get("trading", {}).get("portfolio", {})
        self.initial_cash = float(initial_cash or cfg.get("initial_cash", 100_000))
        self.commission = float(commission or cfg.get("commission", 0.002))
        self.ticker = (ticker or cfg.get("ticker", "XLY")).upper()

    def run_portfolio(
        self,
        strategy: BaseStrategy,
        data: StrategyData,
        output_dir: Optional[Path] = None,
        test_mode: Optional[str] = None,
    ) -> dict:
        if data.prices is None:
            raise ValueError(
                "run_portfolio() requires data.prices. "
                "Load market data before running the trading pipeline."
            )

        test_mode = _coerce_test_mode(test_mode or _test_mode_from_config())
        default_ticker = strategy.default_ticker or self.ticker

        if test_mode == "monte_carlo":
            mc_cfg = _monte_carlo_config()
            signals_df, trades_df, path_results_df, results = _run_weight_monte_carlo(
                strategy=strategy,
                data=data,
                initial_cash=self.initial_cash,
                transaction_cost=self.commission,
                default_ticker=default_ticker,
                mc_cfg=mc_cfg,
            )
            positions_df = _coerce_position_frame(signals_df, default_ticker=default_ticker)
        else:
            signals_df = strategy.generate_signals(data)
            positions_df = _coerce_position_frame(signals_df, default_ticker=default_ticker)
            trades_df, results, equity_curve = _run_weight_backtest(
                positions_df=positions_df,
                prices=data.prices,
                initial_cash=self.initial_cash,
                transaction_cost=self.commission,
            )
            path_results_df = pd.DataFrame()
            results["mode"] = "backtest"
            results["equity_curve"] = {
                ts.isoformat(): float(val) for ts, val in equity_curve.items()
            }

        results["strategy"] = strategy.name
        results["initial_cash"] = self.initial_cash
        results["ticker"] = self.ticker

        if output_dir is not None:
            self._write_portfolio_results(
                results=results,
                raw_signals_df=signals_df,
                positions_df=positions_df,
                trades_df=trades_df,
                path_results_df=path_results_df,
                output_dir=Path(output_dir),
            )

        return results

    def run_forecastex(
        self,
        strategy: BaseStrategy,
        data: StrategyData,
        all_forecasts: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> dict:
        cfg_fx = config.get("trading", {}).get("forecastex", {})
        contract_value = float(cfg_fx.get("contract_value", 100.0))
        bid_ask_spread = float(cfg_fx.get("bid_ask_spread", 0.02))

        signals_df = strategy.generate_signals(data)
        trades_df = _run_forecastex_loop(
            signals_df=signals_df,
            forecasts_df=all_forecasts,
            contract_value=contract_value,
            bid_ask_spread=bid_ask_spread,
        )

        active = trades_df[trades_df["signal"] != 0]
        wins = int((active["pnl"] > 0).sum())
        total = int(len(active))

        results = {
            "strategy": strategy.name,
            "total_pnl": float(trades_df["pnl"].sum()),
            "mean_monthly_pnl": float(trades_df["pnl"].mean()),
            "win_rate": float(wins / total) if total > 0 else float("nan"),
            "num_trades": total,
            "final_cum_pnl": float(trades_df["cumulative_pnl"].iloc[-1]) if not trades_df.empty else 0.0,
        }

        if output_dir is not None:
            self._write_forecastex_results(results=results, trades_df=trades_df, output_dir=Path(output_dir))

        return results

    def _write_portfolio_results(
        self,
        results: dict,
        raw_signals_df: pd.DataFrame,
        positions_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        path_results_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = results["strategy"]
        mode = str(results.get("mode", "backtest")).lower()

        raw_signals_df.reset_index().to_csv(output_dir / f"signals_{name}.csv", index=False)
        positions_df.reset_index().to_csv(output_dir / f"positions_{name}.csv", index=False)

        if mode == "monte_carlo":
            trades_df.to_csv(output_dir / f"monte_carlo_trades_{name}.csv", index=False)
            path_results_df.to_csv(output_dir / f"monte_carlo_path_results_{name}.csv", index=False)
            scalar_results = {k: v for k, v in results.items() if k != "equity_curve"}
            with open(output_dir / f"monte_carlo_results_{name}.json", "w") as f:
                json.dump(scalar_results, f, indent=2)
            if results.get("equity_curve"):
                equity_curve = pd.Series(results["equity_curve"], name="median_equity")
                equity_curve.to_csv(output_dir / f"monte_carlo_equity_curve_{name}.csv")
            return

        trades_df.to_csv(output_dir / f"backtest_trades_{name}.csv", index=False)
        scalar_results = {k: v for k, v in results.items() if k != "equity_curve"}
        with open(output_dir / f"backtest_results_{name}.json", "w") as f:
            json.dump(scalar_results, f, indent=2)

        if results.get("equity_curve"):
            equity_curve = pd.Series(results["equity_curve"], name="equity")
            equity_curve.to_csv(output_dir / f"equity_curve_{name}.csv")

    def _write_forecastex_results(
        self,
        results: dict,
        trades_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = results["strategy"]
        trades_df.reset_index().to_csv(output_dir / f"forecastex_trades_{name}.csv", index=False)
        with open(output_dir / f"forecastex_results_{name}.json", "w") as f:
            json.dump(results, f, indent=2)