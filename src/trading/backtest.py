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
    ) -> dict:
        if data.prices is None:
            raise ValueError(
                "run_portfolio() requires data.prices. "
                "Load market data before running the trading pipeline."
            )

        signals_df = strategy.generate_signals(data)
        positions_df = _coerce_position_frame(signals_df, default_ticker=strategy.default_ticker or self.ticker)
        trades_df, results, equity_curve = _run_weight_backtest(
            positions_df=positions_df,
            prices=data.prices,
            initial_cash=self.initial_cash,
            transaction_cost=self.commission,
        )

        results["strategy"] = strategy.name
        results["initial_cash"] = self.initial_cash
        results["ticker"] = self.ticker
        results["equity_curve"] = {
            ts.isoformat(): float(val) for ts, val in equity_curve.items()
        }

        if output_dir is not None:
            self._write_portfolio_results(
                results=results,
                raw_signals_df=signals_df,
                positions_df=positions_df,
                trades_df=trades_df,
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
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = results["strategy"]

        raw_signals_df.reset_index().to_csv(output_dir / f"signals_{name}.csv", index=False)
        positions_df.reset_index().to_csv(output_dir / f"positions_{name}.csv", index=False)
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