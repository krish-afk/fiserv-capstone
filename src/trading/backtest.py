# src/trading/backtest.py
"""
Backtesting engine for PCE-forecast-driven trading strategies.

Two execution paths
-------------------
BacktestEngine.run_portfolio()
    Wraps backtrader (https://www.backtrader.com/).
    Resamples daily OHLCV to monthly, attaches the pre-computed signal series
    as a custom data line, and runs a generic signal-following adapter strategy.
    Returns a standardised results dict including Sharpe, drawdown, trades.

BacktestEngine.run_forecastex()
    Simple P&L loop for prediction-market-style contracts.
    No external library needed.  P&L is proportional to how close our forecast
    is to the realised value relative to the consensus (market) price.

Both paths accept a BaseStrategy instance and a StrategyData container.
They write results to the experiment output directory (same schema as
forecasts.csv / metrics.csv) so downstream performance.py can read them.

Usage
-----
    engine  = BacktestEngine()
    results = engine.run_portfolio(strategy, data, output_dir=run_dir)
    results = engine.run_forecastex(strategy, data, all_forecasts, output_dir=run_dir)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.trading.strategy import BaseStrategy, StrategyData
from src.utils.config import config

# backtrader is an optional dependency — portfolio path only.
try:
    import backtrader as bt
    import backtrader.analyzers as btanalyzers
    _BT_AVAILABLE = True
except ImportError:
    _BT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom backtrader data feed — monthly OHLCV + pre-computed signal line
# ---------------------------------------------------------------------------

def _make_signal_feed_class():
    """
    Return a bt.feeds.PandasData subclass with an extra 'signal' line.
    Defined at call time so the import guard above doesn't prevent module load.
    """
    class _MonthlySignalFeed(bt.feeds.PandasData):
        """
        Monthly OHLCV feed with an additional 'signal' line.
        The DataFrame passed to this feed must have columns:
            open, high, low, close, volume, signal
        (lowercase, as produced by _build_monthly_feed()).
        """
        lines  = ("signal",)
        params = (
            ("datetime", None),   # index is the datetime
            ("open",     "open"),
            ("high",     "high"),
            ("low",      "low"),
            ("close",    "close"),
            ("volume",   "volume"),
            ("openinterest", -1),  # not present → -1 means ignore
            ("signal",   "signal"),
        )
    return _MonthlySignalFeed


def _make_adapter_strategy_class():
    """
    Return a backtrader Strategy subclass that executes pre-computed signals.
    Long on signal > 0, short on signal < 0, flat on signal == 0.
    """
    class _SignalAdapter(bt.Strategy):
        params = (("verbose", False),)

        def log(self, txt):
            if self.params.verbose:
                dt = self.datas[0].datetime.date(0)
                print(f"[BT] {dt}: {txt}")

        def __init__(self):
            self.signal = self.datas[0].signal

        def next(self):
            sig = self.signal[0]
            pos = self.getposition().size

            if sig > 0 and pos <= 0:
                self.log(f"BUY  signal={sig:.3f}")
                if pos < 0:
                    self.close()
                self.buy()
            elif sig < 0 and pos >= 0:
                self.log(f"SELL signal={sig:.3f}")
                if pos > 0:
                    self.close()
                self.sell()
            elif sig == 0 and pos != 0:
                self.log("FLAT")
                self.close()

    return _SignalAdapter


# ---------------------------------------------------------------------------
# OHLCV resampling + feed construction
# ---------------------------------------------------------------------------

def _build_monthly_feed(prices: pd.DataFrame, ticker: str, signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV for *ticker* to month-end frequency and merge the
    pre-computed signal series.

    Args:
        prices:     Date-indexed DataFrame with flat columns {ticker}_open/high/low/close/volume
        ticker:     E.g. "XLY"
        signals_df: Strategy output — DatetimeIndex, must have a "signal" column
    Returns:
        DataFrame with lowercase OHLCV columns + "signal", month-end DatetimeIndex.
        Ready to pass to _MonthlySignalFeed.
    """
    col = lambda field: f"{ticker}_{field}"
    has_ohlcv = all(col(f) in prices.columns for f in ("open", "high", "low", "close", "volume"))

    if has_ohlcv:
        monthly = pd.DataFrame({
            "open":   prices[col("open")].resample("ME").first(),
            "high":   prices[col("high")].resample("ME").max(),
            "low":    prices[col("low")].resample("ME").min(),
            "close":  prices[col("close")].resample("ME").last(),
            "volume": prices[col("volume")].resample("ME").sum(),
        }).dropna()
    elif col("close") in prices.columns:
        # Only close available — synthesise OHLV from close
        close = prices[col("close")].resample("ME").last().dropna()
        monthly = pd.DataFrame({
            "open":   close,
            "high":   close,
            "low":    close,
            "close":  close,
            "volume": np.ones(len(close)),
        })
    else:
        raise ValueError(
            f"prices DataFrame has no columns for ticker '{ticker}'. "
            f"Expected '{ticker}_close' at minimum."
        )

    # Align signal to monthly index — forward-fill, then fill remaining NaN with 0
    monthly_signal = (
        signals_df["signal"]
        .reindex(monthly.index, method="ffill")
        .fillna(0)
    )
    monthly["signal"] = monthly_signal.values
    monthly.index.name = "datetime"
    return monthly


# ---------------------------------------------------------------------------
# Analyser result extraction
# ---------------------------------------------------------------------------

def _extract_analyzers(thestrats: list) -> dict:
    """
    Pull results from backtrader analysers attached to the strategy run.

    Args:
        thestrats: List returned by cerebro.run()
    Returns:
        Dict of scalar metrics.
    """
    strat = thestrats[0]

    def _safe(analyser_name, *keys, default=float("nan")):
        try:
            a = getattr(strat.analyzers, analyser_name).get_analysis()
            for k in keys:
                a = a[k]
            return float(a)
        except (KeyError, TypeError, AttributeError, ZeroDivisionError):
            return default

    sharpe   = _safe("sharpe",   "sharperatio")
    max_dd   = _safe("drawdown", "max", "drawdown")   # as %
    max_dd_m = _safe("drawdown", "max", "moneydown")

    ta = {}
    try:
        ta = strat.analyzers.trades.get_analysis()
    except AttributeError:
        pass

    total_trades  = _safe_ta(ta, "total",  "total")
    won_trades    = _safe_ta(ta, "won",    "total")
    lost_trades   = _safe_ta(ta, "lost",   "total")
    win_rate      = (won_trades / total_trades) if total_trades > 0 else float("nan")
    avg_won       = _safe_ta(ta, "won",  "pnl", "average")
    avg_lost      = _safe_ta(ta, "lost", "pnl", "average")

    roi_pct = _safe("returns", "rnorm100")

    return {
        "sharpe_ratio":     sharpe,
        "max_drawdown_pct": max_dd,
        "max_drawdown_cash":max_dd_m,
        "return_pct":       roi_pct,
        "num_trades":       total_trades,
        "won_trades":       won_trades,
        "lost_trades":      lost_trades,
        "win_rate":         win_rate,
        "avg_won_pnl":      avg_won,
        "avg_lost_pnl":     avg_lost,
    }


def _safe_ta(ta: dict, *keys, default=float("nan")):
    """Safely traverse a nested TradeAnalyzer dict."""
    try:
        v = ta
        for k in keys:
            v = v[k]
        return float(v)
    except (KeyError, TypeError):
        return default


# ---------------------------------------------------------------------------
# FORECASTEX helpers
# ---------------------------------------------------------------------------

def _run_forecastex_loop(
    signals_df:     pd.DataFrame,
    forecasts_df:   pd.DataFrame,
    contract_value: float = 100.0,
    bid_ask_spread: float = 0.02,
) -> pd.DataFrame:
    """
    Simulate FORECASTEX prediction-market P&L.

    Model
    -----
    Each month we take a position proportional to signal strength.
    The "market price" (consensus) is the mean y_pred across all models —
    a proxy for what the market expects.
    P&L = signal × contract_value × (y_true − consensus)
          − |signal| × bid_ask_spread × contract_value

    This is a simplified model; a full implementation would require actual
    FORECASTEX contract pricing.

    Args:
        signals_df:     Strategy output — DatetimeIndex, signal column in [-1, 1]
        forecasts_df:   Full forecasts.csv (all models) for consensus computation
        contract_value: Notional value per contract unit
        bid_ask_spread: Proportional transaction cost per trade

    Returns:
        DataFrame with columns [signal, our_pred, y_true, consensus, pnl, cumulative_pnl]
    """
    selected = forecasts_df.set_index("date")[["y_true", "y_pred"]].rename(
        columns={"y_pred": "our_pred"}
    )
    consensus = forecasts_df.groupby("date")["y_pred"].mean().rename("consensus")

    df = signals_df.copy()
    df = df.join(selected,   how="left")
    df = df.join(consensus,  how="left")

    df["pnl"] = (
        df["signal"] * contract_value * (df["y_true"] - df["consensus"])
        - df["signal"].abs() * bid_ask_spread * contract_value
    )
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Unified backtesting engine for PCE-forecast-driven strategies.

    Args:
        initial_cash: Starting portfolio value for portfolio backtests.
        commission:   Round-trip commission as a fraction of trade value.
        ticker:       Primary security for portfolio backtests (e.g. "XLY").
    """

    def __init__(
        self,
        initial_cash: float = None,
        commission:   float = None,
        ticker:       str   = None,
    ):
        cfg = config.get("trading", {}).get("portfolio", {})
        self.initial_cash = initial_cash or cfg.get("initial_cash", 100_000)
        self.commission   = commission   or cfg.get("commission",   0.002)
        self.ticker       = ticker       or cfg.get("ticker",       "XLY")

    # ------------------------------------------------------------------
    # Portfolio backtest (backtrader)
    # ------------------------------------------------------------------

    def run_portfolio(
        self,
        strategy:    BaseStrategy,
        data:        StrategyData,
        output_dir:  Optional[Path] = None,
    ) -> dict:
        """
        Run a portfolio backtest using backtrader.

        Args:
            strategy:   Instantiated BaseStrategy.
            data:       StrategyData with at minimum forecasts + prices.
            output_dir: If provided, writes signals CSV and results JSON here.
        Returns:
            Dict of performance metrics: sharpe_ratio, return_pct, win_rate,
            max_drawdown_pct, num_trades, won_trades, lost_trades,
            avg_won_pnl, avg_lost_pnl, plus an "equity_curve" list.
        Raises:
            ImportError if backtrader is not installed.
            ValueError  if prices are missing from data.
        """
        if not _BT_AVAILABLE:
            raise ImportError(
                "The 'backtrader' package is required for portfolio backtests. "
                "Install it with: pip install backtrader"
            )
        if data.prices is None:
            raise ValueError(
                "run_portfolio() requires data.prices. "
                "Load market data via load.load_market_data() first."
            )

        print(f"[INFO] Running portfolio backtest: {strategy.name} on {self.ticker}")

        signals_df    = strategy.generate_signals(data)
        monthly_df    = _build_monthly_feed(data.prices, self.ticker, signals_df)

        MonthlyFeed   = _make_signal_feed_class()
        AdapterStrat  = _make_adapter_strategy_class()

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        feed = MonthlyFeed(dataname=monthly_df)
        cerebro.adddata(feed)

        cerebro.addstrategy(AdapterStrat)

        cerebro.addanalyzer(btanalyzers.SharpeRatio,   _name="sharpe",
                            riskfreerate=0.0, annualize=True, factor=12)
        cerebro.addanalyzer(btanalyzers.DrawDown,      _name="drawdown")
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(btanalyzers.Returns,       _name="returns",
                            tann=12)
        cerebro.addanalyzer(btanalyzers.TimeReturn,    _name="timereturn")

        start_val = cerebro.broker.getvalue()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thestrats = cerebro.run()
        end_val = cerebro.broker.getvalue()

        results = _extract_analyzers(thestrats)
        results["strategy"]        = strategy.name
        results["ticker"]          = self.ticker
        results["initial_cash"]    = start_val
        results["final_value"]     = end_val
        results["absolute_return"] = end_val - start_val

        # Equity curve from TimeReturn analyser
        try:
            tr = thestrats[0].analyzers.timereturn.get_analysis()
            equity_vals = (pd.Series(tr).add(1).cumprod() * start_val)
            results["equity_curve"] = equity_vals.to_dict()
        except Exception:
            results["equity_curve"] = {}

        if output_dir is not None:
            self._write_portfolio_results(results, signals_df, output_dir)

        return results

    # ------------------------------------------------------------------
    # FORECASTEX backtest
    # ------------------------------------------------------------------

    def run_forecastex(
        self,
        strategy:      BaseStrategy,
        data:          StrategyData,
        all_forecasts: pd.DataFrame,
        output_dir:    Optional[Path] = None,
    ) -> dict:
        """
        Simulate FORECASTEX prediction-market trading.

        Args:
            strategy:      Instantiated BaseStrategy.
            data:          StrategyData (only forecasts required).
            all_forecasts: Full forecasts.csv DataFrame (all models, for consensus).
            output_dir:    If provided, writes forecastex_trades.csv and results JSON.
        Returns:
            Dict: total_pnl, mean_monthly_pnl, win_rate, num_trades, final_cum_pnl.
        """
        cfg_fx  = config.get("trading", {}).get("forecastex", {})
        c_val   = cfg_fx.get("contract_value", 100.0)
        bid_ask = cfg_fx.get("bid_ask_spread",  0.02)

        print(f"[INFO] Running FORECASTEX simulation: {strategy.name}")

        signals_df = strategy.generate_signals(data)
        trades_df  = _run_forecastex_loop(signals_df, all_forecasts, c_val, bid_ask)

        active = trades_df[trades_df["signal"] != 0]
        wins   = (active["pnl"] > 0).sum()
        total  = len(active)

        results = {
            "strategy":         strategy.name,
            "total_pnl":        float(trades_df["pnl"].sum()),
            "mean_monthly_pnl": float(trades_df["pnl"].mean()),
            "win_rate":         float(wins / total) if total > 0 else float("nan"),
            "num_trades":       total,
            "final_cum_pnl":    float(trades_df["cumulative_pnl"].iloc[-1]),
        }

        if output_dir is not None:
            self._write_forecastex_results(results, trades_df, output_dir)

        return results

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_portfolio_results(
        self,
        results:    dict,
        signals_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = results["strategy"]

        signals_df.reset_index().to_csv(output_dir / f"signals_{name}.csv", index=False)

        scalar = {k: v for k, v in results.items() if k != "equity_curve"}
        with open(output_dir / f"backtest_results_{name}.json", "w") as f:
            json.dump(scalar, f, indent=2)

        if results.get("equity_curve"):
            eq = pd.Series(results["equity_curve"], name="equity")
            eq.to_csv(output_dir / f"equity_curve_{name}.csv")

        print(f"[INFO] Portfolio backtest results written to {output_dir}")

    def _write_forecastex_results(
        self,
        results:    dict,
        trades_df:  pd.DataFrame,
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = results["strategy"]

        trades_df.reset_index().to_csv(
            output_dir / f"forecastex_trades_{name}.csv", index=False
        )
        with open(output_dir / f"forecastex_results_{name}.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"[INFO] FORECASTEX results written to {output_dir}")
