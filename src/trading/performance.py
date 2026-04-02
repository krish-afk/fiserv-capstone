# src/trading/performance.py
"""
Performance metrics and sensitivity analysis for trading simulations.

Two responsibilities
--------------------
1. compute_metrics()
   Derive Sharpe ratio, annualised ROI, win/loss ratio, max drawdown, and
   additional statistics from a backtest results dict or equity curve.
   Thin wrappers around backtrader analyser outputs — use these to produce
   the final numbers table for reporting.

2. sensitivity_analysis()
   Noise-injection experiment: answer "how much does trading performance
   degrade as PCE forecast accuracy declines?"

   Method:
     - Take the best forecast series (y_pred).
     - Add Gaussian noise with increasing σ → increasing RMSE.
     - Re-run the strategy and backtest for each noise level.
     - Return a DataFrame: rmse | sharpe_mean | sharpe_std | roi_mean | win_rate_mean | …

   This is academically defensible and directly answers whether the strategy's
   edge is a function of forecast quality or simply market beta.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.trading.strategy import BaseStrategy, StrategyData
from src.trading.backtest import BacktestEngine


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> float:
    """
    Annualised Sharpe ratio from a periodic return series.

    Args:
        returns:           Period returns (not cumulative), e.g. monthly.
        risk_free_rate:    Annualised risk-free rate as a decimal.
        periods_per_year:  12 for monthly, 252 for daily.
    Returns:
        Annualised Sharpe ratio, or NaN if returns have zero variance.
    """
    excess = returns - (risk_free_rate / periods_per_year)
    std    = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a positive percentage.

    Args:
        equity_curve: Portfolio value series (not returns).
    Returns:
        Max drawdown as a positive percentage, e.g. 15.3 means −15.3 %.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(-drawdown.min() * 100)


def win_loss_ratio(pnl_series: pd.Series) -> float:
    """
    Ratio of winning trades to losing trades.

    Args:
        pnl_series: Per-trade P&L values.
    Returns:
        win / loss count ratio, or inf if no losses, or NaN if no trades.
    """
    wins   = (pnl_series > 0).sum()
    losses = (pnl_series < 0).sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def compute_metrics(results: dict, equity_curve: pd.Series = None) -> dict:
    """
    Assemble a reporting-ready metrics dict from a BacktestEngine results dict.

    Backtrader analysers already compute most of these; this function normalises
    field names and fills in any gaps (e.g. win/loss ratio) for reporting.

    Args:
        results:      Dict returned by BacktestEngine.run_portfolio().
        equity_curve: Optional pd.Series of portfolio values for drawdown recomputation.
                      If None, uses results["max_drawdown_pct"] as-is.
    Returns:
        Flat dict ready for display or CSV output.
    """
    def _get(key, default=float("nan")):
        v = results.get(key, default)
        return float(v) if v is not None else default

    won  = _get("won_trades",  0)
    lost = _get("lost_trades", 0)
    wl   = (won / lost) if lost > 0 else (float("inf") if won > 0 else float("nan"))

    metrics = {
        "strategy":          results.get("strategy", ""),
        "ticker":            results.get("ticker",   ""),
        "sharpe_ratio":      _get("sharpe_ratio"),
        "return_pct":        _get("return_pct"),
        "absolute_return":   _get("absolute_return"),
        "max_drawdown_pct":  _get("max_drawdown_pct"),
        "win_rate":          _get("win_rate"),
        "win_loss_ratio":    wl,
        "num_trades":        _get("num_trades"),
        "avg_won_pnl":       _get("avg_won_pnl"),
        "avg_lost_pnl":      _get("avg_lost_pnl"),
        "profit_factor":     abs(_get("avg_won_pnl") * won / (_get("avg_lost_pnl") * lost))
                             if (won > 0 and lost > 0) else float("nan"),
    }

    # Recompute drawdown from equity curve if provided (more precise than backtrader estimate)
    if equity_curve is not None and len(equity_curve) > 1:
        metrics["max_drawdown_pct"] = max_drawdown(equity_curve)

    return metrics


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    strategy:        BaseStrategy,
    data:            StrategyData,
    engine:          BacktestEngine,
    noise_sigmas:    list  = None,
    n_trials:        int   = 50,
    random_seed:     int   = 42,
) -> pd.DataFrame:
    """
    Measure how trading performance degrades as PCE forecast accuracy declines.

    Method
    ------
    For each noise level σ in noise_sigmas:
      1. Generate n_trials independent noisy forecast series:
             y_pred_noisy = y_pred + N(0, σ²)
      2. Run run_portfolio() for each noisy series → collect Sharpe, ROI, win rate.
      3. Also compute the RMSE of each noisy series against y_true — this is the
         x-axis of the sensitivity curve.
      4. Aggregate trials → mean ± std per noise level.

    Output can be plotted as: RMSE → Sharpe ratio, with error bands.

    Args:
        strategy:     Instantiated BaseStrategy whose signals will be run.
        data:         StrategyData — data.forecasts must have y_true and y_pred.
                      data.prices must be present (passed through to engine).
        engine:       Configured BacktestEngine instance.
        noise_sigmas: List of σ values to sweep.  Default: [0.0, 0.1, 0.2, 0.5, 1.0, 2.0].
        n_trials:     Number of independent noisy realisations per σ level.
        random_seed:  Base RNG seed for reproducibility (each trial uses seed + i).
    Returns:
        DataFrame with one row per (sigma, trial), columns:
            sigma | trial | rmse | sharpe | return_pct | win_rate | num_trades
        Aggregate summary (mean ± std across trials) can be computed with .groupby("sigma").
    """
    if noise_sigmas is None:
        noise_sigmas = config.get("trading", {}) \
                             .get("sensitivity_analysis", {}) \
                             .get("noise_sigmas", [0.0, 0.1, 0.2, 0.5, 1.0, 2.0])

    base_forecasts = data.forecasts.copy().sort_values("date")
    y_pred_base    = base_forecasts["y_pred"].values
    y_true         = base_forecasts["y_true"].values

    rows = []

    for sigma in noise_sigmas:
        for trial in range(n_trials):
            rng   = np.random.default_rng(random_seed + trial)
            noise = rng.normal(0.0, sigma, size=len(y_pred_base)) if sigma > 0 else np.zeros(len(y_pred_base))

            noisy_preds = y_pred_base + noise
            rmse        = float(np.sqrt(np.mean((noisy_preds - y_true) ** 2)))

            # Build a StrategyData with the noisy forecast series
            noisy_forecasts = base_forecasts.copy()
            noisy_forecasts["y_pred"] = noisy_preds
            noisy_data = StrategyData(
                forecasts = noisy_forecasts,
                prices    = data.prices,
                macro     = data.macro,
                mrts      = data.mrts,
            )

            try:
                results = engine.run_portfolio(strategy, noisy_data, output_dir=None)
            except Exception as e:
                print(f"[WARN] sensitivity_analysis sigma={sigma} trial={trial} failed: {e}")
                results = {}

            rows.append({
                "sigma":      sigma,
                "trial":      trial,
                "rmse":       rmse,
                "sharpe":     results.get("sharpe_ratio",  float("nan")),
                "return_pct": results.get("return_pct",    float("nan")),
                "win_rate":   results.get("win_rate",      float("nan")),
                "num_trades": results.get("num_trades",    float("nan")),
            })

    return pd.DataFrame(rows)


def summarise_sensitivity(sens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sensitivity_analysis() output to mean ± std per sigma level.

    Args:
        sens_df: Output of sensitivity_analysis().
    Returns:
        DataFrame indexed by sigma with columns:
            rmse_mean | sharpe_mean | sharpe_std | return_mean | return_std |
            win_rate_mean | win_rate_std
    """
    agg = sens_df.groupby("sigma").agg(
        rmse_mean     =("rmse",       "mean"),
        sharpe_mean   =("sharpe",     "mean"),
        sharpe_std    =("sharpe",     "std"),
        return_mean   =("return_pct", "mean"),
        return_std    =("return_pct", "std"),
        win_rate_mean =("win_rate",   "mean"),
        win_rate_std  =("win_rate",   "std"),
    )
    return agg
