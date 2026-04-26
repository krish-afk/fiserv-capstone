from __future__ import annotations

"""
PCE macro-conditioned pairs trading strategy.

Macro regime (monthly): PCE forecast sets the structural direction.
Spread entry/exit (daily): rolling z-score of the Engle-Granger residual triggers trades.

Hedge ratio is estimated once via OLS on the full price history and fixed for
the entire backtest — it is never re-estimated on a rolling basis.
"""

import numpy as np
import pandas as pd

from src.trading.strategy import BaseStrategy, StrategyData, register_strategy


@register_strategy
class PCEPairsStrategy(BaseStrategy):
    def __init__(
        self,
        ticker_a: str = "XLY",
        ticker_b: str = "XLP",
        macro_threshold: float = 0.0,
        zscore_window: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        position_size: float = 0.5,
        **params,
    ):
        super().__init__(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            macro_threshold=macro_threshold,
            zscore_window=zscore_window,
            entry_z=entry_z,
            exit_z=exit_z,
            position_size=position_size,
            **params,
        )
        self.ticker_a = ticker_a.upper()
        self.ticker_b = ticker_b.upper()
        self.macro_threshold = float(macro_threshold)
        self.zscore_window = int(zscore_window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.position_size = float(position_size)
        # Cached after first call to generate_signals; None until then.
        self._hedge_ratio: float | None = None

    @property
    def name(self) -> str:
        return f"pce_pairs_{self.ticker_a.lower()}_{self.ticker_b.lower()}"

    @property
    def required_inputs(self) -> set[str]:
        return {"forecasts", "prices"}

    @property
    def tickers(self) -> list[str]:
        return [self.ticker_a, self.ticker_b]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
        """OLS slope from regressing y on x (with intercept, Engle-Granger style)."""
        # np.polyfit returns [slope, intercept]; we only need the slope.
        return float(np.polyfit(x, y, 1)[0])

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)

        # --- 1. Price series ---
        price_a = self.get_price_series(data.prices, self.ticker_a)
        price_b = self.get_price_series(data.prices, self.ticker_b)

        # Align and drop any missing price observations.
        prices = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
        log_a = np.log(prices["a"])
        log_b = np.log(prices["b"])

        # --- 2. Hedge ratio (estimated once; fixed for the entire backtest) ---
        if self._hedge_ratio is None:
            self._hedge_ratio = self._ols_slope(log_b.values, log_a.values)
        hedge_ratio = self._hedge_ratio

        # --- 3. Spread: Engle-Granger residual ---
        spread = log_a - hedge_ratio * log_b

        # --- 4. Rolling z-score (uses only past data within the window) ---
        roll_mean = spread.rolling(self.zscore_window, min_periods=self.zscore_window).mean()
        roll_std = spread.rolling(self.zscore_window, min_periods=self.zscore_window).std()
        z_score = (spread - roll_mean) / roll_std.where(roll_std > 0)

        # --- 5. Forward-fill monthly forecasts to daily frequency ---
        # Monthly y_pred is forward-filled to cover each trading day.
        # Days before the first forecast observation become NaN and are dropped below.
        # The union+ffill pattern ensures ffill propagates across month boundaries.
        forecasts = self._coerce_forecast_index(data.forecasts)[["y_pred"]]
        daily_macro = (
            forecasts
            .reindex(prices.index.union(forecasts.index))
            .sort_index()
            .ffill()
            .reindex(prices.index)
        )

        # --- 6. Inner join: only dates where spread, z-score, and forecast all exist ---
        # (z-score NaN for the first zscore_window days; forecast NaN before first obs)
        work = pd.DataFrame(
            {
                "spread": spread,
                "z_score": z_score,
                "y_pred": daily_macro["y_pred"],
            }
        ).dropna()

        if work.empty:
            return self._make_weight_frame(
                index=pd.DatetimeIndex([], name="date"),
                weights={self.ticker_a: [], self.ticker_b: []},
            )

        # --- 7. Macro regime (binary threshold) ---
        preds = work["y_pred"].values
        macro_regime = np.where(
            preds > self.macro_threshold,
            "bullish",
            np.where(preds < -self.macro_threshold, "bearish", "neutral"),
        )

        # --- 8. State machine: entry / exit ---
        n = len(work)
        weight_a = np.zeros(n, dtype=float)
        weight_b = np.zeros(n, dtype=float)
        active = np.zeros(n, dtype=bool)

        position = 0        # 0 = flat, 1 = long spread, -1 = short spread
        entry_regime = None
        zs = work["z_score"].values

        for i in range(n):
            z = zs[i]
            regime = macro_regime[i]

            # Exit: mean reversion or macro flip
            if position != 0:
                if abs(z) < self.exit_z or regime != entry_regime:
                    position = 0
                    entry_regime = None

            # Entry: only enter from flat
            if position == 0:
                if z < -self.entry_z and regime == "bullish":
                    position = 1          # long A / short B
                    entry_regime = regime
                elif z > self.entry_z and regime == "bearish":
                    position = -1         # short A / long B
                    entry_regime = regime

            if position == 1:
                weight_a[i] = self.position_size
                weight_b[i] = -self.position_size
                active[i] = True
            elif position == -1:
                weight_a[i] = -self.position_size
                weight_b[i] = self.position_size
                active[i] = True

        # --- 9. Metadata ---
        spreads = work["spread"].values
        metadata = [
            {
                "z_score": round(float(zs[i]), 6),
                "spread_value": round(float(spreads[i]), 6),
                "macro_regime": str(macro_regime[i]),
                "hedge_ratio": round(hedge_ratio, 6),
                "active": bool(active[i]),
            }
            for i in range(n)
        ]

        return self._make_weight_frame(
            index=work.index,
            weights={self.ticker_a: weight_a, self.ticker_b: weight_b},
            metadata=metadata,
        )
