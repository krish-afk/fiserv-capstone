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
    # Private helpers
    # ------------------------------------------------------------------

    def _get_log_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Fetch close prices for both tickers, align, and return log-transformed."""
        price_a = self.get_price_series(prices, self.ticker_a)
        price_b = self.get_price_series(prices, self.ticker_b)
        raw = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
        return np.log(raw)

    def _build_spread(self, log_prices: pd.DataFrame) -> pd.Series:
        """Compute the Engle-Granger spread, estimating the hedge ratio once."""
        if self._hedge_ratio is None:
            # OLS: regress log_a on log_b with intercept; slope is the hedge ratio.
            self._hedge_ratio = float(
                np.polyfit(log_prices["b"].values, log_prices["a"].values, 1)[0]
            )
        return log_prices["a"] - self._hedge_ratio * log_prices["b"]

    def _rolling_zscore(self, spread: pd.Series) -> pd.Series:
        """Rolling z-score using only past data within the configured window."""
        roll = spread.rolling(self.zscore_window, min_periods=self.zscore_window)
        std = roll.std()
        return (spread - roll.mean()) / std.where(std > 0)

    def _align_macro(self, forecasts: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.Series:
        """Forward-fill monthly PCE forecasts onto a daily index.

        The union+ffill pattern propagates values across month boundaries.
        Days before the first forecast observation remain NaN and are dropped
        when the caller does the inner join.
        """
        monthly = self._coerce_forecast_index(forecasts)[["y_pred"]]
        return (
            monthly
            .reindex(daily_index.union(monthly.index))
            .sort_index()
            .ffill()
            .reindex(daily_index)
            ["y_pred"]
        )

    def _classify_regime(self, y_pred: np.ndarray) -> np.ndarray:
        """Map forecast values to 'bullish' / 'bearish' / 'neutral' regime labels."""
        return np.where(
            y_pred > self.macro_threshold, "bullish",
            np.where(y_pred < -self.macro_threshold, "bearish", "neutral"),
        )

    def _run_state_machine(
        self, zs: np.ndarray, macro_regime: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Entry/exit loop; returns (weight_a, weight_b, active) arrays."""
        n = len(zs)
        weight_a = np.zeros(n, dtype=float)
        weight_b = np.zeros(n, dtype=float)
        active = np.zeros(n, dtype=bool)

        position = 0        # 0 = flat, 1 = long spread, -1 = short spread
        entry_regime = None

        for i in range(n):
            z, regime = zs[i], macro_regime[i]

            # Exit: mean reversion or macro regime flip since entry.
            if position != 0:
                if abs(z) < self.exit_z or regime != entry_regime:
                    position = 0
                    entry_regime = None

            # Entry: only from flat.
            if position == 0:
                if z < -self.entry_z and regime == "bullish":
                    position, entry_regime = 1, regime   # long A / short B
                elif z > self.entry_z and regime == "bearish":
                    position, entry_regime = -1, regime  # short A / long B

            if position == 1:
                weight_a[i], weight_b[i], active[i] = self.position_size, -self.position_size, True
            elif position == -1:
                weight_a[i], weight_b[i], active[i] = -self.position_size, self.position_size, True

        return weight_a, weight_b, active

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)

        log_prices = self._get_log_prices(data.prices)
        spread = self._build_spread(log_prices)
        z_score = self._rolling_zscore(spread)
        daily_macro = self._align_macro(data.forecasts, log_prices.index)

        # Inner join: drop rows where z-score or forecast is unavailable.
        work = pd.DataFrame(
            {"spread": spread, "z_score": z_score, "y_pred": daily_macro}
        ).dropna()

        if work.empty:
            return self._make_weight_frame(
                index=pd.DatetimeIndex([], name="date"),
                weights={self.ticker_a: [], self.ticker_b: []},
            )

        macro_regime = self._classify_regime(work["y_pred"].values)
        weight_a, weight_b, active = self._run_state_machine(
            work["z_score"].values, macro_regime
        )

        metadata = [
            {
                "z_score": round(float(work["z_score"].iat[i]), 6),
                "spread_value": round(float(work["spread"].iat[i]), 6),
                "macro_regime": str(macro_regime[i]),
                "hedge_ratio": round(self._hedge_ratio, 6),
                "active": bool(active[i]),
            }
            for i in range(len(work))
        ]

        return self._make_weight_frame(
            index=work.index,
            weights={self.ticker_a: weight_a, self.ticker_b: weight_b},
            metadata=metadata,
        )
