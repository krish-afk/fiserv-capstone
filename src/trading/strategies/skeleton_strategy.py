from __future__ import annotations

"""
Starter template for team-written strategy files.

How to use
----------
1. Copy this file.
2. Rename the class and file.
3. Set trading.strategy.file and trading.strategy.class in config.yaml.
4. Add any params you want under trading.strategy.params.

Expected return format
----------------------
Return a DataFrame indexed by date with one or more weight__TICKER columns.
Example columns:
    weight__SPY, weight__TLT, cash_weight, confidence, metadata
"""

import numpy as np
import pandas as pd

from src.trading.strategy import BaseStrategy, StrategyData, register_strategy


@register_strategy
class SkeletonStrategy(BaseStrategy):
    """
    Example strategy that rotates between a risk-on ETF and a defensive ETF
    based on the sign/size of the forecast.

    Replace this logic with whatever your team wants.
    """

    def __init__(
        self,
        risk_on_ticker: str = "SPY",
        defensive_ticker: str = "TLT",
        threshold: float = 0.0,
        target_allocation: float = 0.50,
        use_prices: bool = False,
        **params,
    ):
        super().__init__(
            risk_on_ticker=risk_on_ticker,
            defensive_ticker=defensive_ticker,
            threshold=threshold,
            target_allocation=target_allocation,
            use_prices=use_prices,
            **params,
        )
        self.risk_on_ticker = risk_on_ticker.upper()
        self.defensive_ticker = defensive_ticker.upper()
        self.threshold = float(threshold)
        self.target_allocation = float(target_allocation)
        self.use_prices = bool(use_prices)

    @property
    def name(self) -> str:
        return "skeleton_strategy"

    @property
    def required_inputs(self) -> set[str]:
        # Add "prices", "macro", or "mrts" if your strategy needs them.
        required = {"forecasts"}
        if self.use_prices:
            required.add("prices")
        return required

    @property
    def tickers(self) -> list[str]:
        return [self.risk_on_ticker, self.defensive_ticker]

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)
        forecasts = self._coerce_forecast_index(data.forecasts)

        # --- your core alpha logic goes here ---
        score = forecasts["y_pred"].astype(float)

        if self.use_prices:
            # Example of how a strategy can access market data safely.
            risk_on_close = self.get_price_series(data.prices, self.risk_on_ticker, "close")
            defensive_close = self.get_price_series(data.prices, self.defensive_ticker, "close")

            # Example filter: only allow risk-on if its last close is above defensive ETF's last close ratio trend.
            relative_strength = (risk_on_close / defensive_close).reindex(forecasts.index, method="ffill")
            score = score.where(relative_strength.notna(), other=score)

        risk_on_weight = np.where(score > self.threshold, self.target_allocation, 0.0)
        defensive_weight = np.where(score <= self.threshold, self.target_allocation, 0.0)
        cash_weight = 1.0 - risk_on_weight - defensive_weight
        confidence = np.clip(np.abs(score) / max(abs(self.threshold), 1.0), 0.0, 1.0)

        metadata = [
            {
                "forecast": round(float(pred), 6),
                "threshold": self.threshold,
                "risk_on": self.risk_on_ticker,
                "defensive": self.defensive_ticker,
            }
            for pred in score
        ]

        return self._make_weight_frame(
            index=forecasts.index,
            weights={
                self.risk_on_ticker: risk_on_weight,
                self.defensive_ticker: defensive_weight,
            },
            cash_weight=cash_weight,
            confidence=confidence,
            metadata=metadata,
        )