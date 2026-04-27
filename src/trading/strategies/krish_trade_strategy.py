from __future__ import annotations

"""
Krish-style monthly consumer regime strategy.

This ports the standalone krish-trade logic into the src/trading framework so it
can be selected from config.yaml like any other strategy module.
"""

import numpy as np
import pandas as pd

from src.trading.strategy import BaseStrategy, StrategyData, register_strategy


@register_strategy
class KrishTradeStrategy(BaseStrategy):

    DISPLAY_NAME = "Krish Trade Strategy"
    DESCRIPTION = "Allocates between risk-on and defensive consumer names using combined PCE + MRTS forecasts."
    REQUIRED_INPUTS_SCHEMA = ["forecasts", "mrts"]
    PARAMETER_SCHEMA = [
        {
            "name": "bullish_threshold",
            "label": "Bullish Threshold",
            "type": "number",
            "default": 0.75,
            "required": True,
            "step": 0.01,
        },
        {
            "name": "bearish_threshold",
            "label": "Bearish Threshold",
            "type": "number",
            "default": -0.75,
            "required": True,
            "step": 0.01,
        },
        {
            "name": "pce_weight",
            "label": "PCE Weight",
            "type": "number",
            "default": 0.6,
            "required": True,
            "step": 0.01,
        },
        {
            "name": "mrts_weight",
            "label": "MRTS Weight",
            "type": "number",
            "default": 0.4,
            "required": True,
            "step": 0.01,
        },
        {
            "name": "target_allocation",
            "label": "Target Allocation",
            "type": "number",
            "default": 0.25,
            "required": True,
            "step": 0.01,
        },
        {
            "name": "risk_on_ticker",
            "label": "Risk-On Ticker",
            "type": "ticker",
            "default": "XLY",
            "required": True,
        },
        {
            "name": "defensive_ticker",
            "label": "Defensive Ticker",
            "type": "ticker",
            "default": "XLP",
            "required": True,
        },
    ]
    UI_SPEC = {
        "market_type": "securities",
        "plots": [
            "forecast_vs_actual",
            "forecast_error",
            "equity_curve",
            "cumulative_return_curve",
            "drawdown_curve",
            "period_return_curve",
            "weights_curve",
            "confidence_curve",
        ],
    }
    def __init__(
        self,
        bullish_threshold: float = 0.75,
        bearish_threshold: float = -0.75,
        pce_weight: float = 0.6,
        mrts_weight: float = 0.4,
        target_allocation: float = 0.25,
        risk_on_ticker: str = "XLY",
        defensive_ticker: str = "XLP",
        **params,
    ):
        super().__init__(
            bullish_threshold=bullish_threshold,
            bearish_threshold=bearish_threshold,
            pce_weight=pce_weight,
            mrts_weight=mrts_weight,
            target_allocation=target_allocation,
            risk_on_ticker=risk_on_ticker,
            defensive_ticker=defensive_ticker,
            **params,
        )
        self.bullish_threshold = float(bullish_threshold)
        self.bearish_threshold = float(bearish_threshold)
        self.pce_weight = float(pce_weight)
        self.mrts_weight = float(mrts_weight)
        self.target_allocation = float(target_allocation)
        self.risk_on_ticker = risk_on_ticker.upper()
        self.defensive_ticker = defensive_ticker.upper()

    @property
    def name(self) -> str:
        return "krish_trade"

    @property
    def required_inputs(self) -> set[str]:
        return {"forecasts", "mrts"}

    @property
    def tickers(self) -> list[str]:
        return [self.risk_on_ticker, self.defensive_ticker]

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)

        pce = self._coerce_forecast_index(data.forecasts)[["y_pred"]].rename(
            columns={"y_pred": "pce_pred"}
        )
        mrts = self._coerce_forecast_index(data.mrts)[["y_pred"]].rename(
            columns={"y_pred": "mrts_pred"}
        )

        df = pce.join(mrts, how="inner")
        if df.empty:
            raise ValueError("No overlapping dates found between primary forecasts and MRTS forecasts.")

        combined_score = (self.pce_weight * df["pce_pred"]) + (self.mrts_weight * df["mrts_pred"])

        long_consumer = combined_score > self.bullish_threshold
        defensive_consumer = combined_score < self.bearish_threshold

        risk_on_weight = np.where(long_consumer, self.target_allocation, 0.0)
        defensive_weight = np.where(defensive_consumer, self.target_allocation, 0.0)
        cash_weight = 1.0 - risk_on_weight - defensive_weight

        regime = np.where(
            long_consumer,
            "bullish_consumer",
            np.where(defensive_consumer, "bearish_consumer", "neutral"),
        )
        action = np.where(
            long_consumer,
            f"BUY {self.risk_on_ticker}",
            np.where(defensive_consumer, f"BUY {self.defensive_ticker}", "NO TRADE"),
        )

        confidence = np.clip(np.abs(combined_score) / max(abs(self.bullish_threshold), abs(self.bearish_threshold), 1e-8), 0.0, 1.0)

        metadata = [
            {
                "pce_pred": round(float(pce_pred), 6),
                "mrts_pred": round(float(mrts_pred), 6),
                "combined_score": round(float(score), 6),
                "regime": str(reg),
                "action": str(act),
            }
            for pce_pred, mrts_pred, score, reg, act in zip(
                df["pce_pred"],
                df["mrts_pred"],
                combined_score,
                regime,
                action,
            )
        ]

        return self._make_weight_frame(
            index=df.index,
            weights={
                self.risk_on_ticker: risk_on_weight,
                self.defensive_ticker: defensive_weight,
            },
            cash_weight=cash_weight,
            confidence=confidence,
            metadata=metadata,
        )