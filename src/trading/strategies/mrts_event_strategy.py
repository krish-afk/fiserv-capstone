from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
from typing import Any

from src.trading.strategy import BaseStrategy, StrategyData, register_strategy

@register_strategy
class MRTSForecastMarketStrategy(BaseStrategy):
    DISPLAY_NAME = "MRTS Forecast Market Strategy"
    DESCRIPTION = "Trades a prediction-market-style event contract using model edge versus Bloomberg survey median."
    REQUIRED_INPUTS_SCHEMA = ["forecasts"]
    PARAMETER_SCHEMA = [
        {
            "name": "bloomberg_csv_path",
            "label": "Bloomberg CSV Path",
            "type": "text",
            "default": "",
            "required": True,
            "placeholder": "/absolute/or/project/relative/path.csv",
        },
        {
            "name": "model_rmse",
            "label": "Model RMSE",
            "type": "number",
            "default": 0.15,
            "required": True,
            "step": 0.01,
        },
        {
            "name": "edge_threshold",
            "label": "Edge Threshold",
            "type": "number",
            "default": 0.15,
            "required": True,
            "step": 0.01,
        },
        {
            "name": "contract_price",
            "label": "Contract Price",
            "type": "number",
            "default": 0.5,
            "required": True,
            "step": 0.01,
        },
    ]
    UI_SPEC = {
        "market_type": "prediction_market",
        "plots": [
            "forecast_vs_actual",
            "forecast_error",
            "confidence_curve",
            "edge_curve",
            "probability_curve",
        ],
    }
    def __init__(
        self,
        bloomberg_csv_path: str,
        model_rmse: float = 0.15,
        edge_threshold: float = 0.15,
        contract_price: float = 0.50,
        **params: Any,
    ):
        super().__init__(
            bloomberg_csv_path=bloomberg_csv_path,
            model_rmse=model_rmse,
            edge_threshold=edge_threshold,
            contract_price=contract_price,
            **params
        )
        self.bbg_path = Path(bloomberg_csv_path)
        self.model_rmse = float(model_rmse)
        self.edge_threshold = float(edge_threshold)
        self.contract_price = float(contract_price)

    @property
    def name(self) -> str:
        return "mrts_forecast_market"

    @property
    def required_inputs(self) -> set[str]:
        return {"forecasts"}

    @property
    def tickers(self) -> list[str]:
        # Returning an empty list prevents Will's pipeline from trying to 
        # download YFinance equity data for our Prediction Market.
        return []

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)
        
        forecast_df = self._coerce_forecast_index(data.forecasts)

        if not self.bbg_path.exists():
            raise FileNotFoundError(f"Bloomberg CSV missing at {self.bbg_path}")
            
        bbg_df = pd.read_csv(self.bbg_path, skiprows=5)
        bbg_df["date"] = pd.to_datetime(bbg_df["Date"])
        bbg_df = bbg_df.set_index("date")

        # Align and Forward-Fill Bloomberg Data to Match Forecast Dates
        forecast_df = forecast_df.sort_index(ascending=True)
        bbg_df = bbg_df.sort_index(ascending=True)
        bbg_df = bbg_df.reindex(forecast_df.index, method='ffill')
        
        # Drop rows that don't overlap and sync indices
        bbg_df = bbg_df.dropna(subset=['PX_LAST', 'BN_SURVEY_MEDIAN'])
        forecast_df = forecast_df.loc[bbg_df.index]

        # ==========================================
        # TRANSLATE ABSOLUTE TO MoM % CHANGE
        # ==========================================
        # Grab the previous month's actual reported value
        forecast_df['prev_y_true'] = forecast_df['y_true'].shift(1)
        
        # Formula: ((Prediction / Previous Actual) - 1) * 100
        forecast_df['pred_mom_pct'] = ((forecast_df['y_pred'] / forecast_df['prev_y_true']) - 1.0) * 100.0
        
        # The first row won't have a previous month, so we must drop it
        forecast_df = forecast_df.dropna(subset=['pred_mom_pct'])

        # Merge for signal generation
        merged = forecast_df.join(bbg_df, how="inner")

        signals = []
        confidences = []
        metadata = []

        for date, row in merged.iterrows():
            # Use our new PERCENTAGE prediction for the math!
            y_pred_pct = float(row["pred_mom_pct"]) 
            strike_median = float(row["BN_SURVEY_MEDIAN"])
            actual_release = float(row["PX_LAST"])

            # Now the loc (prediction) and strike are on the exact same scale
            prob_beat_strike = 1.0 - norm.cdf(strike_median, loc=y_pred_pct, scale=self.model_rmse)
            calculated_edge = prob_beat_strike - self.contract_price

            if calculated_edge > self.edge_threshold:
                trade_signal = 1.0
            elif calculated_edge < -self.edge_threshold:
                trade_signal = -1.0
            else:
                trade_signal = 0.0

            signals.append(trade_signal)
            confidences.append(abs(calculated_edge))
            
            metadata.append({
                "y_pred_abs": round(float(row["y_pred"]), 2), # Keep absolute for records
                "y_pred_pct": round(y_pred_pct, 4),           # The traded %
                "bbg_strike": strike_median,
                "bbg_actual": actual_release,
                "model_prob": round(prob_beat_strike, 4),
                "edge": round(calculated_edge, 4)
            })

        return self._make_signals(
            index=merged.index,
            signal=signals,
            confidence=confidences,
            metadata=metadata
        )