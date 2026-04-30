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
        
        # Drop rows that don't overlap
        bbg_df = bbg_df.dropna(subset=['PX_LAST', 'BN_SURVEY_MEDIAN'])
        forecast_df = forecast_df.loc[bbg_df.index]

        # ==========================================
        # 1. DYNAMIC TARGET SCALING
        # ==========================================
        median_val = forecast_df['y_true'].abs().median()
        
        if median_val > 100.0:
            forecast_df['prev_y_true'] = forecast_df['y_true'].shift(1)
            forecast_df['pred_mom_pct'] = ((forecast_df['y_pred'] / forecast_df['prev_y_true']) - 1.0) * 100.0
            forecast_df['actual_mom_pct'] = ((forecast_df['y_true'] / forecast_df['prev_y_true']) - 1.0) * 100.0
        elif median_val < 1.0:
            forecast_df['pred_mom_pct'] = forecast_df['y_pred'] * 100.0
            forecast_df['actual_mom_pct'] = forecast_df['y_true'] * 100.0
        else:
            forecast_df['pred_mom_pct'] = forecast_df['y_pred']
            forecast_df['actual_mom_pct'] = forecast_df['y_true']

        forecast_df = forecast_df.dropna(subset=['pred_mom_pct', 'actual_mom_pct'])
            
        merged = forecast_df.join(bbg_df, how="inner")

        # ==========================================
        # 3. Z-SCORE BLOWOUT PREVENTER
        # ==========================================
        true_rmse = np.sqrt(((merged['pred_mom_pct'] - merged['actual_mom_pct']) ** 2).mean())
        
        # If the user's config RMSE is dangerously tight (< half of reality), it will cause a blowout.
        # We override it with the true error to safely price the options.
        active_rmse = true_rmse if (true_rmse > 0 and self.model_rmse < (true_rmse / 2)) else self.model_rmse

        print("\n" + "▼"*65)
        print("🔍 PREDICTION MARKET MATH DIAGNOSTIC")
        print(f"Configured RMSE in yaml:  {self.model_rmse}")
        print(f"True Historical RMSE:     {true_rmse:.4f}")
        print(f"Active Pricing RMSE:      {active_rmse:.4f} (Used for Z-Scores)")
        print("-" * 65)
        print("Date       | Pred (%) | Strike (%) | Z-Score | Prob Beat | Edge")
        print("-" * 65)

        signals = []
        confidences = []
        metadata = []

        for idx, (date, row) in enumerate(merged.iterrows()):
            y_pred_pct = float(row["pred_mom_pct"]) 
            strike_median = float(row["BN_SURVEY_MEDIAN"])
            actual_release = float(row["PX_LAST"])

            # Statistically sound Z-Score math
            z_score = (strike_median - y_pred_pct) / active_rmse
            prob_beat_strike = 1.0 - norm.cdf(z_score)
            calculated_edge = prob_beat_strike - self.contract_price

            # Print the X-Ray Diagnostic for the first 5 trades
            if idx < 5: 
                print(f"{date.date()} | {y_pred_pct:8.3f} | {strike_median:10.3f} | {z_score:7.2f} | {prob_beat_strike:9.3f} | {calculated_edge:5.3f}")

            if calculated_edge > self.edge_threshold:
                trade_signal = 1.0
            elif calculated_edge < -self.edge_threshold:
                trade_signal = -1.0
            else:
                trade_signal = 0.0

            signals.append(trade_signal)
            confidences.append(abs(calculated_edge))
            
            metadata.append({
                "y_pred_abs": round(float(row["y_pred"]), 2), 
                "y_pred_pct": round(y_pred_pct, 4),           
                "bbg_strike": strike_median,
                "bbg_actual": actual_release,
                "model_prob": round(prob_beat_strike, 4),
                "edge": round(calculated_edge, 4)
            })

        print("▲"*65 + "\n")

        return self._make_signals(
            index=merged.index,
            signal=signals,
            confidence=confidences,
            metadata=metadata
        )