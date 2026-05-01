from __future__ import annotations

import numpy as np
import pandas as pd

from src.trading.strategy import BaseStrategy, StrategyData, register_strategy


@register_strategy
class AdaptiveRegimeStrategy(BaseStrategy):
    """
    Forecast-led adaptive ETF rotation strategy.

    Logic
    -----
    1. Use the primary forecast panel's `y_pred` as the macro score.
    2. Classify each month into one of four regimes using rolling z-scores of:
       - forecast gap vs trailing trend
       - month-over-month change in the forecast
       - forecast uncertainty proxy (rolling forecast error volatility)
    3. Inside the chosen regime basket, rank ETFs by weighted momentum with a
       volatility penalty.
    4. Hold the top-ranked ETF (top_k=1 by default) and park unused capital in cash.

    This matches the "best adaptive" idea from the notebook, but is packaged to
    run inside the team's trading pipeline.
    """

    DISPLAY_NAME = "Adaptive Regime ETF Rotation"
    DESCRIPTION = "Uses forecast regimes plus ETF momentum ranking to rotate across risk-on, defensive, duration, and uncertain baskets."
    REQUIRED_INPUTS_SCHEMA = ["forecasts", "prices"]

    PARAMETER_SCHEMA = [
        {"name": "trend_window", "label": "Trend Window", "type": "number", "default": 3, "required": True, "step": 1},
        {"name": "z_window", "label": "Z-Score Window", "type": "number", "default": 4, "required": True, "step": 1},
        {"name": "uncertainty_window", "label": "Uncertainty Window", "type": "number", "default": 3, "required": True, "step": 1},
        {"name": "target_allocation", "label": "Target Allocation", "type": "number", "default": 1.0, "required": True, "step": 0.05},
        {"name": "top_k", "label": "Top K ETFs", "type": "number", "default": 2, "required": True, "step": 1},
        {"name": "fallback_etf", "label": "Fallback ETF", "type": "ticker", "default": "SHY", "required": True},
        {"name": "risk_on_basket", "label": "Risk-On Basket", "type": "ticker_list", "default": ["XLY", "XRT", "PEJ", "AMZN", "BKNG", "TSLA"], "required": True},
        {"name": "defensive_basket", "label": "Defensive Basket", "type": "ticker_list", "default": ["XLP", "XLV", "XLU", "USMV", "QUAL"], "required": True},
        {"name": "duration_basket", "label": "Duration Basket", "type": "ticker_list", "default": ["TLT", "IEF", "XLP", "XLU", "XLV"], "required": True},
        {"name": "uncertain_basket", "label": "Uncertain Basket", "type": "ticker_list", "default": ["QUAL", "USMV", "XLV", "XLP", "GLD"], "required": True},
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
        trend_window: int = 6,
        z_window: int = 12,
        uncertainty_window: int = 6,
        risk_on_gap_z: float = 0.4,
        risk_on_delta_z: float = 0.0,
        defensive_gap_z: float = 0.0,
        duration_gap_z: float = -0.05,
        falling_delta_z: float = -0.2,
        uncertainty_ratio_thresh: float = 2.0,
        ambiguous_band: float = 0.05,
        top_k: int = 2,
        min_abs_mom: float = 0.0,
        use_ma_filter: bool = False,
        target_allocation: float = 1.0,
        score_w_3m: float = 1.0,
        score_w_6m: float = 0.0,
        score_w_12m: float = 0.0,
        score_w_vol: float = 0.15,
        fallback_etf: str = "QUAL",
        risk_on_basket: list[str] | None = None,
        defensive_basket: list[str] | None = None,
        duration_basket: list[str] | None = None,
        uncertain_basket: list[str] | None = None,
        **params,
    ):
        super().__init__(
            trend_window=trend_window,
            z_window=z_window,
            uncertainty_window=uncertainty_window,
            risk_on_gap_z=risk_on_gap_z,
            risk_on_delta_z=risk_on_delta_z,
            defensive_gap_z=defensive_gap_z,
            duration_gap_z=duration_gap_z,
            falling_delta_z=falling_delta_z,
            uncertainty_ratio_thresh=uncertainty_ratio_thresh,
            ambiguous_band=ambiguous_band,
            top_k=top_k,
            min_abs_mom=min_abs_mom,
            use_ma_filter=use_ma_filter,
            target_allocation=target_allocation,
            score_w_3m=score_w_3m,
            score_w_6m=score_w_6m,
            score_w_12m=score_w_12m,
            score_w_vol=score_w_vol,
            fallback_etf=fallback_etf,
            risk_on_basket=risk_on_basket,
            defensive_basket=defensive_basket,
            duration_basket=duration_basket,
            uncertain_basket=uncertain_basket,
            **params,
        )

        self.trend_window = int(trend_window)
        self.z_window = int(z_window)
        self.uncertainty_window = int(uncertainty_window)
        self.risk_on_gap_z = float(risk_on_gap_z)
        self.risk_on_delta_z = float(risk_on_delta_z)
        self.defensive_gap_z = float(defensive_gap_z)
        self.duration_gap_z = float(duration_gap_z)
        self.falling_delta_z = float(falling_delta_z)
        self.uncertainty_ratio_thresh = float(uncertainty_ratio_thresh)
        self.ambiguous_band = float(ambiguous_band)
        self.top_k = max(1, int(top_k))
        self.min_abs_mom = float(min_abs_mom)
        self.use_ma_filter = bool(use_ma_filter)
        self.target_allocation = float(target_allocation)
        self.score_w_3m = float(score_w_3m)
        self.score_w_6m = float(score_w_6m)
        self.score_w_12m = float(score_w_12m)
        self.score_w_vol = float(score_w_vol)
        self.fallback_etf = fallback_etf.upper()

        self.baskets = {
            "RISK_ON": [t.upper() for t in (risk_on_basket or ["XLY", "XRT", "PEJ", "AMZN", "BKNG", "TSLA"])],
            "DEFENSIVE": [t.upper() for t in (defensive_basket or ["XLP", "XLV", "XLU", "USMV", "QUAL"])],
            "DURATION": [t.upper() for t in (duration_basket or ["TLT", "IEF", "XLP", "XLU", "XLV"])],
            "UNCERTAIN": [t.upper() for t in (uncertain_basket or ["QUAL", "USMV", "XLV", "XLP", "GLD"])],
        }

    @property
    def name(self) -> str:
        return "adaptive_regime_strategy"

    @property
    def required_inputs(self) -> set[str]:
        return {"forecasts", "prices"}

    @property
    def tickers(self) -> list[str]:
        all_tickers = set()
        for basket in self.baskets.values():
            all_tickers.update(basket)
        all_tickers.add(self.fallback_etf)
        return sorted(all_tickers)

    def _rolling_z(self, s: pd.Series, window: int) -> pd.Series:
        mean = s.rolling(window).mean()
        std = s.rolling(window).std(ddof=0).replace(0.0, np.nan)
        return (s - mean) / std

    def _classify_regime(self, row: pd.Series) -> str:
        gap_z = row.get("gap_z", np.nan)
        delta_z = row.get("delta_z", np.nan)
        uncertainty_ratio = row.get("uncertainty_ratio", np.nan)

        # if pd.notna(uncertainty_ratio) and uncertainty_ratio > self.uncertainty_ratio_thresh:
        #     return "UNCERTAIN"

        if pd.notna(gap_z) and pd.notna(delta_z):
            if gap_z >= self.risk_on_gap_z and delta_z >= self.risk_on_delta_z:
                return "RISK_ON"
            if gap_z >= self.defensive_gap_z and delta_z < self.risk_on_delta_z:
                return "DEFENSIVE"
            if gap_z <= self.duration_gap_z or delta_z <= self.falling_delta_z:
                return "DURATION"
            if abs(gap_z) <= self.ambiguous_band and abs(delta_z) <= self.ambiguous_band:
                return "UNCERTAIN"

        return "UNCERTAIN"

    def _compute_asset_features(self, data: StrategyData, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        feats: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            close = self.get_price_series(data.prices, ticker, "close").sort_index()
            close = close.reindex(index, method="ffill")
            ret_1m = close.pct_change()
            mom_3m = close.pct_change(3)
            mom_6m = close.pct_change(6)
            mom_12m = close.pct_change(12)
            vol_6m = ret_1m.rolling(6).std(ddof=0)
            feat = pd.DataFrame(
                {
                    "close": close,
                    "ret_1m": ret_1m,
                    "mom_3m": mom_3m,
                    "mom_6m": mom_6m,
                    "mom_12m": mom_12m,
                    "vol_6m": vol_6m,
                },
                index=index,
            )
            if self.use_ma_filter:
                feat["ma_10m"] = close.rolling(10).mean()
            feats[ticker] = feat
        return feats

    def _rank_assets(self, row_idx: pd.Timestamp, regime: str, asset_features: dict[str, pd.DataFrame]) -> list[str]:
        basket = self.baskets.get(regime, [self.fallback_etf])
        scores: list[tuple[str, float]] = []

        for ticker in basket:
            feat = asset_features[ticker].loc[row_idx]
            if feat[["mom_3m", "vol_6m"]].isna().any():
                continue
            if abs(float(feat["mom_3m"])) < self.min_abs_mom:
                continue
            if self.use_ma_filter and pd.notna(feat.get("ma_10m")) and float(feat["close"]) < float(feat["ma_10m"]):
                continue

            score = (
                self.score_w_3m * float(feat["mom_3m"])
                - self.score_w_vol * float(feat["vol_6m"])
            )
            scores.append((ticker, score))

        if not scores:
            print(f"[DEBUG] {row_idx.date()} | regime={regime} | no valid scores -> fallback {self.fallback_etf}")
            return [self.fallback_etf]

        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] {row_idx.date()} | regime={regime} | top candidates={scores[:3]}")
        return [t for t, _ in scores[: self.top_k]]

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)
        forecasts = self._coerce_forecast_index(data.forecasts).copy()
        forecasts = forecasts.sort_index()

        print("\n[DEBUG] Forecast rows:", len(forecasts))
        print("[DEBUG] Forecast date range:", forecasts.index.min(), "to", forecasts.index.max())

        if "y_pred" not in forecasts.columns:
            raise ValueError("Forecast panel must include a 'y_pred' column.")

        score = forecasts["y_pred"].astype(float)
        actual = forecasts["y"].astype(float) if "y" in forecasts.columns else pd.Series(np.nan, index=forecasts.index)

        forecasts["pred_trend"] = score.rolling(self.trend_window).mean()
        forecasts["pred_delta"] = score.diff()
        forecasts["pred_gap"] = score - forecasts["pred_trend"]
        forecasts["gap_z"] = self._rolling_z(forecasts["pred_gap"], self.z_window)
        forecasts["delta_z"] = self._rolling_z(forecasts["pred_delta"], self.z_window)

        if "residual_std" in forecasts.columns:
            residual_std = forecasts["residual_std"].astype(float)
        else:
            forecast_error = actual - score
            residual_std = forecast_error.shift(1).rolling(self.uncertainty_window).std(ddof=0)

        pred_std = score.shift(1).rolling(self.uncertainty_window).std(ddof=0).replace(0.0, np.nan)
        forecasts["uncertainty_ratio"] = residual_std / pred_std
        forecasts["macro_regime"] = forecasts.apply(self._classify_regime, axis=1)

        asset_features = self._compute_asset_features(data, forecasts.index)

        selected_assets: list[list[str]] = []
        selected_primary: list[str] = []
        confidences: list[float] = []
        metadata: list[dict] = []

        for dt, row in forecasts.iterrows():
            chosen = self._rank_assets(dt, row["macro_regime"], asset_features)
            selected_assets.append(chosen)
            selected_primary.append(chosen[0])

            conf = 0.0
            for v in [row.get("gap_z", np.nan), row.get("delta_z", np.nan)]:
                if pd.notna(v):
                    conf += abs(float(v))
            conf = min(conf / 4.0, 1.0)
            confidences.append(conf)

            metadata.append(
                {
                    "forecast": None if pd.isna(score.loc[dt]) else round(float(score.loc[dt]), 6),
                    "regime": row["macro_regime"],
                    "selected": chosen,
                    "gap_z": None if pd.isna(row.get("gap_z", np.nan)) else round(float(row["gap_z"]), 4),
                    "delta_z": None if pd.isna(row.get("delta_z", np.nan)) else round(float(row["delta_z"]), 4),
                    "uncertainty_ratio": None if pd.isna(row.get("uncertainty_ratio", np.nan)) else round(float(row["uncertainty_ratio"]), 4),
                }
            )

        weight_map = {ticker: np.zeros(len(forecasts), dtype=float) for ticker in self.tickers}
        cash_weight = np.ones(len(forecasts), dtype=float)

        for i, chosen in enumerate(selected_assets):
            alloc = self.target_allocation / len(chosen)
            for ticker in chosen:
                weight_map[ticker][i] = alloc
            cash_weight[i] = max(0.0, 1.0 - self.target_allocation)

        out = self._make_weight_frame(
            index=forecasts.index,
            weights=weight_map,
            cash_weight=cash_weight,
            confidence=np.asarray(confidences, dtype=float),
            metadata=metadata,
        )
        out["macro_regime"] = forecasts["macro_regime"].values
        out["selected_asset"] = selected_primary

        print("\n[DEBUG] Selected asset counts:")
        print(pd.Series(selected_primary).value_counts(dropna=False))

        print("\n[DEBUG] Final decisions:")
        print(pd.DataFrame({
            "regime": forecasts["macro_regime"].values,
            "selected_asset": selected_primary,
        }, index=forecasts.index))
        return out