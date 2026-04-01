# src/models/timeseries.py
import warnings

import pandas as pd
from typing import Optional

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.models.base import BaseForecaster

# ARIMA and ETS produce many convergence/optimization warnings during
# walk-forward evaluation (hundreds of fits per experiment run).
warnings.filterwarnings("ignore")


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA(p,d,q) wrapper. Pure time-series — ignores X_train/X_test.
    Order is passed as a list or tuple; YAML lists are normalized to tuple.
    """

    def __init__(self, order=(1, 1, 1), horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.order = tuple(order)
        self._model_fit = None

    @property
    def name(self) -> str:
        p, d, q = self.order
        return f"arima_{p}{d}{q}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        self._model_fit = ARIMA(y_train, order=self.order).fit()
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        return self._model_fit.forecast(steps=self.horizon)


class ARIMAXForecaster(BaseForecaster):
    """
    ARIMAX(p,d,q) wrapper — ARIMA with exogenous features.
    X_train and X_test are required.

    Exog safety: all features in X are pre-lagged by features.py (shift-based
    lags and rolling stats), so X_test at fold t contains only information
    available at time t. No lookahead.
    """

    def __init__(self, order=(1, 1, 1), horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.order = tuple(order)
        self._model_fit = None

    @property
    def name(self) -> str:
        p, d, q = self.order
        return f"arimax_{p}{d}{q}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "ARIMAXForecaster requires X_train"
        self._model_fit = ARIMA(y_train, order=self.order, exog=X_train).fit()
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "ARIMAXForecaster requires X_test for prediction"
        return self._model_fit.forecast(steps=self.horizon, exog=X_test)


class ETSForecaster(BaseForecaster):
    """
    Exponential Smoothing wrapper covering SES, Holt, and Holt-Winters variants.
    Pure time-series — ignores X_train/X_test.

    Common configurations (set via config.yaml params):
      SES:          trend=null,  seasonal=null
      Holt:         trend='add', seasonal=null
      Holt-damped:  trend='add', damped_trend=true, seasonal=null
      Holt-Winters: trend='add', seasonal='add'|'mul', seasonal_periods=12

    Note: multiplicative seasonal ('mul') requires all y values to be positive.
    PCE MoM% can be negative — use additive seasonal or seasonal=null for MoM panels.
    """

    def __init__(
        self,
        trend: Optional[str] = None,
        damped_trend: bool = False,
        seasonal: Optional[str] = None,
        seasonal_periods: int = 12,
        horizon: int = 1,
        **kwargs,
    ):
        super().__init__(horizon=horizon, **kwargs)
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._model_fit = None

    @property
    def name(self) -> str:
        t = self.trend or "none"
        d = "_damped" if (self.damped_trend and self.trend) else ""
        s = f"_{self.seasonal}{self.seasonal_periods}" if self.seasonal else "_none"
        return f"ets_{t}{d}{s}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        kwargs: dict = {"trend": self.trend, "seasonal": self.seasonal}
        if self.seasonal is not None:
            kwargs["seasonal_periods"] = self.seasonal_periods
        if self.damped_trend and self.trend is not None:
            kwargs["damped_trend"] = True
        self._model_fit = ExponentialSmoothing(y_train, **kwargs).fit(optimized=True)
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        return self._model_fit.forecast(self.horizon)
