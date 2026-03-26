# src/models/timeseries.py
import pandas as pd
from typing import Optional
from src.models.base import BaseForecaster

# TODO: add imports for your specific libraries, e.g.:
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA wrapper. Order (p,d,q) passed at instantiation.
    """

    def __init__(self, order: tuple = (1, 1, 1), horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.order = order
        self._model_fit = None

    @property
    def name(self) -> str:
        p, d, q = self.order
        return f"arima_{p}{d}{q}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        # TODO: instantiate and fit your ARIMA model here
        # self._model_fit = ARIMA(y_train, order=self.order).fit()
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        # TODO: return forecast as a date-indexed pd.Series
        # forecast = self._model_fit.forecast(steps=self.horizon)
        # return forecast
        pass


class ARIMAXForecaster(BaseForecaster):
    """
    ARIMAX wrapper — ARIMA with exogenous features (e.g. FSBI columns).
    X_train and X_test are required for this model.
    """

    def __init__(self, order: tuple = (1, 1, 1), horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.order = order
        self._model_fit = None

    @property
    def name(self) -> str:
        p, d, q = self.order
        return f"arimax_{p}{d}{q}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "ARIMAXForecaster requires exogenous features"
        # TODO: fit ARIMA with exog=X_train
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "ARIMAXForecaster requires X_test for prediction"
        # TODO: return forecast with exog=X_test
        pass


class ETSForecaster(BaseForecaster):
    """
    Exponential Smoothing (Holt-Winters) wrapper.
    """

    def __init__(self, trend: str = "add", seasonal: str = None,
                 seasonal_periods: int = 12, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._model_fit = None

    @property
    def name(self) -> str:
        return f"ets_trend={self.trend}_seasonal={self.seasonal}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        # TODO: instantiate and fit ExponentialSmoothing here
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        # TODO: return forecast as date-indexed pd.Series
        pass
