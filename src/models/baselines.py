# src/models/baselines.py
import pandas as pd
import numpy as np
from typing import Optional
from src.models.base import BaseForecaster

class NaiveForecaster(BaseForecaster):
    """
    Seasonal naive baseline: forecast = last observed value.
    Useful as a sanity-check floor — any real model should beat this.
    """

    @property
    def name(self) -> str:
        return "naive"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        self._last_value = y_train.iloc[-1]
        self._last_date = y_train.index[-1]
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        # TODO: extend to seasonal naive (e.g. same month last year)
        # by storing y_train and indexing back by frequency
        assert self.is_fitted, "Model must be fit before predicting"
        forecast_index = pd.date_range(
            self._last_date, periods=self.horizon + 1, freq="MS"
        )[1:]
        return pd.Series(self._last_value, index=forecast_index)


class MeanForecaster(BaseForecaster):
    """
    Historical mean baseline: forecast = mean of training window.
    """

    @property
    def name(self) -> str:
        return "historical_mean"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        self._mean = y_train.mean()
        self._last_date = y_train.index[-1]
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        forecast_index = pd.date_range(
            self._last_date, periods=self.horizon + 1, freq="MS"
        )[1:]
        return pd.Series(self._mean, index=forecast_index)
