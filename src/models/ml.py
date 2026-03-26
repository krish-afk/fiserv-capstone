# src/models/ml.py
import pandas as pd
from typing import Optional
from src.models.base import BaseForecaster

# TODO: add imports, e.g.:
# from sklearn.linear_model import Ridge
# from xgboost import XGBRegressor


class RidgeForecaster(BaseForecaster):
    """
    Ridge regression wrapper.
    Treats forecasting as supervised regression on a feature matrix.
    Requires X_train/X_test (lag features should be pre-built in features.py).
    """

    def __init__(self, alpha: float = 1.0, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.alpha = alpha
        self._model = None

    @property
    def name(self) -> str:
        return f"ridge_alpha={self.alpha}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "RidgeForecaster requires feature matrix X_train"
        # TODO: instantiate Ridge(alpha=self.alpha) and fit on X_train, y_train
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "RidgeForecaster requires X_test"
        # TODO: return model.predict(X_test) as a date-indexed pd.Series
        pass


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost regression wrapper.
    """

    def __init__(self, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon)
        self.xgb_params = kwargs  # Pass XGB hyperparams directly
        self._model = None

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "XGBoostForecaster requires feature matrix X_train"
        # TODO: instantiate XGBRegressor(**self.xgb_params) and fit
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "XGBoostForecaster requires X_test"
        # TODO: return predictions as date-indexed pd.Series
        pass
