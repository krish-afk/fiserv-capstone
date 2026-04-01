# src/models/timeseries.py
import warnings

import numpy as np
import pandas as pd
from typing import Optional

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

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
        # Extract exog coefficients: params is a Series; filter to those whose
        # index name matches an X_train column (excludes AR/MA/sigma2 params).
        exog_params = {
            name: float(val)
            for name, val in self._model_fit.params.items()
            if name in X_train.columns
        }
        self._fold_artifacts.append({"coef": exog_params})

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "ARIMAXForecaster requires X_test for prediction"
        return self._model_fit.forecast(steps=self.horizon, exog=X_test)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        return {
            "n_folds":      len(self._fold_artifacts),
            "coefficients": self._agg_coef(self._fold_artifacts),
        }


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


class ThetaForecaster(BaseForecaster):
    """
    Theta model (statsmodels ThetaModel). Pure time-series — ignores X_train/X_test.

    deseasonalize=True is appropriate for level or YoY panels where seasonal
    structure is present. For MoM% panels (already first-differenced), set
    deseasonalize=False — the seasonal component is typically negligible and
    the decomposition can be unstable on a near-stationary series.

    period: seasonal period for decomposition. None lets statsmodels detect it
    automatically via a seasonality test (requires a DatetimeIndex with freq).
    For monthly data, period=12 is safe to set explicitly.
    """

    def __init__(
        self,
        deseasonalize: bool = True,
        period: Optional[int] = None,
        horizon: int = 1,
        **kwargs,
    ):
        super().__init__(horizon=horizon, **kwargs)
        self.deseasonalize = deseasonalize
        self.period = period
        self._model_fit = None

    @property
    def name(self) -> str:
        d = "deseas" if self.deseasonalize else "nodeseas"
        return f"theta_{d}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        self._model_fit = ThetaModel(
            y_train,
            period=self.period,
            deseasonalize=self.deseasonalize,
        ).fit()
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        return self._model_fit.forecast(self.horizon)


class ETSXForecaster(BaseForecaster):
    """
    Two-stage OLS-residual ETS hybrid with exogenous features.
    Ported from old/models/time-series/exponential_smoothing.py.

    Stage 1: OLS (with intercept) on X_train → fit beta, compute residuals.
    Stage 2: ETS on residuals (trend/damped_trend controlled by params).
    Forecast: X_test @ beta + ETS.forecast(horizon).

    This separates the exogenous signal (captured by OLS) from the autocorrelated
    residual dynamics (captured by ETS), avoiding the need for ARIMAX-style
    joint estimation.

    trend / damped_trend follow the same conventions as ETSForecaster:
      trend=null             → SES on residuals
      trend='add'            → Holt on residuals
      trend='add', damped_trend=true → Holt-damped on residuals
    """

    def __init__(
        self,
        trend: Optional[str] = None,
        damped_trend: bool = False,
        horizon: int = 1,
        **kwargs,
    ):
        super().__init__(horizon=horizon, **kwargs)
        self.trend = trend
        self.damped_trend = damped_trend
        self._beta: Optional[np.ndarray] = None
        self._ets_fit = None

    @property
    def name(self) -> str:
        t = self.trend or "none"
        d = "_damped" if (self.damped_trend and self.trend) else ""
        return f"ets_x_{t}{d}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "ETSXForecaster requires X_train"
        # Stage 1: OLS with intercept
        X = np.column_stack([np.ones(len(y_train)), X_train.values])
        self._beta, _, _, _ = np.linalg.lstsq(X, y_train.values, rcond=None)
        residuals = y_train.values - X @ self._beta

        # Stage 2: ETS on residuals
        ets_kwargs: dict = {"trend": self.trend, "seasonal": None}
        if self.damped_trend and self.trend is not None:
            ets_kwargs["damped_trend"] = True
        self._ets_fit = ExponentialSmoothing(
            pd.Series(residuals, index=y_train.index), **ets_kwargs
        ).fit(optimized=True)
        self.is_fitted = True
        # beta[0] is the intercept; beta[1:] are the exog coefficients
        self._fold_artifacts.append({
            "coef": dict(zip(X_train.columns, self._beta[1:]))
        })

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "ETSXForecaster requires X_test"
        X_fut = np.column_stack([np.ones(len(X_test)), X_test.values])
        ets_fc = self._ets_fit.forecast(self.horizon).values
        preds  = X_fut @ self._beta + ets_fc
        return pd.Series(preds, index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        return {
            "n_folds":      len(self._fold_artifacts),
            "coefficients": self._agg_coef(self._fold_artifacts),
        }
