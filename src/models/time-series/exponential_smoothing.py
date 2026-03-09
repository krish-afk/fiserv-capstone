# Exponential Smoothing (ETS) predictors for PCE forecasting.
#
# Exposes build_predictors() which returns a list of (name, fn) pairs
# ready for base.run_experiment().

import warnings
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')


# ============================================================
# PREDICTORS
# ============================================================

def predict_ses(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Simple Exponential Smoothing — level only, no trend or seasonality.
    Equivalent to ETS(A,N,N).
    """
    model = ExponentialSmoothing(train, trend=None, seasonal=None)
    return model.fit(optimized=True).forecast(horizon).values


def predict_holt(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Holt's linear trend method — level + additive trend.
    Equivalent to ETS(A,A,N).
    """
    model = ExponentialSmoothing(train, trend='add', seasonal=None)
    return model.fit(optimized=True).forecast(horizon).values


def predict_holt_damped(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Holt's damped trend method — level + damped additive trend.
    Equivalent to ETS(A,Ad,N). Tends to outperform undamped Holt on
    longer horizons.
    """
    model = ExponentialSmoothing(train, trend='add', damped_trend=True,
                                 seasonal=None)
    return model.fit(optimized=True).forecast(horizon).values


# ============================================================
# PREDICTOR FACTORY
# ============================================================

def build_predictors() -> list[tuple]:
    """
    Return the (name, fn) predictor list for this model family.

    All ETS variants are purely endogenous (no exog), so no extra
    arguments are needed and no binding is required.

    Returns
    -------
    list[tuple[str, Callable]]
    """
    return [
        ('ets_ses',         predict_ses),
        ('ets_holt',        predict_holt),
        ('ets_holt_damped', predict_holt_damped),
    ]
