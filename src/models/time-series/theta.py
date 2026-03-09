# Theta model predictors for PCE forecasting.
#
# Exposes build_predictors() which returns a list of (name, fn) pairs
# ready for base.run_experiment().
#
# The Theta model (Assimakopoulos & Nikolopoulos, 2000) decomposes a series
# into two "theta lines" — one capturing long-run trend, one capturing
# short-run variation — then combines their forecasts. statsmodels implements
# this via ThetaModel with theta=2 (the standard choice).

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel

warnings.filterwarnings('ignore')


# ============================================================
# PREDICTORS
# ============================================================

def predict_theta(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Standard Theta model (theta=2) with automatic seasonality detection.

    statsmodels will test for seasonality using the series frequency; for
    monthly data with a DatetimeIndex it defaults to period=12.
    """
    model = ThetaModel(train, deseasonalize=True, use_test=True)
    return model.fit().forecast(horizon).values


def predict_theta_no_season(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Theta model with seasonality adjustment disabled.

    Useful as a comparison against the auto-deseasonalized variant,
    or when the series has no meaningful seasonal period.
    """
    model = ThetaModel(train, deseasonalize=False)
    return model.fit().forecast(horizon).values


# ============================================================
# PREDICTOR FACTORY
# ============================================================

def build_predictors() -> list[tuple]:
    """
    Return the (name, fn) predictor list for this model family.

    Returns
    -------
    list[tuple[str, Callable]]
    """
    return [
        ('theta',           predict_theta),
        ('theta_no_season', predict_theta_no_season),
    ]
