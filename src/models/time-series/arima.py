# ARIMA / ARIMAX predictors for PCE forecasting.
#
# Exposes PREDICTORS: list of (name, fn) pairs ready for base.run_experiment().
# Call build_predictors() to construct them with the appropriate orders and
# exogenous feature slices for a given target series.

import warnings
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from base import select_exog_features

warnings.filterwarnings('ignore')


# ============================================================
# PREDICTORS
# ============================================================

def predict_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for all forecast steps."""
    return np.repeat(train.iloc[-1], horizon)


def predict_arima(train: pd.Series, horizon: int, order: tuple) -> np.ndarray:
    """Fit ARIMA(p,d,q) on train and return a multi-step forecast."""
    return ARIMA(train, order=order).fit().forecast(steps=horizon).values


def predict_arimax(train: pd.Series,
                   horizon: int,
                   order: tuple,
                   exog_df: pd.DataFrame) -> np.ndarray:
    """
    Fit ARIMAX on train with exogenous features and return a forecast.

    Exogenous features for the forecast horizon are assumed known in advance.
    exog_df must cover both the training period and the forecast horizon.
    """
    exog_train = exog_df.loc[train.index]
    # The next `horizon` rows after the training window
    all_dates = exog_df.index
    train_end_pos = all_dates.get_loc(train.index[-1])
    exog_future = exog_df.iloc[train_end_pos + 1: train_end_pos + 1 + horizon]

    assert len(exog_train) == len(train), "exog_train length must match train"
    assert len(exog_future) == horizon, "exog_future length must match horizon"

    result = ARIMA(train, order=order, exog=exog_train).fit()
    return result.forecast(steps=horizon, exog=exog_future).values


# ============================================================
# ORDER SELECTION UTILITY
# ============================================================

def find_best_arima_order(series: pd.Series,
                          p_range: range = range(0, 4),
                          d_range: range = range(0, 2),
                          q_range: range = range(0, 4)) -> tuple:
    """
    Grid search over ARIMA (p, d, q) orders, selecting by AIC.

    Returns
    -------
    tuple
        Best (p, d, q) order.
    """
    best_aic = np.inf
    best_order = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    aic = ARIMA(series, order=(p, d, q)).fit().aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    print(f"Best ARIMA order: {best_order} | AIC: {best_aic:.4f}")
    return best_order


# ============================================================
# PREDICTOR FACTORY
# ============================================================

def build_predictors(target: pd.Series,
                     exog_df: pd.DataFrame,
                     exog_cols: list,
                     order: tuple,
                     max_exog_features: int = 8) -> list[tuple]:
    """
    Build the (name, fn) predictor list for this model family.

    Performs exog feature selection and binds all model-specific parameters
    so the returned callables match the standard (train, horizon) signature.

    Parameters
    ----------
    target : pd.Series
        The target series (used for exog feature selection).
    exog_df : pd.DataFrame
        Full exogenous feature DataFrame.
    exog_cols : list
        Candidate exogenous feature columns.
    order : tuple
        ARIMA (p, d, q) order to use for both ARIMA and ARIMAX.
    max_exog_features : int
        Maximum features passed to ARIMAX after selection.

    Returns
    -------
    list[tuple[str, Callable]]
    """
    selected = select_exog_features(target, exog_df[exog_cols],
                                    max_features=max_exog_features)
    exog_selected = exog_df[selected]

    return [
        ('naive', predict_naive),
        ('arima', partial(predict_arima, order=order)),
        ('arimax', partial(predict_arimax, order=order, exog_df=exog_selected)),
    ]
