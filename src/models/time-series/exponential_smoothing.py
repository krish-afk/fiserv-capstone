# Exponential Smoothing (ETS) predictors for PCE forecasting.
#
# Exposes build_predictors() which returns a list of (name, fn) pairs
# ready for base.run_experiment().

import warnings
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from base import select_exog_features

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


def predict_ets_x(train: pd.Series,
                  horizon: int,
                  exog_df: pd.DataFrame,
                  trend: str | None = None,
                  damped_trend: bool = False) -> np.ndarray:
    """
    Two-stage ETS with exogenous regressors (regression-with-ETS-errors).

    Stage 1: OLS on exog features over the training window → residuals.
    Stage 2: Fit ETS on residuals.
    Forecast: exog_future @ beta + ETS.forecast(horizon).

    exog_df must cover both the training period and the forecast horizon.
    The future exog rows are taken from the horizon steps immediately
    following the training window, matching the ARIMAX convention.
    """
    exog_train = exog_df.loc[train.index].values

    all_dates = exog_df.index
    train_end_pos = all_dates.get_loc(train.index[-1])
    exog_future = exog_df.iloc[train_end_pos + 1: train_end_pos + 1 + horizon].values

    assert exog_train.shape[0] == len(train), "exog_train length must match train"
    assert exog_future.shape[0] == horizon, "exog_future length must match horizon"

    # OLS with intercept
    X_train = np.column_stack([np.ones(len(train)), exog_train])
    X_future = np.column_stack([np.ones(horizon), exog_future])
    beta, _, _, _ = np.linalg.lstsq(X_train, train.values, rcond=None)
    residuals = train.values - X_train @ beta

    # ETS on residuals (damped_trend only valid when trend is set)
    residual_series = pd.Series(residuals, index=train.index)
    effective_damped = damped_trend if trend is not None else False
    model = ExponentialSmoothing(residual_series, trend=trend,
                                 damped_trend=effective_damped,
                                 seasonal=None)
    ets_forecast = model.fit(optimized=True).forecast(horizon).values

    return X_future @ beta + ets_forecast


# ============================================================
# PREDICTOR FACTORY
# ============================================================

def build_predictors(target: pd.Series | None = None,
                     exog_df: pd.DataFrame | None = None,
                     exog_cols: list | None = None,
                     max_exog_features: int = 8,
                     limited_features: list | None = None) -> list[tuple]:
    """
    Return the (name, fn) predictor list for this model family.

    When exog_df, target, and exog_cols are provided, also builds ETS-X
    variants (two-stage regression + ETS on residuals) with two feature sets:
      - auto-selected: top features by correlation with collinearity filter
      - limited: the caller-supplied limited_features list (if provided)

    Parameters
    ----------
    target : pd.Series | None
        Target series for exog feature selection. Required if exog_df given.
    exog_df : pd.DataFrame | None
        Full exogenous feature DataFrame.
    exog_cols : list | None
        Candidate exogenous feature columns for auto-selection.
    max_exog_features : int
        Max features passed to auto-selection.
    limited_features : list | None
        Pre-specified feature list for the *_limited variants.
        If None or empty, limited variants are skipped.

    Returns
    -------
    list[tuple[str, Callable]]
    """
    base_predictors = [
        ('ets_ses',         predict_ses),
        ('ets_holt',        predict_holt),
        ('ets_holt_damped', predict_holt_damped),
    ]

    if exog_df is None or target is None or exog_cols is None:
        return base_predictors

    # --- Auto-selected feature variants ---
    selected = select_exog_features(target, exog_df[exog_cols],
                                    max_features=max_exog_features)
    exog_selected = exog_df[selected]

    ets_x_selected = [
        ('ets_x_ses',
         partial(predict_ets_x, exog_df=exog_selected,
                 trend=None, damped_trend=False)),
        ('ets_x_holt',
         partial(predict_ets_x, exog_df=exog_selected,
                 trend='add', damped_trend=False)),
        ('ets_x_holt_damped',
         partial(predict_ets_x, exog_df=exog_selected,
                 trend='add', damped_trend=True)),
    ]

    # --- Limited feature variants ---
    ets_x_limited = []
    if limited_features:
        valid = [f for f in limited_features if f in exog_df.columns]
        if valid:
            exog_limited = exog_df[valid]
            ets_x_limited = [
                ('ets_x_ses_limited',
                 partial(predict_ets_x, exog_df=exog_limited,
                         trend=None, damped_trend=False)),
                ('ets_x_holt_limited',
                 partial(predict_ets_x, exog_df=exog_limited,
                         trend='add', damped_trend=False)),
                ('ets_x_holt_damped_limited',
                 partial(predict_ets_x, exog_df=exog_limited,
                         trend='add', damped_trend=True)),
            ]

    return base_predictors + ets_x_selected + ets_x_limited
