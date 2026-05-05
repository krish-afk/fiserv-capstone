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
    
class AR3BaselineForecaster(BaseForecaster):
    """
    Strict AR(3) baseline estimated by OLS on exactly three target-lag columns.

    For PCE MoM panels, this should be:
        pce_mom_lag1, pce_mom_lag2, pce_mom_lag3

    For MRTS level panels, this should be:
        mrts_lag1, mrts_lag2, mrts_lag3
    """

    _ALLOWED_TARGET_PREFIXES = {
        "pce",
        "pce_mom",
        "pce_yoy",
        "mrts",
        "mrts_mom",
        "mrts_yoy",
    }

    def __init__(self, order: int = 3, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)

        if int(order) != 3:
            raise ValueError("AR3BaselineForecaster is intentionally fixed to order=3")

        if int(horizon) != 1:
            raise ValueError("AR3BaselineForecaster currently supports only horizon=1")

        self.order = 3
        self._coef = None
        self._lag_columns = None

    @property
    def name(self) -> str:
        return "ar3_baseline"

    def _select_lag_columns(self, X: Optional[pd.DataFrame]) -> list[str]:
        if X is None:
            raise ValueError(
                "AR3BaselineForecaster requires X with exactly three target-lag columns"
            )

        if X.shape[1] != self.order:
            raise ValueError(
                "AR3BaselineForecaster must receive exactly 3 columns, "
                f"but received {X.shape[1]} columns: {list(X.columns)}"
            )

        lag_columns = []
        for lag in range(1, self.order + 1):
            suffix = f"_lag{lag}"
            matches = [col for col in X.columns if str(col).endswith(suffix)]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one column ending in '{suffix}', "
                    f"but found {matches}. Columns received: {list(X.columns)}"
                )

            lag_columns.append(matches[0])

        prefixes = {str(col).rsplit("_lag", 1)[0] for col in lag_columns}
        if len(prefixes) != 1:
            raise ValueError(
                "AR3BaselineForecaster requires all lag columns to come from "
                f"the same target. Got prefixes: {prefixes}"
            )

        target_prefix = next(iter(prefixes))
        if target_prefix not in self._ALLOWED_TARGET_PREFIXES:
            raise ValueError(
                "AR3BaselineForecaster only allows PCE/MRTS target lags. "
                f"Got target prefix '{target_prefix}' from columns {lag_columns}"
            )

        return lag_columns

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        lag_columns = self._select_lag_columns(X_train)

        train_df = pd.concat(
            [y_train.rename("__target__"), X_train[lag_columns]],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(train_df) <= self.order:
            raise ValueError("Not enough observations to estimate AR(3) baseline")

        y = train_df["__target__"].astype("float64").to_numpy()
        X = train_df[lag_columns].astype("float64").to_numpy()

        X_design = np.column_stack([np.ones(len(X)), X])
        self._coef = np.linalg.lstsq(X_design, y, rcond=None)[0]
        self._lag_columns = lag_columns
        self.is_fitted = True

        self._fold_artifacts.append({
            "coef": dict(zip(["const"] + lag_columns, self._coef))
        })

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"

        if X_test is None:
            raise ValueError("AR3BaselineForecaster requires X_test")

        X = X_test[self._lag_columns].astype("float64").to_numpy()
        X_design = np.column_stack([np.ones(len(X)), X])
        preds = X_design @ self._coef

        return pd.Series(preds, index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}

        return {
            "n_folds": len(self._fold_artifacts),
            "coefficients": self._agg_coef(self._fold_artifacts, "coef"),
        }
