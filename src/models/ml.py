# src/models/ml.py
import pandas as pd
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.models.base import BaseForecaster


class RidgeForecaster(BaseForecaster):
    """
    Ridge regression. Scaling handled per-fold via sklearn Pipeline.
    Requires X_train / X_test (pre-lagged features from features.py).
    """

    def __init__(self, alpha: float = 1.0, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.alpha = alpha
        self._pipeline: Optional[Pipeline] = None

    @property
    def name(self) -> str:
        return f"ridge_alpha={self.alpha}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "RidgeForecaster requires X_train"
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=self.alpha)),
        ])
        self._pipeline.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "RidgeForecaster requires X_test"
        preds = self._pipeline.predict(X_test)
        return pd.Series(preds, index=X_test.index)


class LassoForecaster(BaseForecaster):
    """
    Lasso regression with implicit feature selection.
    Scaling handled per-fold via sklearn Pipeline.
    """

    def __init__(self, alpha: float = 1.0, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self.alpha = alpha
        self._pipeline: Optional[Pipeline] = None

    @property
    def name(self) -> str:
        return f"lasso_alpha={self.alpha}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "LassoForecaster requires X_train"
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=self.alpha, max_iter=5000)),
        ])
        self._pipeline.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "LassoForecaster requires X_test"
        return pd.Series(self._pipeline.predict(X_test), index=X_test.index)


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest regressor. Tree-based; no scaling required.
    Handles missing values natively via sklearn's impurity-based splits.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        horizon: int = 1,
        **kwargs,
    ):
        super().__init__(horizon=horizon, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._model: Optional[RandomForestRegressor] = None

    @property
    def name(self) -> str:
        d = "none" if self.max_depth is None else self.max_depth
        return f"rf_n_estimators={self.n_estimators}_max_depth={d}_min_samples_leaf={self.min_samples_leaf}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "RandomForestForecaster requires X_train"
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=1,
        )
        self._model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "RandomForestForecaster requires X_test"
        return pd.Series(self._model.predict(X_test), index=X_test.index)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost regressor. Extra constructor kwargs are passed directly to
    XGBRegressor, so param_grid entries (n_estimators, learning_rate, etc.)
    are forwarded automatically.

    Uses objective='reg:absoluteerror' to optimize MAE, consistent with the
    primary evaluation metric.
    """

    def __init__(self, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon)
        self.xgb_params = kwargs
        self._model: Optional[XGBRegressor] = None

    @property
    def name(self) -> str:
        if not self.xgb_params:
            return "xgboost"
        suffix = "_".join(f"{k}={v}" for k, v in sorted(self.xgb_params.items()))
        return f"xgboost_{suffix}"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "XGBoostForecaster requires X_train"
        self._model = XGBRegressor(
            **self.xgb_params,
            objective="reg:absoluteerror",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        self._model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "XGBoostForecaster requires X_test"
        return pd.Series(self._model.predict(X_test), index=X_test.index)


class GradientBoostingForecaster(BaseForecaster):
    """
    sklearn GradientBoostingRegressor. Sequential boosting; no scaling required.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        horizon: int = 1,
        **kwargs,
    ):
        super().__init__(horizon=horizon, **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self._model: Optional[GradientBoostingRegressor] = None

    @property
    def name(self) -> str:
        return (f"gbm_n_estimators={self.n_estimators}"
                f"_learning_rate={self.learning_rate}"
                f"_max_depth={self.max_depth}")

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "GradientBoostingForecaster requires X_train"
        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )
        self._model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "GradientBoostingForecaster requires X_test"
        return pd.Series(self._model.predict(X_test), index=X_test.index)
