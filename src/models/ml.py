# src/models/ml.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.models.base import BaseForecaster


class OLSForecaster(BaseForecaster):
    """
    OLS regression (no regularization) via statsmodels.
    No scaling applied — OLS estimates are scale-invariant.
    Serves as a sanity-check just above naive baselines: any regularized or
    ensemble model should beat OLS once the feature count is non-trivial.
    """

    def __init__(self, horizon: int = 1, **kwargs):
        super().__init__(horizon=horizon, **kwargs)
        self._result = None

    @property
    def name(self) -> str:
        return "ols"

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        assert X_train is not None, "OLSForecaster requires X_train"
        X = sm.add_constant(X_train, has_constant="add")
        self._result = sm.OLS(y_train, X).fit()
        self.is_fitted = True
        # Record per-fold interpretability — skip the added constant row
        feat_names = X_train.columns.tolist()
        self._fold_artifacts.append({
            "coef":    dict(zip(feat_names, self._result.params[1:].values)),
            "t_stat":  dict(zip(feat_names, self._result.tvalues[1:].values)),
            "p_value": dict(zip(feat_names, self._result.pvalues[1:].values)),
        })

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "OLSForecaster requires X_test"
        X = sm.add_constant(X_test, has_constant="add")
        preds = self._result.predict(X)
        return pd.Series(preds.values, index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        return {
            "n_folds":      len(self._fold_artifacts),
            "coefficients": self._agg_coef(self._fold_artifacts, "coef"),
            "t_stats":      self._agg_coef(self._fold_artifacts, "t_stat"),
            "p_values":     self._agg_coef(self._fold_artifacts, "p_value"),
        }


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
        coefs = self._pipeline.named_steps["model"].coef_
        self._fold_artifacts.append({"coef": dict(zip(X_train.columns, coefs))})

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "RidgeForecaster requires X_test"
        preds = self._pipeline.predict(X_test)
        return pd.Series(preds, index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        return {
            "n_folds":      len(self._fold_artifacts),
            "coefficients": self._agg_coef(self._fold_artifacts),
        }


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
        coefs = self._pipeline.named_steps["model"].coef_
        self._fold_artifacts.append({"coef": dict(zip(X_train.columns, coefs))})

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "LassoForecaster requires X_test"
        return pd.Series(self._pipeline.predict(X_test), index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        feats = list(self._fold_artifacts[0]["coef"].keys())
        # Selection frequency: fraction of folds where |coef| > 0 (Lasso produces
        # exact zeros via coordinate descent; threshold matches sklearn's convention)
        sel_freq = {
            feat: float(np.mean([f["coef"][feat] != 0.0 for f in self._fold_artifacts]))
            for feat in feats
        }
        return {
            "n_folds":           len(self._fold_artifacts),
            "coefficients":      self._agg_coef(self._fold_artifacts),
            "selection_freq":    sel_freq,
        }


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
        self._fold_artifacts.append({
            "importance": dict(zip(X_train.columns, self._model.feature_importances_))
        })

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "RandomForestForecaster requires X_test"
        return pd.Series(self._model.predict(X_test), index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        agg = self._agg_importance(self._fold_artifacts)
        return {"n_folds": len(self._fold_artifacts), "feature_importance": agg}


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
        # gain importance: only features used in splits are returned.
        # Fill the full feature list with 0 for absent features as requested.
        raw = self._model.get_booster().get_score(importance_type="gain")
        importance = {col: raw.get(col, 0.0) for col in X_train.columns}
        self._fold_artifacts.append({"importance": importance})

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "XGBoostForecaster requires X_test"
        return pd.Series(self._model.predict(X_test), index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        agg = self._agg_importance(self._fold_artifacts)
        return {"n_folds": len(self._fold_artifacts), "feature_importance": agg}


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
        self._fold_artifacts.append({
            "importance": dict(zip(X_train.columns, self._model.feature_importances_))
        })

    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        assert self.is_fitted, "Model must be fit before predicting"
        assert X_test is not None, "GradientBoostingForecaster requires X_test"
        return pd.Series(self._model.predict(X_test), index=X_test.index)

    def summarize_artifacts(self) -> dict:
        if not self._fold_artifacts:
            return {}
        agg = self._agg_importance(self._fold_artifacts)
        return {"n_folds": len(self._fold_artifacts), "feature_importance": agg}
