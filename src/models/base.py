# src/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract base class for all PCE forecasting models.
    All model implementations must inherit from this class and
    implement fit(), predict(), and name.
    """

    def __init__(self, horizon: int = 1, **kwargs):
        self.horizon = horizon
        self.is_fitted = False
        # Populated by subclass fit() calls during walk-forward evaluation.
        # Each entry is one fold's artifact dict (e.g. coefficients, importances).
        self._fold_artifacts: List[dict] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique string identifier for this model."""
        pass

    @abstractmethod
    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None):
        """
        Fit the model on training data.

        Args:
            y_train: Target series (PCE), indexed by date
            X_train: Optional exogenous features (e.g. FSBI columns)
        """
        pass

    @abstractmethod
    def predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate forecast(s) after fitting.

        Args:
            X_test: Optional exogenous features for forecast horizon
        Returns:
            pd.Series of forecasted values, indexed by date
        """
        pass

    def fit_predict(self, y_train: pd.Series, X_train=None, X_test=None) -> pd.Series:
        """Convenience wrapper: fit then predict in one call."""
        self.fit(y_train, X_train)
        return self.predict(X_test)

    def summarize_artifacts(self) -> dict:
        """
        Aggregate per-fold artifacts accumulated during walk-forward evaluation.
        Override in subclasses that record interpretability information in fit().
        Default returns {} (no artifact written for this model).
        """
        return {}

    # ------------------------------------------------------------------
    # Static helpers for subclass summarize_artifacts() implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _agg_coef(fold_artifacts: List[dict], key: str = "coef") -> Dict[str, dict]:
        """
        Aggregate per-fold coefficient dicts into {feature: {mean, std}}.
        Feature order is taken from the first fold.
        """
        if not fold_artifacts:
            return {}
        feats = list(fold_artifacts[0][key].keys())
        return {
            feat: {
                "mean": float(np.mean([f[key][feat] for f in fold_artifacts])),
                "std":  float(np.std( [f[key][feat] for f in fold_artifacts])),
            }
            for feat in feats
        }

    @staticmethod
    def _agg_importance(fold_artifacts: List[dict], key: str = "importance") -> dict:
        """
        Aggregate per-fold importance dicts into {feature: {mean, std}} plus
        a top-10 ranking by mean importance.
        """
        if not fold_artifacts:
            return {}
        feats = list(fold_artifacts[0][key].keys())
        by_feat = {
            feat: {
                "mean": float(np.mean([f[key][feat] for f in fold_artifacts])),
                "std":  float(np.std( [f[key][feat] for f in fold_artifacts])),
            }
            for feat in feats
        }
        top10 = sorted(by_feat, key=lambda f: by_feat[f]["mean"], reverse=True)[:10]
        return {"by_feature": by_feat, "top_10_by_mean": top10}
