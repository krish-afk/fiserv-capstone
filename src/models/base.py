# src/models/base.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class BaseForecaster(ABC):
    """
    Abstract base class for all PCE forecasting models.
    All model implementations must inherit from this class and
    implement fit(), predict(), and name().
    """

    def __init__(self, horizon: int = 1, **kwargs):
        self.horizon = horizon
        self.is_fitted = False
        # TODO: Store any shared hyperparameters here

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
