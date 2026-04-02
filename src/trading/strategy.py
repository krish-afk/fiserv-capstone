# src/trading/strategy.py
"""
Strategy layer for PCE-forecast-driven trading simulations.

Architecture
------------
BaseStrategy (ABC)
  ├── DirectionalPCEStrategy   – long/short based on forecast sign
  └── ThresholdPCEStrategy     – trade only when |forecast| > threshold

All strategies accept a StrategyData container and return a standardised
signal DataFrame.  The backtest layer (backtest.py) handles execution;
strategies only generate signals.

Signal schema
-------------
date        DatetimeIndex
signal      float  [-1.0, 1.0]  (-1 = full short, 0 = flat, 1 = full long)
confidence  float  [0.0, 1.0]   abs(signal) by default; override in subclass
metadata    str                 JSON-serialisable debug info (y_pred, etc.)
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import config


# ---------------------------------------------------------------------------
# StrategyData — typed input container
# ---------------------------------------------------------------------------

@dataclass
class StrategyData:
    """
    Container for all inputs a strategy may need.

    Required: forecasts
    Optional: prices, macro, mrts  (strategies declare needs via required_inputs)
    """
    forecasts: pd.DataFrame                     # date | y_true | y_pred | model_name | …
    prices:    Optional[pd.DataFrame] = None    # date-indexed OHLCV, flat cols (XLY_close …)
    macro:     Optional[pd.DataFrame] = None    # date-indexed macro indicators
    mrts:      Optional[pd.DataFrame] = None    # date | y_true | y_pred for MRTS target

    def validate(self, required: set) -> None:
        """
        Raise ValueError if any required input is None.

        Args:
            required: Set of attribute names that must be present
                      (subset of {"forecasts", "prices", "macro", "mrts"})
        """
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            raise ValueError(
                f"Strategy requires inputs {missing} but they are None in StrategyData."
            )


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategy implementations.

    To add a new strategy:
      1. Inherit from BaseStrategy.
      2. Implement ``name`` (property) and ``generate_signals()`` (method).
      3. Override ``required_inputs`` if the strategy needs prices / macro / mrts.
      4. Register it in config.yaml under trading.strategies.

    Subclasses must NOT implement backtest execution — that belongs in backtest.py.
    """

    def __init__(self, **params):
        """
        Args:
            **params: Strategy hyperparameters.  Subclasses should accept
                      explicit kwargs AND forward **params to super().__init__()
                      so they are accessible via self.params for logging.
        """
        self.params = params

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier — used in output filenames and logs."""
        ...

    @abstractmethod
    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        """
        Compute trading signals from forecast and optional inputs.

        Args:
            data: StrategyData container (validated against required_inputs
                  before this method is called by BacktestEngine).
        Returns:
            DataFrame with DatetimeIndex and columns:
                signal      float [-1.0, 1.0]
                confidence  float [0.0, 1.0]
                metadata    str  (JSON string, may be "{}")
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    @property
    def required_inputs(self) -> set:
        """
        Names of StrategyData attributes this strategy requires.
        Default: only forecasts.  Override if prices / macro / mrts are needed.
        """
        return {"forecasts"}

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _make_signals(
        index: pd.DatetimeIndex,
        signal: np.ndarray,
        confidence: np.ndarray = None,
        metadata: list = None,
    ) -> pd.DataFrame:
        """
        Build a standardised signal DataFrame.

        Args:
            index:      DatetimeIndex aligned to forecast dates
            signal:     Array of float values in [-1, 1]
            confidence: Optional array of float values in [0, 1];
                        defaults to abs(signal)
            metadata:   Optional list of dicts; defaults to [{}] * len(index)
        Returns:
            DataFrame with DatetimeIndex and columns [signal, confidence, metadata]
        """
        if confidence is None:
            confidence = np.abs(signal)
        if metadata is None:
            metadata = [{}] * len(index)

        return pd.DataFrame(
            {
                "signal":     signal.astype(float),
                "confidence": confidence.astype(float),
                "metadata":   [json.dumps(m) for m in metadata],
            },
            index=index,
        )


# ---------------------------------------------------------------------------
# Concrete strategy implementations
# ---------------------------------------------------------------------------

class DirectionalPCEStrategy(BaseStrategy):
    """
    Simplest defensible PCE strategy: long when forecast PCE change is positive,
    short when negative, flat when zero.

    Academic rationale: consumer discretionary spending (XLY) has a documented
    positive relationship with PCE growth.  A forecast of rising PCE → long XLY;
    falling PCE → short XLY.

    Params
    ------
    (none — fully determined by forecast sign)
    """

    @property
    def name(self) -> str:
        return "directional_pce"

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)
        df = data.forecasts.copy().sort_values("date").set_index("date")

        raw_signal = np.sign(df["y_pred"].values).astype(float)
        metadata   = [{"y_pred": round(float(v), 6)} for v in df["y_pred"]]

        return self._make_signals(df.index, raw_signal, metadata=metadata)


class ThresholdPCEStrategy(BaseStrategy):
    """
    Trade only when the absolute forecast PCE change exceeds a threshold.
    Below the threshold the strategy is flat (signal = 0).

    This is a natural extension of DirectionalPCEStrategy — the threshold
    filters out low-conviction months when the model forecasts near-zero change,
    reducing whipsaw trades.

    Params
    ------
    threshold : float
        Minimum absolute forecast value required to enter a position.
        Units match the target transform (e.g. 0.2 for MoM % means ±0.2 pp).
        Default: 0.2.
    scale_confidence : bool
        If True, confidence is proportional to |y_pred| / threshold rather
        than binary.  Default: True.
    """

    def __init__(self, threshold: float = 0.2, scale_confidence: bool = True, **params):
        super().__init__(threshold=threshold, scale_confidence=scale_confidence, **params)
        self.threshold         = threshold
        self.scale_confidence  = scale_confidence

    @property
    def name(self) -> str:
        return f"threshold_pce_{self.threshold}"

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)
        df = data.forecasts.copy().sort_values("date").set_index("date")

        preds = df["y_pred"].values.astype(float)
        above = np.abs(preds) >= self.threshold

        raw_signal = np.where(above, np.sign(preds), 0.0)

        if self.scale_confidence:
            confidence = np.where(
                above,
                np.clip(np.abs(preds) / max(self.threshold, 1e-8), 0.0, 1.0),
                0.0,
            )
        else:
            confidence = np.where(above, 1.0, 0.0)

        metadata = [
            {"y_pred": round(float(p), 6), "threshold": self.threshold, "active": bool(a)}
            for p, a in zip(preds, above)
        ]

        return self._make_signals(df.index, raw_signal, confidence, metadata)


# ---------------------------------------------------------------------------
# Strategy registry — mirrors _MODEL_REGISTRY pattern from experiment.py
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict = {
    "DirectionalPCEStrategy": DirectionalPCEStrategy,
    "ThresholdPCEStrategy":   ThresholdPCEStrategy,
}


def build_strategy(name: str, **params) -> BaseStrategy:
    """
    Instantiate a strategy by class name, forwarding params as kwargs.

    Args:
        name:   Class name string (must exist in _STRATEGY_REGISTRY)
        **params: Passed to the strategy constructor
    Returns:
        Initialised BaseStrategy instance
    Raises:
        KeyError if name is not registered
    """
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. "
            f"Registered: {list(_STRATEGY_REGISTRY)}"
        )
    return _STRATEGY_REGISTRY[name](**params)


# ---------------------------------------------------------------------------
# Forecast loading helpers (unchanged interface, kept for pipeline continuity)
# ---------------------------------------------------------------------------

def load_best_forecasts(experiments_dir: Path = None) -> pd.DataFrame:
    """
    Load forecast CSV from the most recent experiment run.

    Returns:
        DataFrame with columns [date, y_true, y_pred, model_name, feature_set]
    """
    if experiments_dir is None:
        experiments_dir = Path(config["paths"]["experiments"])

    run_dirs = sorted(experiments_dir.glob("*_run"))
    if not run_dirs:
        raise FileNotFoundError(f"No experiment runs found in {experiments_dir}")

    latest        = run_dirs[-1]
    forecasts_path = latest / "forecasts.csv"
    print(f"[INFO] Loading forecasts from {latest.name}")
    return pd.read_csv(forecasts_path, parse_dates=["date"])


def select_best_model(
    forecasts_df: pd.DataFrame,
    metrics_df:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter forecast DataFrame to rows from the best-performing model.

    Falls back to config["trading"]["selected_model"] if present; otherwise
    selects the row with the lowest RMSE in metrics_df.

    Args:
        forecasts_df: Full forecasts from all trials
        metrics_df:   Summary metrics DataFrame from run_experiment()
    Returns:
        Forecasts from the single best model / feature_set combination
    """
    sel = config.get("trading", {}).get("selected_model", {})
    if sel.get("model_name") and sel.get("feature_set"):
        best_model = sel["model_name"]
        best_fs    = sel["feature_set"]
        print(f"[INFO] Config-selected model: {best_model} | features: {best_fs}")
    else:
        best_row   = metrics_df.dropna(subset=["rmse"]).nsmallest(1, "rmse").iloc[0]
        best_model = best_row["model_name"]
        best_fs    = best_row["feature_set"]
        print(f"[INFO] Auto-selected model: {best_model} | features: {best_fs}")

    mask = (
        (forecasts_df["model_name"]  == best_model) &
        (forecasts_df["feature_set"] == best_fs)
    )
    return forecasts_df[mask].copy()
