from __future__ import annotations

"""
src/trading/strategy.py

Strategy abstractions + dynamic strategy loading helpers.

What this adds
--------------
1. Team members can drop their own strategy file into src/trading/strategies/.
2. config.yaml can point at the file/class to run.
3. Strategies declare the tickers they need, and the trading pipeline can fetch
   those tickers automatically.
4. Strategies can return either:
      - legacy single-ticker signals (signal/confidence/metadata), or
      - generic multi-asset weights via weight__TICKER columns.

Recommended config.yaml shape
-----------------------------
trading:
  strategy:
    file: krish_trade_strategy.py
    class: KrishTradeStrategy
    params:
      bullish_threshold: 0.75
      bearish_threshold: -0.75

  forecast_panels:
    primary: pce
    mrts: mrts

  market_data:
    tickers: []         # optional extras; strategy tickers are always included
    start_date: 2018-01-01
    end_date: 2026-01-01
    source: yfinance
"""

import importlib.util
import inspect
import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import config


DEFAULT_STRATEGY_DIR = Path(__file__).resolve().parent / "strategies"
_WEIGHT_PREFIX = "weight__"


@dataclass
class StrategyData:
    """
    Container for all inputs a strategy may need.

    Attributes
    ----------
    forecasts
        Primary forecast panel for the strategy. Usually PCE.
        Expected columns include at least: date | y_true | y_pred.
    prices
        Date-indexed OHLCV data with columns in the canonical schema:
        {ticker_lower}_open/high/low/close/volume.
    macro
        Optional macro frame (usually latest FRED pull).
    mrts
        Optional MRTS forecast frame when the strategy needs both PCE and MRTS.
    cfg
        Full loaded config so a strategy can read optional config values.
    context
        Free-form auxiliary objects the pipeline can attach.
    """

    forecasts: pd.DataFrame
    prices: Optional[pd.DataFrame] = None
    macro: Optional[pd.DataFrame] = None
    mrts: Optional[pd.DataFrame] = None
    cfg: Optional[dict] = None
    context: dict[str, Any] = field(default_factory=dict)

    def validate(self, required: set[str]) -> None:
        missing = [name for name in required if getattr(self, name, None) is None]
        if missing:
            raise ValueError(
                f"Strategy requires inputs {missing} but they are None in StrategyData."
            )

    @property
    def config(self) -> dict:
        return self.cfg or {}

    def copy_with(self, **updates: Any) -> "StrategyData":
        payload = {
            "forecasts": self.forecasts,
            "prices": self.prices,
            "macro": self.macro,
            "mrts": self.mrts,
            "cfg": self.cfg,
            "context": dict(self.context),
        }
        payload.update(updates)
        return StrategyData(**payload)


class BaseStrategy(ABC):
    """
    Base class for all strategies.

    Strategy authors should typically implement:
      - name
      - generate_signals(data)
      - tickers (if the strategy trades securities)
      - required_inputs (when macro / mrts / prices are needed)

    Output formats
    --------------
    A strategy may return one of two DataFrame formats:

    1) Legacy single-ticker format
       index=date, columns=[signal, confidence, metadata]

    2) Preferred multi-asset format
       index=date, columns like:
           weight__SPY, weight__TLT, cash_weight, confidence, metadata

       Weight columns are interpreted as target portfolio weights for that period.
       Negative weights are allowed for short exposure.
    """

    def __init__(self, **params: Any):
        self.params = params

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        ...

    @property
    def required_inputs(self) -> set[str]:
        return {"forecasts"}

    @property
    def tickers(self) -> list[str]:
        return []

    @property
    def default_ticker(self) -> Optional[str]:
        if self.tickers:
            return self.tickers[0].upper()
        try:
            return get_active_strategy_config(config).get("params", {}).get("ticker")
        except KeyError:
            return None

    @staticmethod
    def price_col(ticker: str, field: str = "close") -> str:
        return f"{ticker.lower()}_{field.lower()}"

    @classmethod
    def get_price_series(
        cls,
        prices: pd.DataFrame,
        ticker: str,
        field: str = "close",
    ) -> pd.Series:
        if prices is None:
            raise ValueError("prices is None. Strategy requested market data that was not loaded.")

        col = cls.price_col(ticker, field)
        normalized = prices.copy()
        normalized.columns = [str(c).lower() for c in normalized.columns]

        if col not in normalized.columns:
            available = [c for c in normalized.columns if c.endswith(f"_{field.lower()}")]
            raise KeyError(
                f"Column '{col}' not found in prices. "
                f"Available {field} columns (first 10): {available[:10]}"
            )
        return normalized[col]

    @staticmethod
    def _coerce_forecast_index(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
            out = out.sort_values("date").set_index("date")
        else:
            out.index = pd.to_datetime(out.index)
            out = out.sort_index()
        return out

    @staticmethod
    def _make_signals(
        index: pd.DatetimeIndex,
        signal: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        if confidence is None:
            confidence = np.abs(signal)
        if metadata is None:
            metadata = [{} for _ in range(len(index))]

        return pd.DataFrame(
            {
                "signal": np.asarray(signal, dtype=float),
                "confidence": np.asarray(confidence, dtype=float),
                "metadata": [json.dumps(m) for m in metadata],
            },
            index=pd.DatetimeIndex(index, name="date"),
        )

    @staticmethod
    def _make_weight_frame(
        index: pd.DatetimeIndex,
        weights: dict[str, np.ndarray | list[float]],
        cash_weight: Optional[np.ndarray | list[float]] = None,
        confidence: Optional[np.ndarray | list[float]] = None,
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        data: dict[str, Any] = {}
        n = len(index)

        for ticker, values in weights.items():
            col = f"{_WEIGHT_PREFIX}{ticker.upper()}"
            data[col] = np.asarray(values, dtype=float)

        if cash_weight is None:
            exposure = np.zeros(n, dtype=float)
            for values in data.values():
                exposure = exposure + np.abs(values)
            cash_weight = np.clip(1.0 - exposure, 0.0, 1.0)

        if confidence is None:
            confidence = np.ones(n, dtype=float)

        if metadata is None:
            metadata = [{} for _ in range(n)]

        data["cash_weight"] = np.asarray(cash_weight, dtype=float)
        data["confidence"] = np.asarray(confidence, dtype=float)
        data["metadata"] = [json.dumps(m) for m in metadata]

        return pd.DataFrame(data, index=pd.DatetimeIndex(index, name="date"))


class DirectionalPCEStrategy(BaseStrategy):
    def __init__(self, ticker: str = "XLY", **params: Any):
        super().__init__(ticker=ticker, **params)
        self._ticker = ticker.upper()

    @property
    def name(self) -> str:
        return f"directional_pce_{self._ticker.lower()}"

    @property
    def tickers(self) -> list[str]:
        return [self._ticker]

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)
        df = self._coerce_forecast_index(data.forecasts)

        preds = df["y_pred"].astype(float).to_numpy()
        signal = np.sign(preds)
        metadata = [{"y_pred": round(float(v), 6), "ticker": self._ticker} for v in preds]

        return self._make_weight_frame(
            index=df.index,
            weights={self._ticker: signal},
            confidence=np.abs(signal),
            metadata=metadata,
        )


class ThresholdPCEStrategy(BaseStrategy):
    def __init__(
        self,
        ticker: str = "XLY",
        threshold: float = 0.2,
        scale_confidence: bool = True,
        **params: Any,
    ):
        super().__init__(
            ticker=ticker,
            threshold=threshold,
            scale_confidence=scale_confidence,
            **params,
        )
        self._ticker = ticker.upper()
        self.threshold = float(threshold)
        self.scale_confidence = bool(scale_confidence)

    @property
    def name(self) -> str:
        return f"threshold_pce_{self._ticker.lower()}_{self.threshold}"

    @property
    def tickers(self) -> list[str]:
        return [self._ticker]

    def generate_signals(self, data: StrategyData) -> pd.DataFrame:
        data.validate(self.required_inputs)
        df = self._coerce_forecast_index(data.forecasts)
        preds = df["y_pred"].astype(float).to_numpy()

        active = np.abs(preds) >= self.threshold
        signal = np.where(active, np.sign(preds), 0.0)

        if self.scale_confidence:
            confidence = np.where(
                active,
                np.clip(np.abs(preds) / max(self.threshold, 1e-8), 0.0, 1.0),
                0.0,
            )
        else:
            confidence = np.where(active, 1.0, 0.0)

        metadata = [
            {
                "ticker": self._ticker,
                "y_pred": round(float(pred), 6),
                "threshold": self.threshold,
                "active": bool(is_active),
            }
            for pred, is_active in zip(preds, active)
        ]

        return self._make_weight_frame(
            index=df.index,
            weights={self._ticker: signal},
            confidence=confidence,
            metadata=metadata,
        )


_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "DirectionalPCEStrategy": DirectionalPCEStrategy,
    "ThresholdPCEStrategy": ThresholdPCEStrategy,
}


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    if not issubclass(cls, BaseStrategy):
        raise TypeError(f"{cls!r} must inherit from BaseStrategy")
    _STRATEGY_REGISTRY[cls.__name__] = cls
    return cls


def _normalize_strategy_filename(strategy_file: str) -> str:
    strategy_file = strategy_file.strip()
    return strategy_file if strategy_file.endswith(".py") else f"{strategy_file}.py"


def _load_strategy_module(strategy_file: str, strategy_dir: Optional[Path] = None):
    strategy_dir = Path(strategy_dir or DEFAULT_STRATEGY_DIR)
    filename = _normalize_strategy_filename(strategy_file)
    path = strategy_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    module_name = f"src.trading.user_strategy_{re.sub(r'[^a-zA-Z0-9_]', '_', path.stem)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_strategy_class_from_module(module, class_name: Optional[str] = None) -> type[BaseStrategy]:
    if class_name:
        try:
            candidate = getattr(module, class_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Strategy class '{class_name}' not found in module '{module.__name__}'"
            ) from exc

        if not inspect.isclass(candidate) or not issubclass(candidate, BaseStrategy):
            raise TypeError(
                f"{class_name} exists but does not inherit from BaseStrategy"
            )
        return candidate

    candidates: list[type[BaseStrategy]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseStrategy) and obj is not BaseStrategy and obj.__module__ == module.__name__:
            candidates.append(obj)

    if not candidates:
        raise ValueError(
            f"No BaseStrategy subclass found in module '{module.__name__}'."
        )
    if len(candidates) > 1:
        names = [cls.__name__ for cls in candidates]
        raise ValueError(
            f"Multiple strategy classes found in '{module.__name__}': {names}. "
            "Set trading.strategy.class in config.yaml."
        )
    return candidates[0]


def get_active_strategy_config(cfg: Optional[dict] = None) -> dict:
    cfg = cfg or config
    trading_cfg = cfg.get("trading", {})
    name = trading_cfg.get("active_strategy", "")
    if not name:
        raise KeyError("trading.active_strategy is not set in config.")
    strategies = trading_cfg.get("strategies", {})
    if name not in strategies:
        raise KeyError(f"Strategy '{name}' not found under trading.strategies in config.")
    return strategies[name]


def build_strategy(
    name: Optional[str] = None,
    **params: Any,
) -> BaseStrategy:
    if not name:
        raise ValueError("build_strategy() requires a strategy class name.")
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Registered: {sorted(_STRATEGY_REGISTRY)}")
    return _STRATEGY_REGISTRY[name](**params)


def load_strategy_from_file(
    strategy_file: str,
    class_name: Optional[str] = None,
    strategy_dir: Optional[Path] = None,
    params: Optional[dict[str, Any]] = None,
) -> BaseStrategy:
    module = _load_strategy_module(strategy_file, strategy_dir=strategy_dir)
    cls = _resolve_strategy_class_from_module(module, class_name=class_name)
    register_strategy(cls)
    return cls(**(params or {}))


def load_configured_strategy(
    cfg: Optional[dict] = None,
    strategy_dir: Optional[Path] = None,
) -> BaseStrategy:
    cfg = cfg or config
    strategy_cfg = get_active_strategy_config(cfg)
    class_name = strategy_cfg.get("class", "")
    if not class_name:
        raise KeyError("Active strategy config is missing 'class'.")
    params = strategy_cfg.get("params", {}) or {}

    if class_name in _STRATEGY_REGISTRY:
        return _STRATEGY_REGISTRY[class_name](**params)

    scan_dir = Path(strategy_dir or DEFAULT_STRATEGY_DIR)
    for py_file in sorted(scan_dir.glob("*.py")):
        try:
            module = _load_strategy_module(py_file.name, strategy_dir=scan_dir)
        except Exception:
            continue
        if not hasattr(module, class_name):
            continue
        cls = _resolve_strategy_class_from_module(module, class_name=class_name)
        register_strategy(cls)
        return cls(**params)

    raise KeyError(
        f"Strategy class '{class_name}' not found in registry or in {scan_dir}. "
        f"Registered strategies: {sorted(_STRATEGY_REGISTRY)}"
    )


def load_latest_run_frames(experiments_dir: Path = None) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    experiments_dir = Path(experiments_dir or config["paths"]["experiments"])
    run_dirs = sorted(experiments_dir.glob("*_run"))
    if not run_dirs:
        raise FileNotFoundError(f"No experiment runs found in {experiments_dir}")

    latest_run = run_dirs[-1]
    forecasts_path = latest_run / "forecasts.csv"
    metrics_path = latest_run / "metrics.csv"

    if not forecasts_path.exists():
        raise FileNotFoundError(f"Missing forecasts.csv in {latest_run}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv in {latest_run}")

    forecasts_df = pd.read_csv(forecasts_path, parse_dates=["date"])
    metrics_df = pd.read_csv(metrics_path)
    return latest_run, forecasts_df, metrics_df


def load_best_forecasts(experiments_dir: Path = None) -> pd.DataFrame:
    _, forecasts_df, _ = load_latest_run_frames(experiments_dir=experiments_dir)
    return forecasts_df


def _panel_selection_config(panel_name: Optional[str]) -> dict[str, Any]:
    try:
        sel = get_active_strategy_config(config).get("selected_model", {})
    except KeyError:
        sel = {}
    if panel_name and isinstance(sel.get(panel_name), dict):
        return sel[panel_name]
    return sel


def select_best_model(
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    panel_name: Optional[str] = None,
) -> pd.DataFrame:
    working_forecasts = forecasts_df.copy()
    working_metrics = metrics_df.copy()

    if panel_name and "panel_name" in working_forecasts.columns:
        working_forecasts = working_forecasts[working_forecasts["panel_name"] == panel_name].copy()
    if panel_name and "panel_name" in working_metrics.columns:
        working_metrics = working_metrics[working_metrics["panel_name"] == panel_name].copy()

    if working_forecasts.empty:
        raise ValueError(f"No forecasts available for panel '{panel_name}'.")
    if working_metrics.empty:
        raise ValueError(f"No metrics available for panel '{panel_name}'.")

    sel = _panel_selection_config(panel_name)
    if sel.get("model_name") and sel.get("feature_set"):
        best_model = sel["model_name"]
        best_fs = sel["feature_set"]
    else:
        best_row = working_metrics.dropna(subset=["rmse"]).nsmallest(1, "rmse").iloc[0]
        best_model = best_row["model_name"]
        best_fs = best_row["feature_set"]

    mask = (
        (working_forecasts["model_name"] == best_model)
        & (working_forecasts["feature_set"] == best_fs)
    )
    selected = working_forecasts[mask].copy()
    if selected.empty:
        raise ValueError(
            f"Selected model '{best_model}' with feature set '{best_fs}' produced no forecasts"
            + (f" for panel '{panel_name}'" if panel_name else "")
            + "."
        )
    return selected.sort_values("date").reset_index(drop=True)


def load_best_forecasts_for_panel(
    panel_name: str,
    experiments_dir: Path = None,
) -> pd.DataFrame:
    _, forecasts_df, metrics_df = load_latest_run_frames(experiments_dir=experiments_dir)
    return select_best_model(forecasts_df, metrics_df, panel_name=panel_name)