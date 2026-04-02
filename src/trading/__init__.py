# src/trading/__init__.py
from src.trading.strategy import (
    StrategyData,
    BaseStrategy,
    DirectionalPCEStrategy,
    ThresholdPCEStrategy,
    build_strategy,
    load_best_forecasts,
    select_best_model,
)
from src.trading.backtest import BacktestEngine
from src.trading.performance import (
    compute_metrics,
    sensitivity_analysis,
    summarise_sensitivity,
)