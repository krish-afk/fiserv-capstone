# src/trading/simulation.py
import pandas as pd
from src.trading.strategy import generate_signals


def run_simulation(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute simulated trades based on signals and return performance DataFrame.

    Args:
        signals_df: Output of generate_signals()
    Returns:
        DataFrame with columns [date, signal, return, cumulative_return]

    TODO: implement full simulation logic including:
        - Position sizing rules
        - Transaction cost assumptions
        - Stop-loss / risk management
        - Benchmark comparison (buy-and-hold)
    """
    # Stub: return signals with placeholder return columns
    df = signals_df.copy()
    df["return"] = None
    df["cumulative_return"] = None
    return df


def compute_performance_metrics(sim_df: pd.DataFrame) -> dict:
    """
    Compute trading performance metrics from simulation results.

    TODO: implement:
        - Sharpe ratio
        - Maximum drawdown
        - Win/loss ratio
        - ROI
    """
    # Stub: returns empty dict until simulation is implemented
    return {}
