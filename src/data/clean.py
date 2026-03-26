# src/data/clean.py
import pandas as pd
from src.utils.config import config


def validate_schema(df: pd.DataFrame, source_name: str):
    """
    Assert that a raw DataFrame has required columns and a valid date index.
    Raises ValueError with a descriptive message on failure.
    TODO: define expected schema per source and assert here
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{source_name}: index must be DatetimeIndex")
    # TODO: add column presence checks per source


def align_frequencies(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Resample all DataFrames to the configured target frequency.
    Non-PCE sources that arrive at lower frequency are forward-filled
    up to max_fill_periods.

    Args:
        dfs: Dict of {source_name: DataFrame}
    Returns:
        Dict of frequency-aligned DataFrames
    """
    freq       = config["data"]["frequency"]
    method     = config["data"]["alignment"]["method"]
    max_fill   = config["data"]["alignment"]["max_fill_periods"]
    aligned    = {}

    for name, df in dfs.items():
        # Resample numeric columns only
        df_resampled = df.select_dtypes(include="number").resample(freq)

        if method == "ffill":
            aligned[name] = df_resampled.ffill(limit=max_fill)
        elif method == "interpolate":
            aligned[name] = df_resampled.interpolate()
            # TODO: add additional alignment methods as needed
        else:
            aligned[name] = df_resampled.mean()

    return aligned


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute remaining missing values after alignment.
    Default: forward-fill then backward-fill (handles edges).
    TODO: consider source-specific imputation strategies
    """
    return df.ffill().bfill()


def build_master(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all aligned DataFrames into a single master DataFrame.

    IMPORTANT: FSBI remains in long format until panel.py —
    this function pivots it to wide format with prefixed columns
    so the master table has one row per date.

    Args:
        dfs: Dict of aligned DataFrames from align_frequencies()
    Returns:
        Wide master DataFrame, date-indexed, one row per period
    """
    # Start with PCE as the spine
    master = dfs["bea"].copy()

    # Merge FRED indicators
    master = master.join(dfs["fred"], how="left")

    # Pivot FSBI from long to wide — one column per (geography, sector, metric)
    # TODO: implement pivot once FSBI column schema is confirmed
    # fsbi_wide = dfs["fsbi"].pivot_table(
    #     index="date",
    #     columns=["geography", "sector"],
    #     values="index_value"
    # )
    # fsbi_wide.columns = [f"fsbi_{g}_{s}" for g, s in fsbi_wide.columns]
    # master = master.join(fsbi_wide, how="left")

    master = impute_missing(master)
    return master


def run_cleaning(raw_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Master cleaning runner. Validates, aligns, and merges all sources.
    Returns the master DataFrame ready for feature engineering.
    """
    for name, df in raw_dfs.items():
        validate_schema(df, name)

    aligned = align_frequencies(raw_dfs)
    master  = build_master(aligned)

    print(f"[INFO] Master dataset built — "
          f"{len(master)} rows, {master.shape[1]} columns, "
          f"{master.index[0].date()} to {master.index[-1].date()}")
    return master
