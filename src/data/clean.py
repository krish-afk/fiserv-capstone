# src/data/clean.py
import pandas as pd
from src.utils.config import config


def validate_schema(df: pd.DataFrame, source_name: str):
    """
    Assert that a raw DataFrame has the expected structure for its source.

    - BEA / FRED / USCB: must have a DatetimeIndex named 'date'.
    - FSBI: long format, so no DatetimeIndex; must have a 'date' column of
      datetime dtype plus the standard dimension columns.

    Raises ValueError with a descriptive message on failure.
    """
    if source_name == "fsbi":
        if "date" not in df.columns:
            raise ValueError("fsbi: missing 'date' column")
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise ValueError("fsbi: 'date' column must be datetime dtype")
        for col in ["Geo", "Sector Name", "Sub-Sector Name"]:
            if col not in df.columns:
                raise ValueError(f"fsbi: missing expected dimension column '{col}'")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{source_name}: index must be a DatetimeIndex")
        if df.index.name != "date":
            raise ValueError(f"{source_name}: DatetimeIndex must be named 'date'")


def align_frequencies(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Resample wide-format DataFrames (BEA, FRED, USCB) to the configured target frequency.

    FSBI is excluded — it is already monthly and is in long format, so resample()
    would not apply. FSBI is passed through unchanged and separated out in
    run_cleaning() before build_master() is called.

    Non-monthly sources (e.g. quarterly GDP) are forward-filled up to max_fill_periods.
    """
    freq     = config["data"]["frequency"]
    method   = config["data"]["alignment"]["method"]
    max_fill = config["data"]["alignment"]["max_fill_periods"]
    aligned  = {}

    for name, df in dfs.items():
        if name == "fsbi":
            aligned[name] = df
            continue

        df_resampled = df.select_dtypes(include="number").resample(freq)

        if method == "ffill":
            aligned[name] = df_resampled.ffill(limit=max_fill)
        elif method == "interpolate":
            aligned[name] = df_resampled.interpolate()
        else:
            aligned[name] = df_resampled.mean()

    return aligned


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute remaining missing values in a wide-format DataFrame.
    Forward-fill then backward-fill handles interior gaps and edge cases.
    Applied to the master DataFrame after joining all wide sources.
    """
    return df.ffill().bfill()


def build_master(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge BEA + FRED + USCB into a single wide master DataFrame.

    FSBI is intentionally excluded here — it is in long format and requires
    per-panel geographic/sector filtering before pivoting. That logic lives
    in panel.py so the pivot stays narrow regardless of how many geo/sector
    combinations exist in the raw FSBI data.

    Build order:
      1. BEA PCE as the date spine
      2. FRED macro indicators (left join on date)
      3. USCB MRTS (left join on date)

    Remaining NaNs are imputed with global ffill/bfill after joining.
    """
    master = dfs["bea"].copy()
    master = master.join(dfs["fred"], how="left")
    master = master.join(dfs["uscb"], how="left")
    master = impute_missing(master)
    return master


def run_cleaning(raw_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master cleaning runner. Validates, aligns, and merges BEA + FRED + USCB.
    Returns FSBI separately in long format for per-panel processing.

    Returns:
        master:    Wide DataFrame (BEA + FRED + USCB), date-indexed, ready for
                   feature engineering. Written to data/processed/master.csv.
        fsbi_long: Raw long-format FSBI DataFrame. Written to
                   data/processed/fsbi_long.csv. panel.py filters and pivots
                   this per panel at experiment time.
    """
    for name, df in raw_dfs.items():
        validate_schema(df, name)

    aligned   = align_frequencies(raw_dfs)
    fsbi_long = aligned.pop("fsbi")   # separate before master build
    master    = build_master(aligned)

    print(f"[INFO] Master dataset built — "
          f"{len(master)} rows, {master.shape[1]} columns, "
          f"{master.index[0].date()} to {master.index[-1].date()}")
    print(f"[INFO] FSBI long data retained — "
          f"{len(fsbi_long)} rows, "
          f"{fsbi_long['Geo'].nunique()} geographies")
    return master, fsbi_long
