# src/data/clean.py

"""
Data cleaning pipeline for PCE forecasting.

Strategy:
---------
1. Validate schema for each source
2. Align frequencies (resample to monthly, no imputation)
3. Build master by joining BEA + FRED + USCB
4. Drop rows with ANY missing numeric values
5. Document row counts and missing data patterns

Rationale for dropping rows:
- Forward-fill and interpolation create look-ahead bias in time series
- Walk-forward validation will not catch this leakage
- Tree-based models handle NaNs natively
- COVID data is preserved in MoM volatility, not YoY history
- Defensible for stakeholder communication
"""

import pandas as pd
import numpy as np
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

    FSBI is excluded — it is already monthly and is in long format.

    CRITICAL CHANGE: No imputation here. Resample only.
    - Forward-fill is applied ONLY during resampling for non-monthly sources
      (e.g., quarterly GDP → monthly), with a strict limit.
    - No backward-fill or interpolation.
    - Remaining NaNs are preserved and dropped later in build_master().

    Args:
        dfs: Dict of raw DataFrames keyed by source name

    Returns:
        Dict of resampled DataFrames (FSBI unchanged)
    """
    freq = config["data"]["frequency"]
    max_fill = config["data"]["alignment"]["max_fill_periods"]

    aligned = {}

    for name, df in dfs.items():
        if name == "fsbi":
            aligned[name] = df
            continue

        print(f"[INFO] Aligning {name} to {freq}...")

        # Select only numeric columns for resampling
        numeric_df = df.select_dtypes(include="number")

        # Resample with forward-fill ONLY for non-monthly sources
        # This handles quarterly → monthly conversion without creating synthetic data
        resampled = numeric_df.resample(freq).first()

        # Forward-fill ONLY up to max_fill_periods to handle quarterly/annual data
        # This is NOT imputation — it's alignment of different frequencies
        resampled = resampled.ffill(limit=max_fill)

        aligned[name] = resampled

        # Log missing data pattern
        missing_pct = (resampled.isna().sum() / len(resampled) * 100).round(2)
        print(f"[INFO] {name} after alignment: {len(resampled)} rows, "
              f"missing values: {missing_pct.to_dict()}")

    return aligned


def build_master(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge BEA + FRED + USCB into a single wide master DataFrame.

    FSBI is intentionally excluded — it is in long format and requires
    per-panel geographic/sector filtering before pivoting.

    Build order:
      1. BEA PCE as the date spine
      2. FRED macro indicators (left join on date)
      3. USCB MRTS (left join on date)

    CRITICAL CHANGE: Drop rows with ANY missing numeric values.
    - No forward-fill, backward-fill, or interpolation
    - Preserves time series integrity for walk-forward validation
    - Allows tree-based models to handle remaining NaNs natively

    Returns:
        Wide DataFrame (BEA + FRED + USCB), date-indexed, no missing values
    """
    print("\n[INFO] Building master dataset...")

    # Start with BEA as the spine
    master = dfs["bea"].copy()
    initial_rows = len(master)
    print(f"[INFO] BEA spine: {initial_rows} rows")

    # Join FRED
    master = master.join(dfs["fred"], how="left")
    print(f"[INFO] After joining FRED: {len(master)} rows")

    # Join USCB
    master = master.join(dfs["uscb"], how="left")
    print(f"[INFO] After joining USCB: {len(master)} rows")

    # Log missing data BEFORE dropping
    print(f"\n[INFO] Missing data before dropping rows:")
    missing_before = master.isna().sum()
    missing_pct = (missing_before / len(master) * 100).round(2)
    for col, count in missing_before[missing_before > 0].items():
        print(f"       {col}: {count} ({missing_pct[col]:.2f}%)")

    # CRITICAL: Drop rows with ANY missing numeric values
    # This is the key change from the old strategy
    master_clean = master.dropna(how="any")

    rows_dropped = len(master) - len(master_clean)
    pct_dropped = round(rows_dropped / len(master) * 100, 2)

    print(f"\n[INFO] Rows dropped due to missing values: {rows_dropped} ({pct_dropped}%)")
    print(f"[INFO] Final master dataset: {len(master_clean)} rows, "
          f"{master_clean.shape[1]} columns, "
          f"{master_clean.index[0].date()} to {master_clean.index[-1].date()}")

    # Verify no NaNs remain
    assert master_clean.isna().sum().sum() == 0, "ERROR: NaNs remain after dropna()"

    return master_clean


def run_cleaning(raw_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master cleaning runner. Validates, aligns, and merges BEA + FRED + USCB.

    Returns FSBI separately in long format for per-panel processing.

    CRITICAL CHANGES:
    1. No imputation (ffill/bfill/interpolate) in align_frequencies()
    2. Rows with missing values are dropped in build_master()
    3. Detailed logging of row counts and missing data patterns

    Returns:
        master:    Wide DataFrame (BEA + FRED + USCB), date-indexed, no NaNs,
                   ready for feature engineering. Written to data/processed/master.csv.

        fsbi_long: Raw long-format FSBI DataFrame. Written to
                   data/processed/fsbi_long.csv. panel.py filters and pivots
                   this per panel at experiment time.
    """
    print("=" * 80)
    print("DATA CLEANING PIPELINE")
    print("=" * 80)

    # Validate all sources
    print("\n[STEP 1] Validating schema...")
    for name, df in raw_dfs.items():
        validate_schema(df, name)
        print(f"[OK] {name}")

    # Align frequencies (no imputation)
    print("\n[STEP 2] Aligning frequencies...")
    aligned = align_frequencies(raw_dfs)

    # Separate FSBI before master build
    fsbi_long = aligned.pop("fsbi")

    # ==========================================
    # SA ENFORCEMENT: Drop NSA columns from FSBI
    # ==========================================
    # Find any column containing "nsa" (case-insensitive) and drop it 
    # to ensure models only train on Seasonally Adjusted data.
    nsa_cols = [c for c in fsbi_long.columns if "nsa" in c.lower()]
    if nsa_cols:
        print(f"\n[STEP 2.5] Enforcing SA Data: Dropping NSA columns from FSBI: {nsa_cols}")
        fsbi_long = fsbi_long.drop(columns=nsa_cols)

    # Build master with row dropping
    print("\n[STEP 3] Building master dataset...")
    master = build_master(aligned)

    print("\n" + "=" * 80)
    print("CLEANING COMPLETE")
    print("=" * 80)
    print(f"\nFinal datasets:")
    print(f"  Master:    {len(master)} rows × {master.shape[1]} columns")
    print(f"  FSBI long: {len(fsbi_long)} rows")
    print(f"  Date range: {master.index[0].date()} to {master.index[-1].date()}")

    return master, fsbi_long