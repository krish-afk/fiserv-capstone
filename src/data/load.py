# src/data/load.py
import pandas as pd
from pathlib import Path
from src.utils.config import config

RAW_DIR = Path(config["paths"]["raw_data"])


def _load_latest(source_name: str) -> pd.DataFrame:
    """
    Load the most recently written raw CSV for a given source.
    Matches files by prefix so timestamped filenames resolve correctly.
    """
    matches = sorted(RAW_DIR.glob(f"{source_name}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No raw files found for source '{source_name}' in {RAW_DIR}"
        )
    path = matches[-1]
    print(f"[INFO] Loading {path.name}")
    return pd.read_csv(path, parse_dates=[config["data"]["date_col"]])


def load_bea() -> pd.DataFrame:
    """
    Load raw BEA data. Returns DataFrame with columns [date, pce].
    TODO: adjust column renaming to match actual BEA API response schema
    """
    df = _load_latest("bea")
    # TODO: rename columns to standard names
    # df = df.rename(columns={"DATE": "date", "VALUE": "pce"})
    df = df.set_index("date").sort_index()
    return df


def load_fred() -> pd.DataFrame:
    """
    Load raw FRED data. Returns wide DataFrame with one column per series.
    TODO: adjust column renaming to match actual FRED response schema
    """
    df = _load_latest("fred")
    df = df.set_index("date").sort_index()
    return df


def load_fsbi() -> pd.DataFrame:
    """
    Load FSBI data. Returns DataFrame with columns:
    [date, geography, sector, index_value, ...]

    NOTE: FSBI has metro/city and sector dimensions — do NOT aggregate
    here. Keep the long format so panel.py can slice appropriately.
    TODO: confirm actual FSBI column names and adjust accordingly
    """
    df = _load_latest("fsbi")
    # TODO: parse date column, standardize geography/sector column names
    # Expected output columns: date | geography | sector | <metric cols>
    df = df.set_index("date").sort_index()
    return df


def load_all_raw() -> dict[str, pd.DataFrame]:
    """
    Load all raw sources and return as a dict of DataFrames.
    This is the single entry point load.py exposes to clean.py.
    """
    return {
        "bea":  load_bea(),
        "fred": load_fred(),
        "fsbi": load_fsbi(),
    }
