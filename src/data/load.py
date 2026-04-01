# src/data/load.py
import pandas as pd
from pathlib import Path
from src.utils.config import config

RAW_DIR = Path(config["paths"]["raw_data"])


def _find_latest(source_name: str) -> Path:
    """Return path to the most recently written raw CSV for a given source."""
    matches = sorted(RAW_DIR.glob(f"{source_name}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No raw files found for source '{source_name}' in {RAW_DIR}"
        )
    return matches[-1]


def _load(path: Path, **read_kwargs) -> pd.DataFrame:
    """Load a CSV from an explicit path."""
    print(f"[INFO] Loading {path.name}")
    return pd.read_csv(path, **read_kwargs)


def load_bea(path: Path = None) -> pd.DataFrame:
    """
    Load raw BEA data. Returns DataFrame with columns [pce], date-indexed.

    ingest.fetch_bea() writes two columns: 'date' and 'pce'.
    """
    path = path or _find_latest("bea")
    df = _load(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def load_fred(path: Path = None) -> pd.DataFrame:
    """
    Load raw FRED data. Returns wide DataFrame with one column per series.

    ingest.fetch_fred() already renames columns from FRED IDs to human-readable
    snake_case names (e.g., CPIAUCSL → 'cpi'), so no rename is needed here.
    """
    path = path or _find_latest("fred")
    df = _load(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def load_uscb(path: Path = None) -> pd.DataFrame:
    """
    Load raw USCB MRTS data. Returns DataFrame with columns [mrts], date-indexed.

    ingest.fetch_uscb() writes two columns: 'date' and 'mrts'.
    """
    path = path or _find_latest("uscb")
    df = _load(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def load_fsbi(path: Path = None) -> pd.DataFrame:
    """
    Load FSBI data. Returns long-format DataFrame with columns:
        date | Geo | Sector Name | Sub-Sector Name | <metric columns>

    FSBI has metro/city and sector dimensions — keep long format so that
    clean.py can do group-level imputation before pivoting to wide format.

    Raw FSBI uses 'Period' as an integer YYYYMMDD date (e.g., 20190101).
    This function converts it to a proper datetime 'date' column.
    Note: index is NOT set because dates repeat across (Geo, Sector) groups.
    """
    if path is None:
        fsbi_filename = config["data"]["sources"]["fsbi"]["filename"]
        path = RAW_DIR / fsbi_filename
        if not path.exists():
            raise FileNotFoundError(
                f"FSBI file not found at {path}. "
                "Place the pre-downloaded fsbi_raw.csv in data/raw/ and re-run."
            )
    print(f"[INFO] Loading {path.name}")
    df = pd.read_csv(path)

    # Parse Period (YYYYMMDD integer) into a proper datetime column
    df["date"] = pd.to_datetime(df["Period"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.drop(columns=["Period"])
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_all_raw(paths: dict[str, Path] = None) -> dict[str, pd.DataFrame]:
    """
    Load all raw sources and return as a dict of DataFrames.
    This is the single entry point load.py exposes to clean.py.

    Args:
        paths: Optional dict of {source_name: Path} as returned by
               ingest.run_ingestion(). When provided, each loader reads
               the exact file written during this run rather than re-globbing
               for the latest. When None, each loader falls back to globbing
               for the most recently timestamped file in RAW_DIR.

    All four sources are required. A missing file raises FileNotFoundError.
    For FSBI specifically: place fsbi_raw.csv in data/raw/ before running.
    """
    paths = paths or {}
    return {
        "bea":  load_bea(paths.get("bea")),
        "fred": load_fred(paths.get("fred")),
        "uscb": load_uscb(paths.get("uscb")),
        "fsbi": load_fsbi(paths.get("fsbi")),
    }
