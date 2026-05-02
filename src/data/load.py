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

def _canonicalize_fsbi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize FSBI raw-column names so sales and transaction metrics survive
    the raw -> processed -> panel pipeline with predictable feature names.

    Raw FSBI exports may use labels like:
      - "Sales MOM % - SA"
      - "Real Sales MOM % - SA_normalized"
      - "Transaction YOY %  - SA"

    This function canonicalizes them to names like:
      - sales_mom_sa
      - transaction_mom_sa
      - transactional_index_sa
    """
    df = df.copy()

    # Clean header whitespace / hidden BOM characters.
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    aliases = {
        # Dimension columns
        "period": "Period",
        "geo": "Geo",
        "sector name": "Sector Name",
        "sub-sector name": "Sub-Sector Name",
        "sub sector name": "Sub-Sector Name",

        # SA sales metrics
        "sales index - sa": "sales_index_sa",
        "real sales index - sa": "sales_index_sa",
        "real sales index - sa_normalized": "sales_index_sa",
        "real sales index - sa normalized": "sales_index_sa",

        "sales mom % - sa": "sales_mom_sa",
        "real sales mom % - sa": "sales_mom_sa",
        "real sales mom % - sa_normalized": "sales_mom_sa",
        "real sales mom % - sa normalized": "sales_mom_sa",

        "sales yoy % - sa": "sales_yoy_sa",
        "real sales yoy % - sa": "sales_yoy_sa",
        "real sales yoy % - sa_normalized": "sales_yoy_sa",
        "real sales yoy % - sa normalized": "sales_yoy_sa",

        # SA transaction metrics
        "transactional index - sa": "transactional_index_sa",
        "transactional index - sa_normalized": "transactional_index_sa",
        "transactional index - sa normalized": "transactional_index_sa",

        "transaction mom % - sa": "transaction_mom_sa",
        "transaction mom % - sa_normalized": "transaction_mom_sa",
        "transaction mom % - sa normalized": "transaction_mom_sa",

        "transaction yoy % - sa": "transaction_yoy_sa",
        "transaction yoy % - sa_normalized": "transaction_yoy_sa",
        "transaction yoy % - sa normalized": "transaction_yoy_sa",

        # NSA sales metrics — clean.py will drop these later
        "sales index - nsa": "sales_index_nsa",
        "real sales index - nsa": "sales_index_nsa",
        "real sales index - nsa_normalized": "sales_index_nsa",
        "real sales index - nsa normalized": "sales_index_nsa",

        "sales mom % - nsa": "sales_mom_nsa",
        "real sales mom % - nsa": "sales_mom_nsa",
        "real sales mom % - nsa_normalized": "sales_mom_nsa",
        "real sales mom % - nsa normalized": "sales_mom_nsa",

        "sales yoy % - nsa": "sales_yoy_nsa",
        "real sales yoy % - nsa": "sales_yoy_nsa",
        "real sales yoy % - nsa_normalized": "sales_yoy_nsa",
        "real sales yoy % - nsa normalized": "sales_yoy_nsa",

        # NSA transaction metrics — clean.py will drop these later
        "transactional index - nsa": "transactional_index_nsa",
        "transactional index - nsa_normalized": "transactional_index_nsa",
        "transactional index - nsa normalized": "transactional_index_nsa",

        "transaction mom % - nsa": "transaction_mom_nsa",
        "transaction mom % - nsa_normalized": "transaction_mom_nsa",
        "transaction mom % - nsa normalized": "transaction_mom_nsa",

        "transaction yoy % - nsa": "transaction_yoy_nsa",
        "transaction yoy % - nsa_normalized": "transaction_yoy_nsa",
        "transaction yoy % - nsa normalized": "transaction_yoy_nsa",
    }

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        key = " ".join(key.split())
        if key in aliases:
            rename[col] = aliases[key]

    df = df.rename(columns=rename)

    required_dims = ["Period", "Geo", "Sector Name", "Sub-Sector Name"]
    missing_dims = [c for c in required_dims if c not in df.columns]
    if missing_dims:
        raise ValueError(f"FSBI raw file missing required columns: {missing_dims}")

    # Force all metric columns numeric.
    dim_cols = set(required_dims)
    for col in [c for c in df.columns if c not in dim_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def load_fsbi(path: Path = None) -> pd.DataFrame:
    """
    Load FSBI data. Returns long-format DataFrame with columns:
        date | Geo | Sector Name | Sub-Sector Name | <metric columns>

    FSBI has geography and sector dimensions, so keep long format.
    panel.py later filters and pivots it into model features.
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
    df = _canonicalize_fsbi_columns(df)

    # Parse Period robustly.
    # Supports both YYYYMMDD, e.g. 20190101, and ISO dates, e.g. 2019-01-01.
    period = df["Period"].astype(str).str.strip()

    parsed = pd.to_datetime(period, format="%Y%m%d", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(period, errors="coerce"))

    df["date"] = parsed
    df = df.drop(columns=["Period"])
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    transaction_cols = [c for c in df.columns if "transaction" in c.lower()]
    if transaction_cols:
        print(f"[INFO] FSBI transaction columns loaded: {transaction_cols}")
    else:
        print("[WARN] FSBI raw file loaded but no transaction columns were found")

    return df


def load_market_data(path: Path = None) -> pd.DataFrame:
    """
    Load raw market OHLCV data written by ingest.fetch_market_data().

    Returns a date-indexed DataFrame with flat column names:
        {ticker}_open | {ticker}_high | {ticker}_low | {ticker}_close | {ticker}_volume

    Source-agnostic: the same schema is produced regardless of whether the data
    came from yfinance, Alpha Vantage, or a local CSV.  To add a new source,
    update ingest.py to write in this schema.

    Args:
        path: Explicit path to the market CSV.  If None, finds the most recently
              written file matching data/raw/market_*.csv.
    Returns:
        date-indexed DataFrame with all tickers' OHLCV columns.
    """
    path = path or _find_latest("market")
    df = _load(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
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
