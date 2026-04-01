# src/data/store.py
import pandas as pd
from pathlib import Path
from src.utils.config import config

PROCESSED_DIR = Path(config["paths"]["processed_data"])
MASTER_PATH   = PROCESSED_DIR / "master.csv"
FSBI_PATH     = PROCESSED_DIR / "fsbi_long.csv"


def write_master(df: pd.DataFrame):
    """Write master DataFrame (BEA + FRED + USCB, no FSBI) to data/processed/master.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_PATH)
    print(f"[INFO] Master dataset written to {MASTER_PATH}")


def read_master() -> pd.DataFrame:
    """
    Load master DataFrame from data/processed/master.csv.
    Returns the clean wide frame (BEA + FRED + USCB) before feature engineering.
    panel.py joins FSBI and builds features on the panel-specific combined frame.
    """
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {MASTER_PATH}. "
            f"Run the data stage first."
        )
    return pd.read_csv(
        MASTER_PATH,
        index_col=config["data"]["date_col"],
        parse_dates=True,
    )


def write_fsbi(df: pd.DataFrame):
    """Write FSBI long-format DataFrame to data/processed/fsbi_long.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FSBI_PATH, index=False)
    print(f"[INFO] FSBI long data written to {FSBI_PATH} "
          f"({len(df)} rows, {df['Geo'].nunique()} geographies)")


def read_fsbi() -> pd.DataFrame:
    """
    Load FSBI long-format DataFrame from data/processed/fsbi_long.csv.
    panel.py calls this to filter and pivot FSBI per panel at experiment time.
    """
    if not FSBI_PATH.exists():
        raise FileNotFoundError(
            f"FSBI long data not found at {FSBI_PATH}. "
            f"Run the data stage first."
        )
    return pd.read_csv(FSBI_PATH, parse_dates=["date"])
