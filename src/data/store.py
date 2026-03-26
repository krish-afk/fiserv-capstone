# src/data/store.py
import pandas as pd
from pathlib import Path
from src.utils.config import config

PROCESSED_DIR = Path(config["paths"]["processed_data"])
MASTER_PATH   = PROCESSED_DIR / "master.csv"


def write_master(df: pd.DataFrame):
    """Write master DataFrame to data/processed/master.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_PATH)
    print(f"[INFO] Master dataset written to {MASTER_PATH}")


def read_master() -> pd.DataFrame:
    """
    Load master DataFrame from data/processed/master.csv.
    This is the single read point for all downstream components —
    features.py, transform.py, and panel.py all call this.
    """
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {MASTER_PATH}. "
            f"Run the data stage first."
        )
    return pd.read_csv(
        MASTER_PATH,
        index_col=config["data"]["date_col"],
        parse_dates=True
    )
