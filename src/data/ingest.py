# src/data/ingest.py
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.utils.config import config

RAW_DIR = Path(config["paths"]["raw_data"])


def _make_raw_path(source_name: str) -> Path:
    """Generate a timestamped raw output path for a given source."""
    today = datetime.now().strftime("%Y%m%d")
    return RAW_DIR / f"{source_name}_{today}.csv"


def fetch_bea(series_id: str = None) -> Path:
    """
    Pull PCE data from BEA API and write to data/raw/.

    Args:
        series_id: BEA series identifier; defaults to config value
    Returns:
        Path to written CSV file
    """
    series_id = series_id or config["data"]["sources"]["bea"]["series_id"]
    out_path = _make_raw_path("bea")

    # TODO: implement BEA API call
    # Recommended library: requests
    # BEA API docs: https://apps.bea.gov/API/docs/index.htm
    # Key parameters: UserID (API key), Method, DataSetName, SeriesName
    # df = pd.DataFrame(...)  # parse API response into DataFrame
    # df.to_csv(out_path, index=False)

    print(f"[INFO] BEA data written to {out_path}")
    return out_path


def fetch_fred(series: list = None) -> Path:
    """
    Pull macroeconomic indicators from FRED and write to data/raw/.

    Args:
        series: List of FRED series IDs; defaults to config value
    Returns:
        Path to written CSV file
    """
    series = series or config["data"]["sources"]["fred"]["series"]
    out_path = _make_raw_path("fred")

    # TODO: implement using fredapi or pandas_datareader
    # pip install fredapi
    # from fredapi import Fred
    # fred = Fred(api_key=os.environ["FRED_API_KEY"])
    # Each series fetched separately then merged on date index
    # df.to_csv(out_path, index=False)

    print(f"[INFO] FRED data written to {out_path}")
    return out_path


def copy_static(source_name: str, source_path: Path) -> Path:
    """
    Copy a pre-downloaded static CSV into data/raw/ with standard naming.
    Use this for sources like FSBI that arrive as manual downloads.

    Args:
        source_name: Short identifier (e.g. "fsbi")
        source_path: Path to the source file
    Returns:
        Path to standardized copy in data/raw/
    """
    import shutil
    out_path = _make_raw_path(source_name)
    shutil.copy(source_path, out_path)
    print(f"[INFO] {source_name} copied to {out_path}")
    return out_path


def run_ingestion() -> dict[str, Path]:
    """
    Master ingestion runner. Calls all source fetchers and returns
    a dict of {source_name: raw_csv_path}.
    """
    paths = {}
    paths["bea"]  = fetch_bea()
    paths["fred"] = fetch_fred()

    # FSBI is static — point to pre-downloaded file
    fsbi_filename = config["data"]["sources"]["fsbi"]["filename"]
    fsbi_src = RAW_DIR / fsbi_filename
    if fsbi_src.exists():
        paths["fsbi"] = fsbi_src
    else:
        print(f"[WARN] FSBI file not found at {fsbi_src} — skipping")

    return paths
