# src/data/ingest.py
import os
import time
import shutil
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from fredapi import Fred

from src.utils.config import config

RAW_DIR = Path(config["paths"]["raw_data"])

_D            = config["data"]
START_DATE    = _D["start_date"]
END_DATE      = _D["end_date"]
MAX_RETRIES   = _D["max_retries"]
SLEEP_BETWEEN = _D["sleep_between"]


def _make_raw_path(source_name: str) -> Path:
    """Generate a timestamped raw output path for a given source."""
    today = datetime.now().strftime("%Y%m%d")
    return RAW_DIR / f"{source_name}_{today}.csv"


def _fetch_with_retry(fetch_fn, label, max_retries=MAX_RETRIES, sleep_between=SLEEP_BETWEEN):
    """
    Call fetch_fn(), retrying up to max_retries times with sleep_between delay.
    Returns the result on success, or None after all attempts are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(sleep_between)
            else:
                print(f"[SKIP] {label} after {max_retries} attempts: {e}")
                return None


def _call_bea():
    cfg = config["data"]["sources"]["bea"]
    resp = requests.get(
        "https://apps.bea.gov/api/data",
        params={
            "UserID": os.environ["BEA_API_KEY"],
            "Method": "GetData",
            "DataSetName": cfg["data_set"],
            "TableName": cfg["table_name"],
            "Frequency": cfg["frequency"],
            "Year": cfg["year"],
            "ResultFormat": "JSON",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_bea() -> Path:
    """
    Pull PCE data from BEA NIPA API and write to data/raw/.

    Returns:
        Path to written CSV file
    """
    cfg         = config["data"]["sources"]["bea"]
    table_name  = cfg["table_name"]
    line_number = cfg["line_number"]
    out_path    = _make_raw_path("bea")

    raw = _fetch_with_retry(_call_bea, label=f"BEA {table_name}")
    if raw is None:
        return out_path

    records = raw["BEAAPI"]["Results"]["Data"]
    rows = []
    for r in records:
        if int(r["LineNumber"]) != line_number:
            continue
        value = pd.to_numeric(r["DataValue"].replace(",", ""), errors="coerce")
        if pd.isna(value):
            continue
        rows.append({"date": pd.to_datetime(r["TimePeriod"], format="%YM%m"), "value": value})

    df = pd.DataFrame(rows)
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)

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
    series   = series or config["data"]["sources"]["fred"]["series"]
    out_path = _make_raw_path("fred")

    fred = Fred(api_key=os.environ["FRED_API_KEY"])

    frames = []
    for sid in series:
        result = _fetch_with_retry(
            lambda s=sid: fred.get_series(
                s, observation_start=START_DATE, observation_end=END_DATE
            ).rename(s),
            label=f"FRED {sid}",
        )
        if result is not None:
            frames.append(result)

    if not frames:
        print(f"[WARN] No FRED series fetched; {out_path} not written")
        return out_path

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index().reset_index()
    df.to_csv(out_path, index=False)

    print(f"[INFO] FRED data written to {out_path}")
    return out_path


def _call_uscb():
    start_ym = pd.to_datetime(START_DATE).strftime("%Y-%m")
    end_ym   = pd.to_datetime(END_DATE).strftime("%Y-%m")
    resp = requests.get(
        "https://api.census.gov/data/timeseries/eits/marts",
        params={
            "get": "cell_value,time_slot_id,category_code,seasonally_adj",
            "time": f"from {start_ym} to {end_ym}",
            "for": "us:*",
            "key": os.environ["CENSUS_API_KEY"],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_uscb() -> Path:
    """
    Pull MRTS data from US Census Bureau API and write to data/raw/.

    Returns:
        Path to written CSV file
    """
    cfg            = config["data"]["sources"]["uscb"]
    category_code  = str(cfg.get("category_code", "44000"))
    seasonally_adj = cfg.get("seasonally_adj", "no")
    out_path       = _make_raw_path("uscb")

    raw = _fetch_with_retry(_call_uscb, label="USCB MRTS")
    if raw is None:
        return out_path

    headers, *rows = raw
    df = pd.DataFrame(rows, columns=headers)
    df = df[
        (df["category_code"] == category_code)
        & (df["seasonally_adj"] == seasonally_adj)
        & (df["time_slot_id"] == "M")   # "M" = monthly value; excludes CV rows
    ]

    df = df.rename(columns={"cell_value": "value", "time": "date"})[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"[INFO] USCB data written to {out_path}")
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
    paths["uscb"] = fetch_uscb()

    fsbi_filename = config["data"]["sources"]["fsbi"]["filename"]
    fsbi_src = RAW_DIR / fsbi_filename
    if fsbi_src.exists():
        paths["fsbi"] = fsbi_src
    else:
        print(f"[WARN] FSBI file not found at {fsbi_src} — skipping")

    return paths
