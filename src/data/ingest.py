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
    series_name = cfg["series_name"]
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
        rows.append({"date": pd.to_datetime(r["TimePeriod"], format="%YM%m"), series_name: value})

    df = pd.DataFrame(rows)
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"[INFO] BEA data written to {out_path}")
    return out_path


def fetch_fred(series: dict = None) -> Path:
    """
    Pull macroeconomic indicators from FRED and write to data/raw/.

    Args:
        series: Dict of {FRED_series_id: column_name}; defaults to config value.
                Column names are the snake_case names used throughout the pipeline.
    Returns:
        Path to written CSV file
    """
    series_cfg = series or config["data"]["sources"]["fred"]["series"]
    out_path   = _make_raw_path("fred")

    fred = Fred(api_key=os.environ["FRED_API_KEY"])

    frames = []
    for sid, col_name in series_cfg.items():
        result = _fetch_with_retry(
            lambda s=sid, n=col_name: fred.get_series(
                s, observation_start=START_DATE, observation_end=END_DATE
            ).rename(n),
            label=f"FRED {sid} → {col_name}",
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

    print(f"[INFO] FRED data written to {out_path} ({len(frames)} series)")
    return out_path


def _call_uscb():
    series_name = config["data"]["sources"]["uscb"]["series_name"]
    start_ym = pd.to_datetime(START_DATE).strftime("%Y-%m")
    end_ym   = pd.to_datetime(END_DATE).strftime("%Y-%m")
    params = {
        "get": "cell_value,time_slot_id,category_code,seasonally_adj,data_type_code",
        "time": f"from {start_ym} to {end_ym}",
        "key": os.environ["USCB_API_KEY"],
    }
    url = f"https://api.census.gov/data/timeseries/eits/{series_name}"
    # Reconstruct URL with key masked for safe logging
    safe_params = {k: ("***" if k == "key" else v) for k, v in params.items()}
    safe_url = requests.Request("GET", url, params=safe_params).prepare().url
    print(f"[DEBUG] USCB request URL: {safe_url}")
    resp = requests.get(url, params=params, timeout=30)
    print(f"[DEBUG] USCB response status: {resp.status_code}")
    if not resp.ok:
        print(f"[DEBUG] USCB response body: {resp.text[:500]}")
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
    data_type_code = cfg.get("data_type_code", "SM")
    series_name    = cfg.get("series_name", "mrts")
    seasonally_adj = cfg.get("seasonally_adj", "no")
    out_path       = _make_raw_path("uscb")

    raw = _fetch_with_retry(_call_uscb, label="USCB MRTS")
    if raw is None:
        return out_path

    headers, *rows = raw
    df = pd.DataFrame(rows, columns=headers)
    df = df[
        (df["category_code"] == category_code)
        & (df["data_type_code"] == data_type_code)
        & (df["seasonally_adj"] == seasonally_adj)
        & (df["time_slot_id"] == "0")   # "0" = monthly estimate; confirmed from API discovery
    ]

    df = df.rename(columns={"cell_value": series_name, "time": "date"})[["date", series_name]]
    df["date"] = pd.to_datetime(df["date"])
    df[series_name] = pd.to_numeric(df[series_name], errors="coerce")
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


def fetch_market_data(
    tickers: list = None,
    start_date: str = None,
    end_date:   str = None,
    source:     str = "yfinance",
) -> Path:
    """
    Fetch daily OHLCV data for a list of tickers and write to data/raw/.

    Supports source="yfinance" (free, no API key required).  Additional sources
    can be added by extending the dispatch below.

    Args:
        tickers:    List of ticker symbols, e.g. ["XLY", "SPY"].
                    Defaults to config["trading"]["market_data"]["tickers"].
        start_date: ISO date string.  Defaults to config["data"]["start_date"].
        end_date:   ISO date string.  Defaults to config["data"]["end_date"].
        source:     Data source identifier.  Only "yfinance" is currently supported.
    Returns:
        Path to written CSV file (data/raw/market_{date}.csv).
    Raises:
        ImportError if yfinance is not installed.
        ValueError  if an unknown source is specified.
    """
    market_cfg = config.get("trading", {}).get("market_data", {})
    tickers    = tickers    or market_cfg.get("tickers",    ["XLY", "SPY"])
    start_date = start_date or market_cfg.get("start_date", START_DATE)
    end_date   = end_date   or market_cfg.get("end_date",   END_DATE)
    out_path   = _make_raw_path("market")

    if source == "yfinance":
        _fetch_market_yfinance(tickers, start_date, end_date, out_path)
    else:
        raise ValueError(
            f"Unknown market data source '{source}'. Supported: ['yfinance']"
        )

    return out_path


def _fetch_market_yfinance(
    tickers:    list,
    start_date: str,
    end_date:   str,
    out_path:   Path,
) -> None:
    """
    Pull daily OHLCV from Yahoo Finance via yfinance and write to out_path.

    Output schema: date | {ticker}_open | {ticker}_high | {ticker}_low |
                        {ticker}_close | {ticker}_volume  (one set per ticker)

    Args:
        tickers:    E.g. ["XLY", "SPY", "^VIX"]
        start_date: ISO date string
        end_date:   ISO date string
        out_path:   Destination CSV path
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for market data ingestion. "
            "Install it with: pip install yfinance"
        )

    frames = []
    for ticker in tickers:
        result = _fetch_with_retry(
            lambda t=ticker: yf.download(
                t,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            ),
            label=f"yfinance {ticker}",
        )
        if result is None or result.empty:
            print(f"[WARN] No data returned for ticker '{ticker}'")
            continue

        # yfinance returns MultiIndex columns when multi=True; flatten
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [
                f"{ticker.lower()}_{col[0].lower()}" for col in result.columns
            ]
        else:
            result.columns = [f"{ticker.lower()}_{c.lower()}" for c in result.columns]

        result.index = pd.to_datetime(result.index)
        result.index.name = "date"
        frames.append(result)

    if not frames:
        print(f"[WARN] No market data fetched; {out_path} not written")
        return

    df = pd.concat(frames, axis=1).sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df.reset_index().to_csv(out_path, index=False)
    print(f"[INFO] Market data written to {out_path} ({len(tickers)} tickers)")


def run_ingestion() -> dict[str, Path]:
    """
    Master ingestion runner. Calls all source fetchers and returns
    a dict of {source_name: raw_csv_path}.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
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
