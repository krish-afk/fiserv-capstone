import pandas as pd
from functools import reduce

# ----------------------------
# Files you uploaded (paths)
# ----------------------------
FISERV_PATH = "../../data/raw_data/fiserv_fsbi_inflation_adjusted_20260202.csv"

FRED_PATHS = [
    "../../data/raw_data/fred_ConsumerSentimentIndex_20260206.csv",
    "../../data/raw_data/fred_CreditSpreadBAA_20260206.csv",
    "../../data/raw_data/fred_CreditSpreadGS10_20260206.csv",
    "../../data/raw_data/fred_CrudeOilPrices_20260206.csv",
    "../../data/raw_data/fred_ImportPriceIndex_20260206.csv",
    "../../data/raw_data/fred_Income_20260206.csv",
    "../../data/raw_data/fred_JoltsQuitsRate_20260206.csv",
    "../../data/raw_data/fred_MonetaryCPI_20260206.csv",
    "../../data/raw_data/fred_PersonalConsumptionExpenditures_20260206.csv",
    "../../data/raw_data/fred_Unemployment_20260206.csv",
    "../../data/raw_data/fred_USNaturalGasCompositePrice_20260206.csv",
]
# ---------- helpers ----------
def month_start_from_datetime(s: pd.Series) -> pd.Series:
    # Robust across pandas versions; returns month-start timestamps
    return pd.to_datetime(s.values.astype("datetime64[M]"))

def read_fred_monthly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # FRED usually: DATE + SERIES
    date_col = next((c for c in df.columns if c.lower() == "date"), df.columns[0])
    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) != 1:
        raise ValueError(f"{path}: expected 1 value column; got {value_cols}")
    value_col = value_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["Month"] = month_start_from_datetime(df[date_col])

    # If the FRED series is daily/weekly, collapse to monthly (mean).
    # If it's already monthly, this keeps it the same.
    out = (
        df.groupby("Month", as_index=False)[value_col]
          .mean()
          .sort_values("Month")
    )
    return out

def read_fiserv_keep_all(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Period is usually YYYYMMDD (e.g., 20190101). Convert to datetime.
    df["Period_dt"] = pd.to_datetime(df["Period"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Period_dt"])

    # Add Month key for merging, but KEEP all original columns
    df["Month"] = month_start_from_datetime(df["Period_dt"])

    return df

# ---------- main ----------
fiserv = read_fiserv_keep_all(FISERV_PATH)

# Read all FRED monthlies
fred_monthlies = [read_fred_monthly(p) for p in FRED_PATHS]

# Merge each FRED series onto fiserv (LEFT join = keep all fiserv rows)
merged = reduce(lambda left, right: pd.merge(left, right, on="Month", how="left"), [fiserv] + fred_monthlies)

# Optional: drop helper column if you don't want it
# merged = merged.drop(columns=["Period_dt"])

merged = merged.sort_values(["Month", "Period_dt"] if "Period_dt" in merged.columns else ["Month"])

merged.to_csv("fiserv_with_fred_appended.csv", index=False)
print("Saved fiserv_with_fred_appended.csv")
print("Shape:", merged.shape)
