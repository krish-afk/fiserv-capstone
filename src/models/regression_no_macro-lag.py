import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_PATH = "../../data/processed_data/splits/00_full_dataset.csv"
TARGET = "PersonalConsumptionExpenditures_normalized"

INCLUDE_FSBIMOMYOY = True   # include FSBI MoM/YoY columns (not macro lags)
INCLUDE_MACROS = True       # include contemporaneous macro levels
DROP_ALL_MA = True          # recommended for stability in small time series

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

df["Period"] = pd.to_datetime(df["Period"], errors="coerce")
df = df.dropna(subset=["Period"])

# --- Filter to Geo=US and Sector=ALL (and Sub-Sector=ALL if exists)
geo = df["Geo"].astype(str).str.strip()
sector = df["Sector Name"].astype(str).str.strip().str.upper()

mask = (geo == "US") & (sector == "ALL")
if "Sub-Sector Name" in df.columns:
    sub = df["Sub-Sector Name"].astype(str).str.strip().str.upper()
    mask = mask & (sub == "ALL")

ts_df = df.loc[mask].copy().sort_values("Period").reset_index(drop=True)
if ts_df.empty:
    raise ValueError("No rows found for Geo=='US' & Sector Name=='ALL' (and Sub-Sector=='ALL' if present).")

# --- Target + allowed lagged PCE only
pce_lag_cols = [c for c in ts_df.columns if re.fullmatch(r"PersonalConsumptionExpenditures_normalized_lag(1|3|6)", c)]
if not pce_lag_cols:
    print("Warning: No PCE lag cols found matching lag1/lag3/lag6. Using any lag cols that start with PCE lag.")
    pce_lag_cols = [c for c in ts_df.columns if c.startswith("PersonalConsumptionExpenditures_normalized_lag")]

# --- FSBI core (keep small set)
fsbi_core = [
    "Real Sales Index - SA_normalized",
    "Transactional Index - SA_normalized",
    "Real Sales Index - NSA_normalized",
    "Transactional Index - NSA_normalized",
]
fsbi_core = [c for c in fsbi_core if c in ts_df.columns]

fsbi_momyoy = [c for c in ts_df.columns if (
    ("Real Sales MOM % -" in c or "Real Sales YOY % -" in c or "Transaction MOM % -" in c or "Transaction YOY % " in c)
    and c.endswith("_normalized")
)]
fsbi_momyoy = [c for c in fsbi_momyoy if c in ts_df.columns]

# --- Contemporaneous macro levels (NO lags)
macro_level_candidates = [
    "ConsumerSentimentIndex_normalized",
    "CreditSpreadBAA_normalized",
    "CreditSpreadGS10_normalized",
    "CrudeOilPrices_normalized",
    "ImportPriceIndex_normalized",
    "Income_normalized",
    "JoltsQuitsRate_normalized",
    "MonetaryCPI_normalized",
    "Unemployment_normalized",
    "USNaturalGasCompositePrice_normalized",
    "Oil_ImportPrice_interaction",
    "Unemployment_Income_interaction",
]
macros = [c for c in macro_level_candidates if c in ts_df.columns]

# --- Build predictors
predictors = []
predictors += fsbi_core
if INCLUDE_FSBIMOMYOY:
    predictors += fsbi_momyoy
if INCLUDE_MACROS:
    predictors += macros
predictors += pce_lag_cols

# Optionally drop all moving averages to reduce collinearity
if DROP_ALL_MA:
    predictors = [c for c in predictors if not re.search(r"_MA(3|6|12)\b", c)]

predictors = sorted(set(predictors))

# Build model df
model_df = ts_df[["Period", TARGET] + predictors].copy()
model_df[TARGET] = pd.to_numeric(model_df[TARGET], errors="coerce")
for c in predictors:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

# Drop rows with any missing needed columns (important because lags create NaNs at the beginning)
model_df = model_df.dropna(subset=[TARGET] + predictors).copy()

# --- last 3 months holdout
unique_periods = sorted(model_df["Period"].unique())
if len(unique_periods) < 6:
    raise ValueError(f"Not enough months after dropping NA lags. Months available: {len(unique_periods)}")

test_periods = set(unique_periods[-3:])
train_df = model_df[~model_df["Period"].isin(test_periods)].copy()
test_df  = model_df[ model_df["Period"].isin(test_periods)].copy()

X_train = train_df[predictors].astype(float)
y_train = train_df[TARGET].astype(float)

X_all = model_df[predictors].astype(float)
y_all = model_df[TARGET].astype(float)

# Add constant
X_train = sm.add_constant(X_train, has_constant="add")
X_all   = sm.add_constant(X_all,   has_constant="add")

# Sanity check for df_resid
nobs = X_train.shape[0]
k = X_train.shape[1]
df_resid = nobs - k
if df_resid <= 0:
    raise ValueError(
        f"Too many predictors for too few observations after filtering.\n"
        f"Train obs={nobs}, predictors(including const)={k}, df_resid={df_resid}.\n"
        f"Fix: reduce predictors (set INCLUDE_FSBIMOMYOY=False, INCLUDE_MACROS=False, or drop some macros)."
    )

# Fit (use robust HC1 if you want; if it still acts up, switch to nonrobust)
ols = sm.OLS(y_train, X_train).fit(cov_type="HC1")

# Print a safe summary (avoid .summary() crash)
print("\n===== SAFE MODEL REPORT =====")
print(f"Train nobs: {nobs}, predictors: {k}, df_resid: {df_resid}")
print(f"R2: {ols.rsquared:.4f}, Adj R2: {ols.rsquared_adj:.4f}")
print("Top coefficients by |t|:")
top = (ols.tvalues.abs().sort_values(ascending=False).head(15).index)
print(pd.DataFrame({"coef": ols.params[top], "t": ols.tvalues[top], "p": ols.pvalues[top]}))

# Predict all months
model_df["pred_pce"] = ols.predict(X_all)

# Plot all points
plot_df = model_df.sort_values("Period")[["Period", TARGET, "pred_pce"]]

plt.figure(figsize=(12, 6))

plt.plot(plot_df["Period"], plot_df[TARGET], label="Actual PCE (normalized)", linewidth=2)
plt.plot(plot_df["Period"], plot_df["pred_pce"], label="Predicted PCE", linewidth=2)

plt.axvspan(min(test_periods), max(test_periods), alpha=0.15, label="Test window (last 3 months)")

plt.xlabel("Period")
plt.ylabel("PCE (normalized)")
plt.title("Actual vs Predicted PCE — Geo=US, Sector=ALL")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

