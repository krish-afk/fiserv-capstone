import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_PATH = "../../data/processed_data/splits/00_full_dataset.csv"
TARGET = "PersonalConsumptionExpenditures_normalized"

INCLUDE_FSBIMOMYOY = True
INCLUDE_MACROS = True
DROP_ALL_MA = True

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
df["Period"] = pd.to_datetime(df["Period"], errors="coerce")
df = df.dropna(subset=["Period"])

# Filter Geo=US, Sector=ALL (and Sub-Sector=ALL if exists)
geo = df["Geo"].astype(str).str.strip()
sector = df["Sector Name"].astype(str).str.strip().str.upper()

mask = (geo == "US") & (sector == "ALL")
if "Sub-Sector Name" in df.columns:
    sub = df["Sub-Sector Name"].astype(str).str.strip().str.upper()
    mask = mask & (sub == "ALL")

ts_df = df.loc[mask].copy().sort_values("Period").reset_index(drop=True)
if ts_df.empty:
    raise ValueError("No rows found for Geo=='US' & Sector=='ALL' (and Sub-Sector=='ALL' if present).")

# Allowed lagged PCE only
pce_lag_cols = [c for c in ts_df.columns if re.fullmatch(r"PersonalConsumptionExpenditures_normalized_lag(1|3|6)", c)]
if not pce_lag_cols:
    pce_lag_cols = [c for c in ts_df.columns if c.startswith("PersonalConsumptionExpenditures_normalized_lag")]

# FSBI core
fsbi_core = [
    "Real Sales Index - SA_normalized",
    "Transactional Index - SA_normalized",
    "Real Sales Index - NSA_normalized",
    "Transactional Index - NSA_normalized",
]
fsbi_core = [c for c in fsbi_core if c in ts_df.columns]

# FSBI MoM/YoY (optional)
fsbi_momyoy = [c for c in ts_df.columns if (
    ("Real Sales MOM % -" in c or "Real Sales YOY % -" in c or "Transaction MOM % -" in c or "Transaction YOY % " in c)
    and c.endswith("_normalized")
)]

# Macros (levels only; no lags)
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

predictors = []
predictors += fsbi_core
if INCLUDE_FSBIMOMYOY:
    predictors += fsbi_momyoy
if INCLUDE_MACROS:
    predictors += macros
predictors += pce_lag_cols

if DROP_ALL_MA:
    predictors = [c for c in predictors if not re.search(r"_MA(3|6|12)\b", c)]

predictors = sorted(set(predictors))

# Build modeling df
model_df = ts_df[["Period", TARGET] + predictors].copy()
model_df[TARGET] = pd.to_numeric(model_df[TARGET], errors="coerce")
for c in predictors:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

model_df = model_df.dropna(subset=[TARGET] + predictors).copy()
model_df = model_df.sort_values("Period").reset_index(drop=True)

# Last 3 months holdout
unique_periods = sorted(model_df["Period"].unique())
test_periods = set(unique_periods[-3:])
train_mask = ~model_df["Period"].isin(test_periods)
test_mask = model_df["Period"].isin(test_periods)

X_train = model_df.loc[train_mask, predictors].values
y_train = model_df.loc[train_mask, TARGET].values

X_test = model_df.loc[test_mask, predictors].values
y_test = model_df.loc[test_mask, TARGET].values

# Ridge pipeline (standardize -> ridge)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# TimeSeries CV on TRAIN only
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {"ridge__alpha": np.logspace(-4, 4, 30)}  # tune shrinkage
grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best alpha:", grid.best_params_["ridge__alpha"])

# Predict full series (train + test)
y_pred_all = best_model.predict(model_df[predictors].values)
model_df["pred_pce"] = y_pred_all

# Evaluate on last 3 months
y_pred_test = model_df.loc[test_mask, "pred_pce"].values
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Test RMSE (last 3 months): {rmse:.6f}")
print(f"Test MAE  (last 3 months): {mae:.6f}")

# R^2 on train and test
y_pred_train = best_model.predict(X_train)

r2_train = r2_score(y_train, y_pred_train)

print(f"R^2 (train): {r2_train:.4f}")


# Plot lines
plt.figure(figsize=(12, 6))
plt.plot(model_df["Period"], model_df[TARGET], linewidth=2, label="Actual PCE (normalized)")
plt.plot(model_df["Period"], model_df["pred_pce"], linewidth=2, label="Predicted PCE (Ridge)")

plt.axvspan(min(test_periods), max(test_periods), alpha=0.15, label="Test window (last 3 months)")
plt.title("Actual vs Predicted PCE — Geo=US, Sector=ALL (Ridge Regularization)")
plt.xlabel("Period")
plt.ylabel("PCE (normalized)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
