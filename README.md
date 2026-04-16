# PCE & MRTS Forecasting with Fiserv FSBI Data

A research pipeline that uses the Fiserv Small Business Index (FSBI) and macroeconomic indicators to forecast U.S. Personal Consumption Expenditures (PCE) and Monthly Retail Trade Survey (MRTS) data. Forecast outputs drive a trading simulation layer that evaluates directional strategies on the Consumer Discretionary ETF (XLY) and FORECASTEX-style prediction markets.

## Research questions

1. Does FSBI data provide a statistically meaningful leading signal for PCE and MRTS?
2. If yes, can that signal generate excess returns in a simulated trading environment?

---

## Project structure

```
fiserv-capstone/
├── main.py                  # Pipeline entry point
├── config.yaml              # All experiment parameters — edit here, not in code
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── ingest.py        # Pull raw data from BEA, FRED, USCB, yfinance APIs
│   │   ├── load.py          # Load raw CSVs into DataFrames
│   │   ├── clean.py         # Validate, align frequencies, build master dataset
│   │   ├── store.py         # Read/write processed data to data/processed/
│   │   ├── features.py      # Lag, rolling, and growth-rate feature engineering
│   │   ├── transform.py     # Target transforms (MoM %, YoY %) and feature scaling
│   │   └── panel.py         # Build per-experiment (y, X) panels from config
│   ├── models/
│   │   ├── base.py          # BaseForecaster ABC — all models implement this
│   │   ├── baselines.py     # NaiveForecaster, MeanForecaster
│   │   ├── timeseries.py    # ARIMA, ARIMAX, ETS, ETSX, Theta
│   │   ├── ml.py            # OLS, Ridge, Lasso, RandomForest, XGBoost, GBM
│   │   ├── evaluate.py      # walk_forward_evaluate(), compute_metrics()
│   │   └── experiment.py    # Trial grid builder and experiment runner
│   ├── trading/
│   │   ├── strategies/                 # Strategy files loaded dynamically via config
│   │   │   ├── krish_trade_strategy.py # Consumer regime ETF strategy
│   │   │   └── mrts_event_strategy.py  # MRTS prediction market binary strategy
│   │   ├── strategy.py                 # BaseStrategy and universal adapter logic
│   │   ├── backtest.py                 # Standard portfolio equity backtester 
│   │   ├── run_event_backtest.py       # Custom evaluation engine for event contracts
│   │   └── performance.py              # Sharpe, drawdown, sensitivity analysis
│   ├── visualization/
│   │   └── eval_plots.py               # Generates out-of-sample plots (Best Model vs Baseline)
│       └── config.py        # Loads config.yaml and .env into the global config dict
├── data/
│   ├── raw/                 # Written by ingest.py; fsbi_raw.csv placed here manually
│   └── processed/           # master.csv, fsbi_long.csv written after cleaning
├── experiments/             # One timestamped subdirectory per run (metrics, forecasts, artifacts)
└── notebooks/               # Exploratory analysis
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

Create a `.env` file in the project root (next to `config.yaml`):

```bash
BEA_API_KEY=your_bea_key       # https://apps.bea.gov/API/signup/
FRED_API_KEY=your_fred_key     # https://fred.stlouisfed.org/docs/api/api_key.html
USCB_API_KEY=your_census_key   # https://api.census.gov/data/key_signup.html
```

`yfinance` does not require a key.

### 3. Place the FSBI file

Some datasets are not available via public API. Place the pre-downloaded files at:

```
data/raw/fsbi_raw.csv
data/raw/mrts_bbg_consensus.csv
```

### 4. Run the full pipeline

```bash
python main.py
```

This runs all three stages in order: data ingestion, model experimentation, and (when enabled) trading simulation. Each stage can also be run independently:

```bash
python main.py --stage data        # Ingest and process raw data only
python main.py --stage experiment  # Run model grid on already-processed data
python main.py --stage trading     # Run trading simulation on experiment outputs
python main.py --stage plot        # Generate Best Model vs Baseline forecast plots
```

To run a specific panel:

```bash
python main.py --stage experiment --panel pce_national_mom
```

---

## Pipeline stages

### Stage 1 — Data

**Ingestion** (`src/data/ingest.py`) pulls from four sources:

| Source | Data | API |
|--------|------|-----|
| BEA | PCE (monthly, not seasonally adjusted) | `BEA_API_KEY` |
| FRED | 11 macro series (rates, labor, commodities) | `FRED_API_KEY` |
| USCB | MRTS retail trade sales | `USCB_API_KEY` |
| FSBI | Small Business Index — geography × sector × subsector | Manual CSV |
| yfinance | Daily OHLCV for XLY, SPY | None |

Raw CSVs are written to `data/raw/` with a datestamp in the filename. Re-running ingestion always writes new files without deleting old ones.

**Cleaning** (`src/data/clean.py`) validates schema, aligns all non-FSBI sources to monthly frequency via forward-fill, and merges BEA + FRED + USCB into a single wide master DataFrame. FSBI is kept in long format separately because it requires per-panel geographic and sector filtering before pivoting.

Processed files written to `data/processed/`:
- `master.csv` — BEA + FRED + USCB, date-indexed
- `fsbi_long.csv` — FSBI in long format with `Geo`, `Sector Name`, `Sub-Sector Name` dimensions

### Stage 2 — Experiment

**Panels** (`src/data/panel.py`) are defined in `config.yaml` under `data.panels`. Each panel specifies a target column (`pce` or `mrts`), an optional target transform (`mom`, `yoy`, or level), and FSBI filters for geography, sector, and subsector. `build_panel()` joins the filtered FSBI data onto the master dataset, runs feature engineering, and returns `(y, X, y_level)`.

**Feature engineering** (`src/data/features.py`) builds lag, rolling mean/std, and growth-rate columns for all non-target columns. Lag periods and rolling windows are set in `config.yaml` under `features`. Targets (`pce`, `mrts`) are never included as features.

**Experiment grid** (`src/models/experiment.py`) reads `config.yaml` under `experiment` to build a full cross-product of panels × model variants × parameter combinations × feature sets. Each combination is one `Trial`. Walk-forward (expanding window) evaluation is used throughout — no data leakage.

**Outputs** are written to a timestamped directory under `experiments/`:
- `metrics.csv` — one row per trial with RMSE, MAE, ME, MAPE, directional accuracy, R²
- `forecasts.csv` — full `(date, y_true, y_pred)` for every trial
- `artifacts/` — per-trial JSON files with model-specific interpretability data (coefficients, feature importances, selection frequencies)
- `metadata.json` — run configuration summary

### Stage 3 — Trading

Built on top of experiment outputs. `main.py` contains a dynamic router that evaluates the strategy configured in `config.yaml` and deploys the appropriate backtesting engine:

1. **Portfolio backtest:** Uses backtrader to run monthly signal-following on standard equities (e.g., XLY). Metrics: Sharpe ratio, annualised return, max drawdown, win rate.
2. **FORECASTEX simulation:** (`src/trading/run_event_backtest.py`) Models P&L on a prediction-market contract. Compares model point estimates against Bloomberg consensus strike prices, applies normal distribution math to find the probability edge, and outputs Pure vs. Realistic PnL.

**Sensitivity analysis** (`src/trading/performance.py`) measures how trading performance degrades as forecast accuracy declines by injecting Gaussian noise.

### Stage 4 — Visualization

Runs `src/visualization/eval_plots.py` to extract the best-performing model from `metrics.csv` and plot its out-of-sample predictions against ground truth and a Naive baseline. Plots are saved directly into the active `experiments/` run folder.

---

## Configuration

All experiment parameters live in `config.yaml`. No Python changes are needed to add models, feature sets, or panels.

### Adding a panel

```yaml
### Switching the trading strategy

Strategies are loaded dynamically. Specify the `.py` file to load from the `src/trading/strategies/` folder, and pass its required parameters.

**Example A: Standard Equity Strategy (ETF Rotation)**
trading:
  strategy:
    file: "krish_trade_strategy.py"
    class: "KrishTradeStrategy"
    params:
      bullish_threshold: 0.75
      bearish_threshold: -0.75
      risk_on_ticker: "XLY"
      defensive_ticker: "XLP"

**Example B: Prediction Market Strategy (Binary Event Contracts)**
trading:
  strategy:
    file: "mrts_event_strategy.py"
    class: "MRTSForecastMarketStrategy"
    params:
      bloomberg_csv_path: "data/raw/mrts_bbg_consensus.csv" 
      model_rmse: 0.15          
      edge_threshold: 0.02      
      contract_price: 0.50
```

Then add the panel name to `experiment.panels` to include it in the next run.

### Adding a model variant

```yaml
experiment:
  models:
    ml:
      variants:
        - class: RidgeForecaster
          param_grid:
            alpha: [0.001, 0.01, 0.1]
          feature_sets: [fsbi_lags, macro_lags]
```

`param_grid` is expanded into one trial per combination. Use `params` instead for a single fixed configuration.

### Adding a feature set

```yaml
features:
  feature_sets:
    my_subset:
      - "fsbi_*_lag1"
      - "unemployment_rate_lag*"
```

Values are `fnmatch` glob patterns matched against column names after feature engineering. An empty list (`[]`) passes `X=None` to the model (pure time-series path).

### Switching the trading strategy

```yaml
trading:
  portfolio:
    strategy: "ThresholdPCEStrategy"
    params:
      threshold: 0.3          # only trade when |forecast| > 0.3 pp
```

Available strategies: `DirectionalPCEStrategy`, `ThresholdPCEStrategy`. New strategies can be added by subclassing `BaseStrategy` in `src/trading/strategy.py` and registering them in `_STRATEGY_REGISTRY`.

---

## Models

| Class | Type | Notes |
|-------|------|-------|
| `NaiveForecaster` | Baseline | Last observed value |
| `MeanForecaster` | Baseline | Historical training mean |
| `ARIMAForecaster` | Time series | Pure univariate ARIMA(p,d,q) |
| `ARIMAXForecaster` | Time series | ARIMA with exogenous features |
| `ETSForecaster` | Time series | Exponential smoothing (SES / Holt / Holt-Winters) |
| `ETSXForecaster` | Time series | Two-stage OLS + ETS on residuals |
| `ThetaForecaster` | Time series | Theta model |
| `OLSForecaster` | ML | OLS via statsmodels; records coefficients and t-stats |
| `RidgeForecaster` | ML | Ridge with StandardScaler per fold |
| `LassoForecaster` | ML | Lasso; records feature selection frequency across folds |
| `RandomForestForecaster` | ML | Records mean feature importances across folds |
| `XGBoostForecaster` | ML | Gain-based importance; optimises MAE |
| `GradientBoostingForecaster` | ML | sklearn GBM |

All models implement `BaseForecaster` and are registered in `experiment._MODEL_REGISTRY`. Adding a new model requires implementing `fit()`, `predict()`, and `name`, then adding one line to the registry and one entry in `config.yaml`.

---

## Evaluation metrics

Walk-forward (expanding-window) evaluation is used for all models. The minimum training window before the first forecast is set by `forecasting.walk_forward_min_train` in `config.yaml` (default: 36 months).

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error — primary ranking metric |
| RMSE | Root mean squared error |
| ME | Mean error (signed bias) |
| MAPE | Mean absolute percentage error |
| Dir. accuracy | Fraction of periods where sign(y_pred) == sign(y_true) |
| R² | Coefficient of determination (can be negative) |

---

## Data sources

| Source | What | Access |
|--------|------|--------|
| BEA NIPA API | PCE monthly (table T20805, line 1) | Free API key |
| FRED | Fed funds rate, credit spreads, treasury yields, crude oil, natural gas, import prices, unemployment, JOLTS quits, real disposable income, consumer sentiment, sticky CPI | Free API key |
| Census Bureau | MRTS total retail sales (NAICS 44-45) | Free API key |
| Fiserv FSBI | Small Business Index by geography, sector, subsector | Provided dataset |
| Yahoo Finance | Daily OHLCV for XLY, SPY | No key required |
