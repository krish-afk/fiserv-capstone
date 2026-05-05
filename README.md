# PCE & MRTS Forecasting with Fiserv FSBI Data

A research pipeline that uses the Fiserv Small Business Index (FSBI), macroeconomic indicators, and public economic data to forecast U.S. Personal Consumption Expenditures (PCE) and Monthly Retail Trade Survey (MRTS) series. Forecast outputs can be evaluated directly and can also feed a trading layer for consumer equity strategies or FORECASTEX-style prediction-market simulations.

## Research questions

1. Does FSBI data provide a statistically meaningful leading signal for PCE and MRTS?
2. If yes, can that signal generate excess returns in a simulated trading environment?

---

## Project structure

```text
fiserv-capstone/
├── main.py                         # CLI entry point for data, experiment, trading, and plot stages
├── config.yaml                     # Main configuration for data, features, experiments, trading, and plots
├── requirements.txt                # Python dependencies
├── src/
│   ├── data/
│   │   ├── ingest.py               # Pull BEA, FRED, Census data; copy manual FSBI data; fetch market data when needed
│   │   ├── load.py                 # Load raw CSVs into DataFrames
│   │   ├── clean.py                # Validate schemas, align frequencies, build processed datasets
│   │   ├── store.py                # Read/write processed data in data/processed/
│   │   ├── features.py             # Lag, rolling, and growth-rate feature engineering
│   │   ├── transform.py            # Target transforms and feature scaling helpers
│   │   └── panel.py                # Build per-panel (y, X, y_level) datasets from config
│   ├── models/
│   │   ├── base.py                 # BaseForecaster interface
│   │   ├── baselines.py            # Naive, mean, and AR(3) baselines
│   │   ├── timeseries.py           # ARIMA, ARIMAX, ETS, ETSX, Theta
│   │   ├── ml.py                   # OLS, Ridge, Lasso, RandomForest, XGBoost, GradientBoosting
│   │   ├── evaluate.py             # Walk-forward evaluation and metrics
│   │   └── experiment.py           # Trial grid builder and experiment runner
│   ├── trading/
│   │   ├── strategies/
│   │   │   ├── consumer_signal_strategy.py   # PCE/MRTS consumer-regime strategy
│   │   │   ├── mrts_event_strategy.py        # MRTS prediction-market event strategy
│   │   │   ├── adaptive_regime_strategy.py   # Adaptive multi-asset regime strategy
│   │   │   ├── pce_pairs_strategy.py         # PCE-driven pair strategy
│   │   │   └── skeleton_strategy.py          # Template for new strategies
│   │   ├── strategy.py             # BaseStrategy, strategy loading, model selection helpers
│   │   ├── pipeline.py             # Configured trading pipeline and market-data resolution
│   │   ├── backtest.py             # Portfolio and event-contract backtesting engines
│   │   └── performance.py          # Trading performance helpers
│   ├── visualization/
│   │   └── eval_plots.py           # Forecast and trading dashboard plot generation
│   ├── backend/
│   │   ├── app.py                  # Flask API for dashboard runs
│   │   ├── service.py              # Backend orchestration for dashboard-selected runs
│   │   ├── presenters.py           # Dashboard response formatting
│   │   └── run_store.py            # Saved dashboard run metadata/results
│   ├── frontend/                   # Vite + React dashboard app
│   └── utils/
│       ├── config.py               # Loads config.yaml and .env
│       └── logging.py              # Logging utilities
├── data/
│   ├── raw/                        # Manual inputs and timestamped raw pulls; gitignored
│   └── processed/                  # master.csv and fsbi_long.csv; gitignored
└── experiments/                    # Timestamped runs, trading outputs, plots, dashboard runs; gitignored
```

---

## Quickstart

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API keys

Create a `.env` file in the project root, next to `config.yaml`:

```bash
BEA_API_KEY=your_bea_key
FRED_API_KEY=your_fred_key
USCB_API_KEY=your_census_key
```

`yfinance` does not require an API key. It is used only when a trading strategy needs market prices.

### 3. Add manual data files

Place the required manual FSBI file here:

```text
data/raw/fsbi_raw.csv
```

The default active trading strategy is `mrts_event`, which also requires the Bloomberg MRTS consensus file configured in `config.yaml`:

```text
data/raw/bbg_mrts.csv
```

That Bloomberg CSV is read with `skiprows=5` and is expected to include `Date`, `PX_LAST`, and `BN_SURVEY_MEDIAN` columns after the skipped header rows.

### 4. Run the pipeline

Run everything configured in `main.py`:

```bash
python main.py
```

Because the default active strategy is the MRTS event-contract strategy, `python main.py` requires `data/raw/bbg_mrts.csv`. To run only data and model experimentation without trading:

```bash
python main.py --skip-trading
```

You can also run stages independently:

```bash
python main.py --stage data        # Ingest and process raw data only
python main.py --stage experiment  # Run model grid on already-processed data
python main.py --stage trading     # Run trading simulation on latest experiment outputs
python main.py --stage plot        # Generate dashboard-style HTML plots for the latest run
```

Run a specific panel:

```bash
python main.py --stage experiment --panel pce_national_mom
```

Refresh market data for equity strategies:

```bash
python main.py --stage trading --refresh-market-data
```

---

## Pipeline stages

### Stage 1 — Data

`src/data/ingest.py` pulls or loads these sources:

| Source | Data | Access |
|--------|------|--------|
| BEA NIPA API | PCE monthly, table `T20805`, line `1` | `BEA_API_KEY` |
| FRED | Macro indicators such as rates, spreads, commodities, labor, income, sentiment, sticky CPI | `FRED_API_KEY` |
| Census Bureau | MRTS retail trade sales, NAICS `44-45` | `USCB_API_KEY` |
| FSBI | Small Business Index by geography, sector, and subsector | Manual CSV at `data/raw/fsbi_raw.csv` |
| Yahoo Finance | Daily OHLCV for strategy tickers | No key; fetched during trading when needed |

Raw API pulls are written as timestamped files in `data/raw/`. FSBI remains a manual input and is loaded from `data/raw/fsbi_raw.csv`.

`src/data/clean.py` validates schemas, aligns BEA/FRED/Census data to monthly frequency, drops rows with remaining missing values, and keeps FSBI in long format for panel-specific filtering.

Processed files written to `data/processed/`:

- `master.csv` — BEA + FRED + Census wide dataset indexed by date
- `fsbi_long.csv` — FSBI long dataset with `Geo`, `Sector Name`, and `Sub-Sector Name` dimensions

### Stage 2 — Experiment

Panels are defined under `data.panels` in `config.yaml`. Each panel specifies:

- `target`: `pce` or `mrts`
- `target_transform`: `null`, `mom`, or `yoy`
- `geography`, `sectors`, and `subsectors`: FSBI filters used before pivoting features

`src/data/panel.py` joins the filtered FSBI data to the processed master dataset and returns `(y, X, y_level)` for the selected panel.

Feature engineering creates target lags, non-target lags, rolling means/stds, and optional growth-rate features. Named feature sets under `features.feature_sets` are glob patterns matched against engineered column names.

`src/models/experiment.py` expands the configured panels, model variants, parameter grids, and feature sets into trials. Each trial is evaluated with walk-forward expanding-window validation.

Experiment outputs are written to a timestamped folder under `experiments/`:

- `metrics.csv` — one row per trial with `mae`, `rmse`, `me`, `mape`, `mape_ratio`, `dir_acc`, and `r2`
- `forecasts.csv` — date-level `y_true` and `y_pred` values per trial
- `artifacts/` — optional model interpretability artifacts such as coefficients and feature importances
- `metadata.json` — run metadata and configuration summary

### Stage 3 — Trading

Trading runs on top of the latest experiment outputs. `src/trading/pipeline.py` loads the active strategy from `config.yaml`, selects the configured or best model forecast per panel, resolves market data when tickers are required, and dispatches to `BacktestEngine`.

Two strategy families are supported:

1. **Portfolio strategies** — strategies that emit `weight__TICKER` columns or legacy `signal` values. These use historical market prices and produce portfolio backtest outputs.
2. **FORECASTEX-style event strategies** — strategies with no traded tickers. These model event-contract P&L from forecast edge versus a consensus strike.

Trading outputs are written under the latest experiment run, usually in `experiments/<timestamp>/trading/`. Depending on strategy and `trading.test_modes`, outputs may include signals, positions, trades, equity curves, path results, and JSON summaries.

### Stage 4 — Visualization

`src/visualization/eval_plots.py` reads the latest experiment run and writes interactive Plotly HTML dashboards under:

```text
experiments/<timestamp>/plots/
```

Generated plots include forecast-vs-actual charts, top-model comparisons, macro profiler views, and trading performance charts when trading outputs exist.

---

## Configuration recipes

Most experiment, panel, feature, and trading behavior is controlled in `config.yaml`.

### Add a panel

Define a new panel under `data.panels`:

```yaml
data:
  panels:
    pce_retail_us_mom:
      target: "pce"
      target_transform: "mom"
      geography: ["US"]
      sectors: ["Retail Trade"]
      subsectors: ["ALL"]
```

Then include it in the experiment list:

```yaml
experiment:
  panels:
    - pce_national_mom
    - mrts_national
    - pce_retail_us_mom
```

### Add a feature set

```yaml
features:
  feature_sets:
    fsbi_sales_lag1:
      - "fsbi_*sales*_lag1"
      - "fsbi_*sales*_rollmean3"
```

Use `[]` for a pure time-series model path with no exogenous feature matrix.

### Add an existing model class to the grid

```yaml
experiment:
  models:
    ml:
      variants:
        - class: RidgeForecaster
          param_grid:
            alpha: [0.001, 0.01, 0.1, 1.0]
          feature_sets: [fsbi_lags, macro_lags, fsbi_and_macro_lags]
```

`param_grid` expands into one trial per parameter combination. Use `params` for a single fixed configuration. For a brand-new model class, implement `BaseForecaster`, import it in `src/models/experiment.py`, and add it to `_MODEL_REGISTRY`.

### Switch the active trading strategy

Set `trading.active_strategy` to one of the keys under `trading.strategies`:

```yaml
trading:
  active_strategy: "consumer_regime"
  test_modes: ["backtest"]
```

Currently configured strategy keys:

| Key | Class | Type |
|-----|-------|------|
| `consumer_regime` | `ConsumerRegimeStrategy` | Equity portfolio strategy |
| `mrts_event` | `MRTSForecastMarketStrategy` | Prediction-market event strategy |
| `adaptive_regime` | `AdaptiveRegimeStrategy` | Equity portfolio strategy |
| `pce_pairs` | `PCEPairsStrategy` | Pair-trading portfolio strategy |

Example event-contract configuration:

```yaml
trading:
  active_strategy: "mrts_event"
  test_modes: ["backtest"]
  strategies:
    mrts_event:
      class: "MRTSForecastMarketStrategy"
      forecast_panels:
        primary: "mrts_national"
      selected_model:
        mrts_national:
          model_name: null
          feature_set: null
      forecastex:
        enabled: true
      params:
        bloomberg_csv_path: "data/raw/bbg_mrts.csv"
        model_rmse: 0.15
        edge_threshold: 0.15
        contract_price: 0.50
```

Example equity strategy configuration:

```yaml
trading:
  active_strategy: "consumer_regime"
  test_modes: ["backtest", "monte_carlo"]
  strategies:
    consumer_regime:
      class: "ConsumerRegimeStrategy"
      forecast_panels:
        primary: "pce_national_mom"
        mrts: "mrts_national"
      selected_model:
        pce_national_mom:
          model_name: null
          feature_set: null
        mrts_national:
          model_name: null
          feature_set: null
      forecastex:
        enabled: false
      params:
        bullish_threshold: 0.75
        bearish_threshold: -0.75
        pce_weight: 0.6
        mrts_weight: 0.4
        target_allocation: 0.25
        risk_on_ticker: "XLY"
        defensive_ticker: "XLP"
```

To add a new strategy, add a Python file in `src/trading/strategies/`, subclass `BaseStrategy`, implement `name` and `generate_signals()`, and add a config entry under `trading.strategies`.

---

## Models

| Class | Type | Notes |
|-------|------|-------|
| `NaiveForecaster` | Baseline | Last observed value |
| `MeanForecaster` | Baseline | Historical training mean |
| `AR3BaselineForecaster` | Baseline | AR(3)-style baseline using target lag features |
| `ARIMAForecaster` | Time series | Univariate ARIMA |
| `ARIMAXForecaster` | Time series | ARIMA with exogenous features |
| `ETSForecaster` | Time series | Exponential smoothing |
| `ETSXForecaster` | Time series | OLS plus ETS on residuals |
| `ThetaForecaster` | Time series | Theta model |
| `OLSForecaster` | ML/statistical | OLS via statsmodels; records coefficients and t-stats |
| `RidgeForecaster` | ML | Ridge with fold-local scaling |
| `LassoForecaster` | ML | Lasso with feature selection frequency artifacts |
| `RandomForestForecaster` | ML | Random forest with mean feature importances |
| `XGBoostForecaster` | ML | XGBoost regressor with gain-based importance |
| `GradientBoostingForecaster` | ML | scikit-learn gradient boosting regressor |

---

## Evaluation metrics

Walk-forward expanding-window evaluation is used for all trials. The minimum training window is controlled by `forecasting.walk_forward_min_train` in `config.yaml`.

| Metric column | Description |
|---------------|-------------|
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `me` | Mean signed error / bias |
| `mape` | Mean absolute percentage error |
| `mape_ratio` | Ratio form of MAPE where available |
| `dir_acc` | Fraction of periods where `sign(y_pred) == sign(y_true)` |
| `r2` | Coefficient of determination; can be negative |

---

## Running the dashboard

The dashboard has a Flask backend and a Vite + React frontend.

### Backend

From the project root:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
flask --app src.backend.app run --port 5000
```

Useful backend endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | Health check |
| `GET /api/options` | Available panels, models, feature sets, and strategies |
| `POST /api/runs` | Start a dashboard-configured pipeline run |
| `GET /api/runs` | List saved dashboard runs |
| `GET /api/runs/<run_id>` | Get run status |
| `GET /api/runs/<run_id>/results` | Get run results |

### Frontend

In a separate terminal:

```bash
cd src/frontend
npm install
npm run dev
```

The Vite dev server runs on port `5173` and proxies `/api` requests to the Flask backend at `http://127.0.0.1:5000`.

---

## Common troubleshooting

### `FileNotFoundError: FSBI file not found`

Place the manual FSBI file at:

```text
data/raw/fsbi_raw.csv
```

Then rerun:

```bash
python main.py --stage data
```

### `Bloomberg CSV missing at data/raw/bbg_mrts.csv`

The default `mrts_event` strategy requires the Bloomberg MRTS file. Either place the file at `data/raw/bbg_mrts.csv`, change `trading.strategies.mrts_event.params.bloomberg_csv_path`, switch `trading.active_strategy`, or run without trading:

```bash
python main.py --skip-trading
```

### No processed data found

Run the data stage before experiment-only or trading-only runs:

```bash
python main.py --stage data
```

### Market data is stale or missing for an equity strategy

Force a market-data refresh:

```bash
python main.py --stage trading --refresh-market-data
```
