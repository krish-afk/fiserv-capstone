"""
PCE Forecasting — Combined Experiment Runner

Runs ARIMA/ARIMAX, ETS, and Theta model families against the same
targets and CV folds, then produces a unified comparison table.
"""

from datetime import datetime
from functools import partial

import pandas as pd

import arima
import exponential_smoothing as ets
import theta as theta_module
from base import (DATA_DIR, PCE_RAW_COL, PCE_MOM_COL, PCE_YOY_COL,
                  load_dataset, derive_pce_targets, prepare_series,
                  run_experiment, summarize_results, save_results,
                  plot_acf_pacf, run_diebold_mariano)
from arima import find_best_arima_order

# ============================================================
# CONFIGURATION
# ============================================================

FILEPATH         = DATA_DIR / "00_full_dataset.csv"
DATE_COL         = 'Period'
NON_FEATURE_COLS = [DATE_COL, 'Geo', 'Sector Name', 'Sub-Sector Name']
TARGET_COLS      = [PCE_MOM_COL, PCE_YOY_COL]

HORIZONS       = [1, 2, 3]
MIN_TRAIN_SIZE = 24
EXPANDING      = True
MAX_EXOG_FEATS = 8

# Features to test as a focused ARIMAX variant ("arimax_limited").
LIMITED_EXOG_FEATURES = [
    "Real Sales MOM % - SA_normalized",
    "Real Sales YOY % - SA_normalized",
    "Transaction MOM % - SA_normalized",
    "Transaction YOY %  - SA_normalized"
]

# Number of ACF/PACF lags to plot
ACF_LAGS = 24


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- Load & prepare data ---
    df = load_dataset(FILEPATH, DATE_COL)

    # Derive PCE targets as first/12-period differences of the normalized level,
    # matching peer methodology (regression.ipynb: data["PCE_MOM"] = ...diff()).
    # Merge back by date so prepare_series can aggregate and drop NaN rows uniformly.
    pce_targets = derive_pce_targets(df, DATE_COL)
    df = df.merge(pce_targets.reset_index(), on=DATE_COL, how='left')

    # Exclude the raw PCE level and any prebuilt MoM/YoY columns from exog features;
    # those are either the source of our derived targets or redundant transformations.
    PCE_EXCLUDE = [
        PCE_RAW_COL,
        'PersonalConsumptionExpenditures_normalized_MoM_normalized',
        'PersonalConsumptionExpenditures_normalized_YoY_normalized',
    ]
    exog_cols = [col for col in df.columns
                 if col not in NON_FEATURE_COLS
                 and col not in TARGET_COLS
                 and col not in PCE_EXCLUDE]

    target_df, exog_df = prepare_series(df, DATE_COL, TARGET_COLS, exog_cols)

    # --- ACF / PACF for each target ---
    target_short_labels = {PCE_MOM_COL: 'MoM', PCE_YOY_COL: 'YoY'}
    for target_col in TARGET_COLS:
        plot_acf_pacf(target_df[target_col],
                      label=target_short_labels[target_col],
                      lags=ACF_LAGS,
                      timestamp=timestamp)

    # --- Validate limited exog features once ---
    valid_limited = [f for f in LIMITED_EXOG_FEATURES if f in exog_df.columns]
    if LIMITED_EXOG_FEATURES:
        missing = set(LIMITED_EXOG_FEATURES) - set(valid_limited)
        if missing:
            print(f"Warning: LIMITED_EXOG_FEATURES not found in data "
                  f"and will be ignored: {missing}")
        if valid_limited:
            print(f"Limited exog features: {valid_limited}")

    all_results = []

    for target_col in TARGET_COLS:
        target = target_df[target_col]
        print(f"\n{'='*60}")
        print(f"Target: {target_col}")
        print('='*60)

        # --- ARIMA order selection (once per target) ---
        order = find_best_arima_order(target)

        # --- Build predictor lists ---
        arima_predictors = arima.build_predictors(
            target, exog_df, exog_cols, order,
            max_exog_features=MAX_EXOG_FEATS
        )
        ets_predictors   = ets.build_predictors()
        theta_predictors = theta_module.build_predictors()

        all_predictors = arima_predictors + ets_predictors + theta_predictors

        # --- Optional: focused ARIMAX with user-specified features ---
        if valid_limited:
            limited_predictor = (
                'arimax_limited',
                partial(arima.predict_arimax,
                        order=order,
                        exog_df=exog_df[valid_limited]),
            )
            all_predictors = all_predictors + [limited_predictor]

        # --- Run experiment for this target ---
        results = run_experiment(
            target_df=target_df,
            exog_df=exog_df,
            target_cols=[target_col],
            predictors=all_predictors,
            horizons=HORIZONS,
            min_train_size=MIN_TRAIN_SIZE,
            expanding=EXPANDING,
        )
        all_results.append(results)

    # --- Combine, summarize, and save performance heatmaps ---
    combined = pd.concat(all_results, ignore_index=True)
    summary  = summarize_results(combined)
    save_results(summary, timestamp=timestamp)

    # --- Diebold-Mariano tests vs naive baseline ---
    run_diebold_mariano(combined, timestamp=timestamp)

    return combined, summary


if __name__ == "__main__":
    combined, summary = main()
