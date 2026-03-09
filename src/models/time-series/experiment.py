# PCE Forecasting — Combined Experiment Runner
#
# Runs ARIMA/ARIMAX, ETS, and Theta model families against the same
# targets and CV folds, then produces a unified comparison table.
#
# Usage:
#   python experiment.py

import pandas as pd

import arima
import exponential_smoothing as ets
import theta as theta_module
from base import (DATA_DIR, PCE_MOM_COL, PCE_YOY_COL,
                  load_dataset, prepare_series, run_experiment,
                  summarize_results, save_results)
from arima import find_best_arima_order

# ============================================================
# CONFIGURATION
# ============================================================

FILEPATH       = DATA_DIR / "00_full_dataset.csv"
DATE_COL       = 'Period'
NON_FEATURE_COLS = [DATE_COL, 'Geo', 'Sector Name', 'Sub-Sector Name']
TARGET_COLS    = [PCE_MOM_COL, PCE_YOY_COL]

HORIZONS       = [1, 2, 3]
MIN_TRAIN_SIZE = 24
EXPANDING      = True
MAX_EXOG_FEATS = 8


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    # --- Load & prepare data ---
    df = load_dataset(FILEPATH, DATE_COL)

    exog_cols = [col for col in df.columns
                 if col not in NON_FEATURE_COLS and col not in TARGET_COLS]

    target_df, exog_df = prepare_series(df, DATE_COL, TARGET_COLS, exog_cols)

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

    # --- Combine, summarize, and save ---
    combined = pd.concat(all_results, ignore_index=True)
    summary  = summarize_results(combined)
    save_results(summary)

    return combined, summary


if __name__ == "__main__":
    combined, summary = main()
