# Shared infrastructure for PCE forecasting experiments.
#
# Each model module (arima.py, exponential_smoothing.py, theta.py) exposes a
# list of predictors in the form:
#
#   PREDICTORS: list[tuple[str, Callable]]
#
# where each callable has the signature:
#
#   predict(train: pd.Series, horizon: int) -> np.ndarray
#
# run_experiment() accepts that list and knows nothing about the internals of
# any individual model.

from datetime import datetime
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data/live'
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'

PCE_MOM_COL = 'PersonalConsumptionExpenditures_normalized_MoM_normalized'
PCE_YOY_COL = 'PersonalConsumptionExpenditures_normalized_YoY_normalized'


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_dataset(filepath: str, date_col: str = 'Period') -> pd.DataFrame:
    """
    Load a CSV dataset and ensure it's ready for time series modeling.
    Returns a DataFrame sorted by date with the date column parsed.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    assert date_col in df.columns, f"Date column '{date_col}' not found"
    assert df[date_col].notna().all(), "Missing values found in date column"

    print(f"Loaded {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Unique dates: {df[date_col].nunique()}")

    return df


def prepare_series(df: pd.DataFrame,
                   date_col: str,
                   target_cols: list,
                   exog_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate rows to one observation per date and drop NaN-containing rows.

    Target columns are aggregated by median; exog columns by mean.
    Rows with any NaN are dropped (handles YoY/MoM initialization periods).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (target_df, exog_df), both date-indexed with matching rows.
    """
    target_df = df.groupby(date_col)[target_cols].median()
    exog_df = df.groupby(date_col)[exog_cols].mean()

    original_len = len(target_df)
    combined = target_df.join(exog_df, how='inner').dropna()
    target_df = combined[target_cols]
    exog_df = combined[exog_cols]

    dropped = original_len - len(target_df)
    print(f"Dropped {dropped} rows containing NaN values")
    print(f"Remaining time steps: {len(target_df)}")

    assert len(target_df) > 0, "No data remaining after dropping NaNs"
    assert target_df.index.is_monotonic_increasing, "Date index not sorted ascending"

    return target_df, exog_df


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_exog_features(target: pd.Series,
                         exog_df: pd.DataFrame,
                         max_features: int = 8,
                         collinearity_threshold: float = 0.95) -> list:
    """
    Select exogenous features for ARIMAX via correlation ranking with
    greedy collinearity filtering.

    Parameters
    ----------
    target : pd.Series
        The target series.
    exog_df : pd.DataFrame
        All candidate exogenous features, date-indexed.
    max_features : int
        Maximum number of features to select.
    collinearity_threshold : float
        Exclude features with pairwise correlation above this threshold
        against any already-selected feature.

    Returns
    -------
    list
        Selected column names, ordered by correlation with target.
    """
    aligned_target, aligned_exog = target.align(exog_df, join='inner')

    correlations = (aligned_exog
                    .apply(lambda col: col.corr(aligned_target))
                    .abs()
                    .dropna()
                    .sort_values(ascending=False))

    selected = []
    for feature in correlations.index:
        if len(selected) >= max_features:
            break
        if not selected:
            selected.append(feature)
            continue
        candidate_corrs = aligned_exog[selected].corrwith(aligned_exog[feature]).abs()
        if candidate_corrs.max() < collinearity_threshold:
            selected.append(feature)

    print(f"Selected {len(selected)} features from {len(exog_df.columns)} candidates")
    for f in selected:
        print(f"  {f} (r={correlations[f]:.3f})")

    return selected


# ============================================================
# CROSS-VALIDATION
# ============================================================

def get_cv_splits(series: pd.Series,
                  min_train_size: int,
                  horizon: int,
                  expanding: bool = True) -> list[tuple]:
    """
    Generate rolling/expanding window cross-validation splits.

    Parameters
    ----------
    series : pd.Series
        The target time series (date-indexed).
    min_train_size : int
        Minimum number of observations in the first training window.
    horizon : int
        Forecast horizon (number of steps ahead).
    expanding : bool
        If True, training window grows each fold (expanding window).
        If False, training window is fixed size (rolling window).

    Returns
    -------
    list[tuple]
        List of (train_series, test_series) pairs. Each test_series is
        exactly `horizon` steps long.
    """
    splits = []
    n = len(series)

    for i in range(min_train_size, n - horizon + 1):
        train = series.iloc[:i] if expanding else series.iloc[i - min_train_size:i]
        test = series.iloc[i:i + horizon]
        splits.append((train, test))

    print(f"Generated {len(splits)} CV folds "
          f"(train: {len(splits[0][0])}–{len(splits[-1][0])}, "
          f"horizon: {horizon})")

    return splits


# ============================================================
# EVALUATION
# ============================================================

def directional_accuracy(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Proportion of steps where predicted direction matches actual direction."""
    return float(np.mean(np.sign(actuals) == np.sign(predictions)))


def evaluate_forecast(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute MAE, RMSE, and directional accuracy for a single forecast."""
    return {
        'MAE': mean_absolute_error(actuals, predictions),
        'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
        'DirectionalAcc': directional_accuracy(actuals, predictions),
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(target_df: pd.DataFrame,
                   exog_df: pd.DataFrame,
                   target_cols: list,
                   predictors: list[tuple],
                   horizons: list = [1, 2, 3],
                   min_train_size: int = 24,
                   expanding: bool = True) -> pd.DataFrame:
    """
    Run a list of predictors across all targets, horizons, and CV folds.

    Parameters
    ----------
    target_df : pd.DataFrame
        Date-indexed DataFrame with target columns.
    exog_df : pd.DataFrame
        Date-indexed DataFrame with exogenous features (may be unused by some
        predictors, but is passed through for those that need it).
    target_cols : list
        Target column names to forecast.
    predictors : list[tuple[str, Callable]]
        Each entry is (name, fn) where fn has signature:
            fn(train, horizon) -> np.ndarray
        Bind any model-specific arguments (orders, exog slices, etc.) via
        functools.partial or a closure before passing here.
    horizons : list
        Forecast horizons to evaluate.
    min_train_size : int
        Minimum training window size in CV.
    expanding : bool
        Whether to use expanding or rolling CV window.

    Returns
    -------
    pd.DataFrame
        Tidy results with one row per (target, model, horizon, fold).
    """
    results = []

    for target_col in target_cols:
        target = target_df[target_col]

        for horizon in horizons:
            splits = get_cv_splits(target, min_train_size, horizon, expanding)

            for fold_idx, (train, test) in enumerate(splits):
                actuals = test.values
                row_base = {'target': target_col, 'horizon': horizon, 'fold': fold_idx}

                for name, predict_fn in predictors:
                    try:
                        preds = predict_fn(train, horizon)
                        results.append({**row_base, 'model': name,
                                        **evaluate_forecast(actuals, preds)})
                    except Exception as e:
                        print(f"{name} failed | target={target_col} "
                              f"horizon={horizon} fold={fold_idx}: {e}")

    return pd.DataFrame(results)


# ============================================================
# RESULTS AGGREGATION & REPORTING
# ============================================================

def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CV fold results, reporting mean and std per (target, model, horizon).
    Adds an RMSE_rank column ranking models within each (target, horizon) group.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results from run_experiment().

    Returns
    -------
    pd.DataFrame
        Summary with columns like MAE_mean, MAE_std, RMSE_mean, etc.
    """
    metrics = ['MAE', 'RMSE', 'DirectionalAcc']

    summary = (results_df
               .groupby(['target', 'model', 'horizon'])[metrics]
               .agg(['mean', 'std'])
               .round(4))

    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    summary['RMSE_rank'] = (summary
                            .groupby(['target', 'horizon'])['RMSE_mean']
                            .rank(ascending=True)
                            .astype(int))

    return summary


def save_results(summary: pd.DataFrame, timestamp: str | None = None) -> list[Path]:
    """
    Save one plot per (target × horizon) combination — six total.

    Each plot is a grouped bar chart showing mean MAE and RMSE per model,
    with a secondary y-axis line for mean directional accuracy. Error bars
    show one standard deviation across CV folds.

    Files are written to RESULTS_DIR as:
        time_series_experiment_results_{timestamp}_{target_label}_h{horizon}.png

    Parameters
    ----------
    summary : pd.DataFrame
        Output of summarize_results().
    timestamp : str | None
        Timestamp string to embed in filenames. Defaults to now
        (format: YYYYMMDD_HHMMSS).

    Returns
    -------
    list[Path]
        Paths of all saved files.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Short labels for filenames and plot titles
    target_labels = {
        PCE_MOM_COL: 'MoM',
        PCE_YOY_COL: 'YoY',
    }

    saved_paths = []

    for target_col in summary['target'].unique():
        target_label = target_labels.get(target_col, target_col)

        for horizon in sorted(summary['horizon'].unique()):
            subset = (summary
                      .query("target == @target_col and horizon == @horizon")
                      .sort_values('RMSE_mean'))

            models = subset['model'].tolist()
            x = np.arange(len(models))
            bar_width = 0.35

            fig, ax1 = plt.subplots(figsize=(max(8, len(models) * 1.1), 5))

            # --- MAE and RMSE bars ---
            ax1.bar(
                x - bar_width / 2,
                subset['MAE_mean'],
                bar_width,
                yerr=subset['MAE_std'],
                label='MAE',
                color='steelblue',
                alpha=0.85,
                capsize=4,
            )
            ax1.bar(
                x + bar_width / 2,
                subset['RMSE_mean'],
                bar_width,
                yerr=subset['RMSE_std'],
                label='RMSE',
                color='tomato',
                alpha=0.85,
                capsize=4,
            )

            ax1.set_ylabel('Error (lower is better)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
            ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

            # --- Directional accuracy line on secondary axis ---
            ax2 = ax1.twinx()
            ax2.plot(
                x,
                subset['DirectionalAcc_mean'],
                color='seagreen',
                marker='o',
                linewidth=1.8,
                markersize=6,
                label='Directional Acc',
                zorder=5,
            )
            ax2.fill_between(
                x,
                subset['DirectionalAcc_mean'] - subset['DirectionalAcc_std'],
                subset['DirectionalAcc_mean'] + subset['DirectionalAcc_std'],
                color='seagreen',
                alpha=0.12,
            )
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel('Directional Accuracy (higher is better)')
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

            # --- Legend: merge both axes ---
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2,
                       loc='upper right', fontsize=8)

            # --- Title & layout ---
            ax1.set_title(
                f'PCE {target_label} | Horizon = {horizon} month{"s" if horizon > 1 else ""}\n'
                f'Models ranked by mean RMSE (left to right)',
                fontsize=11,
            )
            fig.tight_layout()

            filename = (f'time_series_experiment_results_{timestamp}'
                        f'_{target_label}_h{horizon}.png')
            out_path = RESULTS_DIR / filename
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved: {out_path}")
            saved_paths.append(out_path)

    return saved_paths
