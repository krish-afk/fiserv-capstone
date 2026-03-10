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
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data/live'
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'

PCE_RAW_COL = 'PersonalConsumptionExpenditures_normalized'
PCE_MOM_COL = 'PCE_MOM'
PCE_YOY_COL = 'PCE_YOY'


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


def derive_pce_targets(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Derive PCE target series matching peer methodology (regression.ipynb):

      PCE_MOM = .diff()    of median-grouped PersonalConsumptionExpenditures_normalized
      PCE_YOY = .diff(12)  of median-grouped PersonalConsumptionExpenditures_normalized

    Returns a date-indexed DataFrame ready to merge back into the raw df.
    """
    pce = df.groupby(date_col)[PCE_RAW_COL].median()
    return pd.DataFrame({
        PCE_MOM_COL: pce.diff(),
        PCE_YOY_COL: pce.diff(12),
    })


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
    Save one heatmap per target plus a CSV of the full summary table.

    Each heatmap has models as rows (sorted by composite rank) and
    metric × horizon as columns. Cells are colored by normalized performance
    (green = better) and annotated with actual values, making it easy to
    compare all models at a glance without axis clutter.

    Files written to RESULTS_DIR:
        time_series_experiment_results_{timestamp}_{target_label}_heatmap.png
        time_series_experiment_results_{timestamp}_summary.csv

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

    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    target_labels = {
        PCE_MOM_COL: 'MoM',
        PCE_YOY_COL: 'YoY',
    }

    saved_paths = []

    # --- CSV export ---
    csv_path = run_dir / f'time_series_experiment_results_{timestamp}_summary.csv'
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    saved_paths.append(csv_path)

    # --- One heatmap per target ---
    for target_col in summary['target'].unique():
        target_label = target_labels.get(target_col, target_col)
        target_summary = summary.query("target == @target_col")
        horizons = sorted(target_summary['horizon'].unique())

        # Build a wide matrix: rows = models, columns = (horizon, metric)
        # Lower MAE/RMSE is better; higher DirectionalAcc is better.
        metric_cols = ['MAE_mean', 'RMSE_mean', 'DirectionalAcc_mean']
        metric_labels = ['MAE', 'RMSE', 'Dir.Acc']
        higher_is_better = [False, False, True]

        col_tuples = [(f'H{h}', m) for h in horizons for m in metric_labels]
        col_index = pd.MultiIndex.from_tuples(col_tuples, names=['Horizon', 'Metric'])

        # Sort models by mean RMSE rank across horizons
        model_order = (target_summary
                       .groupby('model')['RMSE_rank']
                       .mean()
                       .sort_values()
                       .index.tolist())

        raw_values = pd.DataFrame(index=model_order, columns=col_index, dtype=float)
        fmt_values = pd.DataFrame(index=model_order, columns=col_index, dtype=str)

        for h in horizons:
            h_data = target_summary.query("horizon == @h").set_index('model')
            for mc, ml, hib in zip(metric_cols, metric_labels, higher_is_better):
                col = (f'H{h}', ml)
                vals = h_data.reindex(model_order)[mc]
                raw_values[col] = vals.values
                fmt_str = '{:.1%}' if ml == 'Dir.Acc' else '{:.4f}'
                fmt_values[col] = [fmt_str.format(v) for v in vals.values]

        # Normalize each column 0→1 where 1 = best performer
        norm_matrix = raw_values.copy().astype(float)
        for (h_label, m_label), hib in zip(col_tuples, higher_is_better * len(horizons)):
            col = (h_label, m_label)
            col_vals = norm_matrix[col]
            col_min, col_max = col_vals.min(), col_vals.max()
            span = col_max - col_min
            if span == 0:
                norm_matrix[col] = 0.5
            elif hib:
                norm_matrix[col] = (col_vals - col_min) / span
            else:
                norm_matrix[col] = (col_max - col_vals) / span

        n_models = len(model_order)
        n_cols = len(col_tuples)
        cell_w, cell_h = 1.6, 0.38
        fig_w = max(10, n_cols * cell_w + 3)
        fig_h = max(4, n_models * cell_h + 2)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        norm_array = norm_matrix.values.astype(float)
        im = ax.imshow(norm_array, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        # Annotate cells with formatted values
        for row_i, model in enumerate(model_order):
            for col_j, col_key in enumerate(col_tuples):
                text = fmt_values.loc[model, col_key]
                brightness = norm_array[row_i, col_j]
                text_color = 'black' if 0.25 < brightness < 0.85 else 'white'
                ax.text(col_j, row_i, text, ha='center', va='center',
                        fontsize=8, color=text_color)

        # Axis labels
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_order, fontsize=8)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [f'{h}\n{m}' for h, m in col_tuples],
            fontsize=8, ha='center'
        )
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        # Separator lines between horizon groups
        for sep in range(len(horizons) - 1):
            ax.axvline((sep + 1) * len(metric_labels) - 0.5,
                       color='white', linewidth=2)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label('Normalized performance\n(green = best)', fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Worst', 'Mid', 'Best'])

        ax.set_title(
            f'PCE {target_label} — Model comparison (rows sorted by mean RMSE rank)',
            fontsize=11, pad=18,
        )
        fig.tight_layout()

        filename = (f'time_series_experiment_results_{timestamp}'
                    f'_{target_label}_heatmap.png')
        out_path = run_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    return saved_paths


# ============================================================
# ACF / PACF ANALYSIS
# ============================================================

def plot_acf_pacf(series: pd.Series,
                  label: str,
                  lags: int = 24,
                  timestamp: str | None = None) -> Path:
    """
    Plot ACF and PACF for a series, with Ljung-Box p-values annotated.

    Saves one PNG to RESULTS_DIR and returns its path.

    Parameters
    ----------
    series : pd.Series
        The target time series to analyse.
    label : str
        Short label used in the filename and plot title (e.g. 'MoM', 'YoY').
    lags : int
        Number of lags to display.
    timestamp : str | None
        Timestamp for the filename. Defaults to now (YYYYMMDD_HHMMSS).

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ljung-Box test: store p-values at a few representative lags
    lb_lags = [1, 6, 12, 24]
    lb_lags = [l for l in lb_lags if l <= lags]
    lb_result = acorr_ljungbox(series.dropna(), lags=lb_lags, return_df=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05,
             title=f'ACF — PCE {label}')
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05,
              method='ywm', title=f'PACF — PCE {label}')

    for ax in axes:
        ax.set_xlabel('Lag (months)')
        ax.axhline(0, color='black', linewidth=0.8)

    # Annotate with Ljung-Box results
    lb_lines = [f'Ljung-Box test (H\u2080: white noise):']
    for lag, row in lb_result.iterrows():
        sig = '**' if row['lb_pvalue'] < 0.01 else ('*' if row['lb_pvalue'] < 0.05 else '')
        lb_lines.append(f'  lag {int(lag):>2}: p = {row["lb_pvalue"]:.3f}{sig}')
    lb_lines.append('* p<0.05   ** p<0.01')

    fig.text(0.98, 0.5, '\n'.join(lb_lines),
             ha='right', va='center', fontsize=8,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f'Autocorrelation structure — PCE {label}', fontsize=12, y=1.01)
    fig.tight_layout()

    out_path = run_dir / f'acf_pacf_{timestamp}_{label}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


# ============================================================
# DIEBOLD-MARIANO TEST
# ============================================================

def _dm_test(losses_model: np.ndarray,
             losses_naive: np.ndarray,
             horizon: int) -> tuple[float, float]:
    """
    Harvey-Leybourne-Newbold corrected Diebold-Mariano test (two-sided).

    Uses squared-error loss. A positive DM statistic means the model has
    higher loss than the naive baseline (naive is better).

    Parameters
    ----------
    losses_model : np.ndarray
        Per-fold MSE for the model under test.
    losses_naive : np.ndarray
        Per-fold MSE for the naive baseline.
    horizon : int
        Forecast horizon (determines HAC lag order).

    Returns
    -------
    (dm_stat, p_value) : tuple[float, float]
        NaN for both if variance is non-positive.
    """
    d = losses_model - losses_naive
    T = len(d)
    if T < 2:
        return np.nan, np.nan

    d_bar = d.mean()

    # Newey-West HAC variance with (h-1) lags
    gamma0 = np.sum((d - d_bar) ** 2) / T
    var_d = gamma0
    for k in range(1, horizon):
        if k < T:
            gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
            var_d += 2 * gamma_k

    if var_d <= 0:
        return np.nan, np.nan

    # Harvey et al. small-sample correction
    correction = (T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T
    var_d_corrected = correction * var_d

    if var_d_corrected <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(var_d_corrected)
    p_value = float(2 * scipy_stats.t.sf(abs(dm_stat), df=T - 1))
    return float(dm_stat), p_value


def run_diebold_mariano(results_df: pd.DataFrame,
                        naive_name: str = 'naive',
                        timestamp: str | None = None) -> pd.DataFrame:
    """
    Run pairwise Diebold-Mariano tests comparing every model against the
    naive baseline, for each (target, horizon) combination.

    Uses per-fold MSE (= RMSE²) as the loss function. Applies the
    Harvey-Leybourne-Newbold small-sample correction and t(T-1) critical
    values. Saves a p-value heatmap and a CSV to RESULTS_DIR.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw fold-level output from run_experiment() — must contain columns
        target, model, horizon, fold, RMSE.
    naive_name : str
        Name of the naive predictor in results_df.
    timestamp : str | None
        Timestamp for filenames. Defaults to now (YYYYMMDD_HHMMSS).

    Returns
    -------
    pd.DataFrame
        One row per (target, model, horizon) with columns:
        DM_stat, p_value, significant_05, better_than_naive.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.copy()
    df['MSE'] = df['RMSE'] ** 2

    records = []
    for (target, horizon), group in df.groupby(['target', 'horizon']):
        naive_data = (group[group['model'] == naive_name]
                      .sort_values('fold')['MSE'].values)
        if len(naive_data) == 0:
            print(f"Warning: naive model '{naive_name}' not found for "
                  f"target={target}, horizon={horizon}. Skipping.")
            continue

        for model, model_group in group.groupby('model'):
            if model == naive_name:
                continue
            model_losses = model_group.sort_values('fold')['MSE'].values
            n = min(len(model_losses), len(naive_data))
            dm_stat, p_val = _dm_test(model_losses[:n], naive_data[:n], horizon)

            records.append({
                'target': target,
                'model': model,
                'horizon': horizon,
                'DM_stat': round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
                'p_value': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                'significant_05': (p_val < 0.05) if not np.isnan(p_val) else False,
                'better_than_naive': (dm_stat < 0) if not np.isnan(dm_stat) else False,
            })

    dm_df = pd.DataFrame(records)

    if dm_df.empty:
        print("No DM test results generated.")
        return dm_df

    # --- CSV ---
    csv_path = run_dir / f'diebold_mariano_{timestamp}.csv'
    dm_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # --- Heatmap of p-values ---
    target_labels = {PCE_MOM_COL: 'MoM', PCE_YOY_COL: 'YoY'}
    horizons = sorted(dm_df['horizon'].unique())
    targets = dm_df['target'].unique()
    models = (dm_df.groupby('model')['p_value'].mean()
              .sort_values().index.tolist())

    col_tuples = [(target_labels.get(t, t), f'H{h}') for t in targets
                  for h in horizons]
    col_labels = [f'{tl}\n{hl}' for tl, hl in col_tuples]

    p_matrix = np.full((len(models), len(col_tuples)), np.nan)
    dm_matrix = np.full((len(models), len(col_tuples)), np.nan)

    for col_j, (t, h) in enumerate(
            (t, h) for t in targets for h in horizons):
        subset = dm_df.query("target == @t and horizon == @h")
        for row_i, model in enumerate(models):
            row = subset[subset['model'] == model]
            if not row.empty:
                p_matrix[row_i, col_j] = row['p_value'].values[0]
                dm_matrix[row_i, col_j] = row['DM_stat'].values[0]

    # Colour: green = significantly better than naive,
    #         red = significantly worse, grey = not significant
    cmap_arr = np.zeros_like(p_matrix)  # 0 = not significant (grey)
    for ri in range(len(models)):
        for ci in range(len(col_tuples)):
            p = p_matrix[ri, ci]
            dm = dm_matrix[ri, ci]
            if np.isnan(p):
                cmap_arr[ri, ci] = 0.5  # mid-grey
            elif p < 0.05 and dm < 0:
                cmap_arr[ri, ci] = 1.0  # better than naive
            elif p < 0.05 and dm > 0:
                cmap_arr[ri, ci] = -1.0  # worse than naive
            # else 0 = not significant

    cell_w, cell_h = 1.8, 0.40
    fig_w = max(10, len(col_tuples) * cell_w + 3)
    fig_h = max(4, len(models) * cell_h + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cmap_arr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    for ri, model in enumerate(models):
        for ci in range(len(col_tuples)):
            p = p_matrix[ri, ci]
            dm = dm_matrix[ri, ci]
            if np.isnan(p):
                cell_text = 'N/A'
            else:
                stars = '**' if p < 0.01 else ('*' if p < 0.05 else '')
                cell_text = f'p={p:.3f}{stars}\nDM={dm:+.2f}'
            brightness = cmap_arr[ri, ci]
            text_color = 'black' if -0.6 < brightness < 0.6 else 'white'
            ax.text(ci, ri, cell_text, ha='center', va='center',
                    fontsize=7, color=text_color, linespacing=1.4)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Separator between target groups
    n_horizons = len(horizons)
    for sep in range(len(targets) - 1):
        ax.axvline((sep + 1) * n_horizons - 0.5, color='white', linewidth=2)

    legend_patches = [
        mpatches.Patch(color='#1a9641', label='Significantly better than naive (p<0.05)'),
        mpatches.Patch(color='#d7191c', label='Significantly worse than naive (p<0.05)'),
        mpatches.Patch(color='#ffffbf', label='No significant difference'),
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              bbox_to_anchor=(1, -0.12), fontsize=8, ncol=3)

    ax.set_title(
        f'Diebold-Mariano test vs. naive baseline\n'
        f'(Harvey-Leybourne-Newbold correction, squared-error loss)',
        fontsize=11, pad=18,
    )
    fig.tight_layout()

    heatmap_path = run_dir / f'diebold_mariano_{timestamp}_heatmap.png'
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {heatmap_path}")

    return dm_df
