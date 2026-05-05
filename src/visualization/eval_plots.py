import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from src.utils.config import config
from src.trading.strategy import load_latest_run_frames, select_best_model

def calculate_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100 
    return mape, dir_acc

# ---------------------------------------------------------
# PRESENTATION / DASHBOARD MODEL COMPARISON HELPERS
# ---------------------------------------------------------
LOWER_IS_BETTER_METRICS = {"rmse", "mae", "mape", "me", "abs_error", "absolute_error"}
HIGHER_IS_BETTER_METRICS = {"dir_acc", "directional_accuracy", "r2"}
AR3_BASELINE_MODEL_NAMES = ("ar3_baseline", "ar_3_baseline", "ar(3)_baseline", "ar3")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")


def _panel_display_name(panel_name: str) -> str:
    pieces = str(panel_name).split("_")
    target = pieces[0].upper() if pieces else str(panel_name)

    if "mom" in pieces:
        return f"{target} MoM"
    if "yoy" in pieces:
        return f"{target} YoY"
    return f"{target} Level"


def _model_family_label(model_name: str) -> str:
    name = str(model_name).lower()

    checks = [
        ("xgboost", "XGBoost"),
        ("randomforest", "Random Forest"),
        ("random_forest", "Random Forest"),
        ("gradientboosting", "Gradient Boosting"),
        ("gradient_boosting", "Gradient Boosting"),
        ("gbm", "Gradient Boosting"),
        ("theta", "Theta"),
        ("etsx", "ETSX"),
        ("arimax", "ARIMAX"),
        ("arima", "ARIMA"),
        ("ridge", "Ridge"),
        ("lasso", "Lasso"),
        ("ols", "OLS"),
        ("ets", "ETS"),
        ("naive", "Naive"),
        ("mean", "Mean"),
        ("ar3", "AR(3)"),
        ("ar_3", "AR(3)"),
        ("ar(3)", "AR(3)"),
    ]

    for needle, label in checks:
        if needle in name:
            return label

    return str(model_name).split("_")[0].replace("-", " ").title()


def _trial_display_label(row: pd.Series, include_feature_set: bool = True) -> str:
    family = _model_family_label(row.get("model_name", "model"))
    feature_set = row.get("feature_set")

    if family == "AR(3)":
        return "AR(3) Baseline"

    if (
        include_feature_set
        and isinstance(feature_set, str)
        and feature_set
        and feature_set.lower() != "none"
    ):
        return f"{family} | {feature_set}"

    return family


def _get_dir_acc_col(metrics_df: pd.DataFrame) -> str:
    for col in ["dir_acc", "directional_accuracy"]:
        if col in metrics_df.columns:
            return col
    raise ValueError("metrics_df must contain 'dir_acc' or 'directional_accuracy'.")


def _get_mape_col(metrics_df: pd.DataFrame) -> str:
    if "mape" not in metrics_df.columns:
        raise ValueError("metrics_df must contain 'mape'.")
    return "mape"


def _top_ranked_trials(
    metrics_df: pd.DataFrame,
    panel_name: str,
    top_k: int = 5,
) -> tuple[pd.DataFrame, str, str]:
    """
    Rank models using:
    1) directional accuracy DESC
    2) MAPE ASC

    Returns:
        top_rows, dir_acc_col, mape_col
    """
    dir_acc_col = _get_dir_acc_col(metrics_df)
    mape_col = _get_mape_col(metrics_df)

    panel_metrics = metrics_df[metrics_df["panel_name"] == panel_name].copy()
    if panel_metrics.empty:
        return panel_metrics, dir_acc_col, mape_col

    panel_metrics[dir_acc_col] = pd.to_numeric(panel_metrics[dir_acc_col], errors="coerce")
    panel_metrics[mape_col] = pd.to_numeric(panel_metrics[mape_col], errors="coerce")

    panel_metrics = panel_metrics.dropna(subset=[dir_acc_col, mape_col])

    panel_metrics = panel_metrics.sort_values(
        by=[dir_acc_col, mape_col],
        ascending=[False, True],
    )

    return panel_metrics.head(int(top_k)).copy(), dir_acc_col, mape_col

def _find_model_row(
    metrics_df: pd.DataFrame,
    panel_name: str,
    model_names: tuple[str, ...],
) -> pd.Series | None:
    """Return the best matching metrics row for a named model on a panel."""
    panel_metrics = metrics_df[metrics_df["panel_name"] == panel_name].copy()
    if panel_metrics.empty:
        return None

    wanted = {m.lower() for m in model_names}
    panel_metrics["__model_name_lower__"] = panel_metrics["model_name"].astype(str).str.lower()
    matches = panel_metrics[panel_metrics["__model_name_lower__"].isin(wanted)].copy()
    if matches.empty:
        return None

    dir_acc_col = _get_dir_acc_col(metrics_df)
    mape_col = _get_mape_col(metrics_df)
    matches[dir_acc_col] = pd.to_numeric(matches[dir_acc_col], errors="coerce")
    matches[mape_col] = pd.to_numeric(matches[mape_col], errors="coerce")
    matches = matches.sort_values(by=[dir_acc_col, mape_col], ascending=[False, True])
    return matches.iloc[0].drop(labels=["__model_name_lower__"], errors="ignore")


def _forecast_series_for_row(
    panel_forecasts: pd.DataFrame,
    row: pd.Series,
    label: str,
) -> pd.Series:
    trial_forecasts = panel_forecasts[
        (panel_forecasts["model_name"] == row["model_name"])
        & (panel_forecasts["feature_set"] == row["feature_set"])
    ].copy()

    if trial_forecasts.empty:
        return pd.Series(dtype="float64", name=label)

    return (
        trial_forecasts
        .set_index("date")
        .sort_index()["y_pred"]
        .rename(label)
    )


def _metric_suffix(row: pd.Series, dir_acc_col: str, mape_col: str) -> str:
    suffix_parts = []

    dir_acc_value = row.get(dir_acc_col)
    if pd.notna(dir_acc_value):
        suffix_parts.append(f"Dir. Acc {_ratio_to_pct_points(dir_acc_value):.2f}%")

    mape_value = row.get(mape_col)
    if pd.notna(mape_value):
        suffix_parts.append(f"MAPE {_metric_to_pct_points(mape_value):.2f}%")

    return " | " + " | ".join(suffix_parts) if suffix_parts else ""

def _metric_to_pct_points(value) -> float:
    # MAPE is now stored as percentage points.
    # Example: 0.774 means 0.774%, 77.4 means 77.4%.
    return float(value)


def _ratio_to_pct_points(value) -> float:
    # dir_acc is still stored as a ratio.
    # Example: 0.9565 means 95.65%.
    return float(value) * 100.0


def _write_plot(fig: go.Figure, html_path: Path, png_path: Optional[Path] = None) -> None:
    fig.write_html(str(html_path))

    if png_path is None:
        return

    try:
        fig.write_image(str(png_path), width=1280, height=720, scale=2)
    except Exception as e:
        print(f"[PLOT] PNG export skipped for {html_path.name}: {e}")

# ---------------------------------------------------------
# CORE PLOT 1: FORECAST OVERLAY
# ---------------------------------------------------------

def generate_top_k_forecast_plot(
    plot_dir: Path,
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    panel_name: str,
    top_k: int = 5,
    baseline_model_names: tuple[str, ...] = AR3_BASELINE_MODEL_NAMES,
):
    """
    Top-K models ranked by:
    1) Directional Accuracy (descending)
    2) MAPE (ascending)

    Then plot those forecasts against the actual series.
    """
    print(f"[PLOT] Generating Top {top_k} Forecasts vs Actual for {panel_name}...")

    top_rows, dir_acc_col, mape_col = _top_ranked_trials(
        metrics_df=metrics_df,
        panel_name=panel_name,
        top_k=top_k,
    )

    if top_rows.empty:
        print(f"[PLOT] No ranked models available for panel '{panel_name}'.")
        return

    panel_forecasts = forecasts_df[forecasts_df["panel_name"] == panel_name].copy()
    if panel_forecasts.empty:
        print(f"[PLOT] No forecasts available for panel '{panel_name}'.")
        return

    panel_forecasts["date"] = pd.to_datetime(panel_forecasts["date"])

    actual = (
        panel_forecasts.sort_values("date")
        .groupby("date", as_index=True)["y_true"]
        .first()
        .rename("Actual")
    )

    plot_df = actual.to_frame()
    labels_seen: set[str] = set()
    selected_labels: list[tuple[str, pd.Series, pd.Series]] = []

    for _, row in top_rows.iterrows():
        trial_forecasts = panel_forecasts[
            (panel_forecasts["model_name"] == row["model_name"])
            & (panel_forecasts["feature_set"] == row["feature_set"])
        ].copy()

        if trial_forecasts.empty:
            continue

        label = _trial_display_label(row)

        if label in labels_seen:
            label = f"{label} #{len(labels_seen) + 1}"

        labels_seen.add(label)

        series = (
            trial_forecasts
            .set_index("date")
            .sort_index()["y_pred"]
            .rename(label)
        )

        plot_df = plot_df.join(series, how="left")
        selected_labels.append((label, series, row))

    plot_df = plot_df.dropna(subset=["Actual"])

    if not selected_labels:
        print(f"[PLOT] No selected forecast series were found for panel '{panel_name}'.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="black", width=3),
        )
    )

    for rank, (label, _, row) in enumerate(selected_labels, start=1):
        suffix = _metric_suffix(row, dir_acc_col, mape_col)

        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[label],
                mode="lines",
                name=f"#{rank} {label}{suffix}",
                line=dict(width=2),
            )
        )
    
    baseline_row = _find_model_row(metrics_df, panel_name, baseline_model_names)
    baseline_added = False
    selected_model_names = {str(row.get("model_name")).lower() for _, _, row in selected_labels}

    if baseline_row is not None and str(baseline_row.get("model_name")).lower() not in selected_model_names:
        baseline_label = "AR(3) Baseline"
        baseline_series = _forecast_series_for_row(panel_forecasts, baseline_row, baseline_label)

        if not baseline_series.empty:
            plot_df = plot_df.join(baseline_series, how="left")
            suffix = _metric_suffix(baseline_row, dir_acc_col, mape_col)
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[baseline_label],
                    mode="lines",
                    name=f"{baseline_label}{suffix}",
                    line=dict(color="gray", width=3, dash="dot"),
                )
            )
            baseline_added = True
    elif baseline_row is None:
        print(f"[PLOT] AR(3) baseline not found for panel '{panel_name}'. Run the experiment stage after adding AR3BaselineForecaster.")

    panel_title = _panel_display_name(panel_name)

    fig.update_layout(
        title=(
            f"{panel_title}: Top {len(selected_labels)} Forecasts"
            f"{' + AR(3) Baseline' if baseline_added else ''} vs Actual"
        ),
        xaxis_title="Date",
        yaxis_title="Target value",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.38,
            xanchor="left",
            x=0,
        ),
        height=700,
        margin=dict(t=90, b=200),
    )

    slug = _slug(panel_name)

    _write_plot(
        fig,
        plot_dir / f"dash_top{len(selected_labels)}_forecasts_vs_actual_{slug}.html",
        plot_dir / f"slide_top{len(selected_labels)}_forecasts_vs_actual_{slug}.png",
    )

def generate_top_k_mape_comparison_plot(
    plot_dir: Path,
    metrics_df: pd.DataFrame,
    panel_name: str,
    top_k: int = 5,
):
    """
    Build a MAPE comparison bar chart for the SAME Top-K models chosen using:
    1) Directional Accuracy DESC
    2) MAPE ASC
    """
    print(f"[PLOT] Generating Top {top_k} MAPE comparison for {panel_name}...")

    top_rows, dir_acc_col, mape_col = _top_ranked_trials(
        metrics_df=metrics_df,
        panel_name=panel_name,
        top_k=top_k,
    )

    if top_rows.empty:
        print(f"[PLOT] No ranked models available for panel '{panel_name}'.")
        return

    plot_rows = []
    labels_seen: set[str] = set()

    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        label = _trial_display_label(row)

        if label in labels_seen:
            label = f"{label} #{rank}"

        labels_seen.add(label)

        plot_rows.append(
            {
                "rank": f"#{rank}",
                "model": label,
                "mape_pct": _metric_to_pct_points(row[mape_col]),
                "dir_acc_pct": _ratio_to_pct_points(row[dir_acc_col]),
            }
        )

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        print(f"[PLOT] No MAPE comparison rows available for panel '{panel_name}'.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["model"],
            y=plot_df["mape_pct"],
            text=[f"{v:.2f}%" for v in plot_df["mape_pct"]],
            textposition="outside",
            customdata=plot_df[["rank", "dir_acc_pct"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Rank: %{customdata[0]}<br>"
                "MAPE: %{y:.2f}%<br>"
                "Directional Accuracy: %{customdata[1]:.2f}%<extra></extra>"
            ),
            name="MAPE",
        )
    )

    panel_title = _panel_display_name(panel_name)

    fig.update_layout(
        title=f"MAPE Error Comparison: Top {len(plot_df)} {panel_title} Models",
        xaxis_title="Model",
        yaxis_title="MAPE (%)",
        template="plotly_white",
        height=650,
        margin=dict(t=90, b=150),
        showlegend=False,
    )

    fig.update_xaxes(tickangle=-25)

    slug = _slug(panel_name)

    _write_plot(
        fig,
        plot_dir / f"dash_top{len(plot_df)}_mape_comparison_{slug}.html",
        plot_dir / f"slide_top{len(plot_df)}_mape_comparison_{slug}.png",
    )

def generate_forecast_plot(
    plot_dir: Path,
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    panel_name: str,
    baseline_model_name: str = "ar3_baseline",
):
    print(f"[PLOT] Generating Forecast Overlay for {panel_name}...")
    best_df = select_best_model(forecasts_df, metrics_df, panel_name=panel_name)
    best_model_full_name = best_df.iloc[0]["model_name"]
    display_model_name = best_model_full_name.split("_")[0].upper()
    best_df = best_df.set_index("date").sort_index()

    baseline_names = (baseline_model_name,) + tuple(
        name for name in AR3_BASELINE_MODEL_NAMES if name != baseline_model_name
    )
    baseline_row = _find_model_row(metrics_df, panel_name, baseline_names)

    plot_df = pd.DataFrame({"Actual": best_df["y_true"], "Best Model": best_df["y_pred"]})
    baseline_label = "AR(3) Baseline"

    if baseline_row is not None:
        panel_forecasts = forecasts_df[forecasts_df["panel_name"] == panel_name].copy()
        panel_forecasts["date"] = pd.to_datetime(panel_forecasts["date"])
        baseline_series = _forecast_series_for_row(panel_forecasts, baseline_row, baseline_label)
        if not baseline_series.empty:
            plot_df = plot_df.join(baseline_series, how="left")
    else:
        print(f"[PLOT] AR(3) baseline not found for panel '{panel_name}'.")

    plot_df = plot_df.dropna(subset=["Actual", "Best Model"])

    best_mape, _ = calculate_metrics(plot_df["Actual"], plot_df["Best Model"])
    base_mape = None
    if baseline_label in plot_df.columns and plot_df[baseline_label].notna().any():
        baseline_eval = plot_df.dropna(subset=[baseline_label])
        base_mape, _ = calculate_metrics(baseline_eval["Actual"], baseline_eval[baseline_label])

    subtitle = f"Best Model (MAPE: {best_mape:.2f}%)"
    if base_mape is not None:
        subtitle += f" vs AR(3) Baseline (MAPE: {base_mape:.2f}%)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Actual"], mode="lines", name="Actual (Ground Truth)", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Best Model"], mode="lines", name=f"Best ({display_model_name})", line=dict(color="blue", width=2, dash="dash")))
    if baseline_label in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[baseline_label], mode="lines", name=baseline_label, line=dict(color="gray", width=2, dash="dot")))

    fig.update_layout(
        title=f"Out-of-Sample Predictions: {panel_name}<br><sup>{subtitle}</sup>",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(str(plot_dir / f"dash_forecast_{panel_name}.html"))

# ---------------------------------------------------------
# ELECTIVE PLOT 1: MACRO ENVIRONMENT PROFILER
# ---------------------------------------------------------
def generate_macro_profiler(plot_dir: Path, forecasts_df: pd.DataFrame, metrics_df: pd.DataFrame, panel_name: str):
    print("[PLOT] Generating Macro Environment Profiler...")
    try:
        best_df = select_best_model(forecasts_df, metrics_df, panel_name=panel_name).set_index('date').sort_index()
        best_df['Abs_Error_Pct'] = np.abs((best_df['y_true'] - best_df['y_pred']) / best_df['y_true']) * 100
        
        master_path = Path(config.get("paths", {}).get("processed_data", "data/processed/")) / "master.csv"
        if not master_path.exists(): return
            
        macro_df = pd.read_csv(master_path, parse_dates=['date']).set_index('date')
        macro_pct = macro_df.pct_change(12) * 100 
        macro_pct = macro_pct.replace([np.inf, -np.inf], np.nan).dropna(axis=1, thresh=len(macro_pct)*0.8)
        
        merged = best_df[['Abs_Error_Pct']].join(macro_pct, how='inner').dropna()
        macro_cols = [c for c in merged.columns if c != 'Abs_Error_Pct'][:8]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=merged.index, y=merged['Abs_Error_Pct'], name='Model Error (%)', marker_color='rgba(0, 0, 255, 0.3)'), secondary_y=False)
        for i, col in enumerate(macro_cols): fig.add_trace(go.Scatter(x=merged.index, y=merged[col], mode='lines', name=f"{col} (% YoY)", line=dict(width=2), visible=(i==0)), secondary_y=True)

        buttons = [dict(label=col, method="update", args=[{"visible": [True] + [j == i for j in range(len(macro_cols))]}, {"title": f"Model Error vs {col} (YoY %)"}]) for i, col in enumerate(macro_cols)]
        fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=1.1, y=1.15)], title=f"Macro Environment Profiler: Error vs {macro_cols[0]} (YoY %)", template="plotly_white", hovermode="x unified")
        fig.write_html(str(plot_dir / "dash_macro_profiler.html"))
    except Exception as e:
        print(f"[PLOT] Warning - Macro Profiler skipped: {e}")

# ---------------------------------------------------------
# UNIVERSAL TRADING PLOTS (EQUITY & KPIS)
# ---------------------------------------------------------
def _plot_universal_performance(equity_csv: Path, trades_csv: Path, json_path: Path, plot_dir: Path, strategy_name: str):
    """Generates Equity Curve and KPIs for ALL strategies."""
    initial_capital = config.get("trading", {}).get("portfolio", {}).get("initial_cash", 100000.0)

    # 1. KPIs from JSON
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    total_return = results.get('return_pct', 0.0)
    win_rate = results.get('win_rate', 0.0) * 100
    max_dd = results.get('max_drawdown_pct', 0.0)

    fig_kpi = go.Figure()
    fig_kpi.add_trace(go.Indicator(mode="number", value=total_return, number={"suffix": "%", "valueformat": ".2f"}, title={"text": "Total Return"}, domain={'row': 0, 'column': 0}))
    fig_kpi.add_trace(go.Indicator(mode="number", value=win_rate, number={"suffix": "%", "valueformat": ".1f"}, title={"text": "Win Rate"}, domain={'row': 0, 'column': 1}))
    fig_kpi.add_trace(go.Indicator(mode="number", value=max_dd, number={"suffix": "%", "valueformat": ".2f"}, title={"text": "Max Drawdown"}, domain={'row': 0, 'column': 2}))
    fig_kpi.update_layout(grid={'rows': 1, 'columns': 3, 'pattern': "independent"}, template="plotly_white", height=250)
    fig_kpi.write_html(str(plot_dir / "dash_kpis.html"))

    # 2. Equity Curve & Drawdown
    df_eq = pd.read_csv(equity_csv, index_col=0, parse_dates=True)
    df_eq.columns = ['Cumulative_Real']
    
    # Load Trades to reconstruct Pure PnL and fix the starting point
    df_trades = pd.read_csv(trades_csv)
    
    if 'gross_pnl' in df_trades.columns:  # Forecastex Strategy
        df_trades['date'] = pd.to_datetime(df_trades['date'])
        
        # Calculate Pure Equity mathematically mirroring the backtester
        pure_equity = initial_capital
        pure_points = {df_eq.index[0]: pure_equity} if not df_eq.empty else {}
        for _, row in df_trades.sort_values('date').iterrows():
            pure_equity += row['gross_pnl']
            pure_points[row['date']] = pure_equity
            
        pure_series = pd.Series(pure_points)
        df_eq['Cumulative_Pure'] = pure_series.reindex(df_eq.index).ffill()

    elif 'gross_return' in df_trades.columns:  # ETF Strategy
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        
        if not df_trades.empty:
            first_entry = df_trades['entry_date'].min()
            if first_entry not in df_eq.index or first_entry < df_eq.index.min():
                df_eq.loc[first_entry, 'Cumulative_Real'] = initial_capital
                df_eq = df_eq.sort_index()

            pure_equity = initial_capital
            pure_points = {first_entry: pure_equity}
            for _, row in df_trades.sort_values('exit_date').iterrows():
                pure_equity *= (1.0 + row['gross_return'])
                pure_points[row['exit_date']] = pure_equity
                
            pure_series = pd.Series(pure_points)
            df_eq['Cumulative_Pure'] = pure_series.reindex(df_eq.index).ffill()

    df_eq['Peak'] = df_eq['Cumulative_Real'].cummax()
    df_eq['Drawdown'] = (df_eq['Cumulative_Real'] - df_eq['Peak']) / df_eq['Peak'] * 100

    # --- FETCH S&P 500 BASELINE DYNAMICALLY ---
    try:
        import yfinance as yf
        start_str = df_eq.index.min().strftime('%Y-%m-%d')
        end_str = (df_eq.index.max() + pd.Timedelta(days=5)).strftime('%Y-%m-%d') # Cushion to ensure final day is captured
        
        spy = yf.download("SPY", start=start_str, end=end_str, progress=False)
        if not spy.empty:
            # Handle yfinance multi-index vs single-index output variations
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = spy["Close"]["SPY"]
            else:
                spy_close = spy["Close"]
            
            # Align exact trading days, backfilling the first day if it lands on a weekend
            spy_aligned = spy_close.reindex(df_eq.index, method="ffill").bfill()
            
            # Normalize SPY value to match the initial capital amount
            df_eq['SPY_Baseline'] = (spy_aligned / spy_aligned.iloc[0]) * initial_capital
    except Exception as e:
        print(f"[PLOT] Warning: Could not fetch SPY baseline: {e}")
    # ------------------------------------------

    fig_perf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    if 'Cumulative_Pure' in df_eq.columns:
        fig_perf.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Cumulative_Pure'], mode='lines', name='Pure Portfolio (0-Fee)', line=dict(color='green', width=2)), row=1, col=1)
    
    fig_perf.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Cumulative_Real'], mode='lines', name='Realistic Portfolio (IBKR fee)', line=dict(color='black', width=2, dash='dash')), row=1, col=1)
    
    # Add the SPY trace if the fetch was successful
    if 'SPY_Baseline' in df_eq.columns:
        fig_perf.add_trace(go.Scatter(x=df_eq.index, y=df_eq['SPY_Baseline'], mode='lines', name='S&P 500 (SPY)', line=dict(color='blue', width=2, dash='dot')), row=1, col=1)

    fig_perf.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Drawdown'], mode='lines', fill='tozeroy', name='Drawdown (%)', line=dict(color='red', width=1), fillcolor='rgba(255,0,0,0.2)'), row=2, col=1)
    fig_perf.update_layout(title=f"Trading Performance & Drawdown ({strategy_name})", template="plotly_white", hovermode="x unified", height=700)
    fig_perf.write_html(str(plot_dir / "dash_trading_performance.html"))

    # 3. Rolling Risk Suite
    df_monthly = df_eq.resample('ME').last().ffill()
    df_monthly['Ret'] = df_monthly['Cumulative_Real'].pct_change().fillna(0)
    
    ann_ret = df_monthly['Ret'].rolling(12, min_periods=2).mean() * 12
    ann_vol = df_monthly['Ret'].rolling(12, min_periods=2).std() * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-6)
    
    downside = df_monthly['Ret'].copy()
    downside[downside > 0] = 0
    down_vol = downside.rolling(12, min_periods=2).std() * np.sqrt(12)
    sortino = ann_ret / (down_vol + 1e-6)
    
    max_dd = df_eq['Drawdown'].resample('ME').min().rolling(12, min_periods=2).min() / 100
    calmar = ann_ret / (np.abs(max_dd) + 1e-6)
    
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(x=df_monthly.index, y=sharpe, name="Sharpe Ratio", mode='lines', line=dict(color='purple', width=2), visible=True))
    fig_risk.add_trace(go.Scatter(x=df_monthly.index, y=sortino, name="Sortino Ratio", mode='lines', line=dict(color='blue', width=2), visible=False))
    fig_risk.add_trace(go.Scatter(x=df_monthly.index, y=calmar, name="Calmar Ratio", mode='lines', line=dict(color='orange', width=2), visible=False))
    
    buttons_risk = [
        dict(label="Sharpe", method="update", args=[{"visible": [True, False, False]}, {"title": "Rolling 12-Month Sharpe Ratio"}]),
        dict(label="Sortino", method="update", args=[{"visible": [False, True, False]}, {"title": "Rolling 12-Month Sortino Ratio"}]),
        dict(label="Calmar", method="update", args=[{"visible": [False, False, True]}, {"title": "Rolling 12-Month Calmar Ratio"}])
    ]
    fig_risk.update_layout(updatemenus=[dict(active=0, buttons=buttons_risk, x=1.15, y=1.15)], title="Rolling 12-Month Sharpe Ratio", template="plotly_white", yaxis_title="Ratio")
    fig_risk.write_html(str(plot_dir / "dash_rolling_risk.html"))

# ---------------------------------------------------------
# EVENT CONTRACT EXCLUSIVES
# ---------------------------------------------------------
def _plot_forecastex_exclusives(csv_path: Path, plot_dir: Path):
    """Generates Volatility Scatter and Heatmap for Prediction Markets."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # 1. Volatility Scatter
    df['Surprise_Magnitude'] = np.abs(df['actual'] - df['strike'])
    df['Abs_Edge'] = np.abs(df['edge'])
    
    wins = df[df['won'] == True]
    losses = df[df['won'] == False]
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=wins['Surprise_Magnitude'], y=wins['Abs_Edge'], mode='markers', name='WIN', marker=dict(color='rgba(0, 150, 0, 0.6)', size=14, line=dict(width=1, color='darkgreen')), text=wins.index.strftime('%b %Y'), hoverinfo='text'))
    fig_scatter.add_trace(go.Scatter(x=losses['Surprise_Magnitude'], y=losses['Abs_Edge'], mode='markers', name='LOSS', marker=dict(color='rgba(255, 0, 0, 0.6)', size=14, line=dict(width=1, color='darkred')), text=losses.index.strftime('%b %Y'), hoverinfo='text'))
    fig_scatter.update_layout(title="Conviction Magnitude vs Economic Surprise", xaxis_title="Economic Surprise |Actual - Consensus| (Log Scale)", yaxis_title="Model Conviction |Calculated Edge|", xaxis_type="log", template="plotly_white")
    fig_scatter.write_html(str(plot_dir / "dash_volatility_scatter.html"))

    # 2. Execution Heatmap (Restored Hover Tooltips)
    df_heat = df.copy()
    df_heat['Year'], df_heat['Month'] = df_heat.index.year, df_heat.index.month
    
    pivot_pnl = df_heat.pivot_table(index='Year', columns='Month', values='net_pnl', aggfunc='sum')
    pivot_edge = df_heat.pivot_table(index='Year', columns='Month', values='edge', aggfunc='mean')
    pivot_outcome = df_heat.pivot_table(index='Year', columns='Month', values='won', aggfunc='first')
    
    for m in range(1, 13):
        if m not in pivot_pnl.columns: 
            pivot_pnl[m] = np.nan
            pivot_edge[m] = np.nan
            pivot_outcome[m] = np.nan
            
    pivot_pnl = pivot_pnl.reindex(range(pivot_pnl.index.min(), pivot_pnl.index.max() + 1))
    pivot_edge = pivot_edge.reindex(pivot_pnl.index)
    pivot_outcome = pivot_outcome.reindex(pivot_pnl.index)
    
    hover_text = []
    for year in pivot_pnl.index:
        row_text = []
        for month in range(1, 13):
            pnl = pivot_pnl.loc[year, month]
            outcome = pivot_outcome.loc[year, month]
            if pd.isna(pnl) or (pnl == 0 and pd.isna(outcome)):
                row_text.append(f"<b>{year}-{month:02d}</b><br>No Trade Triggered")
            else:
                edge = pivot_edge.loc[year, month]
                out_str = "WIN" if outcome == True else "LOSS"
                row_text.append(f"<b>{year}-{month:02d}</b><br>Outcome: {out_str}<br>Realized PnL: ${pnl:,.2f}<br>Raw Edge: {edge:.3f}")
        hover_text.append(row_text)
    
    custom_colorscale = [[0.0, 'rgba(214, 39, 40, 0.9)'], [0.5, 'rgba(240, 240, 240, 1.0)'], [1.0, 'rgba(44, 160, 44, 0.9)']]
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_pnl.fillna(0).values, 
        x=[str(m) for m in range(1, 13)], 
        y=pivot_pnl.index, 
        colorscale=custom_colorscale, 
        zmid=0, 
        showscale=True,
        colorbar=dict(title="PnL ($)"),
        text=hover_text,
        hoverinfo="text",
        xgap=2, ygap=2
    ))
    fig_heat.update_layout(title="Execution Heatmap (Realized PnL by Month)", xaxis_title="Month", yaxis_title="Year", template="plotly_white", yaxis=dict(autorange="reversed", dtick=1))
    fig_heat.write_html(str(plot_dir / "dash_execution_heatmap.html"))

# ---------------------------------------------------------
# SMART ROUTER
# ---------------------------------------------------------
def generate_trading_plots(run_dir: Path, plot_dir: Path):
    trading_dir = run_dir / "trading"
    if not trading_dir.exists():
        print(f"[PLOT] No trading directory found at {trading_dir}. Run --stage trading first.")
        return

    # 1. Clean up old trading plots so leftover files (like heatmaps) don't persist between strategy switches
    old_plots = [
        "dash_trading_performance.html", "dash_kpis.html", "dash_rolling_risk.html", 
        "dash_volatility_scatter.html", "dash_execution_heatmap.html"
    ]
    for p in old_plots:
        if (plot_dir / p).exists():
            (plot_dir / p).unlink()

    # 2. Find the most recently modified JSON results file to determine the latest strategy run
    all_jsons = list(trading_dir.glob("*results_*.json"))
    if not all_jsons:
        print(f"[PLOT] Found /trading folder, but no valid JSON result files inside.")
        return

    # Sort by modification time (newest first) to ensure we always plot the strategy you JUST ran
    all_jsons.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest_json = all_jsons[0]

    if latest_json.name.startswith("event_"):
        # ROUTE: Prediction Market Strategy (Forecastex)
        strategy_name = latest_json.stem.replace("event_backtest_results_", "").replace("event_monte_carlo_results_", "")
        event_trades = list(trading_dir.glob(f"event_*_trades_{strategy_name}.csv"))
        equity_csv = trading_dir / f"event_equity_curve_{strategy_name}.csv"
        
        if event_trades and equity_csv.exists():
            print(f"[PLOT] Routing: Prediction Market Strategy ({strategy_name})")
            _plot_universal_performance(equity_csv, event_trades[0], latest_json, plot_dir, strategy_name)
            _plot_forecastex_exclusives(event_trades[0], plot_dir)
        else:
            print(f"[PLOT] Missing CSV files for Event Strategy: {strategy_name}")

    else:
        # ROUTE: Standard ETF Strategy (Krish/Will)
        strategy_name = latest_json.stem.replace("backtest_results_", "").replace("monte_carlo_results_", "")
        
        # Grab the matching trades/equity curves, ensuring we ignore any event contracts
        port_trades = [f for f in trading_dir.glob(f"*_trades_{strategy_name}.csv") if not f.name.startswith("event_")]
        equity_csvs = [f for f in trading_dir.glob(f"*equity_curve_{strategy_name}.csv") if not f.name.startswith("event_")]

        if port_trades and equity_csvs:
            print(f"[PLOT] Routing: Standard ETF Strategy ({strategy_name})")
            _plot_universal_performance(equity_csvs[0], port_trades[0], latest_json, plot_dir, strategy_name)
        else:
            print(f"[PLOT] Missing CSV files for Standard Strategy: {strategy_name}")

def generate_all_dashboard_plots(panel_name: str = "mrts_national"):
    print("\n" + "=" * 50)
    print("🎨 GENERATING DASHBOARD VISUALIZATIONS")
    print("=" * 50)

    run_dir, forecasts_df, metrics_df = load_latest_run_frames()

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    viz_cfg = config.get("visualization", {}) if isinstance(config, dict) else {}
    top_k = int(viz_cfg.get("top_k_forecast_models", 5))

    generate_forecast_plot(
        plot_dir=plot_dir,
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        panel_name=panel_name,
    )

    generate_top_k_forecast_plot(
        plot_dir=plot_dir,
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        panel_name=panel_name,
        top_k=top_k,
    )

    generate_top_k_mape_comparison_plot(
        plot_dir=plot_dir,
        metrics_df=metrics_df,
        panel_name=panel_name,
        top_k=top_k,
    )

    generate_macro_profiler(
        plot_dir=plot_dir,
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        panel_name=panel_name,
    )

    generate_trading_plots(run_dir, plot_dir)

    print("=" * 50)
    print(f"[DONE] All interactive dashboards saved to {plot_dir}")

if __name__ == "__main__":
    generate_all_dashboard_plots()