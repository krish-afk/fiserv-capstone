import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import config
from src.trading.strategy import load_latest_run_frames, select_best_model

def calculate_metrics(y_true, y_pred):
    """Calculate MAPE and Directional Accuracy."""
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100 
    return mape, dir_acc

# ---------------------------------------------------------
# CORE PLOT 1: FORECAST OVERLAY
# ---------------------------------------------------------
def generate_forecast_plot(run_dir: Path, forecasts_df: pd.DataFrame, metrics_df: pd.DataFrame, panel_name: str, baseline_model_name: str = "naive"):
    print(f"[PLOT] Generating Forecast Overlay for {panel_name}...")
    
    best_df = select_best_model(forecasts_df, metrics_df, panel_name=panel_name)
    best_model_full_name = best_df.iloc[0]['model_name']
    display_model_name = best_model_full_name.split('_')[0].upper()
    best_df = best_df.set_index('date').sort_index()
    
    baseline_df = forecasts_df[
        (forecasts_df['panel_name'] == panel_name) & 
        (forecasts_df['model_name'] == baseline_model_name)
    ].set_index('date').sort_index()

    plot_df = pd.DataFrame({'Actual': best_df['y_true'], 'Best Model': best_df['y_pred']})
    if not baseline_df.empty:
        plot_df['Baseline'] = baseline_df['y_pred']
    plot_df = plot_df.dropna()

    best_mape, best_dir = calculate_metrics(plot_df['Actual'], plot_df['Best Model'])
    base_mape, base_dir = calculate_metrics(plot_df['Actual'], plot_df['Baseline']) if not baseline_df.empty else (0, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual (Ground Truth)', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Best Model'], mode='lines', name=f'Best ({display_model_name})', line=dict(color='blue', width=2, dash='dash')))
    
    if not baseline_df.empty:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Baseline'], mode='lines', name='Baseline (Naive)', line=dict(color='red', width=2, dash='dot')))

    fig.update_layout(
        title=f"Out-of-Sample Predictions: {panel_name}<br><sup>Best Model (MAPE: {best_mape:.2f}%) vs Baseline (MAPE: {base_mape:.2f}%)</sup>",
        xaxis_title="Date", yaxis_title="Value", template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html(str(run_dir / f"dash_forecast_{panel_name}.html"))

# ---------------------------------------------------------
# ELECTIVE PLOT 1: MACRO ENVIRONMENT PROFILER
# ---------------------------------------------------------
def generate_macro_profiler(run_dir: Path, forecasts_df: pd.DataFrame, metrics_df: pd.DataFrame, panel_name: str):
    print("[PLOT] Generating Macro Environment Profiler...")
    try:
        best_df = select_best_model(forecasts_df, metrics_df, panel_name=panel_name).set_index('date').sort_index()
        best_df['Abs_Error_Pct'] = np.abs((best_df['y_true'] - best_df['y_pred']) / best_df['y_true']) * 100
        
        # Load master macro dataset
        master_path = Path(config.get("paths", {}).get("processed_data", "data/processed/")) / "master.csv"
        if not master_path.exists():
            print(f"[PLOT] Skipping Macro Profiler: Could not find {master_path}")
            return
            
        macro_df = pd.read_csv(master_path, parse_dates=['date']).set_index('date')
        
        # Normalize macro variables to YoY % Change for apples-to-apples scaling
        macro_pct = macro_df.pct_change(12) * 100 
        macro_pct = macro_pct.replace([np.inf, -np.inf], np.nan).dropna(axis=1, thresh=len(macro_pct)*0.8)
        
        merged = best_df[['Abs_Error_Pct']].join(macro_pct, how='inner').dropna()
        macro_cols = [c for c in merged.columns if c != 'Abs_Error_Pct'][:8] # Limit to top 8 clean columns

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=merged.index, y=merged['Abs_Error_Pct'], name='Model Error (%)', marker_color='rgba(0, 0, 255, 0.3)'), secondary_y=False)
        
        for i, col in enumerate(macro_cols):
            fig.add_trace(go.Scatter(x=merged.index, y=merged[col], mode='lines', name=f"{col} (% YoY)", line=dict(width=2), visible=(i==0)), secondary_y=True)

        # Dropdown to toggle macro indicators
        buttons = []
        for i, col in enumerate(macro_cols):
            vis = [True] + [j == i for j in range(len(macro_cols))]
            buttons.append(dict(label=col, method="update", args=[{"visible": vis}, {"title": f"Model Error vs {col} (YoY %)"}]))

        fig.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, x=1.1, y=1.15)],
            title=f"Macro Environment Profiler: Error vs {macro_cols[0]} (YoY %)",
            template="plotly_white", hovermode="x unified"
        )
        fig.update_yaxes(title_text="Model Absolute Error (%)", secondary_y=False)
        fig.update_yaxes(title_text="Macro Indicator YoY (%)", secondary_y=True)

        fig.write_html(str(run_dir / "dash_macro_profiler.html"))
    except Exception as e:
        print(f"[PLOT] Error generating Macro Profiler: {e}")

# ---------------------------------------------------------
# CORE TRADING PLOTS & ELECTIVES
# ---------------------------------------------------------
def generate_trading_plots(run_dir: Path):
    trading_csv = Path("experiments/mrts_trading_results.csv")
    if not trading_csv.exists():
        print("[PLOT] No Event Trading results found. Skipping trading plots.")
        return

    print("[PLOT] Generating Trading Performance Visuals...")
    df = pd.read_csv(trading_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    initial_capital = config.get("trading", {}).get("portfolio", {}).get("initial_cash", 100000.0)
    df['Cumulative_Real'] = df['Portfolio_Value_Real']
    df['Peak'] = df['Cumulative_Real'].cummax()
    df['Drawdown'] = (df['Cumulative_Real'] - df['Peak']) / df['Peak'] * 100

    # ELECTIVE 2: Regime & Volatility Scatter
    df['Surprise_Magnitude'] = np.abs(df['Actual'] - df['Strike']) # Volatility Proxy
    fig_scatter = go.Figure()
    
    wins = df[df['Outcome'] == 'WIN']
    losses = df[df['Outcome'] == 'LOSS']
    
    fig_scatter.add_trace(go.Scatter(x=wins['Surprise_Magnitude'], y=wins['Edge'], mode='markers', name='WIN', marker=dict(color='rgba(0, 150, 0, 0.7)', size=12, line=dict(width=1, color='darkgreen')), text=wins.index.strftime('%b %Y')))
    fig_scatter.add_trace(go.Scatter(x=losses['Surprise_Magnitude'], y=losses['Edge'], mode='markers', name='LOSS', marker=dict(color='rgba(255, 0, 0, 0.7)', size=12, symbol='x', line=dict(width=1, color='darkred')), text=losses.index.strftime('%b %Y')))
    
    fig_scatter.update_layout(title="Regime Volatility: Trade Edge vs Economic Surprise", xaxis_title="Economic Surprise |Actual - Consensus| (Volatility Proxy)", yaxis_title="Calculated Probability Edge", template="plotly_white")
    fig_scatter.write_html(str(run_dir / "dash_volatility_scatter.html"))

    # ELECTIVE 3: Rolling Risk Suite (Sharpe/Sortino/Calmar)
    # Resample to monthly to fill non-trading gaps for continuous rolling math
    df_monthly = df.resample('ME').last()
    df_monthly['Cumulative_Real'] = df_monthly['Cumulative_Real'].ffill()
    df_monthly['Ret'] = df_monthly['Cumulative_Real'].pct_change().fillna(0)
    
    ann_ret = df_monthly['Ret'].rolling(12).mean() * 12
    ann_vol = df_monthly['Ret'].rolling(12).std() * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-6)
    
    downside = df_monthly['Ret'].copy()
    downside[downside > 0] = 0
    down_vol = downside.rolling(12).std() * np.sqrt(12)
    sortino = ann_ret / (down_vol + 1e-6)
    
    max_dd = df_monthly['Drawdown'].rolling(12).min() / 100
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
    fig_risk.write_html(str(run_dir / "dash_rolling_risk.html"))

    # ELECTIVE 4: Sparse-Execution Monthly Heatmap
    df_heat = df.copy()
    df_heat['Year'] = df_heat.index.year
    df_heat['Month'] = df_heat.index.month
    df_heat['Score'] = df_heat['Outcome'].map({'WIN': 1, 'LOSS': -1}).fillna(0)
    
    pivot = df_heat.pivot_table(index='Year', columns='Month', values='Score', aggfunc='sum')
    for m in range(1, 13):
        if m not in pivot.columns: pivot[m] = np.nan
    pivot = pivot.reindex(range(pivot.index.min(), pivot.index.max() + 1)).fillna(0)
    
    # Custom 3-state colorscale: Red (-1), Gray (0), Green (1)
    colorscale = [[0.0, '#d62728'], [0.5, '#f0f0f0'], [1.0, '#2ca02c']]
    
    hover_text = [[f"{year}-{month:02d}<br>{'WIN' if pivot.loc[year, month] > 0 else 'LOSS' if pivot.loc[year, month] < 0 else 'No Trade'}" for month in range(1, 13)] for year in pivot.index]
    
    fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=[str(m) for m in range(1, 13)], y=pivot.index, colorscale=colorscale, zmin=-1, zmax=1, showscale=False, text=hover_text, hoverinfo="text", xgap=2, ygap=2))
    fig_heat.update_layout(title="Sparse-Execution Heatmap (Green=Win, Red=Loss, Gray=Cash)", xaxis_title="Month", yaxis_title="Year", template="plotly_white", yaxis=dict(autorange="reversed"))
    fig_heat.write_html(str(run_dir / "dash_execution_heatmap.html"))

def generate_all_dashboard_plots(panel_name: str = "mrts_national"):
    """Master function to trigger all plots."""
    print("\n" + "="*50)
    print("🎨 GENERATING DASHBOARD VISUALIZATIONS")
    print("="*50)
    
    run_dir, forecasts_df, metrics_df = load_latest_run_frames()
    
    # Generate Core Forecast Plot
    generate_forecast_plot(run_dir, forecasts_df, metrics_df, panel_name)
    
    # Generate Elective 1 (Macro Profiler)
    generate_macro_profiler(run_dir, forecasts_df, metrics_df, panel_name)
    
    # Generate Trading Plots (Electives 2, 3, 4)
    generate_trading_plots(run_dir)
    
    print("="*50)
    print("[DONE] All interactive dashboards saved to experiment directory.")

if __name__ == "__main__":
    generate_all_dashboard_plots()