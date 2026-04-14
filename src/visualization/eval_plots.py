import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import config
from src.trading.strategy import load_latest_run_frames, select_best_model

def calculate_metrics(y_true, y_pred):
    """Calculate MAPE and Directional Accuracy."""
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional Accuracy
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100 
    
    return mape, dir_acc

def plot_best_vs_baseline(panel_name: str = "pce_national_mom", baseline_model_name: str = "naive"):
    print(f"\n[PLOT] Generating Evaluation Plot for {panel_name}")
    
    # 1. Load the latest experiment data
    run_dir, forecasts_df, metrics_df = load_latest_run_frames()
    
    # 2. Extract the Best Model's forecasts
    best_df = select_best_model(forecasts_df, metrics_df, panel_name=panel_name)
    best_model_full_name = best_df.iloc[0]['model_name']
    
    # Extract base name for a cleaner legend (e.g., 'xgboost' from 'xgboost_colsample...')
    display_model_name = best_model_full_name.split('_')[0].upper()
    
    best_df = best_df.set_index('date').sort_index()
    
    # 3. Extract the Baseline (Naive) forecasts
    baseline_df = forecasts_df[
        (forecasts_df['panel_name'] == panel_name) & 
        (forecasts_df['model_name'] == baseline_model_name)
    ].set_index('date').sort_index()
    
    if baseline_df.empty:
        raise ValueError(f"Baseline model '{baseline_model_name}' not found in forecasts.csv.")

    # 4. Align the data on dates
    plot_df = pd.DataFrame({
        'Actual': best_df['y_true'],
        'Best Model': best_df['y_pred'],
        'Baseline (Naive)': baseline_df['y_pred']
    }).dropna()

    # 5. Calculate Metrics
    best_mape, best_dir = calculate_metrics(plot_df['Actual'], plot_df['Best Model'])
    base_mape, base_dir = calculate_metrics(plot_df['Actual'], plot_df['Baseline (Naive)'])

    # 6. Plotting
    # Create figure and axis explicitly to better control spacing
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data lines
    ax.plot(plot_df.index, plot_df['Actual'], label='Actual (Ground Truth)', color='black', linewidth=2)
    ax.plot(plot_df.index, plot_df['Best Model'], label=f'Best ({display_model_name})', color='blue', linestyle='--')
    ax.plot(plot_df.index, plot_df['Baseline (Naive)'], label='Baseline (Naive)', color='red', linestyle=':')
    
    ax.set_title(f"Out-of-Sample Predictions: Best Model vs Naive Baseline [{panel_name}]", fontsize=14, pad=15)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Place legend OUTSIDE the axes on the top right
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Add Metrics Text Box OUTSIDE the axes, below the legend
    metrics_text = (
        f"--- METRICS ---\n\n"
        f"BEST MODEL\n"
        f"MAPE: {best_mape:.2f}%\n"
        f"Dir. Acc: {best_dir:.1f}%\n\n"
        f"BASELINE\n"
        f"MAPE: {base_mape:.2f}%\n"
        f"Dir. Acc: {base_dir:.1f}%"
    )
    # Using figure coordinates (x=0.92 puts it to the right of the plot area)
    plt.figtext(0.92, 0.65, metrics_text, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Add the massive hyperparameter string as a tiny footnote
    plt.figtext(0.1, -0.02, f"Model Architecture: {best_model_full_name}", fontsize=8, color='gray')

    # Save the plot (bbox_inches='tight' is critical here so it doesn't crop the outside legend)
    out_path = run_dir / f"eval_plot_{panel_name}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[PLOT] Saved formatted plot to {out_path}")

if __name__ == "__main__":
    plot_best_vs_baseline()