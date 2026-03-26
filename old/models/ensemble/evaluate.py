#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 
# FILE: src/ensemble/evaluate.py
# 
"""
Evaluation Engine for Phase 1 Ensemble

Computes metrics, compares models, and generates visualizations.

Key functions:
- Individual metric functions (rmse, mae, mape, r_squared, directional_accuracy)
- evaluate_model(): Compute all metrics for one model
- compare_models(): Compare all models and generate report
- plot_predictions(): Visualize predictions vs. actuals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import ARTIFACTS_DIR, METRICS, VERBOSE, SAVE_PLOTS


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.

    Lower is better. Penalizes large errors more heavily than MAE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.

    Lower is better. Robust to outliers compared to RMSE.
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.

    Lower is better. Scale-independent, useful for comparing across datasets.
    Expressed as percentage (0-100).
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r_squared(y_true, y_pred):
    """
    Coefficient of Determination (R²).

    Higher is better. Represents fraction of variance in y explained by model.
    Range: (-∞, 1], where 1 is perfect prediction.
    """
    return r2_score(y_true, y_pred)


def directional_accuracy(y_true, y_pred):
    """
    Directional Accuracy.

    Percentage of periods where predicted direction of change matches actual.

    Why this matters for trading:
    - Getting the direction right (up/down) is often more important than
      exact magnitude for trading strategy profitability
    - A model that predicts direction correctly 70% of the time can be
      profitable even if magnitude predictions are imperfect
    - This metric directly measures trading signal quality

    Returns:
    --------
    float
        Percentage (0-100) of periods with correct direction prediction
    """
    # Calculate actual direction of change (1 = up, 0 = down)
    actual_direction = np.diff(y_true) > 0

    # Calculate predicted direction of change
    pred_direction = np.diff(y_pred) > 0

    # Directional accuracy
    if len(actual_direction) == 0:
        return 0.0

    accuracy = np.mean(actual_direction == pred_direction) * 100
    return accuracy


def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate a single model on test set.

    Parameters:
    -----------
    model : sklearn Pipeline
        Fitted model
    model_name : str
        Name of model (for logging)
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target

    Returns:
    --------
    dict
        {"metric_name": metric_value} for all metrics in config.METRICS
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute all metrics
    metrics_dict = {
        "RMSE": rmse(y_test, y_pred),
        "MAE": mae(y_test, y_pred),
        "MAPE": mape(y_test, y_pred),
        "R2": r_squared(y_test, y_pred),
        "Directional_Accuracy": directional_accuracy(y_test, y_pred),
    }

    # Print formatted evaluation block
    if VERBOSE:
        print(f"\n{'=' * 60}")
        print(f" MODEL EVALUATION: {model_name}")
        print(f"{'=' * 60}")
        print(f" RMSE               : {metrics_dict['RMSE']:.6f}")
        print(f" MAE                : {metrics_dict['MAE']:.6f}")
        print(f" MAPE               : {metrics_dict['MAPE']:.2f}%")
        print(f" R²                 : {metrics_dict['R2']:.6f}")
        print(f" Directional Acc.   : {metrics_dict['Directional_Accuracy']:.2f}%")
        print(f"{'-' * 60}\n")

    return metrics_dict


def compare_models(models_dict, X_test, y_test):
    """
    Compare all models and generate ranking report.

    Parameters:
    -----------
    models_dict : dict
        {"model_name": fitted_pipeline}
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target

    Returns:
    --------
    pd.DataFrame
        Comparison table ranked by RMSE (ascending)

    Side effects:
    - Prints formatted comparison table
    - Saves comparison CSV to ARTIFACTS_DIR
    - Saves RMSE comparison plot to ARTIFACTS_DIR
    - Saves prediction plots for each model to ARTIFACTS_DIR
    """

    if VERBOSE:
        print(f"\n{'=' * 80}")
        print(f" PHASE 1 MODEL COMPARISON — RANKED BY RMSE (lower is better)")
        print(f" NOTE: Trained on full feature set. No COVID regime handling.")
        # OLD:
        # print(f" Test period: {X_test.iloc[0]['Period']} to {X_test.iloc[-1]['Period']}")
        
        # NEW:
        print(f" Test period: {X_test.index[0]} to {X_test.index[-1]}")
        print(f" Temporal split: 80/20 (COVID mostly in training)")
        print(f"{'=' * 80}\n")

    # Evaluate each model
    results = []
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, model_name, X_test, y_test)
        metrics["Model"] = model_name
        results.append(metrics)

        # Plot predictions for this model
        if SAVE_PLOTS:
            plot_predictions(model, model_name, X_test, y_test, ARTIFACTS_DIR)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Rank by RMSE
    results_df = results_df.sort_values("RMSE").reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    # Reorder columns for readability
    cols = ["Rank", "Model", "RMSE", "MAE", "MAPE", "R2", "Directional_Accuracy"]
    results_df = results_df[cols]

    # Print comparison table
    if VERBOSE:
        print(results_df.to_string(index=False))
        print(f"\n{'=' * 80}\n")

    # Save comparison CSV
    comparison_path = ARTIFACTS_DIR / "phase1_model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    if VERBOSE:
        print(f"✓ Comparison table saved: {comparison_path}\n")

    # Save RMSE comparison plot
    if SAVE_PLOTS:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results_df["Model"], results_df["RMSE"], color="steelblue", alpha=0.8)
        ax.set_ylabel("RMSE (lower is better)", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_title("Phase 1: RMSE Comparison Across Models", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        rmse_plot_path = ARTIFACTS_DIR / "phase1_rmse_comparison.png"
        plt.savefig(rmse_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        if VERBOSE:
            print(f"✓ RMSE comparison plot saved: {rmse_plot_path}\n")

    return results_df


def plot_predictions(model, model_name, X_test, y_test, artifacts_dir):
    """
    Plot predicted vs. actual PCE over test period.

    Parameters:
    -----------
    model : sklearn Pipeline
        Fitted model
    model_name : str
        Name of model
    X_test : pd.DataFrame
        Test features (must contain Period column)
    y_test : pd.Series
        Test target
    artifacts_dir : Path
        Directory to save plot

    Returns:
    --------
    None

    Side effects:
    - Saves plot to {artifacts_dir}/phase1_{model_name}_predictions.png
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual and predicted
    periods = X_test["Period"].values if "Period" in X_test.columns else range(len(y_test))
    ax.plot(periods, y_test.values, label="Actual", linewidth=2, color="black", alpha=0.7)
    ax.plot(periods, y_pred, label="Predicted", linewidth=2, color="steelblue", alpha=0.7)

    ax.set_xlabel("Period", fontsize=12)
    ax.set_ylabel("PCE", fontsize=12)
    ax.set_title(f"Phase 1: {model_name} — Predicted vs. Actual PCE", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Rotate x-axis labels if too many
    if len(periods) > 20:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save plot
    plot_path = artifacts_dir / f"phase1_{model_name}_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    if VERBOSE:
        print(f"  ✓ Prediction plot saved: {plot_path}")


# In[ ]:




