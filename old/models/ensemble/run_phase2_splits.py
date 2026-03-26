#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
PHASE 2: SPLIT TOURNAMENT ORCHESTRATOR

Objective: Evaluate the predictive power of different feature subsets using the winning Phase 1 model (XGBoost).
This phase tests whether FSBI data alone, FSBI + macro, or all features perform best.

Design:
- Iterates over all CSV files in SPLITS_DIR
- For each split, trains XGBoost with Phase 1 hyperparameters
- Compares performance across splits to quantify FSBI's contribution
- Manages RAM carefully (16GB limit) via explicit garbage collection

Output:
- phase2_split_comparison.csv: Ranked results by MAE
- phase2_split_performance.png: Bar chart comparing MAE and Directional Accuracy
- Individual model artifacts and prediction plots per split
"""

import gc
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import (
    TARGET,
    TRAIN_SIZE,
    RANDOM_STATE,
    HYPERPARAMETER_GRIDS,
    ARTIFACTS_DIR,
    SPLITS_DIR,
)
from base_models import get_models
from train import load_and_split, encode_categorical_features, train_with_cv, save_artifacts
from evaluate import evaluate_model, plot_predictions


def run_phase2_tournament():
    """
    Execute the Phase 2 Split Tournament.
    
    For each CSV in SPLITS_DIR:
    1. Load and split data
    2. Encode categorical features
    3. Train XGBoost with Phase 1 hyperparameters
    4. Evaluate on test set
    5. Save artifacts and plots
    6. Collect garbage to manage RAM
    
    Returns:
        results_df: DataFrame of all split evaluations, ranked by MAE
    """
    
    print("\n" + "=" * 80)
    print("PHASE 2: SPLIT TOURNAMENT — FSBI vs. MACRO vs. ALL FEATURES")
    print("=" * 80)
    print(f"\nSplits directory: {SPLITS_DIR}")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    print(f"Target variable: {TARGET} (Month-over-Month % change)")
    print(f"Train/Test split: {TRAIN_SIZE*100:.0f}/{(1-TRAIN_SIZE)*100:.0f}")
    print("\n" + "=" * 80 + "\n")
    
    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get untrained XGBoost model and hyperparameter grid
    models = get_models()
    xgb_model = models["XGBoost"]
    xgb_param_grid = HYPERPARAMETER_GRIDS["XGBoost"]
    
    # Collect all CSV files in splits directory
    split_files = sorted(SPLITS_DIR.glob("*.csv"))
    
    if not split_files:
        print(f"ERROR: No CSV files found in {SPLITS_DIR}")
        return None
    
    print(f"Found {len(split_files)} split(s) to evaluate:\n")
    for f in split_files:
        print(f"  • {f.name}")
    print("\n" + "=" * 80 + "\n")
    
    results = []
    
    # Tournament loop
    for idx, csv_file in enumerate(split_files, 1):
        split_name = csv_file.stem  # Extract filename without extension
        
        # Print split header
        print("\n" + "=" * 80)
        print(f"SPLIT {idx}/{len(split_files)}: {split_name.upper()}")
        print("=" * 80)
        
        try:
            # Load and split data
            print(f"\n[1/6] Loading data from {csv_file.name}...")
            X_train, X_test, y_train, y_test = load_and_split(
                csv_file,
                TARGET,
                TRAIN_SIZE
            )
            print(f"      Train shape: {X_train.shape} | Test shape: {X_test.shape}")
            
            # Record feature count
            feature_count = X_train.shape[1]
            print(f"[2/6] Feature count: {feature_count}")
            
            # Encode categorical features
            print(f"[3/6] Encoding categorical features...")
            X_train, X_test, encoders = encode_categorical_features(X_train, X_test)
            
            # Train with cross-validation
            print(f"[4/6] Training XGBoost with GridSearchCV (TimeSeriesSplit)...")
            fitted_model, best_params = train_with_cv(
                xgb_model,
                "XGBoost",
                X_train,
                y_train,
                xgb_param_grid
            )
            
            # Save artifacts with split name prefix
            print(f"[5/6] Saving model artifacts...")
            save_artifacts(
                fitted_model,
                f"XGBoost_{split_name}",
                best_params,
                ARTIFACTS_DIR
            )
            
            # Evaluate on test set
            print(f"[6/6] Evaluating on test set...")
            metrics = evaluate_model(fitted_model, "XGBoost", X_test, y_test)
            
            # Inject split metadata
            metrics["Split_Name"] = split_name
            metrics["Feature_Count"] = feature_count
            
            # Append to results
            results.append(metrics)
            
            # Generate prediction plot
            print(f"\nGenerating prediction plot...")
            plot_predictions(
                fitted_model,
                f"XGBoost_{split_name}",
                X_test,
                y_test,
                ARTIFACTS_DIR
            )
            
            print(f"\n✓ Split '{split_name}' completed successfully")
            
        except Exception as e:
            print(f"\n✗ ERROR processing split '{split_name}': {str(e)}")
            results.append({
                "Split_Name": split_name,
                "Feature_Count": 0,
                "RMSE": np.nan,
                "MAE": np.nan,
                "MAPE": np.nan,
                "R2": np.nan,
                "Directional_Accuracy": np.nan,
                "Error": str(e)
            })
        
        finally:
            # CRITICAL: Explicit garbage collection to manage RAM
            print(f"\nCleaning up memory...")
            
            # Safely delete variables only if they were successfully created
            if 'X_train' in locals(): del X_train
            if 'X_test' in locals(): del X_test
            if 'y_train' in locals(): del y_train
            if 'y_test' in locals(): del y_test
            if 'fitted_model' in locals(): del fitted_model
            
            gc.collect()
            print(f"Memory freed. Ready for next split.\n")
    
    # Convert results to DataFrame
    print("\n" + "=" * 80)
    print("COMPILING RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    # Sort by MAE (ascending) and add rank
    results_df = results_df.sort_values("MAE", ascending=True).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    
    # Print ranked table
    print("\n" + "=" * 80)
    print("PHASE 2 SPLIT TOURNAMENT — RANKED BY MAE (lower is better)")
    print("=" * 80)
    print(f"\nTarget: {TARGET} (Month-over-Month % change)")
    print(f"Model: XGBoost (Phase 1 winner)")
    print(f"Train/Test: {TRAIN_SIZE*100:.0f}/{(1-TRAIN_SIZE)*100:.0f} temporal split")
    print(f"Evaluation metric: Mean Absolute Error (MAE)")
    print("\n" + "-" * 80 + "\n")
    
    # Display table
    display_df = results_df[["Rank", "Split_Name", "Feature_Count", "MAE", "RMSE", "R2", "Directional_Accuracy"]].copy()
    display_df["MAE"] = display_df["MAE"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "ERROR")
    display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "ERROR")
    display_df["R2"] = display_df["R2"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "ERROR")
    display_df["Directional_Accuracy"] = display_df["Directional_Accuracy"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "ERROR")
    
    print(display_df.to_string(index=False))
    print("\n" + "=" * 80 + "\n")
    
    # Save comparison CSV
    csv_output_path = ARTIFACTS_DIR / "phase2_split_comparison.csv"
    results_df.to_csv(csv_output_path, index=False)
    print(f"✓ Comparison table saved: {csv_output_path}\n")
    
    # Generate performance comparison chart
    print("Generating performance comparison chart...")
    generate_performance_chart(results_df)
    
    return results_df


def generate_performance_chart(results_df):
    """
    Generate a grouped bar chart comparing MAE and Directional Accuracy across splits.
    
    Args:
        results_df: DataFrame with split results
    """
    
    # Filter out error rows
    valid_df = results_df[results_df["MAE"].notna()].copy()
    
    if len(valid_df) == 0:
        print("No valid results to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MAE comparison
    splits = valid_df["Split_Name"]
    mae_values = valid_df["MAE"]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(splits)))
    ax1.bar(splits, mae_values, color=colors, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Mean Absolute Error (MAE)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Feature Split", fontsize=11, fontweight="bold")
    ax1.set_title("MAE Comparison Across Splits\n(Lower is Better)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for i, (split, mae) in enumerate(zip(splits, mae_values)):
        ax1.text(i, mae + mae_values.max() * 0.02, f"{mae:.6f}", ha="center", fontsize=9)
    
    # Plot 2: Directional Accuracy comparison
    da_values = valid_df["Directional_Accuracy"]
    
    ax2.bar(splits, da_values, color=colors, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Directional Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Feature Split", fontsize=11, fontweight="bold")
    ax2.set_title("Directional Accuracy Comparison Across Splits\n(Higher is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, 100])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for i, (split, da) in enumerate(zip(splits, da_values)):
        ax2.text(i, da + 2, f"{da:.1f}%", ha="center", fontsize=9)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = ARTIFACTS_DIR / "phase2_split_performance.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    print(f"✓ Performance chart saved: {chart_path}\n")
    plt.close()


def main():
    """Main entry point for Phase 2 Split Tournament."""
    
    print("\n" + "=" * 80)
    print("PHASE 2: SPLIT TOURNAMENT ORCHESTRATOR")
    print("=" * 80)
    print("\nObjective: Evaluate FSBI's predictive power vs. macro indicators vs. all features")
    print("Methodology: Train XGBoost on each feature subset, compare performance")
    print("RAM Management: Explicit garbage collection between splits (16GB limit)")
    print("\n" + "=" * 80)
    
    results_df = run_phase2_tournament()
    
    if results_df is not None:
        print("\n" + "=" * 80)
        print("PHASE 2 COMPLETE")
        print("=" * 80)
        print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")
        print("\nNext Steps (Phase 3):")
        print("  1. Analyze feature importance from winning split")
        print("  2. Investigate COVID-19 regime effects on predictions")
        print("  3. Develop trading strategy based on best-performing split")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()


# In[ ]:




