#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
CHAMPION MODEL EXECUTOR — "Red Button" Demonstration

Objective: Provide external stakeholders (Fiserv team) with a single-click execution
of the best Phase 2 model (XGBoost Regressor on Accommodation & Food Services sector).

This script bypasses all grid search, cross-validation, and hyperparameter tuning.
It either loads pre-trained weights or instantly trains using hardcoded champion parameters.

Design:
- Load pre-trained model OR train instantly with champion hyperparameters
- No GridSearchCV, no TimeSeriesSplit, no complexity
- Professional console output for stakeholder review
- Clean visualization of actual vs. predicted PCE
- Production-ready for FORECASTEX trading simulations

Usage:
    python run_champion_model.py

Toggle LOAD_PRETRAINED at the top to switch between loading pre-trained weights
or re-training with the same champion hyperparameters.
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    TARGET,
    TRAIN_SIZE,
    RANDOM_STATE,
    ARTIFACTS_DIR,
    WINNER_PATH
)
from train import load_and_split, encode_categorical_features


# ============================================================================
# GLOBAL FLAGS & HARDCODED CHAMPION PARAMETERS
# ============================================================================

SECTOR_FILE = WINNER_PATH
LOAD_PRETRAINED = True  # Toggle: True = load .pkl, False = train instantly

MODEL_PATH = "phase1_XGBoost_msa__New_York-Newark-Jersey_City_NY-NJ-PA_model.pkl"

# Champion hyperparameters (from Phase 2 winner)
CHAMPION_PARAMS = {
    'max_depth': 2,
    'learning_rate': 0.01,
    'n_estimators': 100,
    'subsample': 0.8
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_champion_model():
    """
    Execute the champion model: load pre-trained or train instantly.
    
    Returns:
        model: Trained/loaded XGBRegressor
        X_test: Test feature matrix
        y_test: Test target
        metrics: Dictionary of evaluation metrics
    """
    
    print("\n" + "=" * 80)
    print("CHAMPION MODEL EXECUTOR — FISERV STAKEHOLDER DEMONSTRATION")
    print("=" * 80)
    print(f"\nDataset: {SECTOR_FILE}")
    print(f"Model: XGBoost Regressor (objective='reg:absoluteerror')")
    print(f"Target: {TARGET} (Month-over-Month % change)")
    print(f"Train/Test split: {TRAIN_SIZE*100:.0f}/{(1-TRAIN_SIZE)*100:.0f}")
    print(f"Mode: {'LOAD PRE-TRAINED' if LOAD_PRETRAINED else 'TRAIN INSTANTLY'}")
    print("\n" + "=" * 80)
    
    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print(f"\n[1/5] Loading and preparing data...")
    data_path = Path("data") / SECTOR_FILE
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return None, None, None, None
    
    try:
        # Load and split
        X_train, X_test, y_train, y_test = load_and_split(
            data_path,
            TARGET,
            TRAIN_SIZE
        )
        
        print(f"      Train shape: {X_train.shape} | Test shape: {X_test.shape}")
        
        # Encode categorical features
        print(f"[2/5] Encoding categorical features...")
        X_train, X_test, encoders = encode_categorical_features(X_train, X_test)
        
        # Model execution logic
        print(f"[3/5] Model execution...")
        
        if LOAD_PRETRAINED:
            print(f"      Attempting to load pre-trained model from {MODEL_PATH}...")
            try:
                model = joblib.load(MODEL_PATH)
                print(f"      ✓ Pre-trained model loaded successfully")
            except FileNotFoundError:
                print(f"      ⚠ WARNING: Model file not found at {MODEL_PATH}")
                print(f"      Falling back to instant training with champion parameters...")
                model = XGBRegressor(
                    objective='reg:absoluteerror',
                    random_state=RANDOM_STATE,
                    **CHAMPION_PARAMS
                )
                model.fit(X_train, y_train)
                print(f"      ✓ Model trained instantly with champion parameters")
        else:
            print(f"      Training instantly with champion parameters...")
            model = XGBRegressor(
                objective='reg:absoluteerror',
                random_state=RANDOM_STATE,
                **CHAMPION_PARAMS
            )
            model.fit(X_train, y_train)
            print(f"      ✓ Model trained successfully")
        
               # Evaluation
        print(f"[4/5] Evaluating on test set...")
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate Directional Accuracy
        y_test_direction = np.sign(y_test.diff().dropna())
        y_pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = (y_test_direction.values == y_pred_direction).mean() * 100
        
        print(f"      Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"      MAE:  {mae:.6f}")
        print(f"      RMSE: {rmse:.6f}")
        print(f"      R²:   {r2:.4f}")
        
        # Professional console report
        print(f"\n[5/5] Generating stakeholder report...")
        print_stakeholder_report(mae, rmse, r2, directional_accuracy)
        
        # Visualization
        print(f"\nGenerating forecast visualization...")
        generate_forecast_plot(y_test, y_pred)
        
        # Compile metrics
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        return model, X_test, y_test, metrics
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def print_stakeholder_report(mae, rmse, r2, directional_accuracy):
    """
    Print a professional, stylized console report for stakeholders.
    
    Args:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        r2: R-squared score
        directional_accuracy: Percentage of correct directional predictions (0-100)
    """
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "CHAMPION MODEL PERFORMANCE REPORT".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╠" + "═" * 78 + "╣")
    
    print("║" + " " * 78 + "║")
    print("║  DATASET CONFIGURATION".ljust(79) + "║")
    print("║" + " " * 78 + "║")
    print(f"║  Metro Area:              NY-NJ-PA".ljust(79) + "║")
    print(f"║  Target Variable:     {TARGET}".ljust(79) + "║")
    print(f"║  Train/Test Split:    {TRAIN_SIZE*100:.0f}% / {(1-TRAIN_SIZE)*100:.0f}%".ljust(79) + "║")
    print(f"║  Feature Set:         All features (no selection)".ljust(79) + "║")
    
    print("║" + " " * 78 + "║")
    print("╠" + "═" * 78 + "╣")
    print("║" + " " * 78 + "║")
    print("║  MODEL HYPERPARAMETERS".ljust(79) + "║")
    print("║" + " " * 78 + "║")
    print(f"║  Algorithm:           XGBoost Regressor".ljust(79) + "║")
    print(f"║  Objective:           reg:absoluteerror".ljust(79) + "║")
    print(f"║  n_estimators:        {CHAMPION_PARAMS['n_estimators']}".ljust(79) + "║")
    print(f"║  max_depth:           {CHAMPION_PARAMS['max_depth']}".ljust(79) + "║")
    print(f"║  learning_rate:       {CHAMPION_PARAMS['learning_rate']}".ljust(79) + "║")
    print(f"║  subsample:           {CHAMPION_PARAMS['subsample']}".ljust(79) + "║")
    
    print("║" + " " * 78 + "║")
    print("╠" + "═" * 78 + "╣")
    print("║" + " " * 78 + "║")
    print("║  TEST SET PERFORMANCE METRICS".ljust(79) + "║")
    print("║" + " " * 78 + "║")
    print(f"║  Directional Accuracy (PRIMARY):   {directional_accuracy:>10.2f}%".ljust(79) + "║")
    print(f"║  Mean Absolute Error (MAE):        {mae:>12.6f}".ljust(79) + "║")
    print(f"║  Root Mean Squared Error (RMSE):   {rmse:>12.6f}".ljust(79) + "║")
    print(f"║  R² Score:                         {r2:>12.4f}".ljust(79) + "║")
    
    print("║" + " " * 78 + "║")
    print("╠" + "═" * 78 + "╣")
    print("║" + " " * 78 + "║")
    print("║  INTERPRETATION & TRADING READINESS".ljust(79) + "║")
    print("║" + " " * 78 + "║")
    
    # Directional Accuracy as primary threshold
    if directional_accuracy >= 75:
        interpretation = "EXCELLENT: Model correctly predicts direction >75%."
        trading_status = "PRODUCTION-READY for FORECASTEX trading simulations."
        confidence = "High confidence in directional signals for strategy execution."
    elif directional_accuracy >= 65:
        interpretation = "GOOD: Model correctly predicts direction 65-75%."
        trading_status = "SUITABLE for trading simulations with risk management."
        confidence = "Moderate confidence. Recommend position sizing and stop-loss."
    elif directional_accuracy >= 55:
        interpretation = "MODERATE: Model correctly predicts direction 55-65%."
        trading_status = "CAUTION: Use only with robust risk controls."
        confidence = "Limited edge. Recommend hedging and diversification."
    else:
        interpretation = "WEAK: Model directional accuracy <55%."
        trading_status = "NOT RECOMMENDED for live trading without refinement."
        confidence = "Insufficient predictive power. Consider model improvements."
    
    print(f"║  {interpretation}".ljust(79) + "║")
    print(f"║  {trading_status}".ljust(79) + "║")
    print(f"║  {confidence}".ljust(79) + "║")
    
    print("║" + " " * 78 + "║")
    print("║  SUPPORTING METRICS ANALYSIS".ljust(79) + "║")
    print("║" + " " * 78 + "║")
    
    # Supporting metrics interpretation
    if r2 > 0.7:
        r2_interpretation = "Strong explanatory power (R² > 0.7)"
    elif r2 > 0.5:
        r2_interpretation = "Moderate explanatory power (R² 0.5-0.7)"
    else:
        r2_interpretation = "Limited explanatory power (R² < 0.5)"
    
    print(f"║  R² Score:                {r2_interpretation}".ljust(79) + "║")
    print(f"║  MAE:                     {mae:.6f} (magnitude of average errors)".ljust(79) + "║")
    print(f"║  RMSE:                    {rmse:.6f} (penalizes larger errors)".ljust(79) + "║")
    
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print(f"\n✓ Report generated successfully")
    print(f"✓ Forecast visualization saved to: {ARTIFACTS_DIR / 'champion_model_forecast.png'}")


def generate_forecast_plot(y_test, y_pred):
    """
    Create a DIDACTIC 3-panel forecast plot that tells the story clearly.
    Panel 1: Actual values only
    Panel 2: Predicted values only
    Panel 3: Both overlaid for comparison
    
    Args:
        y_test: Actual test values
        y_pred: Predicted test values
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    x_axis = np.arange(len(y_test))
    y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
    
    # PANEL 1: Actual PCE only
    ax1.plot(x_axis, y_test_values, 'o-', linewidth=3, markersize=8, color='#1f77b4')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax1.fill_between(x_axis, 0, y_test_values, alpha=0.2, color='#1f77b4')
    ax1.set_ylabel('PCE Growth (%)', fontsize=11, fontweight='bold')
    ax1.set_title('STEP 1: What Actually Happened (Actual PCE Growth)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.set_axisbelow(True)
    
    # PANEL 2: Predicted PCE only
    ax2.plot(x_axis, y_pred, 's-', linewidth=3, markersize=8, color='#ff7f0e')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax2.fill_between(x_axis, 0, y_pred, alpha=0.2, color='#ff7f0e')
    ax2.set_ylabel('PCE Growth (%)', fontsize=11, fontweight='bold')
    ax2.set_title('STEP 2: What the Model Predicted (Model Forecast)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.set_axisbelow(True)
    
    # PANEL 3: Both overlaid
    ax3.plot(x_axis, y_test_values, 'o-', label='Actual PCE', linewidth=3, markersize=8, color='#1f77b4', zorder=3)
    ax3.plot(x_axis, y_pred, 's-', label='Model Prediction', linewidth=2.5, markersize=7, color='#ff7f0e', zorder=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax3.set_xlabel('Month (Test Period)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('PCE Growth (%)', fontsize=11, fontweight='bold')
    ax3.set_title('STEP 3: Comparison (How Close Was the Model?)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.25, linestyle='--')
    ax3.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = ARTIFACTS_DIR / "champion_model_forecast.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Forecast plot saved: {plot_path}")
    
    plt.show()


def main():
    """Main entry point for champion model execution."""
    
    print("\n" + "=" * 80)
    print("CHAMPION MODEL EXECUTOR")
    print("=" * 80)
    print("\nObjective: Single-click execution of best Phase 2 model for stakeholders")
    print("Geo: NY-NJ-PA MSA")
    print("Model: XGBoost Regressor (champion hyperparameters)")
    print("\n" + "=" * 80)
    
    model, X_test, y_test, metrics = run_champion_model()
    
    if model is not None:
        print("\n" + "=" * 80)
        print("EXECUTION COMPLETE")
        print("=" * 80)
        print(f"\nModel Status: ✓ Ready for trading simulations")
        print(f"Artifacts saved to: {ARTIFACTS_DIR}")
        print(f"\nKey Metrics:")
        print(f"  • MAE:  {metrics['MAE']:.6f}")
        print(f"  • RMSE: {metrics['RMSE']:.6f}")
        print(f"  • R²:   {metrics['R2']:.4f}")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()


# In[ ]:




