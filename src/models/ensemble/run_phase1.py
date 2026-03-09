#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 
# FILE: src/ensemble/run_phase1.py
# 
"""
Phase 1 Ensemble Modeling — Main Executable

PURPOSE:
--------
Phase 1 establishes baseline ensemble models for PCE prediction using the
full feature set (no feature selection). This phase uses a NAIVE temporal
split (90/10) without special COVID regime handling.

KEY DESIGN DECISION:
--------------------
The COVID period (2020-2021) is intentionally placed mostly in the training
set. This allows models to learn pandemic dynamics as part of the general
data distribution. Phase 3 will explicitly analyze COVID as a structural
break hypothesis and test model robustness across regimes.

WORKFLOW:
---------
1. train.run_training()
   → Loads data, performs temporal split, trains 4 ensemble models with
     GridSearchCV + TimeSeriesSplit, saves artifacts

2. evaluate.compare_models()
   → Evaluates all models on test set, computes 5 metrics, generates
     comparison table and visualizations

3. Print summary and next steps

OUTPUT:
-------
- Trained models saved to results/phase1/
- Comparison CSV: phase1_model_comparison.csv
- Plots: phase1_rmse_comparison.png, phase1_{model}_predictions.png
- Hyperparameters: phase1_{model}_params.json

NEXT STEPS (Phase 2):
---------------------
- Feature importance analysis (which features drive PCE predictions?)
- Correlation analysis between FSBI and PCE
- Lag analysis (what lag of FSBI is most predictive?)
"""

from train import run_training
from evaluate import compare_models
from config import ARTIFACTS_DIR, VERBOSE


def main():
    """
    Main orchestration function for Phase 1.
    """

    if VERBOSE:
        print("\n" + "=" * 80)
        print("PHASE 1: ENSEMBLE MODELING FOR PCE PREDICTION")
        print("=" * 80)
        print("\nDesign: Naive temporal split (80/20), full feature set, no COVID regime handling")
        print("Models: RandomForest, GradientBoost, XGBoost, LightGBM")
        print("CV: TimeSeriesSplit (5 folds, expanding window)")
        print("=" * 80)

    # Step 1: Train all models
    trained_models, X_test, y_test = run_training()

    # Step 2: Evaluate and compare
    comparison_df = compare_models(trained_models, X_test, y_test)

    # Step 3: Summary
    if VERBOSE:
        print("=" * 80)
        print("PHASE 1 COMPLETE")
        print("=" * 80)
        print(f"\n✓ All artifacts saved to: {ARTIFACTS_DIR}")
        print(f"\nKey outputs:")
        print(f"  - phase1_model_comparison.csv (ranked by RMSE)")
        print(f"  - phase1_rmse_comparison.png (bar chart)")
        print(f"  - phase1_{{model}}_predictions.png (4 time series plots)")
        print(f"  - phase1_{{model}}_model.pkl (fitted models)")
        print(f"  - phase1_{{model}}_params.json (best hyperparameters)")

        best_model = comparison_df.iloc[0]["Model"]
        best_rmse = comparison_df.iloc[0]["RMSE"]
        print(f"\n✓ Best model: {best_model} (RMSE: {best_rmse:.6f})")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


# In[ ]:




