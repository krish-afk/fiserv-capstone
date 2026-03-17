#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 
# FILE: src/ensemble/config.py
# 
"""
Phase 1 Ensemble Modeling Configuration

This module centralizes all hyperparameters, paths, and configuration constants
for the Phase 1 ensemble modeling pipeline. No magic numbers elsewhere in the
codebase — all tunable parameters live here.

Key Design Decision:
- TRAIN_SIZE = 0.80 locks the COVID period (2020-2021) mostly into training,
  allowing the model to learn pandemic dynamics. The test set (2022 onwards)
  represents post-COVID normal regime. This is intentional and documented.
  Phase 3 will explicitly analyze COVID as a structural break hypothesis.
"""

from pathlib import Path

# 
# DATA CONFIGURATION
# 

TARGET = "PersonalConsumptionExpenditures_normalized"
DATA_PATH = Path("/home/leo/Documents/Grad School/QCF/00_full_dataset.csv") #Change your file path here.
WINNER_PATH = Path("/home/leo/Documents/Grad School/QCF/msa__New_York-Newark-Jersey_City_NY-NJ-PA.csv") #Path to tournament-winning split
ARTIFACTS_DIR = Path("results/phase1")

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 
# TRAIN/TEST SPLIT CONFIGURATION
# 

TRAIN_SIZE = 0.90
"""
Temporal split ratio (80% train, 20% test).

Rationale:
- Dataset spans 2019-2026 (7 years)
- 80% split ≈ 5.6 years training, 1.4 years testing
- COVID period (2020-2021) is mostly in training set
- Test set is primarily post-COVID (2022-2026)
- This is a NAIVE temporal split — no special COVID handling
- Phase 3 will analyze COVID as an explicit structural break
"""

RANDOM_STATE = 42
"""
Global random seed for reproducibility across all models and CV splits.
"""

# 
# CROSS-VALIDATION CONFIGURATION
# 

CV_N_SPLITS = 5
"""
Number of folds for TimeSeriesSplit.

Rationale:
- TimeSeriesSplit respects temporal order (no data leakage)
- 5 folds provides reasonable balance between stability and computation time
- Each fold uses expanding window: train on all prior data, test on next fold
"""

# 
# HYPERPARAMETER GRIDS
# 

HYPERPARAMETER_GRIDS = {
    "RandomForest": {
        "model__n_estimators": [100],
        "model__max_depth": [10, None],
        "model__min_samples_split": [5],
    },
    "GradientBoost": {
        "model__n_estimators": [100],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    },
    "XGBoost": {
        # Slower learning over more trees prevents early overfitting
        "model__n_estimators": [100, 300],
        "model__learning_rate": [0.01, 0.05],
        
        # Keep trees shallow. Macro data is too noisy for deep trees (>5)
        "model__max_depth": [2, 3, 5],
        
        # The ultimate overfit killers: Stochastic Gradient Boosting
        # Forces trees to train on random 80% slices of rows and columns
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
        
        # Requires a leaf to have enough "weight" before splitting. 
        # Higher numbers stop it from chasing single-month outliers.
        "model__min_child_weight": [1, 5],
    },
    "LightGBM": {
        "model__n_estimators": [100],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    },
}

"""
Hyperparameter search spaces for GridSearchCV.

Design rationale:
- Modest but meaningful ranges to balance exploration vs. computation time
- Focused on key hyperparameters that typically impact performance
- Ranges chosen based on domain knowledge and prior work
- GridSearchCV will evaluate all combinations within TimeSeriesSplit CV
"""

# 
# EVALUATION METRICS
# 

METRICS = ["RMSE", "MAE", "MAPE", "R2", "Directional_Accuracy"]
"""
List of evaluation metrics to compute for each model.

Definitions:
- RMSE: Root Mean Squared Error (penalizes large errors)
- MAE: Mean Absolute Error (robust to outliers)
- MAPE: Mean Absolute Percentage Error (scale-independent)
- R²: Coefficient of determination (variance explained)
- Directional_Accuracy: % of periods where predicted direction matches actual
  (critical for trading strategy — getting direction right is often more
  important than exact magnitude)
"""

# 
# LOGGING AND OUTPUT
# 

VERBOSE = True
"""
If True, print detailed progress messages during training and evaluation.
"""

SAVE_PLOTS = True
"""
If True, save prediction plots and comparison charts to ARTIFACTS_DIR.
"""

# Phase 2: Split Tournament Configuration
SPLITS_DIR = Path("/home/leo/Documents/Grad School/QCF/splits")  # Directory containing pre-processed feature subset CSVs
# Each CSV represents a different feature combination (e.g., fsbi_only.csv, fsbi_and_macro.csv, baseline_all.csv)


# In[ ]:




