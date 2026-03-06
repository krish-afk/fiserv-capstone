#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 
# FILE: src/ensemble/train.py
# 
"""
Training Pipeline for Phase 1 Ensemble

Orchestrates data loading, temporal splitting, hyperparameter tuning,
and model persistence.

Key functions:
- load_and_split(): Temporal train/test split (no shuffling)
- train_with_cv(): GridSearchCV with TimeSeriesSplit
- save_artifacts(): Persist models and hyperparameters
- run_training(): Main orchestration function
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from config import (
    TARGET,
    DATA_PATH,
    ARTIFACTS_DIR,
    TRAIN_SIZE,
    RANDOM_STATE,
    CV_N_SPLITS,
    HYPERPARAMETER_GRIDS,
    VERBOSE,
)
from base_models import get_models
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


def load_and_split(data_path, target, train_size):
    """
    Load CSV and perform strict temporal train/test split.

    Parameters:
    -----------
    data_path : Path or str
        Path to processed CSV file
    target : str
        Name of target column (e.g., "PCE")
    train_size : float
        Fraction of data for training (0.0 to 1.0)

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
        All splits preserve temporal order (no shuffling)

    Notes:
    ------
    - Data is assumed to be sorted by date (Period column)
    - No shuffling is applied — temporal integrity is preserved
    - Missing values (NaNs) are preserved here; imputation happens in pipeline
    """

    if VERBOSE:
        print("\n" + "=" * 80)
        print("LOADING AND SPLITTING DATA")
        print("=" * 80)

    # Load data
    df = pd.read_csv(data_path, engine="python")

    # Set 'Period' as the index so it's not treated as a feature
    if "Period" in df.columns:
        df = df.set_index("Period")
        df = df.sort_index()
    if VERBOSE:
        print(f"\n✓ Data loaded from: {data_path}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Verify target exists
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset columns.")

    # NEW: Transform target to Month-over-Month % change
    if VERBOSE:
        print("\n  Transforming target to % change to achieve stationarity...")
    
    # Calculate % change and multiply by 100 to get standard percentage (e.g. 1.5%)
    df["Target_Pct_Change"] = df[target].pct_change() * 100
    
    # Drop the single NaN row created at the very beginning by pct_change()
    df = df.dropna(subset=["Target_Pct_Change"])

    # Separate features and target (drop both the original target and the new column from X)
    X = df.drop(columns=[target, "Target_Pct_Change"])
    y = df["Target_Pct_Change"]

    # Temporal split (no shuffling)
    split_idx = int(len(df) * train_size)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if VERBOSE:
        print(f"\n✓ Temporal split completed (train_size={train_size}):")
        print(f"  Training set:   {len(X_train)} rows ({train_size*100:.1f}%)")
        print(f"  Test set:       {len(X_test)} rows ({(1-train_size)*100:.1f}%)")

        # Extract date range from Period column if available
        if "Period" in X.columns:
            train_period_min = X_train.index.min()
            train_period_max = X_train.index.max()
            test_period_min = X_test.index.min()
            test_period_max = X_test.index.max()

            print(f"\n  Training period: {train_period_min} to {train_period_max}")
            print(f"  Test period:     {test_period_min} to {test_period_max}")

        print(f"\n  Features: {X.shape[1]} columns")
        print(f"  Target: {target}")

    return X_train, X_test, y_train, y_test

def encode_categorical_features(X_train, X_test):
    """
    Encode categorical string columns to integers using OrdinalEncoder.
    
    Fits on training data and applies to test data. Unseen categories in the 
    test set are safely mapped to -1 to prevent pipeline crashes.
    """
    
    # NEW DYNAMIC DETECTION: Find all object/string columns automatically
    categorical_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    encoders = {}
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in categorical_cols:
        if col in X_train_encoded.columns:
            # handle_unknown maps unseen test categories to -1 safely
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            
            # OrdinalEncoder expects 2D arrays, hence the double brackets [[col]]
            X_train_encoded[col] = oe.fit_transform(X_train_encoded[[col]].astype(str))
            X_test_encoded[col] = oe.transform(X_test_encoded[[col]].astype(str))
            
            encoders[col] = oe
            print(f"  ✓ Encoded {col}: {len(oe.categories_[0])} unique values")
            
    return X_train_encoded, X_test_encoded, encoders


def train_with_cv(model, model_name, X_train, y_train, param_grid):
    """
    Train model with GridSearchCV and TimeSeriesSplit.

    Parameters:
    -----------
    model : sklearn estimator
        Untrained model object
    model_name : str
        Name of model (for logging)
    X_train : pd.DataFrame
        Training features (may contain NaNs)
    y_train : pd.Series
        Training target
    param_grid : dict
        Hyperparameter grid for GridSearchCV

    Returns:
    --------
    tuple
        (best_estimator, best_params_dict)
        best_estimator is the fitted Pipeline
        best_params_dict contains the best hyperparameters found

    Notes:
    ------
    - XGBoost and LightGBM skip IterativeImputer (handle NaNs natively)
    - RandomForest and GradientBoost use IterativeImputer
    - TimeSeriesSplit ensures no data leakage (expanding window)
    - Scoring metric is negative RMSE (GridSearchCV maximizes, so we negate)
    """

    if VERBOSE:
        print(f"\n{'─' * 80}")
        print(f"TRAINING: {model_name}")
        print(f"{'─' * 80}")

    # Determine if model needs imputation
    needs_imputation = model_name not in ["XGBoost", "LightGBM"]

    if needs_imputation:
        # Create pipeline with imputation
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("model", model),
            ]
        )
        if VERBOSE:
            print(f"  Pipeline: SimpleImputer (constant=0) → {model_name}")
    else:
        # Create pipeline without imputation (model handles NaNs natively)
        pipeline = Pipeline([("model", model)])
        if VERBOSE:
            print(f"  Pipeline: {model_name} (native NaN handling)")

    # Setup TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)

    if VERBOSE:
        print(f"  Cross-validation: TimeSeriesSplit (n_splits={CV_N_SPLITS})")
        print(f"  Hyperparameter grid size: {len(param_grid)} combinations")

    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=2,
        verbose=1 if VERBOSE else 0,
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = np.sqrt(-grid_search.best_score_)  # Convert back to RMSE

    if VERBOSE:
        print(f"\n  ✓ Best CV RMSE: {best_cv_score:.6f}")
        print(f"  ✓ Best hyperparameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")

    return best_estimator, best_params


def save_artifacts(model, model_name, best_params, artifacts_dir):
    """
    Save fitted model and hyperparameters to disk.

    Parameters:
    -----------
    model : sklearn Pipeline
        Fitted model (pipeline)
    model_name : str
        Name of model
    best_params : dict
        Best hyperparameters found
    artifacts_dir : Path
        Directory to save artifacts

    Returns:
    --------
    None

    Side effects:
    - Saves model to {artifacts_dir}/phase1_{model_name}_model.pkl
    - Saves params to {artifacts_dir}/phase1_{model_name}_params.json
    """

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = artifacts_dir / f"phase1_{model_name}_model.pkl"
    joblib.dump(model, model_path)

    # Save hyperparameters
    params_path = artifacts_dir / f"phase1_{model_name}_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    if VERBOSE:
        print(f"\n  ✓ Model saved: {model_path}")
        print(f"  ✓ Params saved: {params_path}")


def run_training():
    """
    Orchestrate full training pipeline.

    Returns:
    --------
    tuple
        (models_dict, X_test, y_test)
        models_dict: {"model_name": fitted_pipeline}
        X_test, y_test: Test split for evaluation
    """

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split(
        DATA_PATH, TARGET, TRAIN_SIZE
    )

    print("\n[TRAINING] Encoding categorical features...")
    X_train, X_test, encoders = encode_categorical_features(X_train, X_test)

    # Get untrained models
    models = get_models()

    # Train each model
    trained_models = {}
    for model_name, model in models.items():
        param_grid = HYPERPARAMETER_GRIDS[model_name]
        best_estimator, best_params = train_with_cv(
            model, model_name, X_train, y_train, param_grid
        )
        trained_models[model_name] = best_estimator
        save_artifacts(best_estimator, model_name, best_params, ARTIFACTS_DIR)

    if VERBOSE:
        print(f"\n{'=' * 80}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 80}")
        print(f"✓ All {len(trained_models)} models trained and saved")

    return trained_models, X_test, y_test


# In[ ]:




