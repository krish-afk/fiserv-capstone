#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 
# FILE: src/ensemble/base_models.py
# 
"""
Model Factory for Phase 1 Ensemble

This module is purely DECLARATIVE — it returns untrained model objects.
No data logic, no fitting, no side effects.

This design allows users to:
1. Swap model architectures trivially
2. Import and use models independently
3. Test new models without touching training logic

All models are initialized with RANDOM_STATE from config for reproducibility.
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from config import RANDOM_STATE


def get_models() -> dict:
    """
    Factory function that returns a dictionary of untrained model objects.

    Returns:
    --------
    dict
        Keys: model names (str)
        Values: untrained sklearn/xgboost/lightgbm regressor objects

    Models included:
    ----------------
    1. RandomForest: Ensemble of decision trees, robust to non-linearity
    2. GradientBoost: Sequential boosting, typically strong baseline
    3. XGBoost: Optimized gradient boosting with native NaN handling
    4. LightGBM: Fast gradient boosting with leaf-wise growth

    Notes:
    ------
    - XGBoost and LightGBM are configured to handle NaNs natively
      (via their split algorithms). They will NOT go through IterativeImputer.
    - RandomForest and GradientBoost will go through IterativeImputer
      in the training pipeline.
    - All models use RANDOM_STATE from config for reproducibility.
    """

    models = {
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,  # -1 to use all available cores was crashing my computer, but could work better. 
            verbose=0,
        ),
        "GradientBoost": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            verbose=0,
        ),
        "XGBoost": XGBRegressor(
            random_state=RANDOM_STATE,
            tree_method="hist", 
            enable_categorical=False,
            verbosity=0,
            n_jobs=1,
            objective="reg:absoluteerror",  # <-- NEW: Forces tree to optimize for MAE
        ),
        "LightGBM": LGBMRegressor(
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=1,
        ),
    }

    return models


# In[ ]:




