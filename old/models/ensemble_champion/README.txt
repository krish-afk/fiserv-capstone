================================================================================
CHAMPION MODEL EXECUTOR - FISERV CAPSTONE PROJECT
================================================================================

QUICKSTART: RUN THE CHAMPION MODEL

Our strongest predictive model for FORECASTEX simulations uses XGBoost on the 
New York-New Jersey-Philadelphia metro area. To instantly evaluate the champion model:

QUICK START (3 STEPS)

1. Install dependencies:
   pip install -r requirements.txt

2. Run the champion model:
   python run_champion_model.py

3. Review the output:
   - Console report with MAE, RMSE, R-squared metrics
   - Forecast visualization saved to results/phase1/champion_model_forecast.png

================================================================================
CONFIGURATION
================================================================================

To re-train the model instead of loading pre-trained weights, edit 
run_champion_model.py and change this line:

   LOAD_PRETRAINED = True

To this:

   LOAD_PRETRAINED = False

Then run the script again. If the pre-trained model file is not found, the 
script automatically trains instantly with champion hyperparameters.

================================================================================
MODEL DETAILS
================================================================================

Sector:              Accommodation and Food Services
Algorithm:           XGBoost Regressor
Objective:           Mean Absolute Error (MAE)
Hyperparameters:     n_estimators=100, max_depth=2, learning_rate=0.01, 
                     subsample=0.8
Train/Test Split:    90/10 temporal

================================================================================
FOR STAKEHOLDERS
================================================================================

This script is designed for non-technical stakeholders to evaluate model 
performance without running complex grid searches or cross-validation. 

Simply execute the script and review:
- Professional console report with key metrics
- Forecast visualization showing actual vs. predicted PCE
- Ready-to-use model for FORECASTEX trading simulations

================================================================================
PROJECT CONTEXT
================================================================================

This capstone project leverages Fiserv's Small Business Index (FSBI) to 
forecast Personal Consumption Expenditures (PCE), a key economic indicator. 
The goal is to explore how small business and same-store spending trends 
correlate with overall consumption patterns and use this information to model 
potential gains on financial trading platforms like FORECASTEX.

Key Objectives:
1. Establish whether FSBI is a good leading predictor of PCE
2. Identify and simulate profitable trading strategies using this information

================================================================================