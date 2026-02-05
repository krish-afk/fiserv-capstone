# Predicting Personal Consumption Expenditures (PCE) & Marginal Rate of Technical Substitution (MRTS) using Fiserv Small Business Index & Spend Trend Data  
## and Modeling Trading Gains on Forecast Platforms (e.g., FORECASTEX)

## Overview
This capstone project leverages Fiserv’s Small Business Index (FSBI) and related spending trend data to forecast U.S. Personal Consumption Expenditures (PCE), a key economic indicator. We then simulate trading strategies on a FORECASTEX-style framework to model potential gains/losses from more accurate PCE predictions. 

## Goals
1. Establish whether FSBI is a good leading predictor of the PCE and MRTS.  
2. If yes, identify and simulate profitable trading strategies using this information.

## Project Objectives
### 1) Data Integration and Preprocessing
- Collect and integrate FSBI/Spend Trend data with publicly available PCE and macroeconomic indicators (e.g., inflation rate, GDP growth, consumer confidence).
- Clean and preprocess data (missing values, encoding categorical features if applicable).
- Align datasets across a consistent timeframe/frequency (monthly or quarterly).

### 2) PCE Prediction Model
- Develop models to forecast future PCE values using FSBI (and optional macro indicators).
- Evaluate multiple approaches, including:
  - Regression models
  - Time series forecasting (e.g., ARIMA, LSTM)
  - Ensemble methods (e.g., Random Forest, Gradient Boosting)
- Track model performance with metrics such as RMSE, MAPE, MAE, R², and cross-validation where appropriate. 

### 3) Trading Simulation (FORECASTEX-style)
- Use predicted PCE values to simulate trading strategies based on market reactions to forecasted economic trends.
- Model gains/losses under different strategies (e.g., sector rotation, volatility-based strategies).
- Evaluate results with ROI, Sharpe ratio, maximum drawdown, and related diagnostics. 

### 4) Visualization and Decision Support
- Build interactive dashboards (e.g., Plotly/Dash or Tableau) to visualize:
  - PCE predictions vs. actuals
  - Small business activity indicators
  - Simulated trading performance
- Provide actionable insights and recommendations based on findings. 

### 5) Final Report and Presentation
- Document the end-to-end process, including data, modeling, trading simulation, and visualizations.
- Summarize findings, limitations, and recommendations. 

## Key Deliverables
1. **Preprocessed Dataset**
   - Integrated clean dataset combining FSBI with PCE and relevant indicators
   - Data transformation scripts and feature engineering documentation 

2. **PCE Forecasting Model**
   - Trained forecasting model and evaluation results (RMSE, MAPE, MAE, R², cross-validation)
   - Documentation of model selection rationale 

3. **Trading Simulation Framework**
   - Simulated strategies based on predicted PCE (e.g., buy/sell on expected PCE growth)
   - Performance reports: ROI, Sharpe ratio, max drawdown, and sensitivity to forecast accuracy
   - Analysis under varying market conditions 
4. **Interactive Visualizations**
   - Dashboard(s) showing predictions, drivers, and trading outcomes
   - Supporting charts/figures for analysis and reporting 

5. **Final Report**
   - Comprehensive methodology, results, trading outcomes, and practical implications 

## Data Sources
This project is based solely on publicly available data plus the provided Fiserv datasets. 

- **Fiserv**: Small Business Index / Spend Trend datasets (provided)
- **PCE**: U.S. Bureau of Economic Analysis (BEA)
- **MRTS**:
- **Macro indicators**: inflation, interest rates, GDP growth, consumer confidence, etc.


## Tools & Technologies (planned)
- Languages: Python (primary), R (optional)
- Libraries/Tooling (as needed):
  - ML: scikit-learn, XGBoost, Keras/TensorFlow, statsmodels, LightGBM
  - Time series: ARIMA/SARIMA, Prophet, LSTM
  - Data: pandas, numpy
  - Visualization: matplotlib, plotly, dash, tableau
  - Trading simulation: Backtrader, Monte Carlo simulation 

## Project Timeline (Will be updated soon)
- **Weeks 1–3**: Data Collection & Preprocessing  
- **Weeks 4–5**: Exploratory Data Analysis (EDA)  
- **Weeks 6–8**: Model Development for PCE Prediction  
- **Weeks 9–10**: Trading Strategy Design & Simulation  
- **Weeks 11–12**: Performance Evaluation & Sensitivity Analysis  
- **Weeks 13–14**: Visualization & Dashboard Development  
- **Week 15**: Final Report & Presentation 

