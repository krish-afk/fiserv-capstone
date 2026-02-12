# ============================================================================
# FISERV + MACRO DATA: ADVANCED PREPROCESSING & FEATURE ENGINEERING PIPELINE
# ============================================================================
# This script performs comprehensive data preprocessing, feature engineering,
# and categorical encoding on combined FISERV and macroeconomic data.
#
# KEY UPDATES:
# 1. Keeps all 2019 data (no row dropping)
# 2. Drops original non-normalized features after normalization
# 3. Encodes categorical variables (one-hot encoding)
# 4. Normalizes engineered growth rates for ML optimization
# 5. Keeps NaNs in engineered features only
# 6. OPTIMIZED: Uses vectorized operations for speed (seconds, not minutes)
# 7. FIXED: Handles "Transaction YOY % - SA" column name with space before SA (probably artifact from previous script)
# 8. FIXED: Removes redundant date columns (Period_dt, Month)
# 9. FIXED: Reorganizes output columns in logical order
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FISERV + MACRO DATA: ADVANCED PREPROCESSING PIPELINE")
print("=" * 80)

# ============================================================================
# FUNCTION 1: LOAD AND PREPROCESS DATA
# ============================================================================
def load_and_preprocess_data(file_path):
    """
    Load the CSV file containing FISERV and macro data.
    Converts Period_dt to datetime format and sorts by date for time-series integrity.
    
    Parameters:
    -----------
    file_path (str): Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame: Loaded and sorted dataframe
    """
    print("\n[STEP 1] Loading data from CSV...")
    df = pd.read_csv(file_path)
    
    # Convert Period_dt to datetime format
    df['Period_dt'] = pd.to_datetime(df['Period_dt'])
    
    # Sort by geographic/sector groupings and date for time-series operations
    df = df.sort_values(
        by=['Geo', 'Sector Name', 'Sub-Sector Name', 'Period_dt']
    ).reset_index(drop=True)
    
    print(f"✓ Data loaded successfully")
    print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  - Date range: {df['Period_dt'].min()} to {df['Period_dt'].max()}")
    print(f"  - 2019 data preserved: {(df['Period_dt'].dt.year == 2019).sum()} rows")
    
    return df

# ============================================================================
# FUNCTION 2: HANDLE MISSING VALUES (OPTIMIZED WITH VECTORIZED OPERATIONS)
# ============================================================================
def handle_missing_values(df):
    """
    Handle missing values using VECTORIZED operations for speed.
    
    CRITICAL OPTIMIZATION: Uses groupby().transform() instead of loops.
    This is 100x+ faster because:
    - Vectorized operations use pandas' C extensions and NumPy
    - Loops involve Python interpreter overhead for each iteration
    - For 159,164 rows, vectorized = seconds, loops = minutes
    
    Strategy:
    1. Linear interpolation WITHIN each geographic/sector group
    2. Forward fill for remaining NaNs at start of each group
    3. Backward fill for remaining NaNs at end of each group
    4. Fill any remaining NaNs with column mean (global fallback)
    
    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe
    
    Returns:
    --------
    pd.DataFrame: Dataframe with interpolated missing values
    """
    print("\n[STEP 2] Handling missing values (OPTIMIZED)...")
    
    # Count missing values before
    missing_before = df.isnull().sum().sum()
    print(f"  - Missing values before: {missing_before}")
    
    # Identify numeric columns (exclude Period, Month, and categorical columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Define grouping columns
    group_cols = ['Geo', 'Sector Name', 'Sub-Sector Name']
    
    # ====================================================================
    # VECTORIZED OPERATION 1: Linear interpolation within groups
    # ====================================================================
    print("  - Performing linear interpolation within groups...")
    df[numeric_cols] = df.groupby(group_cols)[numeric_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # ====================================================================
    # VECTORIZED OPERATION 2: Forward fill within groups
    # ====================================================================
    print("  - Applying forward fill within groups...")
    df[numeric_cols] = df.groupby(group_cols)[numeric_cols].transform(
        lambda x: x.fillna(method='ffill')
    )
    
    # ====================================================================
    # VECTORIZED OPERATION 3: Backward fill within groups
    # ====================================================================
    print("  - Applying backward fill within groups...")
    df[numeric_cols] = df.groupby(group_cols)[numeric_cols].transform(
        lambda x: x.fillna(method='bfill')
    )
    
    # ====================================================================
    # VECTORIZED OPERATION 4: Fill remaining NaNs with column mean
    # ====================================================================
    print("  - Filling remaining NaNs with column mean...")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    missing_after = df.isnull().sum().sum()
    print(f"  - Missing values after: {missing_after}")
    print(f"✓ Missing values handled successfully (vectorized operations)")
    
    return df

# ============================================================================
# FUNCTION 3: NORMALIZE FEATURES
# ============================================================================
def normalize_features(df):
    """
    Normalize features using appropriate scaling methods.
    
    Strategy:
    - StandardScaler (z-score): For macro indicators and growth rates
      * Transforms to mean=0, std=1
      * Ideal for indicators with different units and scales
    
    - MinMaxScaler (0-1): For indices
      * Transforms to [0, 1] range
      * Appropriate for bounded indices
    
    CRITICAL: After normalization, original non-normalized columns are DROPPED
    to reduce dataset size and avoid redundancy. Only _normalized versions are kept.
    
    CRITICAL FIX: Handles "Transaction YOY % - SA" with space before SA
    
    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe
    
    Returns:
    --------
    pd.DataFrame: Dataframe with normalized columns (originals dropped)
    """
    print("\n[STEP 3] Normalizing features...")
    
    # Define macro indicators (different units: %, prices, indices, rates)
    macro_indicators = [
        'ConsumerSentimentIndex',      # Index (0-100 scale)
        'CreditSpreadBAA',             # Percentage points
        'CreditSpreadGS10',            # Percentage points
        'CrudeOilPrices',              # USD per barrel
        'ImportPriceIndex',            # Index
        'Income',                      # Billions of dollars
        'JoltsQuitsRate',              # Percentage
        'MonetaryCPI',                 # Index
        'PersonalConsumptionExpenditures',  # Billions of dollars
        'Unemployment',                # Percentage
        'USNaturalGasCompositePrice'   # USD per MMBtu
    ]
    
    # Define indices (already bounded, 0-100 or similar)
    indices = [
        'Real Sales Index - SA',
        'Transactional Index - SA',
        'Real Sales Index - NSA',
        'Transactional Index - NSA'
    ]
    
    # Define growth rates (percentages that will be normalized for ML)
    # CRITICAL FIX: Include "Transaction YOY % - SA" with exact spacing
    growth_rates = [
        'Real Sales MOM % - SA',
        'Real Sales YOY % - SA',
        'Transaction MOM % - SA',
        'Transaction YOY %  - SA',     
        'Real Sales MOM % - NSA',
        'Real Sales YOY % - NSA',
        'Transaction MOM % - NSA',
        'Transaction YOY % - NSA'
    ]
    
    # Apply StandardScaler to macro indicators
    print("  - Applying StandardScaler to macro indicators...")
    scaler_std = StandardScaler()
    for col in macro_indicators:
        if col in df.columns:
            df[col + '_normalized'] = scaler_std.fit_transform(df[[col]])
    print(f"    ✓ {len(macro_indicators)} macro indicators normalized")
    
    # Apply MinMaxScaler to indices
    print("  - Applying MinMaxScaler to indices...")
    scaler_minmax = MinMaxScaler()
    for col in indices:
        if col in df.columns:
            df[col + '_normalized'] = scaler_minmax.fit_transform(df[[col]])
    print(f"    ✓ {len(indices)} indices normalized")
    
    # Apply StandardScaler to growth rates (NEW: normalize for ML optimization)
    print("  - Applying StandardScaler to growth rates...")
    scaler_growth = StandardScaler()
    for col in growth_rates:
        if col in df.columns:
            df[col + '_normalized'] = scaler_growth.fit_transform(df[[col]])
    print(f"    ✓ {len(growth_rates)} growth rates normalized")
    
    # DROP original non-normalized columns to reduce dataset size
    print("  - Dropping original non-normalized columns...")
    cols_to_drop = macro_indicators + indices + growth_rates
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"    ✓ Dropped {len(cols_to_drop)} original columns")
    
    print(f"✓ Normalization complete (originals dropped, only _normalized versions kept)")
    
    return df

# ============================================================================
# FUNCTION 4: FEATURE ENGINEERING
# ============================================================================
def feature_engineering(df):
    """
    Perform comprehensive feature engineering on NORMALIZED data:
    1. Moving Averages (3, 6, 12-month): Captures trend and momentum
    2. Lagged Features (1, 3, 6-month): Captures temporal dependencies
    3. Growth Rates (MoM, YoY): Captures momentum and seasonality
    4. Interaction Terms: Captures cross-variable relationships
    
    CRITICAL: All operations are grouped by Geo, Sector Name, Sub-Sector Name
    to maintain temporal integrity and avoid data leakage across groups.
    
    CRITICAL FIX: After groupby().rolling() or groupby().shift() operations,
    we use .reset_index(drop=True) to flatten the MultiIndex created by groupby.
    This ensures the resulting Series has a flat index that aligns with the
    original DataFrame's index after sorting.
    
    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe with normalized features
    
    Returns:
    --------
    pd.DataFrame: Dataframe with engineered features added
    """
    print("\n[STEP 4] Feature Engineering...")
    
    # Define columns for time-series feature engineering
    # These are the NORMALIZED versions of original columns
    ts_cols = [
        'Real Sales Index - SA_normalized', 'Transactional Index - SA_normalized',
        'Real Sales MOM % - SA_normalized', 'Real Sales YOY % - SA_normalized',
        'Transaction MOM % - SA_normalized', 'Transaction YOY % - SA_normalized',
        'Real Sales Index - NSA_normalized', 'Transactional Index - NSA_normalized',
        'Real Sales MOM % - NSA_normalized', 'Real Sales YOY % - NSA_normalized',
        'Transaction MOM % - NSA_normalized', 'Transaction YOY % - NSA_normalized',
        'ConsumerSentimentIndex_normalized', 'CreditSpreadBAA_normalized', 
        'CreditSpreadGS10_normalized', 'CrudeOilPrices_normalized', 
        'ImportPriceIndex_normalized', 'Income_normalized', 'JoltsQuitsRate_normalized',
        'MonetaryCPI_normalized', 'PersonalConsumptionExpenditures_normalized', 
        'Unemployment_normalized', 'USNaturalGasCompositePrice_normalized'
    ]
    
    # Grouping columns to maintain temporal integrity
    group_cols = ['Geo', 'Sector Name', 'Sub-Sector Name']
    
    # Counter for tracking engineered features
    feature_count = 0
    
    print("  - Creating moving averages, lags, and growth rates...")
    
    for col in ts_cols:
        if col in df.columns:
            # ================================================================
            # MOVING AVERAGES: Smooth out noise and capture trends
            # ================================================================
            # 3-month MA: Short-term trend
            df[col + '_MA3'] = (df.groupby(group_cols)[col]
                                .rolling(window=3, min_periods=1)
                                .mean()
                                .reset_index(drop=True))
            
            # 6-month MA: Medium-term trend
            df[col + '_MA6'] = (df.groupby(group_cols)[col]
                                .rolling(window=6, min_periods=1)
                                .mean()
                                .reset_index(drop=True))
            
            # 12-month MA: Long-term trend (annual)
            df[col + '_MA12'] = (df.groupby(group_cols)[col]
                                 .rolling(window=12, min_periods=1)
                                 .mean()
                                 .reset_index(drop=True))
            
            # ================================================================
            # LAGGED FEATURES: Capture autoregressive relationships
            # ================================================================
            # 1-month lag: Immediate past value
            df[col + '_lag1'] = (df.groupby(group_cols)[col]
                                 .shift(1)
                                 .reset_index(drop=True))
            
            # 3-month lag: Quarterly relationship
            df[col + '_lag3'] = (df.groupby(group_cols)[col]
                                 .shift(3)
                                 .reset_index(drop=True))
            
            # 6-month lag: Semi-annual relationship
            df[col + '_lag6'] = (df.groupby(group_cols)[col]
                                 .shift(6)
                                 .reset_index(drop=True))
            
            # ================================================================
            # GROWTH RATES: Capture momentum and seasonality
            # NOTE: These are calculated on NORMALIZED data (already standardized)
            # ================================================================
            # Month-over-Month (MoM): Short-term momentum
            df[col + '_MoM'] = (df.groupby(group_cols)[col]
                                .pct_change()
                                .reset_index(drop=True) * 100)
            
            # Year-over-Year (YoY): Seasonal-adjusted growth
            df[col + '_YoY'] = (df.groupby(group_cols)[col]
                                .pct_change(12)
                                .reset_index(drop=True) * 100)
            
            feature_count += 8  # 3 MAs + 3 lags + 2 growth rates
    
    print(f"    ✓ Created {feature_count} time-series features")
    
    # ====================================================================
    # INTERACTION TERMS: Capture cross-variable relationships
    # ====================================================================
    print("  - Creating interaction terms...")
    
    interaction_count = 0
    
    # Unemployment × Income: Labor market tightness × spending capacity
    if 'Unemployment_normalized' in df.columns and 'Income_normalized' in df.columns:
        df['Unemployment_Income_interaction'] = (
            df['Unemployment_normalized'] * df['Income_normalized']
        )
        interaction_count += 1
    
    # Oil Prices × Import Price Index: External price shock transmission
    if 'CrudeOilPrices_normalized' in df.columns and 'ImportPriceIndex_normalized' in df.columns:
        df['Oil_ImportPrice_interaction'] = (
            df['CrudeOilPrices_normalized'] * df['ImportPriceIndex_normalized']
        )
        interaction_count += 1
    
    print(f"    ✓ Created {interaction_count} interaction terms")
    
    print(f"✓ Feature engineering complete")
    print(f"  - Total new features created: {feature_count + interaction_count}")
    
    return df

# ============================================================================
# FUNCTION 5: NORMALIZE ENGINEERED GROWTH RATES (FIXED - HANDLES INFINITIES)
# ============================================================================
def normalize_engineered_growth_rates(df):
    """
    Normalize the engineered growth rates (MoM, YoY) created during feature engineering.
    
    CRITICAL FIX: Handles infinities and NaNs that can arise from pct_change() operations.
    When pct_change() is applied to data with NaNs, it can produce inf/-inf values.
    StandardScaler cannot handle infinities, so we replace them with NaN first.
    
    These NaNs are then left in the output for model-specific handling.
    
    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe with engineered growth rates
    
    Returns:
    --------
    pd.DataFrame: Dataframe with normalized engineered growth rates
    """
    print("\n[STEP 5] Normalizing engineered growth rates...")
    
    # Find all engineered growth rate columns (those ending in _MoM or _YoY)
    engineered_growth_cols = [col for col in df.columns 
                              if col.endswith('_MoM') or col.endswith('_YoY')]
    
    if len(engineered_growth_cols) > 0:
        print(f"  - Found {len(engineered_growth_cols)} engineered growth rate columns")
        
        # CRITICAL: Replace infinities with NaN before normalization
        # pct_change() can produce inf/-inf when dividing by zero or NaN
        print(f"  - Replacing infinities with NaN...")
        for col in engineered_growth_cols:
            # Replace inf and -inf with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Apply StandardScaler to engineered growth rates
        # NaNs will be preserved through the scaling process
        scaler_eng_growth = StandardScaler()
        for col in engineered_growth_cols:
            # Only fit and transform on non-NaN values
            # This prevents NaNs from affecting the scaling parameters
            valid_mask = df[col].notna()
            if valid_mask.sum() > 0:  # Only if there are valid values
                df.loc[valid_mask, col + '_normalized'] = scaler_eng_growth.fit_transform(
                    df.loc[valid_mask, [col]]
                )
            else:
                # If all values are NaN, create a column of NaNs
                df[col + '_normalized'] = np.nan
        
        # Drop original (non-normalized) engineered growth rates
        df = df.drop(columns=engineered_growth_cols)
        
        print(f"    ✓ Normalized {len(engineered_growth_cols)} engineered growth rates")
        print(f"    ✓ Replaced infinities with NaN")
        print(f"    ✓ Dropped original engineered growth rate columns")
    else:
        print(f"  - No engineered growth rate columns found")
    
    print(f"✓ Engineered growth rate normalization complete")
    
    return df

# ============================================================================
# FUNCTION 6: ENCODE CATEGORICAL VARIABLES (FIXED)
# ============================================================================
def encode_categorical_variables(df):
    """
    Encode categorical variables using one-hot encoding.
    
    One-hot encoding converts categorical variables (Geo, Sector Name, Sub-Sector Name)
    into binary dummy variables suitable for ML models.
    
    Strategy:
    - Use OneHotEncoder with drop='first' to avoid multicollinearity
    - This creates n-1 binary columns for each categorical variable
    - Drop original text-based categorical columns after encoding
    
    CRITICAL FIX: Uses sparse_output instead of sparse parameter for compatibility
    with newer scikit-learn versions (>=1.2.0)
    
    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe with categorical columns
    
    Returns:
    --------
    pd.DataFrame: Dataframe with encoded categorical variables (originals dropped)
    """
    print("\n[STEP 6] Encoding categorical variables...")
    
    # Identify categorical columns to encode
    categorical_cols = ['Geo', 'Sector Name', 'Sub-Sector Name']
    
    # Check which categorical columns exist in the dataframe
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    if len(cols_to_encode) > 0:
        print(f"  - Found {len(cols_to_encode)} categorical columns to encode")
        
        # Apply one-hot encoding with drop='first' to avoid multicollinearity
        # This creates n-1 binary columns for each categorical variable
        # CRITICAL FIX: Use sparse_output instead of sparse for scikit-learn >= 1.2.0
        try:
            # Try newer parameter name first (scikit-learn >= 1.2.0)
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        except TypeError:
            # Fall back to older parameter name (scikit-learn < 1.2.0)
            encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
        
        # Fit and transform the categorical columns
        encoded_array = encoder.fit_transform(df[cols_to_encode])
        
        # Get feature names for the encoded columns
        feature_names = encoder.get_feature_names_out(cols_to_encode)
        
        # Create a dataframe with the encoded columns
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
        
        # Concatenate encoded columns with original dataframe
        df = pd.concat([df, encoded_df], axis=1)
        
        print(f"    ✓ Created {len(feature_names)} binary encoded columns")
        
        # Drop original categorical text columns
        df = df.drop(columns=cols_to_encode)
        print(f"    ✓ Dropped original categorical columns")
    else:
        print(f"  - No categorical columns found to encode")
    
    print(f"✓ Categorical encoding complete")
    
    return df

# ============================================================================
# FUNCTION 7: REMOVE REDUNDANT DATE COLUMNS AND REORGANIZE OUTPUT
# ============================================================================
def clean_and_reorganize_columns(df):
    """
    Remove redundant date columns and reorganize output in logical order:
    1. Date column (Period - YYYYMMDD format)
    2. FSBI normalized columns (indices and their growth rates)
    3. Macro normalized columns (in alphabetical order)
    4. Engineered features (moving averages, lags, growth rates, interactions)
    5. Categorical encoded columns
    
    CRITICAL FIX: Removes Period_dt and Month columns (redundant with Period)
    
    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe with all features
    
    Returns:
    --------
    pd.DataFrame: Reorganized dataframe with clean column order
    """
    print("\n[STEP 7] Cleaning and reorganizing columns...")
    
    # Drop redundant date columns
    print("  - Removing redundant date columns...")
    cols_to_drop = ['Period_dt', 'Month']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"    ✓ Dropped redundant columns: {cols_to_drop}")
    
    # Define column order
    print("  - Reorganizing columns in logical order...")
    
    # 1. Date column
    date_cols = ['Period']
    
    # 2. FSBI normalized columns (indices)
    fsbi_index_cols = [
        'Real Sales Index - SA_normalized',
        'Transactional Index - SA_normalized',
        'Real Sales Index - NSA_normalized',
        'Transactional Index - NSA_normalized'
    ]
    
    # 3. FSBI growth rates (MOM and YOY)
    fsbi_growth_cols = [
        'Real Sales MOM % - SA_normalized',
        'Real Sales YOY % - SA_normalized',
        'Transaction MOM % - SA_normalized',
        'Transaction YOY %  - SA_normalized',
        'Real Sales MOM % - NSA_normalized',
        'Real Sales YOY % - NSA_normalized',
        'Transaction MOM % - NSA_normalized',
        'Transaction YOY % - NSA_normalized'
    ]
    
    # 4. Macro normalized columns (in order)
    macro_cols = [
        'ConsumerSentimentIndex_normalized',
        'CreditSpreadBAA_normalized',
        'CreditSpreadGS10_normalized',
        'CrudeOilPrices_normalized',
        'ImportPriceIndex_normalized',
        'Income_normalized',
        'JoltsQuitsRate_normalized',
        'MonetaryCPI_normalized',
        'PersonalConsumptionExpenditures_normalized',
        'Unemployment_normalized',
        'USNaturalGasCompositePrice_normalized'
    ]
    
    # 5. Engineered features (all remaining numeric columns that aren't categorical)
    # Get all columns that aren't in the above lists and aren't categorical encoded
    all_cols = set(df.columns)
    ordered_cols_set = set(date_cols + fsbi_index_cols + fsbi_growth_cols + macro_cols)
    
    # Engineered features are everything else except categorical (which start with 'x0_', 'x1_', etc.)
    engineered_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col not in ordered_cols_set:
            # Check if it's a categorical encoded column (contains 'x0_', 'x1_', etc.)
            if any(col.startswith(f'x{i}_') for i in range(10)):
                categorical_cols.append(col)
            else:
                engineered_cols.append(col)
    
    # Sort engineered columns for consistency
    engineered_cols.sort()
    
    # Build final column order
    final_column_order = (
        [col for col in date_cols if col in df.columns] +
        [col for col in fsbi_index_cols if col in df.columns] +
        [col for col in fsbi_growth_cols if col in df.columns] +
        [col for col in macro_cols if col in df.columns] +
        engineered_cols +
        categorical_cols
    )
    
    # Reorder dataframe
    df = df[final_column_order]
    
    print(f"    ✓ Reorganized columns:")
    print(f"      - Date columns: {len([col for col in date_cols if col in df.columns])}")
    print(f"      - FSBI index columns: {len([col for col in fsbi_index_cols if col in df.columns])}")
    print(f"      - FSBI growth columns: {len([col for col in fsbi_growth_cols if col in df.columns])}")
    print(f"      - Macro columns: {len([col for col in macro_cols if col in df.columns])}")
    print(f"      - Engineered features: {len(engineered_cols)}")
    print(f"      - Categorical columns: {len(categorical_cols)}")
    
    print(f"✓ Column cleaning and reorganization complete")
    
    return df

# ============================================================================
# FUNCTION 8: DATA QUALITY REPORT
# ============================================================================
def data_quality_report(df, stage):
    """
    Generate comprehensive data quality report and summary statistics.
    
    Parameters:
    -----------
    df (pd.DataFrame): Dataframe to report on
    stage (str): Processing stage label (e.g., 'Original', 'After Normalization')
    """
    print(f"\n{'─' * 80}")
    print(f"DATA QUALITY REPORT: {stage}")
    print(f"{'─' * 80}")
    
    print(f"\nDataset Shape:")
    print(f"  - Rows: {df.shape[0]}")
    print(f"  - Columns: {df.shape[1]}")
    
    print(f"\nMissing Values:")
    missing_count = df.isnull().sum()
    if missing_count.sum() == 0:
        print(f"  ✓ No missing values")
    else:
        missing_cols = missing_count[missing_count > 0]
        print(f"  - Columns with NaNs: {len(missing_cols)}")
        for col, count in missing_cols.head(10).items():
            print(f"    • {col}: {count} NaNs ({count/len(df)*100:.2f}%)")
        if len(missing_cols) > 10:
            print(f"    ... and {len(missing_cols) - 10} more columns with NaNs")
    
    print(f"\nData Types:")
    print(f"  - Numeric: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"  - Object: {df.select_dtypes(include=['object']).shape[1]}")
    print(f"  - Datetime: {df.select_dtypes(include=['datetime64']).shape[1]}")
    
    print(f"\nDate Coverage:")
    if 'Period' in df.columns:
        print(f"  - Start: {df['Period'].min()}")
        print(f"  - End: {df['Period'].max()}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main execution function that orchestrates the entire pipeline.
    """
    
    # ====================================================================
    # UPDATE THIS PATH TO YOUR CSV INPUT FILE
    # ====================================================================
    file_path = '/home/leo/Documents/Grad School/QCF/fiserv_with_fred_appended.csv'  # ← CHANGE THIS TO YOUR FILE PATH
    
    try:
        # Step 1: Load data
        df = load_and_preprocess_data(file_path)
        data_quality_report(df, 'ORIGINAL DATA')
        
        # Step 2: Handle missing values (OPTIMIZED)
        df = handle_missing_values(df)
        data_quality_report(df, 'AFTER MISSING VALUE HANDLING')
        
        # Step 3: Normalize features (drops originals)
        df = normalize_features(df)
        data_quality_report(df, 'AFTER NORMALIZATION')
        
        # Step 4: Feature engineering
        df = feature_engineering(df)
        data_quality_report(df, 'AFTER FEATURE ENGINEERING')
        
        # Step 5: Normalize engineered growth rates (drops originals)
        df = normalize_engineered_growth_rates(df)
        data_quality_report(df, 'AFTER NORMALIZING ENGINEERED GROWTH RATES')
        
        # Step 6: Encode categorical variables (drops originals)
        df = encode_categorical_variables(df)
        data_quality_report(df, 'AFTER CATEGORICAL ENCODING')
        
        # Step 7: Clean redundant columns and reorganize
        df = clean_and_reorganize_columns(df)
        data_quality_report(df, 'AFTER FINAL CLEANUP AND REORGANIZATION')
        
        # Step 8: Save to CSV
        output_path = 'processed_fiserv_macro_data.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"✓ PROCESSING COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nOutput file saved: {output_path}")
        print(f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn Order:")
        print(f"  1. Date column (Period)")
        print(f"  2. FSBI normalized indices")
        print(f"  3. FSBI normalized growth rates")
        print(f"  4. Macro normalized indicators")
        print(f"  5. Engineered features (MAs, lags, growth rates, interactions)")
        print(f"  6. Categorical encoded columns")
        print(f"\nKey Features:")
        print(f"  ✓ All 2019 data preserved")
        print(f"  ✓ Original non-normalized columns dropped")
        print(f"  ✓ Growth rates normalized for ML optimization")
        print(f"  ✓ Categorical variables one-hot encoded")
        print(f"  ✓ NaNs kept in output (from lagged/MA features)")
        print(f"  ✓ OPTIMIZED: Vectorized operations (seconds, not minutes)")
        print(f"  ✓ FIXED: 'Transaction YOY % - SA' properly normalized")
        print(f"  ✓ FIXED: Redundant date columns removed")
        print(f"  ✓ FIXED: Columns reorganized in logical order")
        
        return df
    
    except FileNotFoundError:
        print(f"\n✗ ERROR: File '{file_path}' not found.")
        print(f"Please update the file_path variable with the correct path to your CSV file.")
        return None
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print(f"Please check your data and try again.")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# RUN THE PIPELINE
# ============================================================================
if __name__ == '__main__':
    processed_df = main()