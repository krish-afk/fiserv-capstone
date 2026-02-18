#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 
# FISERV + MACRO DATA: PREPROCESSING & FEATURE ENGINEERING PIPELINE
# 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FISERV + MACRO DATA PROCESSING PIPELINE")
print("=" * 80)

def load_and_preprocess_data(file_path):
    print("\n[STEP 1] Loading data...")
    df = pd.read_csv(file_path)
    df['Period_dt'] = pd.to_datetime(df['Period_dt'])
    df = df.sort_values(
        by=['Geo', 'Sector Name', 'Sub-Sector Name', 'Period_dt']
    ).reset_index(drop=True)
    
    print(f"✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def handle_missing_values(df):
    print("\n[STEP 2] Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = ['Geo', 'Sector Name', 'Sub-Sector Name']
    
    df[numeric_cols] = df.groupby(group_cols)[numeric_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    df[numeric_cols] = df.groupby(group_cols)[numeric_cols].transform(
        lambda x: x.fillna(method='ffill')
    )
    df[numeric_cols] = df.groupby(group_cols)[numeric_cols].transform(
        lambda x: x.fillna(method='bfill')
    )
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print(f"✓ Missing values handled")
    return df

def normalize_features(df):
    print("\n[STEP 3] Normalizing features...")
    
    macro_indicators = [
        'ConsumerSentimentIndex', 'CreditSpreadBAA', 'CreditSpreadGS10',
        'CrudeOilPrices', 'ImportPriceIndex', 'Income', 'JoltsQuitsRate',
        'MonetaryCPI', 'PersonalConsumptionExpenditures', 'Unemployment',
        'USNaturalGasCompositePrice'
    ]
    
    indices = [
        'Real Sales Index - SA', 'Transactional Index - SA',
        'Real Sales Index - NSA', 'Transactional Index - NSA'
    ]
    
    growth_rates = [
        'Real Sales MOM % - SA', 'Real Sales YOY % - SA',
        'Transaction MOM % - SA', 'Transaction YOY %  - SA',
        'Real Sales MOM % - NSA', 'Real Sales YOY % - NSA',
        'Transaction MOM % - NSA', 'Transaction YOY % - NSA'
    ]
    
    scaler_std = StandardScaler()
    for col in macro_indicators:
        if col in df.columns:
            df[col + '_normalized'] = scaler_std.fit_transform(df[[col]])
    
    scaler_minmax = MinMaxScaler()
    for col in indices:
        if col in df.columns:
            df[col + '_normalized'] = scaler_minmax.fit_transform(df[[col]])
    
    scaler_growth = StandardScaler()
    for col in growth_rates:
        if col in df.columns:
            df[col + '_normalized'] = scaler_growth.fit_transform(df[[col]])
    
    cols_to_drop = [col for col in macro_indicators + indices + growth_rates if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    print(f"✓ Normalized {len(macro_indicators) + len(indices) + len(growth_rates)} features")
    return df

def feature_engineering(df):
    print("\n[STEP 4] Feature engineering...")
    
    fsbi_index_cols = [
        'Real Sales Index - SA_normalized',
        'Transactional Index - SA_normalized',
        'Real Sales Index - NSA_normalized',
        'Transactional Index - NSA_normalized'
    ]
    
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
    
    group_cols = ['Geo', 'Sector Name', 'Sub-Sector Name']
    
    for col in fsbi_index_cols:
        if col in df.columns:
            df[col + '_lag1'] = df.groupby(group_cols)[col].shift(1).reset_index(drop=True)
            df[col + '_lag3'] = df.groupby(group_cols)[col].shift(3).reset_index(drop=True)
            df[col + '_lag6'] = df.groupby(group_cols)[col].shift(6).reset_index(drop=True)
            df[col + '_MA3'] = df.groupby(group_cols)[col].rolling(window=3, min_periods=1).mean().reset_index(drop=True)
            df[col + '_MA6'] = df.groupby(group_cols)[col].rolling(window=6, min_periods=1).mean().reset_index(drop=True)
            df[col + '_MA12'] = df.groupby(group_cols)[col].rolling(window=12, min_periods=1).mean().reset_index(drop=True)
    
    for col in fsbi_growth_cols:
        if col in df.columns:
            df[col + '_lag1'] = df.groupby(group_cols)[col].shift(1).reset_index(drop=True)
            df[col + '_lag3'] = df.groupby(group_cols)[col].shift(3).reset_index(drop=True)
            df[col + '_lag6'] = df.groupby(group_cols)[col].shift(6).reset_index(drop=True)
            df[col + '_MA3'] = df.groupby(group_cols)[col].rolling(window=3, min_periods=1).mean().reset_index(drop=True)
            df[col + '_MA6'] = df.groupby(group_cols)[col].rolling(window=6, min_periods=1).mean().reset_index(drop=True)
            df[col + '_MA12'] = df.groupby(group_cols)[col].rolling(window=12, min_periods=1).mean().reset_index(drop=True)
    
    for col in macro_cols:
        if col in df.columns:
            df[col + '_lag1'] = df.groupby(group_cols)[col].shift(1).reset_index(drop=True)
            df[col + '_lag3'] = df.groupby(group_cols)[col].shift(3).reset_index(drop=True)
            df[col + '_lag6'] = df.groupby(group_cols)[col].shift(6).reset_index(drop=True)
            df[col + '_MA3'] = df.groupby(group_cols)[col].rolling(window=3, min_periods=1).mean().reset_index(drop=True)
            df[col + '_MA6'] = df.groupby(group_cols)[col].rolling(window=6, min_periods=1).mean().reset_index(drop=True)
            df[col + '_MA12'] = df.groupby(group_cols)[col].rolling(window=12, min_periods=1).mean().reset_index(drop=True)
            df[col + '_MoM'] = df.groupby(group_cols)[col].pct_change().reset_index(drop=True) * 100
            df[col + '_YoY'] = df.groupby(group_cols)[col].pct_change(12).reset_index(drop=True) * 100
    
    if 'Unemployment_normalized' in df.columns and 'Income_normalized' in df.columns:
        df['Unemployment_Income_interaction'] = df['Unemployment_normalized'] * df['Income_normalized']
    if 'CrudeOilPrices_normalized' in df.columns and 'ImportPriceIndex_normalized' in df.columns:
        df['Oil_ImportPrice_interaction'] = df['CrudeOilPrices_normalized'] * df['ImportPriceIndex_normalized']
    
    print(f"✓ Created engineered features")
    return df

def normalize_engineered_growth_rates(df):
    print("\n[STEP 5] Normalizing engineered growth rates...")
    
    engineered_growth_cols = [col for col in df.columns 
                              if col.endswith('_MoM') or col.endswith('_YoY')]
    
    if len(engineered_growth_cols) > 0:
        for col in engineered_growth_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        scaler_eng_growth = StandardScaler()
        for col in engineered_growth_cols:
            valid_mask = df[col].notna()
            if valid_mask.sum() > 0:
                df.loc[valid_mask, col + '_normalized'] = scaler_eng_growth.fit_transform(
                    df.loc[valid_mask, [col]]
                )
            else:
                df[col + '_normalized'] = np.nan
        
        df = df.drop(columns=engineered_growth_cols)
        print(f"✓ Normalized {len(engineered_growth_cols)} growth rate features")
    
    return df

def clean_and_reorganize_columns(df):
    print("\n[STEP 6] Reorganizing columns...")
    
    # Drop redundant date columns
    df = df.drop(columns=[col for col in ['Period_dt', 'Month'] if col in df.columns])
    
    # Define the EXACT order based on diagnostic output
    base_cols = ['Period', 'Geo', 'Sector Name', 'Sub-Sector Name']
    
    macro_normalized = [
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
    
    fsbi_index_normalized = [
        'Real Sales Index - SA_normalized',
        'Transactional Index - SA_normalized',
        'Real Sales Index - NSA_normalized',
        'Transactional Index - NSA_normalized'
    ]
    
    fsbi_growth_normalized = [
        'Real Sales MOM % - SA_normalized',
        'Real Sales YOY % - SA_normalized',
        'Transaction MOM % - SA_normalized',
        'Transaction YOY %  - SA_normalized',
        'Real Sales MOM % - NSA_normalized',
        'Real Sales YOY % - NSA_normalized',
        'Transaction MOM % - NSA_normalized',
        'Transaction YOY % - NSA_normalized'
    ]
    
    # Build the master order for normalized columns
    normalized_cols = fsbi_index_normalized + fsbi_growth_normalized + macro_normalized
    
    # Build engineered features in order
    engineered_cols = []
    
    # For each feature type, add columns in the order of normalized columns
    for suffix in ['_lag1', '_lag3', '_lag6', '_MA3', '_MA6', '_MA12']:
        for base_col in normalized_cols:
            col_name = base_col + suffix
            if col_name in df.columns:
                engineered_cols.append(col_name)
    
    # Add MoM and YoY for macros only
    for base_col in macro_normalized:
        col_name = base_col + '_MoM_normalized'
        if col_name in df.columns:
            engineered_cols.append(col_name)
    
    for base_col in macro_normalized:
        col_name = base_col + '_YoY_normalized'
        if col_name in df.columns:
            engineered_cols.append(col_name)
    
    # Add interactions
    interaction_cols = [col for col in df.columns if 'interaction' in col]
    engineered_cols.extend(sorted(interaction_cols))
    
    # Build final order
    final_order = base_cols + normalized_cols + engineered_cols
    
    # Validate
    print(f"  Base columns: {len(base_cols)}")
    print(f"  Normalized columns: {len(normalized_cols)}")
    print(f"  Engineered columns: {len(engineered_cols)}")
    print(f"  Total in order: {len(final_order)}")
    print(f"  Total in dataframe: {len(df.columns)}")
    
    if len(final_order) != len(df.columns):
        print(f"  ⚠ MISMATCH DETECTED")
        missing = set(df.columns) - set(final_order)
        print(f"  Missing from order: {missing}")
        # Add any missing columns
        final_order.extend(list(missing))
    
    # Reorder and return
    df = df[final_order]
    print(f"✓ Columns reorganized - {len(df.columns)} columns in output")
    
    return df

def data_quality_report(df, stage):
    print(f"\n{'─' * 80}")
    print(f"DATA QUALITY: {stage}")
    print(f"{'─' * 80}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

def main():
    file_path = 'path_to_file' #CHANGE YOUR PATH HERE
    
    try:
        df = load_and_preprocess_data(file_path)
        data_quality_report(df, 'ORIGINAL DATA')
        
        df = handle_missing_values(df)
        df = normalize_features(df)
        df = feature_engineering(df)
        df = normalize_engineered_growth_rates(df)
        data_quality_report(df, 'BEFORE REORGANIZATION')
        
        df = clean_and_reorganize_columns(df)
        data_quality_report(df, 'FINAL OUTPUT')
        
        output_path = 'processed_fiserv_macro_data.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"✓ PROCESSING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Output saved: {output_path}")
        print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        return df
    
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    processed_df = main()


# In[ ]:




