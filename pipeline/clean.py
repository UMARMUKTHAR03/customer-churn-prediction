"""
Data Cleaning Module
Removes duplicate records, handles missing numeric and categorical values with robust imputations.
"""

import pandas as pd
import numpy as np

def clean_dataset(file_path, output_path=None):
    """
    Cleans raw customer data by dropping duplicates and imputing missing values.
    """
    df = pd.read_csv(file_path)
    # Deduplicate
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Impute income if missing
    if 'income' in df_cleaned.columns and df_cleaned['income'].isnull().sum() > 0:
        median_income = df_cleaned['income'].median()
        df_cleaned['income'] = df_cleaned['income'].fillna(median_income)
        
    # Impute nps_score if missing
    if 'nps_score' in df_cleaned.columns and df_cleaned['nps_score'].isnull().sum() > 0:
        median_nps = df_cleaned['nps_score'].median()
        df_cleaned['nps_score'] = df_cleaned['nps_score'].fillna(median_nps)
        
    # Fill any remaining numeric missing values with median
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            
    # Fill any remaining categorical missing values with mode
    cat_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].empty else 'Unknown'
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)
            
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        
    return df_cleaned

if __name__ == '__main__':
    clean_dataset('data/raw/telecom_churn_train.csv', 'data/processed/telecom_churn_train_cleaned.csv')
    print("Cleaning complete.")
