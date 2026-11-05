"""
Categorical Encoding Module
Transforms categorical contract, payment, and temporal attributes into dummy indicators.
"""

import pandas as pd
import numpy as np

def encode_features(input_path, output_path=None):
    """
    Encodes categorical features deterministically for data pipeline persistence.
    """
    df = pd.read_csv(input_path) if isinstance(input_path, str) else input_path.copy()
    
    # Contract type encoding
    contract_cats = ['1-year', '2-year', 'Month-to-month']
    for cat in contract_cats:
        df[f'contract_{cat}'] = (df.get('contract_type', '') == cat).astype(int)
    if 'contract_type' in df.columns:
        df.drop('contract_type', axis=1, inplace=True)
        
    # Payment method encoding
    payment_cats = ['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check']
    for cat in payment_cats:
        df[f'payment_{cat}'] = (df.get('payment_method', '') == cat).astype(int)
    if 'payment_method' in df.columns:
        df.drop('payment_method', axis=1, inplace=True)
        
    # Zip code encoding
    if 'zip_code' in df.columns:
        df['zip_code_encoded'] = pd.to_numeric(df['zip_code'], errors='coerce').fillna(500).astype(int)
        df.drop('zip_code', axis=1, inplace=True)
        
    # Signup month encoding
    signup_m = pd.to_numeric(df.get('signup_month', 6), errors='coerce').fillna(6).astype(int)
    for m in range(1, 13):
        df[f'month_{m}'] = (signup_m == m).astype(int)
    if 'signup_month' in df.columns:
        df.drop('signup_month', axis=1, inplace=True)
        
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df

if __name__ == '__main__':
    encode_features('data/processed/telecom_churn_train_cleaned.csv', 'data/processed/telecom_churn_train_encoded.csv')
    print("Encoding complete.")
