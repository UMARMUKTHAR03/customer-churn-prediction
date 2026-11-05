"""
Telecom Churn Prediction - Preprocessing & Feature Engineering Pipeline
Handles data cleaning, imputation, categorical encoding, and feature synthesis
without target leakage or preprocessing leakage.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TelecomDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Robust transformer for Telecom Churn data.
    Fits missing value statistics and category vocabularies on training data,
    then transforms any single customer record or batch dataframe deterministically.
    """
    def __init__(self):
        self.medians_ = {}
        self.modes_ = {}
        self.zip_quantiles_ = None
        self.contract_categories_ = ['1-year', '2-year', 'Month-to-month']
        self.payment_categories_ = ['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check']
        self.month_categories_ = list(range(1, 13))
        self.feature_names_ = None

    def fit(self, X, y=None):
        df = X.copy()
        # Compute medians for numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            self.medians_[c] = float(df[c].median())
            
        # Compute modes for categorical columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for c in cat_cols:
            self.modes_[c] = df[c].mode()[0] if not df[c].empty else ''
            
        # Standard defaults if columns were completely missing in training
        defaults = {
            'age': 38.0,
            'income': 55000.0,
            'zip_code': 500.0,
            'tenure_months': 12.0,
            'monthly_charges': 65.0,
            'total_charges': 780.0,
            'num_support_calls': 2.0,
            'num_complaints': 1.0,
            'avg_call_duration_min': 4.5,
            'nps_score': 7.0,
            'data_usage_gb': 150.0,
            'minutes_used': 300.0,
            'sms_sent': 50.0,
            'competitor_price_diff': 0.0,
            'signup_month': 6.0,
            'last_interaction_days': 20.0,
            'has_internet': 1.0,
            'has_online_security': 0.0,
            'has_online_backup': 0.0,
            'has_device_protection': 0.0,
            'has_tech_support': 0.0,
            'has_streaming_tv': 0.0,
            'has_streaming_movies': 0.0,
            'has_phone': 1.0,
            'has_multiple_lines': 0.0
        }
        for k, v in defaults.items():
            if k not in self.medians_:
                self.medians_[k] = v

        if 'zip_code' in df.columns:
            try:
                self.zip_quantiles_ = pd.qcut(df['zip_code'].astype(float), q=5, retbins=True, duplicates='drop')[1]
            except Exception:
                self.zip_quantiles_ = None

        return self

    def transform(self, X):
        df = X.copy()
        if isinstance(df, dict):
            df = pd.DataFrame([df])

        # Fill missing values from fitted statistics
        for c, med in self.medians_.items():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med)
            else:
                df[c] = med
                
        for c, m in self.modes_.items():
            if c in df.columns:
                df[c] = df[c].fillna(m)
            else:
                df[c] = m

        # Total charges fallback
        if 'total_charges' not in df.columns or df['total_charges'].isnull().all():
            df['total_charges'] = df['tenure_months'] * df['monthly_charges']

        # Categorical contract dummies
        if 'contract_type' not in df.columns:
            df['contract_type'] = 'Month-to-month'
        contract_series = df['contract_type'].astype(str)
        # Normalize contract string variations
        contract_norm = contract_series.replace({
            'One year': '1-year',
            'Two year': '2-year',
            'Month-to-Month': 'Month-to-month'
        })
        for cat in self.contract_categories_:
            df[f'contract_{cat}'] = (contract_norm == cat).astype(int)

        # Categorical payment dummies
        if 'payment_method' not in df.columns:
            df['payment_method'] = 'Electronic check'
        payment_series = df['payment_method'].astype(str)
        payment_norm = payment_series.apply(lambda p: 
            'Credit card' if 'credit card' in p.lower() else (
            'Bank transfer' if 'bank transfer' in p.lower() else (
            'Mailed check' if 'mailed check' in p.lower() else 'Electronic check'
        )))
        for cat in self.payment_categories_:
            df[f'payment_{cat}'] = (payment_norm == cat).astype(int)

        # Zip code encoding and quantile regions
        if 'zip_code' not in df.columns:
            df['zip_code'] = 500
        zip_num = pd.to_numeric(df['zip_code'], errors='coerce').fillna(500).astype(int)
        df['zip_code_encoded'] = zip_num
        if self.zip_quantiles_ is not None:
            df['zip_region'] = pd.cut(zip_num, bins=self.zip_quantiles_, labels=False, include_lowest=True).fillna(0).astype(int)
        else:
            df['zip_region'] = (zip_num // 200).clip(0, 4)

        # Signup month dummies
        if 'signup_month' not in df.columns:
            df['signup_month'] = 6
        signup_m = pd.to_numeric(df['signup_month'], errors='coerce').fillna(6).astype(int)
        for m in self.month_categories_:
            df[f'month_{m}'] = (signup_m == m).astype(int)

        # Feature Engineering: Services and Behavioral Ratios
        service_cols = ['has_internet', 'has_online_security', 'has_online_backup', 
                        'has_device_protection', 'has_tech_support', 'has_streaming_tv', 
                        'has_streaming_movies', 'has_phone', 'has_multiple_lines']
        for sc in service_cols:
            if sc not in df.columns:
                df[sc] = 0
            else:
                df[sc] = pd.to_numeric(df[sc], errors='coerce').fillna(0).astype(int)

        df['num_services'] = df[service_cols].sum(axis=1)
        df['has_streaming'] = ((df['has_streaming_tv'] == 1) | (df['has_streaming_movies'] == 1)).astype(int)
        df['charge_per_service'] = df['monthly_charges'] / (df['num_services'] + 0.1)
        df['data_per_month'] = df['data_usage_gb'] / (df['tenure_months'] + 0.1)
        df['calls_per_month'] = df['minutes_used'] / (df['tenure_months'] + 0.1)
        df['sms_per_month'] = df['sms_sent'] / (df['tenure_months'] + 0.1)
        df['is_new_customer'] = (df['tenure_months'] < 3).astype(int)
        df['is_long_term'] = (df['tenure_months'] > 24).astype(int)
        df['approx_ltv'] = df['tenure_months'] * df['monthly_charges']

        target_features = [
            'age', 'income', 'tenure_months', 'monthly_charges', 'total_charges', 
            'has_internet', 'has_online_security', 'has_online_backup', 'has_device_protection', 
            'has_tech_support', 'has_streaming_tv', 'has_streaming_movies', 'has_phone', 
            'has_multiple_lines', 'num_support_calls', 'num_complaints', 'avg_call_duration_min', 
            'nps_score', 'data_usage_gb', 'minutes_used', 'sms_sent', 'competitor_price_diff', 
            'last_interaction_days', 'contract_1-year', 'contract_2-year', 'contract_Month-to-month', 
            'payment_Bank transfer', 'payment_Credit card', 'payment_Electronic check', 
            'payment_Mailed check', 'zip_code_encoded', 'month_1', 'month_2', 'month_3', 
            'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 
            'month_11', 'month_12', 'num_services', 'has_streaming', 'charge_per_service', 
            'data_per_month', 'calls_per_month', 'sms_per_month', 'is_new_customer', 
            'is_long_term', 'approx_ltv', 'zip_region'
        ]
        self.feature_names_ = target_features
        return df[target_features]
