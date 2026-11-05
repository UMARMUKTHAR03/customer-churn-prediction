"""
Unit Tests for Data Preprocessing and Feature Engineering Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pipeline.preprocessor import TelecomDataPreprocessor
from pipeline.clean import clean_dataset
from pipeline.encode import encode_features
from pipeline.features import engineer_features

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame([
        {
            'age': 45.0,
            'income': np.nan,  # Missing
            'zip_code': 94105,
            'tenure_months': 4.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'monthly_charges': 85.50,
            'total_charges': np.nan,  # Missing
            'has_internet': 1,
            'has_online_security': 0,
            'has_online_backup': 1,
            'has_device_protection': 0,
            'has_tech_support': 0,
            'has_streaming_tv': 1,
            'has_streaming_movies': 1,
            'has_phone': 1,
            'has_multiple_lines': 1,
            'num_support_calls': 3,
            'num_complaints': 1,
            'avg_call_duration_min': 4.2,
            'nps_score': np.nan,  # Missing
            'data_usage_gb': 120.0,
            'minutes_used': 350.0,
            'sms_sent': 45,
            'competitor_price_diff': 5.0,
            'signup_month': 3,
            'last_interaction_days': 12.0
        },
        {
            'age': 62.0,
            'income': 78000.0,
            'zip_code': 10001,
            'tenure_months': 36.0,
            'contract_type': '2-year',
            'payment_method': 'Credit card',
            'monthly_charges': 60.0,
            'total_charges': 2160.0,
            'has_internet': 1,
            'has_online_security': 1,
            'has_online_backup': 1,
            'has_device_protection': 1,
            'has_tech_support': 1,
            'has_streaming_tv': 0,
            'has_streaming_movies': 0,
            'has_phone': 1,
            'has_multiple_lines': 0,
            'num_support_calls': 0,
            'num_complaints': 0,
            'avg_call_duration_min': 2.5,
            'nps_score': 9.0,
            'data_usage_gb': 80.0,
            'minutes_used': 200.0,
            'sms_sent': 20,
            'competitor_price_diff': -10.0,
            'signup_month': 11,
            'last_interaction_days': 45.0
        }
    ])

def test_preprocessor_fit_transform(sample_raw_data):
    prep = TelecomDataPreprocessor()
    transformed = prep.fit_transform(sample_raw_data)
    
    # Check that 53 features are output
    assert transformed.shape[1] == 53
    assert transformed.shape[0] == 2
    
    # Check that no NaN values remain
    assert transformed.isnull().sum().sum() == 0
    
    # Check key engineered features
    assert 'num_services' in transformed.columns
    assert 'charge_per_service' in transformed.columns
    assert 'approx_ltv' in transformed.columns
    assert 'is_new_customer' in transformed.columns

def test_preprocessor_single_record():
    prep = TelecomDataPreprocessor()
    dummy_train = pd.DataFrame([{'tenure_months': 12.0, 'monthly_charges': 50.0, 'zip_code': 500}])
    prep.fit(dummy_train)
    
    single = {
        'contract_type': 'One year',
        'monthly_charges': 70.0,
        'tenure_months': 10.0
    }
    result = prep.transform(single)
    assert result.shape == (1, 53)
    assert result['contract_1-year'].iloc[0] == 1
    assert result['contract_Month-to-month'].iloc[0] == 0

def test_cleaning_deduplication(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    out_csv = tmp_path / "cleaned.csv"
    df = pd.DataFrame([
        {'age': 30, 'tenure_months': 5, 'income': 40000},
        {'age': 30, 'tenure_months': 5, 'income': 40000},  # Duplicate
        {'age': 40, 'tenure_months': 15, 'income': np.nan}
    ])
    df.to_csv(raw_csv, index=False)
    
    cleaned = clean_dataset(str(raw_csv), str(out_csv))
    assert len(cleaned) == 2  # Duplicate removed
    assert cleaned['income'].isnull().sum() == 0  # Imputed
