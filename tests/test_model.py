"""
Unit Tests for Machine Learning Model Loading and Inference
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def model_and_prep():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
    prep_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
    
    assert os.path.exists(model_path), "Trained model does not exist at models/best_model.joblib"
    assert os.path.exists(prep_path), "Preprocessor does not exist at models/preprocessor.joblib"
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    return model, preprocessor

def test_model_loaded(model_and_prep):
    model, preprocessor = model_and_prep
    assert model is not None
    assert preprocessor is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')

def test_prediction_output(model_and_prep):
    model, preprocessor = model_and_prep
    sample = {
        'contract_type': 'Month-to-month',
        'tenure_months': 2.0,
        'monthly_charges': 110.0,
        'payment_method': 'Electronic check',
        'has_tech_support': 0,
        'has_online_security': 0
    }
    X_trans = preprocessor.transform(pd.DataFrame([sample]))
    
    pred = model.predict(X_trans)
    proba = model.predict_proba(X_trans)
    
    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0][1] <= 1.0
    assert np.isclose(proba[0][0] + proba[0][1], 1.0)

def test_prediction_consistency(model_and_prep):
    model, preprocessor = model_and_prep
    sample = {
        'contract_type': '2-year',
        'tenure_months': 60.0,
        'monthly_charges': 45.0,
        'payment_method': 'Credit card',
        'has_tech_support': 1,
        'has_online_security': 1
    }
    X_trans1 = preprocessor.transform(pd.DataFrame([sample]))
    X_trans2 = preprocessor.transform(pd.DataFrame([sample]))
    
    prob1 = model.predict_proba(X_trans1)[0][1]
    prob2 = model.predict_proba(X_trans2)[0][1]
    
    assert prob1 == prob2, "Predictions for identical inputs must be strictly consistent."

def test_shap_explainer_artifact(model_and_prep):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    explainer_path = os.path.join(base_dir, 'models', 'explainer.joblib')
    assert os.path.exists(explainer_path), "SHAP explainer artifact must exist"
    explainer = joblib.load(explainer_path)
    assert explainer is not None
    model, preprocessor = model_and_prep
    sample = {
        'contract_type': 'Month-to-month',
        'tenure_months': 2.0,
        'monthly_charges': 110.0,
        'payment_method': 'Electronic check'
    }
    X_trans = preprocessor.transform(pd.DataFrame([sample]))
    shap_vals = explainer.shap_values(X_trans)
    assert shap_vals is not None


def test_artifact_and_metadata_consistency():
    import json
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
    prep_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
    explainer_path = os.path.join(base_dir, 'models', 'explainer.joblib')
    metrics_path = os.path.join(base_dir, 'models', 'metrics.json')

    assert os.path.exists(model_path)
    assert os.path.exists(prep_path)
    assert os.path.exists(explainer_path)
    assert os.path.exists(metrics_path)

    model = joblib.load(model_path)
    explainer = joblib.load(explainer_path)
    preprocessor = joblib.load(prep_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Verify model class aligns with metrics metadata
    assert metrics['model_name'] == 'Gradient Boosting'
    assert 'GradientBoostingClassifier' in type(model).__name__
    assert metrics['inference_engine'] == 'Genuine Gradient Boosting'
    assert metrics['model_version'] == 'v3.1-production'
    assert 'GradientBoostingClassifier' in metrics['algorithm']
    assert len(preprocessor.feature_names_) == metrics['dataset_stats']['total_features']

