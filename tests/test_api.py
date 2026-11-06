"""
Integration Tests for Flask REST API Endpoints
"""

import io
import json
import pytest
from backend.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get('/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True

def test_model_info_endpoint(client):
    res = client.get('/model-info')
    assert res.status_code == 200
    data = res.get_json()
    assert 'model_name' in data
    assert 'model_version' in data

def test_analytics_endpoint(client):
    res = client.get('/analytics')
    assert res.status_code == 200
    data = res.get_json()
    assert 'evaluation_metrics' in data
    assert 'confusion_matrix' in data
    assert 'model_comparison' in data
    assert 'roc_curve' in data

def test_predict_single_endpoint(client):
    sample = {
        'customer_id': 'UNIT-001',
        'contract_type': 'Month-to-month',
        'tenure_months': 3.0,
        'monthly_charges': 95.0,
        'payment_method': 'Electronic check',
        'has_tech_support': 0,
        'has_online_security': 0
    }
    res = client.post('/predict', json=sample)
    assert res.status_code == 200
    data = res.get_json()
    assert 'churn_prediction' in data
    assert 'churn_probability' in data
    assert 'risk_level' in data
    assert 'retention_strategy' in data
    assert 'explainability' in data
    assert 0.0 <= data['churn_probability'] <= 100.0

def test_predict_single_missing_body(client):
    res = client.post('/predict', data="", content_type='application/json')
    assert res.status_code == 400
    data = res.get_json()
    assert 'error' in data

def test_predict_batch_json(client):
    batch = {
        'customers': [
            {'customer_id': 'C-1', 'contract_type': 'Month-to-month', 'tenure_months': 1, 'monthly_charges': 110},
            {'customer_id': 'C-2', 'contract_type': '2-year', 'tenure_months': 48, 'monthly_charges': 40}
        ]
    }
    res = client.post('/predict-batch', json=batch)
    assert res.status_code == 200
    data = res.get_json()
    assert data['total_processed'] == 2
    assert len(data['predictions']) == 2
    assert 'summary' in data

def test_predict_batch_csv(client):
    csv_content = (
        "customer_id,contract_type,tenure_months,monthly_charges,payment_method\n"
        "CSV-001,Month-to-month,2,105,Electronic check\n"
        "CSV-002,2-year,60,50,Credit card\n"
    )
    data = {
        'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_batch.csv')
    }
    res = client.post('/predict-batch', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data['total_processed'] == 2
    assert json_data['predictions'][0]['customer_id'] == 'CSV-001'

def test_404_handler(client):
    res = client.get('/non-existent-endpoint-xyz')
    assert res.status_code == 404
    data = res.get_json()
    assert 'error' in data

def test_shap_in_api(client):
    res = client.post('/predict', json={
        'customer_id': 'SHAP-TEST-1',
        'contract_type': 'Month-to-month',
        'tenure_months': 1,
        'monthly_charges': 115.0,
        'payment_method': 'Electronic check'
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'explainability' in data
    assert data['explainability']['is_genuine_shap'] is True
    assert 'top_risk_drivers' in data['explainability']
    assert len(data['explainability']['top_risk_drivers']) > 0

def test_predict_batch_raw_array(client):
    raw_array = [
        {'customer_id': 'ARR-1', 'contract_type': 'Two year', 'tenure_months': 60, 'monthly_charges': 35.0},
        {'customer_id': 'ARR-2', 'contract_type': 'Month-to-month', 'tenure_months': 1, 'monthly_charges': 115.0}
    ]
    res = client.post('/predict-batch', json=raw_array)
    assert res.status_code == 200
    data = res.get_json()
    assert data['total_processed'] == 2
    assert data['summary']['total_customers'] == 2

def test_predict_validation_invalid_numeric(client):
    # Test negative monthly charges
    res = client.post('/predict', json={'monthly_charges': -10.0, 'tenure_months': 12})
    assert res.status_code == 400
    assert 'error' in res.get_json()

    # Test negative tenure
    res2 = client.post('/predict', json={'monthly_charges': 50.0, 'tenure_months': -5})
    assert res2.status_code == 400
    assert 'error' in res2.get_json()

def test_whatif_scenario_re_inference(client):
    # Baseline profile
    baseline_payload = {
        'customer_id': 'WHATIF-BASE',
        'contract_type': 'Month-to-month',
        'tenure_months': 2,
        'monthly_charges': 105.0,
        'has_tech_support': 0,
        'payment_method': 'Electronic check'
    }
    res_base = client.post('/predict', json=baseline_payload)
    assert res_base.status_code == 200
    base_data = res_base.get_json()

    # Counterfactual intervention
    scenario_payload = baseline_payload.copy()
    scenario_payload['contract_type'] = '1-year'
    scenario_payload['monthly_charges'] = 89.25
    scenario_payload['has_tech_support'] = 1
    scenario_payload['payment_method'] = 'Credit card'

    res_scenario = client.post('/predict', json=scenario_payload)
    assert res_scenario.status_code == 200
    scenario_data = res_scenario.get_json()

    # Genuine re-inference should exhibit altered risk and valid probability
    assert scenario_data['churn_probability'] < base_data['churn_probability']
    assert scenario_data['risk_level'] in ['High', 'Medium', 'Low']
    assert 'recovery_basis' in scenario_data['retention_strategy']

def test_batch_row_level_cleansing_and_robustness(client):
    # Batch containing rows with invalid/missing numeric fields to test safe row-level imputation
    batch = {
        'customers': [
            {'customer_id': 'CLEAN-1', 'contract_type': 'Month-to-month', 'tenure_months': -2, 'monthly_charges': 95.0},
            {'customer_id': 'CLEAN-2', 'contract_type': '1-year', 'tenure_months': 24, 'monthly_charges': 'invalid_charge_string'},
            {'customer_id': 'CLEAN-3', 'contract_type': '2-year', 'tenure_months': 48, 'monthly_charges': 65.0}
        ]
    }
    res = client.post('/predict-batch', json=batch)
    assert res.status_code == 200
    data = res.get_json()
    assert data['total_processed'] == 3
    assert len(data['predictions']) == 3

