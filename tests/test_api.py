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

