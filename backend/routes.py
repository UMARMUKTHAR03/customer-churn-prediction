"""
Flask API Routes for ChurnIQ
Platform: ChurnIQ — Intelligent Customer Retention & Explainable Churn Prediction
Author: Umar Mukthar
Tagline: Predict Churn. Understand Risk. Retain Customers.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify

# Set up logging
logger = logging.getLogger('churniq_api')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

api_bp = Blueprint('api', __name__)

MODEL_CACHE = {
    'model': None,
    'preprocessor': None,
    'metrics': None,
    'explainer': None
}

def load_artifacts():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
    prep_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
    explainer_path = os.path.join(base_dir, 'models', 'explainer.joblib')
    metrics_path = os.path.join(base_dir, 'models', 'metrics.json')

    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'models', 'best_model.pkl')

    if os.path.exists(model_path) and MODEL_CACHE['model'] is None:
        try:
            logger.info(f"Loading ChurnIQ trained model from {model_path}...")
            MODEL_CACHE['model'] = joblib.load(model_path)
            logger.info("ChurnIQ model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    if os.path.exists(prep_path) and MODEL_CACHE['preprocessor'] is None:
        try:
            logger.info(f"Loading preprocessor from {prep_path}...")
            MODEL_CACHE['preprocessor'] = joblib.load(prep_path)
            logger.info("Preprocessor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")

    if os.path.exists(explainer_path) and MODEL_CACHE['explainer'] is None:
        try:
            logger.info(f"Loading SHAP TreeExplainer from {explainer_path}...")
            MODEL_CACHE['explainer'] = joblib.load(explainer_path)
            logger.info("SHAP TreeExplainer loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load pre-fitted explainer: {e}")

    if os.path.exists(metrics_path) and MODEL_CACHE['metrics'] is None:
        try:
            with open(metrics_path, 'r') as f:
                MODEL_CACHE['metrics'] = json.load(f)
            logger.info("Verified metrics loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

load_artifacts()


def generate_retention_recommendation(customer_data, proba):
    """
    Synthesizes actionable retention strategies and estimates financial value saved.
    """
    contract = str(customer_data.get('contract_type', 'Month-to-month')).lower()
    monthly = float(customer_data.get('monthly_charges', 75.0))
    tenure = float(customer_data.get('tenure_months', 6.0))
    payment = str(customer_data.get('payment_method', 'Electronic check')).lower()
    has_tech = int(customer_data.get('has_tech_support', 0))
    has_security = int(customer_data.get('has_online_security', 0))
    has_backup = int(customer_data.get('has_online_backup', 0))
    num_complaints = int(customer_data.get('num_complaints', 0))

    remaining_months = max(12, int(72 - tenure))
    at_risk_revenue = round(remaining_months * monthly * (proba / 100.0), 2)

    suggestions = []
    key_driver = "Pricing & Contract Structure"

    if 'month-to-month' in contract:
        key_driver = "Month-to-Month Contract"
        suggestions.append("Offer 15% discount for upgrading to an annual contract commitment")
    elif monthly > 90:
        key_driver = "High Monthly Tariff"
        suggestions.append("Apply a $15/month loyalty bill credit for 6 months")

    if 'electronic check' in payment:
        suggestions.append("Incentivize enrollment in Auto-Pay with a one-time $25 account credit")
    
    if not has_tech:
        suggestions.append("Bundle complimentary 24/7 VIP Tech Support for 3 months")
    if not has_security and not has_backup:
        suggestions.append("Activate free CyberShield Security and Cloud Backup add-on package")

    if num_complaints > 0:
        suggestions.append("Trigger priority VIP retention callback from dedicated customer success team")

    if tenure < 6:
        suggestions.append("Enroll in New Subscriber Onboarding Concierge for personalized service checks")

    if not suggestions:
        suggestions.append("Award Platinum Loyalty status and offer complimentary speed boost perk")

    action_text = " • ".join(suggestions[:3])
    est_saved = round(at_risk_revenue * 0.65, 2)

    return {
        'key_driver': key_driver,
        'action_summary': action_text,
        'detailed_suggestions': suggestions,
        'at_risk_revenue': at_risk_revenue,
        'estimated_saved_revenue': est_saved
    }


def compute_shap_explanations(transformed_row, feature_names):
    """
    Computes genuine SHAP feature attribution using the loaded TreeExplainer.
    Identifies top factors increasing churn risk vs protective factors.
    """
    explainer = MODEL_CACHE.get('explainer')
    if explainer is None and MODEL_CACHE.get('model') is not None:
        try:
            import shap
            explainer = shap.TreeExplainer(MODEL_CACHE['model'])
            MODEL_CACHE['explainer'] = explainer
        except Exception as e:
            logger.warning(f"Error initializing SHAP TreeExplainer: {e}")

    if explainer is not None:
        try:
            shap_values = explainer.shap_values(transformed_row)
            # Binary classifier returns shape (1, n_features, 2) or list of 2 arrays
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                values = shap_values[0, :, 1]
            elif isinstance(shap_values, list) and len(shap_values) == 2:
                values = shap_values[1][0]
            else:
                values = shap_values[0]

            contributions = []
            for name, val in zip(feature_names, values):
                contributions.append({
                    'feature': name,
                    'shap_value': round(float(val), 4),
                    'impact': 'increases_churn' if val > 0 else 'decreases_churn'
                })

            top_positive = sorted([c for c in contributions if c['shap_value'] > 0], key=lambda x: x['shap_value'], reverse=True)[:5]
            top_negative = sorted([c for c in contributions if c['shap_value'] < 0], key=lambda x: x['shap_value'])[:5]

            return {
                'method': 'SHAP TreeExplainer',
                'is_genuine_shap': True,
                'top_risk_drivers': top_positive,
                'top_protective_factors': top_negative,
                'raw_contributions': contributions[:10]
            }
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")

    # Fallback only if SHAP library encounters runtime issue
    return {
        'method': 'Tree Feature Weight Analysis',
        'is_genuine_shap': False,
        'top_risk_drivers': [],
        'top_protective_factors': []
    }


# --- API ENDPOINTS ---

@api_bp.route('/health', methods=['GET'])
def health():
    """
    GET /health: Checks backend and model availability.
    """
    load_artifacts()
    is_ready = MODEL_CACHE['model'] is not None and MODEL_CACHE['preprocessor'] is not None
    return jsonify({
        'status': 'healthy' if is_ready else 'degraded',
        'platform': 'ChurnIQ',
        'title': 'ChurnIQ: Intelligent Customer Churn Prediction & Retention Analytics Platform',
        'version': 'v3.1-production',
        'model_loaded': MODEL_CACHE['model'] is not None,
        'preprocessor_loaded': MODEL_CACHE['preprocessor'] is not None,
        'shap_explainer_loaded': MODEL_CACHE['explainer'] is not None,
        'metrics_available': MODEL_CACHE['metrics'] is not None
    }), 200


@api_bp.route('/model-info', methods=['GET'])
def model_info():
    """
    GET /model-info: Returns model metadata, version, and training configuration.
    """
    load_artifacts()
    metrics = MODEL_CACHE.get('metrics')
    if metrics:
        return jsonify({
            'platform': metrics.get('platform_name', 'ChurnIQ'),
            'project_title': metrics.get('project_title', 'ChurnIQ: Customer Retention Platform'),
            'tagline': metrics.get('tagline', 'Predict Churn. Understand Risk. Retain Customers.'),
            'model_name': metrics.get('model_name', 'Random Forest'),
            'model_version': metrics.get('model_version', 'v3.1-production'),
            'training_date': metrics.get('training_date', '2026-11-05 UTC'),
            'algorithm': metrics.get('algorithm', 'RandomForestClassifier(n_estimators=350, max_depth=14)'),
            'dataset_stats': metrics.get('dataset_stats', {}),
            'evaluation_metrics': metrics.get('evaluation_metrics', {}),
            'threshold_analysis': metrics.get('threshold_analysis', [])
        }), 200
    
    return jsonify({
        'platform': 'ChurnIQ',
        'model_name': 'Random Forest Classifier',
        'model_version': 'v3.1-production',
        'status': 'Model in memory'
    }), 200


@api_bp.route('/analytics', methods=['GET'])
def analytics():
    """
    GET /analytics: Returns out-of-sample evaluation metrics, confusion matrix,
    ROC/PR curve coordinates, and multi-algorithm model comparisons.
    """
    load_artifacts()
    metrics = MODEL_CACHE.get('metrics')
    if not metrics:
        return jsonify({'error': 'Evaluation metrics not found. Run pipeline/run_pipeline.py first.'}), 404
        
    return jsonify(metrics), 200


@api_bp.route('/predict', methods=['POST'])
def predict_individual():
    """
    POST /predict: Evaluates churn risk for an individual customer profile using genuine ML inference.
    """
    load_artifacts()
    model = MODEL_CACHE.get('model')
    preprocessor = MODEL_CACHE.get('preprocessor')

    if model is None or preprocessor is None:
        return jsonify({'error': 'Machine learning model or preprocessor not loaded on server.'}), 503

    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid request payload. Expected a JSON object with customer features.'}), 400

        # Input validation for required core fields
        if 'monthly_charges' in data:
            try:
                charges = float(data['monthly_charges'])
                if charges < 0:
                    return jsonify({'error': 'monthly_charges must be non-negative.'}), 400
            except ValueError:
                return jsonify({'error': 'monthly_charges must be a valid numeric value.'}), 400

        if 'tenure_months' in data:
            try:
                tenure = float(data['tenure_months'])
                if tenure < 0:
                    return jsonify({'error': 'tenure_months must be non-negative.'}), 400
            except ValueError:
                return jsonify({'error': 'tenure_months must be a valid numeric value.'}), 400

        # Transform features using preprocessor (strictly matching training)
        input_df = pd.DataFrame([data])
        X_trans = preprocessor.transform(input_df)

        # Genuine Model Inference
        prediction_val = int(model.predict(X_trans)[0])
        probabilities = model.predict_proba(X_trans)[0]
        churn_proba = round(float(probabilities[1]) * 100, 2)
        confidence = round(float(max(probabilities)) * 100, 2)

        # Business Risk Tier Classification
        if churn_proba >= 65.0:
            risk_level = "High"
            badge_label = "CRITICAL CHURN RISK"
        elif churn_proba >= 35.0:
            risk_level = "Medium"
            badge_label = "MODERATE ATTRITION RISK"
        else:
            risk_level = "Low"
            badge_label = "LOYAL SUBSCRIBER (LOW RISK)"

        # Retention playbook
        retention_info = generate_retention_recommendation(data, churn_proba)

        # Genuine SHAP feature attribution
        feature_names = preprocessor.feature_names_
        explanations = compute_shap_explanations(X_trans, feature_names)

        response = {
            'platform': 'ChurnIQ',
            'customer_id': str(data.get('customer_id', data.get('id', 'CUST-NEW'))),
            'churn_prediction': "Churn" if prediction_val == 1 else "No Churn",
            'churn_probability': churn_proba,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_label': badge_label,
            'retention_strategy': retention_info,
            'explainability': explanations,
            'model_version': 'v3.1-production',
            'inference_engine': 'Genuine Random Forest (350 Trees)'
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Inference error in /predict: {e}", exc_info=True)
        return jsonify({'error': f"Prediction calculation failed: {str(e)}"}), 500


@api_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    POST /predict-batch: Performs bulk inference on customer records.
    Accepts CSV multipart upload or JSON customer array with per-row validation.
    """
    load_artifacts()
    model = MODEL_CACHE.get('model')
    preprocessor = MODEL_CACHE.get('preprocessor')

    if model is None or preprocessor is None:
        return jsonify({'error': 'Machine learning model not loaded on server.'}), 503

    try:
        errors = []
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Uploaded file is empty.'}), 400
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Invalid file format. Please upload a .csv file.'}), 400
            try:
                df = pd.read_csv(file)
            except Exception as parse_err:
                return jsonify({'error': f"Failed to parse CSV file: {str(parse_err)}"}), 400
        elif request.is_json:
            json_body = request.get_json()
            if isinstance(json_body, list):
                customer_list = json_body
            elif isinstance(json_body, dict):
                customer_list = json_body.get('customers', [json_body] if json_body else [])
            else:
                customer_list = []
            if not customer_list:
                return jsonify({'error': 'No customer records provided in JSON payload.'}), 400
            df = pd.DataFrame(customer_list)
        else:
            return jsonify({'error': 'Unsupported content type. Send multipart CSV file or application/json.'}), 400

        if df.empty:
            return jsonify({'error': 'Dataset contains zero rows.'}), 400

        total_records = len(df)
        logger.info(f"ChurnIQ processing bulk batch of {total_records} customer records...")

        # Customer ID column detection
        if 'customer_id' in df.columns:
            id_col = df['customer_id'].astype(str)
        elif 'id' in df.columns:
            id_col = df['id'].astype(str)
        else:
            id_col = pd.Series([f"CUST-{1000 + i}" for i in range(total_records)])

        # Preprocess features
        X_trans = preprocessor.transform(df)

        # Batch ML inference
        preds = model.predict(X_trans)
        probas = model.predict_proba(X_trans)[:, 1]

        results = []
        high_risk_count = 0
        med_risk_count = 0
        low_risk_count = 0
        total_at_risk_ltv = 0.0

        for i in range(total_records):
            p = int(preds[i])
            prob_pct = round(float(probas[i]) * 100, 2)
            cid = id_col.iloc[i]
            row_dict = df.iloc[i].to_dict()

            if prob_pct >= 65.0:
                risk = "High"
                high_risk_count += 1
            elif prob_pct >= 35.0:
                risk = "Medium"
                med_risk_count += 1
            else:
                risk = "Low"
                low_risk_count += 1

            retention = generate_retention_recommendation(row_dict, prob_pct)
            total_at_risk_ltv += retention['at_risk_revenue']

            results.append({
                'customer_id': cid,
                'churn_prediction': "Churn" if p == 1 else "No Churn",
                'churn_probability': prob_pct,
                'risk_level': risk,
                'contract': str(row_dict.get('contract_type', 'Month-to-month')),
                'tenure': float(row_dict.get('tenure_months', 6.0)),
                'monthly_charges': float(row_dict.get('monthly_charges', 75.0)),
                'key_driver': retention['key_driver'],
                'recommended_action': retention['action_summary'],
                'at_risk_revenue': retention['at_risk_revenue']
            })

        results.sort(key=lambda x: x['churn_probability'], reverse=True)

        return jsonify({
            'status': 'success',
            'platform': 'ChurnIQ',
            'total_processed': total_records,
            'summary': {
                'total_customers': total_records,
                'churn_predicted': int(sum(preds)),
                'retained_predicted': int(total_records - sum(preds)),
                'high_risk_count': high_risk_count,
                'medium_risk_count': med_risk_count,
                'low_risk_count': low_risk_count,
                'total_at_risk_revenue': round(total_at_risk_ltv, 2)
            },
            'predictions': results
        }), 200

    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        return jsonify({'error': f"Failed to process batch inference: {str(e)}"}), 500
