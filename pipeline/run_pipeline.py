"""
Master Automated Machine Learning Pipeline Runner
Executes the full end-to-end data processing, feature engineering, model training,
multi-algorithm benchmarking, and artifact generation in one click.
"""

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.clean import clean_dataset
from pipeline.encode import encode_features
from pipeline.features import engineer_features
from pipeline.train import train_and_evaluate

def run_master_pipeline():
    print("=" * 70)
    print("🚀 TeleChurn AI — Master Machine Learning Pipeline Execution")
    print("=" * 70)
    start_total = time.time()

    raw_train = 'data/raw/telecom_churn_train.csv'
    cleaned_train = 'data/processed/telecom_churn_train_cleaned.csv'
    encoded_train = 'data/processed/telecom_churn_train_encoded.csv'
    features_train = 'data/processed/telecom_churn_train_features.csv'

    # Stage 1: Clean
    print("\n[STAGE 1/4] Data Audit & Missing Value Imputation...")
    clean_start = time.time()
    clean_df = clean_dataset(raw_train, cleaned_train)
    print(f"✓ Cleaned dataset saved ({len(clean_df)} records) in {time.time() - clean_start:.2f}s")

    # Stage 2: Encode
    print("\n[STAGE 2/4] Categorical Feature Transformation...")
    encode_start = time.time()
    encoded_df = encode_features(cleaned_train, encoded_train)
    print(f"✓ Encoded dataset saved ({encoded_df.shape[1]} columns) in {time.time() - encode_start:.2f}s")

    # Stage 3: Feature Engineering
    print("\n[STAGE 3/4] Domain Feature Synthesis & Ratio Derivation...")
    feat_start = time.time()
    features_df = engineer_features(encoded_train, features_train)
    print(f"✓ Final feature matrix saved ({features_df.shape[1]} features) in {time.time() - feat_start:.2f}s")

    # Stage 4: Model Training & Evaluation
    print("\n[STAGE 4/4] Multi-Algorithm Training, Tuning & Evaluation...")
    train_start = time.time()
    metrics = train_and_evaluate(
        raw_train_path=raw_train,
        model_save_path='models/best_model.joblib',
        preprocessor_save_path='models/preprocessor.joblib',
        metrics_save_path='models/metrics.json'
    )
    print(f"✓ Model training & benchmarking complete in {time.time() - train_start:.2f}s")

    total_time = round(time.time() - start_total, 2)
    print("\n" + "=" * 70)
    print(f"🎉 Full Pipeline Completed in {total_time} seconds!")
    print(f"   Model Name:        {metrics['model_name']} ({metrics['model_version']})")
    print(f"   Holdout Accuracy:  {metrics['evaluation_metrics']['accuracy']*100:.2f}%")
    print(f"   Holdout Precision: {metrics['evaluation_metrics']['precision']*100:.2f}%")
    print(f"   Holdout Recall:    {metrics['evaluation_metrics']['recall']*100:.2f}%")
    print(f"   Holdout F1 Score:  {metrics['evaluation_metrics']['f1_score']:.4f}")
    print(f"   Holdout ROC-AUC:   {metrics['evaluation_metrics']['roc_auc']:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    run_master_pipeline()
