"""
ChurnIQ — Machine Learning Training, Benchmarking & Evaluation Pipeline
Platform: ChurnIQ Enterprise Retention Intelligence
Author: Umar Mukthar

Performs:
1. Data deduplication and zero-leakage preprocessing fitting.
2. 5-Fold Stratified Cross-Validation across 5 algorithms:
   - Logistic Regression
   - Decision Tree
   - Random Forest (Selected Primary)
   - Gradient Boosting
   - XGBoost
3. Holdout test set evaluation (Accuracy, Precision, Recall, F1, ROC-AUC).
4. Exact ROC Curve & Precision-Recall Curve coordinates.
5. Confusion matrix calculation (TN, FP, FN, TP).
6. Global feature importance and SHAP TreeExplainer serialization.
7. Business decision threshold calibration.
"""

import os
import json
import time
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)
import shap

from pipeline.preprocessor import TelecomDataPreprocessor

def train_and_evaluate(
    raw_train_path='data/raw/telecom_churn_train.csv',
    model_save_path='models/best_model.joblib',
    preprocessor_save_path='models/preprocessor.joblib',
    explainer_save_path='models/explainer.joblib',
    metrics_save_path='models/metrics.json'
):
    start_time = time.time()
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    print("=" * 70)
    print("🚀 ChurnIQ — Model Training & Independent Holdout Benchmark")
    print("=" * 70)

    # 1. Load & Deduplicate
    raw_df = pd.read_csv(raw_train_path)
    initial_rows = len(raw_df)
    df = raw_df.drop_duplicates().reset_index(drop=True)
    dedup_rows = len(df)
    print(f"Dataset Audit: {initial_rows} raw rows -> {dedup_rows} unique rows ({initial_rows - dedup_rows} duplicates removed).")

    X = df.drop('churned', axis=1)
    y = df['churned'].astype(int)

    # 2. Stratified Holdout Test Split (80% Train, 20% Unseen Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Data Partition: {len(X_train)} Train samples | {len(X_test)} Independent Holdout Test samples.")

    # 3. Fit Preprocessing Pipeline strictly on X_train (Zero Data Leakage)
    print("Fitting TelecomDataPreprocessor on training split...")
    preprocessor = TelecomDataPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Save processed features for inspection
    train_features_df = X_train_trans.copy()
    train_features_df['churned'] = y_train.values
    train_features_df.to_csv('data/processed/telecom_churn_train_features.csv', index=False)

    # Save fitted preprocessor
    joblib.dump(preprocessor, preprocessor_save_path)
    print(f"Saved preprocessor to {preprocessor_save_path}")

    # Standard Scaler for Logistic Regression convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_trans)
    X_test_scaled = scaler.transform(X_test_trans)

    # 4. Multi-Algorithm Benchmark & 5-Fold Stratified Cross-Validation
    candidate_models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, C=0.5, random_state=42),
            X_train_scaled,
            X_test_scaled
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42),
            X_train_trans,
            X_test_trans
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=350, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1),
            X_train_trans,
            X_test_trans
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42),
            X_train_trans,
            X_test_trans
        ),
        'XGBoost': (
            XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss'),
            X_train_trans,
            X_test_trans
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_table = []
    trained_estimators = {}

    print("\nRunning 5-Fold Stratified Cross-Validation and Holdout Evaluation...")
    for name, (clf, x_tr, x_te) in candidate_models.items():
        # Cross-validation on train split
        cv_scores = cross_validate(
            clf, x_tr, y_train, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )

        # Fit on full train split and test on holdout set
        clf.fit(x_tr, y_train)
        trained_estimators[name] = clf

        preds = clf.predict(x_te)
        proba = clf.predict_proba(x_te)[:, 1] if hasattr(clf, 'predict_proba') else preds

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, proba))

        cv_acc_mean = float(np.mean(cv_scores['test_accuracy']))
        cv_acc_std = float(np.std(cv_scores['test_accuracy']))

        comparison_table.append({
            'algorithm': name,
            'holdout_accuracy': round(acc, 4),
            'holdout_precision': round(prec, 4),
            'holdout_recall': round(rec, 4),
            'holdout_f1': round(f1, 4),
            'holdout_roc_auc': round(auc, 4),
            'cv_accuracy_mean': round(cv_acc_mean, 4),
            'cv_accuracy_std': round(cv_acc_std, 4)
        })
        print(f"  {name:22s} | Holdout Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f} | CV Acc: {cv_acc_mean*100:.2f}% ± {cv_acc_std*100:.2f}%")

    # Select Best Model: Random Forest (Production Primary)
    best_name = 'Random Forest'
    best_model = trained_estimators[best_name]

    # Evaluate best model thoroughly on holdout test set
    y_pred = best_model.predict(X_test_trans)
    y_proba = best_model.predict_proba(X_test_trans)[:, 1]

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    # Dynamic ROC Curve Points
    fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
    step_roc = max(1, len(fpr) // 30)
    roc_points = [{'fpr': round(float(f), 4), 'tpr': round(float(t), 4)} for f, t in zip(fpr[::step_roc], tpr[::step_roc])]
    if roc_points[-1]['fpr'] != 1.0:
        roc_points.append({'fpr': 1.0, 'tpr': 1.0})

    # Dynamic Precision-Recall Curve Points
    pr_precision, pr_recall, pr_thresh = precision_recall_curve(y_test, y_proba)
    step_pr = max(1, len(pr_precision) // 30)
    pr_points = [{'recall': round(float(r), 4), 'precision': round(float(p), 4)} for r, p in zip(pr_recall[::step_pr], pr_precision[::step_pr])]

    # Gini Feature Importances
    feature_names = preprocessor.feature_names_
    importances = best_model.feature_importances_
    feat_imp = sorted(
        [{'feature': fn, 'importance': round(float(imp), 4)} for fn, imp in zip(feature_names, importances)],
        key=lambda x: x['importance'],
        reverse=True
    )

    # Churn probability distribution buckets
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    test_df_eval = pd.DataFrame({'proba': y_proba, 'actual': y_test.values})
    test_df_eval['bucket'] = pd.cut(test_df_eval['proba'], bins=bins, labels=labels, include_lowest=True)
    
    dist_retained = test_df_eval[test_df_eval['actual'] == 0]['bucket'].value_counts().reindex(labels, fill_value=0).tolist()
    dist_churned = test_df_eval[test_df_eval['actual'] == 1]['bucket'].value_counts().reindex(labels, fill_value=0).tolist()

    # Threshold Optimization Analysis
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    threshold_analysis = []
    for t in thresholds:
        t_preds = (y_proba >= t).astype(int)
        t_prec = float(precision_score(y_test, t_preds, zero_division=0))
        t_rec = float(recall_score(y_test, t_preds, zero_division=0))
        t_f1 = float(f1_score(y_test, t_preds, zero_division=0))
        threshold_analysis.append({
            'threshold': t,
            'precision': round(t_prec, 4),
            'recall': round(t_rec, 4),
            'f1_score': round(t_f1, 4)
        })

    # Fit and Save Genuine SHAP TreeExplainer
    print("Fitting and saving SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(best_model)
    joblib.dump(explainer, explainer_save_path)
    print(f"Saved SHAP TreeExplainer to {explainer_save_path}")

    # Compute Global SHAP Feature Importances on sample
    sample_X = X_test_trans.iloc[:150]
    shap_sample = explainer.shap_values(sample_X)
    if isinstance(shap_sample, np.ndarray) and shap_sample.ndim == 3:
        mean_abs_shap = np.mean(np.abs(shap_sample[:, :, 1]), axis=0)
    elif isinstance(shap_sample, list) and len(shap_sample) == 2:
        mean_abs_shap = np.mean(np.abs(shap_sample[1]), axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(shap_sample), axis=0)

    shap_global = sorted(
        [{'feature': fn, 'mean_shap_value': round(float(ms), 4)} for fn, ms in zip(feature_names, mean_abs_shap)],
        key=lambda x: x['mean_shap_value'],
        reverse=True
    )

    # Assemble Metrics Payload
    execution_time = round(time.time() - start_time, 2)
    metrics_payload = {
        'platform_name': 'ChurnIQ',
        'project_title': 'ChurnIQ: An Explainable AI-Powered Customer Churn Prediction and Retention Intelligence Platform',
        'tagline': 'Predict Churn. Understand Risk. Retain Customers.',
        'model_name': best_name,
        'model_version': 'v3.1-production',
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'algorithm': 'RandomForestClassifier(n_estimators=350, max_depth=14, criterion="gini")',
        'execution_time_sec': execution_time,
        'dataset_stats': {
            'raw_records': initial_rows,
            'deduplicated_records': dedup_rows,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': len(feature_names),
            'churn_ratio_train': round(float(y_train.mean()), 4),
            'churn_ratio_test': round(float(y_test.mean()), 4)
        },
        'evaluation_metrics': {
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'roc_auc': round(float(roc_auc_score(y_test, y_proba)), 4)
        },
        'confusion_matrix': {
            'matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        },
        'model_comparison': comparison_table,
        'top_features': feat_imp[:15],
        'shap_global_importance': shap_global[:15],
        'threshold_analysis': threshold_analysis,
        'distribution': {
            'labels': labels,
            'retained': dist_retained,
            'churned': dist_churned
        },
        'roc_curve': roc_points,
        'pr_curve': pr_points,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    # Save artifacts
    print(f"Saving model to {model_save_path}...")
    joblib.dump(best_model, model_save_path)

    # Save backwards-compatible pickle
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Save metrics JSON
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved verified evaluation metrics to {metrics_save_path}")

    # Generate predictions for raw test set (telecom_churn_test.csv)
    raw_test_file = 'data/raw/telecom_churn_test.csv'
    if os.path.exists(raw_test_file):
        test_raw = pd.read_csv(raw_test_file)
        test_trans = preprocessor.transform(test_raw)
        preds_test = best_model.predict(test_trans)
        proba_test = best_model.predict_proba(test_trans)[:, 1]
        
        preds_output = test_raw.copy()
        preds_output['churn_prediction'] = ['Churn' if p == 1 else 'No Churn' for p in preds_test]
        preds_output['churn_probability'] = np.round(proba_test * 100, 2)
        preds_output['risk_level'] = [
            'High' if prob >= 65 else ('Medium' if prob >= 35 else 'Low')
            for prob in preds_output['churn_probability']
        ]
        preds_output.to_csv('data/processed/final_predictions.csv', index=False)
        print(f"Generated test batch predictions -> data/processed/final_predictions.csv ({len(preds_output)} records).")

    print(f"Pipeline executed successfully in {execution_time} seconds!")
    return metrics_payload

if __name__ == '__main__':
    train_and_evaluate()
