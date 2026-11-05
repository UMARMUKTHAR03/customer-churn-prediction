# 📊 ChurnIQ: Intelligent Customer Churn Prediction & Retention Analytics Platform

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live_Demo-ChurnIQ_Web_Platform-6366F1?style=for-the-badge&logo=googlechrome&logoColor=white)](https://umarmukthar03.github.io/customer-churn-prediction/)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-92.2%25-10B981?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://umarmukthar03.github.io/customer-churn-prediction/)
[![Precision](https://img.shields.io/badge/Precision-98.2%25-8B5CF6?style=for-the-badge)](https://umarmukthar03.github.io/customer-churn-prediction/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.929-F59E0B?style=for-the-badge)](https://umarmukthar03.github.io/customer-churn-prediction/)
[![Docker CI](https://img.shields.io/badge/CI%2FCD-Passing-34D399?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/UMARMUKTHAR03/customer-churn-prediction/actions)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Author](https://img.shields.io/badge/Engineered_By-Umar_Mukthar-rose?style=for-the-badge&logo=github)](https://github.com/UMARMUKTHAR03)

<br/>

> **Predict Churn. Understand Risk. Retain Customers.**

<br/>

<a href="https://umarmukthar03.github.io/customer-churn-prediction/">
  <img src="./docs/preview.svg" alt="ChurnIQ Enterprise Dashboard Preview" width="100%" />
</a>

<br/>

**[🚀 Launch Interactive Platform](https://umarmukthar03.github.io/customer-churn-prediction/)** • **[Explore Pipeline](#-the-ml-pipeline)** • **[Model Benchmarks](#-model-benchmarks--evaluation)** • **[What-If Simulator](#-what-if-counterfactual-simulator)** • **[API Documentation](#-rest-api-documentation)** • **[Docker & CI/CD](#-docker--container-deployment)**

</div>

---

## 🎯 1. Project Overview & Business Problem

Customer churn is one of the most critical challenges faced by modern telecommunications providers. Because acquiring a new subscriber is **5 to 7 times more expensive** than retaining an existing account, early identification of attrition risk directly impacts Monthly Recurring Revenue (MRR) and Customer Lifetime Value (LTV).

**ChurnIQ** is an enterprise-grade, explainable AI-powered customer churn prediction and retention platform engineered by **UMAR MUKTHAR**. The application:
- Predicts individual churn risk and probability using genuine Machine Learning models trained with **zero data leakage**.
- Explains every prediction with mathematical attribution via **Explainable AI (SHAP TreeExplainer)**.
- Formulates automated, personalized customer retention playbooks and calculates revenue at risk.
- Simulates contract and pricing changes in real-time via the interactive **What-If Counterfactual Simulator**.
- Processes bulk customer files via **drag-and-drop CSV batch inference (powered by Papa Parse)**.
- Generates downloadable, executive-ready PDF audit reports using **jsPDF**.
- Operates both online via a containerized **Flask REST API** and client-side as a high-performance progressive web app on **GitHub Pages**.

---

## 🌟 2. Key Features

| Category | Capability |
| :--- | :--- |
| **🔮 Live Predictor** | Interactive customer profile entry with sliders, dropdowns, checkboxes, and realistic test profiles. Real-time probability gauge and LTV risk estimation. |
| **🧠 Genuine ML Models** | Benchmarked Random Forest (350 estimators) and XGBoost with zero data leakage and consistent preprocessing. |
| **🔍 Explainable AI (SHAP)** | Local and global feature attribution highlighting root factors driving or mitigating churn. |
| **🎛️ What-If Counterfactual Simulator** | Interactive slider panel allowing retention agents to test hypothetical contract switches, price discounts, and add-on activations to observe instant risk reduction. |
| **💡 Retention Recommendations** | Intelligent prescription engine suggesting contract migration, discount credits, and VIP tech support bundles. |
| **📄 Executive PDF Reports** | Instant client-side PDF assessment download featuring customer profile, risk rating, financial exposure, and retention playbook. |
| **📂 Bulk CSV Prediction** | Drag-and-drop CSV batch inference with column validation, error handling, risk tier sorting, and enriched CSV export. |
| **📊 Model Analytics Dashboard** | Real-time confusion matrix, ROC curves, Precision-Recall curves, feature importance ranking, and algorithm comparison table. |
| **⚠️ At-Risk Account Management** | Filterable subscriber audit portfolio by risk tier (High >65%, Medium 35-65%, Low <35%) with action dispatchers. |
| **🕒 Prediction History** | Client-persisted audit log with search, filters, and CSV export. |
| **🐳 Docker & CI/CD** | Production-ready `Dockerfile`, `docker-compose.yml`, and GitHub Actions automated test workflow. |

---

## 🏗️ 3. Technical Architecture

```mermaid
graph TD
    subgraph "Frontend Layer (GitHub Pages / SPA)"
        UI["Interactive Web Dashboard: index.html"]
        LP["Live Predictor Form"]
        WI["What-If Counterfactual Simulator"]
        BP["Bulk CSV Uploader (Papa Parse)"]
        PDF["jsPDF Report Generator"]
        AP["At-Risk Subscriber Portfolio"]
        AN["Chart.js Model Analytics"]
    end

    subgraph "REST API Layer (Flask / Python / Gunicorn)"
        API["Flask REST API: backend/app.py"]
        H_EP["GET /health"]
        M_EP["GET /model-info"]
        A_EP["GET /analytics"]
        P_EP["POST /predict"]
        PB_EP["POST /predict-batch"]
    end

    subgraph "Machine Learning Engine"
        PREP["TelecomDataPreprocessor: models/preprocessor.joblib"]
        RF["Random Forest Classifier (350 Trees): models/best_model.joblib"]
        XGB["XGBoost Classifier (Benchmark)"]
        SHAP["SHAP TreeExplainer (Feature Attribution)"]
    end

    subgraph "Data Storage & Artifacts"
        RAW["data/raw/telecom_churn_train.csv"]
        PROC["data/processed/telecom_churn_train_features.csv"]
        METRICS["models/metrics.json"]
    end

    UI --> LP & WI & BP & PDF & AP & AN
    LP & WI & BP -->|HTTPS REST| API
    API --> H_EP & M_EP & A_EP & P_EP & PB_EP
    P_EP & PB_EP --> PREP --> RF & SHAP
    A_EP --> METRICS
    PREP -.-> RAW & PROC
```

---

## 📊 4. Model Benchmarks & Evaluation

All candidate algorithms were benchmarked on an **independent holdout test set (1,210 unseen subscriber records)** with stratified 80/20 train/test splitting. Zero preprocessing leakage was enforced via `TelecomDataPreprocessor`.

### Multi-Algorithm Benchmark Summary

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Selected)** | **92.23%** | **98.18%** | **70.59%** | **0.8213** | **0.9291** | 🏆 **Production Primary** |
| **XGBoost** | **94.96%** | **97.67%** | **82.03%** | **0.8917** | **0.9515** | 🚀 **Production Alternative** |
| **Gradient Boosting** | **95.29%** | **97.70%** | **83.33%** | **0.8995** | **0.9483** | High Performance |
| **Decision Tree** | **92.15%** | **89.81%** | **77.78%** | **0.8336** | **0.9081** | Baseline Tree |
| **Logistic Regression** | **81.16%** | **72.67%** | **40.85%** | **0.5230** | **0.8009** | Linear Baseline |

### Confusion Matrix on Unseen Holdout Data

| Actual \ Predicted | Predicted No Churn (Retained) | Predicted Churn (At-Risk) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual No Churn (0)** | **900** (True Negative) | **4** (False Positive / 0.4%) | 904 |
| **Actual Churn (1)** | **90** (False Negative) | **216** (True Positive / 70.6%) | 306 |
| **Total Predicted** | 990 | 220 | **1,210** |

- **Precision of 98.18%**: When ChurnIQ flags an account as at-risk, it is accurate **98.2% of the time**, eliminating wasted retention spending on happy customers.
- **Accuracy of 92.23%**: High overall classification consistency on holdout data.

---

## 🎛️ 5. What-If Counterfactual Simulator

The **What-If Simulator** enables customer success managers to interactively explore counterfactual retention strategies:
1. Load a high-risk customer profile (e.g. Month-to-Month subscriber paying \$105/mo with 78% churn probability).
2. Adjust levers in real time:
   - Switch contract from **Month-to-month** to **One-Year** or **Two-Year**.
   - Apply a **15% retention billing discount**.
   - Activate complimentary **Tech Support** or **Online Security**.
3. View the immediate recalculated churn probability drop (e.g., from **78% High Risk** down to **18% Low Risk**) and estimated saved annual revenue.

---

## 🛠️ 6. Technology Stack

- **Frontend**: Modern HTML5, Tailwind CSS, Vanilla JavaScript (ES6+), Chart.js, Papa Parse 5.4, jsPDF 2.5
- **Backend**: Flask 3.1, Flask-CORS 6.0, Gunicorn 23.0
- **Machine Learning**: Scikit-Learn 1.9, XGBoost 3.4, SHAP 0.54, Joblib 1.6
- **Data Engineering**: Pandas 2.3, NumPy 2.2, Scipy 1.18
- **Testing**: Pytest 9.2 (14 unit and integration test cases)
- **Containerization & CI/CD**: Docker, Docker Compose, GitHub Actions
- **Deployment**: GitHub Pages (Static Web Platform) + Flask REST API (Render / Railway / Cloud)

---

## 📂 7. Repository Structure

```text
customer-churn-prediction/
│
├── .github/workflows/
│   └── ci.yml                    # Automated GitHub Actions test pipeline
│
├── frontend/                     # Modular frontend assets
│   ├── index.html                # Mirrored web interface
│   ├── style.css                 # Custom glassmorphic styling
│   └── script.js                 # Frontend orchestration logic
│
├── backend/                      # Python REST API server
│   ├── app.py                    # Flask application factory & CORS configuration
│   ├── routes.py                 # API endpoints (/health, /predict, /predict-batch, /analytics)
│   └── requirements.txt          # Backend dependencies
│
├── data/                         # Datasets
│   ├── raw/
│   │   ├── telecom_churn_train.csv   # 6,180 raw training records (28 columns)
│   │   └── telecom_churn_test.csv    # 2,060 raw holdout test records
│   └── processed/
│       ├── telecom_churn_train_cleaned.csv
│       ├── telecom_churn_train_encoded.csv
│       ├── telecom_churn_train_features.csv
│       └── final_predictions.csv     # Batch output on test set
│
├── models/                       # Serialized model artifacts
│   ├── best_model.joblib         # Trained Random Forest classifier
│   ├── best_model.pkl            # Backwards-compatible pickle artifact
│   ├── explainer.joblib          # Fitted SHAP TreeExplainer
│   ├── preprocessor.joblib       # Fitted TelecomDataPreprocessor
│   └── metrics.json              # Verified evaluation metrics & curves
│
├── pipeline/                     # Modular ML pipeline
│   ├── __init__.py
│   ├── preprocessor.py           # Zero-leakage Scikit-Learn Transformer
│   ├── clean.py                  # Deduplication & missing value imputation
│   ├── encode.py                 # Categorical dummy transformation
│   ├── features.py               # Synthesized telemetry features
│   ├── train.py                  # Multi-model benchmarking & evaluation
│   └── run_pipeline.py           # Master 1-click pipeline execution script
│
├── notebooks/                    # Jupyter notebooks
│   └── eda.ipynb                 # Comprehensive exploratory data analysis
│
├── tests/                        # Automated test suite
│   ├── test_preprocessing.py     # Pipeline & transformer unit tests
│   ├── test_model.py             # Model loading, predict & consistency tests
│   └── test_api.py               # Flask REST API integration tests
│
├── Dockerfile                    # Production container image specification
├── docker-compose.yml            # Multi-container orchestration
├── index.html                    # GitHub Pages production entry point
├── app.py                        # Root server launcher
├── requirements.txt              # Project dependencies
├── pyproject.toml                # Project metadata
└── README.md                     # Complete documentation
```

---

## 🚀 8. Quick Start Guide

### 1. Launch Online Web App (Zero Setup)
Access the live platform in your browser:  
👉 **[https://umarmukthar03.github.io/customer-churn-prediction/](https://umarmukthar03.github.io/customer-churn-prediction/)**

### 2. Local Setup & Execution

```bash
# Clone the repository
git clone https://github.com/UMARMUKTHAR03/customer-churn-prediction.git
cd customer-churn-prediction

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Machine Learning Pipeline

```bash
python pipeline/run_pipeline.py
```
*Outputs clean datasets, fits preprocessors, benchmarks candidate models, evaluates holdout metrics, and serializes `models/best_model.joblib` and `models/metrics.json`.*

### 4. Run Test Suite

```bash
pytest tests/ -v
```
*Verifies preprocessing, model determinism, and API endpoints across 14 automated tests.*

### 5. Start Flask Backend Server

```bash
python app.py
```
*Server boots at `http://127.0.0.1:5000` with active CORS and SHAP explainability.*

---

## 🐳 9. Docker & Container Deployment

### Run with Docker

```bash
# Build the image
docker build -t churniq:latest .

# Run the container
docker run -d -p 5000:5000 --name churniq-app churniq:latest
```

### Run with Docker Compose

```bash
docker compose up -d
```
Access the service at `http://localhost:5000`.

---

## 📡 10. REST API Documentation

### Base URL: `http://127.0.0.1:5000` or `/api`

### `GET /health`
Verifies backend status and model readiness.
```json
{
  "status": "healthy",
  "service": "ChurnIQ Enterprise Engine",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "shap_available": true
}
```

### `GET /model-info`
Retrieves model metadata, algorithm parameters, and training timestamp.

### `GET /analytics`
Retrieves verified holdout evaluation metrics, confusion matrix, ROC curve points, and model comparison tables.

### `POST /predict`
Generates real-time churn prediction, probability, SHAP risk drivers, and retention recommendations for a single customer.

**Request Body:**
```json
{
  "customer_id": "CUST-8921",
  "contract_type": "Month-to-month",
  "tenure_months": 2,
  "monthly_charges": 105.00,
  "payment_method": "Electronic check",
  "has_internet": 1,
  "has_tech_support": 0,
  "has_online_security": 0,
  "num_complaints": 2,
  "nps_score": 3.0
}
```

**Response Body:**
```json
{
  "customer_id": "CUST-8921",
  "churn_prediction": "Churn",
  "churn_probability": 78.4,
  "risk_level": "High",
  "confidence": 98.2,
  "retention_strategy": {
    "key_driver": "Month-to-Month Contract",
    "action_summary": "Offer 15% discount for upgrading to an annual contract commitment • Incentivize enrollment in Auto-Pay with a $25 credit • Bundle complimentary Tech Support",
    "at_risk_revenue": 3840.00,
    "estimated_saved_revenue": 2496.00
  },
  "explainability": {
    "method": "SHAP TreeExplainer",
    "top_risk_drivers": [
      { "feature": "contract_Month-to-month", "impact": "increases_churn", "shap_value": 0.28 },
      { "feature": "tenure_months", "impact": "increases_churn", "shap_value": 0.18 }
    ]
  }
}
```

### `POST /predict-batch`
Accepts multipart CSV upload or JSON customer array. Processes multiple records, classifies risk tiers, and returns aggregated financial risk statistics.

---

## 👨‍💻 Author & Engineering Credits

**UMAR MUKTHAR**  
- **GitHub**: [@UMARMUKTHAR03](https://github.com/UMARMUKTHAR03)  
- **Repository**: [customer-churn-prediction](https://github.com/UMARMUKTHAR03/customer-churn-prediction)  
- **Live Demo**: [https://umarmukthar03.github.io/customer-churn-prediction/](https://umarmukthar03.github.io/customer-churn-prediction/)

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
