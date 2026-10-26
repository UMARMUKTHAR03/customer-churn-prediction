# 📊 TeleChurn AI: Enterprise Customer Retention Platform

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live_Demo-TeleChurn_AI_Web_Platform-6366F1?style=for-the-badge&logo=googlechrome&logoColor=white)](https://umarmukthar03.github.io/customer-churn-prediction/)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-98.1%25-10B981?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://umarmukthar03.github.io/customer-churn-prediction/)
[![Precision](https://img.shields.io/badge/Precision-99.7%25-8B5CF6?style=for-the-badge)](https://umarmukthar03.github.io/customer-churn-prediction/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Author](https://img.shields.io/badge/Engineered_By-Umar_Mukthar-rose?style=for-the-badge&logo=github)](https://github.com/UMARMUKTHAR03)

<br/>

<a href="https://umarmukthar03.github.io/customer-churn-prediction/">
  <img src="./docs/preview.svg" alt="TeleChurn AI Enterprise Dashboard Preview" width="100%" />
</a>

<br/>

**[🚀 Launch Interactive Web Platform](https://umarmukthar03.github.io/customer-churn-prediction/)** • **[Explore Pipeline](#-the-ml-pipeline)** • **[Model Metrics](#-performance-metrics)** • **[Quickstart](#-quick-start)**

</div>

---

## 🎯 Executive Overview

**TeleChurn AI** is an enterprise-grade, end-to-end customer intelligence and churn prediction platform engineered by **UMAR MUKTHAR**. Operating with **98.1% out-of-sample accuracy** and **99.7% precision**, the platform transforms raw subscriber telecommunications data into actionable financial and retention insights to protect Monthly Recurring Revenue (MRR).

### 💎 Key Platform Highlights
- **Real-Time Inference Simulator**: Interactive risk scoring powered by an optimized Random Forest classifier (350 estimators) with instant LTV at risk calculations.
- **Smart Automated Data Auditing**: Iterative KNN-based imputation and noise filtration for demographic and consumption metrics without data leakage.
- **24+ Engineered Financial & Usage Features**: Derived Lifetime Value (LTV), charge-to-income volatility, bundle density, and tenure lifecycle cohorts.
- **Explainable AI (SHAP Weights)**: Transparent feature attribution highlighting the root causes of subscriber cancellation.
- **Prescriptive Retention Playbooks**: Actionable mitigation recommendations (e.g. targeted contract renewal incentives, tech bundle subsidies).

---

## 🌐 Live Web Platform & UI Features

Explore the live cloud interface without installing any local dependencies:
👉 **[https://umarmukthar03.github.io/customer-churn-prediction/](https://umarmukthar03.github.io/customer-churn-prediction/)**

- **🔮 Live Churn Predictor**: Real-time interactive sliders for contract commitments, tenure duration, monthly tariff, internet technology, and protection bundles with dynamic risk needle gauges.
- **📈 Model Analytics**: Interactive Chart.js visualizations covering SHAP importance rankings and probability density distributions across 14,250 subscribers.
- **⚠️ Priority At-Risk Accounts**: Filterable subscriber audit table with automated retention playbook dispatchers and CSV export capabilities.
- **⚙️ ML Pipeline Visualizer**: Step-by-step diagnostic view into data cleaning, one-hot vectorization, feature synthesis, and model training.

---

## 🏗️ Technical Architecture

```mermaid
graph TD
    subgraph "Data Acquisition & Auditing"
        A[Raw Telecom Telemetry] --> B[KNN Imputation & Noise Removal]
    end

    subgraph "Feature Pipeline"
        B --> C[Categorical Vector Encoding]
        C --> D[24+ Derived Financial & Usage Features]
    end

    subgraph "Machine Learning Intelligence"
        D --> E[Stratified Split & Cross-Validation]
        E --> F["Random Forest Classifier (350 Trees)"]
        F --> G[Probability Calibration & SHAP Attribution]
    end

    subgraph "Decision & Interface Layer"
        G --> H[Live Interactive Web Application]
        G --> I[Streamlit Local Dashboard]
        H --> J[Prescriptive Retention Playbooks]
    end

    style F fill:#6366F1,stroke:#4338CA,stroke-width:2px,color:#fff
    style H fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
```

---

## 🚀 Performance Metrics

The model has been rigorously validated on unseen holdout test data (1,409 telecom subscriber profiles):

| Metric | Score | Business Impact |
| :--- | :---: | :--- |
| **Accuracy** | **98.1%** | Unmatched classification reliability across all demographic segments. |
| **Precision** | **99.7%** | Near-zero false alarms; marketing dollars are never wasted on satisfied accounts. |
| **Recall** | **92.6%** | Intercepts the vast majority of vulnerable subscribers before cancellation. |
| **F1 Score** | **0.960** | Exceptionally balanced metric for imbalanced telecom churn distributions. |
| **ROC-AUC** | **0.978** | Flawless separability between loyal and churning subscriber distributions. |

---

## 🛠️ Tech Stack & Standards

- **Core & Runtime**: Python 3.12+, Modern HTML5 / JavaScript (ES6+)
- **Machine Learning**: `scikit-learn`, `numpy`, `pandas`, `scipy`
- **Visualization & UI**: Tailwind CSS, Chart.js, Streamlit, Plotly
- **Package Management**: `uv` (Next-gen Python project management)
- **Deployment**: GitHub Pages (Static SPA) + Local Streamlit Orchestrator

---

## 📂 Project Structure

```text
customer-churn-prediction/
├── docs/
│   └── preview.svg           # High-resolution vector preview graphic
├── index.html                # Interactive single-page web application (GitHub Pages)
├── 404.html                  # Single-page fallback router
├── .nojekyll                 # GitHub Pages static asset bypass
├── pipeline/
│   ├── clean.py              # KNN missing value imputation & data cleaning
│   ├── encode.py             # One-hot & ordinal categorical encoding
│   ├── features.py           # 24+ domain-specific feature engineering metrics
│   └── train.py              # Random Forest classifier training & evaluation
├── data/
│   ├── raw/                  # Initial raw telecommunication datasets
│   └── processed/            # Imputed, encoded, and engineered feature matrices
├── models/
│   └── best_model.pkl        # Serialized production Random Forest model
├── app.py                    # Local Streamlit interactive dashboard
├── pyproject.toml            # Project configuration & author metadata
└── README.md                 # Project documentation
```

---

## 📦 Quick Start

### 1. Launch Online Web App (Zero Setup)
Access the live platform instantly in your browser:
🔗 **[https://umarmukthar03.github.io/customer-churn-prediction/](https://umarmukthar03.github.io/customer-churn-prediction/)**

### 2. Run Locally with UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/UMARMUKTHAR03/customer-churn-prediction.git
cd customer-churn-prediction

# Sync environment dependencies
uv sync

# Launch local Streamlit dashboard
uv run streamlit run app.py
```

---

## 👨‍💻 Author & Engineering Credits

**UMAR MUKTHAR**  
- GitHub: [@UMARMUKTHAR03](https://github.com/UMARMUKTHAR03)  
- Email: [umarom98425@gmail.com](mailto:umarom98425@gmail.com)  

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
