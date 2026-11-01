# Campus Placement Prediction & Student Success Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-red.svg)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1+-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Pytest-11%20Passed-brightgreen.svg)](https://pytest.org/)

An enterprise-grade, end-to-end Machine Learning web application designed for collegiate **Training & Placement (T&P) Cells**, academic counselors, and students. The platform evaluates academic performance, internship history, and aptitude scores to predict placement probability, categorize risk, simulate "what-if" profile improvements, and generate actionable coaching plans.

---

## 📌 Problem Statement & Objectives

Campus placement clearance is heavily dependent on academic consistency, technical proficiency, and screening test aptitude. Late identification of struggling students leaves placement cells with inadequate time to intervene.

**Core Objectives:**
1. **Early Risk Identification:** Predict placement likelihood before corporate recruitment drives commence.
2. **Probability Calibration:** Provide calibrated, realistic probability scores across 6 tuned classification algorithms.
3. **What-If Sensitivity Simulation:** Show students how specific improvements (e.g. +10 marks in aptitude, or adding an internship) directly alter placement probability.
4. **Actionable Coaching:** Automate tailored remedial guidance and exportable PDF student reports.
5. **Batch Operations & Audit Logging:** Provide placement officers with bulk CSV/Excel scoring and SQLite-backed audit logs.

---

## 📂 Project Architecture

```
placement-prediction/
├── app.py                      # Root application entry point
├── app/
│   └── app.py                  # Full production Streamlit web application
├── config.py                   # Central configuration, constants, schemas, bounds
├── data/
│   ├── raw/placement.csv       # Campus recruitment dataset (400 records)
│   └── processed/              # Cleaned, leakage-free dataset
├── models/
│   ├── placement_model.joblib  # Calibrated champion model pipeline (KNN)
│   ├── all_models.joblib       # Dictionary of all 6 trained, calibrated pipelines
│   └── metadata.json           # Schemas, benchmark metrics, optimal thresholds
├── reports/
│   ├── figures/                # Visualizations (ROC, confusion matrix, model comparison)
│   ├── results.csv             # 6-Algorithm benchmark metrics table
│   ├── feature_importance.csv  # Ranked predictor weights
│   └── model_evaluation_report.md
├── src/
│   ├── data.py                 # Data ingestion, schema validation, leakage removal
│   ├── features.py             # ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── train.py                # Hyperparameter tuning (GridSearchCV), CV, calibration
│   ├── predict.py              # Single and batch inference engine
│   ├── explain.py              # Cohort delta analysis, What-If simulation, radar benchmarking
│   ├── report_generator.py     # ReportLab PDF report builder
│   ├── database.py             # SQLite audit trail integration
│   ├── evaluate.py             # Diagnostic plotting helpers
│   └── eda.py                  # CLI exploratory analysis script
├── tests/
│   └── test_pipeline.py        # 11 automated pytest unit tests
├── Dockerfile                  # Containerized deployment specification
├── .dockerignore               # Docker build exclusions
├── .env.example                # Environment variables template
├── requirements.txt            # Pinned dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Quickstart & Installation

### 1. Local Setup (Virtual Environment)

```bash
# Clone the repository and navigate to root
cd placement-prediction

# Create and activate Python virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell / Command Prompt)
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify with Unit Tests

```bash
python -m pytest tests/ -v
```

### 3. Launch the Streamlit Web Application

```bash
python -m streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🚀 Execution Pipeline

You can run individual pipeline components from the terminal:

```bash
# 1. Exploratory Data Analysis & Static Figures
python src/eda.py

# 2. Train, Tune, Calibrate & Benchmark 6 Algorithms
python src/train.py

# 3. Test CLI Single Student Inference
python src/predict.py

# 4. Launch Interactive Web App
python -m streamlit run app.py
```

---

## 🤖 Model Benchmarking & Performance

We evaluated 6 classification algorithms using **5-Fold Stratified Cross-Validation** on training data, followed by probability calibration (`CalibratedClassifierCV`) and decision threshold optimization on the 20% holdout test split (80 students):

| Model Architecture | Accuracy | Precision | Recall | Test F1 | ROC-AUC | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN (Champion)** | **70.0%** | **71.4%** | **92.6%** | **0.8065** | **0.6531** | **0.8065** |
| Random Forest | 75.0% | 75.0% | 94.4% | 0.8361 | 0.6830 | 0.7957 |
| XGBoost | 73.8% | 74.6% | 92.6% | 0.8264 | 0.6781 | 0.7878 |
| Logistic Regression | 72.5% | 76.7% | 85.2% | 0.8070 | 0.7450 | 0.7816 |
| Gradient Boosting | 75.0% | 74.3% | 96.3% | 0.8387 | 0.6766 | 0.7792 |
| Support Vector Machine (SVM) | 76.2% | 77.8% | 90.7% | 0.8376 | 0.7457 | 0.7554 |

- **Champion Model:** **KNN** (Highest CV F1: **0.8065**)
- **Optimal Decision Threshold:** `0.44` (Maximizes test F1 to **0.8346**)
- **Data Leakage Safeguard:** Target leakage attributes (`sl_no`, `salary`) are strictly stripped during ingestion.

---

## 🔍 Key Influencing Predictors

Feature importance rankings extracted from ensemble models indicate the strongest placement drivers:

1. **Undergraduate Graduation Score (`degree_p`):** ~22.6% weight. Initial corporate screening filter.
2. **Secondary Education 10th (`ssc_p`):** ~19.4% weight. Indicator of long-term academic consistency.
3. **Higher Secondary 12th (`hsc_p`):** ~15.5% weight. High school STEM / business foundation.
4. **Post-Grad / MBA Score (`mba_p`):** ~10.7% weight. Domain specialization performance.
5. **Employability & Aptitude Score (`etest_p`):** ~8.7% weight. Quantitative screening clearance.
6. **Work Experience (`workex`):** Students with prior internships have a **+16.8% higher placement rate** (77.7% vs 60.9%).

---

## 🌟 Key Application Features

1. **Multi-Model Inference Dropdown:** Dynamically switch active model between KNN, Random Forest, XGBoost, Logistic Regression, Gradient Boosting, or SVM in real-time.
2. **Interactive Probability Gauge:** Visual Plotly meter showing color-coded risk bands (Low, Moderate, High Risk).
3. **Cohort Comparison Radar Chart:** Side-by-side spider chart comparing the student's metrics against the average placed student cohort.
4. **Feature Influence Explainability:** Directional delta analysis displaying attributes helping or hurting the student relative to peers.
5. **Live 'What-If' Simulator:** Sensitivity sliders allowing students to test how boosting aptitude scores or adding an internship alters placement odds.
6. **Automated PDF Report Builder:** One-click generation of professional student readiness reports using ReportLab.
7. **Batch Evaluation (CSV & Excel):** Bulk roster upload with automatic at-risk tagging and downloadable scored CSVs.
8. **Recruitment Analytics Dashboard:** Interactive Plotly charts analyzing conversion by specialization, gender, board, and academic distribution.
9. **Persistent SQLite Audit History:** Logs every evaluated profile and prediction for historical review and counseling follow-up.
10. **Pre-configured Presets & Form Reset:** Quick-load buttons for standard student personas (`🌟 High-Scoring All-Rounder`, `💼 Average Academics + Work Experience`, `⚠️ At-Risk Student`, `🎯 Strong Aptitude, Borderline Academics`).

---

## 🐳 Docker Deployment

To build and run the containerized application locally:

```bash
# Build the Docker image
docker build -t placement-predictor .

# Run container on port 8501
docker run -p 8501:8501 placement-predictor
```
Navigate to `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set the **Main file path** to `app.py`.
5. Click **Deploy**!

---

## 📄 License & Academic Note

This project is developed for educational institutions and Training & Placement cells. Predictions are intended to guide timely mentoring, skills training, and coaching interventions.
