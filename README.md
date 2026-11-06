# PlacementOS Enterprise — Campus Recruitment & Student Career Intelligence

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Pytest-31%20Passed-brightgreen.svg)](https://pytest.org/)

An enterprise-grade, full-stack Monorepo designed for university **Training & Placement (T&P) Cells**, academic counselors, and students. Featuring calibrated ML inference, prescriptive counterfactual recourse, rule-based ATS resume matching, STAR behavioral interview coaching, and institutional cohort analytics.

---

## 🏛️ Monorepo Architecture

```text
placement-prediction/
├── backend/                  # FastAPI REST API (ML inference, recruitment endpoints)
│   ├── app/
│   │   ├── main.py           # FastAPI entry point (/health, /api/predict, /api/models)
│   │   ├── ml_service.py     # Bridge to ML core
│   │   ├── schemas.py        # Pydantic request/response validation
│   │   └── config.py         # Backend environment settings
│   ├── Dockerfile            # Container definition
│   └── requirements.txt      # FastAPI & backend dependencies
│
├── frontend/                 # Next.js 16 (App Router + Tailwind + Lucide + Recharts)
│   ├── src/
│   │   ├── app/              # Layout and views
│   │   ├── components/       # Prediction Studio, Cohort Analytics, Model Governance, ATS Studio
│   │   └── lib/              # API client with offline fallback
│   ├── Dockerfile            # Multi-stage production container
│   └── package.json          # Frontend packages
│
├── ml/                       # Audited Machine Learning Core
│   ├── config.py             # Feature constraints, schemas, paths
│   ├── data/                 # Raw and cleaned dataset files
│   ├── models/               # Serialized pipelines (placement_model.joblib, all_models.joblib)
│   ├── reports/              # Model evaluation metrics and figures
│   ├── src/                  # Core algorithms (predict, explain, counterfactual, tracker)
│   └── tests/                # 31 unit & regression tests (100% pass rate)
│
├── legacy_streamlit/         # Backward-compatible Streamlit dashboard (port 8501)
├── docker-compose.yml        # Orchestration (frontend, backend, postgres, redis)
├── AUDIT.md                  # Comprehensive empirical audit report
└── README.md
```

---

## 🚀 Quickstart Guide (Windows PowerShell)

### 1. Run the Audited ML Test Suite
```powershell
cd ml
python -m pytest -v
cd ..
```
*Expected Result:* `31 passed in ~13s`.

### 2. Run the FastAPI Backend
```powershell
# From project root
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Health Check:* Open `http://127.0.0.1:8000/health` in your browser.

### 3. Run the Next.js Frontend
```powershell
cd frontend
pnpm dev
```
*App URL:* Open `http://localhost:3000` to interact with the modern dashboard.

### 4. Build the Frontend for Production
```powershell
cd frontend
pnpm build
```

### 5. Run the Entire Stack with Docker Compose
```powershell
docker-compose up --build
```
*Services Started:*
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

### 6. Run Legacy Streamlit Dashboard (Optional)
```powershell
cd legacy_streamlit
python -m streamlit run app.py --server.port 8501
```

---

## 🔒 Governance & Empirical Verification Summary
* **Champion Model:** Calibrated Balanced Logistic Regression (`CV BalAcc: 0.7356`, `ROC-AUC: 0.7450`).
* **Optimal Cutoff:** `0.69` (Selected exclusively from training-fold out-of-fold search).
* **Brier Score Loss:** `0.1698` with `+0.0%` empirical mean bias.
* **Input Integrity:** Accepts NumPy arrays and maps them to the exact DataFrame schema required by scikit-learn's `ColumnTransformer`.
