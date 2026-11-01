# Student Placement Prediction (ML Mini Project)

Predict whether a student will be **placed or not placed** from academic and profile details.
Binary classification on a small tabular dataset using scikit-learn, with a Streamlit demo app.

## Folder structure

```
placement-prediction/
|-- data/
|   |-- raw/placement.csv            # dataset (sample is SYNTHETIC - replace with real data)
|   `-- processed/
|-- notebooks/                       # optional: put your Jupyter experiments here
|-- src/
|   |-- generate_sample_data.py      # makes a synthetic dataset so everything runs
|   |-- preprocess.py                # load, clean, encode, scale
|   |-- eda.py                       # exploration + plots
|   |-- evaluate.py                  # plotting helpers
|   |-- train.py                     # compare models, save the best
|   `-- predict.py                   # predict for one student
|-- models/                          # saved model + metadata (created by train.py)
|-- app/app.py                       # Streamlit interface
|-- reports/                         # results.csv and figures/ (created automatically)
|-- requirements.txt
`-- README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Problem Statement and Objectives

**Problem Statement:**
Given a student's academic and profile details, predict whether the student will be placed (1) or not placed (0) so that training and placement cells can identify at-risk students early and provide timely interventions.

**Key Objectives:**
- **Load & Explore:** Ingest the placement dataset and perform comprehensive exploratory data analysis (EDA).
- **Clean & Preprocess:** Drop target-leakage columns (`salary`, `sl_no`), handle missing values, standardize numeric attributes, and one-hot encode categorical features.
- **Train & Compare Models:** Evaluate 5 classification algorithms (Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine).
- **Comprehensive Evaluation:** Assess accuracy, precision, recall, test F1-score, and 5-fold cross-validated F1.
- **Identify Key Influencing Factors:** Extract feature importance rankings to determine the strongest placement drivers.
- **Interactive Web Application:** Deploy a multi-tab Streamlit dashboard for single-student predictions, preset testing, batch evaluation, and actionable recommendations.

## Run Order

```bash
python src/generate_sample_data.py     # Generates synthetic data (skip if you already have real placement.csv)
python src/eda.py                      # EDA summary + plots in reports/figures/ + reports/eda_summary.md
python src/train.py                    # Compares 5 models, saves best pipeline, metrics & reports
python src/predict.py                  # Predict placement and coaching tips from terminal
python -m streamlit run app/app.py     # Open the interactive web dashboard in your browser
```

## Generated Artifacts and Reports

- `reports/results.csv` - 5-model performance comparison table
- `reports/feature_importance.csv` - Ranked predictor importance weights
- `reports/eda_summary.md` - Structured exploratory data analysis report
- `reports/model_evaluation_report.md` - Complete model benchmarking documentation
- `reports/figures/class_balance.png` - Target class balance visualization
- `reports/figures/numeric_by_status.png` - Academic score distributions by placement status
- `reports/figures/correlation.png` - Correlation matrix heatmap
- `reports/figures/model_comparison.png` - Model accuracy vs. cross-validated F1 comparison
- `reports/figures/confusion_matrix.png` - Confusion matrix for the top-performing model
- `reports/figures/feature_importance.png` - Top 10 influential factors bar chart
- `data/processed/placement_cleaned.csv` - Cleaned and encoded tabular dataset
- `models/placement_model.joblib` & `models/metadata.json` - Serialized ML model and schemas


## Troubleshooting

- **`ModuleNotFoundError`**: activate the virtual environment and run `pip install -r requirements.txt`.
- **`KeyError` on a column**: the column names in your CSV differ; update `TARGET` and `DROP_COLS` in `src/preprocess.py`.
- **Very high accuracy (99-100%)**: probably data leakage; check for a column that reveals the outcome.
- **Scores change between runs on a tiny dataset**: normal; rely on the cross-validated F1 and mention it in the report.
- **Streamlit says "Model not found"**: run `python src/train.py` first.

## Notes for the report

- State clearly whether the data is real or synthetic. Results on the synthetic sample only demonstrate that the pipeline works.
- With a few hundred rows, report cross-validation scores, not just one test split.
- Mention limitations: small data, one college only, and predictions should support coaching, not label or exclude students.
