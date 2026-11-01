"""Predict placement for one student using the saved model.

Usage:  python src/predict.py
Or import:  from predict import predict_student
"""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "placement_model.joblib"


def get_recommendations(details: dict, prob: float = None):
    """Generate targeted placement cell advice and action items."""
    recs = []
    degree_p = float(details.get("degree_p", 0))
    etest_p = float(details.get("etest_p", 0))
    workex = str(details.get("workex", "No")).strip().lower()
    mba_p = float(details.get("mba_p", 0))

    if workex == "no":
        recs.append("Practical Experience: Complete a 2-month summer internship or capstone industry project.")
    if etest_p < 70:
        recs.append("Aptitude Practice: Practice quantitative aptitude and logical reasoning daily for campus screening tests.")
    if degree_p < 65:
        recs.append("Technical Skills: Build strong portfolio projects (GitHub/live demos) to compensate for academic cutoff.")
    if mba_p < 65:
        recs.append("Domain Readiness: Prepare business case studies, financial modeling, or HR analytics certifications.")
    if not recs:
        recs.append("Profile Strong: Attend advanced mock technical interviews and focus on company-specific case questions.")
    return recs


def predict_student(details: dict):
    """details: {column: value} using the same column names as the training data (without status/salary)."""
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([details])
    label = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
    return label, prob


if __name__ == "__main__":
    # Example student profile
    student = {
        "gender": "M", "ssc_p": 71.0, "ssc_b": "Central", "hsc_p": 68.0, "hsc_b": "Others",
        "hsc_s": "Commerce", "degree_p": 69.0, "degree_t": "Comm&Mgmt", "workex": "Yes",
        "etest_p": 78.0, "specialisation": "Mkt&Fin", "mba_p": 64.0,
    }
    label, prob = predict_student(student)
    recs = get_recommendations(student, prob)

    print("=" * 50)
    print("      STUDENT PLACEMENT PREDICTION RESULT")
    print("=" * 50)
    status_str = "PLACED (Low Risk) [OK]" if label == 1 else "NOT PLACED (At-Risk) [ATTENTION]"
    print(f"Outcome Prediction : {status_str}")
    if prob is not None:
        print(f"Placement Likelihood: {prob:.1%}")
    print("\nPlacement Cell Recommendations:")
    for r in recs:
        print(f" - {r}")
    print("=" * 50)

