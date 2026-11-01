"""Central configuration and constants for the Placement Prediction system."""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RAW_PATH = DATA_RAW_DIR / "placement.csv"
DATA_PROCESSED_PATH = DATA_PROCESSED_DIR / "placement_cleaned.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "placement_model.joblib"
ALL_MODELS_FILE = MODELS_DIR / "all_models.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_FILE = REPORTS_DIR / "results.csv"
FEATURE_IMP_FILE = REPORTS_DIR / "feature_importance.csv"
EVALUATION_REPORT_FILE = REPORTS_DIR / "model_evaluation_report.md"

DB_PATH = BASE_DIR / "placement_history.db"

# Data Schemas and Leakage Exclusion
TARGET_COL = "status"
POSITIVE_LABEL = "Placed"
DROP_COLS = ["sl_no", "salary"]

NUMERIC_FEATURES = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
]

ALL_FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Numerical ranges and constraints for validation
FEATURE_CONSTRAINTS = {
    "ssc_p": {"min": 0.0, "max": 100.0, "default": 70.0},
    "hsc_p": {"min": 0.0, "max": 100.0, "default": 70.0},
    "degree_p": {"min": 0.0, "max": 100.0, "default": 68.0},
    "etest_p": {"min": 0.0, "max": 100.0, "default": 75.0},
    "mba_p": {"min": 0.0, "max": 100.0, "default": 65.0},
    "gender": {"options": ["M", "F"], "default": "M"},
    "ssc_b": {"options": ["Central", "Others"], "default": "Central"},
    "hsc_b": {"options": ["Central", "Others"], "default": "Others"},
    "hsc_s": {"options": ["Commerce", "Science", "Arts"], "default": "Commerce"},
    "degree_t": {"options": ["Comm&Mgmt", "Sci&Tech", "Others"], "default": "Comm&Mgmt"},
    "workex": {"options": ["Yes", "No"], "default": "Yes"},
    "specialisation": {"options": ["Mkt&Fin", "Mkt&HR"], "default": "Mkt&Fin"},
}

# Human-friendly field labels
FIELD_LABELS = {
    "gender": "Student Gender",
    "ssc_p": "10th Grade Percentage (SSC %)",
    "ssc_b": "10th Board of Examination",
    "hsc_p": "12th Grade Percentage (HSC %)",
    "hsc_b": "12th Board of Examination",
    "hsc_s": "12th Academic Stream",
    "degree_p": "Undergraduate Degree Percentage (%)",
    "degree_t": "Undergraduate Degree Type",
    "workex": "Prior Work Experience / Internship",
    "etest_p": "College Employability & Aptitude Test Score (%)",
    "specialisation": "MBA / PG Specialisation",
    "mba_p": "MBA / PG Percentage (%)",
}

FIELD_HELP = {
    "ssc_p": "Secondary School Certificate marks percentage (10th standard)",
    "ssc_b": "Board of secondary education (Central/State/Others)",
    "hsc_p": "Higher Secondary Certificate marks percentage (12th standard)",
    "hsc_b": "Board of higher secondary education (Central/State/Others)",
    "hsc_s": "Specialization in 12th standard (Commerce, Science, Arts)",
    "degree_p": "Undergraduate graduation percentage (Bachelor's degree)",
    "degree_t": "Undergraduate academic discipline (Comm&Mgmt, Sci&Tech, Others)",
    "workex": "Prior verified industry work experience or summer internship",
    "etest_p": "College placement cell employability and quantitative test score",
    "specialisation": "MBA specialization stream (Mkt&Fin or Mkt&HR)",
    "mba_p": "MBA aggregate academic percentage",
}

# Preset scenarios
PRESETS = {
    "🌟 High-Scoring All-Rounder": {
        "gender": "F", "ssc_p": 82.5, "ssc_b": "Central", "hsc_p": 84.0, "hsc_b": "Central",
        "hsc_s": "Commerce", "degree_p": 78.5, "degree_t": "Comm&Mgmt", "workex": "Yes",
        "etest_p": 88.0, "specialisation": "Mkt&Fin", "mba_p": 72.0,
    },
    "💼 Average Academics + Work Experience": {
        "gender": "M", "ssc_p": 68.0, "ssc_b": "Others", "hsc_p": 65.0, "hsc_b": "Others",
        "hsc_s": "Science", "degree_p": 67.0, "degree_t": "Sci&Tech", "workex": "Yes",
        "etest_p": 75.0, "specialisation": "Mkt&Fin", "mba_p": 64.0,
    },
    "⚠️ At-Risk Student (Needs Support)": {
        "gender": "M", "ssc_p": 50.0, "ssc_b": "Central", "hsc_p": 52.0, "hsc_b": "Central",
        "hsc_s": "Commerce", "degree_p": 56.0, "degree_t": "Comm&Mgmt", "workex": "No",
        "etest_p": 55.0, "specialisation": "Mkt&HR", "mba_p": 54.0,
    },
    "🎯 Strong Aptitude, Borderline Academics": {
        "gender": "F", "ssc_p": 62.0, "ssc_b": "Central", "hsc_p": 63.5, "hsc_b": "Others",
        "hsc_s": "Science", "degree_p": 64.0, "degree_t": "Sci&Tech", "workex": "No",
        "etest_p": 88.0, "specialisation": "Mkt&Fin", "mba_p": 63.0,
    },
}

# Risk Thresholds
THRESHOLD_HIGH_RISK = 0.40
THRESHOLD_MODERATE_RISK = 0.65

RANDOM_SEED = 42
