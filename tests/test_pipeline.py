"""Comprehensive unit test suite for placement prediction data, features, and inference."""
import io
import os
import sqlite3
import tempfile
import numpy as np
import pandas as pd
import pytest

from config import CATEGORICAL_FEATURES, DROP_COLS, NUMERIC_FEATURES, PRESETS
from src.data import DataValidationError, load_dataset, validate_student_record
from src.database import get_prediction_history, init_db, log_prediction
from src.explain import (
    categorize_risk,
    compute_student_feature_contributions,
    generate_coaching_recommendations,
    get_radar_comparison_data,
    simulate_what_if,
)
from src.features import build_preprocessor
from src.predict import (
    batch_predict_students,
    load_inference_pipeline,
    predict_student_placement,
)
from src.report_generator import generate_student_pdf_report


def test_load_dataset_drops_leakage():
    """Verify sl_no and salary are stripped and target is binary 0/1."""
    X, y = load_dataset()
    for col in DROP_COLS:
        assert col not in X.columns, f"Leakage column {col} was not dropped!"
    assert set(y.unique()).issubset({0, 1}), "Target must be binary 0 or 1."
    assert len(X) == len(y)


def test_validation_valid_preset():
    """Verify standard preset profiles pass validation cleanly."""
    for name, profile in PRESETS.items():
        validated_df = validate_student_record(profile)
        assert len(validated_df) == 1
        assert list(validated_df.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_validation_rejects_missing_column():
    """Verify validation raises error if required column is missing."""
    sample = dict(PRESETS["🌟 High-Scoring All-Rounder"])
    del sample["ssc_p"]
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_student_record(sample)


def test_validation_rejects_out_of_bounds_score():
    """Verify validation rejects score > 100 or < 0."""
    sample = dict(PRESETS["🌟 High-Scoring All-Rounder"])
    sample["degree_p"] = 150.0  # Invalid percentage
    with pytest.raises(DataValidationError, match="must be between"):
        validate_student_record(sample)


def test_validation_rejects_invalid_category():
    """Verify validation rejects unexpected categorical option."""
    sample = dict(PRESETS["🌟 High-Scoring All-Rounder"])
    sample["specialisation"] = "UnknownStream"
    with pytest.raises(DataValidationError, match="Invalid value"):
        validate_student_record(sample)


def test_build_preprocessor_shape():
    """Verify ColumnTransformer can fit and transform valid input."""
    X, y = load_dataset()
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)


def test_prediction_output_schema():
    """Verify prediction returns all required fields and valid probability."""
    sample = PRESETS["🌟 High-Scoring All-Rounder"]
    result = predict_student_placement(sample)

    assert "label" in result
    assert "probability" in result
    assert "risk_level" in result
    assert "status_text" in result
    assert "recommendations" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]


def test_batch_prediction():
    """Verify batch inference produces predicted status, probability, and risk level."""
    pipeline = load_inference_pipeline()
    sample_batch = pd.DataFrame([
        PRESETS["🌟 High-Scoring All-Rounder"],
        PRESETS["⚠️ At-Risk Student (Needs Support)"],
    ])
    scored = batch_predict_students(sample_batch, pipeline)
    assert len(scored) == 2
    assert "Predicted_Status" in scored.columns
    assert "Placement_Probability" in scored.columns
    assert "Risk_Level" in scored.columns
    assert scored["Predicted_Status"].iloc[0] == "Placed"


def test_what_if_simulator():
    """Verify that improving E-test score and adding workex improves placement probability."""
    pipeline = load_inference_pipeline()
    base = PRESETS["⚠️ At-Risk Student (Needs Support)"]
    sim = simulate_what_if(pipeline, base, {"etest_p": 95.0, "workex": "Yes"})
    assert "base_probability" in sim
    assert "simulated_probability" in sim
    assert "delta" in sim
    assert sim["simulated_probability"] > sim["base_probability"]


def test_database_logging(tmp_path):
    """Verify SQLite logging and retrieval."""
    test_db = tmp_path / "test_history.db"
    init_db(test_db)

    profile = PRESETS["🌟 High-Scoring All-Rounder"]
    res = {"status_text": "Placed", "probability": 0.85, "risk_level": "Low Risk"}
    row_id = log_prediction(profile, res, "KNN", identifier="TestStudent", db_path=test_db)
    assert row_id == 1

    df_hist = get_prediction_history(limit=10, db_path=test_db)
    assert len(df_hist) == 1
    assert df_hist["student_identifier"].iloc[0] == "TestStudent"
    assert df_hist["probability"].iloc[0] == 0.85


def test_pdf_report_generation():
    """Verify that PDF generation produces valid non-empty PDF bytes."""
    sample = PRESETS["🌟 High-Scoring All-Rounder"]
    res = predict_student_placement(sample)
    pdf_bytes = generate_student_pdf_report(sample, res, model_name="KNN")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
