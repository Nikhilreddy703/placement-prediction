"""Unit test suite for placement prediction data, features, and inference."""
import pytest
import pandas as pd
import numpy as np

from config import DROP_COLS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PRESETS
from src.data import load_dataset, validate_student_record, DataValidationError
from src.features import build_preprocessor
from src.predict import predict_student_placement, load_inference_pipeline


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
    assert transformed.shape[1] > len(NUMERIC_FEATURES)  # One-hot encoded columns added


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
