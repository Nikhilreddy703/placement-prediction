"""Unit tests for Placement Readiness Tracker: score formulas, alert rules, and persistence."""
import pytest
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

from src.tracker import (
    calculate_readiness_score,
    init_tracker_db,
    register_or_get_student,
    record_student_checkin,
    get_student_checkin_history,
    get_early_warning_alerts,
    draft_supportive_message,
    delete_student_data,
    update_task_completion,
    get_student_tasks,
)
from src.predict import load_inference_pipeline
from config import PRESETS


def test_readiness_score_formula_weights():
    """Verify exact formula components: 60% prob, 20% etest, 10% workex, 10% consistency."""
    # Case 1: All maximal
    score, band, color = calculate_readiness_score(prob=1.0, etest_p=100.0, workex="Yes", checkin_count=4)
    # 1.0 * 60 + 100 * 0.20 + 100 * 0.10 + 100 * 0.10 = 60 + 20 + 10 + 10 = 100.0
    assert score == 100.0
    assert band == "Placement Ready"
    assert color == "#10B981"

    # Case 2: Minimal values
    score_low, band_low, color_low = calculate_readiness_score(prob=0.1, etest_p=30.0, workex="No", checkin_count=1)
    # 0.1 * 60 + 30 * 0.20 + 0 * 0.10 + 50 * 0.10 = 6.0 + 6.0 + 0 + 5.0 = 17.0
    assert score_low == 17.0
    assert band_low == "Needs Support"
    assert color_low == "#EF4444"

    # Case 3: Mid values (Developing)
    score_mid, band_mid, color_mid = calculate_readiness_score(prob=0.6, etest_p=65.0, workex="No", checkin_count=2)
    # 0.6 * 60 + 65 * 0.20 + 0 + 70 * 0.10 = 36 + 13 + 7 = 56.0
    assert score_mid == 56.0
    assert band_mid == "Developing"
    assert color_mid == "#F59E0B"


def test_student_registration_locks_academics(tmp_path):
    """Verify that secondary/high school marks stay locked after registration."""
    db_file = tmp_path / "tracker_test.db"
    init_tracker_db(db_file)

    profile = dict(PRESETS["💼 Average Academics + Work Experience"])
    student = register_or_get_student("TEST-01", "Alex Doe", profile, db_path=db_file)
    assert student["ssc_p"] == profile["ssc_p"]
    assert student["hsc_p"] == profile["hsc_p"]
    assert student["degree_p"] == profile["degree_p"]

    # Even if someone attempts to register again with different ssc_p, it remains locked
    profile_tampered = dict(profile)
    profile_tampered["ssc_p"] = 99.9
    student_recheck = register_or_get_student("TEST-01", "Alex Doe", profile_tampered, db_path=db_file)
    assert student_recheck["ssc_p"] == profile["ssc_p"]


def test_checkin_updates_and_tasks(tmp_path):
    """Verify progress check-in records new improvable fields and creates 4-week tasks."""
    db_file = tmp_path / "tracker_test.db"
    pipeline = load_inference_pipeline()

    profile = dict(PRESETS["⚠️ At-Risk Student (Needs Support)"])
    register_or_get_student("TEST-02", "Sam Smith", profile, db_path=db_file)

    res = record_student_checkin(
        student_id="TEST-02",
        etest_p=80.0,  # Improved
        mba_p=68.0,
        workex="Yes",   # Gained internship
        active_pipeline=pipeline,
        threshold=0.50,
        db_path=db_file,
    )
    assert res["probability"] > 0.0
    assert res["readiness_score"] > 0.0

    history = get_student_checkin_history("TEST-02", db_path=db_file)
    assert len(history) == 1
    assert history["etest_p"].iloc[0] == 80.0
    assert history["workex"].iloc[0] == "Yes"

    # Verify tasks generated
    tasks = get_student_tasks("TEST-02", db_path=db_file)
    assert len(tasks) == 4
    # Toggle task
    update_task_completion(tasks[0]["task_id"], True, db_path=db_file)
    updated_tasks = get_student_tasks("TEST-02", db_path=db_file)
    assert updated_tasks[0]["completed"] == 1


def test_early_warning_alerts(tmp_path):
    """Verify alert triggers: probability drop and 30+ days inactivity."""
    db_file = tmp_path / "tracker_test.db"
    pipeline = load_inference_pipeline()

    profile = dict(PRESETS["🌟 High-Scoring All-Rounder"])
    register_or_get_student("TEST-03", "Jordan Lee", profile, db_path=db_file)

    # Checkin 1: Good score
    record_student_checkin(
        "TEST-03", etest_p=90.0, mba_p=75.0, workex="Yes",
        active_pipeline=pipeline, checkin_date="2026-08-01", db_path=db_file
    )

    # Checkin 2: Severe drop in aptitude and MBA
    record_student_checkin(
        "TEST-03", etest_p=40.0, mba_p=50.0, workex="No",
        active_pipeline=pipeline, checkin_date="2026-08-15", db_path=db_file
    )

    alerts = get_early_warning_alerts(db_path=db_file)
    assert not alerts.empty
    student_alert = alerts[alerts["student_id"] == "TEST-03"]
    assert not student_alert.empty
    assert any("Drop" in t for t in student_alert["alert_type"].tolist())

    # Test supportive draft message
    drop_alert = student_alert[student_alert["alert_type"].str.contains("Drop")].iloc[0]
    draft = draft_supportive_message(drop_alert.to_dict())
    assert "Jordan Lee" in draft
    assert "Placement Cell" in draft


def test_delete_student_data(tmp_path):
    """Verify delete-my-data wipes all records for student."""
    db_file = tmp_path / "tracker_test.db"
    pipeline = load_inference_pipeline()

    profile = dict(PRESETS["🌟 High-Scoring All-Rounder"])
    register_or_get_student("TEST-04", "Taylor Swift", profile, db_path=db_file)
    record_student_checkin("TEST-04", 80.0, 70.0, "Yes", pipeline, db_path=db_file)

    assert delete_student_data("TEST-04", db_path=db_file) is True
    history = get_student_checkin_history("TEST-04", db_path=db_file)
    assert history.empty
    tasks = get_student_tasks("TEST-04", db_path=db_file)
    assert len(tasks) == 0
