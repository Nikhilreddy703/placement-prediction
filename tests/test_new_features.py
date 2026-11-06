"""Unit tests for the 6 new advanced features:
1. Counterfactual Actionable Recourse Engine
2. Resume & JD Semantic Matcher
3. Multi-Tier Placement Predictor
4. Tech & Coding Profile Analyzer
5. TPO Command Center
6. AI Mock Interview Simulator
"""
import pytest
import pandas as pd
import numpy as np

from config import PRESETS
from src.predict import load_inference_pipeline
import src.counterfactual as cf
import src.resume_analyzer as ra
import src.tier_predictor as tp
import src.tech_profile as tech
import src.tpo_center as tpo
import src.mock_interview as mi


@pytest.fixture
def trained_pipeline():
    return load_inference_pipeline()


@pytest.fixture
def sample_student():
    return PRESETS["⚠️ At-Risk Student (Needs Support)"]


# -------------------------------------------------------------
# 1. Counterfactual Engine Tests
# -------------------------------------------------------------
def test_counterfactual_find_minimum_changes(trained_pipeline, sample_student):
    res = cf.find_minimum_changes(trained_pipeline, sample_student, target_prob=0.75)
    assert "achievable" in res
    assert "current_probability" in res
    assert "achieved_probability" in res
    assert "changes" in res
    assert "narrative" in res
    if res["achievable"]:
        assert res["achieved_probability"] >= 0.75


def test_counterfactual_feature_sensitivity(trained_pipeline, sample_student):
    df_sens = cf.compute_feature_sensitivity(trained_pipeline, sample_student)
    assert isinstance(df_sens, pd.DataFrame)
    assert "Feature" in df_sens.columns
    assert "Sensitivity_Per_Unit" in df_sens.columns
    assert len(df_sens) >= 2


# -------------------------------------------------------------
# 2. Resume & JD Analyzer Tests
# -------------------------------------------------------------
def test_resume_skill_extraction():
    text = "Experienced software developer proficient in Python, SQL, Docker, and Machine Learning."
    skills = ra.extract_skills_from_text(text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Docker" in skills


def test_resume_ats_match():
    resume_skills = ["Python", "SQL", "Git", "Tableau"]
    jd_skills = ["Python", "SQL", "Docker", "AWS", "Kubernetes"]
    ats = ra.compute_ats_match(resume_skills, jd_skills)
    assert 0 <= ats["match_score"] <= 100
    assert "Python" in ats["matched_skills"]
    assert "SQL" in ats["matched_skills"]
    assert "Docker" in ats["missing_skills"]


# -------------------------------------------------------------
# 3. Multi-Tier Predictor Tests
# -------------------------------------------------------------
def test_tier_predictor(sample_student):
    tiers = tp.predict_tier_probabilities(sample_student, placement_prob=0.60)
    assert "tier1" in tiers
    assert "tier2" in tiers
    assert "tier3" in tiers
    assert "recommended_tier" in tiers
    assert 0 <= tiers["tier1"]["probability"] <= 1
    assert 0 <= tiers["tier2"]["probability"] <= 1
    assert 0 <= tiers["tier3"]["probability"] <= 1


def test_tier_gap_analysis(sample_student):
    tiers = tp.predict_tier_probabilities(sample_student, placement_prob=0.45)
    gaps = tp.get_tier_gap_analysis(sample_student, tiers)
    assert isinstance(gaps, list)


# -------------------------------------------------------------
# 4. Tech Profile Analyzer Tests
# -------------------------------------------------------------
def test_tech_profile_scoring():
    tech_data = {
        "leetcode_easy": 30,
        "leetcode_medium": 20,
        "leetcode_hard": 5,
        "leetcode_contest_rating": 1650,
        "github_repos": 10,
        "github_stars": 15,
        "github_contributions_last_year": 300,
        "github_languages": ["Python", "JavaScript", "SQL"],
        "certifications": ["AWS Certified Cloud Practitioner"],
        "projects_count": 4,
        "hackathons_participated": 2,
    }
    res = tech.compute_tech_readiness_score(tech_data)
    assert 0 <= res["overall_score"] <= 100
    assert "band" in res
    assert "coding_proficiency" in res["dimension_scores"]

    composite = tech.compute_enhanced_placement_score(0.70, res["overall_score"])
    assert 0 <= composite["enhanced_score"] <= 100


# -------------------------------------------------------------
# 5. TPO Command Center Tests
# -------------------------------------------------------------
def test_tpo_cohort_heatmap():
    dummy_students = [
        {"student_id": "S1", "name": "Alice", "probability": 0.85, "specialisation": "Mkt&Fin"},
        {"student_id": "S2", "name": "Bob", "probability": 0.55, "specialisation": "Mkt&HR"},
        {"student_id": "S3", "name": "Charlie", "probability": 0.30, "specialisation": "Mkt&Fin"},
    ]
    heatmap = tpo.generate_cohort_risk_heatmap(dummy_students, pd.DataFrame())
    assert len(heatmap["green_students"]) == 1
    assert len(heatmap["amber_students"]) == 1
    assert len(heatmap["red_students"]) == 1
    assert heatmap["summary"]["total"] == 3


def test_tpo_shortlist_generator():
    df_cohort = pd.DataFrame([
        {"student_id": "S1", "name": "Alice", "degree_p": 75.0, "etest_p": 80.0, "probability": 0.85, "specialisation": "Mkt&Fin", "workex": "Yes", "gender": "F"},
        {"student_id": "S2", "name": "Bob", "degree_p": 60.0, "etest_p": 65.0, "probability": 0.50, "specialisation": "Mkt&HR", "workex": "No", "gender": "M"},
    ])
    criteria = {
        "min_degree_p": 70.0,
        "min_etest_p": 70.0,
        "min_probability": 0.60,
        "specialisation": "All",
        "workex_required": False,
        "gender": "All",
    }
    shortlist = tpo.generate_company_shortlist(df_cohort, criteria)
    assert len(shortlist) == 1
    assert shortlist.iloc[0]["student_id"] == "S1"


# -------------------------------------------------------------
# 6. Mock Interview Simulator Tests
# -------------------------------------------------------------
def test_mock_interview_generation(sample_student):
    q_data = mi.generate_interview_questions("Mkt&Fin", sample_student)
    assert "questions" in q_data
    assert len(q_data["questions"]) >= 5


def test_mock_interview_evaluation():
    question = {
        "question_text": "Walk me through a DCF model.",
        "evaluation_criteria": "DCF free-cash-flow WACC terminal-value discount-rate enterprise-value equity-value",
        "sample_framework": "Project cash flows, estimate WACC, discount, subtract net debt",
    }
    answer = (
        "In a Discounted Cash Flow (DCF) model, we first project the unlevered free cash flows "
        "over a 5-year forecast horizon. We then compute the WACC as our discount rate. Next, "
        "we determine the terminal value using the Gordon Growth formula, discount all cash flows "
        "to present value to find Enterprise Value, and subtract net debt to derive Equity Value."
    )
    eval_res = mi.evaluate_answer_quality(answer, question)
    assert 0 <= eval_res["overall_score"] <= 100
    assert eval_res["relevance_score"] >= 30


# -------------------------------------------------------------
# 7. LLM Coach Tests
# -------------------------------------------------------------
def test_llm_coach_graceful_fallback_without_key():
    import src.llm_coach as llm
    assert llm.is_llm_available(api_key=None) is False
    assert llm.analyze_resume_with_llm("sample resume", "sample jd", api_key=None) is None
    assert llm.evaluate_interview_with_llm("question", "answer", "criteria", api_key=None) is None

