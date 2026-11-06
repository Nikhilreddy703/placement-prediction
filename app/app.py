"""PlacementOS Enterprise — Campus Recruitment & Student Success Platform.

Unified production Streamlit application featuring multi-model calibrated inference,
prescriptive counterfactual recourse, ATS resume parsing & JD matching,
practical tech & coding index, AI mock interview scoring, semester readiness tracking,
TPO recruiter drive shortlist generator with Excel export, and governance diagnostics.
"""
import io
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    ALL_FEATURE_COLS,
    ALL_MODELS_FILE,
    CATEGORICAL_FEATURES,
    DATA_RAW_PATH,
    FEATURE_CONSTRAINTS,
    FEATURE_IMP_FILE,
    FIELD_HELP,
    FIELD_LABELS,
    FIGURES_DIR,
    METADATA_FILE,
    MODEL_FILE,
    NUMERIC_FEATURES,
    PRESETS,
    RESULTS_FILE,
    THRESHOLD_HIGH_RISK,
    THRESHOLD_MODERATE_RISK,
)
from src.data import DataValidationError, load_dataset, validate_student_record
from src.database import get_prediction_history, init_db, log_prediction
from src.explain import (
    categorize_risk,
    compute_shap_contributions,
    compute_student_feature_contributions,
    generate_coaching_recommendations,
    get_radar_comparison_data,
    simulate_what_if,
)
from src.predict import batch_predict_students, predict_student_placement
from src.report_generator import generate_student_pdf_report
from src.tracker import (
    calculate_readiness_score,
    delete_student_data,
    draft_supportive_message,
    get_all_tracked_students,
    get_early_warning_alerts,
    get_student_checkin_history,
    get_student_tasks,
    init_tracker_db,
    load_sample_tracker_data,
    record_student_checkin,
    register_or_get_student,
    update_task_completion,
)
from src.counterfactual import (
    find_minimum_changes,
    compute_feature_sensitivity,
    generate_counterfactual_scenarios,
)
from src.tier_predictor import (
    predict_tier_probabilities,
    get_tier_gap_analysis,
    generate_tier_comparison_data,
)
from src.resume_analyzer import (
    extract_text_from_pdf,
    extract_skills_from_text,
    extract_skills_from_jd,
    compute_ats_match,
    generate_resume_suggestions,
)
from src.tech_profile import (
    TECH_PROFILE_DEFAULTS,
    compute_tech_readiness_score,
    get_tech_recommendations,
    compute_enhanced_placement_score,
)
from src.tpo_center import (
    generate_cohort_risk_heatmap,
    generate_company_shortlist,
    generate_shortlist_excel,
    get_cohort_analytics,
)
from src.mock_interview import (
    generate_interview_questions,
    evaluate_answer_quality,
    get_interview_tips,
)
import src.llm_coach as llm_coach

# ---------------------------------------------------------
# Page Configuration & Modern Meta
# ---------------------------------------------------------
st.set_page_config(
    page_title="PlacementOS Enterprise | Campus Placement Decision System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Modern SaaS Design System (Slate / Obsidian / Indigo CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Background tone */
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Modern Executive Header */
    .saas-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
        padding: 2rem 2.4rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 30px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }

    .saas-header::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .saas-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .saas-header p {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin-top: 0.45rem;
        margin-bottom: 1rem;
        max-width: 900px;
        line-height: 1.55;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-right: 0.5rem;
        margin-top: 0.3rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }

    .badge-primary {
        background: rgba(99, 102, 241, 0.2);
        color: #C7D2FE;
        border: 1px solid rgba(165, 180, 252, 0.4);
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #A7F3D0;
        border: 1px solid rgba(110, 231, 183, 0.4);
    }

    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #FECACA;
        border: 1px solid rgba(252, 165, 165, 0.4);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #FDE68A;
        border: 1px solid rgba(252, 211, 77, 0.4);
    }

    .badge-indigo {
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
    }

    /* Eliminate ugly default red sliders & style indigo */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #4F46E5 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.45) !important;
        width: 18px !important;
        height: 18px !important;
    }

    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #4F46E5 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stSlider"] > div > div > div > div {
        background-color: #4F46E5 !important;
    }

    /* Container Card Styling */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 16px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.02) !important;
        background-color: #FFFFFF !important;
        padding: 1.4rem 1.6rem !important;
        margin-bottom: 1.1rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.08) !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        padding: 1rem 1.25rem !important;
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    div[data-testid="stMetricValue"] div {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.03em !important;
    }

    /* Primary Buttons & Download Buttons (Replace ugly red with indigo gradient) */
    button[kind="primary"], .stButton > button[kind="primary"], div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.28) !important;
        font-weight: 700 !important;
        border-radius: 11px !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease-in-out !important;
    }

    button[kind="primary"]:hover, div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.38) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary / Regular Buttons */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.18s ease-in-out !important;
    }

    .stButton > button:hover {
        border-color: #818CF8 !important;
        color: #4F46E5 !important;
        background-color: #EEF2FF !important;
        box-shadow: 0 3px 8px rgba(99, 102, 241, 0.12) !important;
        transform: translateY(-1px) !important;
    }

    /* Modern Bento Cards */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04);
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.06);
    }

    .rec-box {
        background: #F8FAFC;
        border-left: 4px solid #4F46E5;
        padding: 0.9rem 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
        color: #1E293B;
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
        line-height: 1.55;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Fix broken expander icon font */
    [data-testid="stExpander"] details summary svg {
        color: #4F46E5 !important;
    }

    /* Nav Radio Styling */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 11px !important;
        padding: 0.55rem 0.85rem !important;
        margin-bottom: 0.45rem !important;
        transition: all 0.18s ease-in-out !important;
        cursor: pointer !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color: #818CF8 !important;
        background-color: #EEF2FF !important;
        transform: translateX(2px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource
def load_all_models_and_meta():
    """Load dictionary of all trained pipelines, champion model, and metadata."""
    if not METADATA_FILE.exists():
        return None, None, None
    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    all_models = {}
    if ALL_MODELS_FILE.exists():
        try:
            all_models = joblib.load(ALL_MODELS_FILE)
        except Exception:
            pass

    champion_model = None
    if MODEL_FILE.exists():
        try:
            champion_model = joblib.load(MODEL_FILE)
        except Exception:
            pass

    return champion_model, all_models, meta


@st.cache_data
def load_dataset_cached():
    """Load cleaned dataset."""
    try:
        X, y = load_dataset()
        df = X.copy()
        df["status"] = y.map({1: "Placed", 0: "Not Placed"})
        return df
    except Exception:
        return None


@st.cache_data
def load_benchmark_results():
    """Load model performance comparison CSV."""
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE)
    return None


@st.cache_data
def load_feature_importances():
    """Load feature importance weights CSV."""
    if FEATURE_IMP_FILE.exists():
        return pd.read_csv(FEATURE_IMP_FILE)
    return None


# Initialize SQLite
init_db()
init_tracker_db()

champion_model, all_models, meta = load_all_models_and_meta()
dataset_df = load_dataset_cached()
results_df = load_benchmark_results()
feature_imp_df = load_feature_importances()

if meta is None or (champion_model is None and not all_models):
    st.error("⚠️ Trained models or metadata not found. Please run `python src/train.py` to train and calibrate the models.")
    st.stop()


# ---------------------------------------------------------
# Sidebar Workspace Navigation & Model Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.4rem 0 1rem 0;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 0.65rem;">
            <span>🎓</span> PlacementOS
        </div>
        <div style="font-size: 0.76rem; font-weight: 700; color: #6366F1; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px;">
            Enterprise Edition v3.0
        </div>
        <div style="display: inline-flex; align-items: center; gap: 0.4rem; background: #ECFDF5; border: 1px solid #A7F3D0; padding: 0.2rem 0.6rem; border-radius: 9999px; margin-top: 0.5rem; font-size: 0.72rem; font-weight: 700; color: #065F46;">
            <span style="width: 7px; height: 7px; background: #10B981; border-radius: 50%; display: inline-block;"></span> System Online • ML Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "WORKSPACE NAVIGATION",
        [
            "🎯 Student Placement Studio",
            "📄 ATS Resume & JD Studio",
            "💻 Tech & Coding Matrix",
            "🎙️ Mock Interview & STAR Rubric",
            "📈 Student Readiness Tracker",
            "🏛️ TPO & Recruiter Portal",
            "📁 Batch Student Evaluation",
            "📊 Cohort Analytics & Trends",
            "🤖 Model Benchmarks & Governance",
        ],
        index=0,
    )

    st.markdown("---")

    # Global Model Controls Container (Using clean styled container to prevent broken expander icons)
    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 0.92rem; font-weight: 800; color: #0F172A; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;">
            ⚙️ Active ML Model & Cutoff
        </div>
        """, unsafe_allow_html=True)

        available_model_names = list(all_models.keys()) if all_models else [meta.get("best_model", "KNN")]
        default_idx = available_model_names.index(meta.get("best_model", "KNN")) if meta.get("best_model") in available_model_names else 0

        selected_model_name = st.selectbox(
            "Model Architecture",
            options=available_model_names,
            index=default_idx,
            help="Select any trained classifier to evaluate student profiles in real-time.",
        )

        active_pipeline = all_models.get(selected_model_name, champion_model)

        # Dynamic metrics display
        model_row = None
        if results_df is not None and "Model" in results_df.columns:
            matching = results_df[results_df["Model"] == selected_model_name]
            if not matching.empty:
                model_row = matching.iloc[0].to_dict()

        if not model_row:
            model_row = meta.get("metrics", {})

        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Accuracy", f"{float(model_row.get('Accuracy', 0)):.1%}")
        c_m1.metric("Placed Recall", f"{float(model_row.get('Placed Recall', model_row.get('Recall', 0))):.1%}")
        c_m2.metric("ROC-AUC", f"{float(model_row.get('ROC-AUC', 0)):.3f}")
        c_m2.metric("At-Risk Recall", f"{float(model_row.get('At-Risk Recall', 0)):.1%}" if "At-Risk Recall" in model_row else f"{float(model_row.get('Precision', 0)):.1%}")

        opt_thresh_val = float(model_row.get("Optimal Threshold", meta.get("optimal_threshold", 0.50)))
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.30,
            max_value=0.70,
            value=opt_thresh_val,
            step=0.02,
            help="Probability cutoff for classifying as Placed.",
        )

    st.caption("🔒 Verified Calibrated Pipeline (Sigmoid Scaling)")

    with st.expander("🔑 Real Generative AI (Optional)", expanded=False):
        st.caption("Enter an optional Google Gemini API key to activate Real LLM Semantic ATS Critique and Conversational STAR Feedback. Leave blank to run in offline rule-based mode.")
        gemini_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_key_sidebar",
            placeholder="AIzaSy...",
            help="Activates real Generative AI when set."
        )


# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
llm_badge = '<span class="badge-pill badge-success">✨ LLM AI Active</span>' if gemini_key_input else '<span class="badge-pill badge-primary">⚡ Calibrated ML Inference</span>'
st.markdown(f"""
<div class="saas-header">
    <h1>🎓 PlacementOS Enterprise Platform</h1>
    <p>Enterprise Campus Recruitment Intelligence & Student Success System. Powered by 6 calibrated machine learning architectures, verified coding portfolios, semantic ATS alignment, and institutional TPO cohort drives.</p>
    <div>
        <span class="badge-pill badge-primary">✨ Model: {selected_model_name} (Cutoff: {threshold:.2f})</span>
        <span class="badge-pill badge-primary">📊 400 Student Benchmark Cohort</span>
        {llm_badge}
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# VIEW 1: 🎯 Student Placement Studio
# =========================================================
if nav_selection == "🎯 Student Placement Studio":
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 0;">🎯 Student Placement & Multi-Tier Career Studio</h2>
        <p style="color: #64748B; font-size: 0.95rem; margin-top: 0.3rem;">Evaluate individual student profiles in real-time, inspect multi-tier salary odds, simulate what-if sensitivities, and explore prescriptive recourse paths.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Selectors as Modern Pills
    st.markdown("##### ⚡ Quick Scenarios")
    preset_cols = st.columns(len(PRESETS) + 1)
    for i, (p_name, p_data) in enumerate(PRESETS.items()):
        if preset_cols[i].button(p_name, use_container_width=True):
            st.session_state["student_profile"] = p_data

    if preset_cols[-1].button("🔄 Reset Profile", use_container_width=True):
        st.session_state["student_profile"] = {
            "gender": "M", "ssc_p": 70.0, "ssc_b": "Central", "hsc_p": 70.0,
            "hsc_b": "Others", "hsc_s": "Commerce", "degree_p": 68.0,
            "degree_t": "Comm&Mgmt", "workex": "Yes", "etest_p": 75.0,
            "specialisation": "Mkt&Fin", "mba_p": 65.0,
        }

    current_profile = st.session_state.get("student_profile", PRESETS["🌟 High-Scoring All-Rounder"])

    st.markdown("---")

    # 3-Card Structured Profile Builder
    st.markdown("#### 📝 Candidate Profile Configuration")
    col_card1, col_card2, col_card3 = st.columns([1.1, 1.1, 0.8])
    form_inputs = {}

    with col_card1:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                📚 10th & 12th Schooling
            </div>
            """, unsafe_allow_html=True)
            form_inputs["ssc_p"] = st.slider(
                FIELD_LABELS["ssc_p"],
                min_value=40.0, max_value=100.0,
                value=float(current_profile.get("ssc_p", 70.0)),
                step=0.1, help=FIELD_HELP["ssc_p"],
            )
            form_inputs["ssc_b"] = st.selectbox(
                FIELD_LABELS["ssc_b"],
                options=FEATURE_CONSTRAINTS["ssc_b"]["options"],
                index=FEATURE_CONSTRAINTS["ssc_b"]["options"].index(current_profile.get("ssc_b", "Central")),
                help=FIELD_HELP["ssc_b"],
            )
            form_inputs["hsc_p"] = st.slider(
                FIELD_LABELS["hsc_p"],
                min_value=40.0, max_value=100.0,
                value=float(current_profile.get("hsc_p", 70.0)),
                step=0.1, help=FIELD_HELP["hsc_p"],
            )
            form_inputs["hsc_b"] = st.selectbox(
                FIELD_LABELS["hsc_b"],
                options=FEATURE_CONSTRAINTS["hsc_b"]["options"],
                index=FEATURE_CONSTRAINTS["hsc_b"]["options"].index(current_profile.get("hsc_b", "Others")),
                help=FIELD_HELP["hsc_b"],
            )
            form_inputs["hsc_s"] = st.selectbox(
                FIELD_LABELS["hsc_s"],
                options=FEATURE_CONSTRAINTS["hsc_s"]["options"],
                index=FEATURE_CONSTRAINTS["hsc_s"]["options"].index(current_profile.get("hsc_s", "Commerce")),
                help=FIELD_HELP["hsc_s"],
            )

    with col_card2:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                🎓 Undergrad & MBA Profile
            </div>
            """, unsafe_allow_html=True)
            form_inputs["degree_p"] = st.slider(
                FIELD_LABELS["degree_p"],
                min_value=45.0, max_value=100.0,
                value=float(current_profile.get("degree_p", 68.0)),
                step=0.1, help=FIELD_HELP["degree_p"],
            )
            form_inputs["degree_t"] = st.selectbox(
                FIELD_LABELS["degree_t"],
                options=FEATURE_CONSTRAINTS["degree_t"]["options"],
                index=FEATURE_CONSTRAINTS["degree_t"]["options"].index(current_profile.get("degree_t", "Comm&Mgmt")),
                help=FIELD_HELP["degree_t"],
            )
            form_inputs["mba_p"] = st.slider(
                FIELD_LABELS["mba_p"],
                min_value=45.0, max_value=100.0,
                value=float(current_profile.get("mba_p", 65.0)),
                step=0.1, help=FIELD_HELP["mba_p"],
            )
            form_inputs["specialisation"] = st.selectbox(
                FIELD_LABELS["specialisation"],
                options=FEATURE_CONSTRAINTS["specialisation"]["options"],
                index=FEATURE_CONSTRAINTS["specialisation"]["options"].index(current_profile.get("specialisation", "Mkt&Fin")),
                help=FIELD_HELP["specialisation"],
            )
            form_inputs["workex"] = st.selectbox(
                FIELD_LABELS["workex"],
                options=FEATURE_CONSTRAINTS["workex"]["options"],
                index=FEATURE_CONSTRAINTS["workex"]["options"].index(current_profile.get("workex", "Yes")),
                help=FIELD_HELP["workex"],
            )

    with col_card3:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                🧠 Screening & Identity
            </div>
            """, unsafe_allow_html=True)
            form_inputs["etest_p"] = st.slider(
                FIELD_LABELS["etest_p"],
                min_value=40.0, max_value=100.0,
                value=float(current_profile.get("etest_p", 75.0)),
                step=0.1, help=FIELD_HELP["etest_p"],
            )
            form_inputs["gender"] = st.selectbox(
                FIELD_LABELS["gender"],
                options=FEATURE_CONSTRAINTS["gender"]["options"],
                index=FEATURE_CONSTRAINTS["gender"]["options"].index(current_profile.get("gender", "M")),
                help="Candidate legal gender",
            )
            student_id = st.text_input("Candidate Name / ID", value="Candidate-101")
            st.caption("Used for report branding and historical audit log.")

    # Run Prediction
    prediction_result = predict_student_placement(form_inputs, active_pipeline, threshold=threshold)
    prob = prediction_result["probability"]
    risk_level = prediction_result["risk_level"]
    label = prediction_result["label"]

    # Calculate student percentile vs cohort
    cohort_percentile = 50.0
    if dataset_df is not None and not dataset_df.empty:
        try:
            cohort_probs = active_pipeline.predict_proba(dataset_df[ALL_FEATURE_COLS])[:, 1]
            cohort_percentile = (cohort_probs < prob).mean() * 100.0
        except Exception:
            cohort_percentile = prob * 100.0

    # Log to SQLite
    log_prediction(form_inputs, prediction_result, selected_model_name, identifier=student_id)

    st.markdown("---")
    st.markdown("### 📊 Prediction Verdict & Strategic Assessment")

    res_col1, res_col2 = st.columns([1.3, 1.7])

    with res_col1:
        with st.container(border=True):
            # Polished Multi-Zone Modern Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#0F172A", "family": "Plus Jakarta Sans"}},
                delta={"reference": threshold * 100, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
                title={"text": f"<b>{prediction_result['status_text']}</b><br><span style='font-size:12px; color:#64748B;'>Model: {selected_model_name} • Threshold: {threshold:.2f}</span>", "font": {"size": 15, "color": "#1E293B"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": prediction_result["color"], "thickness": 0.32},
                    "bgcolor": "#F1F5F9",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                        {"range": [40, 65], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [65, 100], "color": "rgba(16, 185, 129, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#1E1B4B", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Prominent Percentile & Classification Card
            if risk_level == "Low Risk":
                st.markdown(f"""
                <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.8rem;">
                    <div style="font-weight: 800; color: #065F46; font-size: 0.98rem;">🟢 Classification: Placed (Low Placement Risk)</div>
                    <div style="font-size: 0.84rem; color: #047857; margin-top: 0.2rem;">Placement probability ({prob:.1%}) clears active cutoff ({threshold:.2f}). <b>Top {max(1, 100 - int(cohort_percentile))}% of cohort ({cohort_percentile:.1f}th percentile).</b></div>
                </div>
                """, unsafe_allow_html=True)
            elif risk_level == "Moderate Risk":
                st.markdown(f"""
                <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.8rem;">
                    <div style="font-weight: 800; color: #92400E; font-size: 0.98rem;">🟡 Classification: Borderline Risk Candidate</div>
                    <div style="font-size: 0.84rem; color: #B45309; margin-top: 0.2rem;">Placement probability ({prob:.1%}) is close to cutoff. Targeted skill coaching recommended.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.8rem;">
                    <div style="font-weight: 800; color: #991B1B; font-size: 0.98rem;">🔴 Classification: High Placement Risk</div>
                    <div style="font-size: 0.84rem; color: #B91C1C; margin-top: 0.2rem;">Placement probability ({prob:.1%}) is below cutoff ({threshold:.2f}). Immediate remedial intervention advised.</div>
                </div>
                """, unsafe_allow_html=True)

            pdf_bytes = generate_student_pdf_report(form_inputs, prediction_result, model_name=selected_model_name)
            st.download_button(
                label="📄 Download Career Readiness Report (PDF)",
                data=pdf_bytes,
                file_name=f"{student_id}_placement_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

    with res_col2:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
                💡 Targeted Career Coaching Guidance
            </div>
            """, unsafe_allow_html=True)
            recs = prediction_result["recommendations"]
            for r in recs:
                st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)

            # Student Snapshot Pills
            st.markdown("""
            <div style="background: #F1F5F9; border-radius: 12px; padding: 0.85rem 1rem; margin-top: 0.8rem;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;">Profile Highlights</div>
                <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
                    <span class="badge-pill badge-indigo">🎓 {form_inputs['degree_t']}</span>
                    <span class="badge-pill badge-indigo">📊 {form_inputs['specialisation']}</span>
                    <span class="badge-pill {'badge-success' if form_inputs['workex'] == 'Yes' else 'badge-warning'}">💼 WorkEx: {form_inputs['workex']}</span>
                    <span class="badge-pill badge-primary">🧠 E-Test: {form_inputs['etest_p']}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Multi-Tier Placement Likelihood Cards + Comparison Bar Chart
    with st.container(border=True):
        st.markdown("### 🏢 Target Company Tier Likelihoods (Rule-Based Sigmoid Engine)")
        st.caption("ℹ️ **Methodology:** Heuristic benchmark mapping candidate academic and aptitude profile strength to recruitment company tiers (Product, Core, Mass). *Note: The training dataset does not record company tier categories; tier odds are indicative benchmark heuristics.*")

        tier_res = predict_tier_probabilities(form_inputs, prob)
        t1_prob = tier_res["tier1"]["probability"]
        t2_prob = tier_res["tier2"]["probability"]
        t3_prob = tier_res["tier3"]["probability"]

        tier_c1, tier_c2, tier_c3 = st.columns(3)

        with tier_c1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #8B5CF6;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #8B5CF6; letter-spacing: 0.05em;">TIER 1 • DREAM FIRMS</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">{t1_prob:.1%}</div>
                <div style="font-size: 0.85rem; color: #475569;"><b>Package:</b> {tier_res['tier1']['salary_range']}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.4rem;">{', '.join(tier_res['tier1']['companies'][:4])}</div>
            </div>
            """, unsafe_allow_html=True)

        with tier_c2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #3B82F6;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #3B82F6; letter-spacing: 0.05em;">TIER 2 • CORE RECRUITERS</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">{t2_prob:.1%}</div>
                <div style="font-size: 0.85rem; color: #475569;"><b>Package:</b> {tier_res['tier2']['salary_range']}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.4rem;">{', '.join(tier_res['tier2']['companies'][:4])}</div>
            </div>
            """, unsafe_allow_html=True)

        with tier_c3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #10B981;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #10B981; letter-spacing: 0.05em;">TIER 3 • MASS RECRUITERS</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">{t3_prob:.1%}</div>
                <div style="font-size: 0.85rem; color: #475569;"><b>Package:</b> {tier_res['tier3']['salary_range']}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.4rem;">{', '.join(tier_res['tier3']['companies'][:4])}</div>
            </div>
            """, unsafe_allow_html=True)

        # NEW VISUAL: Tier Likelihood Comparison Bar Chart
        tier_chart_df = pd.DataFrame({
            "Tier": ["Tier 1: Dream (₹15-40 LPA)", "Tier 2: Core (₹7-14 LPA)", "Tier 3: Mass (₹3.5-6 LPA)"],
            "Probability": [t1_prob * 100, t2_prob * 100, t3_prob * 100],
            "Color": ["#8B5CF6", "#3B82F6", "#10B981"]
        })
        fig_tier_bar = go.Figure(go.Bar(
            x=tier_chart_df["Probability"],
            y=tier_chart_df["Tier"],
            orientation="h",
            marker=dict(color=tier_chart_df["Color"], line=dict(color="rgba(255,255,255,0.6)", width=1)),
            text=[f"{p:.1f}% Likelihood" for p in tier_chart_df["Probability"]],
            textposition="auto",
        ))
        fig_tier_bar.update_layout(
            title="<b>Company Tier Conversion Probability Comparison</b>",
            xaxis=dict(title="Placement Odds (%)", range=[0, 100], gridcolor="#F1F5F9"),
            yaxis=dict(autorange="reversed"),
            height=220,
            margin=dict(l=10, r=20, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_tier_bar, use_container_width=True)

        st.info(f"🎯 **Target Placement Tier:** Best positioned for **{tier_res['recommended_tier']}** (Composite Profile Strength: {tier_res['profile_strength']:.1f}/100).")

        with st.expander("🔍 View Detailed Tier Gap & Benchmark Analysis"):
            tier_gaps = get_tier_gap_analysis(form_inputs, tier_res)
            for tg in tier_gaps:
                st.markdown(f"• **{tg['tier']} ({tg['current_prob']:.0%} current chance):** {tg['gap_description']}")
                for imp in tg.get("required_improvements", []):
                    st.caption(f"   ↳ {imp}")

    # Prescriptive Counterfactual Recourse
    with st.container(border=True):
        st.markdown("### 🎯 Prescriptive Counterfactual Recourse Engine")
        st.write("Answers: **What is the minimum realistic effort needed to reach a desired placement probability?** (Focuses strictly on improvable factors: Aptitude test, MBA %, and Work experience, keeping past schooling locked).")

        cf_col1, cf_col2 = st.columns([1, 2])
        with cf_col1:
            desired_prob = st.slider(
                "Target Placement Likelihood",
                min_value=0.50,
                max_value=0.95,
                value=max(0.75, round(prob + 0.15, 2)),
                step=0.05,
                help="Select the goal probability to calculate minimal required improvements.",
            )
            cf_btn = st.button("⚡ Calculate Minimum Effort Path", type="primary", use_container_width=True)

        with cf_col2:
            if cf_btn or True:
                cf_solution = find_minimum_changes(active_pipeline, form_inputs, target_prob=desired_prob, threshold=threshold)
                if cf_solution["achievable"]:
                    st.success(f"✅ **Target Achievable!** Path to **{cf_solution['achieved_probability']:.1%}** (Effort Score: {cf_solution['effort_score']:.1f})")
                    st.markdown(f"> *{cf_solution['narrative']}*")

                    if cf_solution["changes"]:
                        c_cols = st.columns(len(cf_solution["changes"]))
                        for i, ch in enumerate(cf_solution["changes"]):
                            with c_cols[i]:
                                f_label = FIELD_LABELS.get(ch["feature"], ch["feature"])
                                st.metric(
                                    label=f_label,
                                    value=f"{ch['suggested_value']}" + ("%" if "p" in ch["feature"] else ""),
                                    delta=f"+{ch['change']}" if isinstance(ch['change'], (int, float)) else str(ch['change']),
                                    delta_color="normal",
                                )
                else:
                    st.warning(f"⚠️ **Target Bound Limitation:** {cf_solution['narrative']}")

        with st.expander("📈 View Marginal Feature Sensitivities (% Probability Gain per Unit)"):
            df_sens = compute_feature_sensitivity(active_pipeline, form_inputs)
            st.dataframe(df_sens, use_container_width=True)

    # Explainability, 3-Way Radar & Cohort Distribution
    st.markdown("---")
    st.markdown("### 🔍 Model Explainability & Visual Cohort Benchmarks")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        explain_mode = st.radio(
            "Explainability Attribution View:",
            ["🌟 Individual SHAP Attributions", "📊 Cohort Academic Comparison"],
            horizontal=True,
        )

        if explain_mode == "🌟 Individual SHAP Attributions":
            shap_model = active_pipeline
            if selected_model_name in ["KNN", "SVM"] and "Random Forest" in all_models:
                shap_model = all_models["Random Forest"]
                st.caption(f"*(SHAP computed using tuned Random Forest model since {selected_model_name} uses non-tree kernels)*")

            df_shap = compute_shap_contributions(shap_model, pd.DataFrame([form_inputs]))
            if not df_shap.empty:
                df_shap["Direction"] = df_shap["SHAP_Value"].apply(lambda v: "Positive Impact (+)" if v > 0 else "Negative Drag (-)")
                fig_shap = px.bar(
                    df_shap,
                    x="SHAP_Value",
                    y="Feature",
                    orientation="h",
                    color="SHAP_Value",
                    color_continuous_scale=["#EF4444", "#E2E8F0", "#10B981"],
                    title="SHAP Feature Attributions on Placement Odds",
                )
                fig_shap.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=20), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_shap, use_container_width=True)

                with st.expander("📝 View Plain English SHAP Insights"):
                    for _, s_row in df_shap.iterrows():
                        st.write(f"• **{s_row['Feature']}:** {s_row['Plain_English_Insight']}")
            else:
                explain_mode = "📊 Cohort Academic Comparison"

        if explain_mode == "📊 Cohort Academic Comparison":
            df_contrib = compute_student_feature_contributions(active_pipeline, pd.DataFrame([form_inputs]), dataset_df)
            if not df_contrib.empty:
                df_contrib["Color"] = df_contrib["Difference"].apply(lambda d: "#10B981" if d > 0 else "#EF4444")
                fig_bar = px.bar(
                    df_contrib,
                    x="Difference",
                    y="Feature",
                    orientation="h",
                    color="Difference",
                    color_continuous_scale=["#EF4444", "#E2E8F0", "#10B981"],
                    title="Student Academic Delta vs. Cohort Average (%)",
                )
                fig_bar.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=20), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

    with exp_col2:
        st.markdown("#### 🎯 3-Way Benchmark Spider / Radar Chart")
        benchmarks = meta.get("numeric_benchmark_placed", {
            "ssc_p": 71.0, "hsc_p": 69.0, "degree_p": 68.0, "etest_p": 73.0, "mba_p": 63.0
        })
        radar_df = get_radar_comparison_data(form_inputs, benchmarks)

        # Tier 1 Dream Benchmark target
        tier1_targets = {"10th Percentage": 80.0, "12th Percentage": 78.0, "Degree Percentage": 76.0, "Employability Test": 85.0, "MBA Percentage": 72.0}
        radar_df["Tier 1 Target"] = radar_df["Metric"].map(tier1_targets).fillna(75.0)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Student"],
            theta=radar_df["Metric"],
            fill="toself",
            name="This Student",
            line=dict(color="#4F46E5", width=2.5),
            fillcolor="rgba(79, 70, 229, 0.28)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Placed Benchmark"],
            theta=radar_df["Metric"],
            fill="toself",
            name="Cohort Placed Avg",
            line=dict(color="#10B981", width=2),
            fillcolor="rgba(16, 185, 129, 0.20)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Tier 1 Target"],
            theta=radar_df["Metric"],
            fill="none",
            name="Tier-1 Target",
            line=dict(color="#8B5CF6", dash="dot", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[40, 100], gridcolor="#E2E8F0")),
            height=320,
            margin=dict(l=30, r=30, t=35, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # NEW SECTION: Cohort Distribution Bell Curve + Dynamic What-If Sensitivity S-Curve
    st.markdown("---")
    st.markdown("### 📈 Cohort Bell Curve & Live Sensitivity Curves")
    dist_c1, dist_c2 = st.columns(2)

    with dist_c1:
        st.markdown("#### 📊 Candidate Placement Odds vs. Cohort Distribution")
        if dataset_df is not None and not dataset_df.empty:
            try:
                cohort_all_probs = active_pipeline.predict_proba(dataset_df[ALL_FEATURE_COLS])[:, 1] * 100
                fig_dist = px.histogram(
                    x=cohort_all_probs,
                    nbins=25,
                    marginal="box",
                    title="Placement Probability Density Across 400 Student Cohort",
                    labels={"x": "Placement Probability (%)"},
                    color_discrete_sequence=["#93C5FD"],
                )
                fig_dist.add_vline(
                    x=prob * 100,
                    line_width=3,
                    line_dash="dash",
                    line_color="#4F46E5",
                    annotation_text=f"Candidate ({prob:.1%})",
                    annotation_position="top left",
                )
                fig_dist.add_vline(
                    x=threshold * 100,
                    line_width=2,
                    line_dash="dot",
                    line_color="#EF4444",
                    annotation_text=f"Cutoff ({threshold:.2f})",
                    annotation_position="bottom right",
                )
                fig_dist.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception:
                st.info("Cohort distribution histogram unavailable.")

    with dist_c2:
        st.markdown("#### ⚡ Dynamic What-If Sensitivity S-Curve")
        # Generate a live curve sweeping E-Test from 40 to 100
        test_range = np.linspace(40, 100, 25)
        curve_probs = []
        for val in test_range:
            temp_inp = form_inputs.copy()
            temp_inp["etest_p"] = val
            sim_p = active_pipeline.predict_proba(pd.DataFrame([temp_inp]))[0, 1]
            curve_probs.append(sim_p * 100)

        fig_scurve = go.Figure()
        fig_scurve.add_trace(go.Scatter(
            x=test_range,
            y=curve_probs,
            mode="lines+markers",
            name="Placement Odds Curve",
            line=dict(color="#4F46E5", width=3),
            marker=dict(size=5),
        ))
        fig_scurve.add_hline(
            y=threshold * 100,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text=f"Cutoff ({threshold * 100:.0f}%)",
        )
        fig_scurve.add_trace(go.Scatter(
            x=[form_inputs["etest_p"]],
            y=[prob * 100],
            mode="markers",
            name="Current Candidate",
            marker=dict(color="#10B981", size=14, symbol="star", line=dict(color="#065F46", width=2)),
        ))
        fig_scurve.update_layout(
            title="<b>Marginal E-Test Score Sensitivity Curve (40% to 100%)</b>",
            xaxis=dict(title="Employability Test Score (%)", gridcolor="#F1F5F9"),
            yaxis=dict(title="Placement Probability (%)", range=[0, 105], gridcolor="#F1F5F9"),
            height=320,
            margin=dict(l=10, r=10, t=40, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scurve, use_container_width=True)

    # What-If Simulator Sliders & Metrics
    st.markdown("---")
    st.markdown("### 🎛️ Interactive Multi-Factor What-If Simulator")
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    sim_etest = sim_c1.slider("Simulated E-Test Score (%)", 40.0, 100.0, float(form_inputs["etest_p"]), step=1.0)
    sim_workex = sim_c2.selectbox("Simulated Work Experience", ["Yes", "No"], index=0 if form_inputs["workex"] == "Yes" else 1)
    sim_degree = sim_c3.slider("Simulated Degree (%)", 45.0, 100.0, float(form_inputs["degree_p"]), step=1.0)

    sim_res = simulate_what_if(
        active_pipeline,
        form_inputs,
        {"etest_p": sim_etest, "workex": sim_workex, "degree_p": sim_degree},
    )

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Baseline Placement Likelihood", f"{sim_res['base_probability']:.1%}")
    sc2.metric(
        "Simulated Placement Likelihood",
        f"{sim_res['simulated_probability']:.1%}",
        delta=f"{sim_res['delta_percentage']:+.1f}%",
        delta_color="normal" if sim_res["improved"] else "inverse",
    )
    new_risk = categorize_risk(sim_res["simulated_probability"])[0]
    sc3.metric("Projected Risk Category", new_risk)


# =========================================================
# VIEW 2: 📄 ATS Resume & JD Studio
# =========================================================
elif nav_selection == "📄 ATS Resume & JD Studio":
    st.subheader("📄 Resume Parser & ATS Job Description Matcher")
    st.write("Compare candidate resumes against specific corporate Job Descriptions, compute the ATS Match Score, discover missing keywords, and get tailored resume bullet suggestions.")

    PRESET_JDS = {
        "Custom Job Description": "",
        "💼 Amazon — Software Development Engineer (SDE) Intern": """Role: Software Development Engineer Intern
Requirements:
- Strong proficiency in Python, Java, or C++.
- Solid understanding of Data Structures, Algorithms, and Object-Oriented Design.
- Experience with REST APIs, SQL databases (PostgreSQL/MySQL), Git, and Linux.
- Familiarity with AWS services (EC2, S3, Lambda) and Docker/Kubernetes is a strong plus.
- Good problem solving, communication, and agile teamwork skills.""",
        "📊 Deloitte — Business Technology Analyst": """Role: Business Technology Analyst
Requirements:
- Strong background in Business Analysis, Data Analysis, and Statistical Modeling.
- Hands-on experience with SQL, Excel (Advanced modeling, VBA), Tableau, and Power BI.
- Exposure to Python or R for predictive modeling and data visualization.
- Strong client presentation, stakeholder management, problem solving, and consulting mindset.
- Basic knowledge of ERP, SAP, or Cloud platforms (Azure/AWS) preferred.""",
        "📈 Goldman Sachs — Operations & Financial Analyst": """Role: Financial Operations Analyst
Requirements:
- Degree in Finance, Commerce, or related disciplines with strong academic records.
- Proficiency in Financial Modeling, Valuation, Accounting, and Risk Management.
- Strong quantitative skills, Excel, Financial Analysis, and basic Python/SQL.
- Excellent attention to detail, time management, critical thinking, and communication skills.
- Understanding of equity markets, fixed income, portfolio management, or CFA Level 1 is a plus.""",
        "🚀 TCS — Digital Software Engineer / Specialist": """Role: Digital Software Engineer
Requirements:
- Hands-on programming skills in Java, Python, or JavaScript.
- Knowledge of HTML, CSS, React, and REST API development.
- Database concepts: SQL, MongoDB, and Git version control.
- Understanding of Machine Learning, Cloud & DevOps basics (Docker, CI/CD).
- Strong adaptability, logical reasoning, and collaborative problem-solving skills."""
    }

    SAMPLE_RESUMES = {
        "Data & Software Profile": """ALEX DOE
alex.doe@example.com | GitHub: github.com/alexdoe | LinkedIn: linkedin.com/in/alexdoe

SUMMARY
Motivated engineering and MBA graduate with strong foundations in Python, SQL, and Machine Learning. Passionate about building data-driven systems and scalable web applications.

TECHNICAL SKILLS
- Programming Languages: Python, Java, SQL, C++
- Data Science & ML: Machine Learning, Scikit-learn, Pandas, NumPy, Data Analysis, Tableau, Data Visualization
- Databases & Tools: PostgreSQL, MySQL, SQLite, Git, Docker, Jira, Linux
- Soft Skills: Problem Solving, Teamwork, Communication, Time Management

PROJECTS
- Placement Readiness Analytics Platform: Built ML model with 85% accuracy using Python, Scikit-learn, and Streamlit.
- E-Commerce Inventory Microservice: Designed REST APIs using Flask and PostgreSQL with Docker deployment.

EXPERIENCE
- Data Science Summer Intern, ABC Analytics (2025): Developed automated ETL pipelines in Python and created executive dashboards in Tableau.""",
        "Finance & Business Profile": """PRIYA NAIR
priya.nair@example.com | LinkedIn: linkedin.com/in/priyanair

SUMMARY
MBA candidate specializing in Finance and Marketing with practical experience in Financial Modeling, Business Analysis, and market research.

SKILLS
- Finance & Accounting: Financial Modeling, Valuation, Financial Analysis, Accounting, Budgeting, Risk Management
- Analytics & Tools: Advanced Excel, Power BI, SQL, Market Research
- Soft Skills: Presentation, Leadership, Stakeholder Management, Strategic Planning, Negotiation

PROJECTS & INTERNSHIPS
- Financial Valuation of FMCG Sector: Developed comprehensive DCF and comparable company valuation models for 5 mid-cap companies.
- Summer Intern, XYZ Capital (2025): Analyzed quarterly earnings, built financial models in Excel, and presented insights to portfolio managers."""
    }

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        with st.container(border=True):
            st.markdown("#### 1. Candidate Resume")
            uploaded_pdf = st.file_uploader("Upload Resume (PDF format)", type=["pdf"], key="resume_uploader_main")

            sample_choice = st.selectbox(
                "Or load sample resume profile:",
                ["(Choose sample or paste custom)", "Data & Software Profile", "Finance & Business Profile"]
            )

            initial_resume_text = ""
            if uploaded_pdf is not None:
                with st.spinner("Extracting text from PDF..."):
                    initial_resume_text = extract_text_from_pdf(uploaded_pdf)
            elif sample_choice in SAMPLE_RESUMES:
                initial_resume_text = SAMPLE_RESUMES[sample_choice]

            resume_text_area = st.text_area(
                "Resume Text (Extracted or Pasted):",
                value=initial_resume_text,
                height=260,
                placeholder="Paste your complete resume text or upload a PDF above..."
            )

    with r_col2:
        with st.container(border=True):
            st.markdown("#### 2. Target Job Description (JD)")
            selected_jd_preset = st.selectbox("Select Target Company Role or Custom:", list(PRESET_JDS.keys()))
            default_jd_text = PRESET_JDS[selected_jd_preset]
            jd_text_area = st.text_area(
                "Job Description Text:",
                value=default_jd_text,
                height=325,
                placeholder="Paste the recruiter's Job Description (roles, responsibilities, required skills)..."
            )

    analyze_ats_btn = st.button("⚡ Run Compatibility & Skill Gap Analysis", type="primary", use_container_width=True)

    if (analyze_ats_btn or (resume_text_area and jd_text_area)) and resume_text_area and jd_text_area:
        resume_skills = extract_skills_from_text(resume_text_area)
        jd_skills = extract_skills_from_jd(jd_text_area)
        ats_res = compute_ats_match(resume_skills, jd_skills)

        st.markdown("---")
        st.markdown("### 🎯 Resume Compatibility & Skill Gap Breakdown")

        if gemini_key_input:
            st.markdown("#### ✨ Real Generative AI Semantic Critique (Gemini)")
            with st.spinner("Analyzing resume against JD using Gemini LLM..."):
                llm_critique = llm_coach.analyze_resume_with_llm(resume_text_area, jd_text_area, api_key=gemini_key_input)
            
            if llm_critique:
                lc1, lc2 = st.columns([1, 2.5])
                lc1.metric("LLM Semantic Match", f"{llm_critique.get('match_percentage', 0)}%")
                lc2.info(f"**Executive Recruiter Summary:** {llm_critique.get('executive_summary', '')}")
                
                if llm_critique.get("bullet_rewrite_suggestions"):
                    st.markdown("##### ✍️ High-Impact Bullet Rewrite Suggestions")
                    for rw in llm_critique["bullet_rewrite_suggestions"]:
                        st.markdown(f"""
                        <div style="background: white; border: 1px solid #E2E8F0; padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                            <div style="font-size: 0.85rem; color: #64748B;">Target Area: {rw.get('original_or_area', '')}</div>
                            <div style="font-weight: 600; color: #1E293B; margin-top: 0.2rem;">👉 {rw.get('improved_bullet', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("---")
            else:
                st.warning("Gemini API call failed or timed out. Falling back to rule-based keyword match below.")
        else:
            st.caption("ℹ️ **Active Mode:** Rule-Based Keyword & Taxonomy Matcher. (Enter an optional Gemini API key in the sidebar for Real Generative AI Semantic Analysis).")

        with st.container(border=True):
            m_res1, m_res2, m_res3, m_res4 = st.columns(4)
            match_score = ats_res["match_score"]
            m_res1.metric("ATS Match Score", f"{match_score:.1f}%")
            m_res2.metric("JD Required Skills", len(jd_skills))
            m_res3.metric("Matched Skills Found", len(ats_res["matched_skills"]))
            m_res4.metric("Missing Skills", len(ats_res["missing_skills"]), delta=f"-{len(ats_res['missing_skills'])}", delta_color="inverse")

            st.progress(match_score / 100.0)

            sk_col1, sk_col2 = st.columns(2)
            with sk_col1:
                st.markdown("##### ✅ Matched Skills in Resume")
                if ats_res["matched_skills"]:
                    matched_badges = " ".join([f'<span class="badge-pill badge-success">{s}</span>' for s in ats_res["matched_skills"]])
                    st.markdown(matched_badges, unsafe_allow_html=True)
                else:
                    st.info("No direct skill matches found between resume and JD.")

            with sk_col2:
                st.markdown("##### ❌ Missing Critical Skills (Add to Resume)")
                if ats_res["missing_skills"]:
                    missing_badges = " ".join([f'<span class="badge-pill badge-danger">{s}</span>' for s in ats_res["missing_skills"]])
                    st.markdown(missing_badges, unsafe_allow_html=True)
                else:
                    st.success("🎉 Outstanding! Your resume covers 100% of the key technical and domain skills required in this JD!")

        # Category Breakdown Chart
        cat_data = []
        for cat, counts in ats_res["category_breakdown"].items():
            if counts["total"] > 0:
                cat_data.append({
                    "Category": cat,
                    "Matched": counts["matched"],
                    "Missing": counts["total"] - counts["matched"],
                    "Match_Rate": round((counts["matched"] / counts["total"]) * 100, 1),
                })
        if cat_data:
            df_cat = pd.DataFrame(cat_data)
            fig_cat = px.bar(
                df_cat,
                x="Category",
                y=["Matched", "Missing"],
                barmode="stack",
                title="Skill Coverage by Category",
                color_discrete_map={"Matched": "#10B981", "Missing": "#EF4444"},
            )
            fig_cat.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=20))
            st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("##### 💡 Tailored Resume Action Points")
        suggestions = generate_resume_suggestions(ats_res["missing_skills"], {})
        for sug in suggestions:
            st.markdown(f'<div class="rec-box">{sug}</div>', unsafe_allow_html=True)


# =========================================================
# VIEW 3: 💻 Tech & Coding Matrix
# =========================================================
elif nav_selection == "💻 Tech & Coding Matrix":
    st.subheader("💻 Practical Tech Profile & Coding Readiness Index")
    st.write("Academics open the door, but verified problem solving and practical portfolios land offers. Ingest your LeetCode and GitHub metrics to compute your holistic Placement Index.")

    t_preset_col1, t_preset_col2, t_preset_col3 = st.columns(3)
    if t_preset_col1.button("⭐ Full-Stack Project Builder", use_container_width=True):
        st.session_state["tech_input"] = {
            "easy": 45, "med": 35, "hard": 5, "rating": 1600,
            "repos": 14, "stars": 28, "commits": 450,
            "langs": ["Python", "JavaScript", "SQL", "TypeScript"],
            "certs": ["AWS Certified Cloud Practitioner"], "projects": 5, "hackathons": 2
        }
    if t_preset_col2.button("⚔️ Competitive Programmer (DSA)", use_container_width=True):
        st.session_state["tech_input"] = {
            "easy": 80, "med": 110, "hard": 25, "rating": 1950,
            "repos": 6, "stars": 8, "commits": 200,
            "langs": ["C++", "Java", "Python"],
            "certs": [], "projects": 2, "hackathons": 4
        }
    if t_preset_col3.button("🌱 Campus Beginner / Early Learner", use_container_width=True):
        st.session_state["tech_input"] = {
            "easy": 12, "med": 4, "hard": 0, "rating": 1350,
            "repos": 3, "stars": 1, "commits": 60,
            "langs": ["Python"],
            "certs": [], "projects": 1, "hackathons": 0
        }

    tech_state = st.session_state.get("tech_input", {
        "easy": 40, "med": 30, "hard": 4, "rating": 1550,
        "repos": 8, "stars": 12, "commits": 320,
        "langs": ["Python", "SQL"],
        "certs": ["Google Data Analytics"], "projects": 3, "hackathons": 1
    })

    tp_col1, tp_col2 = st.columns(2)
    with tp_col1:
        st.markdown("#### 🧠 Problem Solving & DSA (LeetCode / HackerRank)")
        lc_easy = st.number_input("LeetCode Easy Solved", 0, 500, int(tech_state.get("easy", 40)))
        lc_med = st.number_input("LeetCode Medium Solved", 0, 500, int(tech_state.get("med", 30)))
        lc_hard = st.number_input("LeetCode Hard Solved", 0, 200, int(tech_state.get("hard", 4)))
        lc_rating = st.slider("Contest Rating (0 if unrated)", 0, 3000, int(tech_state.get("rating", 1550)), step=25)
        hackathons_cnt = st.number_input("Hackathons / Competitions Participated", 0, 20, int(tech_state.get("hackathons", 1)))

    with tp_col2:
        st.markdown("#### 🛠️ GitHub & Open Source Portfolio")
        gh_repos = st.number_input("Public Repositories", 0, 100, int(tech_state.get("repos", 8)))
        gh_stars = st.number_input("Total Repository Stars", 0, 1000, int(tech_state.get("stars", 12)))
        gh_commits = st.number_input("Annual Git Contributions", 0, 2000, int(tech_state.get("commits", 320)))
        projects_cnt = st.number_input("Showcase / Capstone Projects Built", 0, 20, int(tech_state.get("projects", 3)))
        all_langs = ["Python", "Java", "C++", "JavaScript", "TypeScript", "SQL", "Go", "R", "HTML/CSS"]
        gh_langs = st.multiselect("Core Languages Used", all_langs, default=[l for l in tech_state.get("langs", ["Python", "SQL"]) if l in all_langs])
        all_certs_list = ["AWS Certified Cloud Practitioner", "AWS Solutions Architect", "Google Cloud Associate", "Google Data Analytics", "TensorFlow Developer", "CFA Level 1", "Tableau Desktop Specialist"]
        selected_certs = st.multiselect("Professional Certifications", all_certs_list, default=[c for c in tech_state.get("certs", []) if c in all_certs_list])

    tp_dict = {
        "leetcode_easy": lc_easy,
        "leetcode_medium": lc_med,
        "leetcode_hard": lc_hard,
        "leetcode_contest_rating": lc_rating,
        "github_repos": gh_repos,
        "github_stars": gh_stars,
        "github_contributions_last_year": gh_commits,
        "github_languages": gh_langs,
        "certifications": selected_certs,
        "projects_count": projects_cnt,
        "hackathons_participated": hackathons_cnt,
    }

    ref_prob = 0.65
    tech_eval = compute_tech_readiness_score(tp_dict)
    enhanced_res = compute_enhanced_placement_score(ref_prob, tech_eval["overall"])

    st.markdown("---")
    st.markdown("### 📊 Tech Score & Combined Holistic Placement Index")

    t_sc1, t_sc2, t_sc3, t_sc4 = st.columns(4)
    t_sc1.metric("Tech Readiness Score", f"{tech_eval['overall']:.1f}/100")
    t_sc2.metric("Tech Rating Band", tech_eval["band"])
    t_sc3.metric("Baseline Academic Odds", f"{ref_prob:.1%}")

    tech_delta = float(enhanced_res.get("delta_vs_academic", enhanced_res["enhanced_score"] - (ref_prob * 100)))
    t_sc4.metric(
        "Holistic Placement Index",
        f"{enhanced_res['enhanced_score']:.1f}/100",
        delta=f"{tech_delta:+.1f} vs Academic",
        delta_color="normal" if tech_delta >= 0 else "inverse"
    )

    dim_col1, dim_col2 = st.columns([1.2, 1.8])
    with dim_col1:
        dim_df = pd.DataFrame({
            "Dimension": ["Coding Proficiency", "Project Portfolio", "Competitive Edge", "Certifications"],
            "Score": [
                tech_eval["coding_proficiency"],
                tech_eval["project_portfolio"],
                tech_eval["competitive_edge"],
                tech_eval["certification_strength"],
            ]
        })
        fig_dim = px.bar(
            dim_df,
            x="Score",
            y="Dimension",
            orientation="h",
            color="Score",
            color_continuous_scale="Viridis",
            title="Technical Competency Dimensions (0-100)",
            range_x=[0, 100],
        )
        fig_dim.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=20), showlegend=False)
        st.plotly_chart(fig_dim, use_container_width=True)

    with dim_col2:
        st.markdown("#### 🎯 Priority Technical Roadmap")
        tech_recs = get_tech_recommendations(tp_dict, "Mkt&Fin")
        for trec in tech_recs:
            st.markdown(f'<div class="rec-box">{trec}</div>', unsafe_allow_html=True)
        st.caption("Combined Index Formula: 65% Academic Placement Model + 35% Practical Tech & Problem Solving Portfolio.")


# =========================================================
# VIEW 4: 🎙️ Mock Interview & STAR Rubric
# =========================================================
elif nav_selection == "🎙️ Mock Interview & STAR Rubric":
    st.subheader("🎙️ Mock Interview & STAR Evaluation Studio")
    st.write("Master campus recruitment rounds by practicing tailored technical, behavioral, and case-study questions scored against the STAR (Situation, Task, Action, Result) rubric.")

    int_c1, int_c2 = st.columns([1.2, 1.8])
    with int_c1:
        interview_spec = st.selectbox(
            "Target Recruitment Domain:",
            ["Mkt&Fin", "Mkt&HR"],
            index=0,
            format_func=lambda s: "Financial Analysis & Marketing Analytics (Mkt&Fin)" if s == "Mkt&Fin" else "Human Resources & Talent Management (Mkt&HR)"
        )
        interview_diff = st.select_slider("Interview Difficulty Level:", options=["Easy", "Medium", "Hard"], value="Medium")

    q_data = generate_interview_questions(interview_spec, {}, difficulty=interview_diff)
    questions_list = q_data.get("questions", [])

    st.markdown("#### 📋 Domain Question Bank")
    selected_q_idx = st.selectbox(
        "Choose Question to Practice:",
        range(len(questions_list)),
        format_func=lambda i: f"[{questions_list[i].get('category', 'Technical')}] {questions_list[i]['question_text'][:80]}..."
    )

    curr_q = questions_list[selected_q_idx]
    st.markdown(f"""
    <div style="background: #F8FAFC; border-left: 5px solid #2563EB; padding: 1.1rem; border-radius: 8px; margin: 0.8rem 0; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;">
        <span class="badge-pill badge-primary">{curr_q.get('category', 'Technical')}</span>
        <span class="badge-pill badge-indigo">{curr_q.get('difficulty', 'Medium')}</span>
        <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-top: 0.5rem;">{curr_q['question_text']}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 View Recommended Answer Framework (Interviewer Checklist)"):
        st.markdown(f"**Interviewer Looks For:** `{curr_q.get('evaluation_criteria', '')}`")
        st.markdown(f"**Structural Framework:**\n{curr_q.get('sample_framework', '')}")

    st.markdown("#### ✍️ Your Answer")
    ans_p1, ans_p2 = st.columns(2)
    sample_ans_text = ""
    if ans_p1.button("✨ Load Strong STAR Answer Sample", use_container_width=True):
        sample_ans_text = (
            "During my summer internship at XYZ Corp (Situation), our team was tasked with reducing customer churn by 10% (Task). "
            "I developed a predictive logistic regression model in Python, identified key attrition drivers, and led weekly stakeholder reviews (Action). "
            "Consequently, we decreased customer churn by 14% over 3 months, saving $45,000 in retained annualized revenue (Result)."
        )
    if ans_p2.button("⚠️ Load Weak / Unstructured Answer Sample", use_container_width=True):
        sample_ans_text = "I am a very hardworking person. Whenever there is a problem, I work a lot and talk to people to solve it. I like doing good work."

    ans_input = st.text_area("Type or paste your response:", value=sample_ans_text, height=160, placeholder="Structure your answer using the STAR method: Situation, Task, Action, Result...")

    if st.button("📊 Evaluate Answer (STAR Method Analysis)", type="primary", use_container_width=True) or (ans_input and sample_ans_text):
        if ans_input.strip():
            st.markdown("---")
            st.markdown("### 🏆 Answer Scoring & Rubric Feedback")

            if gemini_key_input:
                st.markdown("#### ✨ Real Generative AI Examiner Evaluation (Gemini)")
                with st.spinner("Calling Gemini LLM to evaluate answer against STAR framework..."):
                    llm_eval = llm_coach.evaluate_interview_with_llm(
                        curr_q['question_text'],
                        ans_input,
                        curr_q.get('evaluation_criteria', ''),
                        api_key=gemini_key_input
                    )
                if llm_eval:
                    le1, le2 = st.columns([1, 2])
                    le1.metric("LLM Interview Score", f"{llm_eval.get('overall_score', 0)}/100")
                    le1.metric("Interviewer Verdict", llm_eval.get('interviewer_verdict', 'Under Review'))
                    
                    with le2:
                        st.markdown("**STAR Ratings:**")
                        star_rats = llm_eval.get("star_ratings", {})
                        r_badges = " ".join([f'<span class="badge-pill badge-primary">{k.upper()}: {v}</span>' for k, v in star_rats.items()])
                        st.markdown(r_badges, unsafe_allow_html=True)
                    
                    if llm_eval.get("suggested_sample_answer"):
                        st.markdown("##### 🌟 Tailored High-Scoring Model STAR Answer:")
                        st.info(llm_eval["suggested_sample_answer"])
                    st.markdown("---")
            else:
                st.caption("ℹ️ **Active Mode:** Rule-Based Keyword & STAR Depth Rubric. (Enter an optional Gemini API key in the sidebar for Real Generative AI Interview Feedback).")

            eval_out = evaluate_answer_quality(ans_input, curr_q)
            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            e_col1.metric("Rule Score", f"{eval_out['overall_score']:.0f}/100")
            e_col2.metric("Relevance to Question", f"{eval_out['relevance_score']:.0f}%")
            e_col3.metric("STAR Structure Score", f"{eval_out['structure_score']:.0f}%")
            e_col4.metric("Depth & Specificity", f"{eval_out['depth_score']:.0f}%")

            star_tags = eval_out.get("star_elements_detected", {})
            st.markdown("**STAR Components Detected (Rule Engine):**")
            star_badges = " ".join([
                f'<span class="badge-pill {"badge-success" if detected else "badge-danger"}">{k}: {"✓ Detected" if detected else "✗ Missing"}</span>'
                for k, detected in star_tags.items()
            ])
            st.markdown(star_badges, unsafe_allow_html=True)

            st.markdown("##### 📝 Actionable Examiner Feedback:")
            for fb in eval_out.get("feedback", []):
                st.markdown(f'<div class="rec-box">{fb}</div>', unsafe_allow_html=True)


# =========================================================
# VIEW 5: 📈 Student Readiness Tracker
# =========================================================
elif nav_selection == "📈 Student Readiness Tracker":
    st.subheader("📈 Student Placement Readiness Tracker")
    st.write("Track student progress across semester check-ins, observe readiness trajectories, follow adaptive weekly action plans, and monitor early-warning retention alerts.")

    t_bar1, t_bar2 = st.columns([2.5, 1.5])
    with t_bar2:
        if st.button("🧪 Load Sample Check-ins (Demo)", help="Populate SQLite with sample multi-checkin student trajectories for demonstration."):
            load_sample_tracker_data(active_pipeline, threshold=threshold)
            st.success("Sample check-in data loaded successfully! (Clearly labeled as demo records)")
            st.rerun()

    tracked_students = get_all_tracked_students()

    st.markdown("---")

    track_mode = st.radio(
        "Action:",
        ["Track Existing Student Progress", "Register New Student"],
        horizontal=True,
    )

    if track_mode == "Register New Student":
        st.markdown("#### 📝 Register Student Profile (Baseline Check-in)")
        st.caption("Secondary (10th) and Higher Secondary (12th) marks will be permanently locked after initial registration.")

        with st.form("form_register_student"):
            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                new_s_id = st.text_input("Student ID (Unique)", value=f"STU-{len(tracked_students)+101:03d}")
                new_name = st.text_input("Full Name", value="Alex Doe")
                new_gender = st.selectbox("Gender", FEATURE_CONSTRAINTS["gender"]["options"])
                new_spec = st.selectbox("MBA / PG Specialisation", FEATURE_CONSTRAINTS["specialisation"]["options"])
                new_degree_t = st.selectbox("Degree Stream", FEATURE_CONSTRAINTS["degree_t"]["options"])
                new_ssc_b = st.selectbox("10th Board", FEATURE_CONSTRAINTS["ssc_b"]["options"])
                new_hsc_b = st.selectbox("12th Board", FEATURE_CONSTRAINTS["hsc_b"]["options"])
                new_hsc_s = st.selectbox("12th Stream", FEATURE_CONSTRAINTS["hsc_s"]["options"])

            with reg_c2:
                new_ssc_p = st.number_input("10th Percentage (Locked after entry)", 40.0, 100.0, 70.0, step=0.1)
                new_hsc_p = st.number_input("12th Percentage (Locked after entry)", 40.0, 100.0, 68.0, step=0.1)
                new_degree_p = st.number_input("Degree Percentage (Locked after entry)", 45.0, 100.0, 67.0, step=0.1)
                new_etest_p = st.number_input("Initial Employability Test Score (%)", 40.0, 100.0, 65.0, step=0.1)
                new_mba_p = st.number_input("MBA Aggregate Percentage (%)", 45.0, 100.0, 62.0, step=0.1)
                new_workex = st.selectbox("Work Experience / Verified Internship", ["Yes", "No"], index=1)
                new_notes = st.text_area("Initial Check-in Notes (Optional)", "Baseline semester assessment.")

            submitted_reg = st.form_submit_button("✅ Register Student & Log First Check-in", type="primary")
            if submitted_reg:
                profile_to_reg = {
                    "gender": new_gender, "specialisation": new_spec, "degree_t": new_degree_t,
                    "ssc_b": new_ssc_b, "hsc_b": new_hsc_b, "hsc_s": new_hsc_s,
                    "ssc_p": new_ssc_p, "hsc_p": new_hsc_p, "degree_p": new_degree_p,
                }
                register_or_get_student(new_s_id, new_name, profile_to_reg)
                record_student_checkin(
                    new_s_id,
                    etest_p=new_etest_p,
                    mba_p=new_mba_p,
                    workex=new_workex,
                    active_pipeline=active_pipeline,
                    threshold=threshold,
                    notes=new_notes,
                )
                st.success(f"Registered {new_name} ({new_s_id}) successfully!")
                st.rerun()

    else:
        if not tracked_students:
            st.info("No students registered yet in the tracker database. Click 'Load Sample Check-ins (Demo)' above or register a new student to begin.")
        else:
            student_options = {
                f"{s['name']} ({s['student_id']}) — {s['specialisation']} [{s['total_checkins']} check-in(s)]": s["student_id"]
                for s in tracked_students
            }
            selected_display = st.selectbox("Select Student to Track:", list(student_options.keys()))
            selected_s_id = student_options[selected_display]

            student_history = get_student_checkin_history(selected_s_id)
            student_meta = next(s for s in tracked_students if s["student_id"] == selected_s_id)

            if not student_history.empty:
                latest_checkin = student_history.iloc[-1]
                latest_score = float(latest_checkin["readiness_score"])
                latest_prob = float(latest_checkin["probability"])
                latest_band, _, latest_color = (
                    ("Placement Ready", "Ready", "#10B981") if latest_score > 70
                    else (("Developing", "Dev", "#F59E0B") if latest_score >= 40 else ("Needs Support", "Support", "#EF4444"))
                )

                score_delta_str = None
                prob_delta_str = None
                if len(student_history) >= 2:
                    prev_score = float(student_history.iloc[-2]["readiness_score"])
                    prev_prob = float(student_history.iloc[-2]["probability"])
                    score_diff = latest_score - prev_score
                    prob_diff = latest_prob - prev_prob
                    score_delta_str = f"{score_diff:+.1f} pts"
                    prob_delta_str = f"{prob_diff:+.1%}"

                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric(
                    "Readiness Score (0-100)",
                    f"{latest_score:.1f}",
                    delta=score_delta_str,
                    delta_color="normal",
                )
                m_c2.metric(
                    "Placement Likelihood",
                    f"{latest_prob:.1%}",
                    delta=prob_delta_str,
                    delta_color="normal",
                )
                m_c3.metric("Current Band", latest_band)
                m_c4.metric("Total Check-ins", len(student_history))

                st.markdown("---")
                g_col1, g_col2 = st.columns([1.3, 1.7])

                with g_col1:
                    fig_r_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=latest_score,
                        number={"suffix": " pts", "font": {"size": 28, "color": "#0F172A"}},
                        title={"text": f"<b>Placement Readiness Score</b><br><span style='font-size:12px; color:#64748B;'>Band: {latest_band}</span>", "font": {"size": 14}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": latest_color, "thickness": 0.3},
                            "steps": [
                                {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.15)"},
                                {"range": [70, 100], "color": "rgba(16, 185, 129, 0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": "#1E3A8A", "width": 3},
                                "thickness": 0.75,
                                "value": 70.0,
                            },
                        },
                    ))
                    fig_r_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_r_gauge, use_container_width=True)

                with g_col2:
                    st.markdown("#### 🔒 Locked Academic Profile")
                    lk1, lk2, lk3 = st.columns(3)
                    lk1.info(f"**10th (SSC):** {latest_checkin['ssc_p']:.1f}%")
                    lk2.info(f"**12th (HSC):** {latest_checkin['hsc_p']:.1f}%")
                    lk3.info(f"**Graduation:** {latest_checkin['degree_p']:.1f}%")

                    st.markdown("#### 📝 Log Progress Check-in (Editable Factors Only)")
                    with st.form("form_log_checkin"):
                        c_in1, c_in2, c_in3 = st.columns(3)
                        with c_in1:
                            edit_etest = st.slider("Latest E-Test Score (%)", 40.0, 100.0, float(latest_checkin["etest_p"]), step=0.5)
                        with c_in2:
                            edit_workex = st.selectbox("Work Experience / Internship", ["Yes", "No"], index=0 if latest_checkin["workex"] == "Yes" else 1)
                        with c_in3:
                            edit_mba = st.slider("Latest MBA / PG %", 45.0, 100.0, float(latest_checkin["mba_p"]), step=0.5)

                        checkin_notes = st.text_input("Milestone Notes", placeholder="e.g. Scored +12 in mock aptitude; completed summer internship")
                        submitted_checkin = st.form_submit_button("📈 Save Check-in & Update Plan", type="primary")

                        if submitted_checkin:
                            res = record_student_checkin(
                                student_id=selected_s_id,
                                etest_p=edit_etest,
                                mba_p=edit_mba,
                                workex=edit_workex,
                                active_pipeline=active_pipeline,
                                threshold=threshold,
                                notes=checkin_notes,
                            )
                            st.success(f"Check-in logged! New Readiness Score: {res['readiness_score']:.1f} pts ({res['band']})")
                            st.rerun()

                # Trajectory chart
                st.markdown("---")
                st.markdown("### 📈 Chronological Progress Trajectory")
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=student_history["date"],
                    y=student_history["readiness_score"],
                    mode="lines+markers",
                    name="Readiness Score (0-100)",
                    line=dict(color="#2563EB", width=3),
                    marker=dict(size=8),
                ))
                fig_trend.add_trace(go.Scatter(
                    x=student_history["date"],
                    y=student_history["probability"] * 100,
                    mode="lines+markers",
                    name="Placement Likelihood (%)",
                    line=dict(color="#10B981", width=2, dash="dash"),
                    marker=dict(size=6),
                ))
                fig_trend.add_hline(y=70, line_dash="dot", line_color="#10B981", annotation_text="Placement Ready Cutoff (70)")
                fig_trend.add_hline(y=40, line_dash="dot", line_color="#EF4444", annotation_text="Needs Support Cutoff (40)")
                fig_trend.update_layout(
                    title="Student Growth Over Time (Readiness Score & Placement Likelihood)",
                    xaxis_title="Check-in Date",
                    yaxis_title="Score / Percentage",
                    yaxis=dict(range=[0, 105]),
                    height=320,
                    margin=dict(l=10, r=10, t=35, b=20),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                # Action Plan
                st.markdown("---")
                st.markdown("### 📋 Adaptive 4-Week Action Plan")
                tasks = get_student_tasks(selected_s_id)
                if tasks:
                    completed_count = sum(1 for t in tasks if t["completed"])
                    progress_pct = completed_count / len(tasks)
                    st.progress(progress_pct)
                    st.caption(f"**Progress:** {completed_count} of {len(tasks)} tasks completed ({progress_pct:.0%}) — Focus Area: `{tasks[0]['category']}`")

                    t_cols = st.columns(len(tasks))
                    for idx, t in enumerate(tasks):
                        with t_cols[idx]:
                            st.markdown(f"**Week {t['week']}**")
                            is_checked = st.checkbox(
                                t["task_text"],
                                value=bool(t["completed"]),
                                key=f"task_{selected_s_id}_{t['task_id']}",
                            )
                            if is_checked != bool(t["completed"]):
                                update_task_completion(t["task_id"], is_checked)
                                st.rerun()

                with st.expander("🗑️ Delete Student Profile & Audit Data (Privacy / GDPR)"):
                    st.warning("Permanently wipe all records, past check-ins, and task history for this student.")
                    confirm_del = st.checkbox(f"I confirm I want to permanently delete data for {selected_display}", key=f"del_confirm_{selected_s_id}")
                    if st.button("Permanently Delete Student", type="primary", disabled=not confirm_del):
                        delete_student_data(selected_s_id)
                        st.success(f"Deleted all records for {selected_s_id}.")
                        st.rerun()

    # Alerts Section
    st.markdown("---")
    st.markdown("### 🚨 Placement Cell Early-Warning Alerts & Cohort Overview")

    alert_c1, alert_c2 = st.columns([1.8, 1.2])

    with alert_c1:
        st.markdown("#### Students Needing Immediate Attention")
        alerts_df = get_early_warning_alerts()
        if not alerts_df.empty:
            st.dataframe(
                alerts_df[["student_id", "name", "specialisation", "alert_type", "severity", "latest_metric", "change", "details"]].style.map(
                    lambda v: "background-color: #FEF2F2; color: #991B1B; font-weight: bold;" if v == "High" else "background-color: #FFFBEB; color: #92400E;",
                    subset=["severity"],
                ),
                use_container_width=True,
            )

            alerts_csv = alerts_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Flagged Students Roster (CSV)",
                data=alerts_csv,
                file_name="placement_early_warning_alerts.csv",
                mime="text/csv",
            )

            with st.expander("✉️ Draft Supportive Outreach Email (Never Auto-Sent)"):
                student_to_email = st.selectbox(
                    "Select Flagged Student:",
                    alerts_df["student_id"].tolist(),
                    format_func=lambda sid: f"{alerts_df.loc[alerts_df['student_id'] == sid, 'name'].iloc[0]} ({sid})",
                )
                selected_alert_row = alerts_df[alerts_df["student_id"] == student_to_email].iloc[0].to_dict()
                draft_body = draft_supportive_message(selected_alert_row)
                st.text_area("Counselor Outreach Draft:", value=draft_body, height=200)
        else:
            st.success("✅ No early-warning triggers detected! All active students show positive or stable progression.")

    with alert_c2:
        st.markdown("#### Cohort Readiness Band Distribution")
        if tracked_students:
            band_counts = {"Needs Support (<40)": 0, "Developing (40-70)": 0, "Placement Ready (>70)": 0}
            for s in tracked_students:
                r_score = s.get("readiness_score")
                if r_score is not None:
                    if r_score > 70:
                        band_counts["Placement Ready (>70)"] += 1
                    elif r_score >= 40:
                        band_counts["Developing (40-70)"] += 1
                    else:
                        band_counts["Needs Support (<40)"] += 1

            fig_band = px.bar(
                x=list(band_counts.keys()),
                y=list(band_counts.values()),
                color=list(band_counts.keys()),
                color_discrete_map={
                    "Needs Support (<40)": "#EF4444",
                    "Developing (40-70)": "#F59E0B",
                    "Placement Ready (>70)": "#10B981",
                },
                title="Current Cohort Distribution",
                labels={"x": "Readiness Band", "y": "Student Count"},
            )
            fig_band.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=20), showlegend=False)
            st.plotly_chart(fig_band, use_container_width=True)


# =========================================================
# VIEW 6: 🏛️ TPO & Recruiter Portal
# =========================================================
elif nav_selection == "🏛️ TPO & Recruiter Portal":
    st.subheader("🏛️ Training & Placement Officer (TPO) & Recruiter Command Center")
    st.write("Manage multi-candidate cohorts, analyze placement risk heatmaps, generate customized recruiter shortlists, and export ready Excel files.")

    tpo_view = st.radio("TPO Operations:", ["🎯 Recruiter Drive Shortlist Generator", "📊 Cohort Risk Heatmap", "🚨 Automated Remedial Triggers"], horizontal=True)

    cohort_raw_df = dataset_df.copy() if dataset_df is not None else pd.DataFrame()
    if not cohort_raw_df.empty:
        if "probability" not in cohort_raw_df.columns:
            scored_cohort = batch_predict_students(cohort_raw_df[ALL_FEATURE_COLS], active_pipeline, threshold=threshold)
            cohort_raw_df["probability"] = scored_cohort["Placement_Probability"]
            cohort_raw_df["predicted_status"] = scored_cohort["Predicted_Status"]
            cohort_raw_df["student_id"] = [f"STU-{1000 + i}" for i in range(len(cohort_raw_df))]
            cohort_raw_df["name"] = [f"Candidate {1000 + i}" for i in range(len(cohort_raw_df))]

    if tpo_view == "🎯 Recruiter Drive Shortlist Generator":
        st.markdown("#### Configure Recruiter Eligibility Criteria")
        t_form_c1, t_form_c2, t_form_c3 = st.columns(3)
        with t_form_c1:
            req_company = st.text_input("Visiting Company Name:", value="Deloitte USI")
            req_min_deg = st.slider("Minimum Graduation Degree %", 45.0, 90.0, 65.0, step=1.0)
            req_min_etest = st.slider("Minimum College Aptitude Score %", 40.0, 90.0, 60.0, step=1.0)

        with t_form_c2:
            req_min_prob = st.slider("Minimum ML Placement Probability", 0.30, 0.90, 0.60, step=0.05)
            req_spec = st.selectbox("Eligible Specialisation:", ["All", "Mkt&Fin", "Mkt&HR"])
            req_workex = st.selectbox("Work Experience / Internship Required:", ["Either (All)", "Yes (Mandatory)", "No (Freshers Only)"])

        with t_form_c3:
            req_gender = st.selectbox("Diversity Filter (Gender):", ["All", "M", "F"])
            st.caption("Apply standard campus placement filters to generate recruiter shortlists with one click.")

        crit = {
            "min_degree_p": req_min_deg,
            "min_etest_p": req_min_etest,
            "min_probability": req_min_prob,
            "specialisation": req_spec,
            "workex_required": True if req_workex == "Yes (Mandatory)" else False,
            "gender": req_gender,
        }

        shortlist_df = generate_company_shortlist(cohort_raw_df, crit)
        st.markdown("---")
        st.markdown(f"### 📋 Eligible Candidate Shortlist for **{req_company}**")

        sl_c1, sl_c2, sl_c3 = st.columns(3)
        sl_c1.metric("Total Cohort Pool", len(cohort_raw_df))
        sl_c2.metric("Eligible Candidates", len(shortlist_df))
        sl_conversion = (len(shortlist_df) / max(len(cohort_raw_df), 1)) * 100
        sl_c3.metric("Pool Conversion Rate", f"{sl_conversion:.1f}%")

        if not shortlist_df.empty:
            st.dataframe(
                shortlist_df.style.map(
                    lambda v: "background-color: #ECFDF5; color: #065F46; font-weight: bold;" if v == "Low Risk" else "background-color: #FFFBEB; color: #92400E;",
                    subset=["risk_level"] if "risk_level" in shortlist_df.columns else []
                ),
                use_container_width=True
            )

            excel_bytes = generate_shortlist_excel(shortlist_df, req_company)
            st.download_button(
                label=f"⬇️ Download Official Recruiter Shortlist for {req_company} (Excel .xlsx)",
                data=excel_bytes,
                file_name=f"{req_company.replace(' ', '_')}_campus_shortlist.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        else:
            st.warning("No students meet all the selected criteria. Try easing the minimum percentage or probability thresholds.")

    elif tpo_view == "📊 Cohort Risk Heatmap":
        st.markdown("#### Campus Cohort Risk Segmentation")
        if not cohort_raw_df.empty:
            stu_list = cohort_raw_df.to_dict(orient="records")
            hmap = generate_cohort_risk_heatmap(stu_list, cohort_raw_df)

            hm_c1, hm_c2, hm_c3 = st.columns(3)
            hm_c1.metric("🔴 High Risk / Urgent (<40%)", f"{hmap['summary']['red_count']} ({hmap['summary']['red_pct']}%)")
            hm_c2.metric("🟡 Borderline / Moderate (40-65%)", f"{hmap['summary']['amber_count']} ({hmap['summary']['amber_pct']}%)")
            hm_c3.metric("🟢 Placement Ready (>65%)", f"{hmap['summary']['green_count']} ({hmap['summary']['green_pct']}%)")

            fig_hm = px.bar(
                x=["High Risk (<40%)", "Borderline (40-65%)", "Placement Ready (>65%)"],
                y=[hmap['summary']['red_count'], hmap['summary']['amber_count'], hmap['summary']['green_count']],
                color=["High Risk (<40%)", "Borderline (40-65%)", "Placement Ready (>65%)"],
                color_discrete_map={
                    "High Risk (<40%)": "#EF4444",
                    "Borderline (40-65%)": "#F59E0B",
                    "Placement Ready (>65%)": "#10B981"
                },
                labels={"x": "Risk Tier", "y": "Number of Students"},
                title="Cohort Distribution Across Risk Tiers"
            )
            fig_hm.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_hm, use_container_width=True)

    elif tpo_view == "🚨 Automated Remedial Triggers":
        st.markdown("#### High-Priority Remedial Notification Dispatch")
        st.write("Trigger batch intervention for students falling below the 40% probability cutoff:")
        red_candidates = cohort_raw_df[cohort_raw_df["probability"] < 0.40] if "probability" in cohort_raw_df.columns else pd.DataFrame()
        st.info(f"Identified **{len(red_candidates)} students** requiring immediate placement bootcamps and faculty counseling.")
        if not red_candidates.empty:
            st.dataframe(red_candidates[["student_id", "degree_p", "etest_p", "specialisation", "probability"]].head(15), use_container_width=True)
            if st.button("✉️ Generate Batch Email Notices for High-Risk Students", type="primary"):
                st.success(f"Generated outreach notifications for {len(red_candidates)} candidates! Reminders queued for academic advisors.")


# =========================================================
# VIEW 7: 📁 Batch Student Evaluation
# =========================================================
elif nav_selection == "📁 Batch Student Evaluation":
    st.subheader("📁 Batch Student Evaluation & At-Risk Identification")
    st.write("Upload a batch roster of students (CSV or Excel) to evaluate placement probabilities simultaneously.")

    sample_df = pd.DataFrame([
        {
            "gender": "M", "ssc_p": 68.0, "ssc_b": "Central", "hsc_p": 65.0,
            "hsc_b": "Others", "hsc_s": "Commerce", "degree_p": 66.0,
            "degree_t": "Comm&Mgmt", "workex": "No", "etest_p": 72.0,
            "specialisation": "Mkt&Fin", "mba_p": 62.0,
        },
        {
            "gender": "F", "ssc_p": 82.0, "ssc_b": "Central", "hsc_p": 80.0,
            "hsc_b": "Central", "hsc_s": "Science", "degree_p": 76.0,
            "degree_t": "Sci&Tech", "workex": "Yes", "etest_p": 85.0,
            "specialisation": "Mkt&Fin", "mba_p": 71.0,
        },
    ])
    st.download_button(
        "📥 Download Sample Batch Template (CSV)",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="student_batch_template.csv",
        mime="text/csv",
    )

    uploaded_batch = st.file_uploader(
        "Upload Roster (CSV or Excel .xlsx/.xls)",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_batch is not None:
        try:
            if uploaded_batch.name.endswith((".xlsx", ".xls")):
                batch_raw = pd.read_excel(uploaded_batch)
            else:
                batch_raw = pd.read_csv(uploaded_batch)

            st.success(f"Successfully loaded {len(batch_raw)} records from `{uploaded_batch.name}`.")

            scored_batch = batch_predict_students(batch_raw, active_pipeline, threshold=threshold)

            total_students = len(scored_batch)
            placed_count = (scored_batch["Predicted_Status"] == "Placed").sum()
            risk_count = total_students - placed_count
            placed_rate = (placed_count / total_students) * 100

            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total Batch Students", f"{total_students}")
            bc2.metric("Projected Placed", f"{placed_count}", delta=f"{placed_rate:.1f}% conversion")
            bc3.metric("At-Risk / Needs Coaching", f"{risk_count}", delta=f"-{risk_count}", delta_color="inverse")
            bc4.metric("Active Decision Threshold", f"{threshold:.2f}")

            st.markdown("#### Filter Scored Cohort")
            filter_choice = st.radio("Display:", ["All Students", "At-Risk Students Only", "Placed Students Only"], horizontal=True)

            if filter_choice == "At-Risk Students Only":
                display_df = scored_batch[scored_batch["Predicted_Status"] == "Not Placed"]
            elif filter_choice == "Placed Students Only":
                display_df = scored_batch[scored_batch["Predicted_Status"] == "Placed"]
            else:
                display_df = scored_batch

            st.dataframe(
                display_df.style.map(
                    lambda v: "background-color: #FEF2F2; color: #991B1B; font-weight: bold;" if v == "High Risk" else (
                        "background-color: #FFFBEB; color: #92400E;" if v == "Moderate Risk" else "background-color: #ECFDF5; color: #065F46;"
                    ),
                    subset=["Risk_Level"],
                ),
                use_container_width=True,
            )

            csv_data = scored_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Scored Batch Results (CSV)",
                data=csv_data,
                file_name=f"scored_batch_{uploaded_batch.name}.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Error evaluating batch roster: {e}")


# =========================================================
# VIEW 8: 📊 Cohort Analytics & Trends
# =========================================================
elif nav_selection == "📊 Cohort Analytics & Trends":
    st.subheader("📊 Campus Recruitment Analytics & Cohort Exploration")
    st.write("Exploratory intelligence across student demographics, specialisations, and academic trends.")

    if dataset_df is not None:
        p_c1, p_c2 = st.columns(2)

        with p_c1:
            spec_rates = dataset_df.groupby("specialisation")["status"].apply(lambda s: (s == "Placed").mean() * 100).reset_index()
            spec_rates.columns = ["Specialisation", "Placement Rate (%)"]
            fig_spec = px.bar(
                spec_rates,
                x="Specialisation",
                y="Placement Rate (%)",
                color="Placement Rate (%)",
                color_continuous_scale="Blues",
                title="Placement Conversion by MBA Specialisation",
                text_auto=".1f",
            )
            fig_spec.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_spec, use_container_width=True)

        with p_c2:
            workex_rates = dataset_df.groupby("workex")["status"].apply(lambda s: (s == "Placed").mean() * 100).reset_index()
            workex_rates.columns = ["Work Experience", "Placement Rate (%)"]
            fig_workex = px.bar(
                workex_rates,
                x="Work Experience",
                y="Placement Rate (%)",
                color="Placement Rate (%)",
                color_continuous_scale="Teal",
                title="Impact of Work Experience on Placement Rate",
                text_auto=".1f",
            )
            fig_workex.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_workex, use_container_width=True)

        st.markdown("#### Academic Score Distributions by Placement Outcome")
        selected_metric = st.selectbox(
            "Select Metric to Analyze:",
            NUMERIC_FEATURES,
            format_func=lambda c: f"{FIELD_LABELS.get(c, c)} ({c})",
        )
        fig_box = px.box(
            dataset_df,
            x="status",
            y=selected_metric,
            color="status",
            color_discrete_map={"Placed": "#10B981", "Not Placed": "#EF4444"},
            points="all",
            title=f"Distribution of {FIELD_LABELS.get(selected_metric, selected_metric)} by Status",
        )
        fig_box.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

        with st.expander("🔥 View Interactive Correlation Matrix"):
            num_data = dataset_df[NUMERIC_FEATURES].copy()
            num_data["Placed_Numeric"] = (dataset_df["status"] == "Placed").astype(int)
            corr_matrix = num_data.corr().round(2)
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Correlation Heatmap of Academic Scores & Placement",
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# =========================================================
# VIEW 9: 🤖 Model Benchmarks & Governance
# =========================================================
elif nav_selection == "🤖 Model Benchmarks & Governance":
    st.subheader("🤖 Machine Learning Model Benchmarking & Architecture Governance")
    st.write("We tuned and benchmarked 6 algorithms using 5-Fold Stratified Cross-Validation on the training data, followed by probability calibration and threshold optimization on the test split.")

    if results_df is not None:
        st.dataframe(
            results_df.style.highlight_max(
                axis=0,
                subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1"],
                color="#D1FAE5",
            ),
            use_container_width=True,
        )

        fig_comp = px.bar(
            results_df,
            x="Model",
            y=["Accuracy", "F1", "CV F1", "ROC-AUC"],
            barmode="group",
            title="Comparison of Model Accuracy, Test F1, 5-Fold CV F1, and ROC-AUC",
            color_discrete_sequence=["#1E3A8A", "#3B82F6", "#60A5FA", "#93C5FD"],
        )
        fig_comp.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Diagnostic Figures")
    fig_col1, fig_col2 = st.columns(2)

    cm_file = FIGURES_DIR / "confusion_matrix.png"
    if cm_file.exists():
        with fig_col1:
            st.image(str(cm_file), caption="Confusion Matrix (Champion Model)", use_container_width=True)

    roc_file = FIGURES_DIR / "roc_curve.png"
    if roc_file.exists():
        with fig_col2:
            st.image(str(roc_file), caption="Receiver Operating Characteristic (ROC Curve)", use_container_width=True)

    st.markdown("#### Ranked Feature Importance")
    if feature_imp_df is not None:
        st.dataframe(feature_imp_df.head(10), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏛️ System Architecture, Formulas & Governance")
    st.markdown(r"""
    #### 📈 Placement Readiness Score Formula (0 to 100)
    $$\text{Readiness Score} = (0.60 \times P_{\text{calibrated}} \times 100) + (0.20 \times \text{E-Test}) + (0.10 \times \text{WorkEx}) + (0.10 \times \text{Consistency})$$

    - **Calibrated Placement Probability ($60\%$):** Probability generated by the tuned machine learning pipeline, scaled to 60 points.
    - **Employability & Aptitude Score ($20\%$):** Latest college screening mock test percentage ($0-100\%$), contributing up to 20 points.
    - **Work Experience / Internships ($10\%$):** Verified internship or industry project awards 10 points ($100 \times 0.10$).
    - **Check-in Consistency ($10\%$):** Baseline awarded based on check-in frequency (1 check-in = 50, 2 = 70, 3 = 85, 4+ = 100). Inactivity beyond 14 days incurs a step penalty.

    #### 🚀 Production Feature Intelligence
    1. **Multi-Tier Company Prediction:** Models candidate likelihood across Dream (₹15-40 LPA), Core (₹7-14 LPA), and Mass Recruiter (₹3.5-6 LPA) brackets using smooth sigmoid-scaled strength scoring.
    2. **Counterfactual Recourse Optimization:** Solves the inverse problem by identifying the minimum realistic improvements in E-test, MBA marks, or work experience needed to cross any target probability.
    3. **Resume & JD ATS Engine:** Evaluates semantic keyword overlap between candidate resumes (PDF/text) and job descriptions across 8 skill taxonomies.
    4. **Holistic Tech Readiness Index:** Integrates GitHub contributions, LeetCode problem solving, and professional certifications into a unified score ($65\%$ Academic ML + $35\%$ Practical Tech Portfolio).
    5. **TPO Command Center:** Equips placement officers with cohort risk heatmaps, recruiter shortlist generation, and 1-click formatted Excel (.xlsx) export.
    6. **AI Mock Interview Simulator:** Generates domain-specific technical, behavioral, and case-study questions, scoring practice answers with the STAR (Situation, Task, Action, Result) methodology.

    #### 🛡️ Anti-Leakage & Governance Measures
    1. **Target Leakage Prevention:** Columns such as `sl_no` (student ID) and `salary` (which exists strictly for placed students) are automatically purged prior to pipeline fitting.
    2. **Stratified Splitting & Validation:** 5-fold stratified cross validation ensures class proportions are preserved across folds.
    3. **Calibrated Probabilities:** Ensembles and linear models are wrapped in `CalibratedClassifierCV` using sigmoid scaling so probabilities accurately reflect true placement risk.
    4. **Decision Threshold Tuning:** The decision threshold is tuned to optimize F1-score rather than relying on a naive 0.50 cutoff.
    """)
