"""Campus Placement Prediction & Student Success Analytics Dashboard.

Production Streamlit application featuring multi-model inference, probability calibration,
dynamic explainability, What-If simulator, Radar benchmarking, PDF reports, Plotly analytics,
batch CSV/Excel processing, and SQLite audit logging.
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
    compute_student_feature_contributions,
    generate_coaching_recommendations,
    get_radar_comparison_data,
    simulate_what_if,
)
from src.predict import batch_predict_students, predict_student_placement
from src.report_generator import generate_student_pdf_report

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Campus Placement Prediction & Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Glassmorphic, Modern Gradient CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        padding: 2.2rem 2.2rem 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #CBD5E1;
        margin-top: 0.6rem;
        margin-bottom: 0;
        max-width: 900px;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.8rem;
    }

    .badge-primary {
        background: rgba(59, 130, 246, 0.25);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.25);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .rec-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: #1E293B;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
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

champion_model, all_models, meta = load_all_models_and_meta()
dataset_df = load_dataset_cached()
results_df = load_benchmark_results()
feature_imp_df = load_feature_importances()

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🎓 Campus Placement Predictor & Student Success Analytics</h1>
    <p>Enterprise Machine Learning platform for predicting placement likelihood, identifying at-risk students before recruitment drives, running What-If sensitivity simulations, and generating automated coaching plans.</p>
    <div>
        <span class="badge badge-primary">✨ Calibrated ML Probabilities</span>
        <span class="badge badge-success">🎯 6-Algorithm Benchmarking</span>
        <span class="badge badge-primary">📊 Early Risk Intervention System</span>
    </div>
</div>
""", unsafe_allow_html=True)

if meta is None or (champion_model is None and not all_models):
    st.error("⚠️ Trained models or metadata not found. Please run `python src/train.py` to train and calibrate the models.")
    st.stop()


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Placement Cell Controls")

    user_role = st.selectbox(
        "User Role / View",
        ["🧑‍🎓 Student / Counselor View", "🏛️ Placement Cell Admin"],
        help="Switch between Counselor guidance view and Admin batch analytics.",
    )

    st.markdown("---")
    st.markdown("### 🏆 Active ML Model")

    available_model_names = list(all_models.keys()) if all_models else [meta.get("best_model", "KNN")]
    default_idx = available_model_names.index(meta.get("best_model", "KNN")) if meta.get("best_model") in available_model_names else 0

    selected_model_name = st.selectbox(
        "Select Model Architecture",
        options=available_model_names,
        index=default_idx,
        help="Select any trained classifier to evaluate student profiles in real-time.",
    )

    active_pipeline = all_models.get(selected_model_name, champion_model)

    # Dynamic metrics display for selected model
    model_row = None
    if results_df is not None and "Model" in results_df.columns:
        matching = results_df[results_df["Model"] == selected_model_name]
        if not matching.empty:
            model_row = matching.iloc[0].to_dict()

    if not model_row:
        model_row = meta.get("metrics", {})

    st.markdown(f"**Performance ({selected_model_name}):**")
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Accuracy", f"{float(model_row.get('Accuracy', 0)):.1%}")
    c_m1.metric("Recall", f"{float(model_row.get('Recall', 0)):.1%}")
    c_m2.metric("Precision", f"{float(model_row.get('Precision', 0)):.1%}")
    c_m2.metric("CV F1-Score", f"{float(model_row.get('CV F1', 0)):.3f}")

    opt_thresh_val = float(meta.get("optimal_threshold", 0.50))
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.30,
        max_value=0.70,
        value=opt_thresh_val,
        step=0.02,
        help="Probability cutoff for classifying as Placed. Calibrated optimal value is pre-set.",
    )

    st.markdown("---")
    st.markdown("### 🎯 Placement Cell Objectives")
    st.markdown("""
    - **Early Identification:** Flag at-risk students before campus drives begin.
    - **Targeted Coaching:** Focus remedial sessions on aptitude (`etest_p`) & projects.
    - **Internship Encouragement:** Prioritize students lacking practical experience.
    """)
    st.caption("Campus Recruitment Decision Support System v2.0")


# ---------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Student Predictor",
    "📁 Batch Student Evaluation",
    "📊 Placement Analytics",
    "🤖 Model Benchmarks",
    "📜 Prediction History",
    "ℹ️ About System",
])


# =========================================================
# TAB 1: Student Predictor
# =========================================================
with tab1:
    st.subheader("Predict Placement Likelihood for a Student")
    st.write("Enter the student's academic and profile parameters, or select one of the pre-configured scenarios below:")

    # Preset Scenarios
    preset_cols = st.columns(len(PRESETS) + 1)
    for i, (p_name, p_data) in enumerate(PRESETS.items()):
        if preset_cols[i].button(p_name, use_container_width=True):
            st.session_state["student_profile"] = p_data

    if preset_cols[-1].button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state["student_profile"] = {
            "gender": "M", "ssc_p": 70.0, "ssc_b": "Central", "hsc_p": 70.0,
            "hsc_b": "Others", "hsc_s": "Commerce", "degree_p": 68.0,
            "degree_t": "Comm&Mgmt", "workex": "Yes", "etest_p": 75.0,
            "specialisation": "Mkt&Fin", "mba_p": 65.0,
        }

    current_profile = st.session_state.get("student_profile", PRESETS["🌟 High-Scoring All-Rounder"])

    st.markdown("---")

    # 2-Column Responsive Input Form
    col_left, col_right = st.columns(2)
    form_inputs = {}

    with col_left:
        st.markdown("#### 📚 Secondary & Higher Secondary Education")
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
        form_inputs["gender"] = st.selectbox(
            FIELD_LABELS["gender"],
            options=FEATURE_CONSTRAINTS["gender"]["options"],
            index=FEATURE_CONSTRAINTS["gender"]["options"].index(current_profile.get("gender", "M")),
            help="Student legal gender",
        )

    with col_right:
        st.markdown("#### 🎓 Higher Education & Employability")
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
        form_inputs["workex"] = st.selectbox(
            FIELD_LABELS["workex"],
            options=FEATURE_CONSTRAINTS["workex"]["options"],
            index=FEATURE_CONSTRAINTS["workex"]["options"].index(current_profile.get("workex", "Yes")),
            help=FIELD_HELP["workex"],
        )
        form_inputs["etest_p"] = st.slider(
            FIELD_LABELS["etest_p"],
            min_value=40.0, max_value=100.0,
            value=float(current_profile.get("etest_p", 75.0)),
            step=0.1, help=FIELD_HELP["etest_p"],
        )
        form_inputs["specialisation"] = st.selectbox(
            FIELD_LABELS["specialisation"],
            options=FEATURE_CONSTRAINTS["specialisation"]["options"],
            index=FEATURE_CONSTRAINTS["specialisation"]["options"].index(current_profile.get("specialisation", "Mkt&Fin")),
            help=FIELD_HELP["specialisation"],
        )
        form_inputs["mba_p"] = st.slider(
            FIELD_LABELS["mba_p"],
            min_value=45.0, max_value=100.0,
            value=float(current_profile.get("mba_p", 65.0)),
            step=0.1, help=FIELD_HELP["mba_p"],
        )

    student_id = st.text_input("Student Identifier / Name (Optional for Report & History)", value="Student-101")

    # Run Prediction
    prediction_result = predict_student_placement(form_inputs, active_pipeline, threshold=threshold)
    prob = prediction_result["probability"]
    risk_level = prediction_result["risk_level"]
    label = prediction_result["label"]

    # Log to SQLite
    log_prediction(form_inputs, prediction_result, selected_model_name, identifier=student_id)

    st.markdown("---")
    st.markdown("### 📊 Prediction Results & Risk Assessment")

    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        # Gauge Chart using Plotly
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 32, "color": "#0F172A"}},
            delta={"reference": threshold * 100, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
            title={"text": f"<b>{prediction_result['status_text']}</b><br><span style='font-size:13px; color:#64748B;'>Active Model: {selected_model_name} (Threshold: {threshold:.2f})</span>", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": prediction_result["color"], "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [40, 65], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [65, 100], "color": "rgba(16, 185, 129, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#1E3A8A", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk Classification Badge
        if risk_level == "Low Risk":
            st.success(f"**Classification: Placed (Low Risk)** — Probability ({prob:.1%}) exceeds threshold ({threshold:.2f}).")
        elif risk_level == "Moderate Risk":
            st.warning(f"**Classification: Borderline / Moderate Risk** — Probability ({prob:.1%}). Targeted intervention advised.")
        else:
            st.error(f"**Classification: At-Risk of Not Placed** — Probability ({prob:.1%}) is below threshold ({threshold:.2f}).")

        # PDF Download Button
        pdf_bytes = generate_student_pdf_report(form_inputs, prediction_result, model_name=selected_model_name)
        st.download_button(
            label="📄 Download Student Career Readiness Report (PDF)",
            data=pdf_bytes,
            file_name=f"{student_id}_placement_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    with res_col2:
        st.markdown("#### 💡 Targeted Placement Cell Guidance")
        recs = prediction_result["recommendations"]
        for r in recs:
            st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Visual Analytics: Explainability & Radar Benchmark
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 Model Explainability & Cohort Comparison")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### Feature Influence vs. Cohort Average")
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
            fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("📝 View Detailed Attribute Insights"):
                for _, row in df_contrib.iterrows():
                    st.write(f"• **{FIELD_LABELS.get(row['Feature'], row['Feature'])}:** {row['Insight']}")

    with exp_col2:
        st.markdown("#### Student vs. Average Placed Student (Radar Chart)")
        benchmarks = meta.get("numeric_benchmark_placed", {
            "ssc_p": 71.0, "hsc_p": 69.0, "degree_p": 68.0, "etest_p": 73.0, "mba_p": 63.0
        })
        radar_df = get_radar_comparison_data(form_inputs, benchmarks)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Student"],
            theta=radar_df["Metric"],
            fill="toself",
            name="This Student",
            line_color="#3B82F6",
            fillcolor="rgba(59, 130, 246, 0.25)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Placed Benchmark"],
            theta=radar_df["Metric"],
            fill="toself",
            name="Placed Cohort Avg",
            line_color="#10B981",
            fillcolor="rgba(16, 185, 129, 0.25)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[40, 100])),
            height=300,
            margin=dict(l=30, r=30, t=40, b=20),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ---------------------------------------------------------
    # Live What-If Sensitivity Simulator
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚡ Live 'What-If' Scenario Simulator")
    st.write("Adjust hypothetical improvements in employability test score or internship experience to see the instant impact on placement odds:")

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
# TAB 2: Batch Student Evaluation (CSV & Excel)
# =========================================================
with tab2:
    st.subheader("📁 Batch Student Evaluation & At-Risk Identification")
    st.write("Upload a batch roster of students (CSV or Excel) to evaluate placement probabilities simultaneously.")

    # Download template
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

            # Run Batch Prediction
            scored_batch = batch_predict_students(batch_raw, active_pipeline, threshold=threshold)

            # Summary Metrics
            total_students = len(scored_batch)
            placed_count = (scored_batch["Predicted_Status"] == "Placed").sum()
            risk_count = total_students - placed_count
            placed_rate = (placed_count / total_students) * 100

            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total Batch Students", f"{total_students}")
            bc2.metric("Projected Placed", f"{placed_count}", delta=f"{placed_rate:.1f}% conversion")
            bc3.metric("At-Risk / Needs Coaching", f"{risk_count}", delta=f"-{risk_count}", delta_color="inverse")
            bc4.metric("Active Decision Threshold", f"{threshold:.2f}")

            # Filter options
            st.markdown("#### Filter Scored Cohort")
            filter_choice = st.radio("Display:", ["All Students", "At-Risk Students Only", "Placed Students Only"], horizontal=True)

            if filter_choice == "At-Risk Students Only":
                display_df = scored_batch[scored_batch["Predicted_Status"] == "Not Placed"]
            elif filter_choice == "Placed Students Only":
                display_df = scored_batch[scored_batch["Predicted_Status"] == "Placed"]
            else:
                display_df = scored_batch

            # Color-coded display
            st.dataframe(
                display_df.style.map(
                    lambda v: "background-color: #FEF2F2; color: #991B1B; font-weight: bold;" if v == "High Risk" else (
                        "background-color: #FFFBEB; color: #92400E;" if v == "Moderate Risk" else "background-color: #ECFDF5; color: #065F46;"
                    ),
                    subset=["Risk_Level"],
                ),
                use_container_width=True,
            )

            # Download Scored CSV
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
# TAB 3: Placement Analytics Dashboard
# =========================================================
with tab3:
    st.subheader("📊 Campus Recruitment Analytics & Cohort Exploration")
    st.write("Exploratory intelligence across student demographics, specialisations, and academic trends.")

    if dataset_df is not None:
        p_c1, p_c2 = st.columns(2)

        with p_c1:
            # Placement Rate by Specialisation
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
            # Placement Rate by Work Experience
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

        # Correlation Heatmap
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
# TAB 4: Model Benchmarks & Comparison
# =========================================================
with tab4:
    st.subheader("🤖 Machine Learning Model Benchmarking & Architecture Comparison")
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

        # Plotly comparison bar chart
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


# =========================================================
# TAB 5: Prediction Audit History (SQLite)
# =========================================================
with tab5:
    st.subheader("📜 Historical Prediction Audit Logs")
    st.write("All individual predictions evaluated by counselors or placement coordinators are automatically archived in SQLite for review and auditing.")

    history_df = get_prediction_history(limit=200)

    if not history_df.empty:
        c_h1, c_h2, c_h3 = st.columns(3)
        c_h1.metric("Archived Records", len(history_df))
        placed_hist = (history_df["predicted_status"].str.contains("Placed")).sum()
        c_h2.metric("Predicted Placed", placed_hist)
        c_h3.metric("At-Risk Students", len(history_df) - placed_hist)

        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "⬇️ Export Prediction History (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history_export.csv",
            mime="text/csv",
        )
    else:
        st.info("No prediction logs recorded yet. Make predictions in Tab 1 to see audit entries here.")


# =========================================================
# TAB 6: About System Architecture
# =========================================================
with tab6:
    st.subheader("ℹ️ System Architecture & Governance")
    st.markdown("""
    ### 🏛️ Placement Decision Support System Overview
    This platform assists college Training & Placement (T&P) cells in identifying students at risk of not clearing campus recruitment drives early in their academic journey.

    #### 🛡️ Anti-Leakage & Governance Measures
    1. **Target Leakage Prevention:** Columns such as `sl_no` (student ID) and `salary` (which exists strictly for placed students) are automatically purged prior to pipeline fitting.
    2. **Stratified Splitting & Validation:** 5-fold stratified cross validation ensures class proportions are preserved across folds.
    3. **Calibrated Probabilities:** Ensembles and linear models are wrapped in `CalibratedClassifierCV` using sigmoid/isotonic scaling so probabilities accurately reflect true placement risk.
    4. **Decision Threshold Tuning:** The decision threshold is tuned to optimize F1-score rather than relying on a naive 0.50 cutoff.

    #### 📦 Tech Stack
    - **Language & Framework:** Python 3.12, Streamlit
    - **ML & Preprocessing:** Scikit-Learn (Pipelines, ColumnTransformer), XGBoost
    - **Visualization:** Plotly Interactive Graphs & Matplotlib
    - **Reporting:** ReportLab PDF Engine
    - **Storage:** SQLite Database & Joblib Serialization
    """)
