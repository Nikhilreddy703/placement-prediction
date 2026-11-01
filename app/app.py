"""Campus Placement Prediction & Student Success Analytics Dashboard.

Run from the project folder:
    streamlit run app/app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Student Placement Prediction | Placement Cell Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "placement_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
DATA_PATH = ROOT / "data" / "raw" / "placement.csv"
RESULTS_PATH = ROOT / "reports" / "results.csv"
FIG_DIR = ROOT / "reports" / "figures"
FEATURE_IMP_PATH = ROOT / "reports" / "feature_importance.csv"

# ---------------------------------------------------------
# Custom Styling (Glassmorphic, Modern Gradient, Polished)
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

    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }

    .result-placed {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.2);
    }

    .result-risk {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 20px -5px rgba(245, 158, 11, 0.2);
    }

    .rec-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.7rem;
        font-size: 0.95rem;
        color: #1E293B;
    }

    .metric-chip {
        background: #F1F5F9;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Resources
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


@st.cache_data
def load_raw_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return None


@st.cache_data
def load_feature_imp():
    if FEATURE_IMP_PATH.exists():
        return pd.read_csv(FEATURE_IMP_PATH)
    return None


model, meta = load_model_and_meta()

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🎓 Campus Placement Predictor & Student Success Analytics</h1>
    <p>Predict student placement likelihood using academic performance, skills, aptitude test results, and internship history. Identify students at risk early to provide personalized coaching, skill training, and mock interview support.</p>
    <div>
        <span class="badge badge-primary">✨ Binary ML Classification</span>
        <span class="badge badge-success">🎯 5-Model Comparative Benchmark</span>
        <span class="badge badge-primary">📊 Early Risk Intervention System</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or meta is None:
    st.error("⚠️ Trained model or metadata not found. Please run `python src/train.py` first to generate the models.")
    st.stop()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏆 Active ML Model")
    st.success(f"**Selected Model:** {meta.get('best_model', 'KNN')}")

    st.markdown("#### 📈 Benchmark Metrics")
    m = meta.get("metrics", {})
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Accuracy", f"{m.get('Accuracy', 0):.1%}")
        st.metric("Recall", f"{m.get('Recall', 0):.1%}")
    with col_s2:
        st.metric("Precision", f"{m.get('Precision', 0):.1%}")
        st.metric("CV F1-Score", f"{m.get('CV F1', 0):.3f}")

    st.markdown("---")
    st.markdown("### 🎯 Placement Cell Objectives")
    st.markdown("""
    - **Early Identification:** Flag students with low placement probability before campus drives begin.
    - **Targeted Coaching:** Focus remedial sessions on aptitude (etest_p) and resume building.
    - **Internship Encouragement:** Prioritize students without work experience for live projects.
    """)
    st.markdown("---")
    st.caption("Developed for Academic Training & Placement Cells.")


# ---------------------------------------------------------
# Field Definitions & Friendly Mappings
# ---------------------------------------------------------
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
    "hsc_b": "Board of higher secondary education",
    "hsc_s": "Specialization in 12th standard (Commerce, Science, Arts)",
    "degree_p": "Undergraduate graduation percentage",
    "degree_t": "Undergraduate field (Comm&Mgmt, Sci&Tech, Others)",
    "workex": "Prior full-time work experience or verified internship",
    "etest_p": "Score in college employability & aptitude assessment",
    "specialisation": "Post-graduation specialisation (Marketing & Finance / Marketing & HR)",
    "mba_p": "Post-graduation MBA aggregate percentage",
}

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


def generate_recommendations(profile: dict, prob: float):
    """Generate targeted guidance for the training and placement cell."""
    recs = []
    degree_p = float(profile.get("degree_p", 0))
    etest_p = float(profile.get("etest_p", 0))
    workex = str(profile.get("workex", "No")).strip().lower()
    mba_p = float(profile.get("mba_p", 0))
    ssc_p = float(profile.get("ssc_p", 0))

    if prob < 0.60:
        recs.append("🚨 **Priority Coaching:** Assign this student to the intensive 4-week placement readiness bootcamp.")

    if workex == "no":
        recs.append("💼 **Internship / Capstone:** Encourage completing a 6-to-8 week industry internship or live client capstone to gain practical experience.")

    if etest_p < 70:
        recs.append("🧠 **Aptitude Drills:** Enroll in daily quantitative problem solving, data interpretation, and logical reasoning practice tests.")

    if degree_p < 65:
        recs.append("🛠️ **Skills & Portfolio:** Direct student to build showcase-ready portfolio projects (GitHub, live web apps, case analyses) to offset academic cutoffs.")

    if mba_p < 65:
        recs.append("📊 **Domain Certifications:** Recommend pursuing professional certifications (e.g., Financial Modeling, Google Data Analytics, SHRM HR Fundamentals).")

    if not recs or prob >= 0.75:
        recs.append("⭐ **Profile Ready:** Excellent profile! Focus on advanced company-specific mock interviews and leadership case studies.")

    return recs


# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Student Predictor",
    "📊 Exploratory Data Analysis",
    "🤖 Model Comparison & Evaluation",
    "🔍 Key Influencing Factors",
    "📁 Batch Student Evaluation",
])

# =========================================================
# TAB 1: Student Predictor
# =========================================================
with tab1:
    st.subheader("Predict Placement Likelihood for a Student")
    st.write("Fill in the student's profile details below, or load a quick preset profile to test common student scenarios.")

    # Preset selection
    preset_cols = st.columns(len(PRESETS))
    for i, (p_name, p_data) in enumerate(PRESETS.items()):
        if preset_cols[i].button(p_name, use_container_width=True):
            st.session_state["student_profile"] = p_data

    current_profile = st.session_state.get("student_profile", PRESETS["🌟 High-Scoring All-Rounder"])

    st.markdown("---")

    # Form layout with grouped columns
    col_left, col_right = st.columns(2)

    form_inputs = {}

    with col_left:
        st.markdown("#### 📚 Secondary & Higher Secondary Education")
        form_inputs["ssc_p"] = st.slider(
            FIELD_LABELS.get("ssc_p", "ssc_p"),
            min_value=40.0, max_value=100.0,
            value=float(current_profile.get("ssc_p", 70.0)),
            step=0.1, help=FIELD_HELP.get("ssc_p")
        )
        form_inputs["ssc_b"] = st.selectbox(
            FIELD_LABELS.get("ssc_b", "ssc_b"),
            options=meta["categorical"].get("ssc_b", ["Central", "Others"]),
            index=meta["categorical"].get("ssc_b", ["Central", "Others"]).index(current_profile.get("ssc_b", "Central")) if current_profile.get("ssc_b") in meta["categorical"].get("ssc_b", []) else 0,
            help=FIELD_HELP.get("ssc_b")
        )
        form_inputs["hsc_p"] = st.slider(
            FIELD_LABELS.get("hsc_p", "hsc_p"),
            min_value=40.0, max_value=100.0,
            value=float(current_profile.get("hsc_p", 70.0)),
            step=0.1, help=FIELD_HELP.get("hsc_p")
        )
        form_inputs["hsc_b"] = st.selectbox(
            FIELD_LABELS.get("hsc_b", "hsc_b"),
            options=meta["categorical"].get("hsc_b", ["Central", "Others"]),
            index=meta["categorical"].get("hsc_b", ["Central", "Others"]).index(current_profile.get("hsc_b", "Others")) if current_profile.get("hsc_b") in meta["categorical"].get("hsc_b", []) else 0,
            help=FIELD_HELP.get("hsc_b")
        )
        form_inputs["hsc_s"] = st.selectbox(
            FIELD_LABELS.get("hsc_s", "hsc_s"),
            options=meta["categorical"].get("hsc_s", ["Commerce", "Science", "Arts"]),
            index=meta["categorical"].get("hsc_s", ["Commerce", "Science", "Arts"]).index(current_profile.get("hsc_s", "Commerce")) if current_profile.get("hsc_s") in meta["categorical"].get("hsc_s", []) else 0,
            help=FIELD_HELP.get("hsc_s")
        )
        form_inputs["gender"] = st.selectbox(
            FIELD_LABELS.get("gender", "gender"),
            options=meta["categorical"].get("gender", ["M", "F"]),
            index=meta["categorical"].get("gender", ["M", "F"]).index(current_profile.get("gender", "M")) if current_profile.get("gender") in meta["categorical"].get("gender", []) else 0,
        )

    with col_right:
        st.markdown("#### 🎓 Higher Education & Employability")
        form_inputs["degree_p"] = st.slider(
            FIELD_LABELS.get("degree_p", "degree_p"),
            min_value=45.0, max_value=100.0,
            value=float(current_profile.get("degree_p", 68.0)),
            step=0.1, help=FIELD_HELP.get("degree_p")
        )
        form_inputs["degree_t"] = st.selectbox(
            FIELD_LABELS.get("degree_t", "degree_t"),
            options=meta["categorical"].get("degree_t", ["Comm&Mgmt", "Sci&Tech", "Others"]),
            index=meta["categorical"].get("degree_t", ["Comm&Mgmt", "Sci&Tech", "Others"]).index(current_profile.get("degree_t", "Comm&Mgmt")) if current_profile.get("degree_t") in meta["categorical"].get("degree_t", []) else 0,
            help=FIELD_HELP.get("degree_t")
        )
        form_inputs["workex"] = st.selectbox(
            FIELD_LABELS.get("workex", "workex"),
            options=meta["categorical"].get("workex", ["Yes", "No"]),
            index=meta["categorical"].get("workex", ["Yes", "No"]).index(current_profile.get("workex", "Yes")) if current_profile.get("workex") in meta["categorical"].get("workex", []) else 0,
            help=FIELD_HELP.get("workex")
        )
        form_inputs["etest_p"] = st.slider(
            FIELD_LABELS.get("etest_p", "etest_p"),
            min_value=40.0, max_value=100.0,
            value=float(current_profile.get("etest_p", 75.0)),
            step=0.1, help=FIELD_HELP.get("etest_p")
        )
        form_inputs["specialisation"] = st.selectbox(
            FIELD_LABELS.get("specialisation", "specialisation"),
            options=meta["categorical"].get("specialisation", ["Mkt&Fin", "Mkt&HR"]),
            index=meta["categorical"].get("specialisation", ["Mkt&Fin", "Mkt&HR"]).index(current_profile.get("specialisation", "Mkt&Fin")) if current_profile.get("specialisation") in meta["categorical"].get("specialisation", []) else 0,
            help=FIELD_HELP.get("specialisation")
        )
        form_inputs["mba_p"] = st.slider(
            FIELD_LABELS.get("mba_p", "mba_p"),
            min_value=45.0, max_value=100.0,
            value=float(current_profile.get("mba_p", 65.0)),
            step=0.1, help=FIELD_HELP.get("mba_p")
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Run Placement Risk Prediction", type="primary", use_container_width=True)

    if predict_clicked or "last_prediction" in st.session_state:
        df_row = pd.DataFrame([form_inputs])
        label = int(model.predict(df_row)[0])
        prob = float(model.predict_proba(df_row)[0][1]) if hasattr(model, "predict_proba") else (0.85 if label == 1 else 0.35)
        st.session_state["last_prediction"] = (label, prob, form_inputs)

        st.markdown("### Prediction Outcome")

        col_res1, col_res2 = st.columns([1.2, 1.8])

        with col_res1:
            if label == 1:
                st.markdown(f"""
                <div class="result-placed">
                    <h2 style="color: #065F46; margin-top:0;">✅ Likely to be Placed</h2>
                    <p style="color: #047857; font-size: 1.1rem; font-weight: 600;">Status: Low Placement Risk</p>
                    <hr style="border-color: #6EE7B7;">
                    <p style="color: #065F46;">Estimated Placement Probability: <b>{prob:.1%}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-risk">
                    <h2 style="color: #92400E; margin-top:0;">⚠️ At-Risk of Not Being Placed</h2>
                    <p style="color: #B45309; font-size: 1.1rem; font-weight: 600;">Status: High Coaching Priority</p>
                    <hr style="border-color: #FCD34D;">
                    <p style="color: #92400E;">Estimated Placement Probability: <b>{prob:.1%}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write(f"**Confidence Gauge:**")
            st.progress(prob)

        with col_res2:
            st.markdown("#### 💡 Training & Placement Cell Recommendations")
            recs = generate_recommendations(form_inputs, prob)
            for r in recs:
                st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)


# =========================================================
# TAB 2: Exploratory Data Analysis
# =========================================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.write("Understand the distribution, class balance, and patterns across student academic profiles.")

    raw_df = load_raw_data()
    if raw_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        total_students = len(raw_df)
        placed_students = (raw_df["status"].astype(str).str.lower() == "placed").sum()
        placed_pct = (placed_students / total_students) * 100
        avg_degree = raw_df["degree_p"].mean()
        workex_pct = (raw_df["workex"].astype(str).str.lower() == "yes").mean() * 100

        c1.metric("Total Students", f"{total_students}")
        c2.metric("Placement Rate", f"{placed_pct:.1f}%")
        c3.metric("Avg Degree Score", f"{avg_degree:.1f}%")
        c4.metric("With Work Experience", f"{workex_pct:.1f}%")

        with st.expander("🔍 View Raw Dataset Records"):
            st.dataframe(raw_df.head(20), use_container_width=True)

    st.markdown("#### Visual Analysis")
    eda_col1, eda_col2 = st.columns(2)

    class_balance_img = FIG_DIR / "class_balance.png"
    if class_balance_img.exists():
        with eda_col1:
            st.image(str(class_balance_img), caption="Figure 1: Target Class Balance (Placed vs Not Placed)", use_container_width=True)

    corr_img = FIG_DIR / "correlation.png"
    if corr_img.exists():
        with eda_col2:
            st.image(str(corr_img), caption="Figure 2: Academic Metrics Correlation Heatmap", use_container_width=True)

    numeric_img = FIG_DIR / "numeric_by_status.png"
    if numeric_img.exists():
        st.markdown("#### Academic Score Distributions by Placement Outcome")
        st.image(str(numeric_img), caption="Figure 3: Score Breakdown by Status (Boxplots)", use_container_width=True)


# =========================================================
# TAB 3: Model Comparison & Evaluation
# =========================================================
with tab3:
    st.subheader("🤖 Machine Learning Model Benchmarking")
    st.write("We trained and compared 5 distinct classification algorithms using stratified 5-fold cross-validation on the training set, followed by evaluation on an untouched test set.")

    results_df = load_results()
    if results_df is not None:
        st.dataframe(
            results_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1", "CV F1"], color="#D1FAE5"),
            use_container_width=True
        )

    st.markdown("#### Performance Visualizations")
    m_col1, m_col2 = st.columns(2)

    comp_img = FIG_DIR / "model_comparison.png"
    if comp_img.exists():
        with m_col1:
            st.image(str(comp_img), caption="Figure 4: Test Accuracy vs Cross-Validated F1", use_container_width=True)

    cm_img = FIG_DIR / "confusion_matrix.png"
    if cm_img.exists():
        with m_col2:
            st.image(str(cm_img), caption=f"Figure 5: Confusion Matrix for Best Model ({meta.get('best_model')})", use_container_width=True)

    st.markdown("""
    #### 📚 Evaluation Metrics Explained
    - **Accuracy:** Overall proportion of correct placement predictions.
    - **Precision:** Of all students predicted to be placed, what fraction actually got placed (minimizes false alarms).
    - **Recall (Sensitivity):** Of all students who were actually placed, how many were correctly detected.
    - **F1-Score:** Harmonic mean of precision and recall; balanced metric for tabular classification.
    - **CV F1 (Cross-Validation F1):** Mean F1-score across 5 stratified folds on training data. Used as the golden standard for model selection.
    """)


# =========================================================
# TAB 4: Key Influencing Factors
# =========================================================
with tab4:
    st.subheader("🔍 What Factors Influence Campus Placement Most?")
    st.write("Feature importance analysis identifies the strongest drivers that distinguish placed students from non-placed students.")

    fi_col1, fi_col2 = st.columns([1.3, 1.0])

    fi_img = FIG_DIR / "feature_importance.png"
    with fi_col1:
        if fi_img.exists():
            st.image(str(fi_img), caption="Figure 6: Top Predictive Features", use_container_width=True)

    with fi_col2:
        df_imp = load_feature_imp()
        if df_imp is not None:
            st.markdown("#### Ranked Feature Weights")
            st.dataframe(df_imp.head(10), use_container_width=True)

    st.markdown("""
    #### 💡 Key Takeaways for Students & Placement Cells
    1. **Undergraduate Performance (degree_p):** The single most influential academic predictor. Recruiters frequently set initial screening cutoffs at 65-70%.
    2. **Work Experience & Internships (workex):** Candidates with prior internship or work experience have a noticeably higher placement rate (+16.8% higher conversion).
    3. **Employability Test Score (etest_p):** Strong aptitude scores are essential for passing first-round quantitative and analytical screening filters.
    4. **Secondary & Higher Secondary Marks (ssc_p, hsc_p):** Consistent academic track record gives confidence to corporate hiring managers.
    """)


# =========================================================
# TAB 5: Batch Student Evaluation
# =========================================================
with tab5:
    st.subheader("📁 Batch Student Evaluation (CSV Upload)")
    st.write("Upload a CSV file containing multiple students' details to generate batch predictions and flag at-risk students for coaching.")

    sample_template = pd.DataFrame([{
        "gender": "M", "ssc_p": 68.0, "ssc_b": "Central", "hsc_p": 65.0, "hsc_b": "Others",
        "hsc_s": "Commerce", "degree_p": 66.0, "degree_t": "Comm&Mgmt", "workex": "No",
        "etest_p": 72.0, "specialisation": "Mkt&Fin", "mba_p": 62.0
    }])
    st.download_button(
        "📥 Download Sample CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="student_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Student Profiles (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** student records.")

            # Make predictions
            batch_preds = model.predict(batch_df)
            batch_probs = model.predict_proba(batch_df)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(batch_df)

            result_df = batch_df.copy()
            result_df["Placement_Prediction"] = ["Placed" if p == 1 else "Not Placed" for p in batch_preds]
            result_df["Placement_Probability"] = [round(float(pr), 3) for pr in batch_probs]
            result_df["Risk_Level"] = ["Low Risk" if pr >= 0.60 else ("Moderate Risk" if pr >= 0.40 else "High Risk") for pr in batch_probs]

            # Summary metrics
            b_c1, b_c2, b_c3 = st.columns(3)
            safe_count = (result_df["Placement_Prediction"] == "Placed").sum()
            risk_count = (result_df["Placement_Prediction"] == "Not Placed").sum()
            b_c1.metric("Likely Placed", f"{safe_count}")
            b_c2.metric("At-Risk (Needs Coaching)", f"{risk_count}", delta=f"-{risk_count}", delta_color="inverse")
            b_c3.metric("Predicted Batch Placement Rate", f"{(safe_count / len(result_df)):.1%}")

            st.markdown("#### Scored Batch Results")
            st.dataframe(result_df, use_container_width=True)

            csv_download = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Scored Batch Predictions (CSV)",
                data=csv_download,
                file_name="placement_predictions_batch.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Error evaluating batch CSV: {e}")
