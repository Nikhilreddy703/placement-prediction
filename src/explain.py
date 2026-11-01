"""Explainability, student coaching recommendations, and what-if analysis."""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import (
    NUMERIC_FEATURES,
    THRESHOLD_HIGH_RISK,
    THRESHOLD_MODERATE_RISK,
)


def categorize_risk(prob: float) -> Tuple[str, str, str]:
    """Map placement probability to a risk level, badge text, and color code.

    Returns:
        (risk_level, badge_text, color_hex)
    """
    if prob >= THRESHOLD_MODERATE_RISK:
        return "Low Risk", "Likely to be Placed (Low Risk)", "#10B981"
    elif prob >= THRESHOLD_HIGH_RISK:
        return "Moderate Risk", "Borderline / Moderate Risk", "#F59E0B"
    else:
        return "High Risk", "High Priority / Placement At-Risk", "#EF4444"


def generate_coaching_recommendations(
    profile: Dict[str, Any],
    prob: float,
    pipeline: Optional[Any] = None
) -> List[str]:
    """Generate both rule-based and model-driven personalized action items."""
    recs = []
    degree_p = float(profile.get("degree_p", 0))
    etest_p = float(profile.get("etest_p", 0))
    workex = str(profile.get("workex", "No")).strip().lower()
    mba_p = float(profile.get("mba_p", 0))
    ssc_p = float(profile.get("ssc_p", 0))
    hsc_p = float(profile.get("hsc_p", 0))

    # Priority coaching notice based on risk level
    if prob < THRESHOLD_HIGH_RISK:
        recs.append(
            "🚨 **Urgent Intervention Required:** Schedule weekly 1-on-1 counseling with the career placement cell."
        )
    elif prob < THRESHOLD_MODERATE_RISK:
        recs.append(
            "⚠️ **Targeted Preparation:** Enroll in the intensive 4-week placement readiness workshop."
        )

    # Work experience impact
    if workex == "no":
        recs.append(
            "💼 **Practical Industry Exposure:** Complete a verified 6-8 week summer internship or capstone industry project to offset lack of full-time experience."
        )

    # Employability test score impact
    if etest_p < 70.0:
        gap = round(75.0 - etest_p, 1)
        recs.append(
            f"🧠 **Aptitude & Problem Solving:** Practice quantitative and logical reasoning tests daily. Aiming for +{gap}% in college mock tests significantly boosts company screening clearance."
        )

    # Undergraduate academic performance
    if degree_p < 65.0:
        recs.append(
            "🛠️ **Technical Portfolio:** Build 2-3 showcase projects (GitHub repositories, live deployed applications, case studies) to overcome academic screening cutoffs."
        )

    # Post-grad specialization readiness
    if mba_p < 65.0:
        recs.append(
            "📊 **Domain Certifications:** Obtain recognized domain credentials (e.g., Financial Modeling, Google Data Analytics, or HR Talent Analytics certifications)."
        )

    # High-performing profile encouragement
    if prob >= 0.75 and len(recs) == 0:
        recs.append(
            "⭐ **Placement Ready:** Strong overall profile! Focus on company-specific case competitions, leadership mock interviews, and pre-placement talks."
        )

    return recs


def compute_shap_contributions(
    pipeline: Any,
    student_df: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    """Compute true individual SHAP values for the student's prediction.

    Works for Tree models (Random Forest, XGBoost, Gradient Boosting).
    Returns a sorted DataFrame with [Feature, SHAP_Value, Impact, Plain_English_Insight]
    """
    try:
        import shap

        estimator = pipeline.estimator if hasattr(pipeline, "estimator") else pipeline
        pre = estimator.named_steps.get("pre")
        clf = estimator.named_steps.get("clf")

        if pre is None or clf is None:
            return pd.DataFrame()

        # Check if clf is supported by TreeExplainer
        if not hasattr(clf, "feature_importances_"):
            return pd.DataFrame()

        feature_names = [
            n.replace("num__", "").replace("cat__", "")
            for n in pre.get_feature_names_out()
        ]
        X_trans = pre.transform(student_df)
        explainer = shap.TreeExplainer(clf)
        explanation = explainer(X_trans)
        raw_vals = explanation.values

        # If shape is (1, n_features, 2), slice positive class (Placed)
        if len(raw_vals.shape) == 3 and raw_vals.shape[2] == 2:
            shap_scores = raw_vals[0, :, 1]
        elif len(raw_vals.shape) == 2:
            shap_scores = raw_vals[0, :]
        else:
            shap_scores = raw_vals.flatten()[:len(feature_names)]

        df_shap = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Value": shap_scores,
        })
        df_shap["Abs_SHAP"] = df_shap["SHAP_Value"].abs()
        df_shap = df_shap.sort_values(by="Abs_SHAP", ascending=False).head(top_n).copy()

        insights = []
        impacts = []
        for _, row in df_shap.iterrows():
            feat = row["Feature"]
            val = row["SHAP_Value"]
            if val > 0.005:
                impact = "Boosts Placement Probability (+)"
                insight = f"This attribute strongly pushed the student's probability UP by +{val:.3f} log-odds."
            elif val < -0.005:
                impact = "Lowers Placement Probability (-)"
                insight = f"This attribute pulled the student's probability DOWN by {val:.3f} log-odds."
            else:
                impact = "Neutral Impact (~)"
                insight = "Minimal impact on final prediction."
            insights.append(insight)
            impacts.append(impact)

        df_shap["Impact"] = impacts
        df_shap["Plain_English_Insight"] = insights
        return df_shap.drop(columns=["Abs_SHAP"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def compute_student_feature_contributions(
    pipeline: Any,
    student_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute directional impact of student attributes vs average student profile.

    Returns a DataFrame with [Feature, Student Value, Average Value, Impact, Plain English Insight]
    """
    contributions = []

    # If reference dataset is available, compare numeric features against the cohort
    if reference_df is not None:
        for col in NUMERIC_FEATURES:
            if col in student_df.columns and col in reference_df.columns:
                stud_val = float(student_df[col].iloc[0])
                avg_val = float(reference_df[col].mean())
                diff = stud_val - avg_val

                if diff > 3.0:
                    impact = "Positive (+)"
                    plain = f"Scored {stud_val:.1f}% (+{diff:.1f}% higher than cohort average of {avg_val:.1f}%)."
                elif diff < -3.0:
                    impact = "Negative (-)"
                    plain = f"Scored {stud_val:.1f}% ({abs(diff):.1f}% lower than cohort average of {avg_val:.1f}%)."
                else:
                    impact = "Neutral (~)"
                    plain = f"Scored {stud_val:.1f}% (consistent with cohort average of {avg_val:.1f}%)."

                contributions.append({
                    "Feature": col,
                    "Student Value": round(stud_val, 1),
                    "Cohort Average": round(avg_val, 1),
                    "Difference": round(diff, 1),
                    "Impact": impact,
                    "Insight": plain,
                })

    return pd.DataFrame(contributions)


def simulate_what_if(
    pipeline: Any,
    base_profile: Dict[str, Any],
    alterations: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate changes to student profile and calculate the probability delta."""
    from src.data import validate_student_record

    base_df = validate_student_record(base_profile)
    altered_profile = dict(base_profile)
    altered_profile.update(alterations)
    altered_df = validate_student_record(altered_profile)

    base_prob = float(pipeline.predict_proba(base_df)[0][1])
    new_prob = float(pipeline.predict_proba(altered_df)[0][1])
    delta = new_prob - base_prob

    return {
        "base_probability": base_prob,
        "simulated_probability": new_prob,
        "delta": delta,
        "delta_percentage": delta * 100,
        "improved": delta > 0,
    }


def get_radar_comparison_data(
    student_profile: Dict[str, Any],
    placed_benchmark: Dict[str, float],
) -> pd.DataFrame:
    """Generate normalized metrics (0-100) comparing student to placed student average."""
    metrics = ["10th (SSC)", "12th (HSC)", "Degree", "E-Test", "MBA Aggregate"]
    cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]

    student_scores = [float(student_profile.get(c, 50.0)) for c in cols]
    bench_scores = [float(placed_benchmark.get(c, 68.0)) for c in cols]

    df = pd.DataFrame({
        "Metric": metrics,
        "Student": student_scores,
        "Placed Benchmark": bench_scores,
    })
    return df

