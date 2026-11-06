"""Tech Profile Analyzer — manual-entry system for coding & project readiness.

Provides a schema for students to enter their LeetCode, GitHub, certification,
and project data.  Computes dimension-level and overall tech-readiness scores
and merges them with the ML placement probability for a holistic placement
assessment.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Schema & defaults
# ---------------------------------------------------------------------------

TECH_PROFILE_SCHEMA: Dict[str, type] = {
    "leetcode_easy": int,
    "leetcode_medium": int,
    "leetcode_hard": int,
    "leetcode_contest_rating": int,      # 0-3000
    "github_repos": int,
    "github_stars": int,
    "github_contributions_last_year": int,
    "github_languages": list,            # List[str]
    "certifications": list,              # List[str]
    "projects_count": int,
    "hackathons_participated": int,
}

TECH_PROFILE_DEFAULTS: Dict[str, Any] = {
    "leetcode_easy": 0,
    "leetcode_medium": 0,
    "leetcode_hard": 0,
    "leetcode_contest_rating": 0,
    "github_repos": 0,
    "github_stars": 0,
    "github_contributions_last_year": 0,
    "github_languages": [],
    "certifications": [],
    "projects_count": 0,
    "hackathons_participated": 0,
}

# Dimension weights for overall tech score.
_DIMENSION_WEIGHTS: Dict[str, float] = {
    "coding_proficiency": 0.35,
    "project_portfolio": 0.25,
    "competitive_edge": 0.20,
    "certification_strength": 0.20,
}

# Certification keywords that carry extra weight.
_HIGH_VALUE_CERTS = {
    "aws", "azure", "gcp", "google cloud", "tensorflow", "pytorch",
    "kubernetes", "docker", "data engineering", "machine learning",
    "deep learning", "professional", "associate", "solutions architect",
}

# Well-known certifications for the recommendation engine.
RECOMMENDED_CERTIFICATIONS: Dict[str, List[str]] = {
    "Mkt&Fin": [
        "CFA Level 1", "Bloomberg Market Concepts", "Google Data Analytics",
        "AWS Cloud Practitioner", "Excel Expert (MOS)",
    ],
    "Mkt&HR": [
        "Google Data Analytics", "SHRM-CP", "Tableau Desktop Specialist",
        "AWS Cloud Practitioner", "HubSpot Inbound Marketing",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _safe_int(val: Any, default: int = 0) -> int:
    """Coerce to int, falling back to *default*."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tech_readiness_score(tech_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Score a student's technical readiness across four dimensions.

    Each dimension is scored 0-100.  The overall score is a weighted average
    of the four dimensions and is mapped to a proficiency band.

    Dimensions
    ----------
    - **coding_proficiency** — LeetCode problem-solving breadth and depth.
      Points: easy×1 + medium×3 + hard×5, capped at 100.
    - **project_portfolio** — GitHub repos, stars, and contribution activity.
    - **competitive_edge** — Contest rating and hackathon participation.
    - **certification_strength** — Number and quality of certifications.

    Args:
        tech_profile: Dictionary matching ``TECH_PROFILE_SCHEMA``.

    Returns:
        Dictionary with keys ``coding_proficiency``, ``project_portfolio``,
        ``competitive_edge``, ``certification_strength``, ``overall``,
        ``band``, and ``dimension_details`` (per-dimension breakdowns).
    """
    tp = {k: tech_profile.get(k, TECH_PROFILE_DEFAULTS[k]) for k in TECH_PROFILE_SCHEMA}

    # ---- Coding Proficiency (0-100) ----
    easy = _safe_int(tp["leetcode_easy"])
    medium = _safe_int(tp["leetcode_medium"])
    hard = _safe_int(tp["leetcode_hard"])
    raw_lc = easy * 1 + medium * 3 + hard * 5
    coding_score = _clamp(raw_lc, 0.0, 100.0)

    coding_detail = (
        f"{easy} easy, {medium} medium, {hard} hard solved "
        f"(raw points: {raw_lc})"
    )

    # ---- Project Portfolio (0-100) ----
    repos = _safe_int(tp["github_repos"])
    stars = _safe_int(tp["github_stars"])
    contribs = _safe_int(tp["github_contributions_last_year"])
    langs = tp.get("github_languages") or []
    projects = _safe_int(tp["projects_count"])

    # Sub-scores (each 0-25, total 0-100)
    repo_score = _clamp(repos * 3.0, 0.0, 25.0)          # ~8 repos → 24/25
    star_score = _clamp(stars * 1.5, 0.0, 25.0)           # ~17 stars → 25
    contrib_score = _clamp(contribs / 10.0, 0.0, 25.0)    # ~250 → 25
    proj_score = _clamp(projects * 5.0, 0.0, 25.0)        # 5 projects → 25
    portfolio_score = _clamp(repo_score + star_score + contrib_score + proj_score)

    portfolio_detail = (
        f"{repos} repos, {stars} stars, {contribs} contributions/yr, "
        f"{projects} projects, {len(langs)} languages"
    )

    # ---- Competitive Edge (0-100) ----
    contest_rating = _safe_int(tp["leetcode_contest_rating"])
    hackathons = _safe_int(tp["hackathons_participated"])

    # Contest rating: map 0-3000 → 0-70
    contest_score = _clamp(contest_rating / 3000.0 * 70.0, 0.0, 70.0)
    # Hackathons: each worth 10, max 30
    hack_score = _clamp(hackathons * 10.0, 0.0, 30.0)
    competitive_score = _clamp(contest_score + hack_score)

    competitive_detail = (
        f"Contest rating: {contest_rating}, {hackathons} hackathon(s)"
    )

    # ---- Certification Strength (0-100) ----
    certs: list = tp.get("certifications") or []
    num_certs = len(certs)
    # Each cert = 15 pts base; high-value keywords add 10 bonus each
    base_cert = _clamp(num_certs * 15.0, 0.0, 60.0)
    bonus = sum(
        10.0
        for c in certs
        if any(kw in c.lower() for kw in _HIGH_VALUE_CERTS)
    )
    cert_score = _clamp(base_cert + bonus)

    cert_detail = (
        f"{num_certs} certification(s), "
        f"{int(bonus // 10)} high-value"
    )

    # ---- Overall ----
    overall = round(
        coding_score * _DIMENSION_WEIGHTS["coding_proficiency"]
        + portfolio_score * _DIMENSION_WEIGHTS["project_portfolio"]
        + competitive_score * _DIMENSION_WEIGHTS["competitive_edge"]
        + cert_score * _DIMENSION_WEIGHTS["certification_strength"],
        2,
    )

    if overall >= 80:
        band = "Expert"
    elif overall >= 60:
        band = "Advanced"
    elif overall >= 35:
        band = "Intermediate"
    else:
        band = "Beginner"

    return {
        "coding_proficiency": round(coding_score, 2),
        "project_portfolio": round(portfolio_score, 2),
        "competitive_edge": round(competitive_score, 2),
        "certification_strength": round(cert_score, 2),
        "dimension_scores": {
            "coding_proficiency": round(coding_score, 2),
            "project_portfolio": round(portfolio_score, 2),
            "competitive_edge": round(competitive_score, 2),
            "certification_strength": round(cert_score, 2),
        },
        "overall": overall,
        "overall_score": overall,
        "band": band,
        "dimension_details": {
            "coding_proficiency": coding_detail,
            "project_portfolio": portfolio_detail,
            "competitive_edge": competitive_detail,
            "certification_strength": cert_detail,
        },
    }


def get_tech_recommendations(
    tech_profile: Dict[str, Any],
    specialisation: str = "Mkt&Fin",
) -> List[str]:
    """Generate personalised tech-improvement suggestions.

    Analyses gaps in the four readiness dimensions and returns concrete,
    actionable recommendations.

    Args:
        tech_profile: Dictionary matching ``TECH_PROFILE_SCHEMA``.
        specialisation: MBA specialisation (``Mkt&Fin`` or ``Mkt&HR``).

    Returns:
        List of human-readable recommendation strings.
    """
    scores = compute_tech_readiness_score(tech_profile)
    recs: List[str] = []

    # --- Coding ---
    if scores["coding_proficiency"] < 40:
        easy = _safe_int(tech_profile.get("leetcode_easy", 0))
        medium = _safe_int(tech_profile.get("leetcode_medium", 0))
        if medium < 30:
            recs.append(
                "Solve 50+ medium-level LeetCode problems focusing on "
                "Arrays, Strings, and Dynamic Programming to clear most "
                "online assessments."
            )
        if easy < 50:
            recs.append(
                "Build foundational speed by completing at least 50 easy "
                "LeetCode problems across all major topics."
            )
        recs.append(
            "Practice 1-2 coding problems daily; consistency matters more "
            "than cramming."
        )
    elif scores["coding_proficiency"] < 70:
        recs.append(
            "Push into hard-level problems (Graphs, DP on Trees, Segment "
            "Trees) to differentiate yourself in Tier-1 OAs."
        )

    # --- Portfolio ---
    if scores["project_portfolio"] < 40:
        recs.append(
            "Create 2 end-to-end projects: one data pipeline (ETL → "
            "dashboard) and one full-stack web app. Host them on GitHub "
            "with clear READMEs."
        )
        if _safe_int(tech_profile.get("github_contributions_last_year", 0)) < 100:
            recs.append(
                "Increase GitHub activity to ≥200 contributions/year by "
                "contributing to open-source or maintaining personal projects."
            )
    elif scores["project_portfolio"] < 70:
        recs.append(
            "Add a deployed ML or analytics project with live demo link "
            "to strengthen your portfolio."
        )

    # --- Competitive Edge ---
    if scores["competitive_edge"] < 30:
        recs.append(
            "Participate in at least 2 hackathons (e.g., MLH, Devpost) "
            "to demonstrate teamwork and rapid prototyping skills."
        )
        if _safe_int(tech_profile.get("leetcode_contest_rating", 0)) < 1400:
            recs.append(
                "Start participating in weekly LeetCode contests to build "
                "a competitive rating (target: 1500+)."
            )
    elif scores["competitive_edge"] < 60:
        recs.append(
            "Target a LeetCode contest rating of 1800+ and participate in "
            "Kaggle or Codeforces for cross-platform visibility."
        )

    # --- Certifications ---
    certs = tech_profile.get("certifications") or []
    if scores["certification_strength"] < 30:
        spec_certs = RECOMMENDED_CERTIFICATIONS.get(specialisation, [])
        if spec_certs:
            recs.append(
                f"Earn at least 2 certifications relevant to {specialisation}: "
                f"{', '.join(spec_certs[:3])}."
            )
        else:
            recs.append(
                "Earn a cloud certification (AWS Cloud Practitioner or "
                "Google Cloud Digital Leader) — widely valued by recruiters."
            )
    elif scores["certification_strength"] < 60:
        has_cloud = any(
            kw in c.lower()
            for c in certs
            for kw in ("aws", "azure", "gcp", "google cloud")
        )
        if not has_cloud:
            recs.append(
                "Add a cloud-platform certification (AWS / Azure / GCP) "
                "to complement your existing credentials."
            )

    # --- General if nothing specific flagged ---
    if not recs:
        recs.append(
            "Your tech profile is strong! Focus on mock interviews and "
            "system-design practice for Tier-1 rounds."
        )

    return recs


def compute_enhanced_placement_score(
    placement_prob: float,
    tech_score: float,
) -> Dict[str, Any]:
    """Combine ML placement probability with tech readiness for a holistic score.

    ``enhanced_score = 0.65 × placement_prob × 100 + 0.35 × tech_score``

    Args:
        placement_prob: ML-model placement probability (0.0–1.0).
        tech_score: Overall tech readiness score (0-100) from
                    :func:`compute_tech_readiness_score`.

    Returns:
        Dictionary with ``enhanced_score``, ``academic_component``,
        ``tech_component``, ``band``, and ``interpretation``.
    """
    academic_component = round(0.65 * _clamp(placement_prob, 0.0, 1.0) * 100, 2)
    tech_component = round(0.35 * _clamp(tech_score), 2)
    enhanced = round(academic_component + tech_component, 2)
    delta_vs_academic = round(enhanced - (float(placement_prob) * 100.0), 2)

    if enhanced >= 80:
        band = "Excellent"
        interpretation = (
            "Outstanding profile. You are well-positioned for Tier-1 and "
            "Dream company placements. Focus on interview preparation."
        )
    elif enhanced >= 65:
        band = "Good"
        interpretation = (
            "Solid profile with room for improvement. Target Core/Tier-2 "
            "companies confidently and stretch for Tier-1 with focused effort."
        )
    elif enhanced >= 45:
        band = "Average"
        interpretation = (
            "Moderate readiness. Strengthen weak areas (academics or tech "
            "skills) to move from mass-recruiter range to core companies."
        )
    else:
        band = "Needs Improvement"
        interpretation = (
            "Significant gaps in both academic and technical dimensions. "
            "Prioritise building a coding habit and improving key scores."
        )

    return {
        "enhanced_score": enhanced,
        "academic_component": academic_component,
        "tech_component": tech_component,
        "delta_vs_academic": delta_vs_academic,
        "band": band,
        "interpretation": interpretation,
    }

