"""Multi-Tier Placement Prediction Engine.

Estimates likelihood of placement into Dream / Core / Mass-Recruiter company
tiers by combining an ML-derived placement probability with a heuristic
profile-strength composite score.  All tier boundaries use smooth sigmoid
scaling so there are no hard cutoffs.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weights for the composite profile-strength score (must sum to 1.0).
_PROFILE_WEIGHTS: Dict[str, float] = {
    "ssc_p": 0.10,
    "hsc_p": 0.10,
    "degree_p": 0.20,
    "etest_p": 0.25,
    "mba_p": 0.15,
    "workex": 0.10,
    "specialisation": 0.10,
}

# Tier metadata definitions.
TIER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "tier1": {
        "label": "Dream / Product Companies",
        "salary_range": "₹15-40 LPA",
        "companies": [
            "Google", "Microsoft", "Goldman Sachs", "Amazon",
            "JP Morgan", "Adobe", "Uber", "Flipkart",
        ],
        "strength_center": 80.0,
        "prob_center": 0.75,
    },
    "tier2": {
        "label": "Core / Specialized",
        "salary_range": "₹7-14 LPA",
        "companies": [
            "Deloitte", "Accenture", "Wipro Digital", "Capgemini",
            "EY", "KPMG", "PwC", "ZS Associates",
        ],
        "strength_center": 65.0,
        "prob_center": 0.55,
    },
    "tier3": {
        "label": "Mass Recruiters",
        "salary_range": "₹3.5-6 LPA",
        "companies": [
            "TCS", "Infosys", "Cognizant", "HCL",
            "Tech Mahindra", "Mphasis", "Mindtree", "LTIMindtree",
        ],
        "strength_center": 50.0,
        "prob_center": 0.40,
    },
}

# Benchmark average profiles for radar / bar-chart comparison.
TIER_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "Tier 1 Avg": {
        "ssc_p": 82.0, "hsc_p": 80.0, "degree_p": 78.0,
        "etest_p": 85.0, "mba_p": 75.0,
    },
    "Tier 2 Avg": {
        "ssc_p": 72.0, "hsc_p": 70.0, "degree_p": 70.0,
        "etest_p": 72.0, "mba_p": 68.0,
    },
    "Tier 3 Avg": {
        "ssc_p": 62.0, "hsc_p": 60.0, "degree_p": 62.0,
        "etest_p": 60.0, "mba_p": 58.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, center: float, steepness: float = 0.15) -> float:
    """Smooth sigmoid curve centred at *center*.

    Returns a value in (0, 1).  *steepness* controls how rapidly the
    transition occurs around *center*; higher values → sharper.
    """
    z = steepness * (x - center)
    # Clamp to avoid math overflow
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_profile_strength_score(profile: Dict[str, Any]) -> float:
    """Compute a weighted composite profile-strength score (0-100).

    The score blends academic percentages, work-experience presence, and
    specialisation track (Mkt&Fin receives a bonus).

    Args:
        profile: Student profile dictionary containing at least the keys
                 ``ssc_p``, ``hsc_p``, ``degree_p``, ``etest_p``, ``mba_p``,
                 ``workex``, and ``specialisation``.

    Returns:
        Float in [0, 100] representing overall profile strength.
    """
    score = 0.0

    # Academic numeric features (already 0-100 scale).
    for feat in ("ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"):
        val = float(profile.get(feat, 0.0))
        score += _clamp(val) * _PROFILE_WEIGHTS[feat]

    # Work experience: binary → 0 or 100.
    workex_val = 100.0 if str(profile.get("workex", "No")).strip().lower() == "yes" else 0.0
    score += workex_val * _PROFILE_WEIGHTS["workex"]

    # Specialisation bonus: Mkt&Fin historically correlates with higher packages.
    spec = str(profile.get("specialisation", "")).strip()
    spec_val = 100.0 if spec == "Mkt&Fin" else 50.0
    score += spec_val * _PROFILE_WEIGHTS["specialisation"]

    return round(_clamp(score), 2)


def predict_tier_probabilities(
    profile: Dict[str, Any],
    placement_prob: float,
) -> Dict[str, Any]:
    """Estimate tier-wise placement likelihood using sigmoid-smoothed scoring.

    Combines the ML placement probability with a heuristic profile-strength
    score to produce smooth probability estimates for three company tiers.

    Args:
        profile: Student profile dictionary.
        placement_prob: ML-model placement probability (0.0–1.0).

    Returns:
        Dictionary with ``tier1``, ``tier2``, ``tier3`` sub-dicts (each
        containing *probability*, *label*, *salary_range*, *companies*),
        plus ``recommended_tier`` (highest tier ≥ 50 %) and
        ``profile_strength``.
    """
    strength = compute_profile_strength_score(profile)

    result: Dict[str, Any] = {}
    best_tier: str | None = None

    for tier_key in ("tier1", "tier2", "tier3"):
        defn = TIER_DEFINITIONS[tier_key]

        # Two-axis sigmoid: one on profile strength, one on placement prob.
        s_strength = _sigmoid(strength, defn["strength_center"], steepness=0.12)
        s_prob = _sigmoid(placement_prob * 100, defn["prob_center"] * 100, steepness=0.08)

        # Geometric-mean gives "AND-like" behaviour – both must be high.
        tier_prob = round(math.sqrt(s_strength * s_prob), 4)
        tier_prob = _clamp(tier_prob, 0.0, 1.0)

        result[tier_key] = {
            "probability": tier_prob,
            "label": defn["label"],
            "salary_range": defn["salary_range"],
            "companies": defn["companies"],
        }

        # Track the highest tier with > 50 % probability.
        if tier_prob > 0.50:
            if best_tier is None:  # tier1 checked first → highest priority
                best_tier = tier_key

    result["recommended_tier"] = best_tier or "Below Tier 3"
    result["profile_strength"] = strength

    return result


def get_tier_gap_analysis(
    profile: Dict[str, Any],
    current_tiers: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Identify improvement areas for tiers the student hasn't yet reached.

    For each tier where the student's probability is < 50 %, this function
    computes what changes to the profile would close the gap.

    Args:
        profile: Student profile dictionary.
        current_tiers: Output of :func:`predict_tier_probabilities`.

    Returns:
        List of dicts, each with *tier*, *current_prob*, *gap_description*,
        and *required_improvements* (a list of specific suggestions).
    """
    gaps: List[Dict[str, Any]] = []
    strength = current_tiers.get("profile_strength", compute_profile_strength_score(profile))

    for tier_key in ("tier1", "tier2", "tier3"):
        tier_info = current_tiers[tier_key]
        prob = tier_info["probability"]
        if prob >= 0.50:
            continue  # Student already qualifies

        defn = TIER_DEFINITIONS[tier_key]
        improvements: List[str] = []

        # Check strength gap
        strength_target = defn["strength_center"]
        if strength < strength_target:
            delta = round(strength_target - strength, 1)
            improvements.append(
                f"Increase overall profile strength by ~{delta} points "
                f"(current: {strength:.1f}, target: ≥{strength_target:.0f})."
            )

            # Drill down into specific academic features
            benchmarks = TIER_BENCHMARKS.get(f"Tier {tier_key[-1]} Avg", {})
            for feat in NUMERIC_FEATURES:
                current_val = float(profile.get(feat, 0.0))
                bench_val = benchmarks.get(feat, 70.0)
                if current_val < bench_val:
                    improvements.append(
                        f"Raise {feat} from {current_val:.1f}% to ≥{bench_val:.0f}%."
                    )

        # Work experience suggestion
        if str(profile.get("workex", "No")).strip().lower() != "yes":
            improvements.append(
                "Gain verified work experience or a structured internship "
                "to boost profile competitiveness."
            )

        # Specialisation suggestion for Tier 1
        if tier_key == "tier1" and str(profile.get("specialisation", "")) != "Mkt&Fin":
            improvements.append(
                "Consider Mkt&Fin specialisation, which is historically "
                "associated with higher-paying placements."
            )

        if not improvements:
            improvements.append(
                "Continue improving overall academics and aptitude scores."
            )

        gaps.append({
            "tier": tier_info["label"],
            "tier_key": tier_key,
            "current_prob": round(prob, 4),
            "target_prob": 0.50,
            "gap_description": (
                f"Currently at {prob:.0%} probability for {tier_info['label']}. "
                f"Need ≥ 50 % to be competitive."
            ),
            "required_improvements": improvements,
        })

    return gaps


def generate_tier_comparison_data(
    profile: Dict[str, Any],
    dataset_benchmarks: Dict[str, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame comparing the student to tier-average benchmark profiles.

    The result is suitable for Plotly radar or grouped-bar charts.

    Args:
        profile: Student profile dictionary.
        dataset_benchmarks: Optional custom benchmarks.  Falls back to the
                            built-in ``TIER_BENCHMARKS`` if not supplied.

    Returns:
        Long-format ``pd.DataFrame`` with columns
        ``['Category', 'Metric', 'Value']``.
    """
    benchmarks = dataset_benchmarks or TIER_BENCHMARKS

    rows: List[Dict[str, Any]] = []

    # Student row
    for feat in NUMERIC_FEATURES:
        rows.append({
            "Category": "You",
            "Metric": feat,
            "Value": float(profile.get(feat, 0.0)),
        })

    # Benchmark rows
    for cat_label, bench_profile in benchmarks.items():
        for feat in NUMERIC_FEATURES:
            rows.append({
                "Category": cat_label,
                "Metric": feat,
                "Value": float(bench_profile.get(feat, 0.0)),
            })

    return pd.DataFrame(rows)

