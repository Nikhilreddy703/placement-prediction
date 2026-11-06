"""What-If Counterfactual Engine for placement prediction.

Finds the minimum actionable changes to *improvable* features (etest_p,
mba_p, workex) that push a student's predicted placement probability above
a specified target.  Also provides multi-target scenario generation and
per-feature sensitivity analysis.

Note: Historical features (ssc_p, hsc_p, degree_p) are treated as
immutable — they cannot be changed retroactively.
"""
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import FEATURE_CONSTRAINTS
from src.data import validate_student_record

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only features a student can still improve
IMPROVABLE_NUMERIC = ["etest_p", "mba_p"]
IMPROVABLE_CATEGORICAL = ["workex"]  # flip from "No" to "Yes"
IMMUTABLE_FEATURES = ["ssc_p", "hsc_p", "degree_p"]

# Step sizes for the grid search
_NUMERIC_STEP = 1.0
_MAX_SCORE = 100.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_probability(pipeline: Any, profile: Dict[str, Any]) -> float:
    """Return placement probability for a profile dict."""
    df = validate_student_record(profile)
    if hasattr(pipeline, "predict_proba"):
        return float(pipeline.predict_proba(df)[0][1])
    return 0.85 if int(pipeline.predict(df)[0]) == 1 else 0.35


def _effort_score(changes: List[Dict[str, Any]]) -> float:
    """Compute a simple effort heuristic (lower = easier).

    Numeric change contributes proportionally (each percentage point = 1).
    Flipping workex contributes a fixed 10 (roughly equivalent to gaining
    a 10-percentage-point improvement elsewhere).
    """
    total = 0.0
    for ch in changes:
        if ch["feature"] == "workex":
            total += 10.0
        else:
            total += abs(float(ch["change"]))
    return round(total, 2)


def _build_narrative(
    changes: List[Dict[str, Any]],
    current_prob: float,
    achieved_prob: float,
) -> str:
    """Build a plain-English narrative describing the counterfactual path."""
    if not changes:
        return (
            f"Your current placement probability is already "
            f"{current_prob:.0%} — no changes are needed."
        )

    parts: List[str] = []
    for ch in changes:
        feat = ch["feature"]
        if feat == "workex":
            parts.append("gain verified work experience / internship")
        elif feat == "etest_p":
            parts.append(
                f"improve your Employability Test score from "
                f"{ch['current_value']}% to {ch['suggested_value']}% "
                f"(+{ch['change']} points)"
            )
        elif feat == "mba_p":
            parts.append(
                f"raise your MBA percentage from "
                f"{ch['current_value']}% to {ch['suggested_value']}% "
                f"(+{ch['change']} points)"
            )
        else:
            parts.append(
                f"change {feat} from {ch['current_value']} "
                f"to {ch['suggested_value']}"
            )

    narrative = (
        f"To move from {current_prob:.0%} to {achieved_prob:.0%} placement "
        f"probability, the most efficient path is to: "
        + "; ".join(parts)
        + "."
    )
    return narrative


# ---------------------------------------------------------------------------
# 1. find_minimum_changes
# ---------------------------------------------------------------------------

def find_minimum_changes(
    pipeline: Any,
    profile: Dict[str, Any],
    target_prob: float = 0.80,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Find the smallest set of improvable-feature changes that achieve a target.

    Uses a grid search over etest_p, mba_p increments and workex flip to
    find the combination with the lowest total effort that pushes the
    predicted placement probability above ``target_prob``.

    Args:
        pipeline: Trained sklearn Pipeline with ``predict_proba``.
        profile: Student profile dictionary.
        target_prob: Desired minimum placement probability (0–1).
        threshold: Classification decision threshold (informational only).

    Returns:
        Dictionary with keys:
        - ``achievable`` (bool): Whether the target can be reached.
        - ``current_probability`` (float): Baseline probability.
        - ``target_probability`` (float): The requested target.
        - ``achieved_probability`` (float): Best probability found.
        - ``changes`` (list[dict]): Feature changes in the optimal path.
        - ``effort_score`` (float): Lower = easier path.
        - ``narrative`` (str): Plain-English explanation.
    """
    current_prob = _get_probability(pipeline, profile)

    # Already above target — no changes needed
    if current_prob >= target_prob:
        return {
            "achievable": True,
            "current_probability": round(current_prob, 4),
            "target_probability": target_prob,
            "achieved_probability": round(current_prob, 4),
            "changes": [],
            "effort_score": 0.0,
            "narrative": _build_narrative([], current_prob, current_prob),
        }

    # --- Build candidate increments ---
    cur_etest = float(profile.get("etest_p", 50.0))
    cur_mba = float(profile.get("mba_p", 50.0))
    cur_workex = str(profile.get("workex", "No")).strip()

    # Numeric increments (0 = no change, then 1, 2, ... up to max)
    etest_deltas = [0.0] + [
        d for d in np.arange(_NUMERIC_STEP, _MAX_SCORE - cur_etest + _NUMERIC_STEP, _NUMERIC_STEP)
        if cur_etest + d <= _MAX_SCORE
    ]
    mba_deltas = [0.0] + [
        d for d in np.arange(_NUMERIC_STEP, _MAX_SCORE - cur_mba + _NUMERIC_STEP, _NUMERIC_STEP)
        if cur_mba + d <= _MAX_SCORE
    ]
    workex_options = [cur_workex]
    if cur_workex.lower() == "no":
        workex_options.append("Yes")

    best_result: Optional[Dict[str, Any]] = None
    best_effort = float("inf")

    # To keep the search tractable, limit granularity for large ranges:
    # Use step-5 for coarse pass, then refine around the winner.
    coarse_etest = [0.0] + [d for d in etest_deltas if d % 5 == 0 or d <= 5]
    coarse_mba = [0.0] + [d for d in mba_deltas if d % 5 == 0 or d <= 5]

    def _evaluate(e_delta: float, m_delta: float, wex: str):
        """Evaluate a single candidate and return (effort, prob, changes) or None."""
        changes: List[Dict[str, Any]] = []
        trial = dict(profile)

        if e_delta > 0:
            trial["etest_p"] = round(cur_etest + e_delta, 1)
            changes.append({
                "feature": "etest_p",
                "current_value": cur_etest,
                "suggested_value": round(cur_etest + e_delta, 1),
                "change": round(e_delta, 1),
            })
        if m_delta > 0:
            trial["mba_p"] = round(cur_mba + m_delta, 1)
            changes.append({
                "feature": "mba_p",
                "current_value": cur_mba,
                "suggested_value": round(cur_mba + m_delta, 1),
                "change": round(m_delta, 1),
            })
        if wex.lower() == "yes" and cur_workex.lower() == "no":
            trial["workex"] = "Yes"
            changes.append({
                "feature": "workex",
                "current_value": cur_workex,
                "suggested_value": "Yes",
                "change": "No → Yes",
            })

        if not changes:
            return None

        prob = _get_probability(pipeline, trial)
        if prob >= target_prob:
            effort = _effort_score(changes)
            return effort, prob, changes
        return None

    # --- Coarse search ---
    for wex in workex_options:
        for e_d in coarse_etest:
            for m_d in coarse_mba:
                result = _evaluate(e_d, m_d, wex)
                if result is not None:
                    effort, prob, changes = result
                    if effort < best_effort:
                        best_effort = effort
                        best_result = {
                            "prob": prob,
                            "changes": changes,
                            "effort": effort,
                        }

    # --- Fine-grained refinement around best coarse hit ---
    if best_result is not None:
        # Extract the winning deltas
        best_e = 0.0
        best_m = 0.0
        for ch in best_result["changes"]:
            if ch["feature"] == "etest_p":
                best_e = float(ch["change"])
            elif ch["feature"] == "mba_p":
                best_m = float(ch["change"])
        best_wex = any(ch["feature"] == "workex" for ch in best_result["changes"])
        wex_val = "Yes" if best_wex else cur_workex

        fine_etest = [
            d for d in np.arange(max(0, best_e - 5), best_e + 1, _NUMERIC_STEP)
            if cur_etest + d <= _MAX_SCORE and d >= 0
        ]
        fine_mba = [
            d for d in np.arange(max(0, best_m - 5), best_m + 1, _NUMERIC_STEP)
            if cur_mba + d <= _MAX_SCORE and d >= 0
        ]

        for e_d in fine_etest:
            for m_d in fine_mba:
                result = _evaluate(e_d, m_d, wex_val)
                if result is not None:
                    effort, prob, changes = result
                    if effort < best_effort:
                        best_effort = effort
                        best_result = {
                            "prob": prob,
                            "changes": changes,
                            "effort": effort,
                        }

    # --- Assemble output ---
    if best_result is None:
        # Could not reach target with any combination
        # Find the maximum reachable probability
        max_trial = dict(profile)
        max_trial["etest_p"] = _MAX_SCORE
        max_trial["mba_p"] = _MAX_SCORE
        if cur_workex.lower() == "no":
            max_trial["workex"] = "Yes"
        max_prob = _get_probability(pipeline, max_trial)

        return {
            "achievable": False,
            "current_probability": round(current_prob, 4),
            "target_probability": target_prob,
            "achieved_probability": round(max_prob, 4),
            "changes": [],
            "effort_score": float("inf"),
            "narrative": (
                f"Even with maximum improvements to employability test "
                f"(100%), MBA percentage (100%), and gaining work experience, "
                f"the predicted probability reaches {max_prob:.0%}, which is "
                f"below the {target_prob:.0%} target. Historical academic "
                f"factors (SSC, HSC, Degree) may be limiting the prediction."
            ),
        }

    return {
        "achievable": True,
        "current_probability": round(current_prob, 4),
        "target_probability": target_prob,
        "achieved_probability": round(best_result["prob"], 4),
        "changes": best_result["changes"],
        "effort_score": best_result["effort"],
        "narrative": _build_narrative(
            best_result["changes"], current_prob, best_result["prob"]
        ),
    }


# ---------------------------------------------------------------------------
# 2. generate_counterfactual_scenarios
# ---------------------------------------------------------------------------

def generate_counterfactual_scenarios(
    pipeline: Any,
    profile: Dict[str, Any],
    targets: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Generate counterfactual scenarios for multiple target probabilities.

    Args:
        pipeline: Trained sklearn Pipeline.
        profile: Student profile dictionary.
        targets: List of target probabilities (default: [0.60, 0.70, 0.80, 0.90]).

    Returns:
        List of counterfactual result dicts (same schema as
        ``find_minimum_changes``), one per target.
    """
    if targets is None:
        targets = [0.60, 0.70, 0.80, 0.90]

    scenarios: List[Dict[str, Any]] = []
    for t in sorted(targets):
        result = find_minimum_changes(pipeline, profile, target_prob=t)
        scenarios.append(result)

    return scenarios


# ---------------------------------------------------------------------------
# 3. compute_feature_sensitivity
# ---------------------------------------------------------------------------

def compute_feature_sensitivity(
    pipeline: Any,
    profile: Dict[str, Any],
) -> pd.DataFrame:
    """Compute marginal probability gain per unit change for improvable features.

    For each improvable numeric feature (etest_p, mba_p), evaluates
    the placement probability at +5, +10, and +15 increments and computes
    the average sensitivity per percentage-point change.

    For workex, reports the probability delta of flipping from No to Yes
    (or notes it is already Yes).

    Args:
        pipeline: Trained sklearn Pipeline.
        profile: Student profile dictionary.

    Returns:
        DataFrame with columns:
        [Feature, Current_Value, +5_Change, +10_Change, +15_Change,
         Sensitivity_Per_Unit].
    """
    base_prob = _get_probability(pipeline, profile)
    deltas = [5, 10, 15]
    rows: List[Dict[str, Any]] = []

    for feat in IMPROVABLE_NUMERIC:
        cur_val = float(profile.get(feat, 50.0))
        change_probs: Dict[str, Optional[float]] = {}

        for d in deltas:
            new_val = cur_val + d
            if new_val > _MAX_SCORE:
                change_probs[f"+{d}_Change"] = None
            else:
                trial = dict(profile)
                trial[feat] = round(new_val, 1)
                prob = _get_probability(pipeline, trial)
                change_probs[f"+{d}_Change"] = round(prob - base_prob, 4)

        # Sensitivity: average probability gain per unit across valid deltas
        valid_gains = [
            change_probs[f"+{d}_Change"] / d
            for d in deltas
            if change_probs[f"+{d}_Change"] is not None
        ]
        sensitivity = round(np.mean(valid_gains), 5) if valid_gains else 0.0

        rows.append({
            "Feature": feat,
            "Current_Value": cur_val,
            **change_probs,
            "Sensitivity_Per_Unit": sensitivity,
        })

    # Workex sensitivity
    cur_workex = str(profile.get("workex", "No")).strip()
    if cur_workex.lower() == "no":
        trial = dict(profile)
        trial["workex"] = "Yes"
        prob = _get_probability(pipeline, trial)
        workex_delta = round(prob - base_prob, 4)
        rows.append({
            "Feature": "workex",
            "Current_Value": cur_workex,
            "+5_Change": workex_delta,
            "+10_Change": workex_delta,
            "+15_Change": workex_delta,
            "Sensitivity_Per_Unit": round(workex_delta / 10.0, 5),  # normalized to effort=10
        })
    else:
        rows.append({
            "Feature": "workex",
            "Current_Value": cur_workex,
            "+5_Change": 0.0,
            "+10_Change": 0.0,
            "+15_Change": 0.0,
            "Sensitivity_Per_Unit": 0.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import PRESETS
    from src.predict import load_inference_pipeline

    pipeline = load_inference_pipeline()
    sample = PRESETS["⚠️ At-Risk Student (Needs Support)"]

    print("=" * 60)
    print("      COUNTERFACTUAL ENGINE — SMOKE TEST")
    print("=" * 60)

    result = find_minimum_changes(pipeline, sample, target_prob=0.70)
    print(f"\nAchievable : {result['achievable']}")
    print(f"Current    : {result['current_probability']:.2%}")
    print(f"Achieved   : {result['achieved_probability']:.2%}")
    print(f"Effort     : {result['effort_score']}")
    print(f"Narrative  : {result['narrative']}")
    for ch in result["changes"]:
        print(f"  → {ch['feature']}: {ch['current_value']} → {ch['suggested_value']}")

    print("\n--- Sensitivity Analysis ---")
    sens = compute_feature_sensitivity(pipeline, sample)
    print(sens.to_string(index=False))

    print("\n--- Multi-Target Scenarios ---")
    scenarios = generate_counterfactual_scenarios(pipeline, sample)
    for sc in scenarios:
        status = "✓" if sc["achievable"] else "✗"
        print(
            f"  {status} Target {sc['target_probability']:.0%} → "
            f"Achieved {sc['achieved_probability']:.2%} | "
            f"Effort {sc['effort_score']}"
        )
    print("=" * 60)
