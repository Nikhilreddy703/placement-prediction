"""Inference engine for individual and batch student placement predictions."""
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from config import (
    MODEL_FILE,
    THRESHOLD_MODERATE_RISK,
)
from src.data import validate_student_record
from src.explain import (
    categorize_risk,
    generate_coaching_recommendations,
)


def load_inference_pipeline(model_path: Optional[Union[str, Path]] = None):
    """Load the trained and serialized sklearn pipeline."""
    path = Path(model_path or MODEL_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Please run 'python src/train.py' first."
        )
    return joblib.load(path)


def predict_student_placement(
    profile: Union[Dict[str, Any], pd.DataFrame],
    pipeline: Optional[Any] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Predict placement outcome, probability, risk level, and recommendations.

    Args:
        profile: Dictionary or single-row DataFrame containing student details.
        pipeline: Pre-loaded pipeline, or None to load default MODEL_FILE.
        threshold: Decision threshold for classification (default 0.50).

    Returns:
        Dictionary with keys:
          - 'label': 1 (Placed) or 0 (Not Placed)
          - 'probability': float (0.0 to 1.0)
          - 'risk_level': 'Low Risk', 'Moderate Risk', or 'High Risk'
          - 'status_text': Human-readable summary
          - 'recommendations': List of string action items
    """
    model = pipeline if pipeline is not None else load_inference_pipeline()
    df_clean = validate_student_record(profile)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df_clean)[0][1])
    else:
        prob = 0.85 if int(model.predict(df_clean)[0]) == 1 else 0.35

    label = 1 if prob >= threshold else 0
    risk_level, status_text, color = categorize_risk(prob)
    recs = generate_coaching_recommendations(profile if isinstance(profile, dict) else profile.iloc[0].to_dict(), prob, model)

    return {
        "label": label,
        "probability": round(prob, 4),
        "threshold": threshold,
        "risk_level": risk_level,
        "status_text": status_text,
        "color": color,
        "recommendations": recs,
    }


def batch_predict_students(
    df: pd.DataFrame,
    pipeline: Optional[Any] = None,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Run inference over a DataFrame of students.

    Returns the DataFrame augmented with:
      - 'Predicted_Status' ('Placed' / 'Not Placed')
      - 'Placement_Probability' (0.0 - 1.0)
      - 'Risk_Level' ('Low Risk' / 'Moderate Risk' / 'High Risk')
    """
    model = pipeline if pipeline is not None else load_inference_pipeline()
    df_clean = validate_student_record(df)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_clean)[:, 1]
    else:
        probs = [0.85 if p == 1 else 0.35 for p in model.predict(df_clean)]

    result = df.copy()
    result["Placement_Probability"] = [round(float(p), 4) for p in probs]
    result["Predicted_Status"] = [
        "Placed" if p >= threshold else "Not Placed" for p in probs
    ]
    result["Risk_Level"] = [
        categorize_risk(float(p))[0] for p in probs
    ]
    return result


if __name__ == "__main__":
    from config import PRESETS

    sample = PRESETS["🌟 High-Scoring All-Rounder"]
    print("=" * 60)
    print("          STUDENT PLACEMENT PREDICTION ENGINE")
    print("=" * 60)
    out = predict_student_placement(sample)
    print(f"Outcome         : {out['status_text']}")
    print(f"Probability     : {out['probability']:.1%}")
    print(f"Risk Category   : {out['risk_level']}")
    print("\nActionable Recommendations:")
    for rec in out["recommendations"]:
        clean_rec = rec.encode("ascii", errors="replace").decode("ascii")
        print(f"  - {clean_rec}")
    print("=" * 60)
