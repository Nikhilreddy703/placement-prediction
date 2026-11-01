"""Feature transformation pipeline construction and schema utilities."""
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build an sklearn ColumnTransformer that standardizes numeric columns
    and one-hot encodes categorical columns.

    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.

    Returns:
        Configured ColumnTransformer ready to fit on feature matrices.
    """
    num_cols = numeric_cols if numeric_cols is not None else NUMERIC_FEATURES
    cat_cols = categorical_cols if categorical_cols is not None else CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract clean feature names generated after transformation."""
    try:
        raw_names = preprocessor.get_feature_names_out()
        clean_names = [
            name.replace("num__", "").replace("cat__", "") for name in raw_names
        ]
        return clean_names
    except Exception:
        # Fallback if preprocessor hasn't been fitted
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES

