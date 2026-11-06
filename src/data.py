"""Data loading, cleaning, validation, and leakage removal."""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import (
    ALL_FEATURE_COLS,
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    DROP_COLS,
    FEATURE_CONSTRAINTS,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    TARGET_COL,
)


class DataValidationError(ValueError):
    """Custom exception raised when student input data fails validation."""
    pass


def validate_student_record(data: Union[Dict, pd.DataFrame, np.ndarray, List]) -> pd.DataFrame:
    """Validate a single student profile, DataFrame, or array against expected schema and bounds.

    Returns sanitized single-row or multi-row DataFrame ready for pipeline input with exact column names and dtypes.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        arr = data if data.ndim == 2 else data.reshape(1, -1)
        if arr.shape[1] != len(ALL_FEATURE_COLS):
            raise DataValidationError(
                f"NumPy array has {arr.shape[1]} columns, expected {len(ALL_FEATURE_COLS)} ({ALL_FEATURE_COLS})"
            )
        df = pd.DataFrame(arr, columns=ALL_FEATURE_COLS)
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            arr = np.array(data)
            arr = arr if arr.ndim == 2 else arr.reshape(1, -1)
            if arr.shape[1] != len(ALL_FEATURE_COLS):
                raise DataValidationError(
                    f"List input has {arr.shape[1]} columns, expected {len(ALL_FEATURE_COLS)} ({ALL_FEATURE_COLS})"
                )
            df = pd.DataFrame(arr, columns=ALL_FEATURE_COLS)
    else:
        raise DataValidationError(f"Expected dict, DataFrame, or array-like, received {type(data)}")

    # Check for missing required feature columns
    missing = [col for col in ALL_FEATURE_COLS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns in input: {missing}")

    # Validate numeric features and enforce bounds
    for col in NUMERIC_FEATURES:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            raise DataValidationError(f"Column '{col}' must contain valid numbers: {e}")

        if df[col].isna().any():
            raise DataValidationError(f"Column '{col}' contains NaN or unparseable values.")

        min_val = FEATURE_CONSTRAINTS[col]["min"]
        max_val = FEATURE_CONSTRAINTS[col]["max"]
        if (df[col] < min_val).any() or (df[col] > max_val).any():
            raise DataValidationError(
                f"Column '{col}' values must be between {min_val} and {max_val}."
            )

    # Validate categorical features
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
        allowed = FEATURE_CONSTRAINTS[col]["options"]
        # Allow case-insensitive matching if possible
        allowed_map = {opt.lower(): opt for opt in allowed}
        invalid_mask = ~df[col].str.lower().isin(allowed_map.keys())
        if invalid_mask.any():
            invalid_vals = df.loc[invalid_mask, col].unique().tolist()
            raise DataValidationError(
                f"Invalid value(s) {invalid_vals} for '{col}'. Allowed options: {allowed}"
            )
        # Normalize casing to expected standard
        df[col] = df[col].str.lower().map(allowed_map)

    return df[ALL_FEATURE_COLS]


def load_dataset(
    file_path: Optional[Union[str, Path]] = None,
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Ingest dataset, drop target leakage columns, clean records, and split X and y.

    Args:
        file_path: Path to raw CSV dataset. Defaults to config.DATA_RAW_PATH.
        save_processed: Whether to persist the cleaned tabular dataset to data/processed.

    Returns:
        X (DataFrame): Cleaned predictor features without leakage columns.
        y (Series): Binary target (1 = Placed, 0 = Not Placed).
    """
    path = Path(file_path or DATA_RAW_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # 1. Strip column whitespace
    df.columns = df.columns.str.strip()

    # 2. Drop leakage columns and identifiers
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Verify target presence
    if TARGET_COL not in df.columns:
        raise DataValidationError(f"Target column '{TARGET_COL}' not found in {path}")

    # 4. Clean duplicates and rows missing target or key attributes
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 5. Encode target robustly to binary (1 = Placed, 0 = Not Placed)
    target_series = df[TARGET_COL]
    if pd.api.types.is_numeric_dtype(target_series):
        y = target_series.astype(int)
    else:
        y = (
            target_series.astype(str)
            .str.strip()
            .str.lower()
            == POSITIVE_LABEL.lower()
        ).astype(int)

    X = df.drop(columns=[TARGET_COL])

    # Validate and filter to supported features
    X = validate_student_record(X)

    # 6. Optional: Persist cleaned dataset
    if save_processed:
        try:
            DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
            cleaned_export = X.copy()
            cleaned_export["status_placed"] = y
            cleaned_export.to_csv(DATA_PROCESSED_PATH, index=False)
        except Exception:
            pass

    return X, y

