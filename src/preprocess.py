"""Loading, cleaning and preprocessing for the placement dataset."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "status"                 # target column name in your CSV
POSITIVE_LABEL = "placed"         # value meaning "placed" (comparison is case-insensitive)
# ID column and salary. Salary exists only for placed students, so keeping it would leak the answer.
DROP_COLS = ["sl_no", "salary"]


def load_data(path, save_processed=True):
    """Read the CSV, drop leakage/ID columns, clean, and return X (features) and y (0/1)."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # Handle numeric or text target robustly
    target_col = df[TARGET]
    if pd.api.types.is_numeric_dtype(target_col):
        y = target_col.astype(int)
    else:
        y = (target_col.astype(str).str.strip().str.lower() == POSITIVE_LABEL.lower()).astype(int)

    X = df.drop(columns=TARGET)

    if save_processed:
        try:
            from pathlib import Path
            root = Path(path).resolve().parent.parent.parent
            processed_dir = root / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            cleaned_df = X.copy()
            cleaned_df["status_placed"] = y
            cleaned_df.to_csv(processed_dir / "placement_cleaned.csv", index=False)
        except Exception:
            pass

    return X, y


def build_preprocessor(X):
    """Scale numeric columns and one-hot encode text columns."""
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

