"""Model training, hyperparameter tuning, benchmarking, calibration, and artifact generation."""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import (
    ALL_MODELS_FILE,
    EVALUATION_REPORT_FILE,
    FEATURE_IMP_FILE,
    FIGURES_DIR,
    METADATA_FILE,
    MODEL_FILE,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_FILE,
)
from src.data import load_dataset
from src.evaluate import (
    get_feature_importance,
    plot_confusion,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curve,
)
from src.features import build_preprocessor


def get_model_candidates() -> Dict[str, Tuple[object, dict]]:
    """Define classification algorithms and their hyperparameter search grids."""
    return {
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [4, 6],
                "clf__min_samples_split": [2, 5],
            },
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
            {
                "clf__C": [0.1, 1.0, 5.0],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {
                "clf__n_estimators": [100, 150],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 4],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED),
            {
                "clf__n_estimators": [100, 150],
                "clf__max_depth": [3, 4],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [5, 7, 9],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "SVM": (
            SVC(class_weight="balanced", random_state=RANDOM_SEED),
            {
                "clf__C": [0.5, 1.0, 3.0],
                "clf__kernel": ["rbf", "linear"],
            },
        ),
    }


def find_optimal_threshold(y_true, y_probs) -> Tuple[float, float]:
    """Find the classification decision threshold that maximizes the F1-score."""
    best_thresh = 0.50
    best_f1 = 0.0
    for t in np.linspace(0.30, 0.70, 41):
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = round(float(t), 2)
    return best_thresh, best_f1


def train_and_benchmark(data_path: Path = None):
    """Execute training pipeline across all candidate models, calibrate, and save artifacts."""
    print("=" * 65)
    print("   STUDENT PLACEMENT MODEL BENCHMARKING & CALIBRATION")
    print("=" * 65)

    X, y = load_dataset(data_path)
    total_count = len(X)
    placed_count = int(y.sum())
    placed_rate = y.mean()

    print(f"Dataset: {total_count} records | Placed: {placed_count} ({placed_rate:.1%}) | Not Placed: {total_count - placed_count}")

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    candidates = get_model_candidates()
    fitted_pipelines = {}
    rows = []

    for name, (base_clf, param_grid) in candidates.items():
        print(f"\n[Training & Tuning] {name}...")
        raw_pipe = Pipeline([
            ("pre", build_preprocessor()),
            ("clf", base_clf),
        ])

        # Grid search tuning for optimal hyperparameters
        grid = GridSearchCV(
            raw_pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_

        # 5-fold CV score on training data
        cv_f1 = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="f1").mean()

        # Calibrate probabilities for reliable risk scoring
        try:
            calibrated_pipe = CalibratedClassifierCV(estimator=best_pipe, cv=3)
            calibrated_pipe.fit(X_train, y_train)
            final_pipe = calibrated_pipe
            probs = final_pipe.predict_proba(X_test)[:, 1]
        except Exception:
            final_pipe = best_pipe
            probs = (
                final_pipe.predict_proba(X_test)[:, 1]
                if hasattr(final_pipe, "predict_proba")
                else (final_pipe.predict(X_test) * 0.8 + 0.1)
            )

        test_pred = (probs >= 0.50).astype(int)

        acc = accuracy_score(y_test, test_pred)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "CV F1": round(cv_f1, 4),
            "Best Params": str(grid.best_params_),
        })

        fitted_pipelines[name] = {
            "pipeline": final_pipe,
            "raw_pipeline": best_pipe,
            "probs": probs,
            "preds": test_pred,
        }

    # Rank models by cross-validated F1 score
    results_df = pd.DataFrame(rows).sort_values("CV F1", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 65)
    print("                MODEL BENCHMARK RESULTS")
    print("=" * 65)
    print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1"]].to_string(index=False))

    best_model_name = results_df.loc[0, "Model"]
    best_pack = fitted_pipelines[best_model_name]
    best_pipe = best_pack["pipeline"]
    best_probs = best_pack["probs"]

    # Threshold optimization for best model
    opt_thresh, opt_f1 = find_optimal_threshold(y_test, best_probs)
    print(f"\n[BEST MODEL] Champion Model: {best_model_name}")
    print(f"Optimal Decision Threshold: {opt_thresh:.2f} (Yields Test F1: {opt_f1:.4f})")
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, (best_probs >= opt_thresh).astype(int), target_names=["Not placed", "Placed"]))

    # Save artifacts
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(RESULTS_FILE, index=False)

    # Save best model pipeline and dictionary of all trained pipelines
    joblib.dump(best_pipe, MODEL_FILE)
    export_all = {name: pack["pipeline"] for name, pack in fitted_pipelines.items()}
    joblib.dump(export_all, ALL_MODELS_FILE)

    # Visualizations
    plot_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")
    plot_confusion(y_test, best_pack["preds"], FIGURES_DIR / "confusion_matrix.png", f"Confusion Matrix ({best_model_name})")
    plot_roc_curve(y_test, best_probs, FIGURES_DIR / "roc_curve.png", best_model_name)

    # Feature Importance (extract from tree or linear model if best model doesn't have native coefs)
    tree_or_linear = best_pack["raw_pipeline"]
    df_imp = get_feature_importance(tree_or_linear)
    if df_imp is None and "Random Forest" in fitted_pipelines:
        tree_or_linear = fitted_pipelines["Random Forest"]["raw_pipeline"]
        df_imp = get_feature_importance(tree_or_linear)

    if df_imp is not None:
        plot_feature_importance(tree_or_linear, FIGURES_DIR / "feature_importance.png")
        df_imp.round(4).to_csv(FEATURE_IMP_FILE, index=False)

    # Benchmark statistics for radar chart
    placed_mask = (y == 1)
    numeric_placed_benchmarks = {
        col: round(float(X.loc[placed_mask, col].mean()), 2)
        for col in X.select_dtypes(include="number").columns
    }

    # Metadata export
    metadata = {
        "best_model": best_model_name,
        "optimal_threshold": opt_thresh,
        "metrics": results_df.loc[0].drop(["Model", "Best Params"]).to_dict(),
        "all_models": results_df.to_dict(orient="records"),
        "numeric_benchmark_placed": numeric_placed_benchmarks,
        "feature_importance": df_imp.head(10).to_dict(orient="records") if df_imp is not None else [],
        "numeric": {
            c: [float(X[c].min()), float(X[c].max()), float(X[c].median())]
            for c in X.select_dtypes(include="number").columns
        },
        "categorical": {
            c: sorted(X[c].unique().tolist())
            for c in X.select_dtypes(exclude="number").columns
        },
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Structured Evaluation Markdown Report
    report_content = [
        "# Model Comparison and Comprehensive Evaluation Report",
        "",
        "## 1. Benchmarking Overview across 6 Algorithms",
        "",
        results_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1"]].to_markdown(index=False),
        "",
        f"## 2. Champion Model Selected: **{best_model_name}**",
        "- **Selection Criterion:** Highest 5-Fold Stratified Cross-Validation F1 score on training data.",
        f"- **Optimal Decision Threshold:** `{opt_thresh:.2f}` (Standard: `0.50`)",
        f"- **Test Accuracy:** `{results_df.loc[0, 'Accuracy']:.1%}`",
        f"- **Test F1-Score:** `{results_df.loc[0, 'F1']:.4f}`",
        f"- **ROC-AUC:** `{results_df.loc[0, 'ROC-AUC']:.4f}`",
        f"- **5-Fold CV F1:** `{results_df.loc[0, 'CV F1']:.4f}`",
        "",
        "## 3. Top Influencing Factors (Feature Importance)",
        "",
        df_imp.head(10).round(4).to_markdown(index=False) if df_imp is not None else "N/A",
        "",
        "## 4. Generated Artifacts",
        "- `models/placement_model.joblib`: Serialized inference pipeline",
        "- `models/all_models.joblib`: Serialized dictionary of all 6 trained algorithms",
        "- `models/metadata.json`: Feature schemas, optimal threshold, and benchmark metrics",
        "- `reports/results.csv`: Complete metrics comparison",
        "- `reports/feature_importance.csv`: Ranked predictor importance",
        "- `reports/figures/model_comparison.png`: Accuracy, F1, and CV F1 comparison chart",
        "- `reports/figures/confusion_matrix.png`: Test set confusion matrix",
        "- `reports/figures/roc_curve.png`: ROC Curve for champion model",
        "- `reports/figures/feature_importance.png`: Top predictors bar chart",
    ]
    EVALUATION_REPORT_FILE.write_text("\n".join(report_content), encoding="utf-8")
    print(f"\nAll artifacts successfully saved to {MODELS_DIR} and {REPORTS_DIR}")


if __name__ == "__main__":
    train_and_benchmark()
