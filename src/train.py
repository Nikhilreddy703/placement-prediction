"""Train and compare several models, then save the best pipeline.

Usage:  python src/train.py --data data/raw/placement.csv
Outputs: models/placement_model.joblib, models/metadata.json, reports/results.csv, reports/figures/*.png
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluate import get_feature_importance, plot_confusion, plot_feature_importance, plot_model_comparison
from preprocess import build_preprocessor, load_data

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "placement.csv"))
    args = ap.parse_args()

    X, y = load_data(args.data)
    print(f"Loaded {len(X)} rows, placed rate {100 * y.mean():.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, clf in get_models().items():
        pipe = Pipeline([("pre", build_preprocessor(X)), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()  # CV on training data only
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "CV F1": cv_f1,
        })
        fitted[name] = (pipe, pred)

    results = pd.DataFrame(rows).sort_values("CV F1", ascending=False).reset_index(drop=True)
    print("\n=== Model comparison (test set + 5-fold CV on training data) ===")
    print(results.round(3).to_string(index=False))

    best_name = results.loc[0, "Model"]  # chosen by CV F1, not by looking at the test set
    best_pipe, best_pred = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best_pred, target_names=["Not placed", "Placed"]))

    # Save outputs
    for d in ("models", "reports/figures"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(ROOT / "reports" / "results.csv", index=False)
    plot_confusion(y_test, best_pred, ROOT / "reports" / "figures" / "confusion_matrix.png", f"{best_name}")
    plot_model_comparison(results, ROOT / "reports" / "figures" / "model_comparison.png")

    # Feature importance calculation and export
    df_imp = get_feature_importance(best_pipe)
    if df_imp is None:
        # Best model has no native feature importance (e.g. KNN); extract from Random Forest
        df_imp = get_feature_importance(fitted["Random Forest"][0])
        plot_feature_importance(fitted["Random Forest"][0], ROOT / "reports" / "figures" / "feature_importance.png")
    else:
        plot_feature_importance(best_pipe, ROOT / "reports" / "figures" / "feature_importance.png")

    if df_imp is not None:
        df_imp.round(4).to_csv(ROOT / "reports" / "feature_importance.csv", index=False)

    joblib.dump(best_pipe, ROOT / "models" / "placement_model.joblib")
    meta = {
        "best_model": best_name,
        "metrics": results.loc[0].drop("Model").round(4).to_dict(),
        "all_models": results.round(4).to_dict(orient="records"),
        "numeric": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())]
                    for c in X.select_dtypes(include="number").columns},
        "categorical": {c: sorted(X[c].unique().tolist()) for c in X.select_dtypes(exclude="number").columns},
    }
    (ROOT / "models" / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Save Markdown Evaluation Report
    report_lines = [
        "# Model Comparison and Evaluation Report",
        "",
        "## 1. Performance Overview",
        "",
        results.round(4).to_markdown(index=False),
        "",
        f"## 2. Selected Best Model: **{best_name}**",
        "- **Selection Criterion:** Highest 5-Fold Stratified Cross-Validation F1-score on training set",
        f"- **Test Set Accuracy:** {results.loc[0, 'Accuracy']:.1%}",
        f"- **Test Set F1-Score:** {results.loc[0, 'F1']:.3f}",
        f"- **Cross-Validated F1:** {results.loc[0, 'CV F1']:.3f}",
        "",
        "## 3. Classification Report (Test Set)",
        "```",
        classification_report(y_test, best_pred, target_names=["Not placed", "Placed"]),
        "```",
        "",
        "## 4. Key Factors Influencing Placement",
        "",
        df_imp.head(10).round(4).to_markdown(index=False) if df_imp is not None else "N/A",
        "",
        "## 5. Artifacts Generated",
        "- `models/placement_model.joblib`: Serialized inference pipeline",
        "- `models/metadata.json`: Feature schemas and benchmark metrics",
        "- `reports/results.csv`: Complete metrics comparison",
        "- `reports/feature_importance.csv`: Ranked predictor importance",
        "- `reports/figures/confusion_matrix.png`: Test set confusion matrix",
        "- `reports/figures/model_comparison.png`: Accuracy vs CV F1 chart",
        "- `reports/figures/feature_importance.png`: Top predictors bar chart",
    ]
    (ROOT / "reports" / "model_evaluation_report.md").write_text("\n".join(report_lines))
    print("Saved model, metadata, results table, feature importance, and evaluation report.")



if __name__ == "__main__":
    main()
