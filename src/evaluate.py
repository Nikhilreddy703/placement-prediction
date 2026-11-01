"""Plotting helpers used by eda.py and train.py. Figures are saved to reports/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_balance(y, path):
    counts = y.map({0: "Not placed", 1: "Placed"}).value_counts()
    total = len(y)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(counts.index, counts.values, color=["#1F3A5F", "#3B82F6"], edgecolor="#1E293B", width=0.55)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + (total * 0.015), f"{h} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Target Class Distribution (Placed vs Not Placed)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_numeric_by_status(X, y, path):
    num_cols = X.select_dtypes(include="number").columns
    n = len(num_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows))
    axes = np.array(axes).reshape(-1)
    label = y.map({0: "Not placed", 1: "Placed"})
    for ax, c in zip(axes, num_cols):
        sns.boxplot(x=label, y=X[c], ax=ax, palette=["#1F3A5F", "#7FA7D1"], hue=label, legend=False)
        ax.set_title(c)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation(X, y, path):
    data = X.select_dtypes(include="number").copy()
    data["placed"] = y
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not placed", "Placed"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(results))
    ax.bar(x - 0.2, results["Accuracy"], 0.4, label="Test accuracy", color="#1F3A5F")
    ax.bar(x + 0.2, results["CV F1"], 0.4, label="CV F1", color="#7FA7D1")
    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def get_feature_importance(pipeline):
    """Extract feature importance or coefficients into a sorted DataFrame."""
    pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["clf"]
    names = pre.get_feature_names_out()
    cleaned_names = [n.replace("num__", "").replace("cat__", "") for n in names]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        metric = "Importance"
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
        metric = "Abs_Coefficient"
    else:
        return None

    import pandas as pd
    df_imp = pd.DataFrame({"Feature": cleaned_names, metric: values})
    return df_imp.sort_values(by=metric, ascending=False).reset_index(drop=True)


def plot_feature_importance(pipeline, path, top=10):
    """Works for tree models (feature_importances_) and linear models (coef_)."""
    df_imp = get_feature_importance(pipeline)
    if df_imp is None:
        return False
    
    top_df = df_imp.head(top).iloc[::-1]  # reverse for horizontal bar chart
    val_col = top_df.columns[1]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.barh(top_df["Feature"], top_df[val_col], color="#1F3A5F", edgecolor="#7FA7D1", alpha=0.9)
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.02, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9, color="#1F3A5F")
    ax.set_title(f"Top {top} Most Influential Features on Placement", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Relative Importance / Impact")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True

