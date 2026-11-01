"""Plotting and evaluation utilities for model benchmarking and reports."""
from pathlib import Path
from typing import Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_class_balance(y: pd.Series, path: Path) -> None:
    counts = y.map({0: "Not placed", 1: "Placed"}).value_counts()
    total = len(y)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#1F3A5F", "#3B82F6"],
        edgecolor="#1E293B",
        width=0.55,
    )
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (total * 0.015),
            f"{h} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Target Class Distribution (Placed vs Not Placed)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_numeric_by_status(X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    num_cols = X.select_dtypes(include="number").columns
    n = len(num_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows))
    axes = np.array(axes).reshape(-1)
    label = y.map({0: "Not placed", 1: "Placed"})
    for ax, c in zip(axes, num_cols):
        sns.boxplot(
            x=label,
            y=X[c],
            ax=ax,
            palette=["#1F3A5F", "#7FA7D1"],
            hue=label,
            legend=False,
        )
        ax.set_title(c, fontweight="bold")
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation(X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    data = X.select_dtypes(include="number").copy()
    data["placed"] = y
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True)
    ax.set_title("Academic Metrics Correlation Heatmap", fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path: Path, title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Not placed", "Placed"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, path: Path, model_name: str = "Model") -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", label="Random Chance (0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve - {model_name}", fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return roc_auc


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(results))
    w = 0.25
    ax.bar(x - w, results["Accuracy"], w, label="Test Accuracy", color="#1E3A8A")
    ax.bar(x, results["F1"], w, label="Test F1", color="#3B82F6")
    ax.bar(x + w, results["CV F1"], w, label="5-Fold CV F1", color="#93C5FD")
    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"], rotation=25, ha="right", fontweight="medium")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Model Benchmarking (Accuracy, F1, and 5-Fold Stratified CV F1)", fontweight="bold", pad=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def get_feature_importance(pipeline) -> Optional[pd.DataFrame]:
    """Extract feature importance or coefficients into a sorted DataFrame."""
    pre = pipeline.named_steps.get("pre")
    clf = pipeline.named_steps.get("clf")
    if pre is None or clf is None:
        return None

    # Handle calibrated classifiers
    if hasattr(clf, "estimator"):
        clf = clf.estimator

    names = pre.get_feature_names_out()
    cleaned_names = [n.replace("num__", "").replace("cat__", "") for n in names]

    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        metric = "Importance"
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
        metric = "Importance"
    else:
        return None

    df_imp = pd.DataFrame({"Feature": cleaned_names, metric: values})
    return df_imp.sort_values(by=metric, ascending=False).reset_index(drop=True)


def plot_feature_importance(pipeline, path: Path, top: int = 10) -> bool:
    df_imp = get_feature_importance(pipeline)
    if df_imp is None:
        return False

    top_df = df_imp.head(top).iloc[::-1]
    val_col = top_df.columns[1]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.barh(
        top_df["Feature"],
        top_df[val_col],
        color="#1E3A8A",
        edgecolor="#60A5FA",
        alpha=0.9,
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            fontsize=9,
            color="#1E293B",
        )
    ax.set_title(f"Top {top} Most Influential Features on Placement", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Relative Importance Weight")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
