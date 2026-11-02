"""
Standardized plotting utilities for classification, regression, and clustering.
Generates publication-quality figures with consistent styling.
"""

from typing import List, Optional, Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def set_plot_style() -> None:
    """Sets a clean, modern style for all output figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })


# ==========================================
# Classification Plots
# ==========================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """Plots a normalized or raw confusion matrix heatmap."""
    set_plot_style()
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontweight="bold", pad=12)
    plt.ylabel("True Label", fontweight="semibold")
    plt.xlabel("Predicted Label", fontweight="semibold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str = "Multiclass One-vs-Rest ROC Curves",
    save_path: Optional[str] = None
) -> None:
    """Plots One-vs-Rest ROC curve for each class alongside micro and macro averages."""
    set_plot_style()
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"Micro-average (AUC = {roc_auc['micro']:.3f})",
             color="deeppink", linestyle=":", linewidth=2.5)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f"Macro-average (AUC = {roc_auc['macro']:.3f})",
             color="navy", linestyle=":", linewidth=2.5)
             
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{class_names[i][:15]} (AUC = {roc_auc[i]:.3f})")
                 
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="semibold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="semibold")
    plt.title(title, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


# ==========================================
# Regression Plots
# ==========================================

def plot_regression_actual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted Traffic Volume",
    save_path: Optional[str] = None
) -> None:
    """Scatter plot of actual vs predicted regression values with 45-degree reference line."""
    set_plot_style()
    plt.figure(figsize=(8, 7))
    plt.scatter(y_true, y_pred, alpha=0.25, color="#1f77b4", edgecolors="none", s=15)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    plt.xlabel("Actual Traffic Volume", fontweight="semibold")
    plt.ylabel("Predicted Traffic Volume", fontweight="semibold")
    plt.title(title, fontweight="bold", pad=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_polynomial_under_overfitting(
    degrees: List[int],
    train_mse: List[float],
    test_mse: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plots training vs testing MSE across polynomial degrees to illustrate bias-variance tradeoff."""
    set_plot_style()
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_mse, marker="o", color="#2ca02c", lw=2, label="Train MSE (Bias check)")
    plt.plot(degrees, test_mse, marker="s", color="#d62728", lw=2, label="Test MSE (Variance check)")
    plt.xlabel("Polynomial Degree", fontweight="semibold")
    plt.ylabel("Mean Squared Error (MSE)", fontweight="semibold")
    plt.title("Polynomial Regression: Underfitting vs. Overfitting", fontweight="bold", pad=12)
    plt.xticks(degrees)
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


# ==========================================
# Clustering & PCA Plots
# ==========================================

def plot_elbow_curve(
    k_values: List[int],
    wcss_values: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plots Within-Cluster Sum of Squares (WCSS) vs K (Elbow curve)."""
    set_plot_style()
    plt.figure(figsize=(8, 5.5))
    plt.plot(k_values, wcss_values, marker="o", color="#1f77b4", lw=2.5, markersize=8)
    plt.xlabel("Number of Clusters (K)", fontweight="semibold")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontweight="semibold")
    plt.title("K-Means Elbow Method for Optimal K", fontweight="bold", pad=12)
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plots individual and cumulative explained variance ratio."""
    set_plot_style()
    cum_var = np.cumsum(explained_variance_ratio)
    components = np.arange(1, len(explained_variance_ratio) + 1)
    
    plt.figure(figsize=(9, 5.5))
    plt.bar(components, explained_variance_ratio, alpha=0.6, color="#4c72b0",
            label="Individual Explained Variance")
    plt.step(components, cum_var, where="mid", color="#c44e52", lw=2.5,
             label="Cumulative Explained Variance")
    plt.xlabel("Principal Component Index", fontweight="semibold")
    plt.ylabel("Explained Variance Ratio", fontweight="semibold")
    plt.title("PCA Explained Variance per Component", fontweight="bold", pad=12)
    plt.xticks(components)
    plt.ylim([0, 1.05])
    plt.legend(loc="center right", frameon=True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_pca_2d_scatter(
    Z_2d: np.ndarray,
    color_by: Optional[np.ndarray] = None,
    color_label: str = "Target",
    title: str = "Gas Turbine Data Projected onto First 2 Principal Components",
    save_path: Optional[str] = None
) -> None:
    """Plots 2D projection onto PC1 and PC2."""
    set_plot_style()
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        Z_2d[:, 0], Z_2d[:, 1],
        c=color_by,
        cmap="viridis",
        alpha=0.4,
        s=12,
        edgecolors="none"
    )
    plt.xlabel("Principal Component 1 (PC1)", fontweight="semibold")
    plt.ylabel("Principal Component 2 (PC2)", fontweight="semibold")
    plt.title(title, fontweight="bold", pad=12)
    if color_by is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_label, fontweight="semibold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
