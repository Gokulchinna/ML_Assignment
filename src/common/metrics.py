"""
Evaluation metrics computation module for classification, regression, and clustering.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    r2_score
)


# ==========================================
# Classification Metrics
# ==========================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[list] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive multiclass classification metrics.
    
    Args:
        y_true: True ground-truth labels.
        y_pred: Predicted class labels.
        y_prob: Predicted class probabilities (shape: [N, num_classes]) for ROC-AUC.
        labels: Unique class labels list.
        
    Returns:
        Dictionary of computed metrics and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    roc_auc_macro = None
    roc_auc_weighted = None
    if y_prob is not None:
        try:
            roc_auc_macro = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=labels
            )
            roc_auc_weighted = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted", labels=labels
            )
        except Exception as e:
            roc_auc_macro = np.nan
            roc_auc_weighted = np.nan

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro": float(roc_auc_macro) if roc_auc_macro is not None else None,
        "roc_auc_weighted": float(roc_auc_weighted) if roc_auc_weighted is not None else None,
        "confusion_matrix": cm.tolist()
    }


# ==========================================
# Regression Metrics
# ==========================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MSE, RMSE, and R2.
    
    Args:
        y_true: Ground truth continuous target.
        y_pred: Predicted values.
        
    Returns:
        Dictionary with MSE, RMSE, and R2.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


# ==========================================
# Clustering & PCA Metrics
# ==========================================

def compute_wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Calculates Within-Cluster Sum of Squares (WCSS) from scratch.
    
    Args:
        X: Data points matrix [N, D].
        labels: Cluster assignment vector [N].
        centroids: Cluster centroid matrix [K, D].
        
    Returns:
        Scalar WCSS value.
    """
    wcss = 0.0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            diff = cluster_points - centroids[k]
            wcss += float(np.sum(diff ** 2))
    return wcss


def verify_pca_explained_variance(explained_variance_ratio: np.ndarray, tolerance: float = 1e-4) -> bool:
    """
    Verifies that the sum of all eigenvalues / explained variance ratios equals 1.0.
    """
    total = float(np.sum(explained_variance_ratio))
    return bool(np.isclose(total, 1.0, atol=tolerance))
