"""
Evaluation module for Part 1 (Multiclass Classification).
Generates confusion matrices, comparative ROC curves, and compiles summary metric tables.
"""

from typing import List, Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from src.common.metrics import compute_classification_metrics
from src.common.plotting import plot_confusion_matrix, set_plot_style


def plot_combined_model_roc_curves(
    models_dict: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    class_names: List[str],
    save_path1: str,
    save_path2: str
) -> Dict[str, float]:
    """
    Plots Macro-average ROC curves for all models on a single comparative figure.
    Uses exact multiclass macro One-vs-Rest ROC-AUC consistent with metrics tables.
    Returns dictionary mapping model name to exact Macro ROC-AUC.
    """
    set_plot_style()
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=class_names)
    
    plt.figure(figsize=(9, 7))
    model_macro_aucs = {}
    
    colors = {"KNN (with CV)": "#1f77b4", "Decision Tree": "#2ca02c", "SVM (RBF Kernel)": "#d62728"}
    
    for model_name, res in models_dict.items():
        y_prob = res["y_prob"]
        
        # Exact macro-average One-vs-Rest ROC-AUC matching compute_classification_metrics
        exact_macro_auc = float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro"))
        model_macro_aucs[model_name] = exact_macro_auc
        
        # Smooth macro-average curve interpolation for visual plotting fidelity
        dense_fpr = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(dense_fpr)
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(dense_fpr, fpr_i, tpr_i)
        mean_tpr /= n_classes
        
        color = colors.get(model_name, None)
        plt.plot(dense_fpr, mean_tpr, color=color, lw=2.2,
                 label=f"{model_name} (Macro AUC = {exact_macro_auc:.4f})")
                 
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="semibold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="semibold")
    plt.title("Comparative Multiclass ROC Curves (Macro-Average One-vs-Rest)", fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path1)
    plt.savefig(save_path2)
    plt.close()
    
    return model_macro_aucs


def evaluate_all_classifiers(
    models_dict: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    class_names: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates all classification models and generates confusion matrices,
    comparative ROC curves, and metrics tables.
    """
    comparison_rows = []
    full_metrics = {}
    
    for model_name, res in models_dict.items():
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]
        
        metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            labels=class_names
        )
        full_metrics[model_name] = metrics
        
        # Save individual confusion matrix
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cm_path1 = f"outputs/classification/plots/confusion_matrix_{safe_name}.png"
        cm_path2 = f"reports/figures/confusion_matrix_{safe_name}.png"
        
        plot_confusion_matrix(
            cm=np.array(metrics["confusion_matrix"]),
            class_names=class_names,
            title=f"Confusion Matrix: {model_name}",
            save_path=cm_path1
        )
        plot_confusion_matrix(
            cm=np.array(metrics["confusion_matrix"]),
            class_names=class_names,
            title=f"Confusion Matrix: {model_name}",
            save_path=cm_path2
        )
        
        comparison_rows.append({
            "Classifier": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision_Macro": metrics["precision_macro"],
            "Precision_Weighted": metrics["precision_weighted"],
            "Recall_Macro": metrics["recall_macro"],
            "Recall_Weighted": metrics["recall_weighted"],
            "F1_Macro": metrics["f1_macro"],
            "F1_Weighted": metrics["f1_weighted"],
            "ROC_AUC_Macro": metrics["roc_auc_macro"],
            "ROC_AUC_Weighted": metrics["roc_auc_weighted"],
            "Best_Parameters": str(res["best_params"])
        })
        
    summary_df = pd.DataFrame(comparison_rows)
    return summary_df, full_metrics
