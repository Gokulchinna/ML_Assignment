"""
Stage 7: Unified Multi-Class Evaluation and Comparative Analysis.
Computes and visualizes:
1. Complete classification metrics: Accuracy, Macro/Weighted Precision, Recall, F1-score, and Macro OvR ROC-AUC.
2. Individual and combined Confusion Matrices (raw and normalized).
3. Comparative One-vs-Rest ROC curves on a single plot to identify the winning classifier.
4. Overfitting and model complexity diagnostics.
5. Consolidated CSV/JSON summary tables.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Computes all metrics mandated by the assignment rubric:
    Accuracy, Macro & Weighted Precision, Recall, F1, and Macro OvR ROC-AUC.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # Macro One-vs-Rest ROC-AUC
    n_classes = len(class_names)
    try:
        roc_auc_macro = float(roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        ))
    except Exception as e:
        print(f"  Warning calculating ROC-AUC: {e}")
        roc_auc_macro = float("nan")
        
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro
    }


def plot_single_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str,
    normalize: bool = True
):
    """Generates an annotated heatmap confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 7.5))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, cbar=True)
        plt.title(f"{model_name}: Normalized Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, cbar=True)
        plt.title(f"{model_name}: Raw Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
        
    plt.xlabel("Predicted Class", fontsize=11, fontweight="semibold")
    plt.ylabel("True Class", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_combined_confusion_matrices(
    cms: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str
):
    """Plots a 2x2 multi-panel figure of all 4 normalized confusion matrices."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (model_name, cm) in enumerate(cms.items()):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 7.5}
        )
        axes[idx].set_title(f"({chr(97+idx)}) {model_name}", fontsize=12, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10, fontweight="semibold")
        axes[idx].set_ylabel("True Label", fontsize=10, fontweight="semibold")
        axes[idx].tick_params(axis="x", rotation=40, labelsize=8)
        axes[idx].tick_params(axis="y", rotation=0, labelsize=8)
        
    plt.suptitle("EuroSAT Land-Cover Classification: 4-Way Confusion Matrix Comparison (Normalized)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved 4-way combined confusion matrix grid to: {save_path}")


def plot_comparative_roc_curves(
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str
) -> str:
    """
    Plots the Macro-average One-vs-Rest ROC curve for all 4 models on a single figure.
    Identifies and returns the best model name based on ROC-AUC.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    colors = {
        "K-Nearest Neighbours": "#2ca02c",
        "KNN (with CV)": "#2ca02c",
        "Decision Tree": "#ff7f0e",
        "Support Vector Machine (RBF)": "#1f77b4",
        "SVM (RBF Kernel)": "#1f77b4",
        "Deep CNN (ResNet-18)": "#d62728"
    }
    line_styles = ["-", "--", "-.", ":"]
    
    model_auc_scores = {}
    
    for idx, (model_name, y_prob) in enumerate(probs_dict.items()):
        # Compute macro-average ROC curve across all classes
        all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, c], y_prob[:, c])[0] for c in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
            mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
        mean_tpr /= n_classes
        
        auc_score = float(roc_auc_score(y_true_bin, y_prob, average="macro"))
        model_auc_scores[model_name] = auc_score
        
        color = colors.get(model_name, None)
        ls = line_styles[idx % len(line_styles)]
        plt.plot(all_fpr, mean_tpr, label=f"{model_name} (Macro ROC-AUC = {auc_score:.4f})",
                 color=color, linestyle=ls, linewidth=2.5)
        
    # Baseline diagonal
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", linewidth=1.5, alpha=0.7)
    
    best_model = max(model_auc_scores, key=model_auc_scores.get)
    best_auc = model_auc_scores[best_model]
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="semibold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="semibold")
    plt.title("Comparative Macro-Average One-vs-Rest ROC Curves", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Annotation box for best classifier
    plt.annotate(
        f"Best Classifier: {best_model}\nMacro ROC-AUC: {best_auc:.4f}",
        xy=(0.04, 0.88),
        xycoords="axes fraction",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf2f8", edgecolor="#2b5c8f", lw=1.5)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved comparative ROC curves to: {save_path}")
    print(f"  --> Best Classifier based on ROC analysis: {best_model} (AUC: {best_auc:.4f})")
    
    return best_model


def evaluate_all_classifiers(
    results_dict: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    class_names: List[str],
    plot_dir: str = "outputs/classification/plots",
    metric_dir: str = "outputs/classification/metrics",
    reports_figure_dir: str = "reports/figures",
    reports_results_dir: str = "reports/results"
) -> Tuple[pd.DataFrame, str]:
    """
    Executes unified evaluation across all 4 classifiers:
    - Calculates all metrics
    - Generates confusion matrix heatmaps
    - Plots combined ROC curves
    - Saves comparison CSV and JSON
    - Synchronizes figures to reports/
    """
    print("\n--- [Stage 7: Unified Multi-Class Evaluation] ---")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(reports_figure_dir, exist_ok=True)
    os.makedirs(reports_results_dir, exist_ok=True)
    
    comparison_rows = []
    confusion_matrices = {}
    probs_dict = {}
    
    for model_key, res in results_dict.items():
        name = res["model_name"]
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]
        probs_dict[name] = y_prob
        
        metrics = compute_comprehensive_metrics(y_test, y_pred, y_prob, class_names)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        
        # Save individual confusion matrix plot
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        cm_path = os.path.join(plot_dir, f"confusion_matrix_{clean_name}.png")
        plot_single_confusion_matrix(cm, class_names, name, cm_path, normalize=True)
        
        train_acc = res.get("train_accuracy", float("nan"))
        test_acc = metrics["accuracy"]
        gen_gap = (train_acc - test_acc) * 100.0 if not np.isnan(train_acc) else float("nan")
        
        row = {
            "Classifier": name,
            "Accuracy": round(test_acc, 4),
            "Macro Precision": round(metrics["precision_macro"], 4),
            "Weighted Precision": round(metrics["precision_weighted"], 4),
            "Macro Recall": round(metrics["recall_macro"], 4),
            "Weighted Recall": round(metrics["recall_weighted"], 4),
            "Macro F1-Score": round(metrics["f1_macro"], 4),
            "Weighted F1-Score": round(metrics["f1_weighted"], 4),
            "Macro ROC-AUC": round(metrics["roc_auc_macro"], 4),
            "Train Accuracy": round(train_acc, 4) if not np.isnan(train_acc) else "N/A",
            "Generalization Gap (%)": round(gen_gap, 2) if not np.isnan(gen_gap) else "N/A",
            "Train Time (s)": round(res.get("training_time_s", 0.0), 2),
            "Pred Time (s)": round(res.get("prediction_time_s", 0.0), 2)
        }
        comparison_rows.append(row)
        
    # Combined Confusion Matrix Grid
    all_cm_path = os.path.join(plot_dir, "all_confusion_matrices.png")
    plot_combined_confusion_matrices(confusion_matrices, class_names, all_cm_path)
    
    # Combined ROC Curves
    roc_plot_path = os.path.join(plot_dir, "comparative_roc_curves.png")
    best_classifier = plot_comparative_roc_curves(y_test, probs_dict, class_names, roc_plot_path)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_rows).sort_values("Macro ROC-AUC", ascending=False)
    
    # Save CSVs and JSONs
    csv_metric_path = os.path.join(metric_dir, "classification_comparison.csv")
    csv_report_path = os.path.join(reports_results_dir, "classification_comparison.csv")
    json_metric_path = os.path.join(metric_dir, "classification_comparison.json")
    
    df.to_csv(csv_metric_path, index=False)
    df.to_csv(csv_report_path, index=False)
    
    with open(json_metric_path, "w") as f:
        json.dump({
            "comparison_table": df.to_dict(orient="records"),
            "best_classifier_by_roc_auc": best_classifier
        }, f, indent=2)
        
    # Copy all plots to reports/figures/ for report generation
    for f in os.listdir(plot_dir):
        if f.endswith((".png", ".jpg")):
            src_f = os.path.join(plot_dir, f)
            dst_f = os.path.join(reports_figure_dir, f)
            shutil.copy2(src_f, dst_f)
            
    print("\n--- [Final Classification Comparison Table] ---")
    print(df.to_string(index=False))
    print(f"\n--> Best Classifier based on ROC curve analysis: {best_classifier}")
    
    return df, best_classifier

