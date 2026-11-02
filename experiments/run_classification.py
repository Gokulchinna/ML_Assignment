"""
Experiment driver for Stage 3: Part 1 (Classification Analysis)
Executes K-Nearest Neighbours (with CV), Decision Tree (with depth tuning),
and Support Vector Machine (with RBF Kernel).
Generates confusion matrices, multiclass ROC curves, metrics comparison tables,
and identifies the best classifier based on ROC curve analysis.
"""

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to utf-8 if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.common.reproducibility import set_seed
from src.classification.prepare_data import prepare_classification_data
from src.classification.train_knn import train_knn_classifier
from src.classification.train_decision_tree import train_decision_tree_classifier
from src.classification.train_svm import train_svm_classifier
from src.classification.evaluate import evaluate_all_classifiers, plot_combined_model_roc_curves


def main():
    print("=" * 70)
    print("STAGE 3: PART 1 -- MULTICLASS CLASSIFICATION EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load Configuration
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    seed = config["project"]["seed"]
    set_seed(seed)
    print(f"[Config] Global deterministic seed set to: {seed}")
    
    raw_file = config["data_paths"]["classification"]["raw"]
    test_size = config["classification"]["test_size"]
    tfidf_cfg = config["classification"]["tfidf"]
    knn_cfg = config["classification"]["knn"]
    dt_cfg = config["classification"]["decision_tree"]
    svm_cfg = config["classification"]["svm"]
    
    # Ensure output directories exist
    os.makedirs("outputs/classification/metrics", exist_ok=True)
    os.makedirs("outputs/classification/plots", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    # 2. Data Preparation & 80/20 Stratified Split
    print(f"\n[1/4] Loading and Stratifying PriceRunner Dataset ({raw_file})...")
    X_train, y_train, X_test, y_test, class_names, vectorizer, meta = prepare_classification_data(
        raw_file,
        test_size=test_size,
        max_features=tfidf_cfg["max_features"],
        random_state=seed
    )
    print(f"  * Total samples: {meta['total_samples']:,} across {meta['num_classes']} balanced classes")
    print(f"  * Stratified Train samples (80%): {meta['train_samples']:,}")
    print(f"  * Stratified Test samples  (20%): {meta['test_samples']:,}")
    print(f"  * TF-IDF vocabulary size (fitted strictly on train): {meta['vocabulary_size']:,} n-grams")
    
    models_dict = {}
    runtimes = {}
    
    # 3. Model 1: KNN with Cross-Validation
    print(f"\n[2/4] Training K-Nearest Neighbours with 5-Fold Stratified CV...")
    t0 = time.time()
    knn_model, knn_res = train_knn_classifier(
        X_train, y_train, X_test,
        n_neighbors_grid=knn_cfg["n_neighbors_grid"],
        weights_grid=knn_cfg["weights"],
        cv_folds=knn_cfg["cv_folds"],
        random_state=seed
    )
    runtimes["KNN"] = time.time() - t0
    print(f"  * KNN Best Hyperparameters: {knn_res['best_params']} (CV Accuracy: {knn_res['best_cv_score']:.4f})")
    print(f"  * KNN Workflow Runtime: {runtimes['KNN']:.2f}s")
    models_dict["KNN (with CV)"] = knn_res
    
    # 4. Model 2: Decision Tree with Complexity Tuning
    print(f"\n[3/4] Training Decision Tree with 5-Fold Stratified CV...")
    t0 = time.time()
    dt_model, dt_res = train_decision_tree_classifier(
        X_train, y_train, X_test,
        max_depth_grid=dt_cfg["max_depth_grid"],
        min_samples_split_grid=dt_cfg["min_samples_split_grid"],
        criteria_grid=dt_cfg["criteria"],
        cv_folds=dt_cfg["cv_folds"],
        random_state=seed
    )
    runtimes["Decision Tree"] = time.time() - t0
    print(f"  * Decision Tree Best Hyperparameters: {dt_res['best_params']} (CV Accuracy: {dt_res['best_cv_score']:.4f})")
    print(f"  * Fitted Tree Depth: {dt_res['tree_depth']} levels, {dt_res['n_leaves']} leaves")
    print(f"  * Decision Tree Workflow Runtime: {runtimes['Decision Tree']:.2f}s")
    models_dict["Decision Tree"] = dt_res
    
    # 5. Model 3: Support Vector Machine with RBF Kernel
    print(f"\n[4/4] Training Support Vector Machine with RBF Kernel (with CV)...")
    t0 = time.time()
    svm_model, svm_res = train_svm_classifier(
        X_train, y_train, X_test,
        c_grid=svm_cfg["c_grid"],
        gamma_grid=svm_cfg["gamma_grid"],
        cv_folds=svm_cfg["cv_folds"],
        benchmark_subsample=svm_cfg["benchmark_subsample"],
        random_state=seed
    )
    runtimes["SVM (RBF)"] = time.time() - t0
    print(f"  * RBF SVM Best Hyperparameters: {svm_res['best_params']} (CV Accuracy: {svm_res['best_cv_score']:.4f})")
    print(f"  * RBF SVM Workflow Runtime: {runtimes['SVM (RBF)']:.2f}s")
    models_dict["SVM (RBF Kernel)"] = svm_res
    
    # 6. Comprehensive Metrics Compilation & Confusion Matrices
    print("\n" + "=" * 70)
    print("EVALUATING CLASSIFIERS ON HELD-OUT TEST SET (80/20 STRATIFIED)")
    print("=" * 70)
    summary_df, full_metrics = evaluate_all_classifiers(models_dict, y_test, class_names)
    
    # Add runtime column
    summary_df["Training_Runtime_s"] = [
        f"{runtimes['KNN']:.2f}",
        f"{runtimes['Decision Tree']:.2f}",
        f"{runtimes['SVM (RBF)']:.2f}"
    ]
    
    print("\n--- CLASSIFICATION METRICS COMPARISON TABLE ---")
    print(summary_df.to_string(index=False))
    
    # Save metrics table
    metrics_csv1 = "outputs/classification/metrics/classification_comparison.csv"
    metrics_csv2 = "reports/results/classification_comparison.csv"
    summary_df.to_csv(metrics_csv1, index=False)
    summary_df.to_csv(metrics_csv2, index=False)
    print(f"\n  * Saved classification comparison table to: {metrics_csv1} and {metrics_csv2}")
    
    # Plot Comparative ROC Curves
    roc_plot1 = "outputs/classification/plots/roc_curves_comparison.png"
    roc_plot2 = "reports/figures/roc_curves_comparison.png"
    macro_aucs = plot_combined_model_roc_curves(
        models_dict, y_test, class_names, roc_plot1, roc_plot2
    )
    print(f"  * Saved comparative ROC curves to: {roc_plot1}")
    
    # 7. Identify Best Classifier based on ROC Curve Analysis
    best_classifier = max(macro_aucs, key=macro_aucs.get)
    best_auc = macro_aucs[best_classifier]
    print("\n" + "=" * 70)
    print("BEST CLASSIFIER IDENTIFICATION (ROC CURVE ANALYSIS)")
    print("=" * 70)
    print(f"  >>> Best Classifier: {best_classifier} with Macro ROC-AUC = {best_auc:.4f}")
    for name, auc_val in macro_aucs.items():
        print(f"      - {name}: Macro ROC-AUC = {auc_val:.4f}")
        
    # Invariant checks
    assert len(y_train) == 28248
    assert len(y_test) == 7063
    assert len(class_names) == 10
    assert np.all(np.isfinite(summary_df["Accuracy"]))
    assert np.all(np.isfinite(summary_df["ROC_AUC_Macro"]))
    print("\n[PASS] All classification invariants and metric checks verified successfully.")
    
    print("\n" + "=" * 70)
    print("STAGE 3 COMPLETED SUCCESSFULLY: ALL METRICS & FIGURES EXPORTED")
    print("=" * 70)


if __name__ == "__main__":
    main()
