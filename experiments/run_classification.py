"""
Master Execution Script for Part 1: Land-Cover Classification on EuroSAT RGB Imagery.
Executes all stages end-to-end adhering strictly to the IIIT Sri City ML rubric:
1. Stage 1: Dataset Exploration and Stratified 80/20 Partitioning (Zero Data Leakage).
2. Stage 2: Histogram of Oriented Gradients (HOG) Feature Extraction.
3. Stage 3: K-Nearest Neighbour Classifier with 5-Fold Stratified Cross-Validation.
4. Stage 4: Decision Tree Classifier with Complexity Controls & Overfitting Analysis.
5. Stage 5: Support Vector Machine with RBF Kernel, Margin Analysis & Calibrated Probabilities.
6. Stage 6: Deep Convolutional Neural Network (Adapted ResNet-18) Trained on RGB Spatial Tensors.
7. Stage 7: Unified Multi-Class Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrices, Comparative ROC Curves).
"""

import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

# Unbuffer stdout so progress logs flush immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification.prepare_data import prepare_eurosat_split, get_eurosat_dataloaders
from src.classification.extract_features import get_or_extract_hog_features
from src.classification.train_knn import train_knn_classifier
from src.classification.train_decision_tree import train_decision_tree_classifier
from src.classification.train_svm import train_svm_classifier
from src.classification.train_resnet import train_resnet18_classifier
from src.classification.evaluate import evaluate_all_classifiers


def run_full_classification_pipeline(max_stage: int = 7):
    total_pipeline_start = time.time()
    print("=" * 80)
    print("  IIIT SRI CITY - MACHINE LEARNING ASSIGNMENT")
    print("  PART 1: COMPREHENSIVE IMAGE CLASSIFICATION BENCHMARK ON EUROSAT RGB")
    print("  MODELS: KNN (w/ CV) | DECISION TREE | SVM (RBF KERNEL) | DEEP CNN (RESNET-18)")
    print("=" * 80)
    
    # 0. Load Configuration
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    seed = config["project"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dir = os.path.join(PROJECT_ROOT, config["data_paths"]["classification"]["raw"])
    interim_dir = os.path.join(PROJECT_ROOT, config["data_paths"]["classification"]["interim"])
    plot_dir = os.path.join(PROJECT_ROOT, config["outputs"]["classification"]["plots"])
    metric_dir = os.path.join(PROJECT_ROOT, config["outputs"]["classification"]["metrics"])
    model_dir = os.path.join(PROJECT_ROOT, config["outputs"]["classification"]["models"])
    reports_fig_dir = os.path.join(PROJECT_ROOT, config["outputs"]["reports"]["figures"])
    reports_res_dir = os.path.join(PROJECT_ROOT, config["outputs"]["reports"]["results"])
    
    split_file = os.path.join(interim_dir, "eurosat_split.json")
    hog_cache_file = os.path.join(interim_dir, "eurosat_hog_features.npz")
    
    # -------------------------------------------------------------
    # STAGE 1: Dataset Exploration & Stratified 80/20 Partitioning
    # -------------------------------------------------------------
    if os.path.exists(split_file):
        print(f"\n[Stage 1] Loading verified EuroSAT split from: {split_file}")
        with open(split_file, "r") as f:
            split_meta = json.load(f)
    else:
        print("\n[Stage 1] Generating EuroSAT exploration figures and stratified 80/20 split...")
        split_meta = prepare_eurosat_split(
            data_dir=data_dir,
            interim_dir=interim_dir,
            plot_dir=plot_dir,
            metric_dir=metric_dir,
            test_size=config["classification"].get("test_size", 0.20),
            val_size=config["classification"].get("val_size", 0.10),
            random_state=seed
        )
        
    class_names = split_meta["class_names"]
    print(f"  EuroSAT Land-Cover Classes ({len(class_names)}): {class_names}")
    print(f"  Partition counts: Train/Val={split_meta['counts']['train_val_total']}, Test={split_meta['counts']['test_total']}")
    
    # -------------------------------------------------------------
    # STAGE 2: HOG Feature Extraction for Traditional ML
    # -------------------------------------------------------------
    print("\n[Stage 2] Loading HOG spatial gradient feature representations...")
    X_train, y_train, X_test, y_test, class_names = get_or_extract_hog_features(
        split_meta=split_meta,
        cache_path=hog_cache_file,
        orientations=config["classification"]["hog"].get("orientations", 9),
        pixels_per_cell=tuple(config["classification"]["hog"].get("pixels_per_cell", [8, 8])),
        cells_per_block=tuple(config["classification"]["hog"].get("cells_per_block", [2, 2]))
    )
    print(f"  X_train shape: {X_train.shape} (HOG feature dimensionality: {X_train.shape[1]})")
    print(f"  X_test shape:  {X_test.shape}")
    
    results = {}
    
    # -------------------------------------------------------------
    # STAGE 3: K-Nearest Neighbours with 5-Fold Stratified CV
    # -------------------------------------------------------------
    knn_cfg = config["classification"]["knn"]
    knn_model, knn_res = train_knn_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        k_grid=knn_cfg.get("n_neighbors_grid", [3, 5, 7, 9, 11, 15]),
        weights_grid=knn_cfg.get("weights", ["uniform", "distance"]),
        cv_folds=knn_cfg.get("cv_folds", 5),
        tuning_subsample=5000,
        random_state=seed
    )
    results["knn"] = knn_res
    
    # -------------------------------------------------------------
    # STAGE 4: Decision Tree with Complexity Controls
    # -------------------------------------------------------------
    dt_cfg = config["classification"]["decision_tree"]
    dt_plot_path = os.path.join(plot_dir, "decision_tree_structure.png")
    dt_model, dt_res = train_decision_tree_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        max_depth_grid=dt_cfg.get("max_depth_grid", [10, 15, 20, 25, None]),
        min_samples_split_grid=dt_cfg.get("min_samples_split_grid", [2, 5, 10]),
        criteria_grid=dt_cfg.get("criteria", ["gini", "entropy"]),
        cv_folds=dt_cfg.get("cv_folds", 5),
        tuning_subsample=5000,
        tree_plot_path=dt_plot_path,
        random_state=seed
    )
    results["decision_tree"] = dt_res
    
    # -------------------------------------------------------------
    # STAGE 5: SVM with RBF Kernel
    # -------------------------------------------------------------
    svm_cfg = config["classification"]["svm"]
    svm_model, svm_res = train_svm_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        c_grid=svm_cfg.get("c_grid", [0.1, 1.0, 10.0]),
        gamma_grid=svm_cfg.get("gamma_grid", ["scale", "auto"]),
        cv_folds=svm_cfg.get("cv_folds", 3),
        tuning_subsample=svm_cfg.get("benchmark_subsample", 5000),
        random_state=seed
    )
    results["svm"] = svm_res
    
    # Check if user requested to stop at Stage 5
    if max_stage <= 5:
        print("\n" + "=" * 80)
        print("  STAGE 1 TO 5 COMPLETED SUCCESSFULLY (STOPPING AS REQUESTED)")
        print("=" * 80)
        
        # Save intermediate traditional ML results
        stage_5_results_file = os.path.join(metric_dir, "traditional_ml_stage1_to_5.npz")
        np.savez_compressed(
            stage_5_results_file,
            knn_pred=knn_res["y_pred"],
            knn_prob=knn_res["y_prob"],
            dt_pred=dt_res["y_pred"],
            dt_prob=dt_res["y_prob"],
            svm_pred=svm_res["y_pred"],
            svm_prob=svm_res["y_prob"],
            y_test=y_test
        )
        
        summary_rows = [
            {
                "Model": "KNN (with CV)",
                "Best Params": f"K={knn_res['best_k']}, weights='{knn_res['best_weights']}'",
                "CV Accuracy": round(knn_res["best_cv_score"], 4),
                "Train Accuracy": round(knn_res["train_accuracy"], 4),
                "Test Accuracy": round(knn_res["test_accuracy"], 4),
                "Gen Gap (%)": round((knn_res["train_accuracy"] - knn_res["test_accuracy"])*100, 2),
                "Train Time (s)": round(knn_res["training_time_s"], 2),
                "Pred Time (s)": round(knn_res["prediction_time_s"], 2)
            },
            {
                "Model": "Decision Tree",
                "Best Params": f"depth={dt_res['best_params']['max_depth']}, crit='{dt_res['best_params']['criterion']}'",
                "CV Accuracy": round(dt_res["best_cv_score"], 4),
                "Train Accuracy": round(dt_res["train_accuracy"], 4),
                "Test Accuracy": round(dt_res["test_accuracy"], 4),
                "Gen Gap (%)": round((dt_res["train_accuracy"] - dt_res["test_accuracy"])*100, 2),
                "Train Time (s)": round(dt_res["training_time_s"], 2),
                "Pred Time (s)": round(dt_res["prediction_time_s"], 2)
            },
            {
                "Model": "SVM (RBF Kernel)",
                "Best Params": f"C={svm_res['best_c']}, gamma='{svm_res['best_gamma']}'",
                "CV Accuracy": round(svm_res["best_cv_score"], 4),
                "Train Accuracy": round(svm_res["train_accuracy"], 4),
                "Test Accuracy": round(svm_res["test_accuracy"], 4),
                "Gen Gap (%)": round((svm_res["train_accuracy"] - svm_res["test_accuracy"])*100, 2),
                "Train Time (s)": round(svm_res["training_time_s"], 2),
                "Pred Time (s)": round(svm_res["prediction_time_s"], 2)
            }
        ]
        summary_df = pd.DataFrame(summary_rows)
        print("\n--- [Intermediate Traditional ML Performance Summary (Stages 1-5)] ---")
        print(summary_df.to_string(index=False))
        
        summary_csv = os.path.join(metric_dir, "traditional_ml_stage1_to_5.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSaved Stage 1-5 summary metrics to: {summary_csv}")
        print(f"Saved predictions and probabilities to: {stage_5_results_file}")
        
        duration = time.time() - total_pipeline_start
        print(f"\nExecution finished up to Stage 5 in {duration:.1f}s ({duration/60:.2f} min).")
        return
        
    # -------------------------------------------------------------
    # STAGE 6: Deep CNN (ResNet-18) Trained on Spatial RGB Tensors
    # -------------------------------------------------------------
    cnn_cfg = config["classification"]["cnn"]
    train_loader, val_loader, test_loader = get_eurosat_dataloaders(
        split_meta=split_meta,
        batch_size=cnn_cfg.get("batch_size", 64),
        num_workers=0
    )
    
    ckpt_path = os.path.join(model_dir, "best_resnet18.pth")
    curves_path = os.path.join(plot_dir, "resnet18_learning_curves.png")
    
    resnet_model, resnet_res = train_resnet18_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=len(class_names),
        epochs=cnn_cfg.get("num_epochs", 15),
        lr=cnn_cfg.get("learning_rate", 0.001),
        weight_decay=cnn_cfg.get("weight_decay", 1e-4),
        patience=cnn_cfg.get("early_stopping_patience", 4),
        checkpoint_path=ckpt_path,
        curve_plot_path=curves_path
    )
    results["resnet18"] = resnet_res
    
    # -------------------------------------------------------------
    # STAGE 7: Unified Multi-Class Evaluation & ROC Analysis
    # -------------------------------------------------------------
    comparison_df, best_classifier = evaluate_all_classifiers(
        results_dict=results,
        y_test=y_test,
        class_names=class_names,
        plot_dir=plot_dir,
        metric_dir=metric_dir,
        reports_figure_dir=reports_fig_dir,
        reports_results_dir=reports_res_dir
    )
    
    total_pipeline_duration = time.time() - total_pipeline_start
    print("\n" + "=" * 80)
    print(f"  CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY IN {total_pipeline_duration:.1f}s ({total_pipeline_duration/60:.2f} min)")
    print(f"  WINNING CLASSIFIER: {best_classifier}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroSAT Classification Benchmark Runner")
    parser.add_argument("--max-stage", type=int, default=7, help="Maximum stage to execute (e.g. 5 to stop after SVM)")
    args = parser.parse_args()
    run_full_classification_pipeline(max_stage=args.max_stage)
