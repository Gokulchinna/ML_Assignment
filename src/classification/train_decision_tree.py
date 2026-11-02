"""
Stage 4: Decision Tree Classifier with Complexity Controls and Overfitting Analysis on HOG Features.
Evaluates tree depth, leaf count, and generalization gap under Gini/Entropy criteria.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score


def train_decision_tree_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    max_depth_grid: List[Optional[int]] = [10, 15, 20, 25, None],
    min_samples_split_grid: List[int] = [2, 5, 10],
    criteria_grid: List[str] = ["gini", "entropy"],
    cv_folds: int = 5,
    tuning_subsample: Optional[int] = 5000,
    tree_plot_path: Optional[str] = "outputs/classification/plots/decision_tree_structure.png",
    random_state: int = 42
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Tunes Decision Tree hyperparameters using 5-fold Stratified Cross-Validation on HOG features.
    Documents tree depth, leaf count, and the training-vs-testing overfitting gap.
    """
    from sklearn.model_selection import train_test_split
    print("\n--- [Stage 4: Decision Tree Classifier with 5-Fold CV] ---")
    start_train_time = time.time()
    
    # 1. Hyperparameter tuning via Stratified CV
    if tuning_subsample and tuning_subsample < len(X_train):
        print(f"  Tuning depth and criteria with 5-fold CV on representative {tuning_subsample}-sample stratified subset...")
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train,
            train_size=tuning_subsample,
            stratify=y_train,
            random_state=random_state
        )
    else:
        X_tune, y_tune = X_train, y_train
        
    dt_base = DecisionTreeClassifier(random_state=random_state)
    
    param_grid = {
        "max_depth": max_depth_grid,
        "min_samples_split": min_samples_split_grid,
        "criterion": criteria_grid
    }
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    grid = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )
    
    grid.fit(X_tune, y_tune)
    best_params = grid.best_params_
    cv_acc = float(grid.best_score_)
    print(f"  Optimal parameters found: {best_params} (CV Acc: {cv_acc:.4f})")
    
    # 2. Fit final Decision Tree on full training set
    print(f"  Fitting final Decision Tree on full training set ({len(X_train)} samples)...")
    best_model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        criterion=best_params["criterion"],
        random_state=random_state
    )
    best_model.fit(X_train, y_train)
    training_duration = time.time() - start_train_time
    
    # Timing and predictions
    start_pred_time = time.time()
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)
    prediction_duration = time.time() - start_pred_time
    
    y_train_pred = best_model.predict(X_train)
    
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    cv_acc = float(grid.best_score_)
    
    depth = int(best_model.get_depth())
    leaves = int(best_model.get_n_leaves())
    
    cv_df = pd.DataFrame(grid.cv_results_)[
        ["param_criterion", "param_max_depth", "param_min_samples_split", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(columns={
        "param_criterion": "Criterion",
        "param_max_depth": "Max_Depth",
        "param_min_samples_split": "Min_Samples_Split",
        "mean_train_score": "Mean_Train_Acc",
        "mean_test_score": "Mean_CV_Acc",
        "std_test_score": "Std_CV_Acc",
        "rank_test_score": "Rank"
    }).sort_values("Rank")
    
    print(f"  Best Parameters:     {best_params}")
    print(f"  Fitted Tree Depth:   {depth} levels | Leaf Nodes: {leaves}")
    print(f"  5-Fold CV Accuracy:  {cv_acc:.4f}")
    print(f"  Train Accuracy:      {train_acc:.4f}")
    print(f"  Test Accuracy:       {test_acc:.4f}")
    print(f"  Generalization Gap:  {(train_acc - test_acc)*100:.2f}% (Overfitting evidence)")
    print(f"  Training Time:       {training_duration:.2f}s | Prediction Time: {prediction_duration:.2f}s")
    
    # Generate and save a visually interpretable tree plot (depth limited to 3 for legibility)
    if tree_plot_path:
        os.makedirs(os.path.dirname(tree_plot_path), exist_ok=True)
        plt.figure(figsize=(18, 9))
        plot_tree(
            best_model,
            max_depth=3,
            feature_names=[f"HOG_{i}" for i in range(X_train.shape[1])],
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=8
        )
        plt.title(f"Decision Tree Pruned Structure (Showing Top 3 Levels of {depth}-Depth Tree)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(tree_plot_path, dpi=300)
        plt.close()
        print(f"  Saved Decision Tree structure visualization to: {tree_plot_path}")
        
    results = {
        "model_name": "Decision Tree",
        "best_params": best_params,
        "tree_depth": depth,
        "leaf_count": leaves,
        "best_cv_score": cv_acc,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "generalization_gap_pct": (train_acc - test_acc) * 100.0,
        "training_time_s": training_duration,
        "prediction_time_s": prediction_duration,
        "cv_results_df": cv_df,
        "y_pred": y_test_pred,
        "y_prob": y_test_prob
    }
    
    return best_model, results
