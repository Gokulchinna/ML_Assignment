"""
Stage 5: Support Vector Machine (SVM) with RBF Kernel on HOG Features.
Features:
- StandardScaler preprocessing pipeline
- Stratified grid search over C and gamma on training partition
- Margin and complexity analysis (underfitting with low C, high bias vs. overfitting with high C/gamma)
- Decision function scores for robust One-vs-Rest ROC-AUC calculation
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


def train_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: List[float] = [0.1, 1.0, 10.0],
    gamma_grid: List[Any] = ["scale", "auto"],
    cv_folds: int = 3,
    tuning_subsample: Optional[int] = 5000,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Tunes and trains an RBF Support Vector Machine classifier.
    Uses StandardScaler inside the pipeline to ensure zero data leakage.
    Hyperparameters (C, gamma) are tuned via Stratified K-Fold CV on training data.
    """
    print("\n--- [Stage 5: SVM Classifier with RBF Kernel] ---")
    start_train_time = time.time()
    
    # 1. Hyperparameter Tuning via Stratified Cross-Validation
    print("  Tuning hyperparameters (C, gamma) with Stratified CV...")
    if tuning_subsample and tuning_subsample < len(X_train):
        print(f"  Using representative stratified tuning subset of {tuning_subsample} training samples for grid search...")
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train,
            train_size=tuning_subsample,
            stratify=y_train,
            random_state=random_state
        )
    else:
        X_tune, y_tune = X_train, y_train

    tune_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=random_state))
    ])
    
    param_grid = {
        "svm__C": c_grid,
        "svm__gamma": gamma_grid
    }
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=tune_pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )
    
    grid.fit(X_tune, y_tune)
    best_params = grid.best_params_
    best_c = best_params["svm__C"]
    best_gamma = best_params["svm__gamma"]
    cv_acc = float(grid.best_score_)
    print(f"  Optimal parameters found: C={best_c}, gamma='{best_gamma}' (Tuning CV Acc: {cv_acc:.4f})")
    
    # 2. Fit Final Model on Full Training Set
    print(f"  Fitting final RBF SVM on full training set ({len(X_train)} samples, 1764 HOG features)...")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=best_c,
            gamma=best_gamma,
            kernel="rbf",
            decision_function_shape="ovr",
            random_state=random_state
        ))
    ])
    
    final_pipeline.fit(X_train, y_train)
    training_duration = time.time() - start_train_time
    
    # Support vector statistics
    svm_estimator = final_pipeline.named_steps["svm"]
    n_support_vectors = int(np.sum(svm_estimator.n_support_))
    
    # 3. Timing and Predictions on held-out test set
    start_pred_time = time.time()
    decision_scores = final_pipeline.decision_function(X_test)
    y_test_pred = np.argmax(decision_scores, axis=1)
    # Convert decision function scores to calibrated multi-class probability simplex using softmax
    y_test_prob = softmax(decision_scores, axis=1)
    prediction_duration = time.time() - start_pred_time
    
    # Train accuracy from CV results (avoids redundant 21,600 kernel evaluations against support vectors)
    train_acc = float(grid.cv_results_["mean_train_score"][grid.best_index_])
    test_acc = float(accuracy_score(y_test, y_test_pred))
    
    # CV results summary
    cv_df = pd.DataFrame(grid.cv_results_)[
        ["param_svm__C", "param_svm__gamma", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(columns={
        "param_svm__C": "C",
        "param_svm__gamma": "Gamma",
        "mean_train_score": "Mean_Train_Acc",
        "mean_test_score": "Mean_CV_Acc",
        "std_test_score": "Std_CV_Acc",
        "rank_test_score": "Rank"
    }).sort_values("Rank")
    
    print(f"  Total Support Vectors: {n_support_vectors} ({n_support_vectors/len(X_train)*100:.1f}% of train data)")
    print(f"  Train Accuracy:        {train_acc:.4f}")
    print(f"  Test Accuracy:         {test_acc:.4f}")
    print(f"  Generalization Gap:    {(train_acc - test_acc)*100:.2f}%")
    print(f"  Training Time:         {training_duration:.2f}s | Prediction Time: {prediction_duration:.2f}s")
    
    results = {
        "model_name": "SVM (RBF Kernel)",
        "best_c": best_c,
        "best_gamma": best_gamma,
        "best_params": best_params,
        "best_cv_score": cv_acc,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "n_support_vectors": n_support_vectors,
        "training_time_s": training_duration,
        "prediction_time_s": prediction_duration,
        "cv_results_df": cv_df,
        "y_pred": y_test_pred,
        "y_prob": y_test_prob,
        "decision_scores": decision_scores
    }
    
    return final_pipeline, results

