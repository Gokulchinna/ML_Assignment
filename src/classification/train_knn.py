"""
Stage 3: K-Nearest Neighbours (KNN) Classifier with 5-Fold Stratified Cross-Validation on HOG Features.
Evaluates model complexity vs. overfitting across K in [3, 5, 7, 9, 11, 15].
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score


def train_knn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_grid: List[int] = [3, 5, 7, 9, 11, 15],
    weights_grid: List[str] = ["uniform", "distance"],
    cv_folds: int = 5,
    tuning_subsample: Optional[int] = 5000,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Tunes KNN with StandardScaler using 5-fold Stratified Cross-Validation on HOG features.
    Evaluates on both training and held-out test set to document complexity and overfitting.
    """
    from sklearn.model_selection import train_test_split
    print("\n--- [Stage 3: KNN Classifier with 5-Fold CV] ---")
    start_train_time = time.time()
    
    # 1. Hyperparameter tuning via Stratified CV
    if tuning_subsample and tuning_subsample < len(X_train):
        print(f"  Tuning K and weights with 5-fold CV on representative {tuning_subsample}-sample stratified subset...")
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
        ("knn", KNeighborsClassifier(algorithm="auto", metric="euclidean"))
    ])
    
    param_grid = {
        "knn__n_neighbors": k_grid,
        "knn__weights": weights_grid
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
    best_k = best_params["knn__n_neighbors"]
    best_weights = best_params["knn__weights"]
    cv_acc = float(grid.best_score_)
    print(f"  Optimal parameters found: K={best_k}, weights='{best_weights}' (CV Acc: {cv_acc:.4f})")
    
    # 2. Fit final pipeline on full training set
    print(f"  Fitting final KNN pipeline on full training set ({len(X_train)} samples)...")
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, algorithm="auto", metric="euclidean"))
    ])
    best_model.fit(X_train, y_train)
    training_duration = time.time() - start_train_time
    
    # 3. Fast predictions on held-out test set
    start_pred_time = time.time()
    y_test_prob = best_model.predict_proba(X_test)
    y_test_pred = np.argmax(y_test_prob, axis=1)
    prediction_duration = time.time() - start_pred_time
    
    # Train accuracy from CV results (avoids redundant 21,600 x 21,600 distance matrix calculation)
    train_acc = float(grid.cv_results_["mean_train_score"][grid.best_index_])
    test_acc = float(accuracy_score(y_test, y_test_pred))
    cv_acc = float(grid.best_score_)
    
    # Extract CV results
    cv_df = pd.DataFrame(grid.cv_results_)[
        ["param_knn__n_neighbors", "param_knn__weights", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(columns={
        "param_knn__n_neighbors": "K",
        "param_knn__weights": "Weights",
        "mean_train_score": "Mean_Train_Acc",
        "mean_test_score": "Mean_CV_Acc",
        "std_test_score": "Std_CV_Acc",
        "rank_test_score": "Rank"
    }).sort_values("Rank")
    
    print(f"  Best Parameters: K={best_k}, weights='{best_weights}'")
    print(f"  5-Fold CV Accuracy: {cv_acc:.4f}")
    print(f"  Train Accuracy:     {train_acc:.4f}")
    print(f"  Test Accuracy:      {test_acc:.4f}")
    print(f"  Generalization Gap: {(train_acc - test_acc)*100:.2f}%")
    print(f"  Training Time:      {training_duration:.2f}s | Prediction Time: {prediction_duration:.2f}s")
    
    results = {
        "model_name": "KNN (with CV)",
        "best_k": best_k,
        "best_weights": best_weights,
        "best_params": best_params,
        "best_cv_score": cv_acc,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "training_time_s": training_duration,
        "prediction_time_s": prediction_duration,
        "cv_results_df": cv_df,
        "y_pred": y_test_pred,
        "y_prob": y_test_prob
    }
    
    return best_model, results
