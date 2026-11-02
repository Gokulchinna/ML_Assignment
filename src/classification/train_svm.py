"""
Support Vector Machine (SVM) with RBF kernel and Stratified Cross-Validation.
Uses CalibratedClassifierCV to provide valid multiclass probability estimates for ROC curve analysis.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def train_svm_classifier(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    c_grid: List[float] = [0.5, 1.0, 5.0],
    gamma_grid: List[str] = ["scale", "auto"],
    cv_folds: int = 3,
    benchmark_subsample: int = 10000,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Selects optimal regularization parameter C and kernel coefficient gamma for RBF SVM
    using Stratified Cross-Validation, fits on training data, and generates test predictions
    and probability estimates.
    
    Returns:
        - final_model: Fitted calibrated RBF SVM model.
        - results: Dictionary of best params, CV summary, test predictions, and probabilities.
    """
    param_grid = {
        "C": c_grid,
        "gamma": gamma_grid
    }
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Base RBF SVM without deprecated probability=True
    base_svc = SVC(kernel="rbf", random_state=random_state)
    
    # If training set is very large, perform hyperparameter grid search on a stratified subset
    n_train = X_train.shape[0]
    if n_train > benchmark_subsample:
        # Stratified subsampling for search
        from sklearn.model_selection import train_test_split
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=benchmark_subsample,
            stratify=y_train,
            random_state=random_state
        )
        search_X, search_y = X_sub, y_sub
    else:
        search_X, search_y = X_train, y_train
        
    grid = GridSearchCV(
        estimator=base_svc,
        param_grid=param_grid,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(search_X, search_y)
    
    best_params = grid.best_params_
    best_c = best_params["C"]
    best_gamma = best_params["gamma"]
    
    # Fit final model on full X_train using CalibratedClassifierCV to get clean probabilities
    best_svc = SVC(kernel="rbf", C=best_c, gamma=best_gamma, random_state=random_state)
    calibrated_model = CalibratedClassifierCV(estimator=best_svc, cv=3)
    calibrated_model.fit(X_train, y_train)
    
    y_test_pred = calibrated_model.predict(X_test)
    y_test_prob = calibrated_model.predict_proba(X_test)
    
    cv_df = pd.DataFrame(grid.cv_results_)[
        ["param_C", "param_gamma", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    
    results = {
        "best_params": best_params,
        "best_cv_score": float(grid.best_score_),
        "cv_results_df": cv_df,
        "y_pred": y_test_pred,
        "y_prob": y_test_prob
    }
    
    return calibrated_model, results
