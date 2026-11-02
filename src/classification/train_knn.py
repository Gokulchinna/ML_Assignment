"""
K-Nearest Neighbours (KNN) classifier with Stratified Cross-Validation on TF-IDF features.
Uses brute-force cosine distance for sparse text representations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def train_knn_classifier(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    n_neighbors_grid: List[int] = [3, 5, 7, 11],
    weights_grid: List[str] = ["uniform", "distance"],
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[KNeighborsClassifier, Dict[str, Any]]:
    """
    Tunes KNN hyperparameters using 5-fold Stratified Cross-Validation on X_train,
    fits the optimal estimator, and predicts test labels and probabilities.
    
    Returns:
        - best_model: Fitted KNeighborsClassifier.
        - results: Dictionary of best params, CV summary table, test predictions and probabilities.
    """
    param_grid = {
        "n_neighbors": n_neighbors_grid,
        "weights": weights_grid
    }
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    knn_base = KNeighborsClassifier(algorithm="brute", metric="cosine")
    
    grid = GridSearchCV(
        estimator=knn_base,
        param_grid=param_grid,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)
    
    # Compile CV results summary
    cv_df = pd.DataFrame(grid.cv_results_)[
        ["param_n_neighbors", "param_weights", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    
    results = {
        "best_params": best_params,
        "best_cv_score": float(grid.best_score_),
        "cv_results_df": cv_df,
        "y_pred": y_test_pred,
        "y_prob": y_test_prob
    }
    
    return best_model, results
