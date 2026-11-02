"""
Decision Tree classifier with Stratified Cross-Validation and complexity controls.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def train_decision_tree_classifier(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    max_depth_grid: List[Optional[int]] = [10, 20, 30, None],
    min_samples_split_grid: List[int] = [2, 5, 10],
    criteria_grid: List[str] = ["gini", "entropy"],
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Tunes Decision Tree hyperparameters using 5-fold Stratified Cross-Validation,
    demonstrating the effect of tree depth on complexity and overfitting.
    
    Returns:
        - best_model: Fitted DecisionTreeClassifier.
        - results: Dictionary of best params, CV summary, test predictions and probabilities.
    """
    param_grid = {
        "max_depth": max_depth_grid,
        "min_samples_split": min_samples_split_grid,
        "criterion": criteria_grid
    }
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    dt_base = DecisionTreeClassifier(random_state=random_state)
    
    grid = GridSearchCV(
        estimator=dt_base,
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
    
    cv_df = pd.DataFrame(grid.cv_results_)[
        ["param_max_depth", "param_min_samples_split", "param_criterion", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    
    results = {
        "best_params": best_params,
        "best_cv_score": float(grid.best_score_),
        "cv_results_df": cv_df,
        "y_pred": y_test_pred,
        "y_prob": y_test_prob,
        "tree_depth": int(best_model.get_depth()),
        "n_leaves": int(best_model.get_n_leaves())
    }
    
    return best_model, results
