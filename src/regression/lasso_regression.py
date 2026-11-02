"""
Lasso Regression (L1 regularization) with TimeSeriesSplit cross-validation for alpha selection.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from src.common.metrics import compute_regression_metrics


def train_lasso_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: List[float] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    n_splits: int = 5,
    max_iter: int = 5000,
    random_state: int = 42
) -> Tuple[Lasso, Dict[str, Any]]:
    """
    Selects optimal L1 regularization parameter alpha using TimeSeriesSplit cross-validation,
    fits the final model on full X_train, inspects coefficient sparsity, and evaluates on train and test.
    
    Returns:
        - best_estimator: Fitted Lasso model with optimal alpha.
        - results: Dictionary with best alpha, train/test metrics, sparsity count, and CV scores.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    param_grid = {"alpha": alphas}
    
    grid = GridSearchCV(
        estimator=Lasso(max_iter=max_iter, random_state=random_state),
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_alpha = float(grid.best_params_["alpha"])
    best_model = grid.best_estimator_
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)
    
    # Analyze sparsity
    zero_coefficients = int(np.sum(best_model.coef_ == 0.0))
    non_zero_coefficients = int(np.sum(best_model.coef_ != 0.0))
    
    cv_results = []
    for a, mean_score, std_score in zip(
        alphas,
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"]
    ):
        cv_results.append({
            "alpha": float(a),
            "mean_cv_mse": float(-mean_score),
            "std_cv_mse": float(std_score)
        })
        
    results = {
        "best_alpha": best_alpha,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "zero_coefficients": zero_coefficients,
        "non_zero_coefficients": non_zero_coefficients,
        "cv_results": cv_results,
        "train_pred": y_train_pred,
        "test_pred": y_test_pred
    }
    
    return best_model, results
