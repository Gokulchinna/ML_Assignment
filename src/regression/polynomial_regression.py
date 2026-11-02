"""
Polynomial Regression implementation demonstrating underfitting and overfitting across polynomial degrees.
Applies polynomial expansion strictly to continuous numerical features.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from src.common.metrics import compute_regression_metrics


def evaluate_polynomial_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    continuous_indices: List[int],
    degrees: List[int] = [1, 2, 3]
) -> Tuple[pd.DataFrame, Dict[int, Any]]:
    """
    Evaluates polynomial regression models for degrees d in degrees.
    Polynomial expansion is applied exclusively to continuous features to prevent
    high-dimensional explosion.
    
    Returns:
        - summary_df: Table comparing Train/Test MSE, RMSE, and R2 across degrees.
        - models: Dictionary mapping degree d to (poly_transformer, linear_model).
    """
    non_cont_indices = [i for i in range(X_train.shape[1]) if i not in continuous_indices]
    
    results = []
    fitted_models = {}
    
    for d in degrees:
        poly = PolynomialFeatures(degree=d, include_bias=False)
        
        # Fit polynomial transformer strictly on training continuous features
        X_train_cont_poly = poly.fit_transform(X_train[:, continuous_indices])
        X_test_cont_poly = poly.transform(X_test[:, continuous_indices])
        
        # Concatenate with remaining unexpanded features
        if non_cont_indices:
            X_train_d = np.hstack([X_train_cont_poly, X_train[:, non_cont_indices]])
            X_test_d = np.hstack([X_test_cont_poly, X_test[:, non_cont_indices]])
        else:
            X_train_d = X_train_cont_poly
            X_test_d = X_test_cont_poly
            
        # Fit standard OLS Linear Regression
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train_d, y_train)
        
        y_train_pred = model.predict(X_train_d)
        y_test_pred = model.predict(X_test_d)
        
        train_m = compute_regression_metrics(y_train, y_train_pred)
        test_m = compute_regression_metrics(y_test, y_test_pred)
        
        results.append({
            "Degree": d,
            "Num_Features": X_train_d.shape[1],
            "Train_MSE": train_m["mse"],
            "Train_RMSE": train_m["rmse"],
            "Train_R2": train_m["r2"],
            "Test_MSE": test_m["mse"],
            "Test_RMSE": test_m["rmse"],
            "Test_R2": test_m["r2"]
        })
        fitted_models[d] = {
            "transformer": poly,
            "model": model,
            "train_pred": y_train_pred,
            "test_pred": y_test_pred
        }
        
    summary_df = pd.DataFrame(results)
    return summary_df, fitted_models
