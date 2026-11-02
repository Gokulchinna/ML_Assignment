"""
Evaluation module for K-Means and PCA experiments.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.clustering.kmeans_scratch import KMeansScratch
from src.clustering.pca_scratch import PCAScratch


def evaluate_kmeans_range(
    X_std: np.ndarray,
    k_values: List[int] = [2, 3, 4, 5, 6],
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[int, KMeansScratch]]:
    """
    Evaluates from-scratch K-Means for each K in k_values.
    
    Returns:
        - DataFrame summarizing K, WCSS, convergence iterations, and init check.
        - Dictionary mapping K to the fitted KMeansScratch model.
    """
    results = []
    fitted_models = {}
    
    for k in k_values:
        model = KMeansScratch(
            n_clusters=k,
            max_iter=max_iter,
            tol=1e-4,
            n_init=n_init,
            random_state=random_state
        )
        model.fit(X_std)
        fitted_models[k] = model
        
        # Verify that selected WCSS is <= all other inits for this K
        min_init_wcss = min(model.all_inits_wcss_)
        assert np.isclose(model.wcss_, min_init_wcss), (
            f"K={k}: Best WCSS ({model.wcss_}) does not match minimum init WCSS ({min_init_wcss})"
        )
        
        results.append({
            "K": k,
            "WCSS": model.wcss_,
            "Iterations_Best_Run": model.n_iter_,
            "Min_Init_WCSS": min_init_wcss,
            "Max_Init_WCSS": max(model.all_inits_wcss_),
            "Mean_Init_WCSS": float(np.mean(model.all_inits_wcss_))
        })
        
    summary_df = pd.DataFrame(results)
    return summary_df, fitted_models


def evaluate_pca(
    X_std: np.ndarray,
    feature_names: List[str]
) -> Tuple[PCAScratch, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Fits from-scratch PCA, calculates eigenvalues, explained variance,
    and returns 2-component and 3-component projections.
    
    Returns:
        - Fitted PCAScratch object
        - DataFrame summarizing eigenvalues, EVR, cumulative EVR
        - Z_2d: 2-component projection matrix (N, 2)
        - Z_3d: 3-component projection matrix (N, 3)
    """
    pca = PCAScratch()
    pca.fit(X_std)
    
    # Verify dimensions
    D = len(feature_names)
    assert pca.covariance_matrix_.shape == (D, D), f"Covariance matrix shape mismatch: {pca.covariance_matrix_.shape}"
    assert len(pca.eigenvalues_) == D, f"Eigenvalues length mismatch: {len(pca.eigenvalues_)}"
    assert pca.eigenvectors_.shape == (D, D), f"Eigenvectors shape mismatch: {pca.eigenvectors_.shape}"
    
    # Verify descending ordering: lambda_1 >= lambda_2 >= ...
    is_descending = np.all(np.diff(pca.eigenvalues_) <= 1e-9)
    assert is_descending, "Eigenvalues are not strictly in descending order!"
    
    # Verify explained variance ratio sums to 1.0
    evr_sum = float(np.sum(pca.explained_variance_ratio_))
    assert np.isclose(evr_sum, 1.0, atol=1e-4), f"EVR sum ({evr_sum}) deviates from 1.0"
    
    summary_data = []
    for i in range(D):
        summary_data.append({
            "Principal_Component": f"PC{i+1}",
            "Eigenvalue": float(pca.eigenvalues_[i]),
            "Explained_Variance_Ratio": float(pca.explained_variance_ratio_[i]),
            "Explained_Variance_Pct": float(pca.explained_variance_ratio_[i] * 100.0),
            "Cumulative_Explained_Variance": float(pca.cumulative_explained_variance_[i]),
            "Cumulative_Explained_Variance_Pct": float(pca.cumulative_explained_variance_[i] * 100.0)
        })
    summary_df = pd.DataFrame(summary_data)
    
    # Projections
    Z_2d = pca.transform(X_std, n_components=2)
    Z_3d = pca.transform(X_std, n_components=3)
    
    assert Z_2d.shape == (X_std.shape[0], 2), f"Z_2d shape mismatch: {Z_2d.shape}"
    assert Z_3d.shape == (X_std.shape[0], 3), f"Z_3d shape mismatch: {Z_3d.shape}"
    
    return pca, summary_df, Z_2d, Z_3d
