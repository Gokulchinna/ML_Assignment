"""
Principal Component Analysis (PCA) implementation from scratch using NumPy.
Strictly implements the sample covariance matrix and eigenvalue decomposition
using numpy.linalg.eigh().
"""

from typing import Tuple, Optional
import numpy as np


class PCAScratch:
    """
    Principal Component Analysis implemented from scratch via the sample
    covariance matrix and eigenvalue decomposition.
    
    Parameters:
        n_components: Number of principal components to retain (optional).
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        
        # Fitted attributes
        self.mean_: Optional[np.ndarray] = None
        self.covariance_matrix_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_explained_variance_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray) -> "PCAScratch":
        """
        Fits PCA on data matrix X by computing the sample covariance matrix
        and performing eigenvalue decomposition.
        
        Args:
            X: Input matrix of shape (N, D).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
            
        N, D = X.shape
        if N < 2:
            raise ValueError(f"PCA requires at least 2 samples, but got N={N}.")
            
        # Step 1: Center the data (guarantees zero mean)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute Sample Covariance Matrix Sigma = (1 / (N - 1)) * (X_c^T @ X_c)
        # Shape: (D, D)
        self.covariance_matrix_ = np.dot(X_centered.T, X_centered) / (N - 1.0)
        
        # Step 3: Eigenvalue Decomposition of symmetric covariance matrix using eigh
        raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(self.covariance_matrix_)
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order
        # Note: np.linalg.eigh returns eigenvalues in ascending order
        sort_indices = np.argsort(raw_eigenvalues)[::-1]
        
        # Reorder eigenvalues and eigenvector columns
        self.eigenvalues_ = raw_eigenvalues[sort_indices]
        self.eigenvectors_ = raw_eigenvectors[:, sort_indices]
        
        # Clip numerical precision artifacts (small negative eigenvalues close to zero)
        self.eigenvalues_ = np.maximum(self.eigenvalues_, 0.0)
        
        # Step 5: Compute Explained Variance Ratios
        total_variance = np.sum(self.eigenvalues_)
        if total_variance > 0:
            self.explained_variance_ratio_ = self.eigenvalues_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.eigenvalues_)
            
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)
        
        # Step 6: Select components
        if self.n_components is not None:
            self.components_ = self.eigenvectors_[:, :self.n_components]
        else:
            self.components_ = self.eigenvectors_.copy()
            
        return self

    def transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Projects X onto the principal component subspace.
        
        Args:
            X: Input data (N, D).
            n_components: Number of components to project onto. If None, uses self.n_components.
            
        Returns:
            Projected matrix Z of shape (N, n_components).
        """
        if self.eigenvectors_ is None or self.mean_ is None:
            raise ValueError("PCAScratch instance is not fitted yet.")
            
        k = n_components if n_components is not None else self.n_components
        if k is None:
            k = self.eigenvectors_.shape[1]
            
        projection_matrix = self.eigenvectors_[:, :k]
        X_centered = X - self.mean_
        return np.dot(X_centered, projection_matrix)

    def fit_transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Fits PCA and returns the projection onto n_components.
        """
        self.fit(X)
        return self.transform(X, n_components=n_components)
