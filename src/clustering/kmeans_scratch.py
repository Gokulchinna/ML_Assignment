"""
K-Means Clustering implementation from scratch using NumPy.
Strictly implements Lloyd's algorithm with multiple initializations,
best-WCSS retention, and safe empty-cluster handling.
"""

from typing import Tuple, List, Optional
import numpy as np


class KMeansScratch:
    """
    K-Means clustering implemented entirely from scratch using NumPy.
    
    Parameters:
        n_clusters: Number of clusters (K).
        max_iter: Maximum number of iterations per initialization run.
        tol: Relative tolerance for centroid convergence.
        n_init: Number of independent random initializations.
        random_state: Seed for reproducible random centroid initialization.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: Optional[int] = 42
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        
        # Fitted attributes
        self.cluster_centers_: Optional[np.ndarray] = None  # Shape: (K, D)
        self.labels_: Optional[np.ndarray] = None          # Shape: (N,)
        self.wcss_: float = float("inf")                   # Best within-cluster sum of squares
        self.n_iter_: int = 0                              # Iterations in best run
        self.all_inits_wcss_: List[float] = []             # WCSS of each initialization run
        
    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Initializes K centroids using random sampling without replacement.
        """
        n_samples = X.shape[0]
        random_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[random_indices].copy()
        
    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Computes squared Euclidean distance from every sample to every centroid.
        Output shape: (N, K).
        """
        # Distance: ||x - mu||^2 = ||x||^2 - 2 x . mu^T + ||mu||^2
        x_norm = np.sum(X ** 2, axis=1, keepdims=True)            # (N, 1)
        mu_norm = np.sum(centroids ** 2, axis=1, keepdims=True).T # (1, K)
        dot_prod = np.dot(X, centroids.T)                          # (N, K)
        distances = x_norm - 2.0 * dot_prod + mu_norm
        # Clip numerical inaccuracies below zero
        return np.maximum(distances, 0.0)

    def _single_run(
        self,
        X: np.ndarray,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Executes a single run of Lloyd's algorithm.
        Returns: (centroids, labels, wcss, n_iterations)
        """
        n_samples, n_features = X.shape
        centroids = self._init_centroids(X, rng)
        labels = np.zeros(n_samples, dtype=np.int32)
        n_iter = 0
        
        for it in range(1, self.max_iter + 1):
            n_iter = it
            # Step 1: Assignment Step
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            
            # Step 2: Centroid Update Step
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_mask = (labels == k)
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size > 0:
                    new_centroids[k] = np.mean(X[cluster_mask], axis=0)
                else:
                    # Empty Cluster Handling:
                    # Reassign centroid to the sample with the maximum squared distance
                    # from its currently assigned centroid.
                    current_min_dists = np.min(distances, axis=1)
                    farthest_idx = np.argmax(current_min_dists)
                    new_centroids[k] = X[farthest_idx].copy()
                    
            # Step 3: Convergence Check
            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            if shift < self.tol:
                break
                
        # Final WCSS calculation for this run
        final_distances = self._compute_distances(X, centroids)
        sample_min_sq_dists = np.min(final_distances, axis=1)
        wcss = float(np.sum(sample_min_sq_dists))
        
        return centroids, labels, wcss, n_iter

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        """
        Fits K-Means across n_init initializations and retains the solution
        with the minimum WCSS.
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
            
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(f"Cannot fit {self.n_clusters} clusters with only {n_samples} samples.")
            
        master_rng = np.random.default_rng(self.random_state)
        
        best_wcss = float("inf")
        best_centroids = None
        best_labels = None
        best_iter = 0
        all_wcss = []
        
        for init_idx in range(self.n_init):
            # Derive unique seed for each initialization run
            init_seed = master_rng.integers(0, 1_000_000)
            run_rng = np.random.default_rng(init_seed)
            
            centroids, labels, wcss, n_iter = self._single_run(X, run_rng)
            all_wcss.append(wcss)
            
            if wcss < best_wcss:
                best_wcss = wcss
                best_centroids = centroids
                best_labels = labels
                best_iter = n_iter
                
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.wcss_ = best_wcss
        self.n_iter_ = best_iter
        self.all_inits_wcss_ = all_wcss
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assigns each sample in X to the closest centroid.
        """
        if self.cluster_centers_ is None:
            raise ValueError("KMeansScratch instance is not fitted yet.")
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
