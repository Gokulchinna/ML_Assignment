"""
Linear Regression implemented from scratch using Batch Gradient Descent in NumPy.
"""

from typing import Optional, List
import numpy as np


class LinearRegressionGradientDescent:
    """
    Linear Regression optimized using Batch Gradient Descent.
    
    Parameters:
        learning_rate: Gradient descent step size (eta).
        max_iter: Maximum number of gradient descent iterations.
        tol: Convergence tolerance on change in cost function.
        random_state: Random seed for weight initialization.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = 42
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Learned parameters
        self.weights_: Optional[np.ndarray] = None  # Shape: (D,)
        self.bias_: float = 0.0                     # Scalar intercept
        self.cost_history_: List[float] = []        # MSE cost at each iteration
        self.n_iter_: int = 0
        
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Mean Squared Error cost J = (1 / (2 * N)) * sum((y_pred - y_true)^2)."""
        N = len(y_true)
        return float(np.sum((y_pred - y_true) ** 2) / (2.0 * N))
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGradientDescent":
        """
        Fits weights and bias via batch gradient descent.
        
        Args:
            X: Training features array of shape (N, D).
            y: Continuous target array of shape (N,).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.float64)
            
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)
        
        # Initialize weights with small random Gaussian values; bias at target mean
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=D)
        self.bias_ = float(np.mean(y))
        
        self.cost_history_ = []
        prev_cost = float("inf")
        
        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            
            # Forward pass: y_pred = X @ w + b
            y_pred = np.dot(X, self.weights_) + self.bias_
            error = y_pred - y  # (N,)
            
            # Compute current cost
            current_cost = self._compute_cost(y, y_pred)
            self.cost_history_.append(current_cost)
            
            # Check for convergence
            if abs(prev_cost - current_cost) < self.tol:
                break
            prev_cost = current_cost
            
            # Analytical gradients
            # grad_w = (1 / N) * (X^T @ error)
            # grad_b = (1 / N) * sum(error)
            grad_w = np.dot(X.T, error) / float(N)
            grad_b = float(np.sum(error) / float(N))
            
            # Parameter updates
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts continuous target values for feature matrix X."""
        if self.weights_ is None:
            raise ValueError("LinearRegressionGradientDescent is not fitted yet.")
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.weights_) + self.bias_
