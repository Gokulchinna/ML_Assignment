"""
Reproducibility utilities for deterministic experiments across NumPy, Python, and Scikit-Learn.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds across Python built-in random, OS hash seed, and NumPy.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    # Optional PyTorch seed handling if PyTorch is installed (e.g. for deferred CNN bonus)
    try:
        # pyrefly: ignore[missing-import]
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: int = 42) -> np.random.Generator:
    """
    Returns an isolated NumPy random number generator for reproducible sampling.
    """
    return np.random.default_rng(seed)
