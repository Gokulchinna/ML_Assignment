"""
Data validation utilities for schema enforcement, data hygiene, and integrity checks.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


def check_schema(df: pd.DataFrame, expected_columns: List[str], dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Validates that a DataFrame contains all expected columns.
    
    Args:
        df: Input DataFrame.
        expected_columns: List of expected column names.
        dataset_name: Name of the dataset for logging.
        
    Returns:
        Dictionary with validation results.
    """
    actual_cols = list(df.columns)
    missing = [c for c in expected_columns if c not in actual_cols]
    extra = [c for c in actual_cols if c not in expected_columns]
    
    is_valid = len(missing) == 0
    return {
        "dataset": dataset_name,
        "is_valid": is_valid,
        "actual_columns": actual_cols,
        "missing_columns": missing,
        "extra_columns": extra,
        "num_rows": len(df),
        "num_cols": len(df.columns)
    }


def check_missing_and_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks missing values per column and counts duplicate rows.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary with missing value counts and duplicate counts.
    """
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    duplicate_rows = int(df.duplicated().sum())
    
    return {
        "total_rows": len(df),
        "total_missing": int(total_missing),
        "missing_per_col": {k: int(v) for k, v in missing_counts.items() if v > 0},
        "duplicate_rows": duplicate_rows
    }


def verify_no_split_leakage(train_indices: Any, test_indices: Any) -> bool:
    """
    Verifies that train and test indices are strictly disjoint.
    
    Args:
        train_indices: Array or index of training instances.
        test_indices: Array or index of testing instances.
        
    Returns:
        True if completely disjoint, raises ValueError otherwise.
    """
    train_set = set(train_indices)
    test_set = set(test_indices)
    overlap = train_set.intersection(test_set)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL DATA LEAKAGE: {len(overlap)} samples overlap between train and test splits!")
    return True
