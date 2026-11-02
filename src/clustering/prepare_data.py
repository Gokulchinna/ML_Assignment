"""
Data preparation and standardization pipeline for Gas Turbine dataset (Part 3).
"""

from typing import List, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd


def load_and_preprocess_gas_turbine_data(
    file_paths: List[str],
    features: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Loads all yearly gas turbine datasets, validates consistency, removes exact duplicates,
    and standardizes the continuous features (zero mean, unit variance).
    
    Args:
        file_paths: List of CSV file paths (e.g., gt_2011.csv to gt_2015.csv).
        features: List of continuous feature column names to extract.
        
    Returns:
        Tuple of:
            - Cleaned combined DataFrame
            - Standardized feature matrix X_std (NumPy ndarray)
            - Dictionary of scaling parameters (means, stds) and metadata
    """
    dfs = []
    for fp in file_paths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Gas turbine dataset file not found: {fp}")
        df_year = pd.read_csv(fp)
        df_year.columns = [c.strip() for c in df_year.columns]
        dfs.append(df_year)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    initial_rows = len(combined_df)
    
    # Remove exact duplicates across all columns as identified in data inspection
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    cleaned_rows = len(combined_df)
    duplicates_removed = initial_rows - cleaned_rows
    
    # Validate feature presence
    missing_features = [f for f in features if f not in combined_df.columns]
    if missing_features:
        raise ValueError(f"Features missing from gas turbine dataset: {missing_features}")
        
    X_raw = combined_df[features].to_numpy(dtype=np.float64)
    
    # Manual standardization: (X - mean) / std (NumPy from scratch)
    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0, ddof=0)
    
    # Guard against zero standard deviation
    stds_guarded = np.where(stds == 0, 1.0, stds)
    X_std = (X_raw - means) / stds_guarded
    
    metadata = {
        "initial_rows": initial_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_removed": duplicates_removed,
        "num_features": len(features),
        "features": features,
        "means": means.tolist(),
        "stds": stds.tolist()
    }
    
    return combined_df, X_std, metadata
