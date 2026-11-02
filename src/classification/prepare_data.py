"""
Data preparation, stratification, and TF-IDF vectorization for PriceRunner Product Classification (Part 1).
Strictly fits text vectorizers on the training split only to prevent data leakage.
"""

from typing import Tuple, List, Dict, Any
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_classification_data(
    file_path: str,
    test_size: float = 0.20,
    max_features: int = 5000,
    random_state: int = 42
) -> Tuple[Any, np.ndarray, Any, np.ndarray, List[str], TfidfVectorizer, Dict[str, Any]]:
    """
    Loads raw PriceRunner aggregate dataset, performs 80/20 stratified split,
    and fits TF-IDF vectorizer strictly on training product titles.
    
    Returns:
        - X_train, y_train: Sparse TF-IDF training matrix and target labels.
        - X_test, y_test: Sparse TF-IDF testing matrix and target labels.
        - class_names: Sorted list of 10 unique product category labels.
        - vectorizer: Fitted TfidfVectorizer instance.
        - metadata: Dictionary of dataset properties, class frequencies, and split sizes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Classification dataset not found: {file_path}")
        
    df = pd.read_csv(file_path)
    # Strip leading/trailing whitespace from header names
    df.columns = [c.strip() for c in df.columns]
    
    text_col = "Product Title"
    target_col = "Category Label"
    
    # Ensure text is string and fill any missing strings
    df[text_col] = df[text_col].fillna("").astype(str)
    df[target_col] = df[target_col].astype(str).str.strip()
    
    class_names = sorted(df[target_col].unique().tolist())
    num_classes = len(class_names)
    
    # 80/20 Stratified Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )
    
    # Sublinear TF-IDF Vectorization: fit strictly on training titles
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True
    )
    
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    
    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    
    # Stratification distribution check
    train_dist = train_df[target_col].value_counts(normalize=True).to_dict()
    test_dist = test_df[target_col].value_counts(normalize=True).to_dict()
    
    metadata = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "num_classes": num_classes,
        "class_names": class_names,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_class_proportions": train_dist,
        "test_class_proportions": test_dist
    }
    
    return X_train, y_train, X_test, y_test, class_names, vectorizer, metadata
