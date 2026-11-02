"""
Feature extraction pipeline for traditional ML models (KNN, Decision Tree, SVM).
Extracts Histogram of Oriented Gradients (HOG) features from EuroSAT RGB images
and caches them to disk for rapid, zero-overhead experimentation.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image
from skimage.feature import hog


def _extract_single_hog(img_path: str, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8),
                        cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """Helper to load a single image and compute its HOG feature vector."""
    with Image.open(img_path) as img:
        arr = np.array(img.convert("RGB"))
    feat = hog(
        arr,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        channel_axis=-1,
        feature_vector=True
    )
    return feat.astype(np.float32)


def extract_hog_features_batch(
    file_paths: List[str],
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    max_workers: Optional[int] = None
) -> np.ndarray:
    """
    Extracts HOG feature vectors for a list of image paths using multiprocessing.
    Returns:
        - 2D numpy array of shape (N, feature_dim).
    """
    total = len(file_paths)
    print(f"  Extracting HOG features for {total} images (workers={max_workers})...")
    start_t = time.time()
    
    # Pre-extract sample to determine exact feature dimension
    sample_feat = _extract_single_hog(file_paths[0], orientations, pixels_per_cell, cells_per_block)
    feat_dim = len(sample_feat)
    
    features = np.empty((total, feat_dim), dtype=np.float32)
    
    # Process sequentially or in pool
    # Threading/multiprocessing: ProcessPool is clean for CPU-bound HOG
    workers = min(os.cpu_count() or 4, 16) if max_workers is None else max_workers
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_single_hog, fp, orientations, pixels_per_cell, cells_per_block): i 
                   for i, fp in enumerate(file_paths)}
        for future in as_completed(futures):
            idx = futures[future]
            features[idx] = future.result()
            
    elapsed = time.time() - start_t
    print(f"  HOG extraction complete in {elapsed:.1f}s. Feature matrix shape: {features.shape}")
    return features


def get_or_extract_hog_features(
    split_meta: Dict[str, Any],
    cache_path: str = "data/interim/classification/eurosat_hog_features.npz",
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Retrieves HOG features from cache if available, or computes and saves them.
    Returns:
        - X_train: Training feature matrix (21,600 x 1,764)
        - y_train: Training labels (21,600,)
        - X_test: Testing feature matrix (5,400 x 1,764)
        - y_test: Testing labels (5,400,)
        - class_names: List of 10 class labels
    """
    if os.path.exists(cache_path) and not force_recompute:
        print(f"[Stage 2] Loading precomputed HOG features from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        class_names = data["class_names"].tolist()
        return X_train, y_train, X_test, y_test, class_names

    print("[Stage 2] Computing HOG features for EuroSAT dataset...")
    file_paths = split_meta["file_paths"]
    labels = np.array(split_meta["labels"], dtype=np.int64)
    class_names = split_meta["class_names"]
    
    train_idx = split_meta["indices"]["train_val"]
    test_idx = split_meta["indices"]["test"]
    
    train_paths = [file_paths[i] for i in train_idx]
    test_paths = [file_paths[i] for i in test_idx]
    
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    
    print(f"Extracting training set HOG ({len(train_paths)} images)...")
    X_train = extract_hog_features_batch(train_paths, orientations, pixels_per_cell, cells_per_block)
    
    print(f"Extracting testing set HOG ({len(test_paths)} images)...")
    X_test = extract_hog_features_batch(test_paths, orientations, pixels_per_cell, cells_per_block)
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=np.array(class_names)
    )
    print(f"[Stage 2 Complete] Saved HOG features cache to: {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.1f} MB)")
    return X_train, y_train, X_test, y_test, class_names

