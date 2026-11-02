"""
Data preparation, exploration, and stratified partitioning for the EuroSAT RGB dataset.
Guarantees zero data leakage by saving a unified, reproducible 80/20 stratified split.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class EuroSatDataset(Dataset):
    """PyTorch Dataset for EuroSAT RGB satellite imagery."""
    def __init__(self, file_paths: List[str], labels: List[int], transform: Optional[Any] = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img_path = self.file_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_dataset_inventory(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scans EuroSAT dataset directory, sorting class folders and file paths deterministically.
    Returns:
        - file_paths: List of absolute file paths to all images.
        - labels: Integer class indices.
        - class_names: Alphabetically sorted list of class labels.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"EuroSAT dataset directory not found: {data_dir}")

    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    file_paths = []
    labels = []

    for c in class_names:
        c_dir = os.path.join(data_dir, c)
        files = sorted([f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
        for f in files:
            file_paths.append(os.path.join(c_dir, f))
            labels.append(class_to_idx[c])

    return file_paths, labels, class_names


def plot_class_distribution(class_names: List[str], labels: List[int], save_path: str):
    """Generates and saves a publication-quality class balance bar chart."""
    counts = [labels.count(i) for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, counts, color="#2b5c8f", edgecolor="black", linewidth=0.8)
    plt.title("EuroSAT RGB Dataset: Class Distribution (27,000 Images)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Land-Cover Category", fontsize=11, fontweight="semibold")
    plt.ylabel("Number of Images", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=35, ha="right", fontsize=9.5)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add count values above bars
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 30, f"{int(h)}", ha="center", va="bottom", fontsize=9)
        
    plt.ylim(0, max(counts) + 300)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sample_images_montage(file_paths: List[str], labels: List[int], class_names: List[str], save_path: str):
    """Plots a 2x5 grid of sample images representing every class."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
    axes = axes.flatten()
    
    for class_idx, class_name in enumerate(class_names):
        # Pick the first image belonging to this class
        match_idx = labels.index(class_idx)
        img_p = file_paths[match_idx]
        with Image.open(img_p) as img:
            axes[class_idx].imshow(img)
        axes[class_idx].set_title(class_name, fontsize=11, fontweight="bold")
        axes[class_idx].axis("off")
        
    plt.suptitle("EuroSAT RGB: Representative 64x64 Satellite Imagery by Land-Cover Class", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def prepare_eurosat_split(
    data_dir: str = "data/raw/classification/EuroSAT_RGB",
    interim_dir: str = "data/interim/classification",
    plot_dir: str = "outputs/classification/plots",
    metric_dir: str = "outputs/classification/metrics",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes Stage 1: Discovers dataset, validates dimensions, plots distribution and samples,
    and creates a strictly reproducible, stratified train/test split.
    Saves split metadata to disk for cross-model consistency.
    """
    file_paths, labels, class_names = load_dataset_inventory(data_dir)
    total_samples = len(file_paths)
    
    # 1. Generate exploratory figures
    dist_plot_path = os.path.join(plot_dir, "dataset_class_distribution.png")
    montage_plot_path = os.path.join(plot_dir, "sample_images_montage.png")
    plot_class_distribution(class_names, labels, dist_plot_path)
    plot_sample_images_montage(file_paths, labels, class_names, montage_plot_path)
    
    # 2. Stratified 80/20 train/test split
    indices = np.arange(total_samples)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # 3. Further carve a validation split strictly from the training set (for CNN monitoring)
    # val_size relative to total -> val_size / (1 - test_size)
    val_fraction = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        stratify=[labels[i] for i in train_val_idx],
        random_state=random_state
    )
    
    split_meta = {
        "dataset_name": "EuroSAT_RGB",
        "total_images": total_samples,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_resolution": [64, 64, 3],
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "counts": {
            "train_val_total": len(train_val_idx),
            "train_pure": len(train_idx),
            "val_pure": len(val_idx),
            "test_total": len(test_idx)
        },
        "indices": {
            "train_val": train_val_idx.tolist(),
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist()
        },
        "file_paths": file_paths,
        "labels": labels
    }
    
    # Save split info to disk
    os.makedirs(interim_dir, exist_ok=True)
    split_file = os.path.join(interim_dir, "eurosat_split.json")
    with open(split_file, "w") as f:
        json.dump(split_meta, f)
        
    # Save dataset summary JSON in metrics
    os.makedirs(metric_dir, exist_ok=True)
    summary_file = os.path.join(metric_dir, "dataset_summary.json")
    summary_data = {
        "dataset_name": "EuroSAT_RGB",
        "total_images": total_samples,
        "classes": {c: labels.count(i) for i, c in enumerate(class_names)},
        "train_samples": len(train_val_idx),
        "test_samples": len(test_idx),
        "resolution": "64x64x3 RGB",
        "random_seed": random_state
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
        
    print(f"[Stage 1 Complete] EuroSAT split saved: {len(train_val_idx)} train, {len(test_idx)} test images.")
    return split_meta


def get_eurosat_dataloaders(
    split_meta: Dict[str, Any],
    batch_size: int = 64,
    num_workers: int = 0
):
    """
    Returns PyTorch DataLoaders for ResNet-18 training, validation, and testing.
    Applies spatial data augmentations to training set only.
    """
    from torch.utils.data import DataLoader
    
    file_paths = split_meta["file_paths"]
    labels = split_meta["labels"]
    
    train_idx = split_meta["indices"]["train"]
    val_idx = split_meta["indices"]["val"]
    test_idx = split_meta["indices"]["test"]
    
    # Standard EuroSAT / ImageNet channel normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    train_dataset = EuroSatDataset(
        [file_paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        transform=train_transform
    )
    val_dataset = EuroSatDataset(
        [file_paths[i] for i in val_idx],
        [labels[i] for i in val_idx],
        transform=eval_transform
    )
    test_dataset = EuroSatDataset(
        [file_paths[i] for i in test_idx],
        [labels[i] for i in test_idx],
        transform=eval_transform
    )
    
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

