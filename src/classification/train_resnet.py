"""
Stage 6: Deep Convolutional Neural Network (Bonus: 5 Marks).
ResNet-18 Architecture Custom-Adapted for EuroSAT 64x64 RGB Satellite Imagery:
- Replaces initial 7x7 conv (stride 2) with 3x3 conv (stride 1, padding 1)
- Replaces initial maxpool with Identity to preserve 64x64 spatial resolution
- Preserves full residual skip connections across 4 stages
- Replaces 1000-way FC layer with 10-class linear projection
- Trains with CrossEntropyLoss, AdamW optimizer, and Cosine Annealing learning rate schedule
- Evaluates on untouched test partition and exports probabilities for ROC-AUC
"""

from typing import Dict, Any, Tuple, Optional
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18


class EuroSatResNet18(nn.Module):
    """
    Adapted ResNet-18 tailored for 64x64 EuroSAT images.
    Modifications avoid excessive downsampling in the initial layer.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = resnet18(weights=None)
        # Adapt conv1: 3x3 kernel, stride 1, padding 1 (preserves 64x64 spatial grid)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove initial 3x3 maxpool (stride 2) to maintain spatial fidelity
        base.maxpool = nn.Identity()
        # Adapt final classifier head
        base.fc = nn.Linear(512, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_resnet18_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 10,
    epochs: int = 15,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 5,
    checkpoint_path: str = "outputs/classification/models/best_resnet18.pth",
    curve_plot_path: str = "outputs/classification/plots/resnet18_learning_curves.png",
    device: Optional[torch.device] = None
) -> Tuple[EuroSatResNet18, Dict[str, Any]]:
    """
    Trains adapted ResNet-18 with early stopping, learning rate scheduling,
    and checkpointing of the best model based on validation loss.
    """
    print("\n--- [Stage 6: Deep CNN (ResNet-18) Classifier] ---")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device} (PyTorch {torch.__version__})")
    
    # Optimize CPU multi-threading
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(curve_plot_path), exist_ok=True)
    
    model = EuroSatResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    start_train_time = time.time()
    
    print(f"  Beginning ResNet-18 training for up to {epochs} epochs (Early stopping patience: {patience})...")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += len(labels)
            
        scheduler.step()
        
        epoch_train_loss = train_loss_accum / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss_accum += loss.item() * len(labels)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += len(labels)
                
        epoch_val_loss = val_loss_accum / val_total
        epoch_val_acc = val_correct / val_total
        epoch_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start
        
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(epoch_lr)
        
        print(f"  Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%", flush=True)
        
        # Checkpoint Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    --> Saved best model checkpoint (Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%)", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (no val loss improvement for {patience} epochs).", flush=True)
                break
                
    total_train_time = time.time() - start_train_time
    print(f"  Training completed in {total_train_time:.1f}s ({total_train_time/60:.2f} min).")
    
    # Load Best Model for Evaluation
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # Plot Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Training Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(history["epoch"], history["val_loss"], marker="s", label="Validation Loss", color="#d62728", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epoch", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    axes[0].set_ylabel("Loss", fontsize=10, fontweight="semibold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].plot(history["epoch"], history["train_acc"], marker="o", label="Training Accuracy", color="#1f77b4", linewidth=2)
    axes[1].plot(history["epoch"], history["val_acc"], marker="s", label="Validation Accuracy", color="#2ca02c", linewidth=2)
    axes[1].set_title("Classification Accuracy vs. Epoch", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    axes[1].set_ylabel("Accuracy", fontsize=10, fontweight="semibold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.suptitle("Deep CNN (ResNet-18) Learning Curves on EuroSAT RGB", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    print(f"  Saved learning curves plot to: {curve_plot_path}")
    
    # Evaluate on Untouched Test Partition
    print("  Evaluating best ResNet-18 model on held-out test set (5,400 images)...")
    start_pred_time = time.time()
    
    test_loss_accum = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            
            test_loss_accum += loss.item() * len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    prediction_duration = time.time() - start_pred_time
    
    y_test_pred = np.array(all_preds)
    y_test_prob = np.array(all_probs)
    y_test_true = np.array(all_targets)
    test_acc = float(np.mean(y_test_pred == y_test_true))
    test_loss = test_loss_accum / len(y_test_true)
    
    train_final_acc = float(history["train_acc"][-1])
    val_final_acc = float(max(history["val_acc"]))
    
    print(f"  ResNet-18 Test Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  ResNet-18 Test Loss:        {test_loss:.4f}")
    print(f"  Generalization Gap:         {(train_final_acc - test_acc)*100:.2f}%")
    print(f"  Prediction Time (5.4k img): {prediction_duration:.2f}s ({prediction_duration/len(y_test_true)*1000:.2f}ms/img)")
    
    results = {
        "model_name": "Deep CNN (ResNet-18)",
        "train_accuracy": train_final_acc,
        "val_accuracy": val_final_acc,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "training_time_s": total_train_time,
        "prediction_time_s": prediction_duration,
        "epochs_trained": len(history["epoch"]),
        "history": history,
        "y_pred": y_test_pred,
        "y_prob": y_test_prob
    }
    
    return model, results

