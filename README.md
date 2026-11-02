# IIIT Sri City — Machine Learning Assignment (M2026)

**Course:** Machine Learning (M2026)  
**Submission Deliverable:** [`reports/ML_Assignment_Report.pdf`](reports/ML_Assignment_Report.pdf)  

---

## 1. Assignment Overview & Scope

This repository contains the complete, rigorous implementation of the IIIT Sri City Machine Learning Assignment across all three core modules, including the Deep Learning bonus model:

1. **Part 1: Multiclass Classification**
   - **Dataset:** EuroSAT RGB Land Cover Dataset (27,000 Sentinel-2 satellite images, $64 \times 64 \times 3$, 10 land cover classes).
   - **Protocol:** Strict stratified 80/20 train/test partition ($21{,}600$ train, $5{,}400$ unseen test images) with zero data leakage. Spatial features represented using $1{,}764$-D Histogram of Oriented Gradients (HOG).
   - **Algorithms Evaluated:**
     1. *$K$-Nearest Neighbours (KNN):* 5-Fold Stratified Cross-Validation on $K \in \{1, 3, 5, 7, 9, 11\}$ with uniform vs. distance weighting.
     2. *Decision Tree:* 5-Fold Stratified Cross-Validation across Gini vs. Entropy, tree depths $\in \{5, 10, 15, 20\}$, and leaf split controls.
     3. *Support Vector Machine (SVM):* Non-linear RBF kernel with 3-Fold Stratified Cross-Validation on $C \in \{0.1, 1.0, 10.0\}$ and $\gamma \in \{\text{'scale'}, \text{'auto'}, 0.01\}$, fitted on the full training set with calibrated decision probabilities.
     4. *Deep Convolutional Neural Network (ResNet-18, +5 Marks Bonus):* Custom-adapted for $64 \times 64$ images (stride-1 $3 \times 3$ conv, identity pooling, 4 residual stages, $11.17\text{M}$ parameters) trained with AdamW and cosine annealing on NVIDIA RTX 3060 GPU hardware acceleration.
   - **Unified Evaluation:** Multi-class Accuracy, Precision, Recall, Macro F1, Weighted F1, Macro ROC-AUC, 4-panel normalized confusion matrices, comparative One-vs-Rest ROC curve analysis, and Vapnik-Chervonenkis (VC) generalization gap analysis.

2. **Part 2: Regularized Regression & Polynomial Complexity**
   - **Dataset:** Metro Interstate Traffic Volume (UCI 492, 48,204 hourly records) predicting continuous `traffic_volume`.
   - **Algorithms Evaluated:**
     1. *Linear Regression from Scratch:* Batch Gradient Descent ($\eta=0.05$, 1,000 iterations) with normalized gradients verifying numerical convergence against closed-form Ordinary Least Squares (relative deviation $<0.1\%$).
     2. *Polynomial Regression:* Complexity expansion from Degree 1 to 5. Controlled underfitting and high-variance overfitting demonstrated on a subset ($N=300$), while full-dataset stability reveals severe numerical ill-conditioning at Degree 5 ($\kappa = 6.76 \times 10^{24}$).
     3. *Ridge (L2) & Lasso (L1) Regularization:* Time-series cross-validation over regularization penalty paths ($\alpha \in [10^{-3}, 10^3]$), showing 56.5% feature sparsity with Lasso.
     4. *Outlier Sensitivity:* Robust evaluation of rainfall anomaly ($9{,}831.3\text{ mm}$) impact on parameter stability and sign fidelity.

3. **Part 3: Unsupervised Learning from Scratch**
   - **Dataset:** Gas Turbine CO and NOx Telemetry (UCI 551, 36,733 records with 11 continuous sensor measurements).
   - **Part 3A — K-Means Clustering from Scratch (5 Marks):**
     - Pure NumPy implementation of Lloyd's algorithm with Euclidean distance minimization and random restarts.
     - Evaluated $K \in \{2, 3, 4, 5, 6\}$, computed Within-Cluster Sum of Squares (WCSS), plotted the Elbow inflection curve, and evaluated cluster quality using Silhouette analysis ($S=0.3392$ at $K=3$).
   - **Part 3B — Principal Component Analysis from Scratch (5 Marks):**
     - Analytical eigen-decomposition of the sample covariance matrix $\mathbf{C} = \frac{1}{N-1}\mathbf{X}^T\mathbf{X}$ using `numpy.linalg.eigh()`.
     - Calculated eigenvalues, eigenvectors, individual explained variance ratios, and cumulative scree curves ($83.15\%$ total variance retained in top 3 PCs).
     - Visualized 2D and 3D projections combining K-Means cluster partitions on principal component subspaces.

---

## 2. Repository Architecture

```
ML_Assignment/
├── README.md                                  # Comprehensive documentation and execution guide
├── requirements.txt                           # Pinned dependencies (PyTorch, scikit-learn, ReportLab, etc.)
├── .gitignore                                 # Clean version control (ignoring .venv, cache, raw images)
├── config.yaml                                # Centralized experiment configurations and hyperparameters
│
├── data/
│   ├── raw/                                   # Source datasets
│   │   ├── classification/                    # EuroSAT RGB (27,000 images across 10 land cover folders)
│   │   ├── regression/                        # Metro_Interstate_Traffic_Volume.csv
│   │   └── clustering/                        # gt_2011.csv ... gt_2015.csv
│   └── interim/
│       └── classification/
│           └── eurosat_split.json             # Stratified 80/20 train/test split metadata
│
├── src/                                       # Core modular algorithms and pipelines
│   ├── classification/                        # Part 1 modules
│   │   ├── prepare_data.py                    # Dataset download, stratified splitting, PyTorch loaders
│   │   ├── extract_features.py                # HOG spatial feature extraction (1,764-D)
│   │   ├── train_knn.py                       # KNN with 5-fold cross-validation
│   │   ├── train_decision_tree.py             # Decision tree with complexity regularization
│   │   ├── train_svm.py                       # SVM with RBF kernel and probability calibration
│   │   ├── train_resnet.py                    # Custom ResNet-18 model architecture and training loop
│   │   └── evaluate.py                        # Multi-class metrics, confusion matrices, ROC curves
│   ├── regression/                            # Part 2 modules
│   │   ├── prepare_data.py                    # Temporal preprocessing, cyclical time encoding, scaling
│   │   ├── linear_gradient_descent.py         # Batch Gradient Descent from scratch
│   │   ├── polynomial_regression.py           # Polynomial feature expansion and condition number analysis
│   │   ├── ridge_regression.py                # Ridge (L2) regression with CV alpha tuning
│   │   └── lasso_regression.py                # Lasso (L1) regression with sparsity analysis
│   ├── clustering/                            # Part 3 modules
│   │   ├── prepare_data.py                    # Multi-year telemetry aggregation and z-score standardization
│   │   ├── kmeans_scratch.py                  # Lloyd's K-Means algorithm from scratch
│   │   ├── pca_scratch.py                     # PCA covariance eigen-decomposition from scratch
│   │   └── evaluate.py                        # Elbow method, Silhouette score, Scree plots
│   └── common/                                # Shared utilities
│       ├── data_validation.py                 # Split verification and leakage detection
│       ├── metrics.py                         # Evaluation metrics computation
│       ├── plotting.py                        # Standardized publication visual styles
│       └── reproducibility.py                 # Deterministic seed controller
│
├── experiments/                               # Master experiment entrypoints
│   ├── run_classification.py                  # Executes Part 1 (Stages 1-7 end-to-end)
│   ├── run_regression.py                      # Executes Part 2 (Linear GD, Polynomial, Ridge, Lasso)
│   └── run_clustering.py                      # Executes Part 3 (K-Means & PCA from Scratch)
│
├── outputs/                                   # Auto-generated experiment outputs
│   ├── classification/ (metrics/, models/, plots/)
│   ├── regression/     (metrics/, plots/)
│   └── clustering/     (metrics/, plots/)
│
└── reports/
    ├── figures/                               # High-resolution figures embedded in the academic report
    ├── results/                               # Exported CSV metric tables matching experimental runs
    ├── build_fresh_report.py                  # Script to compile the PDF report
    └── ML_Assignment_Report.pdf # Final academic submission report (15 pages)
```

---

## 3. Environment Setup & Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.12).
- CUDA-capable GPU recommended for ResNet-18 (PyTorch with CUDA 12.4), though all code runs on CPU.

### Step-by-Step Setup

```bash
# 1. Clone the repository and enter the directory
cd ML_Assignment

# 2. Create and activate a Python virtual environment
# Windows:
python -m venv .venv
.\.venv\Scripts\activate
# Linux/macOS:
# python3 -m venv .venv
# source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional for GPU acceleration) Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. Running the Experiments

All experiments are fully deterministic and can be reproduced with a single command:

### 1. Multiclass Classification Benchmark (Part 1)
```bash
python experiments/run_classification.py
```
*Executes all 7 stages: Stratified 80/20 partitioning, HOG extraction (1,764-D), 5-fold CV KNN, 5-fold CV Decision Tree, 3-fold CV RBF-SVM on full dataset, ResNet-18 training, and unified ROC/confusion matrix evaluation.*

*To stop after traditional ML models (Stages 1–5):*
```bash
python experiments/run_classification.py --max-stage 5
```

### 2. Regularized Regression Benchmark (Part 2)
```bash
python experiments/run_regression.py
```
*Executes Batch GD Linear Regression, Polynomial Degrees 1–5 with condition number checks, Ridge/Lasso cross-validation paths, and outlier sensitivity tests.*

### 3. Clustering & PCA from Scratch (Part 3)
```bash
python experiments/run_clustering.py
```
*Executes NumPy-only K-Means across $K=2..6$ with Elbow/Silhouette evaluations and sample covariance eigen-decomposition for PCA dimensionality reduction.*

---

## 5. Verified Empirical Results

### Part 1: Classification Benchmark on EuroSAT Test Set ($N=5{,}400$)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC | Generalization Gap | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep CNN (ResNet-18)** | **94.30%** | **0.9411** | **0.9417** | **0.9412** | **0.9977** | **0.17%** | 339.4 s (GPU) |
| **SVM (RBF Kernel)** | 72.70% | 0.7226 | 0.7157 | 0.7181 | 0.9462 | 27.30% | 8,836.8 s (CPU) |
| **KNN (with 5-Fold CV)** | 36.57% | 0.4033 | 0.3421 | 0.2961 | 0.6800 | 63.43% | 48.1 s |
| **Decision Tree (5-Fold CV)** | 32.89% | 0.3289 | 0.3165 | 0.3183 | 0.7053 | 19.00% | 463.1 s |

> **Key Finding:** ResNet-18 is the **outright best classifier** based on ROC curve analysis (Macro ROC-AUC = **0.9977**). Hierarchical convolutional feature extraction with weight decay eliminates the spatial feature limitations of hand-crafted HOG descriptors, yielding a near-zero generalization gap ($0.17\%$).

---

### Part 2: Regression on Interstate Traffic Volume ($N=48{,}204$)

| Model | Train MSE | Train $R^2$ | Test MSE | Test $R^2$ | Details |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear GD (from Scratch)** | $1.183 \times 10^6$ | 0.7017 | $1.171 \times 10^6$ | 0.6975 | Batch GD (1,000 iters, $\eta=0.05$) |
| **Polynomial Degree 1** | $1.182 \times 10^6$ | 0.7019 | $1.173 \times 10^6$ | 0.6969 | 23 linear features |
| **Polynomial Degree 2** | $1.179 \times 10^6$ | 0.7028 | $1.167 \times 10^6$ | 0.6986 | 33 features, mild non-linearity |
| **Polynomial Degree 4** | $1.176 \times 10^6$ | 0.7036 | $1.165 \times 10^6$ | 0.6991 | 88 features |
| **Polynomial Degree 5** | $3.930 \times 10^6$ | 0.0091 | $3.833 \times 10^6$ | 0.0099 | Severe ill-conditioning ($\kappa = 6.76 \times 10^{24}$) |
| **Ridge Regression (L2)** | $1.183 \times 10^6$ | 0.7018 | $1.172 \times 10^6$ | 0.6972 | Best $\alpha=100.0$ via TimeSeriesSplit |
| **Lasso Regression (L1)** | $1.187 \times 10^6$ | 0.7006 | $1.177 \times 10^6$ | 0.6961 | Best $\alpha=10.0$ (13/23 coefficients zeroed) |

---

### Part 3: Unsupervised Learning on Gas Turbine Telemetry ($N=36{,}733$)

- **$K$-Means from Scratch:**
  - Evaluated Lloyd's algorithm over $K \in \{2, 3, 4, 5, 6\}$ with 10 random restarts.
  - WCSS drops from $276{,}942$ ($K=2$) to $213{,}958$ ($K=3$, a $22.74\%$ reduction), exhibiting an elbow at $K=3$.
  - Silhouette coefficient at $K=3$ confirms cohesive, well-separated operational regimes ($S=0.3392$).
- **PCA from Scratch:**
  - Eigen-decomposition of the $11 \times 11$ sample covariance matrix yielded eigenvalues from $\lambda_1 = 5.280$ to $\lambda_{11} = 0.002$.
  - Variance retention: **PC1 = 48.00%**, **PC2 = 19.90%**, **PC3 = 15.25%**.
  - Top 2 PCs capture **67.90%** of variance; top 3 PCs capture **83.15%**.

---

## 6. Academic PDF Report

The comprehensive 15-page academic report is compiled and available at:
- **Report Path:** [`reports/ML_Assignment_Report.pdf`](reports/ML_Assignment.pdf)
