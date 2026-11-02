# IIIT Sri City — Machine Learning Assignments (M2026)

**Student Name:** [Your Name]  
**Course:** Machine Learning  
**Deadline:** October 31, 2026  

---

## 1. Explicit Assignment Scope vs. Implementation Choices

The assignment requirements from `ML M2026 Assignment.pdf` mandate standard datasets and specific algorithmic models and evaluation protocols. Below is the mapping:

- **Part 1: Multiclass Classification (10 Marks):**
  - **Required by PDF:** K-Nearest Neighbours with cross-validation, Decision Tree, SVM with RBF kernel; 80/20 train/test split; confusion matrix; accuracy, precision, recall, F1-score, and ROC-AUC; discussion of overfitting and complexity; best classifier identification via ROC curve analysis.
  - **Selected Implementation Dataset:** PriceRunner Product Classification (UCI 837) — 10 product categories.
  - **Deferred Bonus (5 Marks):** Deep Convolutional Neural Network (CNN). Per instructions, this will be an image-based CNN (e.g. Fashion-MNIST / CIFAR) implemented only after the entire required assignment and report are complete.

- **Part 2: Regression (10 Marks):**
  - **Required by PDF:** Linear Regression with gradient descent optimization, Polynomial Regression, Ridge Regression, Lasso Regression; MSE, RMSE, and $R^2$ for every model; demonstration of underfitting and overfitting; cross-validation for regularization parameter ($\alpha$).
  - **Selected Implementation Dataset:** Metro Interstate Traffic Volume (UCI 492) predicting `traffic_volume`.

- **Part 3A: K-Means Clustering (5 Marks):**
  - **Required by PDF:** Implement K-Means from scratch; evaluate $K = 2, 3, 4, 5, 6$; calculate WCSS for each $K$; plot the Elbow curve.
  - **Selected Implementation Dataset:** Gas Turbine CO and NOx Emissions (UCI 551) — 11 continuous sensor features.

- **Part 3B: Principal Component Analysis (5 Marks):**
  - **Required by PDF:** Implement PCA using the covariance matrix and eigenvalue decomposition; calculate eigenvalues and eigenvectors; calculate explained-variance ratios; plot cumulative explained variance; reduce original feature space to 2 and 3 principal components; visualize data using the first 2 principal components.
  - **Selected Implementation Dataset:** Gas Turbine CO and NOx Emissions (UCI 551).

---

## 2. Directory Architecture

```
ml-assignment/
├── README.md                           # Documentation, setup, and execution guide
├── requirements.txt                    # Pinned project dependencies
├── .gitignore                          # Clean repository management
├── config.yaml                         # Central configuration parameters
├── ML_Assignment_Report.pdf            # Comprehensive academic submission report
│
├── data/
│   └── raw/                            # Original unmodifiable UCI datasets
│       ├── classification/             # pricerunner_aggregate.csv
│       ├── regression/                 # Metro_Interstate_Traffic_Volume.csv
│       └── clustering/                 # gt_2011.csv ... gt_2015.csv
│
├── src/
│   ├── common/                         # Shared utilities
│   │   ├── data_validation.py          # Schema, missing values, and split leakage verification
│   │   ├── metrics.py                  # Standard and custom evaluation metrics
│   │   ├── plotting.py                 # Standardized publication figures
│   │   └── reproducibility.py          # Deterministic random seed controller
│   ├── classification/                 # Part 1 modules (KNN, Decision Tree, SVM)
│   ├── regression/                     # Part 2 modules (Scratch GD, Polynomials, Ridge, Lasso)
│   └── clustering/                     # Part 3 modules (Scratch K-Means, Scratch PCA)
│
├── experiments/                        # Experiment runner entrypoints
│   ├── run_classification.py          # Executes Part 1 classification experiments
│   ├── run_regression.py              # Executes Part 2 regression experiments
│   ├── run_clustering.py              # Executes Part 3 clustering/PCA experiments
│   └── run_diagnostics.py             # Executes supplementary model diagnostics
│
├── outputs/                            # Auto-generated experiment outputs
│   ├── classification/ (metrics/, plots/)
│   ├── regression/     (metrics/, plots/)
│   └── clustering/     (metrics/, plots/)
│
└── reports/
    ├── figures/                        # High-resolution figures for report
    ├── results/                        # Generated metric tables and summaries
    └── ML_Assignment_Report.pdf        # Academic submission report
```

---

## 3. Environment Setup & Installation

### Prerequisites
- Python 3.12 (or 3.10+) installed.
- Windows PowerShell / Linux Bash.

### Setup Instructions

```powershell
# 1. Create a dedicated virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 3. Upgrade pip and install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Dataset Information & Placement

The raw datasets are located in `data/raw/`:

1. **Classification:**
   - Source: UCI Machine Learning Repository (ID 837)
   - Path: `data/raw/classification/pricerunner_aggregate.csv`
   - Description: 35,311 products categorized into 10 classes.

2. **Regression:**
   - Source: UCI Machine Learning Repository (ID 492)
   - Path: `data/raw/regression/Metro_Interstate_Traffic_Volume.csv`
   - Description: 48,204 hourly traffic volume observations with weather indicators.

3. **Clustering & PCA:**
   - Source: UCI Machine Learning Repository (ID 551)
   - Path: `data/raw/clustering/gt_2011.csv` through `gt_2015.csv`
   - Description: 36,733 hourly gas turbine operational records with 11 continuous features.

---

## 5. Running Experiments & Building the Academic Report

Each experiment can be executed independently or sequentially from the repository root. A deterministic random seed (`seed: 42`) is set globally:

```powershell
# 1. Run Part 3: Clustering & PCA Pipeline (K-Means & PCA from Scratch)
python experiments/run_clustering.py

# 2. Run Part 2: Regression Pipeline (Scratch GD, Polynomial, Ridge, Lasso)
python experiments/run_regression.py

# 3. Run Part 1: Classification Pipeline (KNN, Decision Tree, Calibrated RBF-SVM)
python experiments/run_classification.py

# 4. Compile the Academic PDF Report (8 pages with embedded figures and tables)
python src/reporting/build_academic_report.py
```

All metrics are automatically exported as CSV tables to `outputs/*/metrics/` and mirrored to `reports/results/`. High-resolution figures are saved to `outputs/*/plots/` and mirrored to `reports/figures/`.

---

## 6. Summary of Key Verified Empirical Findings

### Part 1: Multiclass Classification (PriceRunner Dataset, 35,311 records)
- **Evaluation Split:** 80/20 Stratified train/test split (28,248 train, 7,063 test).
- **KNN ($k=15$, distance-weighted, cosine metric):** Test Accuracy: **93.86%**, Macro F1: **0.9385**, Macro ROC-AUC: **0.9933**. Selected as the **Best Classifier by ROC Analysis**.
- **Decision Tree (Gini, unconstrained):** Test Accuracy: **94.14%**, Macro F1: **0.9414**, Macro ROC-AUC: **0.9859**. Overfitting demonstrated: 241 depth, 874 leaves, 96.76% training accuracy (2.62% generalization gap).
- **Support Vector Machine (RBF kernel, $C=10.0, \gamma=\text{scale}$):** Test Accuracy: **95.30%**, Macro F1: **0.9540**, Macro ROC-AUC: **0.9908**. Delivered the highest overall Accuracy and F1-score across all 10 product categories.

### Part 2: Regression Analysis (Metro Traffic Dataset, 48,204 timestamps)
- **Evaluation Split:** Strict chronological 80/20 split (38,563 train, 9,641 test).
- **Linear Regression (Scratch Batch GD):** 1,000 iterations ($\eta=0.05$), Half-MSE loss, normalized gradients ($\|\nabla J\|_2 = 3.765$). Test MSE: **$1.1711\times 10^6$** ($R^2 = 0.6975$).
- **Sanity Check vs. OLS:** Out-of-sample prediction difference between GD and closed-form OLS is only **3.25 vehicles/hour** (relative difference $<0.1\%$).
- **Polynomial Regression:** Evaluated across degrees 1 to 5. Controlled underfitting and overfitting are demonstrated on an independent subset ($N=300$ continuous records): Degree 1 underfits (high bias, Test MSE = $3.86\times 10^6$), Degree 2 achieves close training fit with comparable test error ($3.87\times 10^6$), Degree 3 begins diverging ($4.13\times 10^6$), and Degree 4 exhibits high-variance overfitting (Test MSE explodes 14-fold to $5.41\times 10^7$). On the full dataset ($N=38,563$), sample size shields degrees 2–4 into a plateau ($R^2 \approx 0.699$), while Degree 5 exhibits severe numerical ill-conditioning and instability ($\kappa = 6.76\times 10^{24} \gg 1/\epsilon_{\text{mach}} \approx 4.5\times 10^{15}$), where normal equations become numerically unstable in double precision.
- **Regularization:** Ridge selected $\alpha=100.0$ ($R^2 = 0.6972$); Lasso selected $\alpha=10.0$ ($R^2 = 0.6961$), successfully zeroing 13 out of 23 features (56.5% sparsity).
- **Rainfall Outlier Treatment:** Capping `rain_1h = 9831.3 mm` at the 99.99th percentile ($29.14\text{ mm}$) does not improve held-out test MSE ($1.1733\times 10^6$ vs $1.1727\times 10^6$, 0.05% difference). However, uncapping inflates feature variance 3,500-fold, flipping the regression slope from $-34.88$ (physically consistent traffic reduction) to $+3.69$ (counter-intuitive positive association). Capping is justified on data-quality and parameter interpretability grounds.

### Part 3: Unsupervised Learning (Gas Turbine Dataset, 36,733 records)
- **K-Means (from Scratch):** Implemented Lloyd's algorithm in pure NumPy. Evaluated $K \in \{2, 3, 4, 5, 6\}$ with 10 random restarts per $K$. WCSS drops from 276,942 ($K=2$) to 213,958 ($K=3$, a 22.74% drop), showing a distinct elbow inflection at $K=3$.
- **PCA (from Scratch):** Implemented via sample covariance $\mathbf{C} = \frac{1}{N-1}\mathbf{X}^T\mathbf{X}$ and `numpy.linalg.eigh()`. Eigenvalues: $\lambda_1 = 5.280$ down to $\lambda_{11} = 0.002$. PC1 explains 48.00% (turbine load/thermal capacity) and PC2 explains 19.90% (ambient climate). A 2-PC projection retains **67.90%** of system variance; 3 PCs retain **77.35%**.

---

## 7. Methodological Limitations

1. **SVM Cross-Validation Subsampling:** Due to quadratic sample complexity $\mathcal{O}(N^2 D)$ of the RBF kernel on 28k sparse text vectors, hyperparameter tuning was conducted on a stratified 10,000-sample training partition with 3-fold CV. The resulting optimal parameters ($C=10.0, \gamma=\text{scale}$) were fitted on the full 28,248 dataset with 3-fold `CalibratedClassifierCV`. Hyperparameters were not reselected using a full-dataset 5-fold pipeline CV.
2. **Gradient Descent Stopping:** Batch GD halted because the 1,000-iteration budget was reached ($|\Delta J| = 0.714 > 10^{-6}$). The smaller weight norm ($\|\mathbf{w}\|_2 = 2,084.7$ vs. $2,126.2$ for OLS) reflects finite-iteration truncation. Early stopping induces spectral shrinkage but is not mathematically equivalent to Ridge regularization.
3. **Polynomial Degree 5 Numerical Ill-Conditioning:** Degree 5 error explosion is an artifact of severe floating-point numerical ill-conditioning and instability ($\kappa = 6.76\times 10^{24}$) rather than classical statistical variance.
4. **Rainfall Outlier Sensitivity:** Capping the $9,831.3\text{ mm}$ reading protects physical parameter interpretability (preventing a sign flip to $+3.69$), but does not reduce held-out test MSE.
