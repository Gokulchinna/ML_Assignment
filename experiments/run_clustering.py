"""
Experiment driver for Stage 1: Part 3 (K-Means and PCA from Scratch)
Executes data standardization, from-scratch PCA, from-scratch K-Means (K=2..6),
validates all mathematical identities, and exports metrics and figures.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to utf-8 if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.common.reproducibility import set_seed
from src.common.plotting import (
    plot_elbow_curve,
    plot_pca_variance,
    plot_pca_2d_scatter
)
from src.clustering.prepare_data import load_and_preprocess_gas_turbine_data
from src.clustering.evaluate import evaluate_kmeans_range, evaluate_pca


def main():
    print("=" * 70)
    print("STAGE 1: PART 3 -- K-MEANS AND PCA EXPERIMENTS (FROM SCRATCH)")
    print("=" * 70)
    
    # 1. Load Configuration
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    seed = config["project"]["seed"]
    set_seed(seed)
    print(f"[Config] Global deterministic seed set to: {seed}")
    
    raw_files = config["data_paths"]["clustering"]["raw_files"]
    features = config["clustering"]["features"]
    k_values = config["clustering"]["k_values"]
    kmeans_cfg = config["clustering"]["kmeans"]
    
    # Ensure output directories exist
    os.makedirs("outputs/clustering/metrics", exist_ok=True)
    os.makedirs("outputs/clustering/plots", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    # 2. Data Preparation & Standardization
    print(f"\n[1/4] Loading and Standardizing Gas Turbine Dataset ({len(raw_files)} annual files)...")
    df_clean, X_std, meta = load_and_preprocess_gas_turbine_data(raw_files, features)
    print(f"  * Total raw samples: {meta['initial_rows']:,}")
    print(f"  * Exact duplicate rows removed: {meta['duplicates_removed']}")
    print(f"  * Cleaned samples: {meta['cleaned_rows']:,} rows x {meta['num_features']} features")
    print(f"  * Feature check: Mean of X_std ~ {np.mean(X_std):.2e}, Std of X_std ~ {np.std(X_std):.2f}")
    
    # Save cleaned interim dataset
    interim_path = config["data_paths"]["clustering"]["interim"]
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)
    df_clean.to_csv(interim_path, index=False)
    print(f"  * Cleaned interim data saved to: {interim_path}")

    # 3. Part 3B: Principal Component Analysis from Scratch
    print(f"\n[2/4] Executing PCA from Scratch via Covariance & Eigenvalue Decomposition...")
    pca_model, pca_summary_df, Z_2d, Z_3d = evaluate_pca(X_std, features)
    
    print("\n--- PCA EIGENVALUES & EXPLAINED VARIANCE SUMMARY ---")
    print(pca_summary_df.to_string(index=False))
    
    # Save PCA metrics
    pca_metrics_out1 = "outputs/clustering/metrics/pca_variance_summary.csv"
    pca_metrics_out2 = "reports/results/pca_variance_summary.csv"
    pca_summary_df.to_csv(pca_metrics_out1, index=False)
    pca_summary_df.to_csv(pca_metrics_out2, index=False)
    
    # Save Eigenvectors matrix
    eigenvec_df = pd.DataFrame(
        pca_model.eigenvectors_,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))]
    )
    eigenvec_out1 = "outputs/clustering/metrics/pca_eigenvectors.csv"
    eigenvec_out2 = "reports/results/pca_eigenvectors.csv"
    eigenvec_df.to_csv(eigenvec_out1)
    eigenvec_df.to_csv(eigenvec_out2)
    print(f"  * Saved PCA variance summary to: {pca_metrics_out1} and {pca_metrics_out2}")
    print(f"  * Saved PCA eigenvectors table to: {eigenvec_out1}")
    
    # Plot PCA Explained Variance
    pca_plot_out1 = "outputs/clustering/plots/pca_explained_variance.png"
    pca_plot_out2 = "reports/figures/pca_explained_variance.png"
    plot_pca_variance(pca_model.explained_variance_ratio_, save_path=pca_plot_out1)
    plot_pca_variance(pca_model.explained_variance_ratio_, save_path=pca_plot_out2)
    print(f"  * Saved PCA variance plot to: {pca_plot_out1}")
    
    # Plot PC1 vs PC2 2D Projection
    pca_scatter_out1 = "outputs/clustering/plots/pca_2d_projection.png"
    pca_scatter_out2 = "reports/figures/pca_2d_projection.png"
    # Color by Turbine Energy Yield (TEY) for clear physical interpretation
    tey_col = df_clean["TEY"].to_numpy()
    plot_pca_2d_scatter(
        Z_2d,
        color_by=tey_col,
        color_label="Turbine Energy Yield (TEY in MWH)",
        title="PCA 2D Projection: PC1 vs. PC2 (Gas Turbine Data)",
        save_path=pca_scatter_out1
    )
    plot_pca_2d_scatter(
        Z_2d,
        color_by=tey_col,
        color_label="Turbine Energy Yield (TEY in MWH)",
        title="PCA 2D Projection: PC1 vs. PC2 (Gas Turbine Data)",
        save_path=pca_scatter_out2
    )
    print(f"  * Saved PCA 2D projection scatter to: {pca_scatter_out1}")
    
    # 4. Part 3A: K-Means Clustering from Scratch
    print(f"\n[3/4] Executing K-Means from Scratch across K = {k_values}...")
    print(f"  * Using {kmeans_cfg['n_init']} random initializations per K, max_iter={kmeans_cfg['max_iter']}")
    kmeans_summary_df, fitted_kmeans = evaluate_kmeans_range(
        X_std,
        k_values=k_values,
        n_init=kmeans_cfg["n_init"],
        max_iter=kmeans_cfg["max_iter"],
        random_state=seed
    )
    
    print("\n--- K-MEANS WCSS & CONVERGENCE SUMMARY ---")
    print(kmeans_summary_df.to_string(index=False))
    
    # Save K-Means WCSS metrics
    kmeans_metrics_out1 = "outputs/clustering/metrics/kmeans_wcss_summary.csv"
    kmeans_metrics_out2 = "reports/results/kmeans_wcss_summary.csv"
    kmeans_summary_df.to_csv(kmeans_metrics_out1, index=False)
    kmeans_summary_df.to_csv(kmeans_metrics_out2, index=False)
    print(f"  * Saved K-Means WCSS summary to: {kmeans_metrics_out1} and {kmeans_metrics_out2}")
    
    # Plot Elbow Curve
    elbow_plot_out1 = "outputs/clustering/plots/kmeans_elbow_curve.png"
    elbow_plot_out2 = "reports/figures/kmeans_elbow_curve.png"
    plot_elbow_curve(
        k_values=kmeans_summary_df["K"].tolist(),
        wcss_values=kmeans_summary_df["WCSS"].tolist(),
        save_path=elbow_plot_out1
    )
    plot_elbow_curve(
        k_values=kmeans_summary_df["K"].tolist(),
        wcss_values=kmeans_summary_df["WCSS"].tolist(),
        save_path=elbow_plot_out2
    )
    print(f"  * Saved K-Means Elbow Curve to: {elbow_plot_out1}")
    
    # Plot 2D PCA projection colored by K-Means Cluster Labels (K=3)
    k_demo = 3
    demo_labels = fitted_kmeans[k_demo].labels_
    kmeans_pca_scatter1 = "outputs/clustering/plots/pca_2d_kmeans_clusters.png"
    kmeans_pca_scatter2 = "reports/figures/pca_2d_kmeans_clusters.png"
    plot_pca_2d_scatter(
        Z_2d,
        color_by=demo_labels,
        color_label=f"K-Means Cluster Assignment (K={k_demo})",
        title=f"PCA 2D Projection with K-Means (K={k_demo}) Cluster Assignments",
        save_path=kmeans_pca_scatter1
    )
    plot_pca_2d_scatter(
        Z_2d,
        color_by=demo_labels,
        color_label=f"K-Means Cluster Assignment (K={k_demo})",
        title=f"PCA 2D Projection with K-Means (K={k_demo}) Cluster Assignments",
        save_path=kmeans_pca_scatter2
    )
    print(f"  * Saved PCA 2D scatter with K-Means labels to: {kmeans_pca_scatter1}")

    # 5. Final Mathematical Validations Verification
    print("\n[4/4] Verifying Final Mathematical and Computational Invariants...")
    # Validation 1: Covariance shape
    assert pca_model.covariance_matrix_.shape == (11, 11)
    print("  [PASS] Covariance matrix dimensions: Exactly (11, 11)")
    
    # Validation 2: Eigenvalues ordering
    diffs = np.diff(pca_model.eigenvalues_)
    assert np.all(diffs <= 1e-9)
    print("  [PASS] Eigenvalues sorting: Strictly descending order verified")
    
    # Validation 3: EVR Sum
    evr_sum = np.sum(pca_model.explained_variance_ratio_)
    assert np.isclose(evr_sum, 1.0, atol=1e-4)
    print(f"  [PASS] Explained Variance Ratio Sum: {evr_sum:.6f} ~ 1.0")
    
    # Validation 4: 2D & 3D Projections
    assert Z_2d.shape == (36726, 2)
    assert Z_3d.shape == (36726, 3)
    print(f"  [PASS] Projection dimensions: Z_2d is (36726, 2), Z_3d is (36726, 3)")
    
    # Validation 5: Finite WCSS and monotonic selection
    assert np.all(np.isfinite(kmeans_summary_df["WCSS"]))
    for k in k_values:
        row = kmeans_summary_df[kmeans_summary_df["K"] == k].iloc[0]
        assert row["WCSS"] <= row["Mean_Init_WCSS"] + 1e-5, f"K={k}: WCSS {row['WCSS']} > Mean {row['Mean_Init_WCSS']}"
        assert np.isclose(row["WCSS"], row["Min_Init_WCSS"], atol=1e-5), f"K={k}: WCSS not matching min init"
    print("  [PASS] K-Means selection: For all K in [2..6], retained WCSS is strictly minimal across all inits")
    
    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETED SUCCESSFULLY: ALL METRICS & FIGURES EXPORTED")
    print("=" * 70)


if __name__ == "__main__":
    main()
