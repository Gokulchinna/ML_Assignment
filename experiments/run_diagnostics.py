"""
Supplementary diagnostics for the report (run from the project root).

Writes small CSV tables to outputs/diagnostics/ so that every number quoted in the
report that is not already in outputs/*/metrics/ can be regenerated:

  Part 1: KNN / Decision Tree / SVM cross-validation tables, Decision Tree depth sweep
  Part 2: Ridge / Lasso CV curves, Lasso coefficients, underfitting baselines,
          controlled N=300 polynomial experiment, polynomial condition numbers,
          Gradient Descent vs closed-form OLS, rainfall-outlier capping test
  Part 3: K=1 baseline, extended K-Means table (drop, init spread, silhouette), K=3 profile

Usage:  python experiments/run_diagnostics.py
"""

import os
import sys
import types

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.classification.prepare_data import prepare_classification_data
from src.classification.train_decision_tree import train_decision_tree_classifier
from src.classification.train_knn import train_knn_classifier
from src.clustering.kmeans_scratch import KMeansScratch
from src.clustering.prepare_data import load_and_preprocess_gas_turbine_data
from src.common.metrics import compute_regression_metrics
from src.common.reproducibility import set_seed
from src.regression.lasso_regression import train_lasso_regression
from src.regression.linear_gradient_descent import LinearRegressionGradientDescent
from src.regression.prepare_data import prepare_traffic_data
from src.regression.ridge_regression import train_ridge_regression

OUT = "outputs/diagnostics"


def save(df, name):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"  saved {OUT}/{name}")


def part1(cfg, seed):
    print("\n[Part 1] classification diagnostics")
    X_tr, y_tr, X_te, y_te, classes, _, _ = prepare_classification_data(
        cfg["data_paths"]["classification"]["raw"], cfg["classification"]["test_size"],
        cfg["classification"]["tfidf"]["max_features"], seed)

    # Class counts
    df = pd.read_csv(cfg["data_paths"]["classification"]["raw"])
    df.columns = [c.strip() for c in df.columns]
    counts = df["Category Label"].astype(str).str.strip().value_counts().rename_axis("Category").reset_index(name="Records")
    save(counts, "class_counts.csv")

    # Decision tree: full CV grid + depth sweep
    dt_cfg = cfg["classification"]["decision_tree"]
    dt, dt_res = train_decision_tree_classifier(
        X_tr, y_tr, X_te, dt_cfg["max_depth_grid"], dt_cfg["min_samples_split_grid"],
        dt_cfg["criteria"], dt_cfg["cv_folds"], seed)
    save(dt_res["cv_results_df"], "dt_cv_results.csv")

    rows = []
    for depth, split in ((10, 2), (15, 10), (20, 2), (30, 2), (None, 2)):
        m = DecisionTreeClassifier(max_depth=depth, min_samples_split=split, random_state=seed).fit(X_tr, y_tr)
        rows.append({
            "max_depth": "None" if depth is None else depth, "min_samples_split": split,
            "tree_depth": m.get_depth(), "leaves": m.get_n_leaves(),
            "train_acc": float((m.predict(X_tr) == y_tr).mean()),
            "test_acc": float((m.predict(X_te) == y_te).mean())})
    save(pd.DataFrame(rows), "dt_depth_sweep.csv")

    # KNN CV grid
    knn_cfg = cfg["classification"]["knn"]
    _, knn_res = train_knn_classifier(X_tr, y_tr, X_te, knn_cfg["n_neighbors_grid"], knn_cfg["weights"],
                                      knn_cfg["cv_folds"], seed)
    save(knn_res["cv_results_df"], "knn_cv_results.csv")

    # SVM CV grid (same 10,000-sample stratified subset and 3-fold CV as train_svm.py)
    svm_cfg = cfg["classification"]["svm"]
    X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=svm_cfg["benchmark_subsample"],
                                          stratify=y_tr, random_state=seed)
    grid = GridSearchCV(SVC(kernel="rbf", random_state=seed),
                        {"C": svm_cfg["c_grid"], "gamma": svm_cfg["gamma_grid"]},
                        cv=StratifiedKFold(svm_cfg["cv_folds"], shuffle=True, random_state=seed),
                        scoring="accuracy", n_jobs=-1).fit(X_sub, y_sub)
    svm_df = pd.DataFrame(grid.cv_results_)[["param_C", "param_gamma", "mean_test_score", "std_test_score", "rank_test_score"]]
    save(svm_df.sort_values("rank_test_score"), "svm_cv_results.csv")


def part2(cfg, seed):
    print("\n[Part 2] regression diagnostics")
    raw = cfg["data_paths"]["regression"]["raw"]
    X_tr, y_tr, X_te, y_te, names, cont, _ = prepare_traffic_data(raw, cfg["regression"]["test_size"], seed)
    gd_cfg = cfg["regression"]["gradient_descent"]

    # Gradient descent vs closed-form OLS
    gd = LinearRegressionGradientDescent(gd_cfg["learning_rate"], gd_cfg["max_iters"], float(gd_cfg["tolerance"]), seed).fit(X_tr, y_tr)
    ols = LinearRegression().fit(X_tr, y_tr)
    p_gd, p_ols = gd.predict(X_te), ols.predict(X_te)
    err = gd.predict(X_tr) - y_tr
    grad = np.r_[X_tr.T @ err / len(y_tr), err.mean()]
    gm_tr, gm_te = compute_regression_metrics(y_tr, gd.predict(X_tr)), compute_regression_metrics(y_te, p_gd)
    om_tr, om_te = compute_regression_metrics(y_tr, ols.predict(X_tr)), compute_regression_metrics(y_te, p_ols)
    diff = np.abs(p_gd - p_ols)
    save(pd.DataFrame([
        ("GD iterations run", gd.n_iter_), ("GD final |delta J|", abs(gd.cost_history_[-1] - gd.cost_history_[-2])),
        ("GD gradient norm (w and b)", np.linalg.norm(grad)),
        ("GD train MSE", gm_tr["mse"]), ("OLS train MSE", om_tr["mse"]),
        ("GD test MSE", gm_te["mse"]), ("OLS test MSE", om_te["mse"]),
        ("GD test R2", gm_te["r2"]), ("OLS test R2", om_te["r2"]),
        ("Mean |GD - OLS| test prediction (veh/h)", diff.mean()), ("Median |GD - OLS| (veh/h)", np.median(diff)),
        ("Mean |GD - OLS| / mean test traffic", diff.mean() / y_te.mean()),
        ("||w|| GD", np.linalg.norm(gd.weights_)), ("||w|| OLS", np.linalg.norm(ols.coef_)),
    ], columns=["Quantity", "Value"]), "gd_vs_ols.csv")

    # Underfitting baselines on the full training set
    base = []
    const = np.full_like(y_te, y_tr.mean(), dtype=float)
    base.append(("Constant (training mean)", 1, np.mean((y_tr - y_tr.mean()) ** 2), np.mean((y_te - const) ** 2)))
    m4 = LinearRegression().fit(X_tr[:, cont], y_tr)
    base.append(("Linear, 4 continuous features", 4,
                 np.mean((m4.predict(X_tr[:, cont]) - y_tr) ** 2), np.mean((m4.predict(X_te[:, cont]) - y_te) ** 2)))
    base.append(("Linear, all 23 features", 23, om_tr["mse"], om_te["mse"]))
    b = pd.DataFrame(base, columns=["Model", "Features", "Train_MSE", "Test_MSE"])
    b["Test_R2"] = 1 - b["Test_MSE"] / np.mean((y_te - y_te.mean()) ** 2)
    save(b, "underfitting_baselines.csv")

    # Controlled N=300 polynomial experiment on the 4 continuous features
    idx = np.random.default_rng(seed).choice(len(y_tr), 300, replace=False)
    rows = []
    for d in (1, 2, 3, 4):
        P = PolynomialFeatures(degree=d, include_bias=True)
        A, B = P.fit_transform(X_tr[idx][:, cont]), P.transform(X_te[:, cont])
        m = LinearRegression(fit_intercept=False).fit(A, y_tr[idx])
        rows.append({"Degree": d, "Parameters": A.shape[1],
                     "Train_MSE": float(np.mean((m.predict(A) - y_tr[idx]) ** 2)),
                     "Test_MSE": float(np.mean((m.predict(B) - y_te) ** 2))})
    save(pd.DataFrame(rows), "poly_n300.csv")

    # Condition numbers of the full-data polynomial design matrices (no intercept column)
    non = [i for i in range(X_tr.shape[1]) if i not in cont]
    rows = []
    for d in range(1, 6):
        A = np.hstack([PolynomialFeatures(degree=d, include_bias=False).fit_transform(X_tr[:, cont]), X_tr[:, non]])
        s = np.linalg.svd(A, compute_uv=False)
        rows.append({"Degree": d, "Features": A.shape[1], "Condition_number": float(s[0] / s[-1]), "Sigma_min": float(s[-1])})
    save(pd.DataFrame(rows), "poly_condition_numbers.csv")

    # Ridge / Lasso CV curves and Lasso coefficients
    _, rr = train_ridge_regression(X_tr, y_tr, X_te, y_te, cfg["regression"]["ridge"]["alphas"], cfg["regression"]["cv_splits"], seed)
    save(pd.DataFrame(rr["cv_results"]), "ridge_cv.csv")
    lm, lr = train_lasso_regression(X_tr, y_tr, X_te, y_te, cfg["regression"]["lasso"]["alphas"], cfg["regression"]["cv_splits"], 5000, seed)
    save(pd.DataFrame(lr["cv_results"]), "lasso_cv.csv")
    save(pd.DataFrame({"Feature": names, "Lasso_coefficient": lm.coef_, "OLS_coefficient": ols.coef_}), "coefficients.csv")

    # Rainfall outlier: capped (project default) vs uncapped
    src = open("src/regression/prepare_data.py", encoding="utf-8").read()
    src_unc = src.replace('train_df["rain_1h"] = np.minimum(train_df["rain_1h"], rain_cap)', "pass") \
                 .replace('test_df["rain_1h"] = np.minimum(test_df["rain_1h"], rain_cap)', "pass")
    assert src_unc != src
    mod = types.ModuleType("prepare_uncapped")
    exec(compile(src_unc, "prepare_uncapped", "exec"), mod.__dict__)
    Xu_tr, yu_tr, Xu_te, yu_te, _, _, _ = mod.prepare_traffic_data(raw, cfg["regression"]["test_size"], seed)
    ou = LinearRegression().fit(Xu_tr, yu_tr)
    ri = names.index("rain_1h")
    rain = pd.read_csv(raw, keep_default_na=False)["rain_1h"].sort_values()
    save(pd.DataFrame([
        ("Largest rain_1h in data (mm)", rain.iloc[-1]), ("Second largest rain_1h (mm)", rain.iloc[-2]),
        ("Records above 100 mm", int((rain > 100).sum())),
        ("Test MSE, capped", np.mean((ols.predict(X_te) - y_te) ** 2)), ("Test MSE, uncapped", np.mean((ou.predict(Xu_te) - yu_te) ** 2)),
        ("rain_1h coefficient (per std), capped", ols.coef_[ri]), ("rain_1h coefficient (per std), uncapped", ou.coef_[ri]),
    ], columns=["Quantity", "Value"]), "rain_capping.csv")


def part3(cfg, seed):
    print("\n[Part 3] clustering diagnostics")
    df, X, _ = load_and_preprocess_gas_turbine_data(cfg["data_paths"]["clustering"]["raw_files"], cfg["clustering"]["features"])
    km_cfg = cfg["clustering"]["kmeans"]
    idx = np.random.default_rng(0).choice(len(X), 8000, replace=False)
    prev = wcss1 = float((X ** 2).sum())
    rows = [{"K": 1, "WCSS": wcss1, "Drop_pct": np.nan, "Init_std": np.nan, "Silhouette": np.nan}]
    for k in cfg["clustering"]["k_values"]:
        km = KMeansScratch(n_clusters=k, max_iter=km_cfg["max_iter"], tol=float(km_cfg["tol"]), n_init=km_cfg["n_init"], random_state=seed).fit(X)
        rows.append({"K": k, "WCSS": km.wcss_, "Drop_pct": 100 * (prev - km.wcss_) / prev,
                     "Init_std": float(np.std(km.all_inits_wcss_)),
                     "Silhouette": float(silhouette_score(X[idx], km.labels_[idx]))})
        prev = km.wcss_
        if k == 3:
            prof = df.assign(Cluster=km.labels_).groupby("Cluster")[["TEY", "CDP", "TIT", "AT"]].mean().reset_index()
            prof.insert(1, "Records", np.bincount(km.labels_))
            save(prof, "kmeans_k3_profile.csv")
    out = pd.DataFrame(rows)
    out["N"] = len(X)
    save(out, "kmeans_extended.csv")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    seed = cfg["project"]["seed"]
    set_seed(seed)
    part3(cfg, seed)
    part2(cfg, seed)
    part1(cfg, seed)
    print("\nDiagnostics complete.")
