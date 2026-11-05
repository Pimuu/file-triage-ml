#!/usr/bin/env python3
"""
End-to-end reproducible pipeline:

    data generation -> feature extraction -> stratified split ->
    5-fold CV on train -> fit -> test metrics -> figures ->
    per-template analysis -> evasion stress test -> multi-seed stability

Usage:  python run_pipeline.py [--n 20000] [--seed 42] [--quick]
All outputs go to data/ and results/. Runtime: well under a minute on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data_generation import generate_dataset
from src.evasion import ATTACKS, apply_attack
from src.evaluate import (compute_metrics, plot_confusion_matrices, plot_entropy_distribution,
                          plot_evasion, plot_feature_importance, plot_roc)
from src.features import extract_features
from src.models import build_models

ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT / "data", ROOT / "results"
FIG = RES / "figures"


def proba_of(model, X):
    return model.predict_proba(X)[:, 1]


def run_once(n: int, seed: int, test_size: float = 0.25):
    """Generate, split and fit all models for one seed. Returns everything needed downstream."""
    df = generate_dataset(n_samples=n, seed=seed)
    X = extract_features(df)
    y = df["label"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size,
                                      stratify=y, random_state=seed)
    models = build_models(seed)
    for m in models.values():
        m.fit(X.iloc[idx_tr], y[idx_tr])
    return df, X, y, idx_tr, idx_te, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="dataset size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="skip multi-seed stability run")
    args = ap.parse_args()

    t0 = time.time()
    for d in (DATA, RES, FIG):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. data
    df, X, y, idx_tr, idx_te, models = run_once(args.n, args.seed)
    df.to_csv(DATA / "synthetic_files.csv", index=False)
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    print(f"[data] {len(df):,} files, malicious rate {y.mean():.3f}, "
          f"train {len(idx_tr):,} / test {len(idx_te):,}, {X.shape[1]} features")
    pd.DataFrame({
        "template": df["template"].value_counts().index,
        "count": df["template"].value_counts().values,
    }).to_csv(RES / "dataset_composition.csv", index=False)

    # ------------------------------------------------ 2. cross-validation (train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_rows = []
    for name, m in build_models(args.seed).items():
        s = cross_validate(m, X_tr, y_tr, cv=cv, scoring=["roc_auc", "f1"], n_jobs=1)
        cv_rows.append({"model": name,
                        "cv_roc_auc_mean": s["test_roc_auc"].mean(), "cv_roc_auc_std": s["test_roc_auc"].std(),
                        "cv_f1_mean": s["test_f1"].mean(), "cv_f1_std": s["test_f1"].std()})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RES / "cv_metrics.csv", index=False, float_format="%.4f")
    print("[cv]\n" + cv_df.round(4).to_string(index=False))

    # ------------------------------------------------------- 3. test metrics
    rows, cms, curves = [], {}, {}
    for name, m in models.items():
        p = proba_of(m, X_te)
        t = time.perf_counter()
        for _ in range(5):
            proba_of(m, X_te)
        infer_us = (time.perf_counter() - t) / 5 / len(X_te) * 1e6
        met = compute_metrics(y_te, p)
        met["inference_us_per_file"] = infer_us
        rows.append({"model": name, **met})
        cms[name] = confusion_matrix(y_te, (p >= 0.5).astype(int), labels=[0, 1])
        if name != "Rule baseline":
            fpr, tpr, _ = roc_curve(y_te, p)
            curves[name] = (fpr, tpr, met["roc_auc"])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RES / "metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame([{"model": k, "tn": v[0, 0], "fp": v[0, 1], "fn": v[1, 0], "tp": v[1, 1]}
                  for k, v in cms.items()]).to_csv(RES / "confusion_matrices.csv", index=False)
    print("[test]\n" + metrics[["model", "accuracy", "precision", "recall", "f1", "roc_auc",
                                "pr_auc", "tpr_at_1pct_fpr"]].round(4).to_string(index=False))

    rule = metrics.set_index("model").loc["Rule baseline"]
    plot_confusion_matrices(cms, FIG / "confusion_matrices.png")
    plot_roc(curves, FIG / "roc_curve.png", rule_point=(rule["fpr"], rule["recall"]))
    plot_entropy_distribution(X, y, FIG / "entropy_distribution.png")

    # ------------------------------------------------- 4. feature importance
    rf, lr = models["Random forest"], models["Logistic regression"]
    perm = permutation_importance(rf, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                  random_state=args.seed, n_jobs=-1)
    imp = pd.DataFrame({
        "rf_permutation_mean": perm.importances_mean,
        "rf_permutation_std": perm.importances_std,
        "rf_impurity": rf.feature_importances_,
        "lr_coef": lr[-1].coef_.ravel(),
    }, index=X.columns).sort_values("rf_permutation_mean", ascending=False)
    imp.index.name = "feature"
    imp.to_csv(RES / "feature_importance.csv", float_format="%.5f")
    plot_feature_importance(imp, FIG / "feature_importance.png")

    # --------------------------------------- 5. per-template recall / FPR
    te = df.iloc[idx_te].copy()
    for name, m in models.items():
        te[name] = (proba_of(m, X_te) >= 0.5).astype(int)
    per_t = te.groupby("template")[list(models)].mean()
    per_t.insert(0, "n_test", te.groupby("template").size())
    per_t.insert(1, "label_rate", te.groupby("template")["label"].mean())
    per_t.to_csv(RES / "per_template_flag_rate.csv", float_format="%.4f")
    print("[per-template flag rate]\n" + per_t.round(3).to_string())

    # ----------------------------------------------------- 6. evasion test
    mal_te = df.iloc[idx_te][df.iloc[idx_te]["label_clean"] == 1]
    ev = {}
    for attack in ATTACKS:
        Xa = extract_features(apply_attack(mal_te, attack, seed=args.seed))
        ev[attack] = {name: float((proba_of(m, Xa) >= 0.5).mean()) for name, m in models.items()}
    ev = pd.DataFrame(ev).T
    ev.index.name = "attack"
    ev.to_csv(RES / "evasion_recall.csv", float_format="%.4f")
    plot_evasion(ev, FIG / "evasion_recall.png")
    print("[evasion recall]\n" + ev.round(3).to_string())

    # ------------------------------------------- 7. multi-seed stability
    if not args.quick:
        stab = []
        for s in range(5):
            _, Xs, ys, _, te_s, ms = run_once(args.n, seed=s)
            for name, m in ms.items():
                met = compute_metrics(ys[te_s], proba_of(m, Xs.iloc[te_s]))
                stab.append({"seed": s, "model": name, **{k: met[k] for k in
                             ("accuracy", "precision", "recall", "f1", "roc_auc")}})
        stab = pd.DataFrame(stab)
        stab.to_csv(RES / "seed_runs.csv", index=False, float_format="%.4f")
        agg = stab.drop(columns="seed").groupby("model", sort=False).agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.to_csv(RES / "seed_stability.csv", float_format="%.4f")
        print("[seed stability, 5 seeds]\n" + agg.round(4).to_string())

    # ------------------------------------------------------------ timing
    t = time.perf_counter()
    extract_features(df.head(5000))
    feat_us = (time.perf_counter() - t) / 5000 * 1e6
    total = time.time() - t0
    json.dump({"feature_extraction_us_per_file": round(feat_us, 2),
               "pipeline_wall_seconds": round(total, 1),
               "n_samples": args.n, "seed": args.seed}, open(RES / "timing.json", "w"), indent=2)
    print(f"[timing] feature extraction {feat_us:.1f} us/file; total pipeline {total:.1f}s")


if __name__ == "__main__":
    main()
