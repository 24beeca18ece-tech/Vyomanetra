#!/usr/bin/env python3
"""
Module 4 -- train the Concept Bottleneck Model classifier (shallow MLP, per
report Section 3.4) on the 51-target labeled concept sample
(results/cbm_concepts_full.csv).

Missingness handling per results/module4_classifier_design.md: per-concept
binary missingness indicator + median imputation (fit on the TRAINING fold
only, to avoid leakage), standardized, into a plain (non-NaN-native) MLP --
consistent with the report's own "shallow multilayer perceptron" spec, and
with the informative-missingness decision made for the concept data itself.

N=51 is too small for a single held-out test split to be statistically
meaningful, so performance is estimated via stratified 5-fold cross-
validation. Concept importance (for the Section 3.4 interpretability claim)
is permutation importance averaged across the CV folds' HELD-OUT sets (not
computed on training data, which would be optimistically biased).

Usage: python train_cbm_classifier.py
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CONCEPT_COLS = ["C1_depth_ppm", "C2_shape_flatbottom_ratio", "C3_ingress_egress_symmetry",
                "C4_odd_even_rel_diff", "C5_secondary_eclipse_ppm", "C6_centroid_shift_px",
                "C7_phase_fold_coherence"]
MAYBE_MISSING = ["C3_ingress_egress_symmetry", "C4_odd_even_rel_diff",
                  "C5_secondary_eclipse_ppm", "C6_centroid_shift_px", "C7_phase_fold_coherence"]


def build_feature_matrix(df, impute_medians=None):
    """(X, feature_names, medians). If impute_medians is None, compute from
    `df` (training fold); otherwise reuse given medians (held-out fold)."""
    X_parts, names = [], []
    medians = {} if impute_medians is None else impute_medians
    for c in CONCEPT_COLS:
        col = df[c].astype(float)
        if c in MAYBE_MISSING:
            med = col.median(skipna=True) if impute_medians is None else medians[c]
            if impute_medians is None:
                medians[c] = med
            missing = col.isna().astype(float)
            X_parts.append(col.fillna(med).values.reshape(-1, 1)); names.append(c)
            X_parts.append(missing.values.reshape(-1, 1)); names.append(c.split("_")[0] + "_missing")
        else:
            X_parts.append(col.values.reshape(-1, 1)); names.append(c)
    return np.hstack(X_parts), names, medians


def main():
    df = pd.read_csv(os.path.join(RESULTS, "cbm_concepts_full.csv"))
    sample = df[df["label"].isin(["planet", "not_planet"])].reset_index(drop=True)
    y = (sample["label"] == "planet").astype(int).values
    print(f"Training sample: {len(sample)} targets ({y.sum()} planet, {len(y)-y.sum()} not_planet)", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    importances_per_fold = []
    feature_names = None

    for fold, (tr_idx, te_idx) in enumerate(skf.split(sample, y)):
        df_tr, df_te = sample.iloc[tr_idx], sample.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr, feature_names, medians = build_feature_matrix(df_tr)
        X_te, _, _ = build_feature_matrix(df_te, impute_medians=medians)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        clf = MLPClassifier(hidden_layer_sizes=(8,), activation="relu", alpha=1e-2,
                             max_iter=3000, random_state=42)
        clf.fit(X_tr_s, y_tr)

        pred = clf.predict(X_te_s)
        proba = clf.predict_proba(X_te_s)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_te, pred))
        metrics["precision"].append(precision_score(y_te, pred, zero_division=0))
        metrics["recall"].append(recall_score(y_te, pred, zero_division=0))
        metrics["f1"].append(f1_score(y_te, pred, zero_division=0))
        try:
            metrics["roc_auc"].append(roc_auc_score(y_te, proba))
        except ValueError:
            metrics["roc_auc"].append(np.nan)

        pi = permutation_importance(clf, X_te_s, y_te, n_repeats=30, random_state=42, scoring="accuracy")
        importances_per_fold.append(pi.importances_mean)

        print(f"  fold {fold}: n_test={len(te_idx)} acc={metrics['accuracy'][-1]:.3f} "
              f"prec={metrics['precision'][-1]:.3f} rec={metrics['recall'][-1]:.3f} "
              f"f1={metrics['f1'][-1]:.3f} auc={metrics['roc_auc'][-1]:.3f}", flush=True)

    print("\n=== 5-fold CV performance (mean +/- std across folds) ===")
    for k, v in metrics.items():
        v = np.array(v, dtype=float)
        print(f"  {k:10s} {np.nanmean(v):.3f} +/- {np.nanstd(v):.3f}")

    imp_arr = np.array(importances_per_fold)
    imp_mean, imp_std = imp_arr.mean(axis=0), imp_arr.std(axis=0)
    order = np.argsort(imp_mean)[::-1]
    print("\n=== permutation importance (mean decrease in held-out accuracy, avg over 5 folds) ===")
    for i in order:
        print(f"  {feature_names[i]:20s} {imp_mean[i]:+.4f} +/- {imp_std[i]:.4f}")

    # ---- final model on ALL data (deployment + bonus V1828 Aql check) ----
    X_all, feature_names, medians_all = build_feature_matrix(sample)
    scaler_all = StandardScaler().fit(X_all)
    X_all_s = scaler_all.transform(X_all)
    clf_final = MLPClassifier(hidden_layer_sizes=(8,), activation="relu", alpha=1e-2,
                               max_iter=3000, random_state=42)
    clf_final.fit(X_all_s, y)

    v1828 = df[df["target"] == "V1828 Aql"]
    proba_v = None
    if len(v1828):
        Xv, _, _ = build_feature_matrix(v1828, impute_medians=medians_all)
        Xv_s = scaler_all.transform(Xv)
        proba_v = float(clf_final.predict_proba(Xv_s)[0, 1])
        print(f"\nBonus check -- V1828 Aql (known EB, held out of training): "
              f"P(planet) = {proba_v:.4f} (expect low)", flush=True)

    joblib.dump({"model": clf_final, "scaler": scaler_all, "medians": medians_all,
                 "feature_names": feature_names}, os.path.join(RESULTS, "cbm_classifier.joblib"))

    summary = {
        "n_samples": len(sample), "n_planet": int(y.sum()), "n_not_planet": int(len(y) - y.sum()),
        "cv_metrics": {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
                       for k, v in metrics.items()},
        "permutation_importance": {feature_names[i]: {"mean": float(imp_mean[i]), "std": float(imp_std[i])}
                                    for i in range(len(feature_names))},
        "v1828_aql_p_planet": proba_v,
    }
    with open(os.path.join(RESULTS, "cbm_classifier_report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved results/cbm_classifier.joblib and results/cbm_classifier_report.json")


if __name__ == "__main__":
    main()
