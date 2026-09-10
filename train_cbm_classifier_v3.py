#!/usr/bin/env python3
"""
Module 4 -- retrain the CBM classifier on the scaled-up results/cbm_concepts_v3.csv
(compute_cbm_concepts_v3.py), for the DSAA-track goal of reporting a real,
adequately-powered classifier result rather than the diagnosed small-N
limitation reported for the 80-target v2 sample (see
paper_tai/main.tex, Section IV-B and results/module4_classifier_design.md).

This is deliberately NOT a rewrite of the v2 diagnosis -- it is the same
feature construction (7 concepts + 5 missingness indicators,
missingness-as-signal design, fold-only median imputation) applied to a
larger sample, plus two additions appropriate at larger N:

  1. Two extra model classes (Random Forest, Gradient Boosting) alongside
     the original MLP and logistic-regression variants -- tree ensembles
     are less prone to the single-feature-floor-beats-the-full-model
     overfitting signature Round 3 of the v2 diagnosis found, and are a
     more realistic candidate for a genuinely competitive result at this
     scale.
  2. A held-out test-set evaluation (80/20 stratified split, fit ONCE on
     the 80% training partition, scored ONCE on the untouched 20%) in
     addition to 5-fold CV on the training partition -- so the headline
     number this script reports is an honest, non-reused-data test AUC,
     not a cross-validation mean that has already seen every example.

IMPORTANT: this script does not manufacture a result. If the classifier
still does not clear a meaningful AUC at this larger scale, that is
itself the finding to report -- honestly, the same way the v2 diagnosis
was reported -- not something to suppress or re-run until a better number
appears. Do not cherry-pick a random_state or fold split to inflate the
reported number.

Usage: python train_cbm_classifier_v3.py
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONCEPTS_CSV = os.path.join(RESULTS, "cbm_concepts_v3.csv")

CONCEPT_COLS = ["C1_depth_ppm", "C2_shape_flatbottom_ratio", "C3_ingress_egress_symmetry",
                "C4_odd_even_rel_diff", "C5_secondary_eclipse_ppm", "C6_centroid_shift_px",
                "C7_phase_fold_coherence"]
MAYBE_MISSING = ["C3_ingress_egress_symmetry", "C4_odd_even_rel_diff",
                  "C5_secondary_eclipse_ppm", "C6_centroid_shift_px", "C7_phase_fold_coherence"]

MODEL_VARIANTS = {
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(8,), activation="relu", alpha=1e-2,
                                  max_iter=3000, random_state=42),
    "LogReg": lambda: LogisticRegression(C=1.0, max_iter=2000),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42),
    "GradientBoosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
}


def build_features(df, medians=None, with_indicators=True):
    meds = {} if medians is None else medians
    X_parts, names = [], []
    for c in CONCEPT_COLS:
        col = df[c].astype(float)
        if c in MAYBE_MISSING:
            med = col.median(skipna=True) if medians is None else meds[c]
            if medians is None:
                meds[c] = med
            X_parts.append(col.fillna(med).values.reshape(-1, 1)); names.append(c)
            if with_indicators:
                X_parts.append(col.isna().astype(float).values.reshape(-1, 1))
                names.append(c.split("_")[0] + "_missing")
        else:
            X_parts.append(col.values.reshape(-1, 1)); names.append(c)
    return np.hstack(X_parts), names, meds


def run_cv(sample, y, model_fn, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(sample, y)):
        df_tr, df_te = sample.iloc[tr_idx], sample.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        X_tr, feature_names, meds = build_features(df_tr, with_indicators=True)
        X_te, _, _ = build_features(df_te, medians=meds, with_indicators=True)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        clf = model_fn()
        clf.fit(X_tr_s, y_tr)
        pred = clf.predict(X_te_s)
        proba = clf.predict_proba(X_te_s)[:, 1]
        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = np.nan
        rows.append({
            "fold": fold, "n_test": len(te_idx),
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "roc_auc": auc,
        })

    means = {k: np.nanmean([r[k] for r in rows]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    stds = {k: np.nanstd([r[k] for r in rows]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    print(f"\n=== {label}: 5-fold CV (train partition only) ===")
    for r in rows:
        print(f"  fold {r['fold']}: n={r['n_test']:>4} acc={r['accuracy']:.3f} "
              f"auc={r['roc_auc']:.3f}")
    print(f"  mean: " + " ".join(f"{k}={means[k]:.3f}+/-{stds[k]:.3f}"
          for k in ["accuracy", "roc_auc"]))
    return rows, means, stds


def held_out_test(sample_tr, y_tr, sample_te, y_te, model_fn, label):
    X_tr, feature_names, meds = build_features(sample_tr, with_indicators=True)
    X_te, _, _ = build_features(sample_te, medians=meds, with_indicators=True)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    clf = model_fn()
    clf.fit(X_tr_s, y_tr)
    pred = clf.predict(X_te_s)
    proba = clf.predict_proba(X_te_s)[:, 1]
    try:
        auc = roc_auc_score(y_te, proba)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(y_te, pred)
    print(f"  [{label}] HELD-OUT TEST (n={len(y_te)}, never touched during "
          f"fitting/scaling): accuracy={acc:.3f}  ROC-AUC={auc:.3f}")

    pi = permutation_importance(clf, X_te_s, y_te, n_repeats=30, random_state=42, scoring="roc_auc")
    order = np.argsort(pi.importances_mean)[::-1]
    print(f"    permutation importance (held-out test set), {label}:")
    for i in order:
        print(f"      {feature_names[i]:20s} {pi.importances_mean[i]:+.4f} +/- {pi.importances_std[i]:.4f}")

    return {"accuracy": acc, "roc_auc": auc,
            "permutation_importance": dict(zip(feature_names, zip(pi.importances_mean.tolist(), pi.importances_std.tolist())))}


def main():
    if not os.path.exists(CONCEPTS_CSV):
        raise SystemExit(
            f"{CONCEPTS_CSV} not found. Run select_cbm_sample_v3.py then "
            "compute_cbm_concepts_v3.py first (see their docstrings for the "
            "full, multi-hour pipeline this depends on)."
        )

    df = pd.read_csv(CONCEPTS_CSV)
    sample = df[df["label"].isin(["planet", "not_planet"]) & (df["status"] == "ok")].reset_index(drop=True)
    y_all = (sample["label"] == "planet").astype(int).values
    print(f"Training sample: {len(sample)} targets ({y_all.sum()} planet, {len(y_all)-y_all.sum()} not_planet)")
    if len(sample) < 200:
        print("  NOTE: this sample is not much larger than the 80-target v2 "
              "sample yet -- run select_cbm_sample_v3.py / "
              "compute_cbm_concepts_v3.py to completion (or with a larger "
              "--n-per-class) for a real scale-up before treating any AUC "
              "reported below as reportable in the paper.")

    sample_tr, sample_te, y_tr, y_te = train_test_split(
        sample, y_all, test_size=0.2, stratify=y_all, random_state=42)
    print(f"Held-out test split: {len(sample_tr)} train / {len(sample_te)} test "
          "(stratified, fixed random_state=42, fit and scaled on train only).")

    summary = {"n_samples": len(sample), "n_planet": int(y_all.sum()),
               "n_not_planet": int(len(y_all) - y_all.sum()),
               "n_train": len(sample_tr), "n_test": len(sample_te),
               "variants": {}}

    for name, model_fn in MODEL_VARIANTS.items():
        cv_rows, cv_means, cv_stds = run_cv(sample_tr.reset_index(drop=True), y_tr, model_fn, name)
        test_result = held_out_test(sample_tr, y_tr, sample_te, y_te, model_fn, name)
        summary["variants"][name] = {
            "cv_per_fold": cv_rows, "cv_mean": cv_means, "cv_std": cv_stds,
            "held_out_test": test_result,
        }

    print("\n=== SUMMARY: held-out test ROC-AUC by model ===")
    for name, res in summary["variants"].items():
        print(f"  {name:20s} AUC={res['held_out_test']['roc_auc']:.3f}  "
              f"acc={res['held_out_test']['accuracy']:.3f}")

    out_path = os.path.join(RESULTS, "cbm_classifier_v3_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
