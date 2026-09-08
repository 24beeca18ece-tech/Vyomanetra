#!/usr/bin/env python3
"""
Module 4 -- retrain the CBM classifier on results/cbm_concepts_v2.csv (the
epoch-count+depth jointly stratified 80-target sample), and re-run the
missingness-indicator ablation from module4_classifier_design.md's
Diagnosis B to test whether it's fixed at the root now that the sample no
longer fights the catalog-wide epoch-count/label direction.

Reports, in order:
  1. Realized epoch-count balance by label (n_epochs, the REAL per-target
     value from the concept pipeline, not the archival proxy used for
     sample selection) -- direct test of whether joint stratification
     worked as intended.
  2. TOI-700 / V1828 Aql C7 sanity pair at this new scale.
  3. Three model variants, each with FULL per-fold 5-fold CV metrics (not
     just mean+/-std) plus the V1828 Aql prediction:
       (a) MLP + missingness indicators (primary, per report Section 3.4)
       (b) Logistic regression + missingness indicators
       (c) Logistic regression, indicators removed (the ablation)
     Permutation importance is computed but the script does not draw any
     interpretability conclusion from it -- that is deliberately left for
     a separate pass once the ablation result here is reviewed.

Usage: python train_cbm_classifier_v2.py
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONCEPTS_CSV = os.path.join(RESULTS, "cbm_concepts_v2.csv")

CONCEPT_COLS = ["C1_depth_ppm", "C2_shape_flatbottom_ratio", "C3_ingress_egress_symmetry",
                "C4_odd_even_rel_diff", "C5_secondary_eclipse_ppm", "C6_centroid_shift_px",
                "C7_phase_fold_coherence"]
MAYBE_MISSING = ["C3_ingress_egress_symmetry", "C4_odd_even_rel_diff",
                  "C5_secondary_eclipse_ppm", "C6_centroid_shift_px", "C7_phase_fold_coherence"]


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


def run_cv(sample, y, model_fn, with_indicators, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    imps = []
    feature_names = None
    for fold, (tr_idx, te_idx) in enumerate(skf.split(sample, y)):
        df_tr, df_te = sample.iloc[tr_idx], sample.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        X_tr, feature_names, meds = build_features(df_tr, with_indicators=with_indicators)
        X_te, _, _ = build_features(df_te, medians=meds, with_indicators=with_indicators)
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
        pi = permutation_importance(clf, X_te_s, y_te, n_repeats=30, random_state=42, scoring="accuracy")
        imps.append(pi.importances_mean)

    print(f"\n=== {label}: per-fold 5-fold CV metrics ===")
    print(f"{'fold':>4} {'n_test':>6} {'accuracy':>9} {'precision':>10} {'recall':>7} {'f1':>7} {'roc_auc':>8}")
    for r in rows:
        print(f"{r['fold']:>4} {r['n_test']:>6} {r['accuracy']:>9.3f} {r['precision']:>10.3f} "
              f"{r['recall']:>7.3f} {r['f1']:>7.3f} {r['roc_auc']:>8.3f}")
    means = {k: np.nanmean([r[k] for r in rows]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    stds = {k: np.nanstd([r[k] for r in rows]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    print(f"{'mean':>4} {'':>6} " + " ".join(f"{means[k]:.3f}+/-{stds[k]:.3f}"
          for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]))

    imp_arr = np.array(imps)
    imp_mean, imp_std = imp_arr.mean(axis=0), imp_arr.std(axis=0)
    order = np.argsort(imp_mean)[::-1]
    print(f"\n  permutation importance (avg over 5 held-out folds), {label}:")
    for i in order:
        print(f"    {feature_names[i]:20s} {imp_mean[i]:+.4f} +/- {imp_std[i]:.4f}")

    return rows, means, stds, feature_names, imp_mean.tolist(), imp_std.tolist()


def final_model_and_v1828(sample, y, df_full, model_fn, with_indicators, label):
    X_all, feature_names, medians_all = build_features(sample, with_indicators=with_indicators)
    scaler_all = StandardScaler().fit(X_all)
    clf = model_fn()
    clf.fit(scaler_all.transform(X_all), y)

    v1828 = df_full[df_full["target"] == "V1828 Aql"]
    if len(v1828) == 0:
        print(f"  [{label}] V1828 Aql not found in concepts file -- skipping check")
        return None
    Xv, _, _ = build_features(v1828, medians=medians_all, with_indicators=with_indicators)
    proba_v = float(clf.predict_proba(scaler_all.transform(Xv))[0, 1])
    verdict = "CORRECT (low)" if proba_v < 0.5 else "WRONG (should be low)"
    print(f"  [{label}] V1828 Aql P(planet) = {proba_v:.4f}  -- {verdict}")
    return proba_v


def main():
    df = pd.read_csv(CONCEPTS_CSV)
    sample = df[df["label"].isin(["planet", "not_planet"]) & (df["status"] == "ok")].reset_index(drop=True)
    y = (sample["label"] == "planet").astype(int).values

    # ---- 1. realized epoch-count balance ----
    print("=== 1. Realized epoch-count balance (real n_epochs, from the concept pipeline) ===")
    print(sample.groupby("label")["n_epochs"].describe()[["count", "mean", "50%", "min", "max"]])
    below4 = sample.groupby("label")["n_epochs"].apply(lambda s: (s < 4).mean())
    print("\nfraction with n_epochs < 4, by label:")
    print(below4)

    # ---- 2. TOI-700 / V1828 Aql sanity pair ----
    print("\n=== 2. TOI-700 / V1828 Aql sanity pair ===")
    pair = df[df["target"].isin(["TOI-700", "V1828 Aql"])][["target", "label", "C7_phase_fold_coherence", "n_epochs"]]
    print(pair.to_string())

    # ---- 3. three model variants ----
    print(f"\nTraining sample: {len(sample)} targets ({y.sum()} planet, {len(y)-y.sum()} not_planet)")

    results = {}

    rows_mlp, means_mlp, stds_mlp, names_mlp, imp_mean_mlp, imp_std_mlp = run_cv(
        sample, y, lambda: MLPClassifier(hidden_layer_sizes=(8,), activation="relu", alpha=1e-2,
                                          max_iter=3000, random_state=42),
        with_indicators=True, label="MLP + indicators")
    v_mlp = final_model_and_v1828(sample, y, df,
                                   lambda: MLPClassifier(hidden_layer_sizes=(8,), activation="relu", alpha=1e-2,
                                                          max_iter=3000, random_state=42),
                                   with_indicators=True, label="MLP + indicators")

    rows_lr, means_lr, stds_lr, names_lr, imp_mean_lr, imp_std_lr = run_cv(
        sample, y, lambda: LogisticRegression(C=1.0, max_iter=2000),
        with_indicators=True, label="LogReg + indicators")
    v_lr = final_model_and_v1828(sample, y, df, lambda: LogisticRegression(C=1.0, max_iter=2000),
                                  with_indicators=True, label="LogReg + indicators")

    rows_lr_noind, means_lr_noind, stds_lr_noind, names_lr_noind, imp_mean_lr_noind, imp_std_lr_noind = run_cv(
        sample, y, lambda: LogisticRegression(C=1.0, max_iter=2000),
        with_indicators=False, label="LogReg, NO indicators (ablation)")
    v_lr_noind = final_model_and_v1828(sample, y, df, lambda: LogisticRegression(C=1.0, max_iter=2000),
                                        with_indicators=False, label="LogReg, NO indicators (ablation)")

    print("\n=== SUMMARY: V1828 Aql P(planet) across variants ===")
    print(f"  MLP + indicators:            {v_mlp:.4f}  {'CORRECT' if v_mlp is not None and v_mlp<0.5 else 'WRONG'}")
    print(f"  LogReg + indicators:         {v_lr:.4f}  {'CORRECT' if v_lr is not None and v_lr<0.5 else 'WRONG'}")
    print(f"  LogReg, no indicators:       {v_lr_noind:.4f}  {'CORRECT' if v_lr_noind is not None and v_lr_noind<0.5 else 'WRONG'}")

    summary = {
        "n_samples": len(sample), "n_planet": int(y.sum()), "n_not_planet": int(len(y) - y.sum()),
        "epoch_balance": {lbl: {"median_n_epochs": float(sample[sample['label']==lbl]['n_epochs'].median()),
                                 "frac_below_4": float((sample[sample['label']==lbl]['n_epochs']<4).mean())}
                          for lbl in ["planet", "not_planet"]},
        "mlp_indicators": {"per_fold": rows_mlp, "mean": means_mlp, "std": stds_mlp,
                            "v1828_p_planet": v_mlp,
                            "permutation_importance": dict(zip(names_mlp, zip(imp_mean_mlp, imp_std_mlp)))},
        "logreg_indicators": {"per_fold": rows_lr, "mean": means_lr, "std": stds_lr,
                               "v1828_p_planet": v_lr,
                               "permutation_importance": dict(zip(names_lr, zip(imp_mean_lr, imp_std_lr)))},
        "logreg_no_indicators": {"per_fold": rows_lr_noind, "mean": means_lr_noind, "std": stds_lr_noind,
                                  "v1828_p_planet": v_lr_noind,
                                  "permutation_importance": dict(zip(names_lr_noind, zip(imp_mean_lr_noind, imp_std_lr_noind)))},
    }
    with open(os.path.join(RESULTS, "cbm_classifier_v2_report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSaved {os.path.join(RESULTS, 'cbm_classifier_v2_report.json')}")


if __name__ == "__main__":
    main()
