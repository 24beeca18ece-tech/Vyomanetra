#!/usr/bin/env python3
"""
Module 4 concept-computation validation on a small (14-target) subset before
scaling to the full results/cbm_sample.csv.

Mix: 6 TOI planets (CP/KP) + 6 TOI false positives (FP), spanning a range of
depths, PLUS the two known core-target sanity checks from prototype.py's own
TARGETS list:
  - TOI-700 (clean, well-behaved planet -- expect C7 high, C1 moderate)
  - V1828 Aql (known period-aliased eclipsing binary -- expect C7 LOW, per
    the report's own worked example, Section 5.4/4.1)

Reuses prototype.py's download_lc/detrend/identify/characterize UNCHANGED
(imported, not modified), then modules_4.compute_all_concepts() for C1-C7.

Usage: python validate_cbm_concepts.py
Writes results/cbm_concepts_validation.csv
"""
import os
import sys
import csv
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import download_lc, detrend, identify, characterize, RESULTS, TARGETS  # noqa: E402
import modules_4 as M4  # noqa: E402

OUT_CSV = os.path.join(RESULTS, "cbm_concepts_validation.csv")

VALIDATION_TOIS = {
    "planet": ["TOI-1807.01", "TOI-5111.01", "TOI-3071.01",
               "TOI-1930.01", "TOI-491.01", "TOI-404.01"],
    "not_planet": ["TOI-1357.01", "TOI-1308.01", "TOI-1541.01",
                   "TOI-1572.01", "TOI-184.01", "TOI-3742.01"],
}

COLS = ["target", "label", "status", "error_message", "period_d", "duration_hr",
        "C1_depth_ppm", "C2_shape_flatbottom_ratio", "C3_ingress_egress_symmetry",
        "C4_odd_even_rel_diff", "C5_secondary_eclipse_ppm", "C6_centroid_shift_px",
        "C7_phase_fold_coherence", "n_epochs"]


def blank(target, label, status, err=""):
    r = {c: "" for c in COLS}
    r.update({"target": target, "label": label, "status": status, "error_message": err})
    return r


def process_target(target_name, terms, label):
    print(f"\n{'=' * 66}\n  {target_name}  [{label}]\n{'=' * 66}", flush=True)
    try:
        lc_raw = download_lc(target_name, terms)
    except Exception as exc:
        traceback.print_exc()
        return blank(target_name, label, "download_error", str(exc))
    if lc_raw is None:
        return blank(target_name, label, "no_data")

    try:
        lc_clean = detrend(lc_raw, target_name)
    except Exception as exc:
        traceback.print_exc()
        return blank(target_name, label, "detrend_error", str(exc))

    try:
        bls = identify(lc_clean, target_name)
    except Exception as exc:
        traceback.print_exc()
        return blank(target_name, label, "identify_error", str(exc))

    try:
        fit = characterize(bls["lc_fold"], bls, target_name)
    except Exception as exc:
        traceback.print_exc()
        return blank(target_name, label, "characterize_error", str(exc))

    row = blank(target_name, label, "ok")
    row.update({"period_d": f"{bls['period']:.5f}", "duration_hr": f"{bls['duration']*24:.3f}"})

    try:
        concepts = M4.compute_all_concepts(lc_raw, lc_clean, bls, fit)
        for k in ("C1_depth_ppm", "C2_shape_flatbottom_ratio", "C3_ingress_egress_symmetry",
                  "C4_odd_even_rel_diff", "C5_secondary_eclipse_ppm", "C6_centroid_shift_px",
                  "C7_phase_fold_coherence", "n_epochs"):
            v = concepts[k]
            row[k] = f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else v
        print(f"  C1(depth)={concepts['C1_depth_ppm']:.1f}ppm  "
              f"C2(shape)={concepts['C2_shape_flatbottom_ratio']:.3f}  "
              f"C3(symmetry)={concepts['C3_ingress_egress_symmetry']}  "
              f"C4(odd-even)={concepts['C4_odd_even_rel_diff']}  "
              f"C5(secondary)={concepts['C5_secondary_eclipse_ppm']}  "
              f"C6(centroid)={concepts['C6_centroid_shift_px']}  "
              f"C7(coherence)={concepts['C7_phase_fold_coherence']}  "
              f"(n_epochs={concepts['n_epochs']})", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row["error_message"] = f"concepts: {exc}"

    return row


def append_row(row):
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    sample = pd.read_csv(os.path.join(RESULTS, "cbm_sample.csv")).set_index("target")

    jobs = []
    for label, tois in VALIDATION_TOIS.items():
        for toi in tois:
            r = sample.loc[toi]
            jobs.append((toi, [str(r["tic_id"])], label))
    # core-target sanity checks
    for name in ["TOI-700", "V1828 Aql"]:
        tgt = next(t for t in TARGETS if t["name"] == name)
        jobs.append((name, tgt["terms"], "planet" if name == "TOI-700" else "eclipsing_binary_known"))

    for target_name, terms, label in jobs:
        row = process_target(target_name, terms, label)
        append_row(row)
        print(f"  -> appended to {OUT_CSV}", flush=True)

    print("\nValidation batch complete.")


if __name__ == "__main__":
    main()
