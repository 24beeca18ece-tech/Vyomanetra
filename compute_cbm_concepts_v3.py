#!/usr/bin/env python3
"""
Module 4 -- compute C1-C7 concepts for the scaled-up results/cbm_sample_v3.csv
(select_cbm_sample_v3.py), plus the TOI-700 / V1828 Aql sanity-check pair.

Identical checkpointed pattern to compute_cbm_concepts_v2.py, pointed at the
v3 sample and a separate output file so cbm_concepts_v2.csv (the 80-target,
paper-reported sample) is untouched and remains the source of every number
already reported in paper/main.tex and paper_tai/main.tex.

TIME ESTIMATE (read before running): each target does a full MAST light-curve
download + Savitzky-Golay detrend + BLS period search + trapezoid fit + all
7 concepts. check_epoch_count_selection_bias.py's own docstring estimates
~8+ hours for a full pipeline run on 500 targets -- i.e. roughly a minute per
target. This script is checkpointed and resumable (like every batch script in
this project): if it is killed or the machine restarts, rerunning the same
command skips every target already in cbm_concepts_v3.csv and continues from
where it left off. Plan to run this in the background (e.g. `nohup python
compute_cbm_concepts_v3.py > concepts_v3.log 2>&1 &` on the machine that
actually has TESS/MAST network access) rather than in an interactive
session, and expect it to take several hours to a day or more depending on
how many targets select_cbm_sample_v3.py drew.

Usage: python compute_cbm_concepts_v3.py [--retry-failed] [--limit N]
"""
import os
import sys
import csv
import argparse
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import download_lc, detrend, identify, characterize, RESULTS, TARGETS  # noqa: E402
import modules_4 as M4  # noqa: E402

OUT_CSV = os.path.join(RESULTS, "cbm_concepts_v3.csv")
SAMPLE_CSV = os.path.join(RESULTS, "cbm_sample_v3.csv")

COLS = ["target", "label", "status", "error_message", "period_d", "duration_hr",
        "C1_depth_ppm", "C2_shape_flatbottom_ratio", "C3_ingress_egress_symmetry",
        "C4_odd_even_rel_diff", "C5_secondary_eclipse_ppm", "C6_centroid_shift_px",
        "C7_phase_fold_coherence", "n_epochs"]


def blank(target, label, status, err=""):
    r = {c: "" for c in COLS}
    r.update({"target": target, "label": label, "status": status, "error_message": err})
    return r


def append_row(row):
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def done_set(retry_failed):
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    if retry_failed:
        return set(df.loc[df["status"] == "ok", "target"])
    return set(df["target"])


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
        print(f"  C1={concepts['C1_depth_ppm']:.1f}ppm  C2={concepts['C2_shape_flatbottom_ratio']:.3f}  "
              f"C7={concepts['C7_phase_fold_coherence']}  (n_epochs={concepts['n_epochs']})", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row["error_message"] = f"concepts: {exc}"

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                     help="process at most N pending targets this run, then "
                          "exit cleanly -- useful for chunking a multi-hour "
                          "job into shorter, checkpointed sessions instead of "
                          "one long-lived background process.")
    args = ap.parse_args()

    if not os.path.exists(SAMPLE_CSV):
        raise SystemExit(f"{SAMPLE_CSV} not found. Run select_cbm_sample_v3.py first.")

    sample = pd.read_csv(SAMPLE_CSV)
    jobs = [(r["target"], [str(r["tic_id"])], r["label"]) for _, r in sample.iterrows()]
    for name in ["TOI-700", "V1828 Aql"]:
        tgt = next(t for t in TARGETS if t["name"] == name)
        jobs.append((name, tgt["terms"], "planet" if name == "TOI-700" else "eclipsing_binary_known"))

    done = done_set(args.retry_failed)
    pending = [j for j in jobs if j[0] not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Jobs: {len(jobs)} total, {len(done)} already done, {len(pending)} pending this run.", flush=True)
    if not pending:
        print("Nothing to do.")
        return

    for target_name, terms, label in pending:
        try:
            row = process_target(target_name, terms, label)
        except Exception as exc:
            traceback.print_exc()
            row = blank(target_name, label, "unexpected_error", str(exc))
        append_row(row)
        print(f"  -> appended to {OUT_CSV}", flush=True)

    print("\nBatch complete for this run. Rerun the same command to continue "
          "with any remaining pending targets.")


if __name__ == "__main__":
    main()
