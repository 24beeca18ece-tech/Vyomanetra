#!/usr/bin/env python3
"""
Module 4 labeled-sample selection, v3 -- scale-up of select_cbm_sample_v2.py's
joint depth x epoch-count stratification to a much larger sample, for the
DSAA-track goal of training the CBM classifier on a dataset large enough to
report a real ROC-AUC rather than a diagnosed small-N limitation.

Same joint-stratification logic as v2 (unchanged, since v2's fix -- stratify
by depth decile AND epoch-count-proxy quintile jointly -- is exactly the
fix Round 2 of the classifier diagnosis (paper_tai/main.tex,
Section IV-B, "confound-tracing check") showed was necessary). The only
change is scale: this script expects a LARGER metadata-only pool than the
498-target results/epoch_bias_check.csv used by v2, and draws more targets
per class from it.

Before running this script, grow the metadata pool by re-running the
existing, already-checkpointed/resumable script with a larger --n-per-class:

    python check_epoch_count_selection_bias.py --n-per-class 1000

This appends to the SAME results/epoch_bias_check.csv v2 already reads (it
skips targets already done via done_set(), so it is safe and cheap to
re-run with a bigger target count -- it only fetches the NEW targets). At
~5s/target for the metadata-only MAST search, 1000 per class (2000 total,
vs. the existing 498) takes roughly 2-3 hours and can be safely
interrupted/resumed at any point.

Usage: python select_cbm_sample_v3.py [--n-per-class 400]
Writes results/cbm_sample_v3.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=400,
                     help="targets per class to draw for the heavy pipeline "
                          "(compute_cbm_concepts_v3.py). Default 400 (800 "
                          "total) -- a 10x scale-up from the 80-target v2 "
                          "sample, chosen to be a realistic multi-hour, not "
                          "multi-day, download+detrend+BLS run. Raise this "
                          "only once you've confirmed the pool "
                          "(epoch_bias_check.csv) actually has enough rows "
                          "per class to support it.")
    args = ap.parse_args()

    pool_path = os.path.join(RESULTS, "epoch_bias_check.csv")
    if not os.path.exists(pool_path):
        raise SystemExit(
            f"{pool_path} not found. Run "
            "`python check_epoch_count_selection_bias.py --n-per-class 1000` "
            "first to build a metadata-only pool large enough to draw from."
        )

    pool = pd.read_csv(pool_path)
    pool = pool[pool["status"] == "ok"].copy()
    n_planet_pool = (pool["label"] == "planet").sum()
    n_fp_pool = (pool["label"] == "not_planet").sum()
    print(f"Pool from epoch_bias_check.csv: {len(pool)} targets "
          f"({n_planet_pool} planet, {n_fp_pool} not_planet)", flush=True)
    if min(n_planet_pool, n_fp_pool) < args.n_per_class:
        print(f"  WARNING: pool has only {min(n_planet_pool, n_fp_pool)} in "
              f"the smaller class -- fewer than --n-per-class={args.n_per_class} "
              "requested. Grow the pool first (see module docstring) for a "
              "real scale-up; otherwise this will silently draw fewer targets "
              "than intended.", flush=True)

    print("Re-querying TOI table for full ephemeris fields (depth, duration, t0) ...", flush=True)
    t = NasaExoplanetArchive.query_criteria(
        table="toi", select="toi,tid,pl_orbper,pl_trandurh,pl_trandep,pl_tranmid,st_tmag",
        where="pl_orbper is not null and pl_trandurh is not null "
              "and pl_trandep is not null and pl_tranmid is not null",
    )
    eph = t.to_pandas().drop_duplicates(subset="toi")
    pool["toi"] = pool["toi"].astype(float)
    eph["toi"] = eph["toi"].astype(float)
    merged = pool.merge(eph, on="toi", how="inner", suffixes=("", "_eph"))
    print(f"  {len(merged)}/{len(pool)} pool targets have complete ephemeris fields", flush=True)

    rng = np.random.default_rng(SEED)

    def joint_stratified_pick(sub, n):
        sub = sub.copy()
        if len(sub) <= n:
            return sub
        sub["depth_decile"] = pd.qcut(sub["pl_trandep"].rank(method="first"), 5, labels=False)
        sub["epoch_quintile"] = pd.qcut(sub["n_epochs_proxy"].rank(method="first"), 5, labels=False)
        picked = []
        per_cell = max(1, n // 25)  # 5 depth x 5 epoch cells
        for d in range(5):
            for e in range(5):
                cell = sub[(sub["depth_decile"] == d) & (sub["epoch_quintile"] == e)]
                k = min(per_cell, len(cell))
                if k > 0:
                    picked.append(cell.sample(k, random_state=SEED))
        out = pd.concat(picked) if picked else sub.iloc[:0]
        if len(out) < n:
            remaining = sub.drop(out.index)
            extra = remaining.sample(min(n - len(out), len(remaining)), random_state=SEED)
            out = pd.concat([out, extra])
        return out.drop(columns=["depth_decile", "epoch_quintile"])

    planets = joint_stratified_pick(merged[merged["label"] == "planet"], args.n_per_class)
    fps = joint_stratified_pick(merged[merged["label"] == "not_planet"], args.n_per_class)
    sample = pd.concat([planets, fps]).reset_index(drop=True)

    out = pd.DataFrame({
        "target": "TOI-" + sample["toi"].astype(str),
        "tic_id": "TIC " + sample["tid"].astype(int).astype(str),
        "label": sample["label"],
        "period_d": sample["pl_orbper"],
        "t0_bjd": sample["pl_tranmid"],
        "duration_hr": sample["pl_trandurh"],
        "depth_ppm": sample["pl_trandep"],
        "tmag": sample["st_tmag"],
        "n_epochs_proxy": sample["n_epochs_proxy"],
    })
    out_path = os.path.join(RESULTS, "cbm_sample_v3.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}: {len(out)} targets "
          f"({(out['label']=='planet').sum()} planet, {(out['label']=='not_planet').sum()} not_planet)")

    for lbl in ("planet", "not_planet"):
        sub = out[out["label"] == lbl]
        below4 = (sub["n_epochs_proxy"] < 4).mean()
        print(f"  {lbl}: median n_epochs_proxy={sub['n_epochs_proxy'].median():.1f}, "
              f"fraction below proxy-4 floor={below4:.3f}")

    print("\nNext step (the slow one -- see module docstring for time estimate):")
    print("  python compute_cbm_concepts_v3.py")


if __name__ == "__main__":
    main()
