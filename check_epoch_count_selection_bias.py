#!/usr/bin/env python3
"""
Cheap, catalog-wide check: does the epoch-count/label confound found in the
51-target cbm_sample (planets disproportionately excluded by the n_epochs>=4
floor: 34.6% of planets vs 20.0% of FPs) reflect a real, catalog-wide
selection effect, or is it specific to that one small stratified draw?

Method: draw 500 TOIs (250 CP/KP, 250 FP), stratified ONLY by disposition
(not depth, unlike select_cbm_sample.py) -- period alone is enough for this
check. For each, do a METADATA-ONLY MAST search (lk.search_lightcurve,
~5s/target, no light-curve download -- a full download+detrend+BLS pipeline
run on 500 targets would take ~8+ hours, not a "cheap" check) to count
observed TESS sectors, then approximate:

    n_epochs_proxy = (n_sectors * 27.4 days) / period_days

This is a proxy for the real per_epoch_depths_trapezoid() epoch count (which
needs an actual light curve fit), not a substitute for it -- documented as
such. It tests the SAME underlying mechanism (shorter period -> more
epochs observable in a fixed baseline) using only archival period +
sector-coverage metadata, at ~1% of the cost of a full pipeline run.

Checkpointed like the rest of this project's batch scripts: results appended
to results/epoch_bias_check.csv incrementally, resumable.

Usage: python check_epoch_count_selection_bias.py [--n-per-class 250]
"""
import os
import csv
import time
import argparse

import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_CSV = os.path.join(RESULTS, "epoch_bias_check.csv")
SECTOR_DAYS = 27.4
SEED = 7

COLS = ["toi", "tid", "label", "period_d", "status", "n_sectors", "baseline_days_approx", "n_epochs_proxy"]


def append_row(row):
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def done_set():
    if not os.path.exists(OUT_CSV):
        return set()
    return set(pd.read_csv(OUT_CSV)["toi"].astype(str))


def n_sectors_for(tid):
    sr = lk.search_lightcurve(f"TIC {tid}", mission="TESS", author="SPOC")
    if sr is None or len(sr) == 0:
        return 0
    sectors = set()
    for m in sr.table["mission"]:
        # mission strings look like "TESS Sector 14"
        parts = str(m).split()
        if parts and parts[-1].isdigit():
            sectors.add(int(parts[-1]))
    return len(sectors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=250)
    args = ap.parse_args()

    print("Querying TOI table (disposition-only stratification, period required) ...", flush=True)
    t = NasaExoplanetArchive.query_criteria(
        table="toi", select="toi,tid,tfopwg_disp,pl_orbper",
        where="pl_orbper is not null and tid is not null",
    )
    df = t.to_pandas()
    df["label"] = df["tfopwg_disp"].map({"CP": "planet", "KP": "planet", "FP": "not_planet"})
    df = df.dropna(subset=["label"]).drop_duplicates(subset="tid", keep="first")
    print(f"  {len(df)} eligible rows after label mapping + TIC de-dup", flush=True)

    rng = np.random.default_rng(SEED)
    planets = df[df["label"] == "planet"].sample(
        min(args.n_per_class, (df["label"] == "planet").sum()), random_state=SEED)
    fps = df[df["label"] == "not_planet"].sample(
        min(args.n_per_class, (df["label"] == "not_planet").sum()), random_state=SEED)
    draw = pd.concat([planets, fps]).reset_index(drop=True)
    print(f"  drew {len(draw)} targets ({len(planets)} planet, {len(fps)} not_planet)", flush=True)

    done = done_set()
    pending = draw[~draw["toi"].astype(str).isin(done)]
    print(f"Jobs: {len(draw)} total, {len(done)} already done, {len(pending)} pending.", flush=True)

    for _, r in pending.iterrows():
        toi, tid, label, period = r["toi"], int(r["tid"]), r["label"], float(r["pl_orbper"])
        try:
            n_sec = n_sectors_for(tid)
            baseline = n_sec * SECTOR_DAYS
            proxy = baseline / period if period > 0 else np.nan
            row = {"toi": toi, "tid": tid, "label": label, "period_d": period,
                   "status": "ok", "n_sectors": n_sec,
                   "baseline_days_approx": baseline, "n_epochs_proxy": proxy}
            print(f"  TOI-{toi}  [{label}]  P={period:.3f}d  sectors={n_sec}  "
                  f"n_epochs_proxy={proxy:.1f}", flush=True)
        except Exception as exc:
            row = {"toi": toi, "tid": tid, "label": label, "period_d": period,
                   "status": f"error: {exc}", "n_sectors": "", "baseline_days_approx": "", "n_epochs_proxy": ""}
            print(f"  TOI-{toi}  FAILED: {exc}", flush=True)
        append_row(row)

    print("\nBatch complete.")


if __name__ == "__main__":
    main()
