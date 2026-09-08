#!/usr/bin/env python3
"""
Module 4 labeled-sample selection, v2 -- adds epoch-count stratification on
top of select_cbm_sample.py's depth-decile stratification.

Why v2 exists: the original 51-target cbm_sample.csv (depth-stratified only)
showed a real confound -- 34.6% of its planets were excluded by the
n_epochs>=4 floor vs. 20.0% of its FPs, and this measurably corrupted the
CBM classifier's missingness-indicator features (see
results/module4_classifier_design.md, Diagnosis B).

check_epoch_count_selection_bias.py then tested whether this was a genuine
catalog-wide survey selection effect (in which case no re-sampling would
fix it) or specific to that one small draw. Result, at N=498 (250/250,
disposition-stratified only): catalog-wide, planets have MORE sector
coverage and a higher epoch-count proxy than FPs (median n_epochs_proxy
23.6 vs 15.0; only 6.0% of planets fall below the proxy-4 floor vs. 24.4%
of FPs -- Mann-Whitney p < 1e-4 on the proxy, p < 1e-20 on sector count).
The direction is REVERSED from what corrupted the original small sample --
confirming that draw's confound was a sampling artifact of depth-only
stratification, not a structural catalog property to work around
permanently. The fix is to stratify by epoch-count too, not to redesign
the missingness handling itself.

This script reuses results/epoch_bias_check.csv's already-fetched pool
(498 targets, real period + sector-count-derived n_epochs_proxy from a
metadata-only MAST search -- no need to re-run those ~500 MAST calls) and
stratifies EACH class jointly by depth decile AND epoch-count-proxy
quintile before drawing the final sample, then re-queries the archive
(cheap, no MAST calls) for the full ephemeris fields
(duration_hr, depth_ppm, t0_bjd) needed by compute_cbm_concepts.py.

Usage: python select_cbm_sample_v2.py [--n-per-class 40]
Writes results/cbm_sample_v2.csv
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
    ap.add_argument("--n-per-class", type=int, default=40)
    args = ap.parse_args()

    pool = pd.read_csv(os.path.join(RESULTS, "epoch_bias_check.csv"))
    pool = pool[pool["status"] == "ok"].copy()
    print(f"Pool from epoch_bias_check.csv: {len(pool)} targets "
          f"({(pool['label']=='planet').sum()} planet, {(pool['label']=='not_planet').sum()} not_planet)",
          flush=True)

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
    out_path = os.path.join(RESULTS, "cbm_sample_v2.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}: {len(out)} targets "
          f"({(out['label']=='planet').sum()} planet, {(out['label']=='not_planet').sum()} not_planet)")

    # verify the epoch-count balance actually improved
    for lbl in ("planet", "not_planet"):
        sub = out[out["label"] == lbl]
        below4 = (sub["n_epochs_proxy"] < 4).mean()
        print(f"  {lbl}: median n_epochs_proxy={sub['n_epochs_proxy'].median():.1f}, "
              f"fraction below proxy-4 floor={below4:.3f}")


if __name__ == "__main__":
    main()
