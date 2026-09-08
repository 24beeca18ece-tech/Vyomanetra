#!/usr/bin/env python3
"""
Module 4 (Concept Bottleneck Model) labeled-sample selection.

Unlike the earlier 25-target benchmark (drawn from pscomppars, confirmed
planets only), this queries the NASA Exoplanet Archive's TOI table, which
carries TFOPWG dispositions including real negative-class examples (FP =
false positive) alongside confirmed/candidate planets (CP/KP) -- needed to
train a classifier at all, since Module 4 must discriminate planet from
non-planet, not just characterize planets.

Label mapping:
    CP, KP  -> "planet"      (confirmed planet / known planet)
    FP      -> "not_planet"  (false positive: EB, blend, instrumental, etc.)
PC/APC (candidate/ambiguous, not yet resolved) and FA (false alarm, usually
non-astrophysical) are excluded -- PC/APC have no reliable ground-truth
label yet, and FA is a small, heterogeneous bucket not worth the noise.

Priority is real negative-class coverage over depth-decile balance (unlike
select_benchmark_sample.py) -- stratifying to guarantee a range of planet
AND false-positive depths, with a fixed random seed for reproducibility.

Usage: python select_cbm_sample.py [--n-per-class 25]
Writes results/cbm_sample.csv
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
    ap.add_argument("--n-per-class", type=int, default=25)
    args = ap.parse_args()

    print("Querying NASA Exoplanet Archive TOI table ...", flush=True)
    t = NasaExoplanetArchive.query_criteria(
        table="toi",
        select="toi,tid,ctoi_alias,tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,pl_tranmid,st_tmag",
        where=("pl_orbper is not null and pl_trandurh is not null "
               "and pl_trandep is not null and pl_tranmid is not null "
               "and tid is not null"),
    )
    df = t.to_pandas()
    print(f"  {len(df)} TOI rows with complete ephemeris/depth fields", flush=True)
    print(f"  disposition counts:\n{df['tfopwg_disp'].value_counts()}", flush=True)

    df["label"] = df["tfopwg_disp"].map({"CP": "planet", "KP": "planet", "FP": "not_planet"})
    df = df.dropna(subset=["label"])
    # drop duplicate TIC IDs (multi-planet systems / multiple TOIs on one star)
    # keeping the first -- one light curve per star is enough for concept validation
    df = df.drop_duplicates(subset="tid", keep="first")
    print(f"  {len(df)} rows after label mapping + de-duplication by TIC ID", flush=True)
    print(f"  label counts:\n{df['label'].value_counts()}", flush=True)

    rng = np.random.default_rng(SEED)

    def stratified_pick(sub, n):
        sub = sub.copy()
        if len(sub) <= n:
            return sub
        # stratify across depth deciles within this class, same spirit as
        # select_benchmark_sample.py, so both planet and FP depths span a range
        sub["decile"] = pd.qcut(sub["pl_trandep"].rank(method="first"), 10, labels=False)
        picked = []
        per_decile = max(1, n // 10)
        for d in range(10):
            pool = sub[sub["decile"] == d]
            k = min(per_decile, len(pool))
            if k > 0:
                picked.append(pool.sample(k, random_state=SEED))
        out = pd.concat(picked)
        if len(out) < n:
            remaining = sub.drop(out.index)
            extra = remaining.sample(min(n - len(out), len(remaining)), random_state=SEED)
            out = pd.concat([out, extra])
        return out.drop(columns="decile")

    planets = stratified_pick(df[df["label"] == "planet"], args.n_per_class)
    fps = stratified_pick(df[df["label"] == "not_planet"], args.n_per_class)
    sample = pd.concat([planets, fps]).reset_index(drop=True)

    out = pd.DataFrame({
        "target": "TOI-" + sample["toi"].astype(str),
        "tic_id": "TIC " + sample["tid"].astype(int).astype(str),
        "label": sample["label"],
        "tfopwg_disp": sample["tfopwg_disp"],
        "period_d": sample["pl_orbper"],
        "t0_bjd": sample["pl_tranmid"],
        "duration_hr": sample["pl_trandurh"],
        "depth_ppm": sample["pl_trandep"],
        "tmag": sample["st_tmag"],
    })
    out_path = os.path.join(RESULTS, "cbm_sample.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}: {len(out)} targets "
          f"({(out['label']=='planet').sum()} planet, {(out['label']=='not_planet').sum()} not_planet)")
    print(out[["target", "label", "period_d", "depth_ppm"]].to_string())


if __name__ == "__main__":
    main()
