#!/usr/bin/env python3
"""
Select a depth-stratified benchmark sample of confirmed TESS-discovered
planets from the NASA Exoplanet Archive, for a real ExoVeil vs. BLS/trapezoid
recovery-rate comparison (as opposed to 4 hand-picked targets).

Queries pscomppars (one best-solution row per planet) for TESS discoveries
with non-null pl_orbper, pl_tranmid, pl_trandur, pl_trandep, tic_id -- the
fields needed both to search MAST and to fold each pipeline's output against
ground truth. Bins the ~850 qualifying planets into pl_trandep deciles and
samples across all ten bins (fixed RNG seed) so the benchmark spans easy
(deep, >5000 ppm) to hard (shallow, <500 ppm) cases, matching the spirit of
ExoVeil's own depth-stratified injection-recovery table (Sec. 4 of the
paper).

The sample is saved BEFORE any light curve is downloaded or either pipeline
is run, so it's fixed and auditable independent of what run_benchmark.py
does with it.

Usage: python select_benchmark_sample.py [--n 25] [--seed 42]
Writes: results/benchmark_sample.csv
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-deciles", type=int, default=10)
    args = ap.parse_args()

    print("Querying NASA Exoplanet Archive (pscomppars) for TESS-discovered "
          "confirmed planets with complete transit parameters ...", flush=True)
    tab = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,tic_id,hostname,pl_orbper,pl_tranmid,pl_trandur,pl_trandep,disc_facility",
        where=(
            "disc_facility like '%Transiting Exoplanet Survey Satellite%' "
            "and pl_orbper is not null and pl_tranmid is not null "
            "and pl_trandur is not null and pl_trandep is not null "
            "and tic_id is not null"
        ),
    )
    df = tab.to_pandas()
    print(f"  {len(df)} qualifying planets", flush=True)

    df["depth_ppm"] = df["pl_trandep"] * 1e4  # % -> ppm
    df = df.dropna(subset=["depth_ppm"]).reset_index(drop=True)
    df = df[df["depth_ppm"] > 0].reset_index(drop=True)

    df["decile"] = pd.qcut(df["depth_ppm"], args.n_deciles, labels=False, duplicates="drop")
    n_bins = df["decile"].nunique()
    print(f"  depth range {df['depth_ppm'].min():.1f}-{df['depth_ppm'].max():.1f} ppm, "
          f"split into {n_bins} deciles", flush=True)

    rng = np.random.default_rng(args.seed)
    per_bin = args.n // n_bins
    extra = args.n - per_bin * n_bins  # first `extra` bins get one more

    picked = []
    for b in sorted(df["decile"].unique()):
        pool = df[df["decile"] == b]
        k = per_bin + (1 if b < extra else 0)
        k = min(k, len(pool))
        idx = rng.choice(pool.index.values, size=k, replace=False)
        picked.append(df.loc[idx])

    sample = pd.concat(picked).sort_values("decile").reset_index(drop=True)

    out = pd.DataFrame({
        "target": sample["pl_name"],
        "tic_id": sample["tic_id"],
        "hostname": sample["hostname"],
        "period_d": sample["pl_orbper"],
        "t0_bjd": sample["pl_tranmid"],
        "duration_hr": sample["pl_trandur"],
        "depth_ppm": sample["depth_ppm"],
        "depth_decile": sample["decile"],
        "source": "NASA Exoplanet Archive pscomppars",
    })

    out_path = os.path.join(RESULTS, "benchmark_sample.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} targets to {out_path}", flush=True)
    print(out[["target", "tic_id", "period_d", "depth_ppm", "depth_decile"]].to_string(index=False))


if __name__ == "__main__":
    main()
