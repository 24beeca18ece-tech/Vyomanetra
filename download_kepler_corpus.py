#!/usr/bin/env python3
"""
Module 2 -- pull a modest Kepler DR25 pretraining corpus (hundreds to
~1-2k light curves, per the right-sized PoC scope -- well short of
ExoVeil's own 16,499 curves, let alone the guidance doc's full 200K-star
catalog; that gap is the point of a proof-of-concept, not a flaw).

Sourced from the Kepler `cumulative` KOI table (CONFIRMED + CANDIDATE host
stars) -- gives a diverse set of real Kepler targets with known transit
context, without requiring a full-catalog crawl. Long-cadence data,
multiple quarters stitched (~5 quarters ~= 19-20K points, matching the
single-TESS-sector scale already validated to fit in 8GB in
results/module2_ssm_design.md).

Preprocessing is DELIBERATELY LIGHTER than prototype.py's Module 1
detrend() (normalize + outlier removal only, NO Savitzky-Golay flatten --
see the comment at the flatten call site for why: a self-supervised world
model needs to see the natural stellar variability it's supposed to learn
to predict, and prototype.py's ~301-cadence SG window would strip that out
as a high-pass filter before the model ever saw it). An earlier version of
this script reused prototype.py's flatten() call directly; caught and fixed
before committing to a full corpus pull -- see
results/module2_ssm_design.md for the full account.

Checkpointed like every other batch script in this project: each target's
processed (time, flux, dt) arrays are saved as one .npz file as soon as
they're ready; already-downloaded targets are skipped on rerun.

Usage: python download_kepler_corpus.py [--n-targets 300] [--n-quarters 5]
Writes results/kepler_corpus/<kepid>.npz and results/kepler_corpus_manifest.csv
"""
import os
import csv
import argparse
import traceback

import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORPUS_DIR = os.path.join(RESULTS, "kepler_corpus")
MANIFEST_CSV = os.path.join(RESULTS, "kepler_corpus_manifest.csv")
os.makedirs(CORPUS_DIR, exist_ok=True)

COLS = ["kepid", "status", "error_message", "n_points", "n_quarters", "span_days"]


def append_row(row):
    write_header = not os.path.exists(MANIFEST_CSV)
    with open(MANIFEST_CSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def done_set():
    if not os.path.exists(MANIFEST_CSV):
        return set()
    return set(pd.read_csv(MANIFEST_CSV)["kepid"].astype(str))


def process_one(kepid, n_quarters):
    npz_path = os.path.join(CORPUS_DIR, f"{kepid}.npz")
    row = {"kepid": kepid, "status": "ok", "error_message": "",
           "n_points": "", "n_quarters": "", "span_days": ""}
    try:
        sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler",
                                   cadence="long")
        if sr is None or len(sr) == 0:
            row.update(status="no_data")
            return row
        n = min(len(sr), n_quarters)
        lcc = sr[:n].download_all()
        if lcc is None or len(lcc) == 0:
            row.update(status="no_data")
            return row
        lc = lcc.stitch().remove_nans()
        if len(lc) < 2000:
            row.update(status="too_short", n_points=len(lc))
            return row

        # NOTE: deliberately NOT running prototype.py-style Savitzky-Golay
        # flattening here. This corpus feeds a SELF-SUPERVISED WORLD MODEL
        # whose whole point (per the guidance doc, Sec 1) is to learn to
        # predict natural stellar variability -- starspot rotation,
        # granulation, pulsations -- from photometric history. A ~301-cadence
        # SG window at Kepler long cadence is a ~6.3-day high-pass filter,
        # which would strip out exactly the multi-day stellar-rotation signal
        # the model is supposed to learn, before it ever sees the data. Only
        # normalize (divide by median) and remove NaNs/extreme outliers
        # (cosmic rays, bad cadences) -- keep the natural baseline intact.
        lc_norm = lc.normalize()
        lc_clean = lc_norm.remove_outliers(sigma_upper=5, sigma_lower=1e4).remove_nans()

        t = np.asarray(lc_clean.time.value, dtype=np.float64)
        f = np.asarray(lc_clean.flux.value, dtype=np.float32)
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        if len(t) < 2000:
            row.update(status="too_short_after_clean", n_points=len(t))
            return row

        dt = np.diff(t, prepend=t[0] - np.median(np.diff(t)) if len(t) > 1 else 1.0)
        dt = np.clip(dt, 1e-4, 10.0).astype(np.float32)

        np.savez_compressed(npz_path, time=t.astype(np.float32), flux=f, dt=dt)
        row.update(n_points=len(t), n_quarters=n, span_days=float(t[-1] - t[0]))
        return row
    except Exception as exc:
        traceback.print_exc()
        row.update(status="error", error_message=f"{type(exc).__name__}: {exc}")
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=300)
    ap.add_argument("--n-quarters", type=int, default=5)
    args = ap.parse_args()

    print("Querying Kepler cumulative KOI table ...", flush=True)
    t = NasaExoplanetArchive.query_criteria(
        table="cumulative", select="kepid,koi_disposition",
        where="koi_disposition = 'CONFIRMED' or koi_disposition = 'CANDIDATE'",
    )
    df = t.to_pandas().drop_duplicates(subset="kepid")
    print(f"  {len(df)} unique KIC host stars available", flush=True)

    rng = np.random.default_rng(42)
    pool = df.sample(min(args.n_targets * 2, len(df)), random_state=42)  # 2x buffer for failures
    kepids = pool["kepid"].astype(int).tolist()

    done = done_set()
    pending = [k for k in kepids if str(k) not in done]
    print(f"Targets: pool={len(kepids)}, already done={len(done)}, pending (up to buffer)={len(pending)}",
          flush=True)

    n_ok = sum(1 for k in done if os.path.exists(os.path.join(CORPUS_DIR, f"{k}.npz")))
    for kepid in pending:
        if n_ok >= args.n_targets:
            print(f"Reached target of {args.n_targets} successful downloads, stopping.")
            break
        print(f"\n[{n_ok}/{args.n_targets}] KIC {kepid} ...", flush=True)
        row = process_one(kepid, args.n_quarters)
        append_row(row)
        if row["status"] == "ok":
            n_ok += 1
            print(f"  OK: {row['n_points']} points, {row['n_quarters']} quarters, "
                  f"{row['span_days']:.1f} day span", flush=True)
        else:
            print(f"  {row['status']}: {row['error_message']}", flush=True)

    print(f"\nBatch complete. {n_ok} successful downloads in {CORPUS_DIR}/")


if __name__ == "__main__":
    main()
