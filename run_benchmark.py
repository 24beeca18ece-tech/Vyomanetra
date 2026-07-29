#!/usr/bin/env python3
"""
Benchmark ExoVeil vs. the BLS/trapezoid pipeline over a depth-stratified
sample of confirmed TESS planets (results/benchmark_sample.csv, produced by
select_benchmark_sample.py) -- a real recovery-rate statistic instead of the
4-target anecdote in run_exoveil_baseline.py / merge_comparison.py.

For each target: download TESS SPOC 2-min light curves (generalized from
prototype.py's download_lc(), keyed on TIC ID instead of a hardcoded name
list), run prototype.py's own detrend() -> identify() -> characterize() ->
classify() -> significance() BLS/trapezoid chain, and separately run
ExoVeil's detect_from_array() on the identical cleaned flux array. Both
pipelines' derived timing is folded against the NASA Exoplanet Archive's own
pl_orbper/pl_tranmid/pl_trandur for that target -- the actual ground truth,
not each other.

Checkpointed: each target's full result row is appended to
results/benchmark_results.csv immediately after that target finishes (file
opened/closed per row, so a crash loses at most the in-flight target).
Targets already present in that CSV are skipped on rerun -- safe to Ctrl-C
and resume. Per-target failures (no data, download error, BLS error, ExoVeil
error) are caught individually and logged as a row with a `status` field
rather than aborting the batch.

prototype.py is not modified -- only imported.

Usage:
    python run_benchmark.py                # process all targets in the sample
    python run_benchmark.py --retry-failed  # also re-attempt non-"ok" rows
    python run_benchmark.py --limit 5       # process only the next 5 pending targets
"""
import os
import sys
import csv
import json
import argparse
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import (  # noqa: E402
    download_lc as _download_lc_by_terms,
    detrend, identify, characterize, classify, significance, RESULTS,
)

try:
    from exoveil import ExoVeil
except ImportError:
    print("ERROR: exoveil not installed. Run:  pip install exoveil")
    sys.exit(1)

SAMPLE_CSV = os.path.join(RESULTS, "benchmark_sample.csv")
RESULTS_CSV = os.path.join(RESULTS, "benchmark_results.csv")

# Local matched-filter SNR blow-up guard (same rationale as merge_comparison.py's
# SNR_SANITY_MAX): exoveil.detect.detect_events() can divide by a near-zero
# local-noise estimate on sharp, high-contrast eclipses (deep decile targets
# in this sample can be EB-like). Flag rather than silently trust.
SNR_SANITY_MAX = 1000.0

CSV_COLS = [
    "target", "tic_id", "status", "error_message",
    "depth_decile", "archive_period_d", "archive_t0_btjd", "archive_duration_hr",
    "archive_depth_ppm",
    "n_sectors_used", "n_cadences",
    "bls_period_d", "bls_t0_btjd", "bls_depth_ppm", "bls_snr", "bls_flag",
    "bls_phase_offset_hr", "bls_in_transit",
    "exoveil_n_events", "exoveil_top_snr_raw", "exoveil_snr_flag",
    "exoveil_top_depth_ppm", "exoveil_top_time_btjd",
    "exoveil_any_in_transit", "exoveil_n_events_in_transit",
]


def download_lc_by_tic(tic_id, hostname=None, max_sectors=4):
    """Generalized version of prototype.py's download_lc(): search by TIC ID
    first (exact, unambiguous across the whole sample), fall back to hostname.
    Returns (lc, n_sectors_used) or (None, 0).
    """
    terms = [tic_id] + ([hostname] if hostname else [])
    lc = _download_lc_by_terms(tic_id, terms)
    if lc is None:
        return None, 0
    # download_lc() doesn't report sector count directly; approximate via
    # the number of distinct calendar gaps is unreliable, so just report
    # "downloaded" -- exact sector list isn't needed for the ground-truth
    # fold, only the cleaned flux array is.
    return lc, None


def fold_phase(t_event, period, t0, half_dur_days):
    raw_phase = ((t_event - t0) / period) % 1.0
    centered = raw_phase - 1.0 if raw_phase > 0.5 else raw_phase
    phase_days = centered * period
    return phase_days, phase_days * 24.0, abs(phase_days) <= half_dur_days


def already_processed(retry_failed):
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    if retry_failed:
        return set(df.loc[df["status"] == "ok", "target"])
    return set(df["target"])


def append_row(row):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def blank_row(target, tic_id, decile, arch_period, arch_t0, arch_dur, arch_depth, status, err=""):
    row = {c: "" for c in CSV_COLS}
    row.update({
        "target": target, "tic_id": tic_id, "status": status, "error_message": err,
        "depth_decile": decile, "archive_period_d": arch_period,
        "archive_t0_btjd": arch_t0, "archive_duration_hr": arch_dur,
        "archive_depth_ppm": arch_depth,
    })
    return row


def process_one(model, target, tic_id, hostname, decile, arch_period_d, arch_t0_bjd, arch_dur_hr, arch_depth_ppm):
    print(f"\n{'=' * 62}\n  TARGET: {target}  ({tic_id})  decile={decile}\n{'=' * 62}", flush=True)
    arch_t0_btjd = arch_t0_bjd - 2457000.0
    half_dur_days = (arch_dur_hr / 24.0) / 2.0

    # -- download + detrend ------------------------------------------------
    try:
        lc, _ = download_lc_by_tic(tic_id, hostname)
    except Exception as exc:
        traceback.print_exc()
        return blank_row(target, tic_id, decile, arch_period_d, arch_t0_btjd, arch_dur_hr,
                          arch_depth_ppm, "download_error", f"{type(exc).__name__}: {exc}")
    if lc is None:
        print(f"  *** SKIPPED -- no TESS SPOC data found for {target}")
        return blank_row(target, tic_id, decile, arch_period_d, arch_t0_btjd, arch_dur_hr,
                          arch_depth_ppm, "no_data")

    try:
        lc_c = detrend(lc, target)
    except Exception as exc:
        traceback.print_exc()
        return blank_row(target, tic_id, decile, arch_period_d, arch_t0_btjd, arch_dur_hr,
                          arch_depth_ppm, "detrend_error", f"{type(exc).__name__}: {exc}")

    n_cadences = len(lc_c)

    # -- BLS / trapezoid / classify / significance --------------------------
    bls_row = {}
    try:
        bls = identify(lc_c, target)
        fit = characterize(bls["lc_fold"], bls, target)
        clf_row = {"depth_ppm": fit["depth_ppm"], "period": bls["period"]}
        clf = classify(clf_row)
        sig = significance(bls["lc_fold"], bls, fit)

        bls_t0_btjd = bls["t0"]
        _, phase_hr, in_transit = fold_phase(bls_t0_btjd, arch_period_d, arch_t0_btjd, half_dur_days)

        bls_row = {
            "bls_period_d": bls["period"], "bls_t0_btjd": bls_t0_btjd,
            "bls_depth_ppm": fit["depth_ppm"], "bls_snr": sig["snr"], "bls_flag": clf["flag"],
            "bls_phase_offset_hr": f"{phase_hr:.4f}", "bls_in_transit": in_transit,
        }
    except Exception as exc:
        traceback.print_exc()
        print(f"  *** BLS pipeline failed on {target}: {exc}")
        bls_row = {"bls_period_d": "", "bls_t0_btjd": "", "bls_depth_ppm": "",
                   "bls_snr": "", "bls_flag": f"BLS_ERROR: {exc}",
                   "bls_phase_offset_hr": "", "bls_in_transit": ""}

    # -- ExoVeil --------------------------------------------------------
    exo_row = {}
    try:
        t = lc_c.time.value
        f = lc_c.flux.value
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        result = model.detect_from_array(t, f)
        events = result.get("events", []) if isinstance(result, dict) else []

        folded = []
        for ev in events:
            _, ph, intr = fold_phase(ev["time"], arch_period_d, arch_t0_btjd, half_dur_days)
            folded.append({**ev, "phase_hr": ph, "in_transit": intr})
        folded.sort(key=lambda e: e["snr"], reverse=True)
        n_in_transit = sum(1 for e in folded if e["in_transit"])
        top = folded[0] if folded else None
        top_snr = top["snr"] if top else None
        snr_flag = "SNR undefined (near-zero local noise)" if (top_snr is not None and top_snr > SNR_SANITY_MAX) else ""

        exo_row = {
            "exoveil_n_events": len(folded),
            "exoveil_top_snr_raw": f"{top_snr:.6g}" if top_snr is not None else "",
            "exoveil_snr_flag": snr_flag,
            "exoveil_top_depth_ppm": top["depth_ppm"] if top else "",
            "exoveil_top_time_btjd": top["time"] if top else "",
            "exoveil_any_in_transit": n_in_transit > 0,
            "exoveil_n_events_in_transit": n_in_transit,
        }
    except Exception as exc:
        traceback.print_exc()
        print(f"  *** ExoVeil failed on {target}: {exc}")
        exo_row = {"exoveil_n_events": "", "exoveil_top_snr_raw": "", "exoveil_snr_flag": f"EXOVEIL_ERROR: {exc}",
                   "exoveil_top_depth_ppm": "", "exoveil_top_time_btjd": "",
                   "exoveil_any_in_transit": "", "exoveil_n_events_in_transit": ""}

    row = blank_row(target, tic_id, decile, arch_period_d, arch_t0_btjd, arch_dur_hr, arch_depth_ppm, "ok")
    row["n_cadences"] = n_cadences
    row.update(bls_row)
    row.update(exo_row)
    print(f"  BLS: P={bls_row.get('bls_period_d')} in_transit={bls_row.get('bls_in_transit')} flag={bls_row.get('bls_flag')}")
    print(f"  ExoVeil: n_events={exo_row.get('exoveil_n_events')} any_in_transit={exo_row.get('exoveil_any_in_transit')}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(SAMPLE_CSV):
        print(f"ERROR: {SAMPLE_CSV} not found. Run select_benchmark_sample.py first.")
        sys.exit(1)
    sample = pd.read_csv(SAMPLE_CSV)

    done = already_processed(args.retry_failed)
    pending = sample[~sample["target"].isin(done)]
    if args.limit:
        pending = pending.head(args.limit)

    print(f"Sample: {len(sample)} targets. Already done: {len(done)}. "
          f"Pending this run: {len(pending)}.", flush=True)
    if len(pending) == 0:
        print("Nothing to do.")
        return

    print("Loading ExoVeil pretrained model ...", flush=True)
    model = ExoVeil.from_pretrained()

    for _, r in pending.iterrows():
        try:
            row = process_one(
                model, r["target"], r["tic_id"], r.get("hostname"), r["depth_decile"],
                float(r["period_d"]), float(r["t0_bjd"]), float(r["duration_hr"]), float(r["depth_ppm"]),
            )
        except Exception as exc:
            traceback.print_exc()
            row = blank_row(r["target"], r["tic_id"], r["depth_decile"], r["period_d"],
                             float(r["t0_bjd"]) - 2457000.0, r["duration_hr"], r["depth_ppm"],
                             "unexpected_error", f"{type(exc).__name__}: {exc}")
        append_row(row)
        print(f"  -> appended to {RESULTS_CSV}", flush=True)

    print("\nDone with this batch.")


if __name__ == "__main__":
    main()
