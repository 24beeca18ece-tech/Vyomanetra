#!/usr/bin/env python3
"""
VyomAnetra -- ExoVeil baseline comparison
------------------------------------------
Runs Priyanshu (2026) ExoVeil (arXiv:2606.02778, pip install exoveil) on the
same four TESS targets used in prototype.py, so the paper can report a real
head-to-head table against the classical BLS/trapezoid pipeline instead of
just describing ExoVeil in prose.

Setup (do this in your own venv -- the exoveil install pulls in PyTorch,
so budget 5-15 min depending on your connection):

    pip install exoveil

Usage:
    python run_exoveil_baseline.py            # all 4 targets
    python run_exoveil_baseline.py --target "TOI-700"

Notes / things to sanity-check when you actually run this (I could not
execute it myself -- see chat):
  - ExoVeil's `detect_from_array(time, flux)` signature and return fields
    (snr, depth_ppm, category, ...) are taken from the paper's usage
    snippet and Section 3-5 description. Print `result.__dict__` (or
    `vars(result)` / `result` directly) the first time you run this and
    adjust FIELD NAMES below if they don't match -- the PyPI package
    (v0.2.1) may have evolved slightly since the paper text was written.
  - ExoVeil is Kepler-trained; it claims zero-shot transfer to TESS
    (2-min cadence) without retraining, which is exactly the regime of
    these 4 targets. V1828 Aql (P=0.66 d, an eclipsing binary) is well
    outside what ExoVeil was validated on (it's built/tested for planet
    transits) -- expect it to either flag something totally different or
    fail informatively. That's a legitimate, reportable result either way.
  - This reuses download_lc() + detrend() from prototype.py so both
    pipelines see the *same* cleaned flux array -- apples to apples.
"""

import os
import sys
import csv
import json
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, RESULTS  # noqa: E402

try:
    from exoveil import ExoVeil
except ImportError:
    print("ERROR: exoveil not installed. Run:  pip install exoveil")
    sys.exit(1)


def run_one(model, tgt):
    name, terms = tgt["name"], tgt["terms"]
    print(f"\n{'=' * 62}\n  TARGET: {name}\n{'=' * 62}", flush=True)

    lc = download_lc(name, terms)
    if lc is None:
        print(f"  *** SKIPPED -- no TESS data found for {name}")
        return None
    lc_c = detrend(lc, name)

    t = lc_c.time.value
    f = lc_c.flux.value
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]

    try:
        result = model.detect_from_array(t, f)
    except Exception as exc:
        print(f"  *** ExoVeil failed on {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None

    # CONFIRMED against exoveil 0.2.1 (PyPI): detect_from_array() returns a
    # dict with keys {events, n_events, time, residual, predicted, variance}.
    # It does NOT match the paper's usage snippet (no top-level snr/depth_ppm/
    # category/period/t0 attributes, no XGBoost planet/FP classification, no
    # conformal interval exposed in this version). It's genuinely single-
    # transit detection: `events` is a list of per-event dicts (each with
    # time, snr, duration_pts, depth_ppm, near_gap, aleatoric,
    # uncertainty_category), sorted by SNR descending, with NO orbital period
    # or epoch -- there is nothing to compare 1:1 against BLS's period/t0.
    print(f"  raw result keys: {result.keys() if isinstance(result, dict) else type(result)}")
    print(f"  n_events={result.get('n_events') if isinstance(result, dict) else 'n/a'}")

    row = {"target": name}
    if isinstance(result, dict) and "error" in result:
        row["n_events"] = 0
        row["error"] = result["error"]
        for k in ("top_time_btjd", "top_snr", "top_depth_ppm", "top_duration_pts",
                  "top_uncertainty_category", "top_near_gap"):
            row[k] = None
    else:
        events = result.get("events", []) if isinstance(result, dict) else []
        row["n_events"] = len(events)
        row["error"] = None
        top = events[0] if events else None
        row["top_time_btjd"] = top["time"] if top else None
        row["top_snr"] = top["snr"] if top else None
        row["top_depth_ppm"] = top["depth_ppm"] if top else None
        row["top_duration_pts"] = top["duration_pts"] if top else None
        row["top_uncertainty_category"] = top["uncertainty_category"] if top else None
        row["top_near_gap"] = top["near_gap"] if top else None
        row["all_events"] = json.dumps(events)
    print(f"  ExoVeil result: {row}")
    return row


def main():
    only = None
    if "--target" in sys.argv:
        idx = sys.argv.index("--target")
        only = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    print("Loading ExoVeil pretrained model ...", flush=True)
    model = ExoVeil.from_pretrained()

    rows = []
    for tgt in TARGETS:
        if only and tgt["name"].lower() != only.lower():
            continue
        r = run_one(model, tgt)
        if r is not None:
            rows.append(r)

    out_csv = os.path.join(RESULTS, "exoveil_baseline.csv")
    if rows:
        cols = ["target", "n_events", "top_time_btjd", "top_snr", "top_depth_ppm",
                "top_duration_pts", "top_uncertainty_category", "top_near_gap",
                "error", "all_events"]
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {out_csv}")

    out_json = os.path.join(RESULTS, "exoveil_baseline.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"Saved {out_json}")

    print("\nNext: merge exoveil_baseline.csv with results/summary_table.csv")
    print("into one comparison table (BLS+trapezoid vs ExoVeil) for the paper.")


if __name__ == "__main__":
    main()
