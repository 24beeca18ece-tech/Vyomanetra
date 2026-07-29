#!/usr/bin/env python3
"""
Recompute BLS (period, t0, duration) for each target by calling prototype.py's
own download_lc() / detrend() / identify() -- prototype.py itself is not
modified. summary_table.csv only stores period_days (t0/epoch was never
written out by prototype.py), and phase-folding ExoVeil's single-transit
event times against the BLS ephemeris needs t0 too.

BLS's period grid is deterministic given the same cleaned flux array, so the
recomputed period should reproduce results/summary_table.csv's period_days
exactly (sanity-checked in main()).

Usage: python compute_bls_ephemeris.py
Writes results/bls_ephemeris.json
"""
import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, identify, RESULTS  # noqa: E402


def main():
    with open(os.path.join(RESULTS, "summary_table.csv"), encoding="utf-8") as f:
        known_period = {r["target"]: float(r["period_days"]) for r in csv.DictReader(f)}

    ephem = {}
    for tgt in TARGETS:
        name, terms = tgt["name"], tgt["terms"]
        print(f"\n{'=' * 62}\n  Recomputing BLS ephemeris: {name}\n{'=' * 62}", flush=True)
        lc = download_lc(name, terms)
        if lc is None:
            print(f"  *** SKIPPED -- no data for {name}")
            continue
        lc_c = detrend(lc, name)
        bls = identify(lc_c, name)

        rec = {"period": float(bls["period"]), "t0": float(bls["t0"]),
               "duration": float(bls["duration"]), "power": float(bls["power"])}
        ephem[name] = rec

        prev = known_period.get(name)
        match = "OK" if prev is not None and abs(prev - rec["period"]) < 1e-6 else "MISMATCH"
        print(f"  period={rec['period']:.6f} d  t0={rec['t0']:.6f} BTJD  "
              f"dur={rec['duration']*24:.3f} h  "
              f"[vs summary_table.csv period {prev}: {match}]", flush=True)

    out = os.path.join(RESULTS, "bls_ephemeris.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ephem, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
