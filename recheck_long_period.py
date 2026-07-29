#!/usr/bin/env python3
"""
Fairness recheck for the two benchmark targets whose archive period exceeds
the BLS search cap in prototype.py's identify():

    max_p = min(span / 2, 30.0)

  - TOI-2449 b  P = 106.14 d
  - NGTS-20 b   P =  54.19 d

Both are structurally unrecoverable under that cap regardless of data quality,
so counting them as ordinary misses overstates BLS's failure rate. This script
reruns the BLS stage with the cap raised, to separate two distinct causes:

  (a) the hardcoded 30 d ceiling  -> fixable by configuration alone
  (b) the span/2 ceiling          -> NOT fixable by configuration; you need a
                                     longer baseline (more sectors), because
                                     recovering period P requires >=2 transits
                                     and therefore span >= 2P

It runs three configurations per target so the two causes are distinguishable:
  1. baseline    : 4 sectors (as the benchmark used), max_p = min(span/2, 30)
  2. cap-raised  : 4 sectors,            max_p = min(span/2, MAX_P_EXTENDED)
  3. more-sectors: up to N sectors,      max_p = min(span/2, MAX_P_EXTENDED)

Everything else (detrending, duration grid, fold-against-archive test) is held
identical to run_benchmark.py. prototype.py is not modified -- identify() is
re-implemented here only because its max_p is hardcoded; the BLS call, grid
construction and duration grid mirror it exactly, except for the period
ceiling and (in config 3) a denser period grid so that grid resolution is not
itself a confound.

Usage: python recheck_long_period.py
Writes: results/long_period_recheck.csv  (+ console report)
"""
import os
import sys
import csv
import warnings

import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import detrend, RESULTS  # noqa: E402

MAX_P_EXTENDED = 150.0     # comfortably exceeds both targets' archive periods
N_SECTORS_MANY = 20        # config 3: pull a much longer baseline
N_PERIODS_BASE = 10000     # same as prototype.identify()
N_PERIODS_FINE = 60000     # config 2/3: keep grid resolution from confounding

TARGETS = [
    # name, TIC, archive period (d), archive t0 (BJD), archive T14 (h)
    ("TOI-2449 b", "TIC 170729775", 106.14468, 2459153.5934000001, 8.26),
    ("NGTS-20 b",  "TIC 257527578",  54.18915, 2458432.9797999998, 4.55),
]


def bls_identify(lc, max_p_ceiling, n_periods):
    """Mirror of prototype.identify()'s BLS stage with a configurable period
    ceiling and grid density. Returns dict incl. the binding constraint."""
    t, f = lc.time.value, lc.flux.value
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]

    span = t[-1] - t[0]
    min_p = 0.5
    max_p = min(span / 2, max_p_ceiling)
    binding = "span/2" if (span / 2) < max_p_ceiling else "ceiling"

    periods = np.linspace(min_p, max_p, n_periods)
    max_dur = min(0.4, min_p * 0.8)
    durations = np.linspace(0.02, max_dur, 25)

    bls = BoxLeastSquares(t, f)
    res = bls.power(periods, durations)
    idx = np.argmax(res.power)
    return {
        "period": float(res.period[idx]), "t0": float(res.transit_time[idx]),
        "duration": float(res.duration[idx]), "power": float(res.power[idx]),
        "span": float(span), "max_p": float(max_p), "binding": binding,
        "n_points": len(t),
    }


def fold_in_transit(t_epoch, period, t0, half_dur_days):
    raw = ((t_epoch - t0) / period) % 1.0
    centered = raw - 1.0 if raw > 0.5 else raw
    pd_ = centered * period
    return pd_ * 24.0, abs(pd_) <= half_dur_days


def get_lc(tic, n_sectors):
    sr = lk.search_lightcurve(tic, mission="TESS", author="SPOC")
    if sr is None or len(sr) == 0:
        return None, 0
    n = min(len(sr), n_sectors)
    lcc = sr[:n].download_all()
    lc = lcc.stitch().remove_nans()
    return lc, n


def main():
    rows = []
    for name, tic, arch_p, arch_t0_bjd, arch_dur_hr in TARGETS:
        arch_t0 = arch_t0_bjd - 2457000.0
        half_dur = (arch_dur_hr / 24.0) / 2.0
        print(f"\n{'=' * 70}\n  {name}  ({tic})   archive P = {arch_p:.4f} d\n{'=' * 70}", flush=True)

        configs = [
            ("1_baseline_4sec_cap30",  4,              30.0,           N_PERIODS_BASE),
            ("2_capraised_4sec",       4,              MAX_P_EXTENDED, N_PERIODS_FINE),
            ("3_capraised_manysec",    N_SECTORS_MANY, MAX_P_EXTENDED, N_PERIODS_FINE),
        ]

        cache = {}
        for label, n_sec, ceiling, n_per in configs:
            print(f"\n  -- config {label}: <={n_sec} sectors, ceiling {ceiling} d --", flush=True)
            if n_sec not in cache:
                lc, got = get_lc(tic, n_sec)
                if lc is None:
                    print("     *** no data")
                    cache[n_sec] = (None, 0, None)
                else:
                    lc_c = detrend(lc, f"{name}_recheck_{n_sec}sec")
                    cache[n_sec] = (lc, got, lc_c)
            lc, got, lc_c = cache[n_sec]
            if lc_c is None:
                continue

            res = bls_identify(lc_c, ceiling, n_per)
            phase_hr, in_tr = fold_in_transit(res["t0"], arch_p, arch_t0, half_dur)
            reachable = res["max_p"] >= arch_p

            print(f"     sectors used   : {got}")
            print(f"     baseline span  : {res['span']:.2f} d")
            print(f"     effective max_p: {res['max_p']:.2f} d   (binding constraint: {res['binding']})")
            print(f"     archive P reachable? {'YES' if reachable else 'NO -- P > max_p'}")
            print(f"     BLS period     : {res['period']:.5f} d   (archive {arch_p:.5f})")
            print(f"     BLS t0         : {res['t0']:.5f} BTJD")
            print(f"     fold vs archive: {phase_hr:+.3f} h  -> {'IN-TRANSIT' if in_tr else 'out-of-transit'}")

            rows.append({
                "target": name, "tic_id": tic, "config": label,
                "n_sectors_requested": n_sec, "n_sectors_used": got,
                "span_d": f"{res['span']:.4f}", "max_p_d": f"{res['max_p']:.4f}",
                "binding_constraint": res["binding"],
                "archive_period_d": arch_p, "archive_period_reachable": reachable,
                "bls_period_d": f"{res['period']:.6f}", "bls_t0_btjd": f"{res['t0']:.6f}",
                "bls_power": f"{res['power']:.6g}",
                "phase_offset_hr": f"{phase_hr:.4f}", "in_transit": in_tr,
            })

    out = os.path.join(RESULTS, "long_period_recheck.csv")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
