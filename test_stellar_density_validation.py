#!/usr/bin/env python3
"""
Bounded test (not part of the checkpointed batch): does fixing a_over_rs
from a stellar-density prior (Kepler's third law, TIC radius/mass) resolve
the identifiability problem that multi-start LSQ could not?

Same 8 worst grazing-b offenders as test_multistart_validation.py. For each:
  1. Pull TIC radius/mass (astroquery, same pattern as the ExoTETHyS LD
     lookup) -> rho_star_solar = mass/radius^3 -> a_over_rs via Kepler III.
  2. Refit with a_over_rs FIXED at that value (fit_least_squares_fixed_a_rs,
     still multi-start over b since b is the one remaining free geometry
     parameter).
  3. Report depth vs. archive, and chi2 vs. the free-a_over_rs (T14,b)
     fit's chi2 on the SAME window, to check whether fixing a_over_rs
     actually differentiates the chi-square surface or just relocates to an
     equally-flat optimum.

Usage: python test_stellar_density_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402

MAX_POINTS = 4000

WORST = ["TOI-5126 b", "LHS 1903 d", "TOI-5126 c", "DS Tuc A b", "TOI-2411 b",
         "HD 108236 c", "TOI-2431 b", "TOI-244 b"]


def get_stellar_density(tic_id):
    # tic_id from benchmark_sample.csv already includes the "TIC " prefix
    t = Catalogs.query_object(str(tic_id), catalog="TIC", radius=0.02)
    row = t[0]
    rad, mass = float(row["rad"]), float(row["mass"])
    if not (np.isfinite(rad) and np.isfinite(mass) and rad > 0):
        raise ValueError(f"invalid TIC rad/mass for TIC {tic_id}: rad={rad}, mass={mass}")
    rho_solar = mass / rad ** 3
    return rho_solar, rad, mass


def main():
    sample = pd.read_csv(os.path.join(RESULTS, "benchmark_sample.csv")).set_index("target")
    bmr = pd.read_csv(os.path.join(RESULTS, "benchmark_results.csv")).set_index("target")

    print(f"{'target':<14} {'rho*':>6} {'a/Rs_fix':>9} {'old_b':>6} {'new_b':>6} "
          f"{'old_depth':>10} {'new_depth':>10} {'archive':>9} {'old_chi2':>9} {'new_chi2':>9}",
          flush=True)

    for name in WORST:
        r = sample.loc[name]
        period, t0_bjd, dur_hr, depth_guess = (float(r["period_d"]), float(r["t0_bjd"]) - 2457000.0,
                                                float(r["duration_hr"]), float(r["depth_ppm"]))
        dur_d = dur_hr / 24.0
        tic_id = str(r["tic_id"])

        try:
            rho_solar, rad, mass = get_stellar_density(tic_id)
            a_rs_fixed = M.stellar_density_to_a_rs(rho_solar, period)
        except Exception as exc:
            print(f"{name:<14} stellar density lookup FAILED: {exc}", flush=True)
            continue

        try:
            lc = download_lc(tic_id, [tic_id, r.get("hostname")])
        except Exception as exc:
            print(f"{name:<14} download failed: {exc}")
            continue
        if lc is None:
            print(f"{name:<14} no data")
            continue
        lc_c = detrend(lc, f"{name}_rhostar_val")
        t = np.asarray(lc_c.time.value, dtype=float)
        f = np.asarray(lc_c.flux.value, dtype=float)
        try:
            ferr = np.asarray(lc_c.flux_err.value, dtype=float)
        except Exception:
            ferr = np.full_like(f, np.nan)
        ok = np.isfinite(t) & np.isfinite(f)
        t, f, ferr = t[ok], f[ok], ferr[ok]
        if not np.all(np.isfinite(ferr)) or np.nanmedian(ferr) <= 0:
            ferr = np.full_like(f, np.nanstd(f))

        tt, ff, ee, _ = M.select_transit_window(t, f, ferr, period, t0_bjd, dur_d, max_points=MAX_POINTS)

        # OLD: free a_over_rs (T14,b) multi-start, for a same-window comparison
        th_old, ok_old, chi2_old, b0_old = M.fit_least_squares_multistart(
            tt, ff, ee, period, t0_bjd, dur_d, depth_guess)
        d_old, _, _, _ = M.derived_quantities(th_old)
        b_old = float(th_old[4])

        # NEW: a_over_rs fixed from stellar density
        th_new, ok_new, chi2_new, b0_new = M.fit_least_squares_fixed_a_rs(
            tt, ff, ee, period, t0_bjd, dur_d, depth_guess, a_rs_fixed)
        if th_new is None:
            print(f"{name:<14} fixed-a/Rs fit FAILED to converge from any start", flush=True)
            continue
        d_new, _, _ = M.derived_quantities_fixed_a_rs(th_new, a_rs_fixed)
        b_new = float(th_new[3])

        archive_depth = float(bmr.loc[name, "archive_depth_ppm"])

        print(f"{name:<14} {rho_solar:6.2f} {a_rs_fixed:9.2f} {b_old:6.3f} {b_new:6.3f} "
              f"{d_old:10.1f} {d_new:10.1f} {archive_depth:9.1f} {chi2_old:9.1f} {chi2_new:9.1f}",
              flush=True)


if __name__ == "__main__":
    main()
