#!/usr/bin/env python3
"""
Validate multi-start LSQ (not part of the checkpointed batch) on the worst
grazing-transit offenders from the 25-target benchmark before committing to
a full rerun. For each target: re-fit with fit_least_squares_multistart(),
report the winning b_start/chi2/depth vs the OLD single-start (b=0.3 seed)
result already in results/modules_3_5_results.csv, and vs archive_depth_ppm.

Also spot-checks assumption (2) from the task: does bootstrap_lsq(), which
warm-starts from theta_best and never re-seeds b, actually inherit the fixed
(lower-b) optimum, or does it need its own multi-start too?

Usage: python test_multistart_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402

MAX_POINTS = 4000

# worst grazing-b offenders from the last benchmark run (lsq_impact_b desc)
WORST = ["TOI-5126 b", "LHS 1903 d", "TOI-5126 c", "DS Tuc A b", "TOI-2411 b",
         "HD 108236 c", "TOI-2431 b", "TOI-244 b"]


def main():
    sample = pd.read_csv(os.path.join(RESULTS, "benchmark_sample.csv")).set_index("target")
    old = pd.read_csv(os.path.join(RESULTS, "modules_3_5_results.csv"))
    old = old[old["mode"] == "benchmark"].set_index("target")
    bmr = pd.read_csv(os.path.join(RESULTS, "benchmark_results.csv")).set_index("target")

    print(f"{'target':<14} {'old_b':>7} {'new_b':>7} {'old_depth':>11} {'new_depth':>11} "
          f"{'archive':>10} {'old_chi2':>10} {'new_chi2':>10}", flush=True)

    for name in WORST:
        r = sample.loc[name]
        period, t0_bjd, dur_hr, depth_guess = (float(r["period_d"]), float(r["t0_bjd"]) - 2457000.0,
                                                float(r["duration_hr"]), float(r["depth_ppm"]))
        dur_d = dur_hr / 24.0

        tic_id = str(r["tic_id"])
        try:
            lc = download_lc(tic_id, [tic_id, r.get("hostname")])
        except Exception as exc:
            print(f"{name:<14} download failed: {exc}")
            continue
        if lc is None:
            print(f"{name:<14} no data")
            continue
        lc_c = detrend(lc, f"{name}_multistart_val")
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

        # OLD single-start (b=0.3) for a same-session, same-window comparison
        th_old, ok_old = M.fit_least_squares(tt, ff, ee, period, t0_bjd, dur_d, depth_guess, b_start=0.3)
        m_old = M.transit_model(th_old, tt)
        chi2_old = float(np.sum(((ff - m_old) / ee) ** 2))
        d_old, _, _, _ = M.derived_quantities(th_old)

        # NEW multi-start
        th_new, ok_new, chi2_new, b0_used = M.fit_least_squares_multistart(
            tt, ff, ee, period, t0_bjd, dur_d, depth_guess)
        d_new, _, _, _ = M.derived_quantities(th_new)

        archive_depth = float(bmr.loc[name, "archive_depth_ppm"])
        b_old, b_new = float(th_old[4]), float(th_new[4])

        print(f"{name:<14} {b_old:7.3f} {b_new:7.3f} {d_old:11.1f} {d_new:11.1f} "
              f"{archive_depth:10.1f} {chi2_old:10.1f} {chi2_new:10.1f}"
              f"  (won from b_start={b0_used})", flush=True)

        # spot-check assumption (2): does bootstrap inherit the new optimum
        # without its own multi-start?
        if name == WORST[0]:
            print(f"\n  --- spot-check: does bootstrap_lsq warm-start hold the new optimum? ---")
            boot = M.bootstrap_lsq(tt, ff, ee, th_new, period, t0_bjd, dur_d, depth_guess, n_boot=50)
            print(f"  bootstrap depth (block, n=50): "
                  f"{boot.get('depth_ppm_block_med', float('nan')):.1f} "
                  f"(-{boot.get('depth_ppm_block_lo', float('nan')):.1f}/"
                  f"+{boot.get('depth_ppm_block_hi', float('nan')):.1f}) ppm | "
                  f"impact_b (block): {boot.get('impact_b_block_med', float('nan')):.3f} "
                  f"(-{boot.get('impact_b_block_lo', float('nan')):.3f}/"
                  f"+{boot.get('impact_b_block_hi', float('nan')):.3f})")
            print(f"  (compare to new LSQ point est.: depth={d_new:.1f} ppm, b={b_new:.3f} -- "
                  f"if the bootstrap median stays near b={b_new:.3f} rather than drifting back "
                  f"toward b={b_old:.3f}, the warm-start assumption holds)\n")


if __name__ == "__main__":
    main()
