#!/usr/bin/env python3
"""
Convergence-fix validation (not part of the checkpointed batch): re-fit
TOI-700 -- the "good" core target -- with a 3x-tighter walker init ball
(init_scale=0.3) and NSTEPS raised from 8000 to 18000, to check whether
mcmc_converged actually flips True before committing this step count to all
4 (and later 25) targets.

Usage: python test_toi700_convergence.py
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402

NWALKERS = 32
NSTEPS = 18000
INIT_SCALE = 0.3
MAX_POINTS = 4000


def main():
    with open(os.path.join(RESULTS, "bls_ephemeris.json"), encoding="utf-8") as fh:
        eph = json.load(fh)["TOI-700"]
    period, t0, dur_d = eph["period"], eph["t0"], eph["duration"]

    tgt = next(t for t in TARGETS if t["name"] == "TOI-700")
    lc = download_lc(tgt["name"], tgt["terms"])
    lc_c = detrend(lc, "TOI-700_convtest")
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

    tt, ff, ee, _ = M.select_transit_window(t, f, ferr, period, t0, dur_d, max_points=MAX_POINTS)
    print(f"fitting {len(tt)} cadences", flush=True)

    th_ls, lsq_ok = M.fit_least_squares(tt, ff, ee, period, t0, dur_d, 2247.17)
    d_ls, du_ls, inc_ls, a_rs_ls = M.derived_quantities(th_ls)
    print(f"LSQ: depth={d_ls:.1f} ppm  T14={du_ls:.3f} h  b={th_ls[4]:.3f}", flush=True)

    t_start = time.time()
    chain, diag = M.run_mcmc(tt, ff, ee, th_ls, period, t0, dur_d,
                             nwalkers=NWALKERS, nsteps=NSTEPS, init_scale=INIT_SCALE,
                             progress=False)
    elapsed = time.time() - t_start
    print(f"\nran {NSTEPS} steps, init_scale={INIT_SCALE} in {elapsed:.1f} s", flush=True)
    if chain is None:
        print(f"MCMC FAILED TO INITIALISE: {diag}")
        return
    post = M.summarize_posterior(chain)
    print(f"MCMC: depth={post['depth_ppm_med']:.1f} "
          f"(-{post['depth_ppm_lo']:.1f}/+{post['depth_ppm_hi']:.1f}) ppm | "
          f"converged={diag['converged']} (acc={diag['acceptance']:.3f}, "
          f"tau={diag['tau_max']:.1f}, n_samples={diag['n_samples']})", flush=True)
    print(f"required for convergence: (nsteps - burn) > 50*tau -> "
          f"{NSTEPS - int(0.4*NSTEPS)} > {50*diag['tau_max']:.0f} = "
          f"{(NSTEPS - int(0.4*NSTEPS)) > 50*diag['tau_max']}", flush=True)


if __name__ == "__main__":
    main()
