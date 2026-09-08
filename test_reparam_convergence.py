#!/usr/bin/env python3
"""
Validate the (T14, b) reparametrization (not part of the checkpointed batch):
re-fit TOI-700, Pi Mensae (literature ephemeris), and V1828 Aql (BLS
ephemeris) with the new PARAM_NAMES, at the SAME NSTEPS=8000 / init_scale=1.0
used for the original (a_over_rs, b) run -- an apples-to-apples comparison of
tau, not a bigger step budget.

Baseline (old parametrization, NSTEPS=8000) for reference:
  TOI-700:   tau=316  acc=0.28  converged=False
  Pi Mensae (BLS ephem, aliased): tau=258  acc=0.26  converged=False
  V1828 Aql: tau=388  acc=0.23  converged=False

Usage: python test_reparam_convergence.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402

NWALKERS = 32
NSTEPS = 8000          # same as the original core run -- apples-to-apples
MAX_POINTS = 4000

# Larsen et al. 2026 (arXiv:2607.12088), Table 2 -- confirmed in item 1 to
# resolve Pi Mensae onto a physically sensible, stable solution.
PI_MEN_LIT_PERIOD = 6.267823
PI_MEN_LIT_T0 = 1325.5042
PI_MEN_LIT_DUR_HR = 2.969


def fit_target(name, terms, period, t0, dur_hr, depth_guess_ppm):
    dur_d = dur_hr / 24.0
    print(f"\n{'=' * 66}\n  {name}   P={period:.5f} d  T14={dur_hr:.3f} h\n{'=' * 66}", flush=True)
    lc = download_lc(terms[0], terms)
    lc_c = detrend(lc, f"{name}_reparam_test")
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
    print(f"  fitting {len(tt)} cadences", flush=True)

    th_ls, lsq_ok = M.fit_least_squares(tt, ff, ee, period, t0, dur_d, depth_guess_ppm)
    d_ls, du_ls, inc_ls, a_rs_ls = M.derived_quantities(th_ls)
    print(f"  LSQ: depth={d_ls:.1f} ppm  T14={du_ls:.3f} h  b={th_ls[4]:.3f}  "
          f"a/Rs={a_rs_ls:.2f}", flush=True)

    chain, diag = M.run_mcmc(tt, ff, ee, th_ls, period, t0, dur_d,
                             nwalkers=NWALKERS, nsteps=NSTEPS, progress=False)
    if chain is None:
        print(f"  MCMC FAILED TO INITIALISE: {diag}")
        return None
    post = M.summarize_posterior(chain)
    print(f"  MCMC: depth={post['depth_ppm_med']:.1f} "
          f"(-{post['depth_ppm_lo']:.1f}/+{post['depth_ppm_hi']:.1f}) ppm | "
          f"T14={post['duration_hr_med']:.3f} h | b={post['impact_b_med']:.3f} | "
          f"a/Rs={post['a_over_rs_med']:.2f} | "
          f"converged={diag['converged']} (acc={diag['acceptance']:.3f}, "
          f"tau={diag['tau_max']:.1f}, n_samples={diag['n_samples']})", flush=True)
    req = 50 * diag["tau_max"]
    avail = NSTEPS - int(0.4 * NSTEPS)
    print(f"  convergence margin: {avail} available vs {req:.0f} required "
          f"(50*tau) -> {'PASS' if avail > req else 'FAIL'}", flush=True)
    return diag


def main():
    with open(os.path.join(RESULTS, "bls_ephemeris.json"), encoding="utf-8") as fh:
        eph = json.load(fh)

    results = {}

    tgt = next(t for t in TARGETS if t["name"] == "TOI-700")
    e = eph["TOI-700"]
    results["TOI-700"] = fit_target("TOI-700", tgt["terms"], e["period"], e["t0"],
                                     e["duration"] * 24.0, 2247.17)

    tgt = next(t for t in TARGETS if t["name"] == "Pi Mensae")
    results["Pi Mensae (lit)"] = fit_target("Pi Mensae (literature ephemeris)", tgt["terms"],
                                             PI_MEN_LIT_PERIOD, PI_MEN_LIT_T0, PI_MEN_LIT_DUR_HR,
                                             300.0)

    tgt = next(t for t in TARGETS if t["name"] == "V1828 Aql")
    e = eph["V1828 Aql"]
    results["V1828 Aql"] = fit_target("V1828 Aql", tgt["terms"], e["period"], e["t0"],
                                       e["duration"] * 24.0, 25480.52)

    print(f"\n{'=' * 66}\n  SUMMARY (NSTEPS={NSTEPS}, new T14/b parametrization)\n{'=' * 66}")
    for name, diag in results.items():
        if diag is None:
            print(f"  {name:20s} MCMC init failed")
            continue
        print(f"  {name:20s} tau={diag['tau_max']:7.1f}  acc={diag['acceptance']:.3f}  "
              f"converged={diag['converged']}")


if __name__ == "__main__":
    main()
