#!/usr/bin/env python3
"""
Diagnostic (not part of the checkpointed batch): does Pi Mensae's bad Module-3
fit trace back to a bad (aliased) input ephemeris, or to Module 3 itself?

results/modules_3_5_results.csv already has a Pi Mensae row fit against
results/bls_ephemeris.json (P=27.214921 d -- the BLS pipeline's own aliased
solution, previously shown in the ExoVeil ephemeris check to NOT match the
true planet). This script re-runs the *identical* Module 3 code (same
NWALKERS/NSTEPS as run_modules_3_5.py's core mode, no convergence-fix changes
yet) on Pi Mensae only, but seeded with the literature ephemeris for Pi Men c
from Larsen et al. 2026 (arXiv:2607.12088, Table 2) -- the same constants
merge_comparison.py already uses for the ExoVeil phase-fold check:
    P  = 6.267823 d
    T0 = 1325.5042 BTJD
    T14 = 2.969 h

If the MCMC converges better and the recovered depth lands near the
literature's ~300 ppm (mini-Neptune, Rp~2.06 Re around a Sun-like star) under
the correct ephemeris, that is strong evidence the earlier failure was
aliased-ephemeris-in/garbage-out, not a Module 3 modelling bug.

Usage: python pi_men_alias_check.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402

# Larsen et al. 2026 (arXiv:2607.12088), Table 2 -- same constants as
# merge_comparison.py's PI_MEN_C_* block.
LIT_PERIOD = 6.267823
LIT_T0_BTJD = 1325.5042
LIT_DUR_HR = 2.969
LIT_DEPTH_GUESS_PPM = 300.0  # (Rp/Rs)^2 for Rp~2.06 Re, Rs~1.1 Rsun

NWALKERS = 32
NSTEPS = 8000        # same as run_modules_3_5.py's current core-mode setting
MAX_POINTS = 4000    # same as run_modules_3_5.py


def main():
    tgt = next(t for t in TARGETS if t["name"] == "Pi Mensae")
    print(f"Downloading Pi Mensae ...", flush=True)
    lc = download_lc(tgt["name"], tgt["terms"])
    lc_c = detrend(lc, "Pi_Mensae_alias_check")
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

    dur_d = LIT_DUR_HR / 24.0
    tt, ff, ee, _ = M.select_transit_window(t, f, ferr, LIT_PERIOD, LIT_T0_BTJD, dur_d,
                                            max_points=MAX_POINTS)
    print(f"fitting {len(tt)} cadences (of {len(t)}) under LITERATURE ephemeris "
          f"P={LIT_PERIOD} d T0={LIT_T0_BTJD} BTJD", flush=True)

    th_ls, lsq_ok = M.fit_least_squares(tt, ff, ee, LIT_PERIOD, LIT_T0_BTJD, dur_d,
                                         LIT_DEPTH_GUESS_PPM)
    d_ls, du_ls, inc_ls, a_rs_ls = M.derived_quantities(th_ls)
    b_ls = float(th_ls[4])
    print(f"LSQ: ok={lsq_ok}  depth={d_ls:.1f} ppm  T14={du_ls:.3f} h  "
          f"b={b_ls:.3f} (inc={inc_ls:.3f} deg)", flush=True)

    chain, diag = M.run_mcmc(tt, ff, ee, th_ls, LIT_PERIOD, LIT_T0_BTJD, dur_d,
                             nwalkers=NWALKERS, nsteps=NSTEPS)
    if chain is None:
        print(f"MCMC FAILED TO INITIALISE: {diag}")
        return

    post = M.summarize_posterior(chain)
    print(f"MCMC: depth={post['depth_ppm_med']:.1f} "
          f"(-{post['depth_ppm_lo']:.1f}/+{post['depth_ppm_hi']:.1f}) ppm | "
          f"T14={post['duration_hr_med']:.3f} h | b={post['impact_b_med']:.3f} | "
          f"converged={diag['converged']} (acc={diag['acceptance']:.2f}, "
          f"tau={diag['tau_max']:.0f}, n_samples={diag['n_samples']})", flush=True)

    th_best = np.median(chain, axis=0)
    fap = M.bootstrap_fap(tt, ff, th_best, dur_d, n_boot=2000)
    print(f"FAP: snr={fap['snr_obs']:.2f}  iid={fap['fap_iid']:.4g}  "
          f"block={fap['fap_block']:.4g}", flush=True)

    out = {
        "ephem_source": "literature (arXiv:2607.12088, Table 2)",
        "period": LIT_PERIOD, "t0": LIT_T0_BTJD, "duration_hr": LIT_DUR_HR,
        "n_fit_points": len(tt),
        "lsq_ok": lsq_ok, "lsq_depth_ppm": d_ls, "lsq_duration_hr": du_ls, "lsq_impact_b": b_ls,
        "mcmc_converged": diag["converged"], "mcmc_acceptance": diag["acceptance"],
        "mcmc_tau_max": diag["tau_max"], "mcmc_n_samples": diag["n_samples"],
        "depth_ppm_med": post["depth_ppm_med"], "depth_ppm_lo": post["depth_ppm_lo"],
        "depth_ppm_hi": post["depth_ppm_hi"],
        "duration_hr_med": post["duration_hr_med"],
        "impact_b_med": post["impact_b_med"],
        "a_over_rs_med": post["a_over_rs_med"], "a_over_rs_lo": post["a_over_rs_lo"],
        "a_over_rs_hi": post["a_over_rs_hi"],
        "snr_matched_filter": fap["snr_obs"], "fap_iid": fap["fap_iid"], "fap_block": fap["fap_block"],
    }
    out_path = os.path.join(RESULTS, "pi_men_alias_check.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
