#!/usr/bin/env python3
"""
Test the limb-darkening hypothesis (not part of the checkpointed batch): fix
q1/q2 to tabulated stellar-atmosphere values (ExoTETHyS tables, accessed via
pylightcurve's plc.exotethys()) instead of sampling them, dropping NDIM from
8 to 6. Same apples-to-apples check as the (T14,b) reparametrization test:
NSTEPS=8000 on TOI-700 first.

Stellar parameters (Teff, logg, [Fe/H]) pulled from the TIC catalog via
astroquery.mast.Catalogs (queried once; results hardcoded below to avoid a
live query inside the timed run):
  TOI-700   : TIC 150428135  Teff=3494.0 K  logg=4.809  [Fe/H]=nan -> 0.0
  Pi Mensae : TIC 261136679  Teff=5992.1 K  logg=4.359  [Fe/H]=0.089
  V1828 Aql : TIC 404635917  Teff=40000.0 K logg=5.499  [Fe/H]=nan -> 0.0
    ** V1828 Aql's TIC Teff (40000 K) is far outside both the PHOENIX
       (2300-10000 K) and ATLAS (3500-10000 K) grids exotethys ships --
       expected for TIC's automated pipeline on an eclipsing-binary source.
       Clamped to 9900 K as a documented, low-confidence approximation for
       this diagnostic only; NOT a trustworthy stellar characterization.

Gate: run TOI-700 only first. If tau drops comfortably below the ~160
needed for 50*tau < 4800 (post-burn budget at NSTEPS=8000), continue to Pi
Mensae (literature ephemeris) and V1828 Aql. Otherwise stop and report --
per instructions, do not proceed to dynesty without checking in first.

Usage: python test_ld_fixed_convergence.py [--all]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402
import pylightcurve as plc  # noqa: E402

NWALKERS = 32
NSTEPS = 8000
MAX_POINTS = 4000

PI_MEN_LIT_PERIOD = 6.267823
PI_MEN_LIT_T0 = 1325.5042
PI_MEN_LIT_DUR_HR = 2.969

STELLAR_PARAMS = {
    "TOI-700":   dict(teff=3494.0, logg=4.80892, mh=0.0),
    "Pi Mensae": dict(teff=5992.1, logg=4.3589, mh=0.0887839),
    "V1828 Aql": dict(teff=9900.0, logg=5.49946, mh=0.0),  # Teff clamped, see docstring
}


def get_ld(teff, logg, mh, label):
    for model in ("phoenix", "atlas"):
        try:
            res = plc.exotethys(logg, teff, mh, "TESS", method="quad", stellar_model=model)
            print(f"  LD lookup [{label}] via {model}: u1={res[0]:.4f} u2={res[1]:.4f}", flush=True)
            return float(res[0]), float(res[1])
        except plc.PyLCError as e:
            print(f"  LD lookup [{label}] {model} failed: {e}", flush=True)
    raise RuntimeError(f"no LD table covers {label}'s stellar parameters")


def fit_target(name, terms, period, t0, dur_hr, depth_guess_ppm, u1, u2):
    dur_d = dur_hr / 24.0
    print(f"\n{'=' * 66}\n  {name}   P={period:.5f} d  T14={dur_hr:.3f} h  "
          f"[fixed LD: u1={u1:.4f} u2={u2:.4f}]\n{'=' * 66}", flush=True)
    lc = download_lc(terms[0], terms)
    lc_c = detrend(lc, f"{name}_ldfix_test")
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

    M.set_fixed_limb_darkening(u1, u2)
    try:
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
              f"converged={diag['converged']} (acc={diag['acceptance']:.3f}, "
              f"tau={diag['tau_max']:.1f}, n_samples={diag['n_samples']})", flush=True)
        req = 50 * diag["tau_max"]
        avail = NSTEPS - int(0.4 * NSTEPS)
        print(f"  convergence margin: {avail} available vs {req:.0f} required "
              f"(50*tau) -> {'PASS' if avail > req else 'FAIL'}", flush=True)
        return diag
    finally:
        M.clear_fixed_limb_darkening()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                     help="run Pi Mensae + V1828 Aql too (default: TOI-700 gate only)")
    args = ap.parse_args()

    with open(os.path.join(RESULTS, "bls_ephemeris.json"), encoding="utf-8") as fh:
        eph = json.load(fh)

    results = {}

    tgt = next(t for t in TARGETS if t["name"] == "TOI-700")
    e = eph["TOI-700"]
    sp = STELLAR_PARAMS["TOI-700"]
    u1, u2 = get_ld(sp["teff"], sp["logg"], sp["mh"], "TOI-700")
    results["TOI-700"] = fit_target("TOI-700", tgt["terms"], e["period"], e["t0"],
                                     e["duration"] * 24.0, 2247.17, u1, u2)

    gate_diag = results["TOI-700"]
    gate_pass = gate_diag is not None and gate_diag["tau_max"] < 160.0

    if gate_diag is not None:
        print(f"\nGATE CHECK: TOI-700 tau={gate_diag['tau_max']:.1f} "
              f"({'below' if gate_pass else 'NOT below'} 160 threshold) -> "
              f"{'proceeding' if (args.all or gate_pass) else 'stopping, per instructions'}",
              flush=True)

    if not (args.all or gate_pass):
        print("\nStopping before Pi Mensae/V1828 Aql -- gate did not pass. "
              "Rerun with --all to force both anyway.")
        return

    tgt = next(t for t in TARGETS if t["name"] == "Pi Mensae")
    sp = STELLAR_PARAMS["Pi Mensae"]
    u1, u2 = get_ld(sp["teff"], sp["logg"], sp["mh"], "Pi Mensae")
    results["Pi Mensae (lit)"] = fit_target("Pi Mensae (literature ephemeris)", tgt["terms"],
                                             PI_MEN_LIT_PERIOD, PI_MEN_LIT_T0, PI_MEN_LIT_DUR_HR,
                                             300.0, u1, u2)

    tgt = next(t for t in TARGETS if t["name"] == "V1828 Aql")
    e = eph["V1828 Aql"]
    sp = STELLAR_PARAMS["V1828 Aql"]
    u1, u2 = get_ld(sp["teff"], sp["logg"], sp["mh"], "V1828 Aql")
    results["V1828 Aql"] = fit_target("V1828 Aql", tgt["terms"], e["period"], e["t0"],
                                       e["duration"] * 24.0, 25480.52, u1, u2)

    print(f"\n{'=' * 66}\n  SUMMARY (NSTEPS={NSTEPS}, fixed limb darkening)\n{'=' * 66}")
    for name, diag in results.items():
        if diag is None:
            print(f"  {name:20s} MCMC init failed")
            continue
        print(f"  {name:20s} tau={diag['tau_max']:7.1f}  acc={diag['acceptance']:.3f}  "
              f"converged={diag['converged']}")


if __name__ == "__main__":
    main()
