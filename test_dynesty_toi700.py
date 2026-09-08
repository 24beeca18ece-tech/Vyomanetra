#!/usr/bin/env python3
"""
Bounded test (not part of the checkpointed batch): does nested sampling
(dynesty) converge where emcee's affine-invariant stretch move could not?

Base config: the best-performing setup from the three convergence attempts
so far -- (T14, b) parametrization + limb darkening FIXED to TIC/ExoTETHyS
values (NDIM=6), tau=228.2 on TOI-700 at NSTEPS=8000 (still far from
converged: needed 50*tau=11409, had 4800).

log_prior's box bounds are converted to a prior_transform (unit cube -> the
same bounds log_prior enforces). The one non-box constraint in log_prior --
the derived a_over_rs must be finite and in [1.2, 500] -- is NOT expressible
as a simple box on (T14, b) since it depends on both plus rp and period; it
is already enforced downstream in log_likelihood UNCHANGED (transit_model
returns nan for an invalid a_over_rs, log_likelihood catches non-finite
model and returns -inf), so no likelihood changes were needed either.

Time-boxed: a manual generator loop (not the blocking run_nested()) checks
wall-clock time each iteration so a runaway run can be stopped cleanly and
reported rather than hanging indefinitely.

Usage: python test_dynesty_toi700.py
"""
import os
import sys
import json
import time

import numpy as np
import dynesty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402
import pylightcurve as plc  # noqa: E402

MAX_POINTS = 4000
NLIVE = 250
DLOGZ = 0.05
MAX_WALL_SECONDS = 15 * 60  # hard time-box per the task instructions

STELLAR = dict(teff=3494.0, logg=4.80892, mh=0.0)  # TOI-700, TIC 150428135


def build_prior_transform(p0, t0_0, dur_0):
    """Box bounds mirror log_prior's for the fixed-LD, 6-param case exactly.
    b's upper bound (1+rp) is conditioned on rp, which is transformed first
    -- a standard, still-mechanical hierarchical prior_transform."""
    rp_lo, rp_hi = 0.001, 0.9
    t14_lo = max(1e-4, 0.4 * dur_0)
    t14_hi = min(0.5 * p0, 2.5 * dur_0)
    p_lo, p_hi = p0 * 0.98, p0 * 1.02
    t0_lo, t0_hi = t0_0 - dur_0, t0_0 + dur_0
    lj_lo, lj_hi = -14.0, -3.0

    def prior_transform(u):
        period = p_lo + u[0] * (p_hi - p_lo)
        t0 = t0_lo + u[1] * (t0_hi - t0_lo)
        rp = rp_lo + u[2] * (rp_hi - rp_lo)
        t14 = t14_lo + u[3] * (t14_hi - t14_lo)
        b = u[4] * (1.0 + rp)
        ln_jit = lj_lo + u[5] * (lj_hi - lj_lo)
        return np.array([period, t0, rp, t14, b, ln_jit])

    return prior_transform


def main():
    with open(os.path.join(RESULTS, "bls_ephemeris.json"), encoding="utf-8") as fh:
        e = json.load(fh)["TOI-700"]
    period, t0, dur_d = e["period"], e["t0"], e["duration"]
    dur_hr = dur_d * 24.0

    u1, u2 = None, None
    for model in ("phoenix", "atlas"):
        try:
            res = plc.exotethys(STELLAR["logg"], STELLAR["teff"], STELLAR["mh"],
                                 "TESS", method="quad", stellar_model=model)
            u1, u2 = float(res[0]), float(res[1])
            print(f"LD via {model}: u1={u1:.4f} u2={u2:.4f}", flush=True)
            break
        except plc.PyLCError as exc:
            print(f"{model} LD lookup failed: {exc}", flush=True)
    M.set_fixed_limb_darkening(u1, u2)

    tgt = next(t for t in TARGETS if t["name"] == "TOI-700")
    lc = download_lc(tgt["name"], tgt["terms"])
    lc_c = detrend(lc, "TOI-700_dynesty_test")
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

    # LSQ, for comparison -- same fixed-LD config
    th_ls, lsq_ok = M.fit_least_squares(tt, ff, ee, period, t0, dur_d, 2247.17)
    d_ls, du_ls, inc_ls, a_rs_ls = M.derived_quantities(th_ls)
    print(f"LSQ: depth={d_ls:.1f} ppm  T14={du_ls:.3f} h  b={th_ls[4]:.3f}  "
          f"a/Rs={a_rs_ls:.2f}", flush=True)

    prior_transform = build_prior_transform(period, t0, dur_d)

    def loglike(theta):
        return M.log_likelihood(theta, tt, ff, ee)  # unchanged, already validated

    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=6,
                                     nlive=NLIVE, bound="multi", sample="auto")

    print(f"\nRunning dynesty NestedSampler: nlive={NLIVE}, dlogz={DLOGZ}, "
          f"time-box={MAX_WALL_SECONDS}s", flush=True)
    t_start = time.time()
    it = 0
    timed_out = False
    for it, _ in enumerate(sampler.sample(dlogz=DLOGZ)):
        if it % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  iter {it}  elapsed={elapsed:.0f}s  "
                  f"logz={sampler.results.logz[-1] if len(sampler.results.logz) else float('nan'):.2f}",
                  flush=True)
        if time.time() - t_start > MAX_WALL_SECONDS:
            timed_out = True
            print(f"  *** TIME-BOX EXCEEDED at iter {it} ({time.time()-t_start:.0f}s) -- stopping", flush=True)
            break

    elapsed = time.time() - t_start
    if not timed_out:
        sampler.add_final_live()
    results = sampler.results
    print(f"\ndynesty finished: {it+1} iterations, {elapsed:.1f}s, "
          f"timed_out={timed_out}, ncall={results.ncall.sum() if hasattr(results,'ncall') else 'n/a'}",
          flush=True)

    # weighted posterior summary
    from dynesty.utils import resample_equal
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / np.sum(weights)
    samples_eq = resample_equal(results.samples, weights)
    names = ["period", "t0", "rp_over_rs", "T14_days", "impact_b", "ln_jitter"]
    print(f"\nposterior (equal-weighted, n={len(samples_eq)}):")
    for i, name in enumerate(names):
        lo, med, hi = np.percentile(samples_eq[:, i], [16, 50, 84])
        print(f"  {name:12s} {med:12.6f}  -{med-lo:.6f} +{hi-med:.6f}")

    # derived depth posterior
    idx = np.random.default_rng(0).choice(len(samples_eq), size=min(800, len(samples_eq)), replace=False)
    deps = []
    for k in idx:
        d, du, inc, a_rs = M.derived_quantities(samples_eq[k])
        if np.isfinite(d):
            deps.append(d)
    deps = np.asarray(deps)
    lo, med, hi = np.percentile(deps, [16, 50, 84])
    print(f"\nDEPTH posterior: {med:.1f} (-{med-lo:.1f}/+{hi-med:.1f}) ppm  (n={len(deps)})")
    print(f"vs LSQ: {d_ls:.1f} ppm")
    print(f"vs trapezoid (results/summary_table.csv): 2247.17 +/- 175.02 ppm")

    print(f"\nlogz = {results.logz[-1]:.3f} +/- {results.logzerr[-1]:.3f}")
    M.clear_fixed_limb_darkening()


if __name__ == "__main__":
    main()
