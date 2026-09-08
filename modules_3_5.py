#!/usr/bin/env python3
"""
VYOMANETRA Modules 3 & 5 -- physically-grounded transit characterization.

Replaces the prototype's trapezoid fit (Module 3) and depth/noise SNR ratio
(Module 5) with:

  Module 3 -- Mandel & Agol (2002) limb-darkened transit model, fit by
              least-squares for a starting point and then sampled with emcee
              for full joint posteriors on (P, t0, depth, duration, impact
              parameter, limb-darkening coefficients).

  Module 5 -- Bootstrap false-alarm probability from a resampled
              out-of-transit null distribution, plus split/jackknife conformal
              prediction intervals giving distribution-free calibrated
              confidence on the depth.

prototype.py is NOT modified; this is a library imported by
run_modules_3_5.py, mirroring how the ExoVeil baseline scripts were layered
alongside the original pipeline so the published baseline stays reproducible.

--------------------------------------------------------------------------
TRANSIT MODEL CHOICE -- why pylightcurve and not batman / pylightcurve-torch
--------------------------------------------------------------------------
The task suggested batman-package or PyLightcurve-torch. Both were attempted:

  * batman-package builds a C extension and requires MSVC 14+, which is not
    installed on this machine (no prebuilt cp314 Windows wheel exists). It
    could not be installed without pulling in Visual Studio Build Tools.

  * pylightcurve-torch 1.1.0 is incompatible with the installed torch 2.13:
    it calls `Tensor.new_zeros(n, 1)` with varargs at 18 call sites, and
    modern torch requires a size tuple. Patching site-packages would make the
    environment non-reproducible.

  * pylightcurve 4.0.4 (pure NumPy) installs cleanly. It is the reference
    implementation from which pylightcurve-torch is derived -- the same
    Mandel-Agol integrals -- and is well established (Tsiaras et al.).

Since emcee is a gradient-free affine-invariant ensemble sampler, autodiff
buys nothing for THIS task; the differentiability argument only matters if
Module 2's planned Mamba backbone later needs end-to-end gradients through
the transit model. Migration path if that happens: swap `transit_model()`
below for a pylightcurve-torch call (pin torch<2.4 or upstream a fix) --
it is the single point of contact with the model library.

Validated on import-time smoke test: for rp/Rs = 0.1, quadratic LD (0.4,
0.25), the model returns 11788 ppm vs a geometric rp^2 of 10000 ppm, i.e.
the ~1.18x limb-darkening enhancement expected for a central transit.

--------------------------------------------------------------------------
LIMB DARKENING PARAMETERIZATION
--------------------------------------------------------------------------
Quadratic LD coefficients (u1, u2) are sampled through the Kipping (2013)
(q1, q2) reparameterization:

    u1 = 2*sqrt(q1)*q2 ,  u2 = sqrt(q1)*(1 - 2*q2) ,  q1, q2 in [0, 1]

which maps the unit square exactly onto the physically allowed triangle
(u1+u2 < 1, u1 > 0, u1+2*u2 > 0), so the sampler cannot wander into
unphysical limb darkening and no rejection is wasted at the boundary.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

import pylightcurve as plc  # noqa: E402
import emcee  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import genpareto  # noqa: E402

# Free parameters, in sampling order.
#
# NOTE we sample T14 (total transit duration, days) and the impact parameter
# b, NOT a_over_rs and b. a_over_rs and b are only weakly, degenerately
# identified from single-band photometry alone at fixed duration -- many
# (a_over_rs, b) pairs reproduce the same light curve (a near-grazing, large
# a/Rs transit can look almost identical to a central, smaller a/Rs one once
# duration and depth are matched). This was confirmed empirically: tightening
# the walker-init ball 3x AND raising NSTEPS from 8000 to 18000 made tau
# WORSE on TOI-700 (316 -> 455) instead of better -- the signature of a
# geometric identifiability problem, not an under-sampling problem.
#
# T14, in contrast, is close to a direct observable (ingress-to-egress time),
# so sampling it directly removes the degenerate direction. a_over_rs is
# recovered algebraically for the transit_model() call via the standard
# circular-orbit relation (valid for i near 90 deg, i.e. a/Rs >> 1 -- true
# for every target here):
#     a/Rs = sqrt((1+rp)^2 - b^2) / sin(pi * T14 / P)
# (Seager & Mallen-Ornelas 2003; Winn 2010 Eq. 14 in the sin(i)~1 limit.)
#
# b itself is still sampled directly rather than inclination, for the same
# reason as before: at large a/Rs, inclination is pathologically conditioned
# (89 deg can imply b > 1, i.e. no transit, killing the gradient), while b's
# prior volume (0 <= b < 1+rp) is trivial to enforce for any a/Rs.
PARAM_NAMES_FULL = ["period", "t0", "rp_over_rs", "T14_days", "impact_b", "q1", "q2", "ln_jitter"]
PARAM_NAMES_FIXED_LD = ["period", "t0", "rp_over_rs", "T14_days", "impact_b", "ln_jitter"]
PARAM_NAMES = PARAM_NAMES_FULL
NDIM = len(PARAM_NAMES)

# When set (via set_fixed_limb_darkening), (u1, u2) are held fixed at tabulated
# stellar-atmosphere values instead of sampled as (q1, q2) -- testing whether
# TESS's single bandpass being unable to constrain limb darkening is the
# source of the persistent tau~260-320 seen under BOTH the (a_over_rs,b) and
# (T14,b) parametrizations. This drops NDIM from 8 to 6. Call
# clear_fixed_limb_darkening() to go back to sampling q1,q2.
_LD_FIXED = None


def set_fixed_limb_darkening(u1, u2):
    global _LD_FIXED, PARAM_NAMES, NDIM
    _LD_FIXED = (float(u1), float(u2))
    PARAM_NAMES = PARAM_NAMES_FIXED_LD
    NDIM = len(PARAM_NAMES)


def clear_fixed_limb_darkening():
    global _LD_FIXED, PARAM_NAMES, NDIM
    _LD_FIXED = None
    PARAM_NAMES = PARAM_NAMES_FULL
    NDIM = len(PARAM_NAMES)


def duration_b_to_a_rs(t14_d, b, rp, period_d):
    """(T14, b, rp, period) -> a/Rs, via the circular-orbit, i~90 deg relation
    a/Rs = sqrt((1+rp)^2 - b^2) / sin(pi*T14/P). Returns nan for an
    unphysical combination (grazing beyond 1+rp, or T14 >= P/2).
    """
    x = np.sin(np.pi * t14_d / period_d)
    if x <= 1e-8:
        return np.nan
    num = (1.0 + rp) ** 2 - b ** 2
    if num <= 0:
        return np.nan
    return float(np.sqrt(num) / x)


def b_to_inc_deg(b, a_rs):
    """Impact parameter -> inclination in degrees."""
    return float(np.rad2deg(np.arccos(np.clip(b / max(a_rs, 1e-9), 0.0, 1.0))))


# ---------------------------------------------------------------- model ----
def q_to_u(q1, q2):
    """Kipping (2013) q1,q2 -> quadratic limb-darkening u1,u2."""
    sq = np.sqrt(np.clip(q1, 0.0, 1.0))
    u1 = 2.0 * sq * q2
    u2 = sq * (1.0 - 2.0 * q2)
    return u1, u2


def _unpack_theta(theta):
    """Split theta into (period, t0, rp, t14, b, u1, u2), reading (u1,u2) from
    _LD_FIXED when limb darkening is fixed (6-dim theta) or from the sampled
    (q1,q2) otherwise (8-dim theta)."""
    if _LD_FIXED is not None:
        period, t0, rp, t14, b, _ = theta
        u1, u2 = _LD_FIXED
    else:
        period, t0, rp, t14, b, q1, q2, _ = theta
        u1, u2 = q_to_u(q1, q2)
    return period, t0, rp, t14, b, u1, u2


def transit_model(theta, t):
    """Mandel-Agol quadratic limb-darkened flux at times `t` for params theta.

    Single point of contact with the transit-model library (see module
    docstring for the swap-in path to a differentiable backend).
    """
    period, t0, rp, t14, b, u1, u2 = _unpack_theta(theta)
    a_rs = duration_b_to_a_rs(t14, b, rp, period)
    if not np.isfinite(a_rs) or a_rs <= 0:
        return np.full(len(t), np.nan)
    inc = b_to_inc_deg(b, a_rs)
    f = plc.transit([u1, u2], rp, period, a_rs, 0.0, inc, 0.0, t0, t, method="quad")
    return np.asarray(f, dtype=float).ravel()


def derived_quantities(theta):
    """Depth (ppm), T14 (hours, from plc's own exact formula -- a self-
    consistency check against the sampled T14), inclination (deg) and
    a/Rs (dimensionless) for a parameter vector."""
    period, t0, rp, t14, b, u1, u2 = _unpack_theta(theta)
    a_rs = duration_b_to_a_rs(t14, b, rp, period)
    if not np.isfinite(a_rs) or a_rs <= 0:
        return np.nan, np.nan, np.nan, np.nan
    inc = b_to_inc_deg(b, a_rs)
    try:
        depth = float(plc.transit_depth([u1, u2], rp, period, a_rs, 0.0, inc, 0.0, method="quad"))
    except Exception:
        depth = np.nan
    try:
        dur = float(plc.transit_duration(rp, period, a_rs, 0.0, inc, 0.0)) * 24.0
    except Exception:
        dur = np.nan
    return depth * 1e6, dur, inc, a_rs


# ------------------------------------------------------- priors / likelihood ----
def log_prior(theta, p0, t0_0, dur_0):
    """Weakly-informative priors centred on the BLS/archive ephemeris.

    Period and epoch are allowed to wander by +/-2% and +/-1 duration
    respectively -- enough to correct a slightly-off input ephemeris without
    letting the sampler jump to a different alias. T14 gets a TIGHT prior
    centred on the input duration estimate dur_0 (the trapezoid/BLS duration)
    -- unlike a_over_rs, T14 is close to directly observable, so a broad,
    uninformative range isn't needed and would just let the sampler wander
    into the very degeneracy this reparametrization is meant to remove.
    """
    if _LD_FIXED is not None:
        period, t0, rp, t14, b, ln_jit = theta
    else:
        period, t0, rp, t14, b, q1, q2, ln_jit = theta

    if not (p0 * 0.98 < period < p0 * 1.02):
        return -np.inf
    if not (t0_0 - dur_0 < t0 < t0_0 + dur_0):
        return -np.inf
    if not (0.001 < rp < 0.9):            # up to EB-scale radius ratios
        return -np.inf
    t14_lo = max(1e-4, 0.4 * dur_0)
    t14_hi = min(0.5 * p0, 2.5 * dur_0)
    if not (t14_lo < t14 < t14_hi):       # tight, physically-centred T14 prior
        return -np.inf
    if not (0.0 <= b < 1.0 + rp):         # geometry: an actual transit occurs
        return -np.inf
    a_rs = duration_b_to_a_rs(t14, b, rp, period)
    if not (np.isfinite(a_rs) and 1.2 <= a_rs <= 500.0):  # planet outside the star
        return -np.inf
    if _LD_FIXED is None:
        if not (0.0 < q1 < 1.0 and 0.0 < q2 < 1.0):
            return -np.inf
    if not (-14.0 < ln_jit < -3.0):
        return -np.inf
    return 0.0


def log_likelihood(theta, t, f, ferr):
    model = transit_model(theta, t)
    if not np.all(np.isfinite(model)):
        return -np.inf
    jitter = np.exp(theta[-1])
    s2 = ferr ** 2 + jitter ** 2
    return -0.5 * np.sum((f - model) ** 2 / s2 + np.log(2.0 * np.pi * s2))


def log_posterior(theta, t, f, ferr, p0, t0_0, dur_0):
    lp = log_prior(theta, p0, t0_0, dur_0)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t, f, ferr)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# --------------------------------------------------------------- windowing ----
def select_transit_window(t, f, ferr, period, t0, duration_d, n_dur=3.0, max_points=6000):
    """Keep cadences within n_dur transit durations of a predicted mid-transit.

    Fitting the full multi-sector series is wasteful -- almost every cadence
    is out of transit and constrains only the baseline. Restricting to transit
    neighbourhoods keeps the likelihood affordable while retaining local
    out-of-transit baseline on both sides of every event.
    """
    phase = (t - t0 + 0.5 * period) % period - 0.5 * period
    half = n_dur * duration_d
    m = np.abs(phase) < half
    if m.sum() < 20:  # ephemeris badly off -- fall back to everything
        m = np.ones_like(t, dtype=bool)

    tt, ff, ee = t[m], f[m], ferr[m]
    if len(tt) > max_points:  # decimate evenly, preserving phase coverage
        idx = np.linspace(0, len(tt) - 1, max_points).astype(int)
        tt, ff, ee = tt[idx], ff[idx], ee[idx]
    return tt, ff, ee, m


# ------------------------------------------------------------ Module 3 fit ----
def fit_least_squares(t, f, ferr, p0, t0_0, dur_d, depth_guess_ppm,
                       start_theta=None, max_nfev=3000, xtol=1e-12, b_start=0.3):
    """Single-start least-squares fit. `start_theta`, if given, overrides the
    default starting guess (used by bootstrap_lsq(), whose resamples start
    extremely close to the already-converged best fit, so they can use a far
    smaller `max_nfev`/looser `xtol` than the original fit needs). `b_start`
    sets the seed impact parameter (used by fit_least_squares_multistart() to
    probe multiple local minima -- see that function's docstring for why a
    single seed isn't reliable)."""
    rp_g = float(np.clip(np.sqrt(max(depth_guess_ppm, 1.0) / 1e6), 0.005, 0.5))
    # T14 starts at the input duration estimate itself (trapezoid/BLS dur_d) --
    # it's the direct observable, not a derived one like a/Rs was.
    t14_g = float(np.clip(dur_d, 1e-4, 0.49 * p0))
    t14_lo = max(1e-4, 0.4 * dur_d)
    t14_hi = min(0.49 * p0, 2.5 * dur_d)
    if _LD_FIXED is not None:
        start = np.array([p0, t0_0, rp_g, t14_g, b_start, -8.0])
        lo = np.array([p0 * 0.99, t0_0 - dur_d, 0.001, t14_lo, 0.0, -13.0])
        hi = np.array([p0 * 1.01, t0_0 + dur_d, 0.9, t14_hi, 1.0 + rp_g, -4.0])
    else:
        start = np.array([p0, t0_0, rp_g, t14_g, b_start, 0.3, 0.3, -8.0])
        lo = np.array([p0 * 0.99, t0_0 - dur_d, 0.001, t14_lo, 0.0, 0.01, 0.01, -13.0])
        hi = np.array([p0 * 1.01, t0_0 + dur_d, 0.9, t14_hi, 1.0 + rp_g, 0.99, 0.99, -4.0])
    if start_theta is not None:
        start = np.clip(np.asarray(start_theta, dtype=float), lo + 1e-9, hi - 1e-9)
    else:
        start = np.clip(start, lo + 1e-9, hi - 1e-9)

    def resid(th):
        m = transit_model(th, t)
        if not np.all(np.isfinite(m)):
            return np.full_like(f, 1e6)
        return (f - m) / ferr

    try:
        sol = least_squares(resid, start, bounds=(lo, hi), max_nfev=max_nfev, xtol=xtol)
        return sol.x, True
    except Exception:
        return start, False


def fit_least_squares_multistart(t, f, ferr, p0, t0_0, dur_d, depth_guess_ppm,
                                  b_starts=(0.0, 0.2, 0.4, 0.6, 0.8), **kwargs):
    """Run fit_least_squares from several seed impact parameters and keep
    whichever converges to the lowest chi-square.

    Confirmed failure mode of the single b=0.3-seed fit: across the 25-target
    benchmark sample, fitted impact parameter correlated with depth deficit
    at Pearson r=-0.63 (13/25 targets landed at b>0.6, median depth ~89%
    BELOW the archive value; the rest, at b<=0.6, still averaged a smaller
    but non-trivial deficit). The chi-square surface in (rp, impact_b) has
    multiple local minima of comparable depth (grazing-large-planet vs.
    central-small-planet degeneracy -- the same one that caused emcee/dynesty
    non-convergence, see module3_convergence_diagnosis.md); a single fixed
    starting point has no way to know which basin it lands in. Sampling
    several b starts and keeping the lowest chi-square is a cheap (~130ms per
    warm-started single-start fit) way to make the point estimate robust to
    this without needing a full posterior.

    Returns (theta_best, ok, chi2_best, b_start_used).
    """
    best_theta, best_ok, best_chi2, best_b0 = None, False, np.inf, None
    for b0 in b_starts:
        th, ok = fit_least_squares(t, f, ferr, p0, t0_0, dur_d, depth_guess_ppm,
                                    b_start=b0, **kwargs)
        m = transit_model(th, t)
        if not np.all(np.isfinite(m)):
            continue
        chi2 = float(np.sum(((f - m) / ferr) ** 2))
        if chi2 < best_chi2:
            best_theta, best_ok, best_chi2, best_b0 = th, ok, chi2, b0
    if best_theta is None:
        th, ok = fit_least_squares(t, f, ferr, p0, t0_0, dur_d, depth_guess_ppm, **kwargs)
        return th, ok, np.inf, None
    return best_theta, best_ok, best_chi2, best_b0


# ---------------------------- bounded test: a_over_rs fixed from stellar density ----
def stellar_density_to_a_rs(rho_star_solar, period_d):
    """a/Rs from stellar density (solar units) and orbital period (days) via
    Kepler's third law, for a low-mass companion (Mp << Mstar):
        a/Rs = (G * rho_star * P^2 / (3*pi))^(1/3)
    (Seager & Mallen-Ornelas 2003; Winn 2010 Eq. 30). rho_star_solar is
    density relative to the Sun's (1.408 g/cm^3), i.e. simply
    (M/Msun) / (R/Rsun)^3 -- no unit conversion needed for that ratio itself.
    """
    G_CGS = 6.674e-8
    RHO_SUN_CGS = 1.408
    p_sec = period_d * 86400.0
    rho_cgs = rho_star_solar * RHO_SUN_CGS
    return float((G_CGS * rho_cgs * p_sec ** 2 / (3.0 * np.pi)) ** (1.0 / 3.0))


def transit_model_fixed_a_rs(theta, t, a_rs):
    """Like transit_model(), but a_over_rs is FIXED (not fit): theta =
    [period, t0, rp, impact_b, q1, q2, ln_jitter] (7-dim, free LD). Bounded
    test of whether externally constraining the geometry scale (e.g. from a
    stellar-density prior) breaks the rp/b/duration/a_over_rs degeneracy
    that neither reparametrization, limb-darkening-fixing, nor multi-start
    LSQ resolved (see module3_convergence_diagnosis.md)."""
    period, t0, rp, b, q1, q2, _ = theta
    u1, u2 = q_to_u(q1, q2)
    inc = b_to_inc_deg(b, a_rs)
    f = plc.transit([u1, u2], rp, period, a_rs, 0.0, inc, 0.0, t0, t, method="quad")
    return np.asarray(f, dtype=float).ravel()


def derived_quantities_fixed_a_rs(theta, a_rs):
    """derived_quantities() counterpart for transit_model_fixed_a_rs()."""
    period, t0, rp, b, q1, q2, _ = theta
    u1, u2 = q_to_u(q1, q2)
    inc = b_to_inc_deg(b, a_rs)
    try:
        depth = float(plc.transit_depth([u1, u2], rp, period, a_rs, 0.0, inc, 0.0, method="quad"))
    except Exception:
        depth = np.nan
    try:
        dur = float(plc.transit_duration(rp, period, a_rs, 0.0, inc, 0.0)) * 24.0
    except Exception:
        dur = np.nan
    return depth * 1e6, dur, inc


def fit_least_squares_fixed_a_rs(t, f, ferr, p0, t0_0, dur_d, depth_guess_ppm, a_rs,
                                  b_starts=(0.0, 0.2, 0.4, 0.6, 0.8),
                                  max_nfev=3000, xtol=1e-12):
    """Multi-start-over-b LSQ fit with a_over_rs fixed externally. b is still
    probed from several starts (it's the one remaining geometry parameter
    that could have distinct local optima); period/t0/rp/LD bounds mirror
    fit_least_squares()'s. Returns (theta_best, ok, chi2_best, b_start_used).
    """
    rp_g = float(np.clip(np.sqrt(max(depth_guess_ppm, 1.0) / 1e6), 0.005, 0.5))
    lo = np.array([p0 * 0.99, t0_0 - dur_d, 0.001, 0.0, 0.01, 0.01, -13.0])
    hi = np.array([p0 * 1.01, t0_0 + dur_d, 0.9, 1.0 + rp_g, 0.99, 0.99, -4.0])

    def resid(th):
        m = transit_model_fixed_a_rs(th, t, a_rs)
        if not np.all(np.isfinite(m)):
            return np.full_like(f, 1e6)
        return (f - m) / ferr

    best_theta, best_ok, best_chi2, best_b0 = None, False, np.inf, None
    for b0 in b_starts:
        start = np.clip(np.array([p0, t0_0, rp_g, b0, 0.3, 0.3, -8.0]), lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(resid, start, bounds=(lo, hi), max_nfev=max_nfev, xtol=xtol)
            th, ok = sol.x, True
        except Exception:
            th, ok = start, False
        m = transit_model_fixed_a_rs(th, t, a_rs)
        if not np.all(np.isfinite(m)):
            continue
        chi2 = float(np.sum(((f - m) / ferr) ** 2))
        if chi2 < best_chi2:
            best_theta, best_ok, best_chi2, best_b0 = th, ok, chi2, b0
    if best_theta is None:
        return None, False, np.inf, None
    return best_theta, best_ok, best_chi2, best_b0


def run_mcmc(t, f, ferr, theta_start, p0, t0_0, dur_d,
             nwalkers=32, nsteps=3000, burn_frac=0.4, seed=42, progress=False,
             init_scale=1.0):
    """Sample the posterior with emcee. Returns (chain, diagnostics).

    Walkers are ALREADY initialised as a tight Gaussian ball around
    `theta_start` (the LSQ optimum), not as wide draws from the prior -- the
    base scatter below (`init_scale=1.0`) is a few times 1e-3--1e-5 in each
    parameter's natural units. `init_scale` < 1 tightens it further (e.g.
    0.3 = 3x tighter) to shrink burn-in wander without touching the physics.

    Historical note: with the OLD (a_over_rs, b) parametrization, tightening
    this ball 3x and raising nsteps from 8000 to 18000 made tau WORSE on
    TOI-700 (316 -> 455). Switching to (T14, b) -- removing the suspected
    a_over_rs/b degeneracy -- did NOT fix it either: tau stayed at 308.3 on
    TOI-700 at the same NSTEPS=8000, statistically indistinguishable from the
    original 316. That ruled out a_over_rs/b as the bottleneck and motivated
    testing limb darkening (q1, q2) instead via set_fixed_limb_darkening() --
    the two dimensions single-band TESS photometry is least able to pin down.
    """
    rng = np.random.default_rng(seed)
    # T14 (index 3) scatter is relative to its own value (duration varies by
    # orders of magnitude across targets), not a flat absolute like the old
    # a_over_rs scatter of 0.5 was.
    if _LD_FIXED is not None:
        base_scat = init_scale * np.array(
            [theta_start[0] * 1e-5, 1e-3, 1e-3, max(1e-4, 0.03 * theta_start[3]),
             0.05, 0.2])
    else:
        base_scat = init_scale * np.array(
            [theta_start[0] * 1e-5, 1e-3, 1e-3, max(1e-4, 0.03 * theta_start[3]),
             0.05, 0.05, 0.05, 0.2])
    pos = []
    tries = 0
    while len(pos) < nwalkers and tries < nwalkers * 400:
        tries += 1
        scat = theta_start + rng.normal(0, 1, NDIM) * base_scat
        scat[4] = float(np.clip(scat[4], 0.0, 0.98 * (1.0 + scat[2])))  # keep b transiting
        if _LD_FIXED is None:
            scat[5] = np.clip(scat[5], 0.02, 0.98)
            scat[6] = np.clip(scat[6], 0.02, 0.98)
        if np.isfinite(log_prior(scat, p0, t0_0, dur_d)):
            pos.append(scat)
    if len(pos) < nwalkers:
        return None, {"ok": False, "reason": "could not initialise walkers in prior volume"}
    pos = np.array(pos)

    sampler = emcee.EnsembleSampler(
        nwalkers, NDIM, log_posterior, args=(t, f, ferr, p0, t0_0, dur_d))
    sampler.run_mcmc(pos, nsteps, progress=progress)

    burn = int(burn_frac * nsteps)
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
        tau_max = float(np.nanmax(tau))
    except Exception:
        tau_max = np.nan
    thin = max(1, int(tau_max / 2)) if np.isfinite(tau_max) else 1
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)

    acc = float(np.mean(sampler.acceptance_fraction))
    n_eff = len(chain)
    converged = bool(np.isfinite(tau_max) and (nsteps - burn) > 50 * tau_max
                     and 0.1 < acc < 0.8 and n_eff > 200)
    diag = {"ok": True, "acceptance": acc, "tau_max": tau_max,
            "n_samples": n_eff, "converged": converged,
            "nwalkers": nwalkers, "nsteps": nsteps}
    return chain, diag


def summarize_posterior(chain, n_derived=800, seed=0):
    """Median and 16th/84th percentiles for sampled and derived parameters."""
    out = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = np.percentile(chain[:, i], [16, 50, 84])
        out[f"{name}_med"] = med
        out[f"{name}_lo"] = med - lo
        out[f"{name}_hi"] = hi - med

    if _LD_FIXED is not None:
        # fixed, not sampled -- report the constant with zero spread for
        # output-schema consistency with the free-LD case.
        for name, val in [("u1", _LD_FIXED[0]), ("u2", _LD_FIXED[1])]:
            out[f"{name}_med"], out[f"{name}_lo"], out[f"{name}_hi"] = val, 0.0, 0.0
    else:
        u1s, u2s = q_to_u(chain[:, 5], chain[:, 6])
        for name, arr in [("u1", u1s), ("u2", u2s)]:
            lo, med, hi = np.percentile(arr, [16, 50, 84])
            out[f"{name}_med"], out[f"{name}_lo"], out[f"{name}_hi"] = med, med - lo, hi - med

    # derived depth / duration / inc / a_over_rs on a random subset (transit_depth
    # is not cheap). a_over_rs is no longer a sampled parameter -- see PARAM_NAMES
    # note -- so it is recomputed here from each sample's (T14, b, rp, period)
    # for reporting/comparison purposes only.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chain), size=min(n_derived, len(chain)), replace=False)
    dep, dur, incs, a_rss = [], [], [], []
    for k in idx:
        d, du, inc, a_rs = derived_quantities(chain[k])
        dep.append(d); dur.append(du); incs.append(inc); a_rss.append(a_rs)
    for name, arr in [("depth_ppm", dep), ("duration_hr", dur), ("inc_deg", incs),
                       ("a_over_rs", a_rss)]:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            out[f"{name}_med"] = out[f"{name}_lo"] = out[f"{name}_hi"] = np.nan
            continue
        lo, med, hi = np.percentile(arr, [16, 50, 84])
        out[f"{name}_med"], out[f"{name}_lo"], out[f"{name}_hi"] = med, med - lo, hi - med
    return out


# ---------------------------------- Module 3 PRIMARY uncertainty: LSQ + bootstrap ----
def _summ68(arr):
    """Median and 68% (16th/84th percentile) interval, conformal_interval-style."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return np.nan, np.nan, np.nan
    lo, med, hi = np.percentile(arr, [16, 50, 84])
    return med, med - lo, hi - med


def bootstrap_lsq(t, f, ferr, theta_best, p0, t0_0, dur_d, depth_guess_ppm,
                   n_boot=300, seed=3):
    """Parametric (residual) bootstrap for LSQ parameter uncertainty.

    THIS is Module 3's PRIMARY uncertainty source, replacing MCMC/nested-
    sampling as the headline result -- see results/module3_convergence_
    diagnosis.md for why. Four independent attempts (free (a_over_rs,b),
    free (T14,b), (T14,b) with limb darkening fixed from tabulated stellar
    values, and dynesty nested sampling on that same fixed-LD config) all
    ran into the same underlying transit-geometry degeneracy (it resurfaces
    as different parameter pairs -- a_over_rs/b, then rp_over_rs/impact_b --
    under different samplers), never converging cleanly. LSQ point estimates,
    in contrast, were stable and physically sensible in every one of those
    attempts. This function quantifies LSQ's uncertainty directly via
    residual resampling instead of relying on posterior sampling to do it.

    NOT to be confused with Module 5's bootstrap_fap(): that resamples ONLY
    out-of-transit residuals to build a null (no-signal) distribution for a
    significance/detection test. This function resamples the FULL fit's
    residuals (in- and out-of-transit, from the best-fit model itself) and
    refits least-squares on each resample -- the classic residual bootstrap
    for regression PARAMETER uncertainty (Efron & Tibshirani 1993), unrelated
    to detection significance.

    Both iid and block (block length ~ 1 transit duration, same construction
    as Module 5's block FAP) resampling are run; block is the honest default
    for TESS's correlated residuals, iid is reported alongside for
    comparison. Returns med/lo/hi (68% interval) for depth_ppm, duration_hr,
    rp_over_rs and impact_b, under both schemes.
    """
    model = transit_model(theta_best, t)
    if not np.all(np.isfinite(model)):
        return {"ok": False, "note": "best-fit model non-finite"}
    resid = f - model

    n = len(t)
    dt = np.nanmedian(np.diff(np.sort(t)))
    blk = int(max(2, round(dur_d / dt))) if (np.isfinite(dt) and dt > 0) else 20
    blk = min(blk, max(2, n // 4))

    rng = np.random.default_rng(seed)

    def draw_iid():
        return rng.choice(resid, size=n, replace=True)

    def draw_block():
        out = np.empty(n)
        filled = 0
        while filled < n:
            s = rng.integers(0, n - blk) if n > blk else 0
            chunk = resid[s:s + blk]
            take = min(len(chunk), n - filled)
            out[filled:filled + take] = chunk[:take]
            filled += take
        return out

    collect = {k: {"depth": [], "dur": [], "rp": [], "b": []} for k in ("iid", "block")}
    n_ok = {"iid": 0, "block": 0}
    for _ in range(n_boot):
        for scheme, draw in (("iid", draw_iid), ("block", draw_block)):
            f_boot = model + draw()
            try:
                th_b, _ = fit_least_squares(t, f_boot, ferr, p0, t0_0, dur_d, depth_guess_ppm,
                                             start_theta=theta_best, max_nfev=300, xtol=1e-8)
                d, du, inc, a_rs = derived_quantities(th_b)
            except Exception:
                continue
            if not np.isfinite(d):
                continue
            c = collect[scheme]
            c["depth"].append(d); c["dur"].append(du); c["rp"].append(th_b[2]); c["b"].append(th_b[4])
            n_ok[scheme] += 1

    out = {"ok": n_ok["block"] > 20, "n_boot": n_boot,
           "n_ok_iid": n_ok["iid"], "n_ok_block": n_ok["block"],
           "block_len_cadences": blk}
    for scheme in ("iid", "block"):
        c = collect[scheme]
        for qty, key in [("depth", "depth_ppm"), ("dur", "duration_hr"),
                          ("rp", "rp_over_rs"), ("b", "impact_b")]:
            med, lo, hi = _summ68(c[qty])
            out[f"{key}_{scheme}_med"] = med
            out[f"{key}_{scheme}_lo"] = lo
            out[f"{key}_{scheme}_hi"] = hi
    return out


# ------------------------------------------------- Module 5: bootstrap FAP ----
def _fit_gpd_tail(null_vals, tail_frac=0.10):
    """Peaks-over-threshold GPD fit to the upper tail of a null statistic.

    Standard extreme-value approach (Coles 2001) for extrapolating a p-value
    below a bootstrap's empirical resolution floor of 1/(n_boot+1): fit a
    Generalized Pareto Distribution to the exceedances over a threshold u
    (the (1-tail_frac) empirical quantile of the null draws), then use the
    GPD's analytic survival function for values beyond the observed range
    instead of reporting a hard floor.

    Returns None if there are too few null draws or tail exceedances to fit.
    """
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    n = len(null_vals)
    if n < 100:
        return None
    u = float(np.quantile(null_vals, 1.0 - tail_frac))
    exceed = null_vals[null_vals > u] - u
    if len(exceed) < 20:
        return None
    try:
        xi, _, sigma = genpareto.fit(exceed, floc=0.0)
    except Exception:
        return None
    clamped = False
    if xi < 0.0:
        # A negative MLE shape implies a BOUNDED tail (finite upper endpoint
        # u - sigma/xi): P(null > x) is then mathematically exactly zero for
        # any x beyond that endpoint, not just small. With only n_exceed~O(100-
        # 200) points the shape estimate is noisy enough that this happens even
        # when the true tail (matched-filter SNR under resampled ~Gaussian
        # noise) is not actually bounded -- it would wrongly report a literal
        # impossibility for a real, strong detection. Conservatively fall back
        # to the unbounded xi=0 (exponential) tail, whose MLE scale is just the
        # mean exceedance -- standard practice for this failure mode in POT
        # fitting (Coles 2001, Sec. 4.2).
        xi = 0.0
        sigma = float(np.mean(exceed))
        clamped = True
    return {"u": u, "p_u": len(exceed) / n, "xi": float(xi), "sigma": float(sigma),
            "n_exceed": len(exceed), "xi_clamped": clamped}


def _gpd_tail_prob(x, gpd):
    """P(null > x) under the fitted GPD tail; only valid for x >= gpd['u']."""
    if x < gpd["u"]:
        return np.nan
    sf = genpareto.sf(x - gpd["u"], gpd["xi"], loc=0.0, scale=gpd["sigma"])
    return gpd["p_u"] * float(sf)


def _gpd_tail_log10prob(x, gpd):
    """log10 P(null > x) under the fitted GPD tail, computed via logsf so it
    stays finite (no underflow to a bare 0) for very extreme x -- e.g. a
    strong detection's SNR can be far enough into the tail that the linear
    probability is below the smallest representable double (~1e-308); the
    log10 value is still a real, reportable number in that regime.
    """
    if x < gpd["u"]:
        return np.nan
    logsf = genpareto.logsf(x - gpd["u"], gpd["xi"], loc=0.0, scale=gpd["sigma"])
    return float(np.log10(gpd["p_u"]) + logsf / np.log(10.0))


def _validate_gpd_tail(null_vals, gpd, n_check=8):
    """Max |empirical - GPD| survival probability over sub-thresholds spanning
    [u, max(null)] -- the region where the empirical fraction is still
    directly measurable, so a large discrepancy here means the GPD fit
    should not be trusted for extrapolating past max(null).
    """
    null_vals = np.asarray(null_vals, dtype=float)
    lo, hi = gpd["u"], float(np.nanmax(null_vals))
    if not np.isfinite(hi) or hi <= lo:
        return np.nan
    xs = np.linspace(lo, lo + 0.98 * (hi - lo), n_check)
    diffs = [abs(np.mean(null_vals > x) - _gpd_tail_prob(x, gpd)) for x in xs]
    diffs = [d for d in diffs if np.isfinite(d)]
    return float(np.max(diffs)) if diffs else np.nan


def fap_gpd_extrapolated(snr_obs, null_vals, n_boot, tail_frac=0.10):
    """FAP for `snr_obs`, extrapolated past the empirical bootstrap floor
    (1/(n_boot+1)) via the GPD tail fit above. Falls back to the plain
    empirical estimate (with a note) if the fit fails, or if snr_obs sits
    below the fitted threshold u (where the empirical count is already
    reliable and no extrapolation is needed).
    """
    fap_emp = (1 + np.sum(np.asarray(null_vals) >= snr_obs)) / (n_boot + 1)
    gpd = _fit_gpd_tail(null_vals, tail_frac=tail_frac)
    if gpd is None:
        return {"fap_gpd": np.nan, "fap_gpd_log10": np.nan, "gpd_u": np.nan, "gpd_xi": np.nan,
                "gpd_sigma": np.nan, "gpd_validation_maxdiff": np.nan,
                "note": "GPD fit failed or too few null draws/exceedances"}
    validation = _validate_gpd_tail(null_vals, gpd)
    clamp_note = " (xi clamped to 0: MLE shape was negative/bounded)" if gpd.get("xi_clamped") else ""
    if snr_obs < gpd["u"]:
        log10_emp = float(np.log10(fap_emp)) if fap_emp > 0 else np.nan
        return {"fap_gpd": float(fap_emp), "fap_gpd_log10": log10_emp,
                "gpd_u": gpd["u"], "gpd_xi": gpd["xi"],
                "gpd_sigma": gpd["sigma"], "gpd_validation_maxdiff": validation,
                "note": "snr_obs below GPD threshold u; empirical estimate already reliable, not extrapolated" + clamp_note}
    fap_gpd = _gpd_tail_prob(snr_obs, gpd)
    fap_gpd_log10 = _gpd_tail_log10prob(snr_obs, gpd)
    return {"fap_gpd": float(fap_gpd) if np.isfinite(fap_gpd) else np.nan,
            "fap_gpd_log10": fap_gpd_log10,
            "gpd_u": gpd["u"], "gpd_xi": gpd["xi"], "gpd_sigma": gpd["sigma"],
            "gpd_validation_maxdiff": validation, "note": clamp_note.strip()}


def matched_filter_snr(deficit, template, sigma):
    """Amplitude SNR of `template` in data, by matched filter.

    template = 1 - model, i.e. the transit shape: zero out of transit, POSITIVE
    in transit, peaking at the fractional transit depth.

    `deficit` must use the SAME sign convention -- it is the flux DEFICIT
    (baseline - flux), not (flux - baseline). Passing the latter flips the sign
    of the returned amplitude and makes every FAP come out at 1.0.

    A_hat = sum(m*d)/sum(m^2);  sigma_A = sigma/sqrt(sum(m^2));  SNR = A/sigma_A

    NOTE A_hat is a DIMENSIONLESS scaling of the template, not a depth: A_hat=1
    means "the observed dip matches the template amplitude exactly". Multiply
    by max(template) to convert to a fractional depth.
    """
    denom = np.sum(template ** 2)
    if denom <= 0 or sigma <= 0:
        return 0.0, 0.0
    amp = np.sum(template * deficit) / denom
    sig_amp = sigma / np.sqrt(denom)
    return amp, amp / sig_amp


def bootstrap_fap(t, f, theta_best, duration_d, n_boot=2000, block=True, seed=1):
    """False-alarm probability from a resampled out-of-transit null.

    The observed statistic is the matched-filter SNR of the best-fit transit
    template at the true ephemeris. The null distribution is built by
    resampling ONLY out-of-transit residuals into a synthetic, transit-free
    light curve on the same time stamps and recomputing the same statistic.

    Two resampling schemes are reported:
      * iid    -- classic bootstrap, assumes white noise. Optimistic (too low
                  a FAP) whenever residual red noise survives detrending.
      * block  -- moving-block bootstrap with block length ~ one transit
                  duration, which preserves correlation on the timescale that
                  actually matters for transit detection. This is the honest
                  number to quote for TESS photometry.

    FAP = (1 + #{null >= observed}) / (n_boot + 1)  -- the add-one keeps the
    estimate conservative and non-zero at finite n_boot (a reported FAP can
    never be smaller than 1/(n_boot+1)).
    """
    rng = np.random.default_rng(seed)
    model = transit_model(theta_best, t)
    template = 1.0 - model
    in_tr = template > (0.02 * np.nanmax(template) if np.nanmax(template) > 0 else np.inf)
    oot = ~in_tr
    if oot.sum() < 50 or in_tr.sum() < 5:
        return {"fap_iid": np.nan, "fap_block": np.nan, "snr_obs": np.nan,
                "n_oot": int(oot.sum()), "n_in": int(in_tr.sum()),
                "note": "insufficient in/out-of-transit cadences"}

    # deficit convention throughout: positive where flux is below baseline,
    # matching the sign of `template` (see matched_filter_snr docstring).
    base = np.nanmedian(f[oot])
    resid_oot = base - f[oot]
    sigma = float(np.nanstd(resid_oot))
    _, snr_obs = matched_filter_snr(base - f, template, sigma)

    n = len(t)
    # cadence-based block length ~ one transit duration
    dt = np.nanmedian(np.diff(np.sort(t)))
    blk = int(max(2, round(duration_d / dt))) if (np.isfinite(dt) and dt > 0) else 20
    blk = min(blk, max(2, len(resid_oot) // 4))

    def draw_iid():
        return rng.choice(resid_oot, size=n, replace=True)

    def draw_block():
        out = np.empty(n)
        filled = 0
        while filled < n:
            s = rng.integers(0, len(resid_oot) - blk) if len(resid_oot) > blk else 0
            chunk = resid_oot[s:s + blk]
            take = min(len(chunk), n - filled)
            out[filled:filled + take] = chunk[:take]
            filled += take
        return out

    snr_iid, snr_blk = [], []
    for _ in range(n_boot):
        snr_iid.append(matched_filter_snr(draw_iid(), template, sigma)[1])
        if block:
            snr_blk.append(matched_filter_snr(draw_block(), template, sigma)[1])

    snr_iid = np.asarray(snr_iid)
    fap_iid = (1 + np.sum(snr_iid >= snr_obs)) / (n_boot + 1)
    if block:
        snr_blk = np.asarray(snr_blk)
        fap_blk = (1 + np.sum(snr_blk >= snr_obs)) / (n_boot + 1)
    else:
        fap_blk = np.nan

    # GPD tail extrapolation past the empirical floor (see fap_gpd_extrapolated
    # docstring) -- reported alongside, never replacing, the empirical values.
    gpd_iid = fap_gpd_extrapolated(snr_obs, snr_iid, n_boot)
    gpd_blk = (fap_gpd_extrapolated(snr_obs, snr_blk, n_boot) if block
               else {"fap_gpd": np.nan, "fap_gpd_log10": np.nan, "gpd_validation_maxdiff": np.nan})

    return {"fap_iid": float(fap_iid), "fap_block": float(fap_blk),
            "fap_iid_gpd": gpd_iid["fap_gpd"], "fap_block_gpd": gpd_blk["fap_gpd"],
            "fap_iid_gpd_log10": gpd_iid["fap_gpd_log10"], "fap_block_gpd_log10": gpd_blk["fap_gpd_log10"],
            "gpd_iid_validation_maxdiff": gpd_iid["gpd_validation_maxdiff"],
            "gpd_block_validation_maxdiff": gpd_blk["gpd_validation_maxdiff"],
            "snr_obs": float(snr_obs), "n_oot": int(oot.sum()), "n_in": int(in_tr.sum()),
            "null_iid_p99": float(np.percentile(snr_iid, 99)),
            "null_block_p99": float(np.percentile(snr_blk, 99)) if block else np.nan,
            "block_len_cadences": int(blk), "n_boot": int(n_boot), "note": ""}


# --------------------------------------- Module 5: conformal prediction ----
def per_epoch_depths(t, f, theta_best, period, t0):
    """Matched-filter depth measured independently at each observed epoch."""
    model = transit_model(theta_best, t)
    template = 1.0 - model
    peak = np.nanmax(template)
    if not np.isfinite(peak) or peak <= 0:
        return np.array([]), np.array([])

    epoch = np.round((t - t0) / period).astype(int)
    in_tr = template > 0.02 * peak
    base = np.nanmedian(f[~in_tr]) if (~in_tr).sum() > 10 else np.nanmedian(f)

    depths, epochs = [], []
    for e in np.unique(epoch[in_tr]):
        m = (epoch == e)
        if (m & in_tr).sum() < 3:
            continue
        # deficit convention; amp is a dimensionless template scaling, so
        # multiply by the template peak to get an absolute fractional depth.
        amp, _ = matched_filter_snr(base - f[m], template[m], 1.0)
        if np.isfinite(amp):
            depths.append(amp * peak * 1e6)  # ppm
            epochs.append(e)
    return np.asarray(depths), np.asarray(epochs)


def conformal_interval(values, alpha=0.1):
    """Distribution-free jackknife-style conformal interval for the mean depth.

    Nonconformity score for epoch i is |d_i - median(d_{-i})| (leave-one-out,
    so each score is computed against an estimate that did not see that
    epoch). The interval is the point estimate +/- the finite-sample-corrected
    (1-alpha) quantile of those scores.

    Unlike the MCMC credible interval this makes no Gaussian or model-
    correctness assumption -- it is calibrated by the actual epoch-to-epoch
    scatter, so it widens honestly when individual transits disagree.
    Requires >=4 epochs to be meaningful.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 4:
        return {"conformal_point": float(np.median(v)) if n else np.nan,
                "conformal_half_width": np.nan, "conformal_lo": np.nan,
                "conformal_hi": np.nan, "n_epochs": n,
                "note": "need >=4 epochs for a calibrated interval"}

    scores = np.array([abs(v[i] - np.median(np.delete(v, i))) for i in range(n)])
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:  # too few points to certify this level; fall back to the max score
        half = float(np.max(scores))
        note = f"n={n} too small for exact {1-alpha:.0%} level; using max score (conservative)"
    else:
        half = float(np.sort(scores)[k - 1])
        note = ""
    point = float(np.median(v))
    return {"conformal_point": point, "conformal_half_width": half,
            "conformal_lo": point - half, "conformal_hi": point + half,
            "n_epochs": n, "alpha": alpha, "note": note}
