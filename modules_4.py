#!/usr/bin/env python3
"""
VYOMANETRA Module 4 -- Concept Bottleneck Model: concept computation (C1-C7).

Spec source: VYOMANETRA_Report.pdf, Table 2 (Sec 3.4) and Eq. 2 (Sec 4.1).
Seven interpretable astrophysical concepts feed the CBM classifier:

  C1  Transit depth score        -- moderate (planet) vs. can be very deep (FP)
  C2  Shape metric (U vs V)      -- high flat-bottom ratio (planet) vs. low (FP)
  C3  Ingress/egress symmetry    -- symmetric (planet) vs. slightly asymmetric (FP)
  C4  Odd-even depth difference  -- same depth every transit (planet) vs.
                                     alternating = binary (FP)
  C5  Secondary eclipse depth    -- shallow/absent (planet) vs. significant = binary (FP)
  C6  Centroid shift             -- none/on-axis (planet) vs. present = blend (FP)
  C7  Phase-fold coherence       -- consistent per-epoch depths, C7~1 (planet) vs.
                                     erratic/aliased period, C7->0 (FP)
                                     (Novel Contribution 4, report Eq. 2)

Reuse policy (per task instructions -- don't rebuild what already exists):
  * C1, C2 come directly from prototype.py's characterize() trapezoid fit
    (depth_ppm, flat_dur/total_dur) -- prototype.py is imported, not modified.
  * C7's per-epoch depths are measured via the SAME matched-filter pattern as
    modules_3_5.py's Module 5 per_epoch_depths()/matched_filter_snr(), but
    templated on prototype.py's own trapezoid() model instead of a Mandel-
    Agol fit. This is a deliberate choice, not an oversight: running the
    Module 3 LSQ/MCMC machinery per target here would reintroduce the
    identifiability failures quantified at length in
    results/module3_convergence_diagnosis.md, for a task (relative epoch-to-
    epoch consistency) that does not need Mandel-Agol-level physical realism.
  * C4 reuses that same per-epoch depth array (odd/even split), and C6 reuses
    the SPOC light curve's own mom_centr1/mom_centr2 (+pos_corr1/2) columns
    -- no separate target-pixel-file download needed (confirmed available in
    the lightkurve product already used by download_lc()).
  * C3 and C5 are genuinely new (no prior code computed either): C3 measures
    mid-depth crossing-time symmetry about T0 from the binned phase-fold; C5
    measures the flux deficit at the phase-0.5 (secondary eclipse) location.
"""
import numpy as np

from prototype import trapezoid
import modules_3_5 as M


# ------------------------------------------------------------- C1, C2 -----
def compute_c1_c2(fit):
    """C1: transit depth (ppm), straight from characterize(). C2: flat-
    bottom-to-total-duration ratio -- high (~1) for a U-shaped (flat-
    bottomed) planet transit, low (~0) for a V-shaped (grazing/EB) eclipse.
    """
    c1 = float(fit["depth_ppm"])
    total = float(fit["total_dur"])
    c2 = float(fit["flat_dur"] / total) if (np.isfinite(total) and total > 0) else np.nan
    return c1, c2


# ------------------------------------------------------------------ C3 ----
def compute_c3_symmetry(ph_h, fl, fit, n_bins=150):
    """Mid-depth crossing-time symmetry about T0=0.

    prototype.py's trapezoid() model is symmetric BY CONSTRUCTION (a single
    ingress_hrs used for both sides), so C3 cannot be read off that fit --
    it needs an independent measurement. This bins the phase-folded light
    curve, finds the two times (one each side of T0) where the binned flux
    crosses the half-depth level, and reports
        C3 = min(|t_left|, |t_right|) / max(|t_left|, |t_right|)
    in (0, 1] -- 1 means the transit is symmetric about T0 (crossings
    equidistant); values well below 1 flag an asymmetric ingress/egress
    (e.g. a grazing or blended geometry). This is a symmetry PROXY (crossing
    time, not a literal ingress-vs-egress duration ratio) -- documented
    here rather than overclaimed as a direct measurement.
    """
    bl = fit["baseline"]
    depth_frac = fit["depth_ppm"] / 1e6
    Ttot = fit["total_dur"]
    if not (np.isfinite(depth_frac) and depth_frac > 0 and np.isfinite(Ttot) and Ttot > 0):
        return np.nan, {"note": "invalid trapezoid fit"}

    zoom = max(Ttot * 2.0, 2.0)
    m = np.isfinite(ph_h) & np.isfinite(fl) & (np.abs(ph_h) < zoom)
    if m.sum() < 30:
        return np.nan, {"note": "insufficient points near transit"}
    ph_z, fl_z = ph_h[m], fl[m]

    edges = np.linspace(-zoom, zoom, n_bins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    bmed = np.array([
        np.nanmedian(fl_z[(ph_z >= edges[i]) & (ph_z < edges[i + 1])])
        if np.sum((ph_z >= edges[i]) & (ph_z < edges[i + 1])) > 0 else np.nan
        for i in range(n_bins)
    ])
    ok = np.isfinite(bmed)
    ctrs, bmed = ctrs[ok], bmed[ok]
    if ok.sum() < 10:
        return np.nan, {"note": "insufficient binned coverage"}

    mid_level = bl - depth_frac / 2.0
    left = ctrs < 0
    right = ctrs > 0
    if left.sum() < 3 or right.sum() < 3:
        return np.nan, {"note": "insufficient one-sided coverage"}

    def closest_crossing(tt, ff, level):
        """Linear-interpolated crossing(s) of `level`, robust to bin-to-bin
        noise (does not require strictly monotonic endpoints) -- returns
        whichever crossing sits closest to T0=0."""
        order = np.argsort(tt)
        tt, ff = tt[order], ff[order]
        sgn = np.sign(ff - level)
        sgn[sgn == 0] = 1  # treat an exact hit as on the positive side, breaks ties
        change = np.where(np.diff(sgn) != 0)[0]
        if len(change) == 0:
            return np.nan
        crossings = []
        for i in change:
            f0, f1 = ff[i], ff[i + 1]
            if f1 == f0:
                continue
            frac = (level - f0) / (f1 - f0)
            crossings.append(tt[i] + frac * (tt[i + 1] - tt[i]))
        if not crossings:
            return np.nan
        crossings = np.asarray(crossings)
        return float(crossings[np.argmin(np.abs(crossings))])

    t_cross_left = closest_crossing(ctrs[left], bmed[left], mid_level)
    t_cross_right = closest_crossing(ctrs[right], bmed[right], mid_level)

    if not (np.isfinite(t_cross_left) and np.isfinite(t_cross_right)):
        return np.nan, {"note": "could not locate both mid-depth crossings"}

    a, b = abs(t_cross_left), abs(t_cross_right)
    if max(a, b) <= 0:
        return np.nan, {"note": "degenerate crossing times"}
    c3 = min(a, b) / max(a, b)
    return float(c3), {"t_cross_left_hr": float(t_cross_left), "t_cross_right_hr": float(t_cross_right)}


# ------------------------------------------------- per-epoch depths (C4, C7) --
def per_epoch_depths_trapezoid(t, f, period, t0, fit):
    """Per-epoch depths via matched filter against prototype.py's OWN
    trapezoid() fit (not a Mandel-Agol model -- see module docstring for
    why). Mirrors modules_3_5.per_epoch_depths()'s exact matched-filter
    pattern (deficit convention, amp*peak->ppm), reusing
    modules_3_5.matched_filter_snr() directly.
    """
    ph_h = ((t - t0 + 0.5 * period) % period - 0.5 * period) * 24.0
    template_flux = trapezoid(ph_h, fit["baseline"], fit["depth_ppm"], fit["total_dur"], fit["ingress_dur"])
    template = fit["baseline"] - template_flux  # deficit shape: 0 out-of-transit, peak = depth_ppm/1e6
    peak = np.nanmax(template)
    if not np.isfinite(peak) or peak <= 0:
        return np.array([]), np.array([])

    in_tr = template > 0.02 * peak
    if in_tr.sum() < 5:
        return np.array([]), np.array([])
    base = np.nanmedian(f[~in_tr]) if (~in_tr).sum() > 10 else np.nanmedian(f)

    epoch = np.round((t - t0) / period).astype(int)
    depths, epochs = [], []
    for e in np.unique(epoch[in_tr]):
        m = (epoch == e)
        if (m & in_tr).sum() < 3:
            continue
        amp, _ = M.matched_filter_snr(base - f[m], template[m], 1.0)
        if np.isfinite(amp):
            depths.append(amp * peak * 1e6)  # ppm
            epochs.append(e)
    return np.asarray(depths), np.asarray(epochs)


def compute_c4_odd_even(depths, epochs):
    """Relative |median(odd) - median(even)| / mean depth. ~0 for a planet
    (same depth every transit); large for an eclipsing binary with
    alternating primary/secondary-like depths."""
    if len(depths) < 4:
        return np.nan, {"note": "too few epochs (<4)"}
    odd = depths[epochs % 2 != 0]
    even = depths[epochs % 2 == 0]
    if len(odd) < 2 or len(even) < 2:
        return np.nan, {"note": "insufficient odd/even split"}
    med_odd, med_even = float(np.median(odd)), float(np.median(even))
    denom = (med_odd + med_even) / 2.0
    c4 = abs(med_odd - med_even) / denom if denom > 0 else np.nan
    return c4, {"median_odd_ppm": med_odd, "median_even_ppm": med_even,
                "n_odd": int(len(odd)), "n_even": int(len(even))}


def compute_c7_phase_fold_coherence(depths):
    """C7 = 1 - MAD(depths)/median(depths). Report Eq. 2 exactly.

    Edge case not addressed by Eq. 2 literally: an aliased fold can produce
    a MAJORITY of epochs with ~zero measured depth (the true eclipse landed
    elsewhere in phase for that epoch, so the fixed window catches only
    baseline) and a minority with large, erratic depths (confirmed on
    V1828 Aql: 28/46 epochs exactly 0, the other 18 ranging ~56,000-
    120,000 ppm). That drives the median to exactly 0, making the ratio
    undefined -- but a majority-zero/minority-huge split IS itself strong
    evidence of incoherence, so it is scored as C7=0 (maximally incoherent)
    rather than NaN. Only the fully-degenerate case (every epoch exactly
    zero, i.e. no signal detected anywhere) stays NaN, since that isn't a
    coherence question at all.
    """
    depths = np.asarray(depths, dtype=float)
    depths = depths[np.isfinite(depths)]
    if len(depths) < 2:
        return np.nan
    med = np.median(depths)
    if med == 0 or not np.isfinite(med):
        return 0.0 if np.any(depths != 0) else np.nan
    mad = np.median(np.abs(depths - med))
    return float(1.0 - mad / abs(med))


# ------------------------------------------------------------------ C5 ----
def compute_c5_secondary_eclipse(t, f, period, t0, duration_d, fit):
    """Flux deficit at the phase-0.5 (secondary eclipse) location, folded
    on t0_sec = t0 + P/2, excluding points too close to the primary window
    (relevant for short-period targets where windows could overlap)."""
    t0_sec = t0 + period / 2.0
    ph_sec_d = (t - t0_sec + 0.5 * period) % period - 0.5 * period
    ph_prim_d = (t - t0 + 0.5 * period) % period - 0.5 * period

    half = 1.5 * duration_d
    not_primary = np.abs(ph_prim_d) > half
    sec_window = (np.abs(ph_sec_d) < half) & not_primary
    if sec_window.sum() < 10:
        return np.nan, {"note": "insufficient secondary-window coverage"}

    oot = (~sec_window) & not_primary & (np.abs(ph_sec_d) > half)
    if oot.sum() < 20:
        return np.nan, {"note": "insufficient out-of-eclipse baseline"}

    base = float(np.nanmedian(f[oot]))
    sec_flux = float(np.nanmedian(f[sec_window]))
    c5_ppm = (base - sec_flux) * 1e6  # positive = a real dip at phase 0.5
    c5_ratio = c5_ppm / fit["depth_ppm"] if fit["depth_ppm"] > 0 else np.nan
    return c5_ppm, {"c5_ratio_to_primary": c5_ratio, "n_points": int(sec_window.sum())}


# ------------------------------------------------------------------ C6 ----
def _segment_center(t, c, gap_days=1.0):
    """Median-center `c` within each contiguous time segment (a gap >
    gap_days delimits a new segment). download_lc() stitches multiple TESS
    sectors into one light curve; mom_centr1/2 are ABSOLUTE detector pixel
    coordinates, and different sectors/cameras/CCDs place the same star at
    completely different absolute pixel positions. Comparing raw absolute
    centroids across a stitched multi-sector curve therefore measures
    sector-to-sector pointing differences, not a real transit-correlated
    shift (confirmed empirically: an uncentered TOI-700 check gave a
    150-pixel "shift", ~50 arcmin -- physically absurd for a real centroid
    shift). Centering each segment on its own median removes that offset
    while preserving genuine sub-segment centroid motion.
    """
    gaps = np.diff(t)
    breaks = np.where(gaps > gap_days)[0] + 1
    seg_ids = np.zeros(len(t), dtype=int)
    seg_ids[breaks] = 1
    seg_ids = np.cumsum(seg_ids)
    out = np.empty_like(c)
    for s in np.unique(seg_ids):
        m = seg_ids == s
        out[m] = c[m] - np.nanmedian(c[m])
    return out


def compute_c6_centroid_shift(lc_raw, period, t0, duration_d):
    """In-transit vs. out-of-transit photometric centroid shift, in pixels,
    from the SPOC light curve's own mom_centr1/mom_centr2 columns (flux-
    weighted moment centroids) -- no target-pixel-file download needed.
    Each sector/segment is median-centered first (see _segment_center) to
    remove sector-to-sector absolute pointing offsets, then pos_corr1/2
    (spacecraft pointing correction) is subtracted where available, so the
    measured shift isn't just residual pointing jitter either.
    """
    try:
        t = np.asarray(lc_raw.time.value, dtype=float)
        c1 = np.asarray(lc_raw["mom_centr1"], dtype=float)
        c2 = np.asarray(lc_raw["mom_centr2"], dtype=float)
    except Exception:
        return np.nan, {"note": "no centroid columns available on this light curve"}

    ok = np.isfinite(t) & np.isfinite(c1) & np.isfinite(c2)
    if ok.sum() < 20:
        return np.nan, {"note": "insufficient centroid data"}
    t, c1, c2 = t[ok], c1[ok], c2[ok]

    order = np.argsort(t)
    t, c1, c2 = t[order], c1[order], c2[order]
    c1 = _segment_center(t, c1)
    c2 = _segment_center(t, c2)

    try:
        pc1 = np.asarray(lc_raw["pos_corr1"], dtype=float)[ok][order]
        pc2 = np.asarray(lc_raw["pos_corr2"], dtype=float)[ok][order]
        if np.all(np.isfinite(pc1)) and np.all(np.isfinite(pc2)):
            c1 = c1 - _segment_center(t, pc1)
            c2 = c2 - _segment_center(t, pc2)
    except Exception:
        pass

    ph_d = (t - t0 + 0.5 * period) % period - 0.5 * period
    in_tr = np.abs(ph_d) < duration_d / 2.0
    oot = np.abs(ph_d) > duration_d
    if in_tr.sum() < 5 or oot.sum() < 20:
        return np.nan, {"note": "insufficient in/out-of-transit centroid coverage"}

    shift1 = float(np.nanmedian(c1[in_tr]) - np.nanmedian(c1[oot]))
    shift2 = float(np.nanmedian(c2[in_tr]) - np.nanmedian(c2[oot]))
    shift_pix = float(np.hypot(shift1, shift2))
    noise_pix = float(np.hypot(np.nanstd(c1[oot]), np.nanstd(c2[oot])) / np.sqrt(in_tr.sum()))
    sig = shift_pix / noise_pix if noise_pix > 0 else np.nan
    return shift_pix, {"shift_significance": sig, "n_in": int(in_tr.sum()), "n_oot": int(oot.sum())}


# ---------------------------------------------------------------- driver --
def compute_all_concepts(lc_raw, lc_clean, bls, fit):
    """Compute C1-C7 for one target. `lc_raw` = pre-detrend LightCurve (for
    C6's centroid columns); `lc_clean` = post-detrend; `bls` = identify()'s
    return dict (period, t0, duration, lc_fold); `fit` = characterize()'s
    return dict."""
    period, t0, duration_d = bls["period"], bls["t0"], bls["duration"]

    t = np.asarray(lc_clean.time.value, dtype=float)
    f = np.asarray(lc_clean.flux.value, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]

    ph_h = bls["lc_fold"].time.value * 24.0
    fl = bls["lc_fold"].flux.value

    c1, c2 = compute_c1_c2(fit)
    c3, c3_info = compute_c3_symmetry(ph_h, fl, fit)

    depths, epochs = per_epoch_depths_trapezoid(t, f, period, t0, fit)
    c4, c4_info = compute_c4_odd_even(depths, epochs)
    c7 = compute_c7_phase_fold_coherence(depths)

    c5, c5_info = compute_c5_secondary_eclipse(t, f, period, t0, duration_d, fit)
    c6, c6_info = compute_c6_centroid_shift(lc_raw, period, t0, duration_d)

    return {
        "C1_depth_ppm": c1, "C2_shape_flatbottom_ratio": c2,
        "C3_ingress_egress_symmetry": c3,
        "C4_odd_even_rel_diff": c4,
        "C5_secondary_eclipse_ppm": c5,
        "C6_centroid_shift_px": c6,
        "C7_phase_fold_coherence": c7,
        "n_epochs": len(depths),
        "_c3_info": c3_info, "_c4_info": c4_info, "_c5_info": c5_info, "_c6_info": c6_info,
    }
