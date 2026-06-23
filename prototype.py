#!/usr/bin/env python3
"""
VyomAnetra - Exoplanet Transit Detection Pipeline (Prototype)
"""

import os, sys, warnings, traceback, csv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk

warnings.filterwarnings("ignore")

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

TARGETS = [
    {"name": "TOI-700",   "terms": ["TOI-700",   "TIC 150428135"],            "label": "Planet"},
    {"name": "TOI-1338",  "terms": ["TOI-1338",  "TIC 260128333"],            "label": "Planet"},
    {"name": "Pi Mensae", "terms": ["Pi Mensae", "Pi Men", "HD 39091"],       "label": "Planet"},
    {"name": "V1828 Aql", "terms": ["V1828 Aql", "V* V1828 Aql", "HD 187091"], "label": "Eclipsing Binary"},
]

# -- Trapezoid transit model ------------------------------------------
def trapezoid(t_hrs, baseline, depth_ppm, total_hrs, ingress_hrs):
    """Symmetric trapezoid centred on t=0."""
    depth   = depth_ppm / 1e6
    half_T  = total_hrs / 2.0
    tau     = max(abs(ingress_hrs), 1e-8)
    half_fl = max(half_T - tau, 0.0)

    f  = np.full_like(t_hrs, baseline, dtype=np.float64)
    at = np.abs(t_hrs)

    f[at <= half_fl] = baseline - depth                          # flat bottom
    slope = (at > half_fl) & (at < half_T)
    if np.any(slope):
        f[slope] = baseline - depth * (half_T - at[slope]) / tau # ingress / egress
    return f

# -- MODULE 1  Detrending ---------------------------------------------
def download_lc(name, terms):
    """Try each search term against TESS/SPOC, then any author."""
    for author in ["SPOC", None]:
        tag = "SPOC" if author else "any-author"
        for term in terms:
            print(f"    search '{term}' ({tag}) ...", flush=True)
            try:
                kw = {"mission": "TESS"}
                if author:
                    kw["author"] = author
                sr = lk.search_lightcurve(term, **kw)
                if sr is None or len(sr) == 0:
                    continue
                n = min(len(sr), 4)
                print(f"    => {len(sr)} result(s); downloading {n} sector(s) ...", flush=True)
                lcc = sr[:n].download_all()
                lc  = lcc.stitch()
                lc  = lc.remove_nans()
                print(f"    => {len(lc)} cadences", flush=True)
                return lc
            except Exception as exc:
                print(f"    => failed: {exc}")
    return None


def detrend(lc, name):
    safe    = name.replace(" ", "_")
    lc_norm = lc.normalize()

    wl = min(301, len(lc_norm) // 3)
    if wl % 2 == 0:
        wl += 1
    wl = max(wl, 5)

    lc_flat  = lc_norm.flatten(window_length=wl)
    # only clip upward outliers so deep eclipses survive
    lc_clean = lc_flat.remove_outliers(sigma_upper=5, sigma_lower=1e4)
    lc_clean = lc_clean.remove_nans()
    print(f"    {len(lc_clean)} pts after flatten + clean", flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].scatter(lc_norm.time.value,  lc_norm.flux.value,  s=0.3, c="gray")
    axes[0].set_ylabel("Normalised Flux"); axes[0].set_title(f"{name} -Raw (normalised)")
    axes[1].scatter(lc_clean.time.value, lc_clean.flux.value, s=0.3, c="black")
    axes[1].set_ylabel("Flattened Flux"); axes[1].set_xlabel("Time [BTJD]")
    axes[1].set_title(f"{name} -After Savitzky-Golay Detrending")
    plt.tight_layout()
    p = os.path.join(RESULTS, f"{safe}_detrending.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"    saved {p}", flush=True)
    return lc_clean

# -- MODULE 2  BLS period search --------------------------------------
def identify(lc, name):
    safe = name.replace(" ", "_")
    t, f = lc.time.value, lc.flux.value
    ok   = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]

    span  = t[-1] - t[0]
    min_p = 0.5
    max_p = min(span / 2, 30.0)
    periods   = np.linspace(min_p, max_p, 10000)
    max_dur   = min(0.4, min_p * 0.8)               # must be < min_p
    durations = np.linspace(0.02, max_dur, 25)       # days

    print(f"    BLS: {min_p:.1f}-{max_p:.1f} d ({len(periods)} trials) ...", flush=True)
    bls = BoxLeastSquares(t, f)
    res = bls.power(periods, durations)

    idx = np.argmax(res.power)
    per, t0, dur, pwr = res.period[idx], res.transit_time[idx], res.duration[idx], res.power[idx]
    print(f"    period = {per:.5f} d | T0 = {t0:.5f} | dur = {dur*24:.2f} h | power = {pwr:.6f}", flush=True)

    lc_fold = lc.fold(period=per, epoch_time=t0)
    ph = lc_fold.time.value          # days from mid-transit
    fl = lc_fold.flux.value

    # plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(res.period, res.power, "k-", lw=0.5)
    axes[0].axvline(per, c="red", ls="--", label=f"P = {per:.4f} d")
    axes[0].set_xlabel("Period [d]"); axes[0].set_ylabel("BLS Power")
    axes[0].set_title(f"{name} -BLS Periodogram"); axes[0].legend()

    axes[1].scatter(ph, fl, s=0.3, c="gray", alpha=0.3)
    nb   = 200
    edg  = np.linspace(np.nanmin(ph), np.nanmax(ph), nb+1)
    ctrs = 0.5 * (edg[:-1] + edg[1:])
    bmed = [np.nanmedian(fl[(ph >= edg[i]) & (ph < edg[i+1])])
            if np.sum((ph >= edg[i]) & (ph < edg[i+1])) > 0 else np.nan
            for i in range(nb)]
    axes[1].plot(ctrs, bmed, "r.", ms=3, label="binned median")
    axes[1].set_xlabel("Phase [d]"); axes[1].set_ylabel("Flux")
    axes[1].set_title(f"{name} -Phase-Folded (P = {per:.4f} d)"); axes[1].legend()
    plt.tight_layout()
    p = os.path.join(RESULTS, f"{safe}_phase_folded.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"    saved {p}", flush=True)

    return {"period": per, "t0": t0, "duration": dur, "power": pwr, "lc_fold": lc_fold}

# -- MODULE 3  Trapezoid fit ------------------------------------------
def characterize(lc_fold, bls, name):
    safe = name.replace(" ", "_")
    ph_d = lc_fold.time.value
    fl   = lc_fold.flux.value
    ok   = np.isfinite(ph_d) & np.isfinite(fl)
    ph_d, fl = ph_d[ok], fl[ok]
    ph_h = ph_d * 24.0

    try:
        fl_err = lc_fold.flux_err.value[ok]
        if np.all(~np.isfinite(fl_err)) or np.nanmedian(fl_err) <= 0:
            fl_err = None
    except Exception:
        fl_err = None

    bl   = np.nanmedian(fl)
    mask = np.abs(ph_d) < bls["duration"] / 2
    d_g  = max((bl - np.nanmedian(fl[mask])) * 1e6, 10.0) if mask.sum() > 3 else 500.0
    T_g  = bls["duration"] * 24.0
    tau_g = T_g * 0.15

    p0 = [bl, d_g, T_g, tau_g]
    lo = [bl - 0.1,  1.0,  0.05,  0.001]
    hi = [bl + 0.1, 5e5, 72.0,  36.0]

    try:
        kw = dict(p0=p0, bounds=(lo, hi), maxfev=100000)
        if fl_err is not None:
            kw["sigma"] = fl_err; kw["absolute_sigma"] = True
        popt, pcov = curve_fit(trapezoid, ph_h, fl, **kw)
    except Exception as exc:
        print(f"    curve_fit failed ({exc}); using BLS estimates", flush=True)
        popt = np.array(p0)
        pcov = np.full((4, 4), np.nan)

    perr = np.sqrt(np.abs(np.diag(pcov)))
    bl_f, dp, Td, tau = popt
    bl_e, dp_e, Td_e, tau_e = perr
    fb   = max(Td - 2 * tau, 0.0)
    fb_e = np.sqrt(Td_e**2 + 4 * tau_e**2) if np.isfinite(Td_e) else np.nan

    print(f"    baseline       {bl_f:.6f} +/- {bl_e:.6f}", flush=True)
    print(f"    depth          {dp:.1f} +/- {dp_e:.1f} ppm", flush=True)
    print(f"    total dur      {Td:.3f} +/- {Td_e:.3f} h", flush=True)
    print(f"    ingress/egress {tau:.3f} +/- {tau_e:.3f} h", flush=True)
    print(f"    flat bottom    {fb:.3f} +/- {fb_e:.3f} h", flush=True)

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(ph_h, fl, s=0.3, c="gray", alpha=0.3)
    nb   = 200
    edg  = np.linspace(ph_h.min(), ph_h.max(), nb+1)
    ctrs = 0.5 * (edg[:-1] + edg[1:])
    bmed = [np.nanmedian(fl[(ph_h >= edg[i]) & (ph_h < edg[i+1])])
            if np.sum((ph_h >= edg[i]) & (ph_h < edg[i+1])) > 0 else np.nan
            for i in range(nb)]
    ax.plot(ctrs, bmed, "b.", ms=3, label="binned median", zorder=3)

    tm = np.linspace(ph_h.min(), ph_h.max(), 2000)
    ax.plot(tm, trapezoid(tm, *popt), "r-", lw=2, label="trapezoid fit", zorder=4)
    zoom = max(Td * 2.5, 3.0)
    ax.set_xlim(-zoom, zoom)
    ax.set_xlabel("Time from mid-transit [h]"); ax.set_ylabel("Flux")
    ax.set_title(f"{name} -Trapezoid Fit  (depth = {dp:.0f} ppm)")
    ax.legend(); plt.tight_layout()
    p = os.path.join(RESULTS, f"{safe}_trapezoid_fit.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"    saved {p}", flush=True)

    return dict(baseline=bl_f, depth_ppm=dp, depth_err=dp_e,
                total_dur=Td, total_dur_err=Td_e,
                ingress_dur=tau, ingress_err=tau_e,
                flat_dur=fb, flat_dur_err=fb_e)

# -- MODULE 4  Rule-based classifier -----------------------------------
def classify(row):
    """Flag detections that are likely not genuine planet transits."""
    depth = row["depth_ppm"]
    period = row["period"]

    if depth > 10000:
        flag = "LIKELY ECLIPSING BINARY"
    elif period > 20 and depth < 5000:
        flag = "POSSIBLE STELLAR VARIABILITY/ALIAS"
    else:
        flag = "PLANET CANDIDATE"

    print(f"    depth = {depth:.1f} ppm, period = {period:.4f} d => {flag}", flush=True)
    return {"flag": flag}


def reclassify_existing():
    """Read summary_table.csv, apply classify() rules, rewrite both output files."""
    csv_path = os.path.join(RESULTS, "summary_table.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run the full pipeline first.")
        return

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{'=' * 62}")
    print("  Module 4 — Rule-Based Classification (standalone)")
    print(f"{'=' * 62}\n")

    for r in rows:
        r["depth_ppm"] = float(r["depth_ppm"])
        r["period"] = float(r["period_days"])
        clf = classify(r)
        r["flag"] = clf["flag"]

    # Rewrite CSV
    csv_cols = ["target", "label", "period_days", "depth_ppm", "depth_err_ppm",
                "total_duration_hr", "ingress_egress_hr", "flat_bottom_hr", "snr", "noise_ppm", "flag"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for r in rows:
            w.writerow([r[c] for c in csv_cols])
    print(f"\nSaved {csv_path}")

    # Rewrite TXT
    W = 160
    txt_path = os.path.join(RESULTS, "summary_table.txt")
    hdr = (f"{'Target':<15} {'Label':<20} {'Period(d)':<12} {'Depth(ppm)':<13} "
           f"{'Duration(h)':<13} {'Ingress(h)':<12} {'FlatBot(h)':<12} {'SNR':<10} {'Flag':<40}")
    lines = []
    lines.append("=" * W)
    lines.append(f"{'FINAL SUMMARY TABLE':^{W}}")
    lines.append("=" * W)
    lines.append(hdr)
    lines.append("-" * W)
    for r in rows:
        lines.append(
            f"{r['target']:<15} {r['label']:<20} {float(r['period_days']):<12.4f} "
            f"{float(r['depth_ppm']):<13.1f} {float(r['total_duration_hr']):<13.3f} "
            f"{float(r['ingress_egress_hr']):<12.3f} {float(r['flat_bottom_hr']):<12.3f} "
            f"{float(r['snr']):<10.1f} {r['flag']:<40}")
    lines.append("=" * W)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {txt_path}")

    print(f"\n{'=' * W}")
    print(f"{'FINAL SUMMARY TABLE':^{W}}")
    print(f"{'=' * W}")
    print(hdr)
    print("-" * W)
    for r in rows:
        print(f"{r['target']:<15} {r['label']:<20} {float(r['period_days']):<12.4f} "
              f"{float(r['depth_ppm']):<13.1f} {float(r['total_duration_hr']):<13.3f} "
              f"{float(r['ingress_egress_hr']):<12.3f} {float(r['flat_bottom_hr']):<12.3f} "
              f"{float(r['snr']):<10.1f} {r['flag']:<40}")
    print("=" * W)


# -- MODULE 5  Statistical significance -------------------------------
def significance(lc_fold, bls, fit):
    ph = lc_fold.time.value
    fl = lc_fold.flux.value
    ok = np.isfinite(ph) & np.isfinite(fl)
    ph, fl = ph[ok], fl[ok]

    dur   = bls["duration"]
    in_tr = np.abs(ph) < dur / 2
    oot   = np.abs(ph) > dur
    n_in  = int(in_tr.sum())

    noise = np.nanstd(fl[oot]) * 1e6 if oot.sum() > 10 else np.nan
    dp    = fit["depth_ppm"]
    snr   = dp / (noise / np.sqrt(n_in)) if (n_in > 0 and noise > 0) else np.nan

    print(f"    depth      = {dp:.1f} ppm", flush=True)
    print(f"    OOT noise  = {noise:.1f} ppm", flush=True)
    print(f"    N in-transit = {n_in}", flush=True)
    print(f"    SNR        = {snr:.1f}", flush=True)
    return {"snr": snr, "noise_ppm": noise}

# -- Run one target through the whole pipeline ------------------------
def process(tgt):
    name, terms, label = tgt["name"], tgt["terms"], tgt["label"]
    print(f"\n{'=' * 62}")
    print(f"  TARGET: {name}   (expected: {label})")
    print(f"{'=' * 62}", flush=True)
    row = {"name": name, "label": label}

    print(f"\n  -- Module 1 - Detrending --", flush=True)
    lc = download_lc(name, terms)
    if lc is None:
        print(f"  *** SKIPPED -no TESS data found for {name}", flush=True)
        return None
    lc_c = detrend(lc, name)

    print(f"\n  -- Module 2 - BLS Identification --", flush=True)
    bls = identify(lc_c, name)
    row["period"] = bls["period"]

    print(f"\n  -- Module 3 - Trapezoid Characterization --", flush=True)
    fit = characterize(bls["lc_fold"], bls, name)
    row.update(fit)

    print(f"\n  -- Module 4 - Rule-Based Classification --", flush=True)
    clf = classify(row)
    row.update(clf)

    print(f"\n  -- Module 5 - Statistical Significance --", flush=True)
    sig = significance(bls["lc_fold"], bls, fit)
    row.update(sig)
    return row

# -- MAIN -------------------------------------------------------------
def main():
    print("=" * 62)
    print("  VyomAnetra - Exoplanet Transit Detection Pipeline")
    print("  Prototype v0.1")
    print("=" * 62, flush=True)

    rows = []
    for tgt in TARGETS:
        try:
            r = process(tgt)
            if r is not None:
                rows.append(r)
        except Exception as exc:
            print(f"\n  *** ERROR on {tgt['name']}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    W = 160
    print(f"\n\n{'=' * W}")
    print(f"{'FINAL SUMMARY TABLE':^{W}}")
    print(f"{'=' * W}")
    hdr = (f"{'Target':<15} {'Label':<20} {'Period(d)':<12} {'Depth(ppm)':<13} "
           f"{'Duration(h)':<13} {'Ingress(h)':<12} {'FlatBot(h)':<12} {'SNR':<10} {'Flag':<40}")
    print(hdr)
    print("-" * W)
    for r in rows:
        print(f"{r['name']:<15} {r['label']:<20} {r['period']:<12.4f} "
              f"{r['depth_ppm']:<13.1f} {r['total_dur']:<13.3f} "
              f"{r['ingress_dur']:<12.3f} {r['flat_dur']:<12.3f} "
              f"{r['snr']:<10.1f} {r['flag']:<40}")
    print("=" * W)

    print(f"\nPlots in {RESULTS}/:")
    for fn in sorted(os.listdir(RESULTS)):
        if fn.endswith(".png"):
            kb = os.path.getsize(os.path.join(RESULTS, fn)) / 1024
            print(f"  {fn}  ({kb:.0f} KB)")

    # Save summary_table.txt
    txt_path = os.path.join(RESULTS, "summary_table.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * W}\n")
        f.write(f"{'FINAL SUMMARY TABLE':^{W}}\n")
        f.write(f"{'=' * W}\n")
        f.write(hdr + "\n")
        f.write("-" * W + "\n")
        for r in rows:
            f.write(f"{r['name']:<15} {r['label']:<20} {r['period']:<12.4f} "
                    f"{r['depth_ppm']:<13.1f} {r['total_dur']:<13.3f} "
                    f"{r['ingress_dur']:<12.3f} {r['flat_dur']:<12.3f} "
                    f"{r['snr']:<10.1f} {r['flag']:<40}\n")
        f.write("=" * W + "\n")
    print(f"\nSaved {txt_path}")

    # Save summary_table.csv
    csv_path = os.path.join(RESULTS, "summary_table.csv")
    csv_cols = ["target", "label", "period_days", "depth_ppm", "depth_err_ppm",
                "total_duration_hr", "ingress_egress_hr", "flat_bottom_hr", "snr", "noise_ppm", "flag"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for r in rows:
            w.writerow([
                r["name"], r["label"], f"{r['period']:.6f}",
                f"{r['depth_ppm']:.2f}", f"{r['depth_err']:.2f}",
                f"{r['total_dur']:.4f}", f"{r['ingress_dur']:.4f}",
                f"{r['flat_dur']:.4f}", f"{r['snr']:.2f}", f"{r['noise_ppm']:.2f}",
                r["flag"],
            ])
    print(f"Saved {csv_path}")

def run_single_target(target_name):
    """Run the pipeline for one target and merge result into existing CSV/TXT."""
    tgt = None
    for t in TARGETS:
        if t["name"].lower() == target_name.lower():
            tgt = t
            break
    if tgt is None:
        print(f"ERROR: target '{target_name}' not found in TARGETS list.")
        return

    print("=" * 62)
    print(f"  VyomAnetra — Single-Target Run: {tgt['name']}")
    print("=" * 62, flush=True)

    r = process(tgt)
    if r is None:
        print(f"\n  *** No result for {tgt['name']}.")
        return

    csv_path = os.path.join(RESULTS, "summary_table.csv")
    csv_cols = ["target", "label", "period_days", "depth_ppm", "depth_err_ppm",
                "total_duration_hr", "ingress_egress_hr", "flat_bottom_hr", "snr", "noise_ppm", "flag"]

    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = [row for row in reader if row["target"] != tgt["name"]]

    new_row = {
        "target": r["name"], "label": r["label"], "period_days": f"{r['period']:.6f}",
        "depth_ppm": f"{r['depth_ppm']:.2f}", "depth_err_ppm": f"{r['depth_err']:.2f}",
        "total_duration_hr": f"{r['total_dur']:.4f}", "ingress_egress_hr": f"{r['ingress_dur']:.4f}",
        "flat_bottom_hr": f"{r['flat_dur']:.4f}", "snr": f"{r['snr']:.2f}",
        "noise_ppm": f"{r['noise_ppm']:.2f}", "flag": r["flag"],
    }
    existing.append(new_row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for row in existing:
            w.writerow([row[c] for c in csv_cols])
    print(f"\nSaved {csv_path}")

    W = 160
    txt_path = os.path.join(RESULTS, "summary_table.txt")
    hdr = (f"{'Target':<15} {'Label':<20} {'Period(d)':<12} {'Depth(ppm)':<13} "
           f"{'Duration(h)':<13} {'Ingress(h)':<12} {'FlatBot(h)':<12} {'SNR':<10} {'Flag':<40}")
    lines = ["=" * W, f"{'FINAL SUMMARY TABLE':^{W}}", "=" * W, hdr, "-" * W]
    for row in existing:
        lines.append(
            f"{row['target']:<15} {row['label']:<20} {float(row['period_days']):<12.4f} "
            f"{float(row['depth_ppm']):<13.1f} {float(row['total_duration_hr']):<13.3f} "
            f"{float(row['ingress_egress_hr']):<12.3f} {float(row['flat_bottom_hr']):<12.3f} "
            f"{float(row['snr']):<10.1f} {row['flag']:<40}")
    lines.append("=" * W)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {txt_path}")

    print(f"\n{'=' * W}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    if "--reclassify" in sys.argv:
        reclassify_existing()
    elif "--target" in sys.argv:
        idx = sys.argv.index("--target")
        if idx + 1 < len(sys.argv):
            run_single_target(sys.argv[idx + 1])
        else:
            print("ERROR: --target requires a target name argument.")
    else:
        main()
