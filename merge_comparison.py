#!/usr/bin/env python3
"""
Merge results/summary_table.csv (BLS/trapezoid pipeline) with
results/exoveil_baseline.csv (ExoVeil, Priyanshu 2026) into one
comparison table for the paper's results section.

Also phase-folds ExoVeil's (period-less, single-event) detections against
the BLS-derived period + t0 (results/bls_ephemeris.json, produced by
compute_bls_ephemeris.py) so the table states explicitly whether the two
pipelines' top detections land on the same transit, instead of just listing
them side by side.

Usage: python merge_comparison.py
Requires: results/summary_table.csv, results/exoveil_baseline.csv,
          results/exoveil_baseline.json, results/bls_ephemeris.json
Writes:   results/comparison_table.csv and results/comparison_table.md
"""
import os
import csv
import json
import math

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# SNR values above this are physically implausible for this detector (typical
# real events across all 4 targets run 5-16) and indicate the local-noise
# denominator in exoveil.detect.detect_events() collapsed near zero, not a
# genuine high-significance detection.
SNR_SANITY_MAX = 1000.0

# Refined Pi Men c transit ephemeris (Table 2 of arXiv:2607.12088, "TESS
# Photometry and Radial Velocity Analysis of the sub-Neptune Exoplanet Pi
# Mensae c", July 2026 -- uses 20 TESS sectors, S1-S95). Period uncertainty
# is ~1 order of magnitude smaller than the prior (Gandolfi/Huang 2018-era,
# also tabulated in that paper's Table 1) solution. Used only for the Pi
# Mensae cross-check below -- independent of prototype.py's own (aliased)
# BLS period for this target.
PI_MEN_C_PERIOD = 6.267823            # days (Table 2)
PI_MEN_C_PERIOD_ERR = 1e-6            # days (Table 2)
PI_MEN_C_T0_BTJD = 1325.5042          # BTJD (Table 2)
PI_MEN_C_T0_ERR = 0.0003              # days (Table 2)
PI_MEN_C_DUR_HR = 2.969               # T14, hours (Table 1; not restated in Table 2)

# Prior (pre-2607.12088) solution, for the "order of magnitude" comparison.
PI_MEN_C_PERIOD_OLD_ERR = 0.000016    # days (Table 1)
PI_MEN_C_T0_OLD_ERR = 0.00028         # days (NASA Exoplanet Archive, same era)

# Sectors actually used by prototype.py's download_lc("Pi Mensae", ...):
# confirmed by replaying its exact search (lk.search_lightcurve("Pi Mensae",
# mission="TESS", author="SPOC")) and taking sr[:min(len(sr),4)], same
# selection logic download_lc() uses. First 4 (2-min cadence) results, in
# MAST's returned order, are Sectors 1, 4, 8, 11.
PI_MEN_SECTORS_USED = [1, 4, 8, 11]


def fmt(v, spec=None):
    if v in (None, "", "n/a"):
        return "n/a"
    if spec:
        try:
            return format(float(v), spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def fold_phase(t_event, period, t0, half_dur_days):
    """Return (phase_days, phase_hr, in_transit) for one event time.

    phase_days is the signed offset from the nearest predicted transit
    center, in days, folded into [-period/2, +period/2). in_transit is True
    if that offset falls within the BLS/trapezoid fitted transit half-duration.
    """
    raw_phase = ((t_event - t0) / period) % 1.0
    centered = raw_phase - 1.0 if raw_phase > 0.5 else raw_phase
    phase_days = centered * period
    in_transit = abs(phase_days) <= half_dur_days
    return phase_days, phase_days * 24.0, in_transit


with open(os.path.join(RESULTS, "summary_table.csv"), encoding="utf-8") as f:
    bls = {r["target"]: r for r in csv.DictReader(f)}

with open(os.path.join(RESULTS, "exoveil_baseline.json"), encoding="utf-8") as f:
    exo_full = {r["target"]: r for r in json.load(f)}

with open(os.path.join(RESULTS, "bls_ephemeris.json"), encoding="utf-8") as f:
    ephem = json.load(f)

targets = list(bls.keys())  # preserves prototype.py TARGETS order

cols = [
    "target", "true_label",
    "bls_period_d", "bls_t0_btjd", "bls_depth_ppm", "bls_snr", "bls_flag",
    "exoveil_n_events",
    "exoveil_top_snr_raw", "exoveil_snr_flag",
    "exoveil_top_depth_ppm", "exoveil_top_time_btjd", "exoveil_uncertainty",
    "exoveil_top_phase_hr", "exoveil_top_in_transit",
    "n_events_in_transit_of_total",
    "n_events_in_transit_of_lit_ephemeris",
    "notes",
]

rows = []
for t in targets:
    b = bls[t]
    e = exo_full.get(t, {})
    events = json.loads(e["all_events"]) if e.get("all_events") else []
    eph = ephem.get(t)

    period = eph["period"] if eph else None
    t0 = eph["t0"] if eph else None
    half_dur_days = (float(b["total_duration_hr"]) / 24.0) / 2.0 if eph else None

    # Fold every returned event, not just the top one, so we can report how
    # many (if any) land in-transit -- the top-SNR event is not necessarily
    # the one that matches the BLS ephemeris.
    folded = []
    if eph:
        for ev in events:
            pd, ph, intr = fold_phase(ev["time"], period, t0, half_dur_days)
            folded.append({**ev, "phase_days": pd, "phase_hr": ph, "in_transit": intr})
    n_in_transit = sum(1 for f_ in folded if f_["in_transit"])

    top = folded[0] if folded else None
    top_snr_raw = top["snr"] if top else None
    snr_flag = ""
    if top_snr_raw is not None and top_snr_raw > SNR_SANITY_MAX:
        snr_flag = "SNR undefined (near-zero local noise)"

    # -- Notes: mechanism / caveats + phase-fold verdict, per target --------
    note_parts = []
    lit_ephemeris_flag = "n/a"  # only computed for Pi Mensae (has a precise published ephemeris)

    if top is not None:
        verdict = "IN-TRANSIT (phase-agrees with BLS)" if top["in_transit"] else "OUT-OF-TRANSIT (does not match BLS ephemeris)"
        note_parts.append(
            f"Top ExoVeil event folds to {top['phase_hr']:+.2f} h from nearest "
            f"predicted transit center (needs within "
            f"±{half_dur_days*24:.2f} h) -> {verdict}."
        )
        if n_in_transit:
            note_parts.append(
                f"{n_in_transit}/{len(folded)} returned ExoVeil events fall "
                f"in-transit against the BLS ephemeris."
            )
        else:
            note_parts.append("0 of the returned ExoVeil events fall in-transit against the BLS ephemeris.")

    if t == "V1828 Aql":
        note_parts.append(
            "EB (P=0.66d) outside ExoVeil's validated regime. Top event's raw "
            f"SNR ({top_snr_raw:.3e}) is a numerical artifact, not a real "
            "detection significance: exoveil.detect.detect_events() weights "
            "residuals by 1/local_MAD(residual); the EB's sharp, near-total "
            "eclipse drives the local MAD toward ~0 in that window, blowing "
            "up the ratio. Second-ranked event (SNR=62.0, t=3528.67 BTJD) is "
            "a more representative 'top' detection for this target -- "
            "included in all_events for reference."
        )
        second = folded[1] if len(folded) > 1 else None
        if second:
            v2 = "in-transit" if second["in_transit"] else "out-of-transit"
            note_parts.append(
                f"That 2nd-ranked event folds to {second['phase_hr']:+.2f} h "
                f"from transit center -> {v2}."
            )
        note_parts.append(
            "Illustrates the single-transit matched filter breaking "
            "numerically on non-planet, high-contrast periodic signals."
        )
    elif t == "TOI-1338":
        note_parts.append(
            "Circumbinary system: BLS flags 'LIKELY ECLIPSING BINARY' at "
            "59,620 ppm (almost certainly the EB eclipse, not the "
            "circumbinary planet transit) -- so this phase-fold check "
            "verifies agreement with the EB signal, not with TOI-1338 b. "
            "Neither pipeline isolates the actual planet signal in this run."
        )
    elif t == "Pi Mensae":
        top2 = folded[:2]
        detail = "; ".join(
            f"event {i+1} (SNR {f_['snr']:.2f}, depth {f_['depth_ppm']:.0f} ppm) "
            f"folds to {f_['phase_hr']:+.2f} h -> {'in-transit' if f_['in_transit'] else 'out-of-transit'}"
            for i, f_ in enumerate(top2)
        )
        pm_half_dur = (PI_MEN_C_DUR_HR / 24.0) / 2.0
        pm_folded = [fold_phase(f_["time"], PI_MEN_C_PERIOD, PI_MEN_C_T0_BTJD, pm_half_dur) for f_ in folded]
        pm_hits = sum(1 for (_, _, intr) in pm_folded if intr)
        lit_ephemeris_flag = f"{pm_hits}/{len(folded)}"
        pm_detail = "; ".join(
            f"event {i+1} folds to {ph:+.2f} h -> {'IN-TRANSIT' if intr else 'out-of-transit'}"
            for i, (pd, ph, intr) in enumerate(pm_folded[:2])
        )

        # Propagate ephemeris uncertainty to the epoch of our own data
        # (N_cycles x sigma_P, combined in quadrature with sigma_T0), at the
        # event farthest from T0 in our baseline -- the worst case.
        ev_times = [f_["time"] for f_ in folded]
        t_far = max(ev_times, key=lambda tt: abs(tt - PI_MEN_C_T0_BTJD))
        n_cycles = (t_far - PI_MEN_C_T0_BTJD) / PI_MEN_C_PERIOD
        sigma_t_new_hr = math.sqrt(PI_MEN_C_T0_ERR**2 + (n_cycles * PI_MEN_C_PERIOD_ERR)**2) * 24.0
        sigma_t_old_hr = math.sqrt(PI_MEN_C_T0_OLD_ERR**2 + (n_cycles * PI_MEN_C_PERIOD_OLD_ERR)**2) * 24.0
        frac_of_window = sigma_t_new_hr / (pm_half_dur * 24.0)

        note_parts.append(
            f"Our Pi Mensae light curve draws on TESS sectors "
            f"{', '.join(str(s) for s in PI_MEN_SECTORS_USED)} (confirmed via "
            "the lightkurve search results, same selection order "
            "download_lc() uses), spanning BTJD "
            f"{min(ev_times):.1f}-{max(ev_times):.1f}."
        )
        note_parts.append(
            f"Checked against BLS's own 27.21 d period (not Pi Men c's real "
            f"period): {detail}. A mismatch here reflects BLS locking onto a "
            "different (likely aliased) period rather than necessarily "
            "meaning ExoVeil is wrong."
        )
        note_parts.append(
            "Cross-checked instead against the refined Pi Men c ephemeris of "
            f"Larsen et al. (2026, arXiv:2607.12088, \"TESS Photometry and "
            "Radial Velocity Analysis of the sub-Neptune Exoplanet Pi Mensae "
            f"c...\", Table 2; 20 TESS sectors S1-S95): P={PI_MEN_C_PERIOD:.6f}"
            f"±{PI_MEN_C_PERIOD_ERR:.0e} d, T0={PI_MEN_C_T0_BTJD:.4f}"
            f"±{PI_MEN_C_T0_ERR:.4f} BTJD, T14={PI_MEN_C_DUR_HR:.3f} h -- "
            f"top-2: {pm_detail}. Extending to **all {len(folded)} returned "
            f"events**: {pm_hits}/{len(folded)} land in-transit."
        )
        note_parts.append(
            f"Ephemeris-uncertainty check: propagating T0/P uncertainty "
            f"(N_cycles x sigma_P, quadrature with sigma_T0) out to our "
            f"most-distant event ({n_cycles:.1f} cycles from T0) gives a "
            f"timing uncertainty of only {sigma_t_new_hr*60:.2f} min "
            f"({frac_of_window*100:.2f}% of the ±{pm_half_dur*24:.2f} h "
            "half-duration window) using the refined ephemeris -- vs. "
            f"{sigma_t_old_hr*60:.2f} min with the prior "
            "(pre-2607.12088, ~10x larger period-error) solution. Even the "
            "OLD ephemeris was already precise enough here; the refined one "
            "makes the margin larger still. Since the propagated uncertainty "
            "is <1% of the search window, the near-total absence of "
            "in-transit events (0-1/20 depending on ephemeris used, vs. "
            "top-2 SNR ~15.6/15.6) is NOT an ephemeris-precision artifact -- "
            "it stands as a genuine non-detection of a known, previously-"
            "confirmed planet by ExoVeil at threshold=5.0 in this run, not "
            "just a top-event mismatch."
        )
    elif t == "TOI-700":
        note_parts.append(
            "12 candidate events in total; phase-fold uses the BLS 16.05 d "
            "period/t0 (itself possibly an alias of the system's actual "
            "~10-37 d multi-planet ephemerides, which BLS is not guaranteed "
            "to recover from a single dominant peak)."
        )

    rows.append({
        "target": t,
        "true_label": b["label"],
        "bls_period_d": b["period_days"],
        "bls_t0_btjd": f"{t0:.6f}" if t0 is not None else "n/a",
        "bls_depth_ppm": b["depth_ppm"],
        "bls_snr": b["snr"],
        "bls_flag": b["flag"],
        "exoveil_n_events": e.get("n_events", "n/a"),
        "exoveil_top_snr_raw": f"{top_snr_raw:.6g}" if top_snr_raw is not None else "n/a",
        "exoveil_snr_flag": snr_flag,
        "exoveil_top_depth_ppm": top["depth_ppm"] if top else "n/a",
        "exoveil_top_time_btjd": top["time"] if top else "n/a",
        "exoveil_uncertainty": top["uncertainty_category"] if top else "n/a",
        "exoveil_top_phase_hr": f"{top['phase_hr']:+.2f}" if top else "n/a",
        "exoveil_top_in_transit": ("YES" if top["in_transit"] else "NO") if top else "n/a",
        "n_events_in_transit_of_total": f"{n_in_transit}/{len(folded)}" if folded else "n/a",
        "n_events_in_transit_of_lit_ephemeris": lit_ephemeris_flag,
        "notes": " ".join(note_parts),
    })

# --- CSV ---
csv_path = os.path.join(RESULTS, "comparison_table.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"Saved {csv_path}")

# --- Markdown ---
md_path = os.path.join(RESULTS, "comparison_table.md")
lines = []
lines.append("# VyomAnetra vs. ExoVeil -- baseline comparison\n")
lines.append(
    "BLS/trapezoid pipeline (this work) vs. ExoVeil "
    "(Priyanshu 2026, arXiv:2606.02778, PyPI `exoveil==0.2.1`), run on the "
    "identical Savitzky-Golay-flattened flux array for each target "
    "(`download_lc()` + `detrend()` shared between both pipelines).\n"
)
lines.append(
    "**Note on ExoVeil's output**: the shipped v0.2.1 package performs raw, "
    "non-phase-folded, single-transit-event detection only -- "
    "`detect_from_array()` returns a list of candidate events "
    "(`time`, `snr`, `depth_ppm`, `duration_pts`, `uncertainty_category`), "
    "each independently scored. It does **not** expose an orbital "
    "period/epoch, and does **not** expose the planet-vs-false-positive "
    "XGBoost classification or conformal-interval output described in the "
    "paper text -- those either aren't wired into the public API at this "
    "version, or require a separate call not documented in the package.\n"
)
lines.append(
    "**Phase-fold methodology**: `summary_table.csv` only stores BLS's "
    "period, not its epoch (t0), so t0 was recomputed by re-running "
    "prototype.py's own `identify()` (`compute_bls_ephemeris.py`, "
    "prototype.py itself untouched) -- recovered periods matched "
    "`summary_table.csv` exactly. Each ExoVeil event time is folded against "
    "(BLS period, BLS t0); \"in-transit\" means the folded offset falls "
    "within ±(fitted total transit duration)/2 from `summary_table.csv`. "
    "\"ExoVeil top event\" is the highest-SNR event returned "
    "(default `threshold=5.0`).\n"
)
lines.append(
    "**Note on the V1828 Aql SNR value**: the top event's raw SNR "
    "(~1.09e7) is flagged `SNR undefined (near-zero local noise)` rather "
    "than clipped or dropped -- it is kept in `exoveil_top_snr_raw` for "
    "transparency, but should not be read as a detection significance. See "
    "per-target notes for the mechanism.\n"
)
lines.append(
    "**Pi Mensae only**: additionally cross-checked against the refined "
    "Pi Men c ephemeris of Larsen et al. (2026, arXiv:2607.12088, Table 2: "
    f"P={PI_MEN_C_PERIOD:.6f}±{PI_MEN_C_PERIOD_ERR:.0e} d, "
    f"T0={PI_MEN_C_T0_BTJD:.4f}±{PI_MEN_C_T0_ERR:.4f} BTJD, "
    f"T14={PI_MEN_C_DUR_HR:.3f} h from their Table 1) rather than the BLS "
    "period for this target, which is itself flagged 'POSSIBLE STELLAR "
    "VARIABILITY/ALIAS' and is not Pi Men c's real period. The refined "
    "ephemeris's period uncertainty is ~1 order of magnitude smaller than "
    "the prior (Gandolfi/Huang 2018-era) solution, per that paper's "
    "abstract. Propagated to our data's epoch this uncertainty is <1% of "
    "the transit half-duration window (see per-target note) -- small "
    "enough that the resulting in-transit count is a genuine non-detection "
    "check, not an ephemeris-precision artifact. See the \"Events in-transit "
    "(lit. ephem.)\" column and per-target notes for the full propagation.\n"
)

header = ("| Target | True label | BLS P (d) | BLS t0 (BTJD) | BLS depth (ppm) | BLS SNR | "
          "BLS flag | ExoVeil #events | ExoVeil top SNR (raw) | SNR flag | "
          "ExoVeil top depth (ppm) | ExoVeil top t (BTJD) | Uncertainty | "
          "Phase offset (h) | In-transit? | Events in-transit (BLS ephem.) | "
          "Events in-transit (lit. ephem.) | Notes |")
sep = "|" + "---|" * 17
lines.append(header)
lines.append(sep)
for r in rows:
    lines.append(
        f"| {r['target']} | {r['true_label']} | "
        f"{fmt(r['bls_period_d'], '.4f')} | {r['bls_t0_btjd']} | "
        f"{fmt(r['bls_depth_ppm'], '.1f')} | {fmt(r['bls_snr'], '.1f')} | "
        f"{r['bls_flag']} | {r['exoveil_n_events']} | "
        f"{r['exoveil_top_snr_raw']} | {r['exoveil_snr_flag'] or '-'} | "
        f"{fmt(r['exoveil_top_depth_ppm'], '.1f')} | "
        f"{fmt(r['exoveil_top_time_btjd'], '.3f')} | "
        f"{r['exoveil_uncertainty']} | {r['exoveil_top_phase_hr']} | "
        f"{r['exoveil_top_in_transit']} | {r['n_events_in_transit_of_total']} | "
        f"{r['n_events_in_transit_of_lit_ephemeris']} | "
        f"{r['notes']} |"
    )

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved {md_path}")
