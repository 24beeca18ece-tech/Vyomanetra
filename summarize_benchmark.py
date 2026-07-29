#!/usr/bin/env python3
"""
Summarize results/benchmark_results.csv into recovery-rate statistics for the
paper: BLS/trapezoid vs. ExoVeil, overall and broken down by transit-depth
decile, scored against the NASA Exoplanet Archive ephemeris as ground truth.

IMPORTANT METHODOLOGICAL POINT (drives how these numbers must be read):
ExoVeil returns up to 20 independently-scored candidate events per target, and
"recovered" is defined as ">=1 event lands in-transit". The probability that a
*randomly placed* event lands in-transit is ~ (T14 / P), so with N events the
chance of at least one false hit is 1 - (1 - T14/P)^N, which for typical
values (T14=3h, P=5d, N=20) is ~40%. A raw "fraction of targets with >=1
in-transit event" therefore has a large chance-coincidence floor and is NOT
directly comparable to BLS's single-epoch check (chance ~ T14/P, ~2.5%).

This script reports both the raw rate and the per-target chance expectation,
plus an excess-over-chance figure, so the comparison is honest. A "strict"
BLS metric (period must also match the archive period within tolerance, or a
low-order harmonic of it) is reported alongside the epoch-only metric.

Usage: python summarize_benchmark.py
Writes: results/benchmark_summary.md
"""
import os

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_CSV = os.path.join(RESULTS, "benchmark_results.csv")
SAMPLE_CSV = os.path.join(RESULTS, "benchmark_sample.csv")
RECHECK_CSV = os.path.join(RESULTS, "long_period_recheck.csv")
OUT_MD = os.path.join(RESULTS, "benchmark_summary.md")

PERIOD_TOL = 0.01  # 1% fractional tolerance for "BLS period matches archive"
HARMONICS = [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0]

# prototype.py's identify() searches periods up to max_p = min(span/2, 30.0).
# Any target whose true period exceeds that ceiling is structurally
# unrecoverable by BLS regardless of data quality or detrending -- it is a
# configuration limit, not a detection failure, and counting it as an ordinary
# miss understates BLS. Rates are reported both including and excluding these.
BLS_MAX_PERIOD_CAP = 30.0


def period_matches(bls_p, arch_p):
    """True if BLS period is within tolerance of the archive period or a
    low-order harmonic/subharmonic of it (BLS commonly locks onto 2x/0.5x)."""
    if not np.isfinite(bls_p) or not np.isfinite(arch_p) or arch_p <= 0:
        return False, None
    for h in HARMONICS:
        if abs(bls_p - arch_p * h) / (arch_p * h) < PERIOD_TOL:
            return True, h
    return False, None


def main():
    df = pd.read_csv(RESULTS_CSV)
    sample = pd.read_csv(SAMPLE_CSV)
    n_sample = len(sample)

    ok = df[df["status"] == "ok"].copy()
    n_ok = len(ok)
    n_failed = len(df) - n_ok
    missing = set(sample["target"]) - set(df["target"])

    for c in ["bls_in_transit", "exoveil_any_in_transit"]:
        ok[c] = ok[c].astype(str).str.lower().eq("true")
    for c in ["archive_period_d", "archive_duration_hr", "archive_depth_ppm",
              "bls_period_d", "exoveil_n_events", "exoveil_n_events_in_transit"]:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    # ---- chance-coincidence baselines -----------------------------------
    # p_single = fraction of the orbital phase that is "in transit"
    ok["p_single"] = (ok["archive_duration_hr"] / 24.0) / ok["archive_period_d"]
    ok["p_single"] = ok["p_single"].clip(0, 1)
    n_ev = ok["exoveil_n_events"].fillna(0)
    ok["p_exoveil_chance"] = 1.0 - np.power(1.0 - ok["p_single"], n_ev)

    # ---- strict BLS metric: epoch in-transit AND period matches ----------
    strict = []
    harm = []
    for _, r in ok.iterrows():
        m, h = period_matches(r["bls_period_d"], r["archive_period_d"])
        strict.append(bool(r["bls_in_transit"]) and m)
        harm.append(h if m else None)
    ok["bls_period_match"] = [h is not None for h in harm]
    ok["bls_harmonic"] = harm
    ok["bls_strict_recovered"] = strict

    # ---- structurally out-of-range targets -------------------------------
    ok["bls_out_of_range"] = ok["archive_period_d"] > BLS_MAX_PERIOD_CAP
    in_range = ok[~ok["bls_out_of_range"]]
    n_oor = int(ok["bls_out_of_range"].sum())
    n_in_range = len(in_range)

    # ---- overall ---------------------------------------------------------
    bls_epoch = ok["bls_in_transit"].sum()
    bls_strict = ok["bls_strict_recovered"].sum()
    bls_permatch = ok["bls_period_match"].sum()
    exo_any = ok["exoveil_any_in_transit"].sum()
    exo_chance_exp = ok["p_exoveil_chance"].sum()
    bls_chance_exp = ok["p_single"].sum()

    # same, restricted to targets BLS could in principle find
    bls_epoch_ir = in_range["bls_in_transit"].sum()
    bls_strict_ir = in_range["bls_strict_recovered"].sum()
    bls_permatch_ir = in_range["bls_period_match"].sum()
    exo_any_ir = in_range["exoveil_any_in_transit"].sum()
    exo_chance_ir = in_range["p_exoveil_chance"].sum()

    lines = []
    lines.append("# ExoVeil vs. BLS/trapezoid -- depth-stratified recovery benchmark\n")
    lines.append(
        f"Sample: **{n_sample}** confirmed TESS-discovered planets drawn from the NASA "
        "Exoplanet Archive (`pscomppars`), stratified across transit-depth deciles "
        "(`select_benchmark_sample.py`, seed 42; sample frozen in "
        "`results/benchmark_sample.csv` before any run). Both pipelines see the "
        "**identical** Savitzky-Golay-detrended flux array per target "
        "(`prototype.py`'s `download_lc()`/`detrend()`), and both are scored against "
        "the archive's own `pl_orbper`/`pl_tranmid`/`pl_trandur` as ground truth -- "
        "not against each other.\n"
    )
    lines.append(
        f"**Completion**: {n_ok}/{n_sample} targets processed successfully "
        f"(status `ok`), {n_failed} failed/skipped"
        + (f", {len(missing)} never attempted" if missing else "")
        + ".\n"
    )

    # ---- API-completeness caveat, stated FIRST --------------------------
    lines.append("## Caveat 1: what is actually being benchmarked is not ExoVeil's full published method\n")
    lines.append(
        "The shipped PyPI package `exoveil==0.2.1` exposes, through "
        "`detect_from_array()`, only the **first two stages** of the pipeline "
        "described in Priyanshu (2026): the Transformer world model and the "
        "variance-weighted matched-filter event detector. Its return value is a raw "
        "list of up to 20 threshold-crossing events (`time`, `snr`, `depth_ppm`, "
        "`duration_pts`, `near_gap`, `aleatoric`, `uncertainty_category`).\n"
    )
    lines.append(
        "The **XGBoost planet-vs-false-positive classifier** (reported at AUC 0.938 on "
        "Kepler DR25) and the **conformal-prediction calibration stage** -- the "
        "components whose entire function is to filter and rank that raw candidate "
        "list down to calibrated detections -- are **not reachable through the public "
        "API at this version**. We inspected the installed package "
        "(`exoveil.core`, `exoveil.detect`) and found no exposed entry point for "
        "either stage.\n"
    )
    lines.append(
        "**This bounds what the results below can claim.** They characterise the "
        "*publicly installable artifact* as it ships, evaluated end-to-end on real "
        "TESS photometry. They do **not** measure the performance of ExoVeil's full "
        "method as published, because the stage specifically designed to suppress the "
        "false positives dominating the raw event list cannot be run. A chance-level "
        "raw-candidate rate is exactly what an unfiltered matched-filter front-end "
        "*should* produce before classification -- the classifier is the missing "
        "half of the method. The correct reading is therefore a **reproducibility gap "
        "between the paper and the shipped package**, not evidence that the published "
        "method performs at chance. Any paper text drawing on this benchmark must "
        "state that distinction explicitly; claiming the latter from these data would "
        "misrepresent the prior work.\n"
    )

    # ---- the chance-floor caveat -----------------------------------------
    lines.append("## Caveat 2: the two 'recovery' metrics are not directly comparable\n")
    lines.append(
        "ExoVeil returns up to 20 independently-scored candidate events per target and "
        "exposes no period. Scoring it as \"recovered if **any** event lands in-transit\" "
        "gives it 20 chances per target, whereas BLS gets one epoch. The probability a "
        "randomly-placed event falls in-transit is ~ T14/P; over N events the chance of "
        "at least one coincidental hit is 1-(1-T14/P)^N.\n"
    )
    lines.append(
        f"Summed over this sample, the **expected number of targets ExoVeil would "
        f"\"recover\" by pure chance is {exo_chance_exp:.1f}/{n_ok} "
        f"({100*exo_chance_exp/n_ok:.0f}%)**, versus {bls_chance_exp:.1f}/{n_ok} "
        f"({100*bls_chance_exp/n_ok:.0f}%) for BLS's single-epoch test. Any ExoVeil "
        "rate must be read against that floor.\n"
    )

    # ---- out-of-range caveat --------------------------------------------
    lines.append("## Caveat 3: two targets are outside BLS's configured search range\n")
    oor = ok[ok["bls_out_of_range"]]
    lines.append(
        f"`prototype.py`'s `identify()` searches periods up to "
        f"`max_p = min(span/2, {BLS_MAX_PERIOD_CAP:.0f})` days. "
        f"**{n_oor} of {n_ok}** benchmark targets have an archive period above that "
        "ceiling and are therefore *structurally* unrecoverable by BLS as configured "
        "-- no detrending or SNR improvement could find them, because the true period "
        "is never trialled:\n"
    )
    for _, r in oor.iterrows():
        lines.append(
            f"- **{r['target']}** -- archive P = {r['archive_period_d']:.2f} d "
            f"(> {BLS_MAX_PERIOD_CAP:.0f} d cap), decile {int(r['depth_decile'])}"
        )
    lines.append(
        "\nBLS rates are reported below **both** including these (the honest "
        "as-configured number) and excluding them (the fair per-target detection "
        "number). ExoVeil has no period search and so no equivalent ceiling -- its "
        "rate is unaffected in kind, but the excluded-subset column is shown for "
        "like-for-like comparison on the same targets.\n"
    )
    lines.append(
        "**Important**: the follow-up experiment below shows the 30 d cap is *not* "
        "the actual explanation for either miss -- raising it recovers neither "
        "target, for two different underlying reasons. Excluding them from the "
        "denominator remains justified (neither is recoverable from the available "
        "data), but the paper should not describe this as a mere tuning oversight.\n"
    )

    # ---- headline table --------------------------------------------------
    lines.append("## Overall recovery rates\n")
    lines.append(
        f"`n={n_ok}` = all processed targets; `n={n_in_range}` excludes the "
        f"{n_oor} out-of-range targets named above.\n"
    )
    lines.append("| Pipeline | Metric | Recovered (all, n=%d) | Rate | Recovered (in-range, n=%d) | Rate | Chance exp. (in-range) |"
                 % (n_ok, n_in_range))
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(
        f"| BLS/trapezoid | epoch falls in-transit | {bls_epoch}/{n_ok} | "
        f"{100*bls_epoch/n_ok:.0f}% | {bls_epoch_ir}/{n_in_range} | "
        f"**{100*bls_epoch_ir/n_in_range:.0f}%** | "
        f"{100*in_range['p_single'].sum()/n_in_range:.0f}% |"
    )
    lines.append(
        f"| BLS/trapezoid | **strict**: epoch in-transit AND period matches archive "
        f"(±{PERIOD_TOL*100:.0f}%, incl. 2x/3x/0.5x harmonics) | {bls_strict}/{n_ok} | "
        f"{100*bls_strict/n_ok:.0f}% | {bls_strict_ir}/{n_in_range} | "
        f"**{100*bls_strict_ir/n_in_range:.0f}%** | -- |"
    )
    lines.append(
        f"| BLS/trapezoid | period matches archive (regardless of epoch) | "
        f"{bls_permatch}/{n_ok} | {100*bls_permatch/n_ok:.0f}% | "
        f"{bls_permatch_ir}/{n_in_range} | {100*bls_permatch_ir/n_in_range:.0f}% | -- |"
    )
    lines.append(
        f"| ExoVeil (raw events; see Caveat 1) | >=1 of <=20 events in-transit | "
        f"{exo_any}/{n_ok} | {100*exo_any/n_ok:.0f}% | {exo_any_ir}/{n_in_range} | "
        f"**{100*exo_any_ir/n_in_range:.0f}%** | {100*exo_chance_ir/n_in_range:.0f}% |"
    )
    lines.append("")
    lines.append(
        f"Chance floors for the full sample: ExoVeil {100*exo_chance_exp/n_ok:.0f}%, "
        f"BLS single-epoch {100*bls_chance_exp/n_ok:.0f}%. On the in-range subset: "
        f"ExoVeil {100*exo_chance_ir/n_in_range:.0f}% vs. its actual "
        f"{100*exo_any_ir/n_in_range:.0f}% "
        f"(**{100*(exo_any_ir-exo_chance_ir)/n_in_range:+.0f} pp** excess); BLS "
        f"{100*in_range['p_single'].sum()/n_in_range:.0f}% vs. its actual "
        f"{100*bls_epoch_ir/n_in_range:.0f}% "
        f"(**{100*(bls_epoch_ir-in_range['p_single'].sum())/n_in_range:+.0f} pp** excess).\n"
    )

    # ---- by decile -------------------------------------------------------
    lines.append("## Breakdown by transit-depth decile\n")
    lines.append(
        "Decile 0 = shallowest (hardest), decile 9 = deepest (easiest). "
        "`n` is targets processed in that decile.\n"
    )
    lines.append(
        "| Decile | Depth range (ppm) | n | BLS epoch-in-transit | BLS strict | "
        "ExoVeil >=1 in-transit | ExoVeil chance exp. | ExoVeil excess |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in sorted(ok["depth_decile"].dropna().unique()):
        sub = ok[ok["depth_decile"] == d]
        n = len(sub)
        lo, hi = sub["archive_depth_ppm"].min(), sub["archive_depth_ppm"].max()
        b_e = sub["bls_in_transit"].sum()
        b_s = sub["bls_strict_recovered"].sum()
        e_a = sub["exoveil_any_in_transit"].sum()
        e_c = sub["p_exoveil_chance"].sum()
        lines.append(
            f"| {int(d)} | {lo:.0f}-{hi:.0f} | {n} | {b_e}/{n} ({100*b_e/n:.0f}%) | "
            f"{b_s}/{n} ({100*b_s/n:.0f}%) | {e_a}/{n} ({100*e_a/n:.0f}%) | "
            f"{e_c:.1f}/{n} ({100*e_c/n:.0f}%) | {100*(e_a-e_c)/n:+.0f} pp |"
        )
    lines.append("")

    # ---- shallow vs deep aggregate --------------------------------------
    shallow = ok[ok["depth_decile"] <= 4]
    deep = ok[ok["depth_decile"] >= 5]
    lines.append("## Shallow (deciles 0-4) vs. deep (deciles 5-9)\n")
    lines.append("| Group | n | BLS epoch | BLS strict | ExoVeil >=1 | ExoVeil chance exp. |")
    lines.append("|---|---|---|---|---|---|")
    for label, grp in [("Shallow (0-4)", shallow), ("Deep (5-9)", deep)]:
        n = len(grp)
        if n == 0:
            continue
        lines.append(
            f"| {label} | {n} | {grp['bls_in_transit'].sum()}/{n} "
            f"({100*grp['bls_in_transit'].sum()/n:.0f}%) | "
            f"{grp['bls_strict_recovered'].sum()}/{n} "
            f"({100*grp['bls_strict_recovered'].sum()/n:.0f}%) | "
            f"{grp['exoveil_any_in_transit'].sum()}/{n} "
            f"({100*grp['exoveil_any_in_transit'].sum()/n:.0f}%) | "
            f"{grp['p_exoveil_chance'].sum():.1f}/{n} "
            f"({100*grp['p_exoveil_chance'].sum()/n:.0f}%) |"
        )
    lines.append("")

    # ---- long-period recheck --------------------------------------------
    if os.path.exists(RECHECK_CSV):
        rc = pd.read_csv(RECHECK_CSV)
        rc["in_transit"] = rc["in_transit"].astype(str).str.lower().eq("true")
        rc["archive_period_reachable"] = rc["archive_period_reachable"].astype(str).str.lower().eq("true")
        lines.append("## Follow-up: does raising the BLS period cap recover the two out-of-range targets?\n")
        lines.append(
            "`recheck_long_period.py` reruns the BLS stage on just these two targets "
            "with the period ceiling raised from 30 d to 150 d (and a denser period "
            "grid, so grid resolution is not a confound), in three configurations: "
            "as-benchmarked (4 sectors, 30 d cap); cap raised; and cap raised with up "
            "to 20 sectors requested. Everything else is held identical.\n"
        )
        lines.append("| Target | Config | Sectors used | Baseline span (d) | Effective max_p (d) | Binding constraint | Archive P reachable? | BLS P (d) | In-transit? |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in rc.iterrows():
            lines.append(
                f"| {r['target']} | {r['config']} | {r['n_sectors_used']} | "
                f"{r['span_d']:.1f} | {r['max_p_d']:.2f} | `{r['binding_constraint']}` | "
                f"{'YES' if r['archive_period_reachable'] else 'NO'} | "
                f"{r['bls_period_d']:.4f} | {'YES' if r['in_transit'] else 'no'} |"
            )
        lines.append("")
        lines.append("### Verdict: the two targets fail for *different* reasons\n")
        lines.append(
            "- **TOI-2449 b (P = 106.14 d)** -- the 30 d ceiling *was* the binding "
            "constraint (baseline span is 2635 d, so `span/2` = 1317 d never bound). "
            "Raising the ceiling to 150 d put the true period inside the search grid. "
            "**BLS still did not recover it**: it returned 0.5150 d, a spurious "
            "short-period peak, still out-of-transit. Only 3 sectors exist for this "
            "target, scattered across a 2635 d baseline, so very few of its 106 d-period "
            "transits fall inside an observed window. **The cap is not the explanation "
            "-- this is a genuine sampling limitation.**\n"
        )
        lines.append(
            "- **NGTS-20 b (P = 54.19 d)** -- the 30 d ceiling was **not** the binding "
            "constraint at all; `span/2` was. Only 2 sectors exist, giving a 25.43 d "
            "baseline, so `max_p = 12.71 d` regardless of the ceiling setting. Raising "
            "the cap changed nothing (12.71 d either way) and the true period was never "
            "reachable in any configuration. Recovering a 54 d period requires "
            "`span >= 2P = 108 d` of data; **only 25 d exists**, and requesting 20 "
            "sectors still returned 2 because that is all MAST has. This is a hard "
            "data-availability limit, not a tuning choice -- no configuration of this "
            "pipeline could ever recover it from the available photometry.\n"
        )
        lines.append(
            "Neither target is rescued by widening the search, so excluding them from "
            "the BLS denominator is justified on structural grounds -- but note the "
            "justification differs: TOI-2449 b is *searchable but under-sampled*, "
            "NGTS-20 b is *not searchable at all* with the existing baseline.\n"
        )

    # ---- SNR blow-ups ----------------------------------------------------
    n_blowup = ok["exoveil_snr_flag"].astype(str).str.contains("undefined").sum()
    lines.append("## ExoVeil SNR numerical blow-ups\n")
    lines.append(
        f"**{n_blowup}/{n_ok}** targets produced a top-event SNR above the sanity "
        "ceiling (1000), i.e. a degenerate value from the local-MAD noise estimator "
        "collapsing toward zero (`exoveil.detect.detect_events()` weights residuals "
        "by 1/local_MAD). This reproduces at scale the failure mode first seen on "
        "V1828 Aql in the 4-target run -- it is **not** confined to eclipsing "
        "binaries. Affected targets:\n"
    )
    for _, r in ok[ok["exoveil_snr_flag"].astype(str).str.contains("undefined")].iterrows():
        lines.append(f"- {r['target']} (decile {int(r['depth_decile'])}, "
                     f"raw SNR {r['exoveil_top_snr_raw']})")
    lines.append("")

    # ---- per-target appendix --------------------------------------------
    lines.append("## Per-target detail\n")
    lines.append("| Target | Decile | Archive P (d) | Archive depth (ppm) | BLS P (d) | "
                 "BLS period match | BLS epoch in-transit | ExoVeil #events | "
                 "ExoVeil in-transit | ExoVeil chance | Note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ok.sort_values(["depth_decile", "target"]).iterrows():
        # NB: None stored in a pandas column comes back as NaN, so test with
        # pd.isna() rather than `is None` (which would render every non-match
        # as "yes (nanx)").
        h = r["bls_harmonic"]
        hm = "no" if pd.isna(h) else (f"yes ({h:g}x)" if h != 1.0 else "yes")
        lines.append(
            f"| {r['target']} | {int(r['depth_decile'])} | {r['archive_period_d']:.4f} | "
            f"{r['archive_depth_ppm']:.0f} | "
            f"{r['bls_period_d']:.4f} | {hm} | "
            f"{'YES' if r['bls_in_transit'] else 'no'} | "
            f"{int(r['exoveil_n_events']) if np.isfinite(r['exoveil_n_events']) else '-'} | "
            f"{int(r['exoveil_n_events_in_transit']) if np.isfinite(r['exoveil_n_events_in_transit']) else '-'} | "
            f"{100*r['p_exoveil_chance']:.0f}% | "
            f"{'**outside configured BLS search range (P > %.0f d)**' % BLS_MAX_PERIOD_CAP if r['bls_out_of_range'] else ''} |"
        )
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")

    # console echo of headline numbers
    print(f"\nProcessed OK: {n_ok}/{n_sample}  (failed/skipped: {n_failed})")
    print(f"Out-of-range for BLS (P > {BLS_MAX_PERIOD_CAP:.0f} d): {n_oor}  "
          f"-> in-range subset n={n_in_range}")
    print(f"                        {'ALL':>12}   {'IN-RANGE':>12}")
    print(f"BLS  epoch-in-transit : {bls_epoch}/{n_ok} ({100*bls_epoch/n_ok:3.0f}%)   "
          f"{bls_epoch_ir}/{n_in_range} ({100*bls_epoch_ir/n_in_range:3.0f}%)   "
          f"[chance {100*in_range['p_single'].sum()/n_in_range:.0f}%]")
    print(f"BLS  strict (P match) : {bls_strict}/{n_ok} ({100*bls_strict/n_ok:3.0f}%)   "
          f"{bls_strict_ir}/{n_in_range} ({100*bls_strict_ir/n_in_range:3.0f}%)")
    print(f"ExoVeil >=1 in-transit: {exo_any}/{n_ok} ({100*exo_any/n_ok:3.0f}%)   "
          f"{exo_any_ir}/{n_in_range} ({100*exo_any_ir/n_in_range:3.0f}%)   "
          f"[chance {100*exo_chance_ir/n_in_range:.0f}%]")
    print(f"ExoVeil SNR blow-ups  : {n_blowup}/{n_ok}")


if __name__ == "__main__":
    main()
