#!/usr/bin/env python3
"""
Apply Module 3 (Mandel-Agol fit) and Module 5 (bootstrap FAP + conformal) to
the 4 core validation targets and to the 25-target benchmark sample.

Module 3's PRIMARY reported uncertainty is LSQ point estimate + parametric
(residual) bootstrap (bootstrap_lsq_* columns) -- NOT the MCMC posterior.
See results/module3_convergence_diagnosis.md for why: four independent
convergence attempts (emcee under 3 parametrizations, then dynesty nested
sampling) all failed, in ways that point to a genuine transit-geometry
degeneracy in the data rather than a fixable sampler/parametrization issue.
MCMC is still run and reported (mcmc_* / period_med / depth_ppm_med / etc.
columns) as a secondary, explicitly-caveated characterization -- see each
row's posterior_diagnostic_note.

    python run_modules_3_5.py --mode core        # TOI-700, TOI-1338, Pi Men, V1828 Aql
    python run_modules_3_5.py --mode benchmark   # results/benchmark_sample.csv
    python run_modules_3_5.py --mode both
    python run_modules_3_5.py --mode core --retry-failed
    python run_modules_3_5.py --mode benchmark --limit 3

Checkpointed exactly like run_benchmark.py: each target's row is appended to
results/modules_3_5_results.csv as soon as it finishes, and targets already
present are skipped on rerun, so a long batch survives interruption.

Ephemeris source per mode:
  core      -- results/bls_ephemeris.json (the pipeline's OWN BLS solution), so
               the MCMC is validating/refining the same solution the trapezoid
               fit in results/summary_table.csv was built on. Comparing against
               an external ephemeris would confound "is the MCMC consistent
               with the trapezoid" with "was BLS right in the first place".
  benchmark -- the NASA Exoplanet Archive ephemeris in
               results/benchmark_sample.csv (ground truth), since the point
               there is statistical power on real planets, not agreement with
               a BLS solution that the earlier benchmark showed is often
               aliased.

prototype.py is imported, never modified.
"""

import os
import sys
import csv
import json
import argparse
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import TARGETS, download_lc, detrend, RESULTS  # noqa: E402
import modules_3_5 as M  # noqa: E402

OUT_CSV = os.path.join(RESULTS, "modules_3_5_results.csv")
EPHEM_JSON = os.path.join(RESULTS, "bls_ephemeris.json")
SAMPLE_CSV = os.path.join(RESULTS, "benchmark_sample.csv")

NWALKERS = 32
# Chains on sparse targets are strongly autocorrelated (tau ~ 200 when only a
# few hundred in-transit cadences exist), so emcee's strict 50*tau criterion
# needs long runs. 8000 steps clears it for most targets; the per-target
# `mcmc_converged` flag records honestly where it does not, rather than
# silently reporting an under-sampled posterior as if it were converged.
NSTEPS = 8000
MAX_POINTS = 4000
N_BOOT = 2000
LSQ_BOOT_N = 300  # parameter bootstrap (Module 3 PRIMARY) -- see bootstrap_lsq() docstring

# LD prior box in fit_least_squares() is [0.01, 0.99] for q1/q2 (free-LD, 8-param
# mode -- the mode this batch runner always uses). If the LSQ fit pins q1 or q2
# within this tolerance of either edge, the optimizer is very likely exploiting
# limb-darkening shape freedom at its boundary to force-fit a light curve the
# quadratic Mandel-Agol model doesn't actually describe (confirmed on V1828 Aql:
# q1=0.9668/q2=0.9894 -> unphysical u1=1.94/u2=-0.96, and the boundary optimum
# was numerically "sticky" enough that bootstrap_lsq()'s resamples all collapsed
# to the same point, understating its own uncertainty by orders of magnitude --
# see results/module3_convergence_diagnosis.md addendum). Flagging this
# automatically catches the same failure mode on any benchmark target without
# requiring a manual per-target inspection.
LD_BOUNDARY_LO, LD_BOUNDARY_HI = 0.01, 0.99
LD_BOUNDARY_TOL = 0.01

POSTERIOR_DIAGNOSTIC_NOTE = (
    "MCMC posterior below is SECONDARY/exploratory, NOT the primary uncertainty. "
    "4 convergence attempts (emcee x3 parametrizations, dynesty nested sampling) "
    "all failed to converge -- points to a real transit-geometry degeneracy "
    "(a_over_rs/b, then rp_over_rs/b under dynesty), not a fixable sampler issue. "
    "See results/module3_convergence_diagnosis.md. Primary uncertainty is "
    "lsq_boot_* (parametric residual bootstrap on the LSQ fit)."
)

COLS = [
    "target", "mode", "status", "error_message",
    "ephem_source", "ephem_period_d", "ephem_t0_btjd", "ephem_duration_hr",
    "n_fit_points",
    # Module 3 -- least squares (PRIMARY point estimate)
    "lsq_ok", "lsq_depth_ppm", "lsq_duration_hr", "lsq_impact_b", "lsq_rp_over_rs",
    "lsq_q1", "lsq_q2", "boundary_pinned_ld", "boundary_pinned_ld_note",
    # Module 3 -- PRIMARY uncertainty: parametric (residual) bootstrap on the LSQ fit.
    # block_* is the honest default (correlated TESS residuals); iid_* for comparison.
    "lsq_boot_n", "lsq_boot_n_ok_iid", "lsq_boot_n_ok_block", "lsq_boot_block_len",
    "lsq_boot_depth_ppm_block_med", "lsq_boot_depth_ppm_block_lo", "lsq_boot_depth_ppm_block_hi",
    "lsq_boot_duration_hr_block_med", "lsq_boot_duration_hr_block_lo", "lsq_boot_duration_hr_block_hi",
    "lsq_boot_rp_over_rs_block_med", "lsq_boot_rp_over_rs_block_lo", "lsq_boot_rp_over_rs_block_hi",
    "lsq_boot_impact_b_block_med", "lsq_boot_impact_b_block_lo", "lsq_boot_impact_b_block_hi",
    "lsq_boot_depth_ppm_iid_med", "lsq_boot_depth_ppm_iid_lo", "lsq_boot_depth_ppm_iid_hi",
    "lsq_boot_duration_hr_iid_med", "lsq_boot_duration_hr_iid_lo", "lsq_boot_duration_hr_iid_hi",
    "lsq_boot_rp_over_rs_iid_med", "lsq_boot_rp_over_rs_iid_lo", "lsq_boot_rp_over_rs_iid_hi",
    "lsq_boot_impact_b_iid_med", "lsq_boot_impact_b_iid_lo", "lsq_boot_impact_b_iid_hi",
    "posterior_diagnostic_note",
    # Module 3 -- MCMC posterior (SECONDARY/exploratory -- see posterior_diagnostic_note)
    "mcmc_converged", "mcmc_acceptance", "mcmc_tau_max", "mcmc_n_samples",
    "period_med", "period_lo", "period_hi",
    "t0_med", "t0_lo", "t0_hi",
    "rp_over_rs_med", "rp_over_rs_lo", "rp_over_rs_hi",
    "a_over_rs_med", "a_over_rs_lo", "a_over_rs_hi",
    "inc_deg_med", "inc_deg_lo", "inc_deg_hi",
    "impact_b_med", "impact_b_lo", "impact_b_hi",
    "depth_ppm_med", "depth_ppm_lo", "depth_ppm_hi",
    "duration_hr_med", "duration_hr_lo", "duration_hr_hi",
    "u1_med", "u1_lo", "u1_hi", "u2_med", "u2_lo", "u2_hi",
    # Module 5
    "snr_matched_filter", "fap_iid", "fap_block",
    "fap_iid_gpd", "fap_block_gpd", "fap_iid_gpd_log10", "fap_block_gpd_log10",
    "gpd_iid_validation_maxdiff", "gpd_block_validation_maxdiff",
    "null_iid_p99", "null_block_p99", "block_len_cadences", "n_boot",
    "n_epochs", "conformal_depth_ppm", "conformal_half_width_ppm",
    "conformal_lo_ppm", "conformal_hi_ppm", "conformal_note",
]


def blank(target, mode, status, err="", **kw):
    r = {c: "" for c in COLS}
    r.update({"target": target, "mode": mode, "status": status, "error_message": err})
    r.update(kw)
    return r


def append_row(row):
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def done_set(retry_failed):
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    if retry_failed:
        return set(df.loc[df["status"] == "ok", "target"])
    return set(df["target"])


def process(target, mode, terms, ephem_source, period, t0_btjd, dur_hr, depth_guess_ppm):
    print(f"\n{'=' * 66}\n  {target}   [{mode}]   P={period:.5f} d  T14={dur_hr:.3f} h"
          f"\n{'=' * 66}", flush=True)
    dur_d = dur_hr / 24.0

    try:
        lc = download_lc(target if mode == "core" else terms[0], terms)
    except Exception as exc:
        traceback.print_exc()
        return blank(target, mode, "download_error", f"{type(exc).__name__}: {exc}")
    if lc is None:
        print("  *** no TESS data")
        return blank(target, mode, "no_data")

    try:
        lc_c = detrend(lc, f"{target}_m35")
    except Exception as exc:
        traceback.print_exc()
        return blank(target, mode, "detrend_error", f"{type(exc).__name__}: {exc}")

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

    tt, ff, ee, _ = M.select_transit_window(t, f, ferr, period, t0_btjd, dur_d,
                                            max_points=MAX_POINTS)
    print(f"  fitting {len(tt)} cadences (of {len(t)})", flush=True)

    row = blank(target, mode, "ok",
                ephem_source=ephem_source, ephem_period_d=period,
                ephem_t0_btjd=t0_btjd, ephem_duration_hr=dur_hr,
                n_fit_points=len(tt))

    # ---- Module 3: least squares (PRIMARY point estimate) --------------
    try:
        th_ls, lsq_ok = M.fit_least_squares(tt, ff, ee, period, t0_btjd, dur_d, depth_guess_ppm)
        # derived_quantities returns (depth_ppm, T14_hr, inclination_deg, a_over_rs);
        # b is a SAMPLED parameter, so read it straight off theta[4].
        d_ls, du_ls, inc_ls, a_rs_ls = M.derived_quantities(th_ls)
        b_ls = float(th_ls[4])
        rp_ls = float(th_ls[2])
        q1_ls, q2_ls = float(th_ls[5]), float(th_ls[6])
        row.update({"lsq_ok": lsq_ok, "lsq_depth_ppm": f"{d_ls:.2f}",
                    "lsq_duration_hr": f"{du_ls:.4f}", "lsq_impact_b": f"{b_ls:.4f}",
                    "lsq_rp_over_rs": f"{rp_ls:.5f}",
                    "lsq_q1": f"{q1_ls:.5f}", "lsq_q2": f"{q2_ls:.5f}"})
        print(f"  LSQ (PRIMARY point est.): depth={d_ls:.1f} ppm  T14={du_ls:.3f} h  "
              f"b={b_ls:.3f} (inc={inc_ls:.3f} deg)", flush=True)

        # boundary-pinning check (generalizes the V1828 Aql diagnosis -- see
        # LD_BOUNDARY_TOL comment above): flag automatically instead of relying
        # on manual per-target inspection to catch this failure mode.
        pinned_q1 = (q1_ls <= LD_BOUNDARY_LO + LD_BOUNDARY_TOL) or (q1_ls >= LD_BOUNDARY_HI - LD_BOUNDARY_TOL)
        pinned_q2 = (q2_ls <= LD_BOUNDARY_LO + LD_BOUNDARY_TOL) or (q2_ls >= LD_BOUNDARY_HI - LD_BOUNDARY_TOL)
        pinned = pinned_q1 or pinned_q2
        row["boundary_pinned_ld"] = pinned
        if pinned:
            which = ", ".join([n for n, p in (("q1", pinned_q1), ("q2", pinned_q2)) if p])
            u1_ls, u2_ls = M.q_to_u(q1_ls, q2_ls)
            note = (f"{which} pinned within {LD_BOUNDARY_TOL} of a prior bound "
                    f"(q1={q1_ls:.4f}, q2={q2_ls:.4f} -> u1={u1_ls:.3f}, u2={u2_ls:.3f}). "
                    "LSQ/bootstrap likely fitting an unphysical LD shape to compensate "
                    "for a model mismatch (e.g. grazing/blended/EB geometry); the "
                    "lsq_boot_* interval for this target may be an artificially tight "
                    "numerical artifact, not genuine precision -- see "
                    "results/module3_convergence_diagnosis.md.")
            row["boundary_pinned_ld_note"] = note
            print(f"  *** BOUNDARY-PINNED LD: {note}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row["status"] = "lsq_error"; row["error_message"] = str(exc)
        return row

    # ---- Module 3: PRIMARY uncertainty -- parametric bootstrap on the LSQ fit ----
    row["posterior_diagnostic_note"] = POSTERIOR_DIAGNOSTIC_NOTE
    try:
        boot = M.bootstrap_lsq(tt, ff, ee, th_ls, period, t0_btjd, dur_d, depth_guess_ppm,
                               n_boot=LSQ_BOOT_N)
        row.update({
            "lsq_boot_n": boot.get("n_boot", ""),
            "lsq_boot_n_ok_iid": boot.get("n_ok_iid", ""),
            "lsq_boot_n_ok_block": boot.get("n_ok_block", ""),
            "lsq_boot_block_len": boot.get("block_len_cadences", ""),
        })
        for qty in ("depth_ppm", "duration_hr", "rp_over_rs", "impact_b"):
            for scheme in ("block", "iid"):
                for stat in ("med", "lo", "hi"):
                    k = f"{qty}_{scheme}_{stat}"
                    col = f"lsq_boot_{k}"
                    v = boot.get(k, np.nan)
                    row[col] = f"{v:.5f}" if np.isfinite(v) else ""
        print(f"  LSQ bootstrap (PRIMARY uncertainty, block, n={boot.get('n_ok_block','?')}): "
              f"depth={boot.get('depth_ppm_block_med', float('nan')):.1f} "
              f"(-{boot.get('depth_ppm_block_lo', float('nan')):.1f}/"
              f"+{boot.get('depth_ppm_block_hi', float('nan')):.1f}) ppm", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row["error_message"] = (row["error_message"] + f" | lsq_boot: {exc}").strip(" |")

    # ---- Module 3: MCMC posterior (SECONDARY/exploratory) --------------
    try:
        chain, diag = M.run_mcmc(tt, ff, ee, th_ls, period, t0_btjd, dur_d,
                                 nwalkers=NWALKERS, nsteps=NSTEPS)
    except Exception as exc:
        traceback.print_exc()
        row["status"] = "mcmc_error"; row["error_message"] = str(exc)
        return row
    if chain is None:
        row["status"] = "mcmc_init_failed"; row["error_message"] = diag.get("reason", "")
        return row

    row.update({"mcmc_converged": diag["converged"],
                "mcmc_acceptance": f"{diag['acceptance']:.4f}",
                "mcmc_tau_max": f"{diag['tau_max']:.2f}" if np.isfinite(diag["tau_max"]) else "",
                "mcmc_n_samples": diag["n_samples"]})
    post = M.summarize_posterior(chain)
    for k, v in post.items():
        if k in row:
            row[k] = f"{v:.6f}" if np.isfinite(v) else ""
    print(f"  MCMC [secondary/exploratory]: depth={post['depth_ppm_med']:.1f} "
          f"(-{post['depth_ppm_lo']:.1f}/+{post['depth_ppm_hi']:.1f}) ppm | "
          f"b={post['impact_b_med']:.3f} | converged={diag['converged']} "
          f"(acc={diag['acceptance']:.2f}, tau={diag['tau_max']:.0f})", flush=True)

    # ---- Module 5: FAP + conformal ------------------------------------
    th_best = np.median(chain, axis=0)
    try:
        fap = M.bootstrap_fap(tt, ff, th_best, dur_d, n_boot=N_BOOT)
        row.update({
            "snr_matched_filter": f"{fap['snr_obs']:.4f}" if np.isfinite(fap["snr_obs"]) else "",
            "fap_iid": f"{fap['fap_iid']:.6f}" if np.isfinite(fap["fap_iid"]) else "",
            "fap_block": f"{fap['fap_block']:.6f}" if np.isfinite(fap["fap_block"]) else "",
            "fap_iid_gpd": f"{fap['fap_iid_gpd']:.6g}" if np.isfinite(fap.get("fap_iid_gpd", np.nan)) else "",
            "fap_block_gpd": f"{fap['fap_block_gpd']:.6g}" if np.isfinite(fap.get("fap_block_gpd", np.nan)) else "",
            "fap_iid_gpd_log10": f"{fap['fap_iid_gpd_log10']:.3f}" if np.isfinite(fap.get("fap_iid_gpd_log10", np.nan)) else "",
            "fap_block_gpd_log10": f"{fap['fap_block_gpd_log10']:.3f}" if np.isfinite(fap.get("fap_block_gpd_log10", np.nan)) else "",
            "gpd_iid_validation_maxdiff": f"{fap['gpd_iid_validation_maxdiff']:.5f}" if np.isfinite(fap.get("gpd_iid_validation_maxdiff", np.nan)) else "",
            "gpd_block_validation_maxdiff": f"{fap['gpd_block_validation_maxdiff']:.5f}" if np.isfinite(fap.get("gpd_block_validation_maxdiff", np.nan)) else "",
            "null_iid_p99": f"{fap.get('null_iid_p99', np.nan):.4f}",
            "null_block_p99": f"{fap.get('null_block_p99', np.nan):.4f}",
            "block_len_cadences": fap.get("block_len_cadences", ""),
            "n_boot": fap.get("n_boot", ""),
        })
        print(f"  FAP: snr={fap['snr_obs']:.2f}  iid={fap['fap_iid']:.4g}  "
              f"block={fap['fap_block']:.4g}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row["error_message"] = (row["error_message"] + f" | FAP: {exc}").strip(" |")

    try:
        depths, _ = M.per_epoch_depths(tt, ff, th_best, th_best[0], th_best[1])
        cf = M.conformal_interval(depths, alpha=0.1)
        row.update({
            "n_epochs": cf["n_epochs"],
            "conformal_depth_ppm": f"{cf['conformal_point']:.2f}" if np.isfinite(cf["conformal_point"]) else "",
            "conformal_half_width_ppm": f"{cf['conformal_half_width']:.2f}" if np.isfinite(cf["conformal_half_width"]) else "",
            "conformal_lo_ppm": f"{cf['conformal_lo']:.2f}" if np.isfinite(cf["conformal_lo"]) else "",
            "conformal_hi_ppm": f"{cf['conformal_hi']:.2f}" if np.isfinite(cf["conformal_hi"]) else "",
            "conformal_note": cf.get("note", ""),
        })
        print(f"  Conformal (90%): {cf['conformal_point']:.1f} +/- "
              f"{cf['conformal_half_width']:.1f} ppm over {cf['n_epochs']} epochs", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row["error_message"] = (row["error_message"] + f" | conformal: {exc}").strip(" |")

    return row


def core_jobs():
    if not os.path.exists(EPHEM_JSON):
        print(f"ERROR: {EPHEM_JSON} missing -- run compute_bls_ephemeris.py first.")
        return []
    with open(EPHEM_JSON, encoding="utf-8") as fh:
        eph = json.load(fh)
    depths = {}
    st = os.path.join(RESULTS, "summary_table.csv")
    if os.path.exists(st):
        with open(st, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                depths[r["target"]] = float(r["depth_ppm"])
    jobs = []
    for tgt in TARGETS:
        n = tgt["name"]
        if n not in eph:
            continue
        e = eph[n]
        jobs.append((n, "core", tgt["terms"], "bls_ephemeris.json",
                     e["period"], e["t0"], e["duration"] * 24.0, depths.get(n, 1000.0)))
    return jobs


def benchmark_jobs():
    if not os.path.exists(SAMPLE_CSV):
        print(f"ERROR: {SAMPLE_CSV} missing -- run select_benchmark_sample.py first.")
        return []
    df = pd.read_csv(SAMPLE_CSV)
    jobs = []
    for _, r in df.iterrows():
        jobs.append((r["target"], "benchmark", [r["tic_id"], r.get("hostname")],
                     "NASA Exoplanet Archive", float(r["period_d"]),
                     float(r["t0_bjd"]) - 2457000.0, float(r["duration_hr"]),
                     float(r["depth_ppm"])))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["core", "benchmark", "both"], default="both")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    jobs = []
    if args.mode in ("core", "both"):
        jobs += core_jobs()
    if args.mode in ("benchmark", "both"):
        jobs += benchmark_jobs()

    done = done_set(args.retry_failed)
    pending = [j for j in jobs if j[0] not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Jobs: {len(jobs)} total, {len(done)} already done, {len(pending)} pending.",
          flush=True)
    print(f"LSQ+bootstrap (PRIMARY): n_boot={LSQ_BOOT_N} (iid+block). "
          f"MCMC (secondary): {NWALKERS} walkers x {NSTEPS} steps, <={MAX_POINTS} cadences. "
          f"FAP: {N_BOOT} bootstraps.", flush=True)
    if not pending:
        print("Nothing to do.")
        return

    for j in pending:
        try:
            row = process(*j)
        except Exception as exc:
            traceback.print_exc()
            row = blank(j[0], j[1], "unexpected_error", f"{type(exc).__name__}: {exc}")
        append_row(row)
        print(f"  -> appended to {OUT_CSV}", flush=True)

    print("\nBatch complete.")


if __name__ == "__main__":
    main()
