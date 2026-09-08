#!/usr/bin/env python3
"""
Module 2 -- evaluate the trained SSM world model on the SAME 29 targets
already used for the ExoVeil baseline comparison: the 4 core targets
(results/comparison_table.md: TOI-700, TOI-1338, Pi Mensae, V1828 Aql) plus
the 25-target confirmed-planet benchmark sample (results/benchmark_sample.csv).

Detection methodology: run the trained model at inference with NO masking
(masking is a training-time-only technique to force learning from context;
at inference a real, unmodeled transit shows up as a large residual because
the model was never trained to predict a transit-shaped dip specifically at
that time). Residual = observed - predicted; matched-filter this residual
against a box template at the archive/BLS ephemeris's known duration,
centered at the known t0 -- this is a SINGLE-hypothesis test at a known
ephemeris, structurally the same as the BLS/trapezoid baseline (NOT
ExoVeil's multi-candidate-event-list structure), so the correct chance
floor here is the BLS-style single-epoch p_single = (T14/24)/P, reusing
summarize_benchmark.py's own formula -- NOT ExoVeil's
1-(1-p_single)^N_events floor, which applies to a ranked multi-event list
this detector does not produce. Reports both raw recovery rate and the
chance-corrected excess, exactly per the standing instruction not to quote
a raw rate without the chance-floor correction.

Usage: python eval_ssm_world_model.py --backbone mamba
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import download_lc, TARGETS, RESULTS  # noqa: E402
from modules_2_ssm import SSMWorldModel, count_parameters  # noqa: E402
import modules_3_5 as M35  # noqa: E402


def load_targets():
    """(name, terms, period, t0_btjd, duration_hr, source) for all 29 eval targets."""
    jobs = []
    with open(os.path.join(RESULTS, "bls_ephemeris.json"), encoding="utf-8") as fh:
        eph = json.load(fh)
    for tgt in TARGETS:
        n = tgt["name"]
        if n in eph:
            e = eph[n]
            jobs.append((n, tgt["terms"], e["period"], e["t0"], e["duration"] * 24.0, "core"))

    sample = pd.read_csv(os.path.join(RESULTS, "benchmark_sample.csv"))
    for _, r in sample.iterrows():
        jobs.append((r["target"], [str(r["tic_id"]), r.get("hostname")],
                     float(r["period_d"]), float(r["t0_bjd"]) - 2457000.0,
                     float(r["duration_hr"]), "benchmark"))
    return jobs


def prep_light_curve(name, terms):
    """Same light preprocessing as the training corpus: normalize + outlier
    removal only, NO Savitzky-Golay flatten -- consistency with what the
    model was actually trained to predict."""
    lc = download_lc(name, terms)
    if lc is None:
        return None, None
    lc_norm = lc.normalize()
    lc_clean = lc_norm.remove_outliers(sigma_upper=5, sigma_lower=1e4).remove_nans()
    t = np.asarray(lc_clean.time.value, dtype=np.float64)
    f = np.asarray(lc_clean.flux.value, dtype=np.float32)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    return t, f


def select_window(t, f, t0, period, L, dur_d):
    """Center a fixed-length L window on a transit epoch that actually has
    real photometric coverage nearby, so the model has real context on both
    sides. Also returns the exact epoch time (center_t) this window was
    centered on, so callers can build a SINGLE-epoch detection template --
    the window may span many orbital periods for a short-period target, but
    only this one transit is the epoch actually being tested.

    An earlier version anchored on the nearest transit cycle to the raw
    calendar-time midpoint 0.5*(t[0]+t[-1]) of the full multi-sector
    baseline. For targets whose requested sectors are widely non-contiguous
    (e.g. 4 TESS sectors spread across a 3-year baseline with ~100 days of
    actual coverage), that midpoint can fall inside a many-months gap
    between sector groups, silently "centering" the window on a transit
    epoch that was never observed (verified for two benchmark targets:
    nearest real cadence 190-464 days from the predicted center, vs. a
    transit half-duration of minutes -- see
    results/module2_ssm_design.md). Fixed by only considering candidate
    transit cycles that have a REAL cadence within a small tolerance
    (a few hours, scaled to the transit duration) of their predicted
    center, then picking the one nearest the median of the ACTUAL observed
    timestamps (robust to sparse/clustered sectors, unlike the raw
    first/last endpoint mean). If no cycle has real nearby coverage, this
    correctly returns None (caller marks the target as skipped) rather than
    forcing a window that doesn't contain the transit it claims to."""
    if len(t) <= L:
        return None
    tol_days = max(3.0 * dur_d, 4.0 / 24.0)  # a few hours, scaled by T14
    n_min = int(np.ceil((t[0] - t0) / period))
    n_max = int(np.floor((t[-1] - t0) / period))
    if n_max < n_min:
        return None
    candidates = t0 + np.arange(n_min, n_max + 1) * period
    idx = np.clip(np.searchsorted(t, candidates), 1, len(t) - 1)
    gaps = np.minimum(np.abs(t[idx] - candidates), np.abs(t[idx - 1] - candidates))
    covered = gaps <= tol_days
    if not covered.any():
        return None
    valid = candidates[covered]
    center_t = valid[np.argmin(np.abs(valid - np.median(t)))]
    center_idx = int(np.searchsorted(t, center_t))
    start = max(0, min(len(t) - L, center_idx - L // 2))
    t_win, f_win = t[start:start + L], f[start:start + L]
    if np.min(np.abs(t_win - center_t)) > tol_days:
        return None  # window clipping near the array edge pushed the covered epoch out
    return t_win, f_win, center_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["mamba", "s4d"], default="mamba")
    ap.add_argument("--seq-len", type=int, default=16384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(RESULTS, f"ssm_world_model_{args.backbone}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SSMWorldModel(d_model=192, n_layers=6, d_state=16, backbone=args.backbone,
                          chunk_size=256).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {ckpt_path}: {count_parameters(model):,} params, "
          f"trained {len(ckpt['losses'])} steps, final loss={ckpt['losses'][-1]:.6f}", flush=True)

    jobs = load_targets()
    print(f"Evaluating on {len(jobs)} targets (4 core + 25 benchmark)", flush=True)

    rows = []
    for name, terms, period, t0, dur_hr, source in jobs:
        dur_d = dur_hr / 24.0
        print(f"\n{name} [{source}]  P={period:.4f}d  T14={dur_hr:.2f}h", flush=True)
        try:
            t, f = prep_light_curve(name, terms)
            if t is None or len(t) < args.seq_len:
                print(f"  skip: insufficient data ({0 if t is None else len(t)} pts)", flush=True)
                rows.append(dict(target=name, source=source, status="insufficient_data",
                                  period_d=period, duration_hr=dur_hr))
                continue

            win = select_window(t, f, t0, period, args.seq_len, dur_d)
            if win is None:
                rows.append(dict(target=name, source=source, status="window_failed",
                                  period_d=period, duration_hr=dur_hr))
                continue
            t_win, f_win, center_t = win
            dt_phys = np.diff(t_win, prepend=t_win[0] - np.median(np.diff(t_win))).clip(1e-4, 10.0)

            flux_t = torch.from_numpy(f_win.astype(np.float32)).unsqueeze(0).to(device)
            dt_t = torch.from_numpy(dt_phys.astype(np.float32)).unsqueeze(0).to(device)

            # Mask the transit window at inference -- NOT unmasked. Found empirically
            # (results/module2_ssm_design.md): this model was trained with loss computed
            # ONLY on masked positions, so it never received any gradient signal at
            # unmasked positions and its output there is uncalibrated garbage (~0.15-0.23
            # instead of ~1.0). An earlier version of this script ran with no masking at
            # all, misreading the guidance doc's "masked tokens hide the transit from the
            # forward pass" as describing training only -- it describes inference too
            # ("forced to predict the flux using only the unperturbed stellar history
            # before and after the block"). Mask the SINGLE transit epoch select_window()
            # centered this window on (block size ~ the real transit duration, matching
            # training's 100-600-cadence block regime) so the model must predict that one
            # transit region from context, exactly as trained.
            #
            # NOTE: template marks ONLY this one epoch (delta_t from center_t), not every
            # period-repeat inside the window. An earlier version phase-folded the whole
            # window against the full period (`(t_win - t0 + 0.5*period) % period`),
            # which for short-period targets marks dozens of real transit epochs at once
            # -- a coherent multi-epoch phase-fold at the literature-true period, NOT the
            # single-epoch statistic the p_single = (T14/24)/P chance floor below assumes.
            # That mismatch let even a trivial constant-mean predictor "detect" 79% of
            # targets (see results/module2_ssm_design.md, control-experiment section) by
            # just re-detecting the real, already-published transit signal via coherent
            # fold at its own true ephemeris -- not by measuring any predictor's skill.
            delta_t = t_win - center_t
            template = (np.abs(delta_t) < dur_d / 2).astype(np.float32)
            median_dt = float(np.median(dt_phys))
            transit_cadences = max(1, int(round(dur_d / median_dt))) if median_dt > 0 else 100
            pad = max(10, int(0.5 * transit_cadences))
            mask_np = np.zeros(args.seq_len, dtype=bool)
            in_tr_idx = np.where(np.abs(delta_t) < dur_d / 2 + pad * median_dt)[0]
            mask_np[in_tr_idx] = True
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(flux_t, dt_t, mask_t).cpu().numpy()[0]

            residual_deficit = f_win - pred  # NOTE: sign convention -- transit = predicted-too-high
            if template.sum() < 3:
                rows.append(dict(target=name, source=source, status="no_intransit_coverage",
                                  period_d=period, duration_hr=dur_hr))
                continue

            oot = template < 0.5
            sigma = float(np.nanstd(residual_deficit[oot])) if oot.sum() > 10 else float(np.nanstd(residual_deficit))
            deficit = -residual_deficit  # want positive = predicted-above-observed (a dip vs prediction)
            amp, snr = M35.matched_filter_snr(deficit, template, sigma)

            recovered = bool(snr > 5.0)  # fixed detection threshold, documented not tuned per-target
            rows.append(dict(target=name, source=source, status="ok", period_d=period,
                              duration_hr=dur_hr, snr=float(snr), recovered=recovered,
                              n_intransit=int(template.sum())))
            print(f"  SNR={snr:.2f}  recovered={recovered}", flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc()
            rows.append(dict(target=name, source=source, status=f"error: {exc}",
                              period_d=period, duration_hr=dur_hr))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, f"ssm_eval_{args.backbone}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    ok = df[df["status"] == "ok"].copy()
    n_ok = len(ok)
    n_total = len(df)
    print(f"\n{n_ok}/{n_total} targets evaluated (rest skipped: insufficient data/errors)")
    if n_ok == 0:
        print("No evaluable targets -- nothing to chance-correct.")
        return

    raw_rate = ok["recovered"].mean()
    n_recovered = int(ok["recovered"].sum())

    # chance floor: BLS-style single-epoch test, p_single = (T14/24)/P,
    # per summarize_benchmark.py's own formula -- NOT ExoVeil's multi-event
    # floor, since this detector tests a single known ephemeris, not a
    # ranked candidate list.
    ok["p_single"] = (ok["duration_hr"] / 24.0) / ok["period_d"]
    ok["p_single"] = ok["p_single"].clip(0, 1)
    chance_expected = ok["p_single"].sum()

    print(f"\n=== Chance-corrected recovery ===")
    print(f"Raw recovery: {n_recovered}/{n_ok} ({100*raw_rate:.1f}%)")
    print(f"Chance-floor expectation (single-epoch test, sum of T14/P): "
          f"{chance_expected:.2f}/{n_ok} ({100*chance_expected/n_ok:.1f}%)")
    print(f"Excess over chance: {n_recovered - chance_expected:+.2f} targets "
          f"({100*(raw_rate - chance_expected/n_ok):+.1f} percentage points)")


if __name__ == "__main__":
    main()
