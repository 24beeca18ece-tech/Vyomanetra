#!/usr/bin/env python3
"""
Module 2 -- replace the invalid analytic chance floor (p_single = (T14/24)/P)
in the SSM-world-model eval with a proper EMPIRICAL false-alarm rate, for all
three conditions (trivial constant-mean, Mamba, S4D).

Why: the anti-phase diagnostic (t0 shifted by period/2, see
results/module2_ssm_design.md) showed the analytic p_single floor is itself
invalid for this statistic on real TESS photometry -- even with zero real
transit signal, the false-alarm rate was ~6x the naive formula (14.8% vs.
2.5%), consistent with the well-known red/correlated-noise problem that
Module 3's `bootstrap_fap` (modules_3_5.py) was built to handle for the
BLS/trapezoid pipeline via an out-of-transit block-bootstrap.

`bootstrap_fap` itself is NOT called directly here: its interface expects a
fitted Mandel-Agol `theta_best` and internally derives the in/out-of-transit
template from that nonlinear model, whereas this eval's "template" is a
simple box mask and its "residual" comes from an arbitrary predictor (a
constant, or an SSM's masked-position output) rather than a Mandel-Agol fit
residual -- there is no theta_best to hand it. Re-purposing it would mean
either fitting a Mandel-Agol model to every target here too (out of scope --
Module 3 already does that separately) or quietly changing what the function
computes, which is worse than not reusing it. Instead, per the instruction to
avoid re-deriving statistics from scratch, this script reuses:
  (1) the exact same `matched_filter_snr` (modules_3_5.py) -- not
      reimplemented,
  (2) the exact same conservative small-sample FAP convention
      `(1 + #{null >= observed}) / (n + 1)` that `bootstrap_fap` uses (so a
      target with zero crossings in its null sample is never reported as a
      hard zero),
  (3) the exact same `select_window`/template/mask construction already
      fixed and verified in eval_ssm_world_model.py -- generalized to accept
      an arbitrary query epoch instead of only the true one, so the SAME
      code path that produces the real, on-phase statistic also produces
      every null draw (this is a direct generalization of the one-shot
      anti-phase control already run, not new detection logic).

Null generation: per target, draw many random orbital-phase offsets
(excluding a buffer around the true phase), find a real-data-covered
transit-shaped window at that offset (reusing select_window), and compute
the SAME detection statistic there. The empirical per-target false-alarm
rate is the fraction of those null draws that would cross the SAME SNR>5
threshold used for the real detection call -- i.e. an empirically measured
per-target chance floor, replacing the analytic p_single term in the
aggregate excess-over-chance table.

Usage: python eval_empirical_fap.py [--n-boot 500] [--batch-size 32]
"""
import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import RESULTS  # noqa: E402
from eval_ssm_world_model import load_targets, prep_light_curve, select_window  # noqa: E402
from modules_2_ssm import SSMWorldModel, count_parameters  # noqa: E402
import modules_3_5 as M35  # noqa: E402

SEQ_LEN = 16384
SNR_THRESHOLD = 5.0


def build_template_mask(t_win, center_t, dur_d):
    """Identical logic to the (now-fixed) single-epoch template/mask
    construction in eval_ssm_world_model.py -- shared here so both the
    observed statistic and every null draw go through the exact same code."""
    delta_t = t_win - center_t
    template = (np.abs(delta_t) < dur_d / 2).astype(np.float32)
    if template.sum() < 3:
        return template, None
    median_dt = float(np.median(np.diff(t_win)))
    transit_cadences = max(1, int(round(dur_d / median_dt))) if median_dt > 0 else 100
    pad = max(10, int(0.5 * transit_cadences))
    mask_np = np.zeros(len(t_win), dtype=bool)
    mask_np[np.abs(delta_t) < dur_d / 2 + pad * median_dt] = True
    return template, mask_np


def snr_from_pred(f_win, pred, template):
    residual_deficit = f_win - pred
    oot = template < 0.5
    sigma = float(np.nanstd(residual_deficit[oot])) if oot.sum() > 10 else float(np.nanstd(residual_deficit))
    deficit = -residual_deficit
    _, snr = M35.matched_filter_snr(deficit, template, sigma)
    return snr


def draw_query_epochs(t, f, t0, period, dur_d, n_want, rng, max_attempts_factor=6, exclude_true=True):
    """Random orbital-phase offsets, excluding a buffer around the true
    (phase=0) epoch, each mapped to a real-data-covered window via
    select_window (redrawing on failure/exclusion until n_want valid draws
    or the attempt budget is exhausted)."""
    exclude_phase = min(0.45, max(3.0 * dur_d, 4.0 / 24.0) / period)
    wins = []
    attempts = 0
    max_attempts = n_want * max_attempts_factor
    while len(wins) < n_want and attempts < max_attempts:
        attempts += 1
        offset = rng.uniform(0.0, 1.0)
        fold = min(offset, 1.0 - offset)
        if exclude_true and fold < exclude_phase:
            continue
        t0_query = t0 + offset * period
        win = select_window(t, f, t0_query, period, SEQ_LEN, dur_d)
        if win is None:
            continue
        wins.append(win)
    return wins


def eval_target_trivial(t, f, t0, period, dur_d, n_boot, rng):
    win = select_window(t, f, t0, period, SEQ_LEN, dur_d)
    if win is None:
        return None
    t_win, f_win, center_t = win
    template, mask_np = build_template_mask(t_win, center_t, dur_d)
    if mask_np is None:
        return None
    pred = np.full_like(f_win, f_win.mean())
    snr_obs = snr_from_pred(f_win, pred, template)

    null_snrs = []
    for t_win_n, f_win_n, center_t_n in draw_query_epochs(t, f, t0, period, dur_d, n_boot, rng):
        template_n, mask_n = build_template_mask(t_win_n, center_t_n, dur_d)
        if mask_n is None:
            continue
        pred_n = np.full_like(f_win_n, f_win_n.mean())
        null_snrs.append(snr_from_pred(f_win_n, pred_n, template_n))
    return snr_obs, np.array(null_snrs)


def model_predict_batch(model, flux_list, dt_list, mask_list, device):
    flux_t = torch.from_numpy(np.stack(flux_list).astype(np.float32)).to(device)
    dt_t = torch.from_numpy(np.stack(dt_list).astype(np.float32)).to(device)
    mask_t = torch.from_numpy(np.stack(mask_list)).to(device)
    with torch.no_grad():
        pred = model(flux_t, dt_t, mask_t).cpu().numpy()
    return pred


def eval_target_model(model, device, t, f, t0, period, dur_d, n_boot, rng, batch_size):
    win = select_window(t, f, t0, period, SEQ_LEN, dur_d)
    if win is None:
        return None
    t_win, f_win, center_t = win
    template, mask_np = build_template_mask(t_win, center_t, dur_d)
    if mask_np is None:
        return None
    dt_phys = np.diff(t_win, prepend=t_win[0] - np.median(np.diff(t_win))).clip(1e-4, 10.0).astype(np.float32)
    pred = model_predict_batch(model, [f_win], [dt_phys], [mask_np], device)[0]
    snr_obs = snr_from_pred(f_win, pred, template)

    null_wins = draw_query_epochs(t, f, t0, period, dur_d, n_boot, rng)
    null_snrs = []
    for i in range(0, len(null_wins), batch_size):
        chunk = null_wins[i:i + batch_size]
        flux_list, dt_list, mask_list, template_list, fwin_list = [], [], [], [], []
        for t_win_n, f_win_n, center_t_n in chunk:
            template_n, mask_n = build_template_mask(t_win_n, center_t_n, dur_d)
            if mask_n is None:
                continue
            dt_n = np.diff(t_win_n, prepend=t_win_n[0] - np.median(np.diff(t_win_n))).clip(1e-4, 10.0).astype(np.float32)
            flux_list.append(f_win_n); dt_list.append(dt_n); mask_list.append(mask_n)
            template_list.append(template_n); fwin_list.append(f_win_n)
        if not flux_list:
            continue
        preds = model_predict_batch(model, flux_list, dt_list, mask_list, device)
        for f_win_n, pred_n, template_n in zip(fwin_list, preds, template_list):
            null_snrs.append(snr_from_pred(f_win_n, pred_n, template_n))
    return snr_obs, np.array(null_snrs)


def empirical_fap(null_snrs, threshold=SNR_THRESHOLD):
    """Fraction of null draws crossing the same recovery threshold, with the
    add-one conservative correction reused from bootstrap_fap's
    `(1 + #{...}) / (n + 1)` convention -- never reports a hard zero at
    finite sample size."""
    n = len(null_snrs)
    if n == 0:
        return np.nan
    return (1 + int(np.sum(null_snrs >= threshold))) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--targets-limit", type=int, default=None, help="debug: cap number of targets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for backbone in ["mamba", "s4d"]:
        ckpt = torch.load(os.path.join(RESULTS, f"ssm_world_model_{backbone}.pt"),
                           map_location=device, weights_only=False)
        m = SSMWorldModel(d_model=192, n_layers=6, d_state=16, backbone=backbone, chunk_size=256).to(device)
        m.load_state_dict(ckpt["model_state"]); m.eval()
        models[backbone] = m
        print(f"loaded {backbone}: {count_parameters(m):,} params", flush=True)

    jobs = load_targets()
    if args.targets_limit:
        jobs = jobs[:args.targets_limit]

    rows = {"trivial": [], "mamba": [], "s4d": []}
    t_start = time.time()
    for name, terms, period, t0, dur_hr, source in jobs:
        dur_d = dur_hr / 24.0
        print(f"\n{name} [{source}]  P={period:.4f}d  T14={dur_hr:.2f}h", flush=True)
        t, f = prep_light_curve(name, terms)
        if t is None or len(t) < SEQ_LEN:
            print(f"  skip: insufficient data", flush=True)
            for cond in rows:
                rows[cond].append(dict(target=name, source=source, status="insufficient_data",
                                        period_d=period, duration_hr=dur_hr))
            continue

        rng = np.random.default_rng(args.seed)
        rt = eval_target_trivial(t, f, t0, period, dur_d, args.n_boot, rng)
        if rt is None:
            for cond in rows:
                rows[cond].append(dict(target=name, source=source, status="no_intransit_coverage",
                                        period_d=period, duration_hr=dur_hr))
            continue
        snr_obs_triv, null_triv = rt
        fap_triv = empirical_fap(null_triv)
        rows["trivial"].append(dict(target=name, source=source, status="ok", period_d=period,
                                     duration_hr=dur_hr, snr=float(snr_obs_triv),
                                     recovered=bool(snr_obs_triv > SNR_THRESHOLD),
                                     empirical_fap=fap_triv, n_null=len(null_triv)))
        print(f"  trivial: SNR={snr_obs_triv:.2f}  empirical_fap={fap_triv:.4f} (n_null={len(null_triv)})", flush=True)

        for backbone in ["mamba", "s4d"]:
            rm = eval_target_model(models[backbone], device, t, f, t0, period, dur_d,
                                    args.n_boot, rng, args.batch_size)
            if rm is None:
                rows[backbone].append(dict(target=name, source=source, status="no_intransit_coverage",
                                            period_d=period, duration_hr=dur_hr))
                continue
            snr_obs, null_snrs = rm
            fap = empirical_fap(null_snrs)
            rows[backbone].append(dict(target=name, source=source, status="ok", period_d=period,
                                        duration_hr=dur_hr, snr=float(snr_obs),
                                        recovered=bool(snr_obs > SNR_THRESHOLD),
                                        empirical_fap=fap, n_null=len(null_snrs)))
            print(f"  {backbone}: SNR={snr_obs:.2f}  empirical_fap={fap:.4f} (n_null={len(null_snrs)})", flush=True)

        print(f"  [elapsed {time.time()-t_start:.0f}s]", flush=True)

    print(f"\n=== Empirically-calibrated recovery (n_boot={args.n_boot} random-phase nulls/target) ===")
    for cond in ["trivial", "mamba", "s4d"]:
        df = pd.DataFrame(rows[cond])
        df.to_csv(os.path.join(RESULTS, f"ssm_eval_empirical_{cond}.csv"), index=False)
        ok = df[df["status"] == "ok"].copy()
        n_ok = len(ok)
        if n_ok == 0:
            print(f"{cond}: no evaluable targets"); continue
        n_rec = int(ok["recovered"].sum())
        chance_expected = ok["empirical_fap"].sum()
        raw_rate = n_rec / n_ok
        print(f"\n[{cond}]  n={n_ok}")
        print(f"  Raw recovery: {n_rec}/{n_ok} ({100*raw_rate:.1f}%)")
        print(f"  Empirical chance-floor expectation (sum of per-target empirical FAP): "
              f"{chance_expected:.2f}/{n_ok} ({100*chance_expected/n_ok:.1f}%)")
        print(f"  Excess over empirical chance: {n_rec - chance_expected:+.2f} targets "
              f"({100*(raw_rate - chance_expected/n_ok):+.1f} percentage points)")

    print(f"\nTotal wall-clock: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
