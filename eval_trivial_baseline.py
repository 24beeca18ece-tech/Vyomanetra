#!/usr/bin/env python3
"""
Control experiment for Module 2's SSM world-model eval (eval_ssm_world_model.py).

Runs the IDENTICAL downstream detection pipeline -- same 29-target list, same
light-curve prep, same window selection, same matched-filter/chance-floor
logic -- but replaces the model's prediction with a trivial constant: the
window's own mean flux, at every position (not just masked ones), matching
how the SSM's full-window residual was computed (residual = f_win - pred,
matched-filtered over the whole window, sigma from out-of-transit positions).

Purpose: this detector has PERFECT knowledge of the true ephemeris (period,
t0, duration) baked into where it looks for a dip -- unlike a blind BLS
search. If a trivial "flat baseline + matched filter at the known ephemeris"
control also shows a large excess over its chance floor, that means the
excess in the Mamba/S4D numbers is coming from the detector/ephemeris-lookup
machinery, not from the SSM's prediction doing any real work, and the
chance-floor accounting or candidate-window selection needs a line-by-line
audit before trusting the S4D result. If the trivial control lands near its
chance floor, the SSM's excess is real added signal.

Usage: python eval_trivial_baseline.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype import RESULTS  # noqa: E402
from eval_ssm_world_model import load_targets, prep_light_curve, select_window  # noqa: E402
import modules_3_5 as M35  # noqa: E402

SEQ_LEN = 16384


def main():
    jobs = load_targets()
    print(f"Evaluating trivial-mean control on {len(jobs)} targets (4 core + 25 benchmark)", flush=True)

    rows = []
    for name, terms, period, t0, dur_hr, source in jobs:
        dur_d = dur_hr / 24.0
        print(f"\n{name} [{source}]  P={period:.4f}d  T14={dur_hr:.2f}h", flush=True)
        try:
            t, f = prep_light_curve(name, terms)
            if t is None or len(t) < SEQ_LEN:
                print(f"  skip: insufficient data ({0 if t is None else len(t)} pts)", flush=True)
                rows.append(dict(target=name, source=source, status="insufficient_data",
                                  period_d=period, duration_hr=dur_hr))
                continue

            win = select_window(t, f, t0, period, SEQ_LEN, dur_d)
            if win is None:
                rows.append(dict(target=name, source=source, status="window_failed",
                                  period_d=period, duration_hr=dur_hr))
                continue
            t_win, f_win, center_t = win

            # trivial predictor: the window's own mean, at every position --
            # same shape/role as the SSM's full-window `pred` array, so the
            # downstream residual/matched-filter/chance-floor logic below is
            # byte-for-byte identical to eval_ssm_world_model.py from here on.
            pred = np.full_like(f_win, f_win.mean())

            # Template marks ONLY the single epoch select_window() centered this
            # window on (delta_t from center_t), not every period-repeat inside the
            # window -- see eval_ssm_world_model.py's matching comment for why the
            # earlier full-period phase-fold was a coherent multi-epoch statistic
            # mismatched against the single-epoch p_single chance floor.
            delta_t = t_win - center_t
            template = (np.abs(delta_t) < dur_d / 2).astype(np.float32)
            if template.sum() < 3:
                rows.append(dict(target=name, source=source, status="no_intransit_coverage",
                                  period_d=period, duration_hr=dur_hr))
                continue

            residual_deficit = f_win - pred
            oot = template < 0.5
            sigma = float(np.nanstd(residual_deficit[oot])) if oot.sum() > 10 else float(np.nanstd(residual_deficit))
            deficit = -residual_deficit
            amp, snr = M35.matched_filter_snr(deficit, template, sigma)

            recovered = bool(snr > 5.0)
            rows.append(dict(target=name, source=source, status="ok", period_d=period,
                              duration_hr=dur_hr, snr=float(snr), recovered=recovered,
                              n_intransit=int(template.sum())))
            print(f"  SNR={snr:.2f}  recovered={recovered}", flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc()
            rows.append(dict(target=name, source=source, status=f"error: {exc}",
                              period_d=period, duration_hr=dur_hr))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, "ssm_eval_trivial_baseline.csv")
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

    ok["p_single"] = (ok["duration_hr"] / 24.0) / ok["period_d"]
    ok["p_single"] = ok["p_single"].clip(0, 1)
    chance_expected = ok["p_single"].sum()

    print(f"\n=== Chance-corrected recovery (trivial constant-mean baseline) ===")
    print(f"Raw recovery: {n_recovered}/{n_ok} ({100*raw_rate:.1f}%)")
    print(f"Chance-floor expectation (single-epoch test, sum of T14/P): "
          f"{chance_expected:.2f}/{n_ok} ({100*chance_expected/n_ok:.1f}%)")
    print(f"Excess over chance: {n_recovered - chance_expected:+.2f} targets "
          f"({100*(raw_rate - chance_expected/n_ok):+.1f} percentage points)")


if __name__ == "__main__":
    main()
