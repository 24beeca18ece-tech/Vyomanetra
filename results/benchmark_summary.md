# ExoVeil vs. BLS/trapezoid -- depth-stratified recovery benchmark

Sample: **25** confirmed TESS-discovered planets drawn from the NASA Exoplanet Archive (`pscomppars`), stratified across transit-depth deciles (`select_benchmark_sample.py`, seed 42; sample frozen in `results/benchmark_sample.csv` before any run). Both pipelines see the **identical** Savitzky-Golay-detrended flux array per target (`prototype.py`'s `download_lc()`/`detrend()`), and both are scored against the archive's own `pl_orbper`/`pl_tranmid`/`pl_trandur` as ground truth -- not against each other.

**Completion**: 25/25 targets processed successfully (status `ok`), 0 failed/skipped.

## Caveat 1: what is actually being benchmarked is not ExoVeil's full published method

The shipped PyPI package `exoveil==0.2.1` exposes, through `detect_from_array()`, only the **first two stages** of the pipeline described in Priyanshu (2026): the Transformer world model and the variance-weighted matched-filter event detector. Its return value is a raw list of up to 20 threshold-crossing events (`time`, `snr`, `depth_ppm`, `duration_pts`, `near_gap`, `aleatoric`, `uncertainty_category`).

The **XGBoost planet-vs-false-positive classifier** (reported at AUC 0.938 on Kepler DR25) and the **conformal-prediction calibration stage** -- the components whose entire function is to filter and rank that raw candidate list down to calibrated detections -- are **not reachable through the public API at this version**. We inspected the installed package (`exoveil.core`, `exoveil.detect`) and found no exposed entry point for either stage.

**This bounds what the results below can claim.** They characterise the *publicly installable artifact* as it ships, evaluated end-to-end on real TESS photometry. They do **not** measure the performance of ExoVeil's full method as published, because the stage specifically designed to suppress the false positives dominating the raw event list cannot be run. A chance-level raw-candidate rate is exactly what an unfiltered matched-filter front-end *should* produce before classification -- the classifier is the missing half of the method. The correct reading is therefore a **reproducibility gap between the paper and the shipped package**, not evidence that the published method performs at chance. Any paper text drawing on this benchmark must state that distinction explicitly; claiming the latter from these data would misrepresent the prior work.

## Caveat 2: the two 'recovery' metrics are not directly comparable

ExoVeil returns up to 20 independently-scored candidate events per target and exposes no period. Scoring it as "recovered if **any** event lands in-transit" gives it 20 chances per target, whereas BLS gets one epoch. The probability a randomly-placed event falls in-transit is ~ T14/P; over N events the chance of at least one coincidental hit is 1-(1-T14/P)^N.

Summed over this sample, the **expected number of targets ExoVeil would "recover" by pure chance is 8.2/25 (33%)**, versus 0.7/25 (3%) for BLS's single-epoch test. Any ExoVeil rate must be read against that floor.

## Caveat 3: two targets are outside BLS's configured search range

`prototype.py`'s `identify()` searches periods up to `max_p = min(span/2, 30)` days. **2 of 25** benchmark targets have an archive period above that ceiling and are therefore *structurally* unrecoverable by BLS as configured -- no detrending or SNR improvement could find them, because the true period is never trialled:

- **NGTS-20 b** -- archive P = 54.19 d (> 30 d cap), decile 6
- **TOI-2449 b** -- archive P = 106.14 d (> 30 d cap), decile 8

BLS rates are reported below **both** including these (the honest as-configured number) and excluding them (the fair per-target detection number). ExoVeil has no period search and so no equivalent ceiling -- its rate is unaffected in kind, but the excluded-subset column is shown for like-for-like comparison on the same targets.

**Important**: the follow-up experiment below shows the 30 d cap is *not* the actual explanation for either miss -- raising it recovers neither target, for two different underlying reasons. Excluding them from the denominator remains justified (neither is recoverable from the available data), but the paper should not describe this as a mere tuning oversight.

## Overall recovery rates

`n=25` = all processed targets; `n=23` excludes the 2 out-of-range targets named above.

| Pipeline | Metric | Recovered (all, n=25) | Rate | Recovered (in-range, n=23) | Rate | Chance exp. (in-range) |
|---|---|---|---|---|---|---|
| BLS/trapezoid | epoch falls in-transit | 5/25 | 20% | 5/23 | **22%** | 3% |
| BLS/trapezoid | **strict**: epoch in-transit AND period matches archive (±1%, incl. 2x/3x/0.5x harmonics) | 4/25 | 16% | 4/23 | **17%** | -- |
| BLS/trapezoid | period matches archive (regardless of epoch) | 7/25 | 28% | 7/23 | 30% | -- |
| ExoVeil (raw events; see Caveat 1) | >=1 of <=20 events in-transit | 9/25 | 36% | 9/23 | **39%** | 35% |

Chance floors for the full sample: ExoVeil 33%, BLS single-epoch 3%. On the in-range subset: ExoVeil 35% vs. its actual 39% (**+4 pp** excess); BLS 3% vs. its actual 22% (**+19 pp** excess).

## Breakdown by transit-depth decile

Decile 0 = shallowest (hardest), decile 9 = deepest (easiest). `n` is targets processed in that decile.

| Decile | Depth range (ppm) | n | BLS epoch-in-transit | BLS strict | ExoVeil >=1 in-transit | ExoVeil chance exp. | ExoVeil excess |
|---|---|---|---|---|---|---|---|
| 0 | 151-330 | 3 | 1/3 (33%) | 1/3 (33%) | 2/3 (67%) | 1.4/3 (48%) | +19 pp |
| 1 | 469-540 | 3 | 0/3 (0%) | 0/3 (0%) | 3/3 (100%) | 1.7/3 (56%) | +44 pp |
| 2 | 771-853 | 3 | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) | 0.8/3 (27%) | -27 pp |
| 3 | 1060-1168 | 3 | 0/3 (0%) | 0/3 (0%) | 1/3 (33%) | 1.3/3 (45%) | -12 pp |
| 4 | 1230-1490 | 3 | 1/3 (33%) | 1/3 (33%) | 1/3 (33%) | 1.0/3 (32%) | +1 pp |
| 5 | 1808-2315 | 2 | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 0.2/2 (12%) | -12 pp |
| 6 | 2936-3708 | 2 | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 0.3/2 (17%) | -17 pp |
| 7 | 5705-5726 | 2 | 2/2 (100%) | 1/2 (50%) | 0/2 (0%) | 0.4/2 (20%) | -20 pp |
| 8 | 8801-10168 | 2 | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 0.7/2 (34%) | -34 pp |
| 9 | 110216-166590 | 2 | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) | 0.3/2 (16%) | +84 pp |

## Shallow (deciles 0-4) vs. deep (deciles 5-9)

| Group | n | BLS epoch | BLS strict | ExoVeil >=1 | ExoVeil chance exp. |
|---|---|---|---|---|---|
| Shallow (0-4) | 15 | 2/15 (13%) | 2/15 (13%) | 7/15 (47%) | 6.2/15 (42%) |
| Deep (5-9) | 10 | 3/10 (30%) | 2/10 (20%) | 2/10 (20%) | 2.0/10 (20%) |

## Follow-up: does raising the BLS period cap recover the two out-of-range targets?

`recheck_long_period.py` reruns the BLS stage on just these two targets with the period ceiling raised from 30 d to 150 d (and a denser period grid, so grid resolution is not a confound), in three configurations: as-benchmarked (4 sectors, 30 d cap); cap raised; and cap raised with up to 20 sectors requested. Everything else is held identical.

| Target | Config | Sectors used | Baseline span (d) | Effective max_p (d) | Binding constraint | Archive P reachable? | BLS P (d) | In-transit? |
|---|---|---|---|---|---|---|---|---|
| TOI-2449 b | 1_baseline_4sec_cap30 | 3 | 2635.0 | 30.00 | `ceiling` | NO | 29.5132 | no |
| TOI-2449 b | 2_capraised_4sec | 3 | 2635.0 | 150.00 | `ceiling` | YES | 0.5150 | no |
| TOI-2449 b | 3_capraised_manysec | 3 | 2635.0 | 150.00 | `ceiling` | YES | 0.5150 | no |
| NGTS-20 b | 1_baseline_4sec_cap30 | 2 | 25.4 | 12.71 | `span/2` | NO | 6.4638 | no |
| NGTS-20 b | 2_capraised_4sec | 2 | 25.4 | 12.71 | `span/2` | NO | 4.6091 | no |
| NGTS-20 b | 3_capraised_manysec | 2 | 25.4 | 12.71 | `span/2` | NO | 4.6091 | no |

### Verdict: the two targets fail for *different* reasons

- **TOI-2449 b (P = 106.14 d)** -- the 30 d ceiling *was* the binding constraint (baseline span is 2635 d, so `span/2` = 1317 d never bound). Raising the ceiling to 150 d put the true period inside the search grid. **BLS still did not recover it**: it returned 0.5150 d, a spurious short-period peak, still out-of-transit. Only 3 sectors exist for this target, scattered across a 2635 d baseline, so very few of its 106 d-period transits fall inside an observed window. **The cap is not the explanation -- this is a genuine sampling limitation.**

- **NGTS-20 b (P = 54.19 d)** -- the 30 d ceiling was **not** the binding constraint at all; `span/2` was. Only 2 sectors exist, giving a 25.43 d baseline, so `max_p = 12.71 d` regardless of the ceiling setting. Raising the cap changed nothing (12.71 d either way) and the true period was never reachable in any configuration. Recovering a 54 d period requires `span >= 2P = 108 d` of data; **only 25 d exists**, and requesting 20 sectors still returned 2 because that is all MAST has. This is a hard data-availability limit, not a tuning choice -- no configuration of this pipeline could ever recover it from the available photometry.

Neither target is rescued by widening the search, so excluding them from the BLS denominator is justified on structural grounds -- but note the justification differs: TOI-2449 b is *searchable but under-sampled*, NGTS-20 b is *not searchable at all* with the existing baseline.

## ExoVeil SNR numerical blow-ups

**10/25** targets produced a top-event SNR above the sanity ceiling (1000), i.e. a degenerate value from the local-MAD noise estimator collapsing toward zero (`exoveil.detect.detect_events()` weights residuals by 1/local_MAD). This reproduces at scale the failure mode first seen on V1828 Aql in the 4-target run -- it is **not** confined to eclipsing binaries. Affected targets:

- TOI-5788 b (decile 0, raw SNR 12024400.0)
- TOI-2193 A b (decile 0, raw SNR 46015100.0)
- TOI-2431 b (decile 1, raw SNR 9871310.0)
- HD 108236 c (decile 1, raw SNR 3988020.0)
- TOI-6651 b (decile 2, raw SNR 4523860.0)
- TOI-244 b (decile 3, raw SNR 32240700.0)
- TOI-6109 b (decile 4, raw SNR 76274800.0)
- LHS 1903 d (decile 5, raw SNR 250576000.0)
- DS Tuc A b (decile 6, raw SNR 345473.0)
- TOI-6564 b (decile 8, raw SNR 134448000.0)

## Per-target detail

| Target | Decile | Archive P (d) | Archive depth (ppm) | BLS P (d) | BLS period match | BLS epoch in-transit | ExoVeil #events | ExoVeil in-transit | ExoVeil chance | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| TOI-1860 b | 0 | 1.0662 | 151 | 1.0665 | yes | YES | 20 | 1 | 60% |  |
| TOI-2193 A b | 0 | 2.1226 | 191 | 2.1227 | yes | no | 20 | 2 | 48% |  |
| TOI-5788 b | 0 | 6.3408 | 330 | 12.4074 | no | no | 20 | 0 | 36% |  |
| HD 108236 c | 1 | 6.2034 | 469 | 25.9787 | no | no | 20 | 1 | 32% |  |
| TOI-2411 b | 1 | 0.7827 | 527 | 24.7160 | no | no | 20 | 2 | 68% |  |
| TOI-2431 b | 1 | 0.2242 | 540 | 5.8312 | no | no | 20 | 1 | 67% |  |
| TOI-4311 c | 2 | 15.0704 | 853 | 16.8033 | no | no | 14 | 0 | 14% |  |
| TOI-5126 c | 2 | 17.8999 | 771 | 28.7107 | no | no | 20 | 0 | 22% |  |
| TOI-6651 b | 2 | 5.0570 | 784 | 3.7823 | no | no | 20 | 0 | 45% |  |
| TOI-1533 b | 3 | 3.6458 | 1168 | 16.1277 | no | no | 17 | 0 | 33% |  |
| TOI-2337 b | 3 | 2.9943 | 1121 | 27.2798 | no | no | 18 | 4 | 84% |  |
| TOI-244 b | 3 | 7.3972 | 1060 | 16.4965 | no | no | 20 | 0 | 18% |  |
| TOI-5126 b | 4 | 5.4588 | 1230 | 28.7107 | no | no | 20 | 0 | 44% |  |
| TOI-6109 b | 4 | 5.6905 | 1470 | 27.6840 | no | no | 20 | 0 | 15% |  |
| TOI-824 b | 4 | 1.3930 | 1490 | 4.1790 | yes (3x) | YES | 13 | 1 | 37% |  |
| LHS 1903 d | 5 | 12.5663 | 1808 | 10.4867 | no | no | 20 | 0 | 17% |  |
| TOI-1470 c | 5 | 18.0882 | 2315 | 2.5269 | no | no | 13 | 0 | 8% |  |
| DS Tuc A b | 6 | 8.1383 | 2936 | 27.8846 | no | no | 20 | 0 | 28% |  |
| NGTS-20 b | 6 | 54.1891 | 3708 | 6.4638 | no | no | 20 | 0 | 7% | **outside configured BLS search range (P > 30 d)** |
| TOI-1135 b | 7 | 8.0277 | 5705 | 16.0569 | yes (2x) | YES | 19 | 0 | 36% |  |
| TOI-7510 c | 7 | 22.5687 | 5726 | 16.5294 | no | YES | 4 | 0 | 4% |  |
| TOI-2449 b | 8 | 106.1447 | 10168 | 29.5132 | no | no | 18 | 0 | 6% | **outside configured BLS search range (P > 30 d)** |
| TOI-6564 b | 8 | 3.9854 | 8801 | 3.9843 | yes | no | 20 | 0 | 62% |  |
| TOI-519 b | 9 | 1.2652 | 110216 | 1.2641 | yes | no | 8 | 1 | 28% |  |
| TOI-6894 b | 9 | 3.3708 | 166590 | 3.3714 | yes | YES | 2 | 1 | 3% |  |

