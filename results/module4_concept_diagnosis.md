# Module 4 concept-computation diagnosis: TOI-1930.01 outlier and C4 NaN policy

Two open items from the 14-target concept-validation run, resolved before
scaling to the full 50-target `cbm_sample.csv`.

## 1. TOI-1930.01 (C7 = -3.333, C4 = 2.0): genuine period alias, not a bug

**Diagnosis: confirmed astrophysical/pipeline edge case -- a second, independent
instance of the exact BLS period-aliasing failure mode Novel Contribution 4
(C7) was designed to catch, on a real confirmed planet this time rather
than V1828 Aql's known eclipsing binary.**

TOI-1930.01's archive ephemeris (`cbm_sample.csv`, TIC 160578764):
P=2.878664 d, depth=8710 ppm. Our pipeline's own BLS search (`identify()`,
unmodified from prototype.py) instead found **P=8.62511 d**. Ratio:

    8.62511 / 2.878664 = 2.9962  ~=  3.000

BLS locked onto the **3rd harmonic** of the true period. Consequences,
traced end to end:

- The trapezoid fit at the wrong (8.625 d) period found only a shallow,
  averaged remnant: depth=220.7 +/- 161.6 ppm -- a factor of ~40x below
  the true 8710 ppm, the same "averaged remnant of partially-overlapping
  real eclipses" mechanism the report describes for V1828 Aql's own alias
  (Section 5.4).
- Per-epoch depths at the aliased period (8 epochs: [0,1,2,86,87,208,209,210]):
  `[-102.7, 34.4, -11.8, 0.0, 0.0, 364.8, 1098.0, 307.7]` ppm -- wildly
  inconsistent, including two exact zeros (the true transit landed outside
  that epoch's aliased window entirely) and both negative and >1000 ppm
  values elsewhere.
- median=17.2 ppm, MAD=74.4 ppm -> **C7 = 1 - 74.4/17.2 = -3.33**, exactly
  reproducing the reported value.

This confirms C7 is not bounded to [0,1] by construction -- Eq. 2 can go
negative whenever MAD exceeds the median, which happens precisely when a
handful of epochs show large, real (if erratic) signal while the median
epoch shows almost nothing. That is itself informative (a strongly negative
C7 is, if anything, a MORE extreme incoherence flag than 0), not an error
to clip or suppress.

**No concept-function code changes were made for this item** -- the
diagnosis confirmed the existing computation is behaving correctly on
contaminated (aliased) upstream input from Module 2. This is a Module 2
(BLS) limitation being correctly surfaced by Module 4, exactly as the
report's own Section 4.1 argues C7 should.

## 2. C4 NaN rate: not simply an epoch-count problem -- documented inclusion/handling policy

Quantified NaN rate against `n_epochs` across all 14 validation targets:

| n_epochs bucket | NaN rate |
|---|---|
| < 4 | 2/2 (100%) |
| = 4 | 1/3 (33%) |
| 5-16 | 0/4 (0%) |
| 17-50 | 2/4 (50%) |
| > 50 | 0/1 (0%) |

**Not monotonic in epoch count.** Two of the five NaN cases occur at
substantial epoch counts (TOI-491.01, n=20; V1828 Aql, n=46) where a naive
"more epochs = more reliable" assumption would predict a clean value. A
minimum-epoch floor alone would not eliminate this.

Root-caused both non-trivial (n>=4) NaN cases:

- **V1828 Aql (n=46):** the already-diagnosed period-alias case from the
  original C7 edge-case fix (majority of epochs measure ~0 depth because
  the true eclipse falls outside the aliased window at that epoch). BLS
  period IS wrong here.
- **TOI-491.01 (n=20):** checked the BLS-vs-archive period ratio --
  3.33228 / 3.332681 = 1.00012, i.e. NOT an alias (correct period to
  within 0.01%). Per-epoch depths instead show two clean BLOCKS of exactly
  zero (epochs 0-6 and 303-309, 13/20 epochs) bracketing one block of
  strong, internally-consistent detections (epochs 221-228: 520-1528 ppm).
  This is a DIFFERENT root cause from aliasing: sparse/noisy per-epoch
  photometric coverage at a shallow (435 ppm) depth causes some epochs'
  matched-filter estimate to land at exactly zero rather than showing
  noise scatter around a small nonzero value -- a data-coverage
  limitation, not a period error.

Both mechanisms produce a median-of-depths exactly (or near) zero, which is
what breaks C4's (and C7's) ratio -- but they are mechanistically distinct
(period error vs. sparse coverage), while producing the same NaN symptom.

**Policy adopted:**

1. **Hard inclusion floor: `n_epochs >= 4`** for any per-epoch concept
   (C4, C7) to be computed at all -- already enforced in
   `compute_c4_odd_even()`/`per_epoch_depths_trapezoid()`. Below this, the
   concept is not just noisy, it is mathematically underdetermined (can't
   split into odd/even, or MAD of 1-2 points is meaningless). Targets with
   `n_epochs < 4` keep C1/C2/C3/C5/C6 (which don't depend on epoch count)
   but get NaN for C4/C7 by design, not by data misfortune.

2. **Above that floor, a remaining NaN is treated as an informative
   missingness signal, not excluded or silently imputed with a fabricated
   value.** Both diagnosed n>=4 NaN cases (V1828 Aql, TOI-491.01) reflect
   real per-target data-quality problems (aliasing or sparse coverage)
   that a downstream classifier should be allowed to see, not paper over.
   Concretely: C4/C7 are left as NaN in `results/cbm_concepts_full.csv`,
   and CBM training (next stage, not yet started) will either (a) use a
   NaN-native model (gradient-boosted trees handle this directly), or (b)
   add an explicit `C4_computable`/`C7_computable` boolean feature
   alongside a neutral imputed value, so "this concept could not be
   reliably measured" is itself visible to the classifier rather than
   hidden inside a plausible-looking number.

3. This means a nonzero fraction of the 50-target sample (empirically
   ~20-35% of targets, per the validation-subset rate, dominated by the
   `n_epochs<4` floor plus a residual ~10-15% majority-zero rate at higher
   counts) will carry NaN for C4 and/or C7. That is expected and will be
   reported explicitly alongside the full-sample concept computation, not
   discovered later during classifier training.
