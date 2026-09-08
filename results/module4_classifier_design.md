# Module 4 CBM classifier: pre-training checks and missingness design

Two checks resolved before training, per instructions, plus the resulting
classifier design decisions.

---

## Check 1: label split, excluded vs. included by the n_epochs>=4 floor

Full concept set: 51 targets (50-target `cbm_sample.csv` + TOI-700, which
carries label "planet" and was folded into this analysis since it's a real
additional planet data point, not just a sanity check).

| Group | n | planet | not_planet | planet % |
|---|---|---|---|---|
| Excluded (n_epochs < 4) | 14 | 9 | 5 | 64.3% |
| Included (n_epochs >= 4) | 37 | 17 | 20 | 45.9% |
| **Full sample** | **51** | **26** | **25** | **51.0%** |

**The overall 26/25 split is fine (~51/49%, negligible drift from the
nominal 25/25 draw).** But the exclusion is NOT applied evenly across
classes: **34.6% of all planets (9/26) get excluded by the n_epochs<4
floor, vs. only 20.0% of all not_planet targets (5/25)** -- planets are
~1.7x more likely to be excluded than false positives.

**Why, mechanistically:** longer-period systems produce fewer transits
within a fixed TESS baseline (hence low n_epochs), and longer-period
candidates are disproportionately the ones still carrying a KP/CP
(confirmed/known planet) disposition specifically because they needed
additional follow-up (RV, multi-sector confirmation) to validate -- while
many FPs in this sample are short-period grazing EBs or blends, which
produce many transits/eclipses within the same baseline (TOI-6764.01, the
379,261 ppm outlier from the earlier report, has multiple epochs precisely
because it's a very short-period system). This is a real, direction-
consistent selection effect, not sampling noise.

**Consequence for training: literal exclusion of n_epochs<4 targets would
disproportionately remove planets from the training set**, shrinking and
biasing the effective sample beyond what the raw 26/25 label count
suggests. This is the decisive argument (beyond the "informative
missingness" principle already adopted for the concept values themselves)
for KEEPING all 51 targets in the training set with C3/C4/C5/C6/C7 masked
as missing where they could not be computed, rather than dropping rows.

---

## Check 2: missingness representation for the classifier

Extended the audit beyond C4/C7 (the two concepts flagged during
validation) to all seven concepts, on the full 51-target set:

| Concept | NaN rate |
|---|---|
| C1 (depth) | 0.0% |
| C2 (shape) | 0.0% |
| C3 (ingress/egress symmetry) | 3.9% |
| C4 (odd-even) | 35.3% |
| C5 (secondary eclipse) | 7.8% |
| C6 (centroid shift) | 15.7% |
| C7 (phase-fold coherence) | 9.8% |

C1/C2 come straight from the trapezoid fit and are never missing. C3, C4,
C5, C6, C7 all have some missingness (not just C4/C7) -- all five need the
same treatment, not a special case for two of them.

**Decision: per-concept missingness indicator + median-imputed placeholder,
fed to a standard (non-NaN-native) shallow MLP -- not a tree-based model
that masks missing values internally.**

Two options were on the table:

1. **Indicator + imputation into a plain feed-forward classifier.**
2. **A NaN-native architecture** (e.g. gradient-boosted trees), which
   handles missing splits internally without needing indicators.

**Chose (1), for two reasons:**

- **Consistency with the report's own spec.** Section 3.4 explicitly
  specifies "a shallow multilayer perceptron" as Module 4's classification
  stage (Table label: "classified from those concepts alone ... using a
  shallow multilayer perceptron"). Switching to a tree ensemble for
  missingness convenience would be a silent architecture deviation from
  what the report commits to and what will be evaluated against it.
- **The indicator flag is itself informative, matching the "informative
  missingness" principle already adopted for the data** (see
  `module4_concept_diagnosis.md`). A bare imputed value (e.g. the training
  median) is indistinguishable from a real low/typical measurement unless
  the model can see that it was manufactured -- the companion binary flag
  (`C{n}_missing`) gives the MLP that information directly as an input
  dimension, rather than hiding it inside tree-split logic that isn't
  reportable as a concept-level interpretability result for Section 3.4.

**Implementation:** feature vector = 7 concept values (median-imputed
using the TRAINING FOLD's own median for C3/C4/C5/C6/C7, to avoid leaking
test-fold information through the imputation statistic) + 5 missingness
indicators (C3_missing, ..., C7_missing; none needed for C1/C2) = **12
input features**, standardized (zero mean / unit variance, fit per training
fold) before the MLP.

---

## Training results: two diagnosed classifier-stage limitations

The classifier (shallow MLP, 8 hidden units, per Section 3.4) was trained
and evaluated via stratified 5-fold cross-validation on the full 51-target
sample. Two problems surfaced, both run to a specific, stated cause before
being reported here -- neither is presented as an unexplained "the model
didn't work."

### Diagnosis A: small-N cross-validation instability

5-fold CV performance (mean +/- std across folds):

| Metric | Mean +/- std | Per-fold range |
|---|---|---|
| Accuracy | 0.627 +/- 0.132 | 0.40 - 0.80 |
| Precision | 0.636 +/- 0.116 | 0.44 - 0.80 |
| Recall | 0.733 +/- 0.084 | 0.60 - 0.80 |
| F1 | 0.673 +/- 0.083 | 0.57 - 0.80 |
| ROC-AUC | 0.740 +/- 0.150 | 0.48 - 0.92 |

With only ~10 held-out samples per fold, single-fold metrics are extremely
noisy by construction (one fold's AUC, 0.48, is statistically
indistinguishable from random guessing; another's, 0.92, looks strong) --
the fold-to-fold spread is a direct, expected consequence of N=51, not
evidence the concepts lack signal. A logistic-regression baseline (far
less capacity than an 8-unit MLP) shows the same instability (accuracy
0.585 +/- 0.152, AUC 0.691 +/- 0.197, one fold's AUC at 0.36), which rules
out "the MLP is overparameterized for this N" as the sole explanation --
the ceiling here is set by sample size, not model choice.

**Conclusion: this is a genuine small-sample-size limitation, not a
concept-computation problem** (which remains validated at both the
14-target and 52-target scale; see `module4_concept_diagnosis.md`). It
directly matches the report's own Section 6.3 anticipation: "Apply the
complete pipeline to the ISRO/PRL-curated TESS dataset once released" --
the report never claimed the classifier would be trainable to a reportable
standard on a hand-assembled, few-dozen-target prototype sample.

### Diagnosis B: missingness indicators encode the epoch-count/label confound from Check 1, not pure astrophysical signal

An independent, ground-truth sanity check -- predicting V1828 Aql (a known
eclipsing binary, held out of training entirely) -- failed outright:

| Model variant | V1828 Aql P(planet) | CV accuracy | CV ROC-AUC |
|---|---|---|---|
| MLP, with missingness indicators | **0.997** (should be low) | 0.627 +/- 0.132 | 0.740 +/- 0.150 |
| Logistic regression, with indicators | **0.798** (should be low) | 0.585 +/- 0.152 | 0.691 +/- 0.197 |
| Logistic regression, indicators removed | **0.452** (correctly < 0.5) | 0.547 +/- 0.087 | 0.545 +/- 0.097 |

Removing the missingness indicators entirely FIXES the V1828 Aql
prediction (0.997/0.798 -> 0.452) but WORSENS average cross-validated
performance (accuracy 0.585 -> 0.547, AUC 0.691 -> 0.545, close to chance).
This is not a coincidence and traces directly to Check 1's finding:
**34.6% of planets in this sample are excluded by the n_epochs<4 floor
vs. 20.0% of false positives** -- i.e., "a per-epoch concept is missing"
is itself weakly correlated with the planet label in this specific small
draw, for a real but incidental reason (longer-period systems are
more likely to still be confirmed planets under follow-up, and produce
fewer epochs in a fixed baseline -- see Check 1). A model with access to
the missingness indicators can partially exploit that correlation to
improve AVERAGE accuracy on this draw, at the cost of applying the wrong
prior to any target whose missingness arises from a DIFFERENT mechanism --
exactly V1828 Aql's case, where C4 is missing because of period aliasing
(BLS locking onto the wrong period), not sparse epoch coverage.

Supporting evidence: permutation importance on the full (indicator-
inclusive) MLP ranks `C4_missing` and `C6_missing` as the two most
important features (+0.069 each), ahead of every substantive concept
value including C1 (depth, +0.039) and C7 (phase-fold coherence, +0.018)
-- the model is leaning more on *whether* a concept could be measured than
on *what* it measured.

**Conclusion: the missingness-indicator design (Check 2) is not wrong in
principle** -- it is the right way to expose "this concept could not be
reliably measured" to the classifier, consistent with the informative-
missingness philosophy adopted for the data itself. **But at N=51, the
indicators are entangled with a selection effect (Check 1) that turned out
to be an artifact of that one small depth-only-stratified draw -- see the
catalog-wide check below, which settles this decisively.**

### Catalog-wide check: is the epoch-count/label confound a real survey property, or a sampling artifact?

`check_epoch_count_selection_bias.py` drew 500 TOIs (250 CP/KP, 250 FP),
stratified ONLY by disposition (not depth), and computed a cheap,
archival-only epoch-count proxy per target:
`n_epochs_proxy = (n_sectors_observed x 27.4 days) / period_days`, where
`n_sectors_observed` comes from a metadata-only MAST search (~5s/target,
no light-curve download -- a full pipeline run on 500 targets would have
taken 8+ hours, not a "cheap" check). 498/500 succeeded.

| Metric | planet median | not_planet median | Mann-Whitney p |
|---|---|---|---|
| Period (d) | 4.14 | 2.83 | 4.7e-9 |
| N sectors observed | 5.0 | 2.0 | 1.4e-20 |
| n_epochs_proxy | 23.6 | 15.0 | 9.6e-5 |
| Fraction below proxy-4 floor | **6.0%** | **24.4%** | -- |

**Result: the catalog-wide direction is the OPPOSITE of what corrupted the
51-target sample.** Planets have MORE sector coverage and a HIGHER median
epoch-count proxy than false positives, catalog-wide, at very high
statistical significance -- and only 6.0% of planets fall below a
proxy-4 floor vs. 24.4% of false positives (compare to the 51-target
sample's 34.6% planets / 20.0% FPs, the reverse ratio). This most likely
reflects real follow-up dynamics (confirmed/known planets accumulate
additional TESS extended-mission sector coverage over time; many FPs are
dispositioned and dropped after initial identification, with less
follow-up observation) -- but regardless of mechanism, **the direction
flatly contradicts what the small depth-stratified draw showed**.

**Conclusion: the 51-target sample's confound was a sampling artifact of
depth-only stratification, not a structural catalog-wide survey bias.**
Depth-decile stratification, applied independently within each class, can
by chance draw a subset where the epoch-count distribution runs backwards
from the population -- exactly what happened here. This is diagnostic,
not just reassuring: it means the fix is a resampling one (stratify by
epoch-count too), not a fundamental limitation to design around
permanently.

**Action taken:** `select_cbm_sample_v2.py` re-stratifies each class
jointly by depth quintile AND epoch-count-proxy quintile (reusing the
already-fetched 498-target pool from the catalog-wide check, so no
additional MAST queries were needed), and draws a larger sample: 80
targets (40 planet / 40 not_planet), saved to `results/cbm_sample_v2.csv`.
Post-hoc check on the new sample: planet fraction below the proxy-4 floor
= 5.0%, not_planet = 27.5% -- consistent with, not fighting against, the
catalog-wide direction, so the classifier's missingness indicators should
no longer be confounded with the label via this mechanism when concepts
are recomputed on this sample.

### Bottom line for the paper

**Report the classifier stage as a validated end-to-end CBM pipeline with
an honestly documented small-N limitation, not as a working, deployable
classifier.** The concept-computation stage (C1-C7) is confirmed correct
and stable at both the 14- and 52-target validation scales (see
`module4_concept_diagnosis.md`) -- that result stands independently of the
classifier's current performance. The classifier itself, and any
concept-weighting/interpretability claim drawn from it for Section 3.4,
should NOT be presented as final until N increases substantially (per the
report's own next-steps in Section 6.3) and Diagnosis B's confound is
re-checked on a larger, less coincidentally-correlated sample.

---

## Diagnosis C: rebalancing the confound did not fix the classifier -- and a data-integrity pass rules out a second pipeline bug

Concepts were recomputed on the full 80-target `cbm_sample_v2.csv` (81
usable rows after one failed download; 41 planet / 39 not_planet) and
both classifiers retrained with the exact same pipeline as before.

**Realized epoch-count balance (real n_epochs, not the archival proxy):**
not_planet median=11.0 (25.6% below the n<4 floor), planet median=8.0
(29.3% below). Gap narrowed from 14.6 points (34.6%/20.0% in the original
sample) to 3.7 points -- stratification worked as intended, though the
realized pipeline-measured balance is not a perfect mirror of the
archival proxy used to draw the sample (expected: the real matched-filter
epoch count has stricter requirements than the proxy's raw sector-based
estimate). TOI-700/V1828 Aql sanity pair unchanged: C7=0.9074/0.0 exactly,
at this third scale.

**Classifier result: worse, not fixed, on every metric.**

| Variant | CV accuracy | CV ROC-AUC | V1828 Aql P(planet) |
|---|---|---|---|
| MLP + indicators | 0.425 +/- 0.127 | 0.490 +/- 0.210 | 0.9557 (WRONG) |
| LogReg + indicators | 0.450 +/- 0.108 | 0.491 +/- 0.197 | 0.7151 (WRONG) |
| LogReg, no indicators (the ablation that scored 0.452/correct on the old sample) | 0.475 +/- 0.075 | 0.470 +/- 0.130 | 0.6435 (WRONG) |

Every variant, including the one that previously worked, now misclassifies
V1828 Aql, and mean CV accuracy for all three sits AT OR BELOW the 51.3%
majority-class baseline. This rules out "the epoch-count confound was the
whole problem": fixing it did not fix the classifier, and the specific
fix that worked before (dropping the indicators) no longer does either --
strong evidence that fix was itself a coincidence of the old sample's
particular composition, not a real, generalizable correction.

### Data-integrity pass (four checks, run before trusting this result)

Given the exact same spot -- "the classifier looks broken" -- previously
concealed a real bug (a failed download polluting every CV fold with NaN
in C1/C2, which should never be missing; caught and fixed before the
numbers above were produced), a second-bug check was run before writing
this up as a real finding rather than an artifact.

1. **Join integrity (`cbm_sample_v2.csv` labels vs. computed concept
   rows):** 80/80 unique targets on both sides, zero duplicates, zero
   orphaned rows, **zero label mismatches** across the join. 8 random
   joined rows spot-checked by eye (target / TIC ID / label / period all
   consistent). Clean.
2. **CV fold stratification:** confirmed `StratifiedKFold`, not plain
   `KFold`. Actual per-fold test-set class counts: 9/16, 8/16, 8/16, 8/16,
   8/16 planet -- every fold within one sample of an exact 50/50 split.
   Clean; not the "3/13-style unstratified imbalance" failure mode this
   check was designed to catch.
3. **Feature scaling / imputation leakage:** `StandardScaler().fit(X_tr)`
   fits on the training fold only, applied via `.transform()` to both
   train and held-out fold; imputation medians are likewise computed from
   the training fold only and reused (not refit) on the held-out fold.
   Verified directly in `train_cbm_classifier_v2.py`. Clean.
4. **Diagnostic floor -- single-feature (C1 depth) classifier, same CV
   setup:**

   | fold | accuracy | roc_auc |
   |---|---|---|
   | 0 | 0.562 | 0.730 |
   | 1 | 0.500 | 0.359 |
   | 2 | 0.562 | 0.516 |
   | 3 | 0.562 | 0.453 |
   | 4 | 0.562 | 0.375 |
   | **mean** | **0.550 +/- 0.025** | 0.487 +/- 0.134 |

   Beats the 0.5125 majority baseline, consistently (note the tight
   +/-0.025 std vs. the 12-feature models' +/-0.11-0.13 swings), with a
   physically correct coefficient sign (higher depth -> lower P(planet)).
   AUC is weak (~0.49, near chance) so C1 alone is not a good classifier
   either -- but it is clearly not broken.

**Interpretation of check 4, per the pre-registered decision rule:** C1
alone beats baseline while the full 12-feature model does not -- the
signature of overfitting/feature interactions on too little data, not a
pipeline bug (the alternative -- C1 alone also failing to beat baseline --
would have pointed back to a hidden bug). Combined with checks 1-3 coming
back clean, there is no remaining candidate explanation for a hidden
alignment or leakage bug.

### Conclusion

**The classifier-stage result is real, not an artifact: N=80, with a
12-dimensional feature vector (7 concepts + 5 missingness indicators),
overfits badly enough to underperform a single weak feature used alone.**
This is a normal, well-characterized small-sample-size failure mode, not
a defect in the concept-computation pipeline (which remains validated,
independently, at three scales now: 14, 52, and 81 targets) or in the
training code (four-point integrity pass, all clean). Rebalancing the
epoch-count/label confound (Check 1/2 and the catalog-wide check above)
was the right thing to do and is documented as such, but it was
addressing a real, secondary problem, not curing the primary one, which
is simply insufficient N for 12 input dimensions.

**Recommendation, unchanged from Diagnosis A/B and now on firmer footing:**
Module 4's classifier stage and any Section 3.4 concept-weighting claim
should be reported as designed-and-exercised-end-to-end on a hand-
assembled prototype sample, explicitly not yet trained to a reportable
standard, pending the full ISRO/PRL-curated dataset (report Section 6.3).
No further tuning of this specific N=80 sample is expected to change that
conclusion, since the diagnostic floor shows the ceiling here is set by
sample size relative to feature count, not by a fixable bug or a wrong
missingness design.
