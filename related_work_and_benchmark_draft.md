# Draft additions for VYOMANETRA_Report.pdf -- journal revision

Two pieces, meant to be integrated into the LaTeX source (not the PDF -- point
me at the .tex when you have it and I'll merge these in properly with real
citation keys/bibtex rather than inline text).

---

## 1. New section: Related Work (insert before or as part of Section 4)

The exoplanet transit-vetting literature has moved rapidly since this
project's initial idea submission (July 2026) and now includes several
directly comparable systems worth situating VYOMANETRA against explicitly,
rather than only citing the single-transit problem in the abstract.

**Classification-based vetting.** The dominant paradigm treats vetting as
binary classification on phase-folded light curves. AstroNet (Shallue &
Vanderburg 2018) pioneered a two-column CNN over global/local views. ExoMiner
(Valizadegan et al. 2022) extended this with multiple diagnostic branches,
achieving AUC 0.98 and validating 301 new Kepler planets; ExoMiner++
(Valizadegan et al. 2025) adapted it to TESS's 147,568 TCEs. RAVEN
(Hadjigeorghiou et al. 2025) used Bayesian gradient-boosted trees on synthetic
false-positive scenarios (AUC > 0.97). ExoNet (Islam 2026) added calibrated
multimodal fusion of phase-folded flux with stellar parameters. All of these
require phase-folded input with a known period, which is precisely the
structural limitation Section 3.2/4 (Module 2) targets.

**Single-transit / non-phase-folded detection.** The closest peer work to
VYOMANETRA's Module 2 upgrade is ExoVeil (Priyanshu 2026, arXiv:2606.02778), a
causal Transformer world-model trained with transit-masked self-supervised
learning on Kepler light curves, paired with matched-filter detection on the
prediction residuals. ExoVeil reports AUC 0.938 on Kepler DR25, 32% single-
transit recovery at 1000 ppm, and 100% zero-shot recovery on 47 confirmed TESS
planets without retraining. Also relevant: Hansen & Dittmann (2024), a CNN
ensemble using onboard Kepler spacecraft diagnostics (>80% single-transit
recovery to 800-day periods, but coupled to Kepler-specific telemetry); Vivien
et al. (2025)'s Panopticon, a U-Net++ segmentation model for single-transit
detection on simulated PLATO data; Salinas et al. (2025), a Transformer over
TESS full-frame images identifying 214 candidates; and TransitNet (2026,
arXiv:2606.18932), an attention-augmented framework for low-SNR blind
searches.

**Positioning VYOMANETRA.** ExoVeil's own limitations section explicitly
states: "A linear-complexity backbone (e.g., Mamba) would enable processing
full 65,000-point Kepler light curves in a single pass, potentially improving
both prediction quality and detection sensitivity" -- naming exactly the
architectural direction VYOMANETRA's Module 2 upgrade pursues. No system
surveyed above combines a linear-complexity state-space backbone with
transit-masked self-supervision, nor integrates a Concept Bottleneck Model for
physically-grounded interpretability (Section 3.4) -- ExoVeil's own
"explainability" layer is limited to aleatoric/epistemic uncertainty
categories, not concept-to-physics mapping.

---

## 2. New section: Independent Benchmark of ExoVeil (insert after Section 5,
before Discussion)

To ground Module 2's motivation empirically rather than by literature-gap
argument alone, we ran the publicly released ExoVeil package
(`exoveil==0.2.1`, pip-installable, pretrained weights) against our own
BLS/trapezoid baseline (Modules 1-3) on identical, independently-detrended
TESS flux.

**Four-target reproduction.** On the same four targets as Section 5, ExoVeil
was run via its `detect_from_array()` inference API. Two findings stand out.
First, the shipped v0.2.1 package exposes only the raw world-model +
matched-filter event list; the XGBoost classifier and conformal-prediction
stage described in Priyanshu (2026) have no reachable entry point in the
public API. This is a reproducibility gap between the published method and
the released package, not evidence against ExoVeil's full pipeline as
described. Second, on V1828 Aql (the eclipsing binary), ExoVeil's top-ranked
event carried a raw SNR of ~1.09x10^7 -- a numerical artifact of its
local-MAD noise estimator collapsing toward zero on the EB's sharp,
near-total eclipse, not a real detection significance.

**25-target confirmed-planet benchmark.** To move beyond a 4-target anecdote,
we drew a depth-stratified sample of 25 confirmed TESS-discovered planets
from the NASA Exoplanet Archive and scored both pipelines against the
archive's own ephemeris (period, epoch, duration) as ground truth. BLS/
trapezoid recovered the correct transit epoch in 5/25 targets (20%, +17
percentage points over a 3% chance floor for a single-epoch test). ExoVeil,
scored as "recovered if >=1 of its returned events (up to 20 per target)
falls in-transit," recovered 9/25 (36%) -- but because ExoVeil returns up to
20 independently-scored candidates per target with no period information,
the probability of a coincidental in-transit hit compounds across events. The
expected chance-level recovery rate under this scoring, 1-(1-T14/P)^N summed
over the sample, is 33% (8.2/25) -- meaning ExoVeil's actual 36% is only +3
percentage points over coincidence, statistically indistinguishable from
chance. This chance-correction is essential: naively reporting ExoVeil's raw
36% against BLS's 20% would misleadingly suggest ExoVeil outperforms BLS,
when the opposite is closer to true once each pipeline's number of "guesses"
per target is accounted for.

The same run surfaced a second, independent finding: 10/25 targets (40%,
spanning depth deciles across the full range, not only eclipsing binaries)
produced a degenerate top-event SNR from the same local-MAD collapse observed
in the V1828 Aql case -- indicating this is a systemic numerical-stability
defect in the shipped implementation rather than an edge case specific to
non-planet signals.

**Interpretation.** These results should be read as characterizing the
publicly reproducible portion of ExoVeil's pipeline, not as a claim that
ExoVeil's full published method (with its classifier and conformal stages
intact) performs at chance -- that stage is untestable from the released
package. What the benchmark does establish is a concrete, quantified case for
Module 2's planned upgrade: the strongest publicly available alternative to
BLS for single-transit detection has a reproducible implementation that
(a) does not clearly outperform simple BLS once scored fairly, and (b) has a
measurable numerical-stability defect affecting close to half of real TESS
targets tested. This motivates VYOMANETRA's Module 2 design choice to expose
calibrated, per-candidate significance (via the planned bootstrap FAP /
conformal prediction in Module 5) rather than an uncorrected multi-candidate
SNR list.

---

## Sources for the above (add to .bib)

- Priyanshu, P. (2026). One Transit Is All You Need: Detecting Exoplanets
  Through Learned Stellar Behaviour with ExoVeil. arXiv:2606.02778.
- Shallue, C. J. & Vanderburg, A. (2018). AJ, 155, 94. (AstroNet)
- Valizadegan, H. et al. (2022). ApJ, 926, 120. (ExoMiner)
- Valizadegan, H. et al. (2025). AJ, 170, 287. (ExoMiner++)
- Hadjigeorghiou, A. et al. (2025). arXiv:2509.17645, submitted to MNRAS. (RAVEN)
- Islam, M. R. (2026). arXiv:2604.15560. (ExoNet)
- Hansen, M. T. & Dittmann, J. A. (2024). AJ, 168, 291.
- Vivien, H. G. et al. (2025). A&A, 694, A293. (Panopticon)
- Salinas, H. et al. (2025). MNRAS, 538, 2031.
- TransitNet (2026). arXiv:2606.18932.
- NASA Exoplanet Archive (pscomppars table) -- benchmark sample source.
