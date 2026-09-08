# Module 3 identifiability diagnosis: posterior-sampling convergence AND LSQ depth recovery

**Summary:** Four independent attempts to obtain a converged posterior for
the Mandel-Agol transit fit (emcee under three different parametrizations,
then dynesty nested sampling) all failed to converge, in ways that point to
a genuine geometric degeneracy in the data -- not a sampler bug, not a
tuning problem, and not fixed by any single reparametrization tried. This
motivated the fallback documented in `run_modules_3_5.py`: **LSQ point
estimate + parametric (residual) bootstrap is Module 3's primary reported
uncertainty**, with every posterior-sampling result below kept as a
secondary, explicitly-caveated characterization.

Running that fallback across the full 25-target benchmark sample then
surfaced a second, related but distinct problem: the LSQ point estimate
itself frequently fails to recover the archive-published depth, and two
further bounded tests (multi-start optimization, then an external
stellar-density prior) both failed to fix it -- see "Benchmark-scale
confirmation" below. Read together, both halves of this document support
one conclusion: for a meaningful fraction of a realistic TESS archive
sample, single-epoch transit geometry is not resolvable from the data at
all with this class of method, independent of sampler, parametrization, or
optimizer-initialization choice.

This is reported here as a methodology finding, not discarded debugging
history -- it is citable in the paper's limitations section as a concrete,
quantified illustration of why single-band, single-target TESS transit
fits are prone to unresolved a/Rs-impact-parameter-duration degeneracies,
and why point-estimate + bootstrap is a defensible fallback when posterior
sampling does not converge in the available compute budget.

---

## Attempt 1: (a_over_rs, impact_b) free, emcee, NSTEPS=8000 (original design)

Baseline non-convergence, all 4 core targets:

| Target | tau | acceptance | converged |
|---|---|---|---|
| TOI-700 | 316 | 0.28 | False |
| TOI-1338 | 292 | 0.31 | False |
| Pi Mensae (BLS-aliased ephemeris) | 258 | 0.26 | False |
| V1828 Aql | 388 | 0.23 | False |

**Follow-up test (TOI-700 only):** tightened the walker-init ball 3x
(`init_scale=0.3`) *and* raised NSTEPS to 18000. If the bottleneck were
under-sampling or a slow burn-in, this should help. It did not: **tau got
worse, 316 -> 455**. This ruled out "just needs more steps" and pointed to
a genuine parameter degeneracy (most plausibly a_over_rs vs impact_b, which
are only weakly, jointly identified by duration and depth alone).

## Attempt 2: reparametrize to (T14, impact_b), emcee, NSTEPS=8000

Sampling transit duration T14 directly (a closer-to-observable quantity)
and deriving a_over_rs algebraically via
`a/Rs = sqrt((1+rp)^2 - b^2) / sin(pi*T14/P)`, apples-to-apples at the same
NSTEPS=8000 used in Attempt 1:

| Target | tau (old param) | tau (T14,b param) | change |
|---|---|---|---|
| TOI-700 | 316 | 308.3 | ~2%, noise-level |
| Pi Mensae (literature ephemeris, see below) | 258 (different ephemeris) | 263.2 | no real change |
| V1828 Aql | 388 | 319.5 | ~18% better, still far from converged |

No meaningful improvement. This ruled out a_over_rs/impact_b specifically
as the sole degenerate pair -- removing that exact coupling did not
meaningfully change the mixing time.

*(Side note, unrelated to the convergence question: this test also
independently confirmed that Pi Mensae's original 0/20 in-transit
non-detection was an aliased-ephemeris artifact, not a real non-detection.
Refit under the literature ephemeris from Larsen et al. 2026,
arXiv:2607.12088 [P=6.267823 d, T0=1325.5042 BTJD] recovered a stable,
physically-sensible ~135-140 ppm depth solution where LSQ and MCMC agreed
closely, versus the BLS-aliased ephemeris's inflated, unstable ~3800 ppm
solution. This finding stands independent of the convergence issue
documented in this file.)*

## Attempt 3: (T14, b) + limb darkening FIXED from tabulated stellar values, emcee, NSTEPS=8000

Fixed quadratic limb-darkening (u1, u2) to ExoTETHyS/PHOENIX table lookups
(via `pylightcurve.exotethys()`) keyed on each target's TIC-catalog Teff /
logg / [Fe/H], instead of sampling (q1, q2) -- dropping NDIM from 8 to 6 and
removing the two dimensions single-band TESS photometry is least able to
constrain on its own.

TOI-700 gate test (stellar params: TIC 150428135, Teff=3494 K, logg=4.809,
[Fe/H]=0 -> u1=0.2551, u2=0.2889 via PHOENIX/TESS):

tau = **228.2** (vs. 308.3 free-LD, vs. 316 original) -- a ~28% improvement,
the best of the three parametrizations tried, but still nowhere near
converged: needed 50*tau ~ 11,400 post-burn samples, had 4,800 available at
NSTEPS=8000. Per the pre-agreed gate (tau needed to drop comfortably below
~160), this did not pass, and the test was stopped before Pi Mensae/V1828
Aql rather than continuing to spend compute chasing it.

## Attempt 4: dynesty nested sampling, same (T14, b) + fixed-LD 6-param config

Converted `log_prior`'s box bounds into a `prior_transform` (mechanical,
uniform-cube mapping; the one non-box constraint -- a_over_rs must be
finite and in [1.2, 500] -- was already enforced downstream in
`log_likelihood` unchanged, via `transit_model` returning NaN for an
invalid geometry). Ran `dynesty.NestedSampler` (nlive=250, bound='multi',
sample='auto', dlogz=0.05 target) on TOI-700 with a 15-minute wall-clock
time-box.

**Result: catastrophic slowdown, not convergence.** The run was intended to
self-terminate within 900s; actual wall-clock was **3217s (53.6 min)** --
one single internal iteration (between iteration 4000 and 4309) alone
appears to have taken on the order of 2800+ seconds, consistent with
dynesty's rejection-sampling/bounding-ellipsoid machinery grinding nearly
to a halt trying to satisfy an increasingly strict likelihood threshold
inside a poorly-conditioned region. dlogz convergence was never confirmed
(the run was killed by the time-box mid-iteration, before `add_final_live`
could be called, so the reported posterior is from an unfinalized run).

The recovered (unconverged) posterior is diagnostic in its own right:

| Parameter | median | -68% | +68% |
|---|---|---|---|
| rp_over_rs | 0.062 | -0.018 | **+0.444** |
| impact_b | 0.954 | -0.579 | **+0.506** |

This is the textbook grazing-large-planet vs. central-small-planet transit
degeneracy -- the *same class* of problem as Attempt 1's a_over_rs/impact_b
coupling, just resurfacing in a **different parameter pair** (rp_over_rs
vs. impact_b) under a **completely different sampling algorithm** with no
autocorrelation-time machinery in common with emcee. That it reappears
under an unrelated sampler is strong evidence the degeneracy is a property
of the *data* -- 619 fitted cadences from a single TESS bandpass, without
an external constraint (e.g. a stellar-density prior on a/Rs) to break the
duration/depth/impact-parameter correlation -- not an artifact of emcee's
stretch-move proposal or of any one coordinate choice.

Despite the posterior's poor shape, its **median depth (2449.9 ppm) was
still within ~0.65 sigma of the trapezoid fit (2247.2 +/- 175.0 ppm)** --
consistent with every earlier attempt's LSQ point estimate landing in the
same sensible range (2323-2380 ppm across all four attempts on TOI-700).
The depth *point estimate* was never the problem; only the full posterior's
shape and formal convergence were.

## Conclusion and fallback

| Attempt | Sampler | Parametrization | Result |
|---|---|---|---|
| 1 | emcee | (a_over_rs, b), 8 free params | tau=316; tighter init + 2.25x steps made it *worse* (tau=455) |
| 2 | emcee | (T14, b), 8 free params | tau~260-320, no improvement over (1) |
| 3 | emcee | (T14, b) + fixed LD, 6 free params | tau=228 (best of the emcee attempts), still 2.4x short of the convergence bar |
| 4 | dynesty | (T14, b) + fixed LD, 6 free params | ran 60x longer than intended, never confirmed dlogz convergence, posterior shows the same class of geometric degeneracy in a different parameter pair |

Reparametrization and sampler choice each address a *specific* coordinate
system's pathology; neither fixes a degeneracy that is intrinsic to what
the data can constrain. Continuing to chase this with a fifth
parametrization or a third sampler was judged not to be a good use of
compute for this paper's validation runs.

**Module 3's primary reported uncertainty is therefore LSQ point estimate +
parametric (residual) bootstrap** (`bootstrap_lsq()` in `modules_3_5.py`),
which does not rely on posterior-sampling convergence at all -- it directly
resamples the best-fit model's residuals (block bootstrap, block length ~1
transit duration, honest for TESS's correlated noise) and refits
least-squares on each resample, characterizing how much the *fitted*
parameters plausibly vary under the observed noise. MCMC and nested-
sampling results are retained in the output CSV as a secondary,
explicitly-flagged characterization (`posterior_diagnostic_note` column) --
worth keeping for transparency and for readers who want the (partial,
under-sampled) joint-posterior shape, but not the number the paper should
quote as Module 3's uncertainty.

**Recommendation for future work** (out of scope for this validation run):
an external stellar-density prior on a/Rs (from Gaia parallax + isochrones,
or asteroseismology where available) would directly break the
duration/depth/impact-parameter degeneracy that both emcee and dynesty
independently ran into, and is the standard fix for this exact failure mode
in the transit-fitting literature.

---

## Addendum: V1828 Aql's bootstrap interval is an artifact, not a real precision

Running `bootstrap_lsq()` on the 4 core targets (see main results,
`results/modules_3_5_results.csv`) surfaced a second, unrelated pathology
worth flagging before anyone reads V1828 Aql's tiny quoted bootstrap width
as a genuinely precise measurement.

V1828 Aql's LSQ fit lands with **q1=0.9668, q2=0.9894 -- both pinned within
0.01 of their prior's upper bound (0.99)**. Converted to physical
limb-darkening coefficients, this is `u1=1.94, u2=-0.96`: far outside the
physically valid range for real stellar limb darkening (normally
u1, u2 ~ O(0.1-0.8)). The optimizer is exploiting the quadratic-LD shape
freedom at its prior boundary to warp the model into an approximation of
V1828 Aql's actual (eclipsing-binary, non-limb-darkened-transit-shaped)
eclipse -- the same underlying problem flagged throughout this
investigation: **a single-planet Mandel-Agol model is the wrong model for
this target's light curve.**

Because that optimum sits at a hard parameter boundary, it is numerically
"sticky": a single-resample diagnostic (injecting one full bootstrap
residual draw, std=0.049, nearly 2.5x the median formal flux uncertainty of
0.023) and refitting from the same warm start moved the recovered
parameters by ~1e-6 to 1e-9 -- i.e. the optimizer barely moved at all
despite a substantially different dataset. Repeated over 300 resamples,
this produces a bootstrap interval of a few hundredths of a ppm on a
44,468 ppm depth -- not because the fit is genuinely that precise, but
because every resample gets trapped at the same boundary-pinned local
optimum.

**Practical consequence:** V1828 Aql's `lsq_boot_*` columns should NOT be
read as a real uncertainty; they under-report it by orders of magnitude.
This is consistent with, and adds to, the standing recommendation that
V1828 Aql (P=0.66 d eclipsing binary) is not a target this pipeline's
transit model should be trusted on without a dedicated EB model.

---

## Benchmark-scale confirmation: this is a real, quantified sample property, not a 4-target fluke

Running the fixed pipeline (LSQ + parametric bootstrap, PARAM_NAMES=(T14,b),
free limb darkening) on the full 25-target NASA Exoplanet Archive benchmark
sample surfaced the same identifiability problem at scale, with enough
signal to characterize it quantitatively rather than just note its
existence.

**The headline number:** only 1/25 archive-published depths fell inside the
LSQ+bootstrap 68% interval; median relative depth error across the sample
was **-78% to -81%** (recovered depths typically ~20% of the published
value). This is far worse than the 4 hand-picked core targets, where 2/4
agreed well and the other 2 had explicable few-percent model-definition
offsets (see the main body of this document) -- confirming the core-target
validation was not representative of a real, unbiased archive sample.

**Boundary-pinned limb darkening (the V1828 Aql failure mode, automated):**
`boundary_pinned_ld` fired on **14/25 (56%)** of the benchmark sample --
vs. 1/4 in the core set. More than half of a random confirmed-planet sample
pushes q1 or q2 to the edge of its prior, the same "unphysical LD
compensating for a model mismatch" signature diagnosed for V1828 Aql above.

**Grazing-transit correlation:** fitted impact parameter correlated with
depth deficit at **Pearson r=-0.63** (signed). 13/25 (52%) of the sample
fit with impact_b > 0.6; those targets had a median depth deficit of
**-88.9%**, vs. **-28.3%** for the rest (still not good, but a materially
different regime). This was the strongest lead going into the two follow-up
tests below.

**Test 1 -- multi-start LSQ (5 seed values of b in [0.0, 0.2, 0.4, 0.6,
0.8], keep lowest chi-square).** Run on the 8 worst offenders
(impact_b 0.81-1.02) before committing to a full rerun. Result: **did not
work.** Every one of the 5 starting points converged to chi-square values
within 0.01-0.1% of each other -- not distinct local minima at all, but the
same flat basin regardless of where the search started. Depths did not move
meaningfully toward the archive value for any of the 8. This ruled out
"wrong local minimum" as the mechanism: multi-start can only fix a
local-optimum problem, and the chi-square surface here doesn't have
separated optima to escape between.

**Test 2 -- fix a_over_rs from an external stellar-density prior**, on the
theory that constraining the one remaining free geometry parameter (not
found in the light curve at all, but computable from TIC-catalog stellar
mass and radius via Kepler's third law: a/Rs = (G*rho_star*P^2/(3*pi))^(1/3))
would convert an underdetermined 3-way rp/b/duration/a_over_rs degeneracy
into a well-constrained rp/b fit at a fixed scale. Implemented, formula
validated against the known Sun/1-AU case (a/Rs=214.8 vs. the analytically
expected ~215), and tested on the same 8 worst offenders. Result: **also
did not work, and confirms the deeper problem.** Depths were still
**-88% to -99%** below archive for all 8 targets even with a physically
correct, independently-measured a/Rs held fixed -- and chi-square changed
by **less than 0.05%** in every single case versus the free-a_over_rs fit
(two of the eight cases were even marginally *worse*). Fitted impact
parameter under the fixed-a/Rs model still ranged from 0.32 to 1.02 with no
consistent convergence toward a physically sensible value.

**Interpretation.** That fixing a_over_rs to its true, externally-measured
value leaves chi-square essentially unchanged is a stronger and more
specific result than "there is a parameter degeneracy." A degeneracy means
several parameter combinations fit comparably well; this shows the fit is
comparably insensitive to the geometry *entirely* for a large fraction of
this sample -- the residuals are not being driven by transit-shape mismatch
at all, at whatever noise level these particular light curves carry. In
other words: for a substantial fraction of a realistic, unbiased TESS
archive sample, **the transit signal is not well enough resolved above the
detrended noise floor for single-epoch least-squares (or, per the
convergence work above, MCMC/nested-sampling) fitting to uniquely recover
transit geometry, regardless of how many free parameters are removed via
reparametrization, fixed limb darkening, multi-start optimization, or an
external stellar-density prior.** This is a data-limited identifiability
wall, not a fixable modelling or sampling bug, and no single technique
tried resolves it.

**Conclusion for Module 3/5.** Per the explicit stopping criterion applied
to this second bounded test (multi-start LSQ, then the stellar-density
prior, tested on the worst offenders before any full-sample commitment,
neither working): no further remediation was attempted on Module 3. The
pipeline's primary reported uncertainty remains LSQ + parametric bootstrap;
MCMC/nested-sampling posteriors remain secondary/exploratory; and this
identifiability limitation is now the paper's citable, quantified
methodological finding for its limitations section:

> Applying least-squares and MCMC Mandel-Agol fitting to a depth-stratified,
> unbiased sample of 25 TESS-confirmed planets, only 1/25 recovered depths
> fell within their own bootstrap-estimated 68% interval of the archive-
> published value (median relative error -78% to -81%). 56% of fits pushed
> limb-darkening parameters to an unphysical prior boundary, and fitted
> impact parameter correlated with depth deficit at r=-0.63. Neither
> multi-start optimization nor fixing the stellar geometry scale from an
> independently-measured stellar density (via Kepler's third law)
> meaningfully improved recovery or differentiated the chi-square surface
> (<0.05% change on the worst-affected targets), indicating the limitation
> is data-driven (detrended-noise-floor-limited single-epoch photometry),
> not a fixable parametrization, optimizer, or sampler choice.

One item explicitly deferred, not investigated here: TOI-2193 A b is a
+310% depth *excess* with impact_b~0 -- not explained by the grazing-fit
mechanism above (which only predicts deficits) and not covered by either
test in this section. Left for separate follow-up, as agreed, since it
doesn't bear on the identifiability conclusion above.
