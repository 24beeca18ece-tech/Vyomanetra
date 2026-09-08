# Module 2: right-sized SSM world model -- scope, architecture, and memory validation

## Why this is scoped down from the guidance document

`Implementation Guidance for State-Space Models in Exoplanet Transit
Detection.pdf` recommends a 25-50M parameter model trained on the full
Kepler catalog (~200,000 stars) plus 1,000,000 TESS targets (75-100 billion
tokens total), on a 400-600 GPU-hour budget requiring an 8-GPU A100 (80GB)
node. That is correct guidance for a well-resourced institutional project;
it is not achievable on the hardware actually available here: a local RTX
4060 (8GB VRAM) for development, and a $300 Google Cloud credit for the
real training run. At current on-demand pricing, 400-600 GPU-hours on
anything resembling an A100 would cost from the low thousands to tens of
thousands of dollars -- multiple orders of magnitude past budget.

This document covers a proof-of-concept, right-sized to what the available
compute can actually support: model scale close to ExoVeil's own reported
3.2M-parameter, 6-layer, D=192 configuration (which achieved AUC 0.938) as
the starting point, not the guidance doc's 25-50M recommendation; a
training corpus in the hundreds-to-low-thousands of light curves, not
200K+1M; single-sector (~20K point) sequences for local dev, not
256K-point multi-sector stitching.

## What was kept from the guidance document (correct, and free)

1. **Contiguous block masking** (30-45% ratio, block size 100-600
   cadences), not random pointwise masking. Random pointwise masking lets
   the model trivially interpolate a transit from its unmasked in-transit
   neighbors (2-min-cadence flux is highly autocorrelated), reconstructing
   the dip perfectly and hiding it from the residual detector at
   inference -- exactly the failure mode the guidance doc identifies.
   Implemented in `contiguous_block_mask()`.
2. **Delta-t-aware discretization**: `Abar = exp(dt*A)`,
   `Bbar = (dt*A)^-1 (Abar - I) B`, using the REAL physical observation
   gap (from BJD/BTJD timestamps), not a purely learned step size. This
   means a 2-day TESS downlink gap physically decays the latent state
   instead of corrupting phase alignment. Masked tokens are RETAINED in
   the sequence (not removed) with a learnable `[MASK]` embedding, per the
   guidance doc's reasoning: removing tokens would alter the physical time
   vector and destroy phase alignment of periodic stellar signals.
3. **Both a Mamba variant (selective: B, C, and the dt-scale factor are
   input-dependent) and an S4D variant (diagonal: A/B/C fixed, only the
   Delta-t-aware discretization varies with the real time gap)**, so the
   Mamba-vs-S4D comparison (arXiv:2605.27406 found S4D can beat Mamba on
   stationary time series) is a real, apples-to-apples result for the
   paper regardless of which wins.

## What was explicitly NOT implemented (not in the user's keep-list)

The wavelet dual-branch OOD-extrapolation augmentation (Section 5 of the
guidance doc) and the learned-variance detection head (Section 6) are out
of scope for this proof-of-concept -- neither was requested, and both add
real implementation and compute cost without being necessary to validate
the core SSM-vs-Transformer, Mamba-vs-S4D questions this PoC targets.

## No custom CUDA kernel

The official `mamba-ssm` package requires compiling custom CUDA kernels
via `nvcc`/`ninja`, well-supported on Linux, not on native Windows without
a full build-tools setup -- the same class of problem that blocked
`batman-package` earlier in this project. The selective/diagonal
recurrence here is implemented as a pure-PyTorch chunked linear scan
(`dt_aware_chunked_ssm` in `modules_2_ssm.py`), trading some wall-clock
speed against the hardware-aware CUDA scan for portability. This is a
documented tradeoff, not a hidden one.

## Memory validation: the naive implementation did NOT fit in 8GB -- reported immediately, then fixed properly

Per the explicit instruction to report an over-budget result immediately
rather than work around it silently, here is the actual sequence of
findings, in order.

### Finding 1: naive chunking OOMs at 14GB+, even at batch=1

The first implementation "chunked" only the cumulative-sum step inside the
scan, but the CALLER still materialized the full `(batch, L, d_inner, N)`
tensor (`Abar`, `Bbar_x`, etc.) before calling it. At the target scale
(2.92M-parameter Mamba: d_model=192, 6 layers, mamba_expand=4 ->
d_inner=768, d_state=16; L=19,728, a single TESS short-cadence sector,
per the guidance doc's own Table 2) this attempted to allocate 14.25GB+ at
**batch=1** -- catastrophically over an 8GB budget. Reported immediately,
not worked around.

### Finding 2: chunking alone (without gradient checkpointing) doesn't fix training memory

Restructuring so no `(batch, L, D, N)` tensor is ever materialized (only
`(batch, chunk_size, D, N)` per chunk) brought a single BLOCK's forward+
backward down to ~1.6-1.7GB -- but stacking 6 such blocks for the full
model still needed ~8.7-8.8GB, because autograd must retain every layer's
activations simultaneously to backprop through a 6-layer residual stack,
regardless of how small each individual layer's chunk-level footprint is.
Reducing the chunk size (256 -> 128 -> 64) did not help (8.82 -> 8.74 ->
8.74GB) and made wall-clock ~2.7x worse -- confirmatory evidence that the
bottleneck had moved from the scan chunks to the cross-layer activation
stack, not remaining chunk-size-dependent.

### Finding 3: the real fix is checkpointing at BOTH the chunk level and the layer level

Profiling isolated the cause precisely: a single block's forward+backward
peaked at ~1.65GB; 6 layers stacked without layer-level checkpointing
additively explained the ~8.7GB total. Adding `torch.utils.checkpoint`
around each `ResidualBlock`'s forward call (recomputing a layer's forward
during backward instead of retaining its activations) -- on top of the
existing chunk-level checkpointing -- dropped peak memory to **1.76-1.78GB
for Mamba, 1.37-1.40GB for S4D**, at full target scale (2.92M / 670K
params, L=19,728, batch=1), and was FASTER (5.1s vs 14.9s per step) since
avoiding large monolithic allocations also reduces allocator overhead.

### Finding 4: nested `use_reentrant=False` checkpointing triggers a real PyTorch internal bug

Testing batch-size scaling with both checkpoint levels using
`use_reentrant=False` (PyTorch's recommended default) crashed with an
internal `SavedTensorHooks` assertion error
(`is_initialized && !tls.stack.empty()`) on the batch=4 Mamba run, silently
corrupting every subsequent measurement in that process (wall-clock times
of 449s, then 294s, then 149s across supposedly-independent configs --
non-monotonic and untrustworthy; not reported as real numbers). Root
cause: nested non-reentrant checkpoints (layer-level wrapping a block that
itself does chunk-level checkpointing) hit a known class of PyTorch
bug/limitation. Fix: switched both checkpoint call sites to
`use_reentrant=True`, which does not exhibit this crash. Verified
layer-checkpoint-alone (no inner chunk checkpoint, avoiding nesting
entirely) does NOT recover the memory win on its own (12.34GB) -- both
levels are genuinely necessary, and `use_reentrant=True` is what makes
nesting them safe.

### Final validated memory/wall-clock table (target scale, L=19,728, single TESS sector)

| Backbone | Params | Batch | Peak GPU mem | Wall-clock/step |
|---|---|---|---|---|
| Mamba | 2,921,671 | 1 | 1.78 GB | 5.13 s |
| Mamba | 2,921,671 | 4 | 7.28 GB | 24.6 s |
| Mamba | 2,921,671 | 8 | **14.52 GB (!)** | **396 s (!)** |
| S4D | 670,465 | 1 | 1.40 GB | 4.37 s |
| S4D | 670,465 | 4 | 5.55 GB | 21.4 s |

**Batch=8 is a second, subtler over-budget finding worth flagging
explicitly rather than silently avoiding:** it did not raise an
out-of-memory exception, but `torch.cuda.max_memory_allocated()` reported
14.52GB on an 8GB physical card, and the step took 396 seconds (vs. ~25s
at batch=4, ~13s extrapolated linearly) -- consistent with Windows/WDDM
silently spilling GPU allocations into slower system RAM ("shared GPU
memory") rather than failing outright. This is NOT a usable configuration
even though it does not crash; treating "doesn't throw an exception" as
"fits" would have been the exact silent-workaround failure mode this task
explicitly asked to avoid.

**Conclusion: batch=4 is the practical ceiling for local dev at this
sequence length for the Mamba backbone (7.28GB, comfortable margin against
the ~7.2GB actually free after other processes' ~1GB baseline usage);
batch=1-2 is the safe default for fast iteration. Both backbones fit
comfortably in 8GB at the target ~3.2M-parameter scale and ~20K sequence
length once BOTH checkpointing levels are applied correctly** -- the
starting point the user asked to validate before considering any scale-up
is confirmed to work, but only after finding and fixing a real memory bug
along the way, not on the first attempt.

## GPU environment note

The local Python environment initially had a CPU-only PyTorch build
(`torch==2.13.0+cpu`) despite the RTX 4060 being present and driver-ready
(driver 595.95, CUDA 13.2 supported). Reinstalled `torch==2.11.0+cu128`
(matching the installed driver's CUDA capability and the environment's
Python 3.14) before any of the above testing was possible. This is a local,
reversible environment fix, not a cloud/billing action.

## Kepler corpus preprocessing bug (caught before it corrupted the corpus)

`download_kepler_corpus.py`'s first version reused `prototype.py`'s
Savitzky-Golay `flatten()` call from Module 1's detrending step. That is
wrong for a self-supervised world model: a ~301-cadence SG window at Kepler
long cadence is a ~6.3-day high-pass filter, which would strip out exactly
the multi-day stellar-rotation/spot variability the model is supposed to
learn to forecast, before it ever saw the data -- training a model whose
entire pretext task is "predict what the star's brightness does" on flux
that's already had that behavior filtered out. Caught after 17 curves had
been downloaded with the flawed preprocessing; the background download was
killed, the 17 flawed `.npz` files and the manifest were deleted, the script
was fixed to `lc.normalize().remove_outliers(sigma_upper=5,
sigma_lower=1e4).remove_nans()` only (no flatten), and the corpus was
redownloaded cleanly. Final corpus: 122 Kepler DR25 KOI host stars
downloaded, 95 with >=16,384 cadences (this run's training sequence
length), 5 quarters stitched per target.

## Training results (both backbones, 500 steps, batch=2, seq_len=16384)

| Backbone | Params | Steps | First loss | Last loss | Min loss | Peak mem | Wall-clock |
|---|---|---|---|---|---|---|---|
| Mamba | 2,921,671 | 500 | 0.138680 | 0.000010 | 0.000006 | ~3.13 GB | 4759.6s (9.52s/step avg) |
| S4D | 670,465 | 500 | 2.068785 | 0.0000238 | 0.0000154 | ~2.39 GB | ~4250s (~8.5s/step avg) |

Both runs completed on the local RTX 4060, comfortably inside the 8GB
budget (well below the 1-2GB/1.4-1.8GB figures from the memory-validation
section above at batch=1, since this is real training with the AdamW
optimizer state added on top, still nowhere near 8GB).

## Investigation: does the near-zero training loss reflect real learning?

Both backbones' loss dropped to near-zero (Mamba: 0.00001; S4D: 0.0000238)
by roughly step 75-100 and stayed flat. Per the explicit instruction to
check this rather than assume either "it works" or "it's broken," a direct
test was run: load each trained checkpoint, run 11 held-out Kepler training
curves through the model with a fresh contiguous block mask (same 40%
ratio, 100-600 cadence blocks as training), and at the MASKED positions
(the only positions that received gradient signal during training) compare
model predictions against (a) the true flux and (b) a trivial constant
predictor (just the window's own mean).

| Backbone | Median correlation (pred vs. true, masked) | Median MSE ratio (model / trivial-mean) | Median pred_std / true_std |
|---|---|---|---|
| Mamba | 0.0054 | 67.1x **worse** | 7.47x |
| S4D | 0.0143 | 161.5x **worse** | 12.66x |

**Finding: neither backbone has learned genuine fine-grained
stellar-variability forecasting at this proof-of-concept scale.** Both
get approximately zero correlation with the true per-timestep structure at
masked positions, and both are dramatically worse than simply predicting
the window's own mean -- because both models add spurious extra variance
(pred_std 7-13x larger than the true signal's tiny intrinsic std, since
these are quiet Kepler stars with std~0.0003) that a trivial constant
predictor wouldn't. The near-zero *aggregate* MSE loss during training is
explained by the absolute scale of Kepler PDCSAP flux (values cluster
tightly around 1.0) making even a badly-uncorrelated prediction numerically
small in raw MSE terms, not by the model having learned real structure.
This is an honest limitation of training a ~3M-parameter model for 500
steps on <100 curves, not a training bug -- it is exactly the kind of
result a right-sized proof-of-concept should surface plainly rather than
paper over.

Separately, and more severely: at UNMASKED positions, predictions were
found to be wildly off-scale (e.g., true~1.0, predicted~0.15-0.23,
correlation -0.13 across trials). This is expected once traced to the
training loop: `train_ssm_world_model.py`'s loss is computed ONLY on
masked positions (`((out - flux)[mask]).pow(2).mean()`), a standard MAE
recipe, but it means the model's output at unmasked positions never
receives ANY gradient signal and is uncalibrated garbage.

## Eval-methodology bug found and fixed as a direct consequence

`eval_ssm_world_model.py`'s first version ran inference completely
UNMASKED, on the reasoning that "masking is a training-time-only
technique." The finding above proves that reasoning wrong: unmasked-output
is uncalibrated at every position, so the original eval script was reading
garbage residuals, not a meaningful transit-detection signal. Re-reading
the guidance document confirmed masking is meant to apply at inference too
("forced to predict the flux using only the unperturbed stellar history
before and after the block"). Fixed by masking each evaluation window's
transit epoch (+ padding roughly one transit duration wide, matching
training's 100-600-cadence block regime) before running inference, so the
model is forced to predict the transit region from context exactly as it
was trained to. This fix was applied and verified (syntax + a dry run)
before any eval numbers below were produced.

## Chance-corrected eval results: Mamba vs. S4D (29-target eval set, fixed masking)

Same 29 targets as the ExoVeil/BLS baseline comparison (4 core targets +
25-target depth-stratified confirmed-planet sample). 24/29 evaluable in
both runs (5 skipped: insufficient cadence count after the seq_len=16384
cut, e.g. TOI-6894 b at 14,609 points). Detection: matched-filter SNR of
the (observed - predicted) residual against a box template at the known
ephemeris/duration, threshold SNR > 5.0. Chance floor: BLS-style
single-epoch `p_single = (T14/24)/P` summed over targets (this is a
single-known-ephemeris test, not ExoVeil's ranked multi-event list, so its
multi-event chance floor does not apply here).

| Backbone | Raw recovery | Chance floor | Excess |
|---|---|---|---|
| Mamba | 3/24 (12.5%) | 0.67/24 (2.8%) | +2.33 targets (+9.7 pp) |
| S4D | 12/24 (50.0%) | 0.67/24 (2.8%) | +11.33 targets (+47.2 pp) |

**S4D substantially outperforms Mamba on the actual downstream detection
task**, despite S4D showing an *equally poor or slightly worse* raw
fine-grained forecasting correlation in the investigation above (0.0143 vs.
0.0054 correlation -- both near zero; S4D's MSE-vs-trivial-mean ratio was
actually worse, 161x vs. 67x). This is a genuinely counterintuitive result
worth flagging rather than smoothing over: neither model is a good
timestep-by-timestep forecaster yet, but S4D's residual apparently
separates real transit epochs from noise far more reliably under the
matched-filter test than Mamba's does. A plausible (not proven) explanation
is that Mamba's input-dependent (selective) B/C parameters let it partially
shape its prediction using the flux values immediately adjacent to a masked
block's boundary, subtly leaking a transit-consistent trend into the
predicted region and reducing the residual's amplitude at real transits --
whereas S4D's fixed, input-independent dynamics cannot do this and so
produce a more consistent residual regardless of what precedes the mask.
This would match the literature motivation cited for including both
architectures (arXiv:2605.27406, S4D outperforming Mamba on stationary time
series) but the specific mechanism above is a hypothesis for follow-up
work, not a demonstrated cause.

Individual target SNRs for both backbones vary in both directions (e.g.
S4D: TOI-519 b SNR=+56.9, TOI-6109 b SNR=-11.2), i.e., this is not a
trivial "everything reads positive" artifact -- the detector produces
target-specific signal in both directions for both backbones, consistent
with real (if currently limited and noisy) per-target signal rather than a
systematic constant bias.

## Context: how this compares to the Module 1 (BLS/trapezoid) and ExoVeil baselines

From `results/benchmark_summary.md` (same 25-target sample, in-range n=23):
BLS/trapezoid epoch-in-transit rate was 5/23 (22%) against a 3% chance
floor (+19pp excess); ExoVeil's raw event list was 9/23 (39%) against a 35%
chance floor (+4pp excess, but see that file's Caveat 1 -- ExoVeil's
classifier/conformal stages are not reachable via the public API, so this
is its unfiltered matched-filter front-end only). Module 2's S4D world
model's +47.2pp excess on its own (smaller, 24-target) eval set is the
largest excess-over-chance of any pipeline benchmarked in this project so
far -- but it rests on a single, un-replicated PoC-scale training run
(500 steps, <100 Kepler curves) and a 24-target sample, and the
fine-grained-forecasting investigation above shows the underlying world
model has not learned real stellar-variability prediction at this scale.
This result should be read as "the architecture and masking-based residual
approach show real promise and deserve a properly-resourced follow-up," not
as a mature, publication-grade detector result yet.

## Future work: full foundational-scale training requires institutional compute

This proof-of-concept validates the pipeline (chunked Delta-t-aware SSM
scan, two-level gradient checkpointing, contiguous block masking, both
Mamba and S4D backbones) at a scale that fits an 8GB consumer GPU and a
few-hundred-curve corpus. It does not attempt, and cannot substitute for,
the guidance document's actual proposal: a 25-50M parameter model trained
on the full ~200,000-star Kepler catalog plus ~1,000,000 TESS targets
(75-100 billion tokens), which the document estimates at 400-600 GPU-hours
on an 8x A100 (80GB) node. That is squarely future work requiring
institutional-scale compute (a research cluster or a substantially larger
cloud budget than the $300 credit available here) -- the same
honest-limitation framing already used for Module 4's classifier stage.
