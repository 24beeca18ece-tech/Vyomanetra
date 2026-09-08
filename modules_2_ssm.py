#!/usr/bin/env python3
"""
VYOMANETRA Module 2 -- right-sized SSM world model (Mamba + S4D variants).

Deliberately scoped DOWN from the guidance document's foundational-scale plan
(25-50M params, 200K Kepler + 1M TESS targets, 400-600 GPU-hours on an
8-GPU A100 node) to a proof-of-concept that fits the available hardware: a
local RTX 4060 (8GB) for development, a $300 GCP credit for the real run.
See results/module2_ssm_design.md for the full budget rationale.

Kept from the guidance doc (these are correct and free):
  (a) Contiguous block masking, 30-45% ratio, block size 100-600 cadences
      -- NOT random pointwise masking (see masking module docstring below
      for why pointwise masking lets the model trivially interpolate
      through a transit and hide it from the residual detector).
  (b) Delta-t-aware discretization: Abar = exp(dt*A), Bbar = (dt*A)^-1 (Abar-I) B
      -- so real TESS/Kepler observation gaps decay the latent state
      physically instead of corrupting phase alignment.
  (c) Both a Mamba (selective, input-dependent A/B/dt) and an S4D (diagonal,
      input-independent, FFT-convolution) variant, since arXiv:2605.27406
      found S4D can beat Mamba on stationary time series -- a legitimate
      comparison for the paper either way.

Explicitly NOT implemented (out of scope for this PoC, not in the user's
keep-list): the wavelet dual-branch OOD-extrapolation augmentation, the
learned-variance detection head, and the 256-512 model dimension / 12-24
layer scale-up. Target scale: ~3.2M parameters, 6 layers, D_model=192,
matching ExoVeil's own reported scale (which achieved AUC 0.938) rather
than the guidance doc's 25-50M recommendation.

No custom CUDA kernel (the official mamba-ssm package needs Linux + nvcc
compilation of custom kernels, the same class of problem that blocked
batman-package earlier in this project on this Windows machine). The
selective/diagonal recurrence here is a pure-PyTorch CHUNKED linear scan:
sequential over chunks of `chunk_size` timesteps (default 256), vectorized
within each chunk via the stable log-space cumulative-product identity for
a diagonal linear recurrence. This trades some wall-clock speed against the
hardware-aware CUDA scan for portability -- exactly the kind of tradeoff
documented, not hidden, per this project's established practice.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


# ============================================================ core scan ====
#
# IMPORTANT MEMORY NOTE (found empirically, not assumed -- see
# results/module2_ssm_design.md): chunking ALONE, i.e. only shrinking the
# instantaneous tensor size processed at once, does NOT bound training
# memory. Autograd must retain every chunk's intermediate tensors (dtA,
# Abar, Bbar_x, the cumsum terms, etc.) until backward() runs, so the TOTAL
# memory summed across all L/chunk_size chunks is still ~O(L) -- an early
# version of this file chunked the scan's cumsum but materialized the full
# (batch, L, D, N) Abar/Bbar_x tensors BEFORE calling it, OOMing at 14GB+
# for a single batch=1 sample at the target L=19728, D=768, N=16 scale.
# The fix below chunks the ENTIRE per-timestep computation (never forms a
# (batch, L, D, N) tensor) AND wraps each chunk's forward computation in
# torch.utils.checkpoint, so autograd recomputes a chunk's intermediates
# during backward instead of storing all of them -- true O(chunk_size)
# peak memory in the state-expansion dimension, independent of L.
def _dt_ssm_chunk_step(u_c, dt_c, h_carry, A, B_c, C_c):
    """One chunk's forward computation (checkpointed -- recomputed during
    backward, not stored). u_c: (batch,c,D). dt_c: (batch,c). h_carry:
    (batch,D,N). A: (D,N). B_c, C_c: (batch,c,N) [selective] or (D,N) [static].
    Returns (y_chunk (batch,c,D), h_last (batch,D,N))."""
    dtA = dt_c.unsqueeze(-1).unsqueeze(-1) * A          # (batch, c, D, N)
    A_bar = torch.exp(dtA)
    B_bar_scale = (A_bar - 1.0) / dtA.clamp(max=-1e-8)

    if B_c.dim() == 3:      # selective (Mamba): (batch, c, N)
        Bx = u_c.unsqueeze(-1) * B_c.unsqueeze(2)        # (batch, c, D, N)
    else:                    # static (S4D): (D, N)
        Bx = u_c.unsqueeze(-1) * B_c
    B_bar_x = B_bar_scale * Bx

    log_a = torch.log(A_bar.clamp_min(1e-12))
    a_cumprod = torch.exp(torch.cumsum(log_a, dim=1))
    carry_contrib = a_cumprod * h_carry.unsqueeze(1)
    scaled = B_bar_x / a_cumprod.clamp_min(1e-12)
    within_contrib = a_cumprod * torch.cumsum(scaled, dim=1)
    h_chunk = carry_contrib + within_contrib             # (batch, c, D, N)

    if C_c.dim() == 3:
        y_chunk = torch.einsum("bcdn,bcn->bcd", h_chunk, C_c)
    else:
        y_chunk = torch.einsum("bcdn,dn->bcd", h_chunk, C_c)
    return y_chunk, h_chunk[:, -1]


def dt_aware_chunked_ssm(u, dt_eff, A, B, C, chunk_size=256, use_checkpoint=True):
    """Fully chunked + gradient-checkpointed Delta-t-aware diagonal SSM.
    Never materializes a (batch, L, D, N) tensor -- only (batch,
    chunk_size, D, N) at a time, and (with use_checkpoint=True, the
    default) does not even retain THAT across the backward pass.

    u: (batch, L, D). dt_eff: (batch, L). A: (D, N). B, C: (D, N) [S4D,
    static] or (batch, L, N) [Mamba, input-selective]. Returns y: (batch, L, D).
    """
    batch, L, D = u.shape
    N = A.shape[-1]
    device, dtype = u.device, u.dtype
    B_selective = B.dim() == 3
    C_selective = C.dim() == 3

    y_chunks = []
    h_carry = torch.zeros(batch, D, N, device=device, dtype=dtype)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        u_c = u[:, start:end]
        dt_c = dt_eff[:, start:end]
        B_c = B[:, start:end] if B_selective else B
        C_c = C[:, start:end] if C_selective else C

        if use_checkpoint and torch.is_grad_enabled():
            y_chunk, h_carry = ckpt.checkpoint(
                _dt_ssm_chunk_step, u_c, dt_c, h_carry, A, B_c, C_c, use_reentrant=True)
        else:
            y_chunk, h_carry = _dt_ssm_chunk_step(u_c, dt_c, h_carry, A, B_c, C_c)
        y_chunks.append(y_chunk)

    return torch.cat(y_chunks, dim=1)


def chunked_diagonal_scan(A_bar, B_bar_x, chunk_size=256):
    """Retained for the unit tests / reference-correctness check against
    dt_aware_chunked_ssm -- NOT used by the model itself (superseded by the
    fully-chunked, checkpointed dt_aware_chunked_ssm above, which never
    materializes A_bar/B_bar_x at full length in the first place)."""
    batch, L, D, N = A_bar.shape
    device, dtype = A_bar.device, A_bar.dtype
    h_out = torch.empty(batch, L, D, N, device=device, dtype=dtype)
    h_carry = torch.zeros(batch, D, N, device=device, dtype=dtype)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        a_chunk = A_bar[:, start:end]
        bx_chunk = B_bar_x[:, start:end]

        log_a = torch.log(a_chunk.clamp_min(1e-12))
        a_cumprod = torch.exp(torch.cumsum(log_a, dim=1))
        carry_contrib = a_cumprod * h_carry.unsqueeze(1)
        scaled = bx_chunk / a_cumprod.clamp_min(1e-12)
        within_contrib = a_cumprod * torch.cumsum(scaled, dim=1)

        h_chunk = carry_contrib + within_contrib
        h_out[:, start:end] = h_chunk
        h_carry = h_chunk[:, -1]

    return h_out


# ======================================================= dt-aware mamba ====
class DtAwareMambaBlock(nn.Module):
    """Selective SSM block (Mamba-style) with physically-grounded Delta-t-
    aware discretization. A is diagonal, per-channel; B, C, and the
    selective step-size SCALE are input-dependent (the "selective" part of
    Mamba); the true observation gap dt_t is the primary driver of the
    discretization step, per the guidance doc's Abar=exp(dt*A) formulation,
    modulated by a small learned per-timestep scale so the model retains
    Mamba's input-selectivity on top of the physical time gap.
    """

    def __init__(self, d_model, d_state=16, expand=2, chunk_size=256):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.chunk_size = chunk_size

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4,
                                 padding=3, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1)  # -> B, C, dt_scale
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # A: (d_inner, d_state), negative real part for stability, standard
        # Mamba "S4D-real" init (A_log parameterized, A = -exp(A_log))
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip connection gain

    def forward(self, x, dt_phys):
        """x: (batch, L, d_model). dt_phys: (batch, L) physical time gaps
        (days) between consecutive observations, dt_phys[:,0] defined as a
        small nominal value (no "previous" cadence for the first point)."""
        batch, L, _ = x.shape
        xz = self.in_proj(x)                       # (batch, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        proj = self.x_proj(x_conv)                  # (batch, L, 2N+1)
        B_sel, C_sel, dt_scale_raw = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        dt_scale = F.softplus(dt_scale_raw)          # (batch, L, 1), >0, input-selective

        dt_eff = (dt_phys * dt_scale.squeeze(-1)).clamp(1e-4, 10.0)  # (batch, L)

        A = -torch.exp(self.A_log)                   # (d_inner, N), negative
        # Abar = exp(dt * A) ; Bbar = (dt*A)^-1 (Abar - I) B  [guidance doc formula],
        # computed per-chunk (never as a full (batch,L,d_inner,N) tensor) inside
        # dt_aware_chunked_ssm -- see that function's docstring for why chunking
        # alone (without gradient checkpointing) was NOT sufficient to fit 8GB.
        y = dt_aware_chunked_ssm(x_conv, dt_eff, A, B_sel, C_sel, chunk_size=self.chunk_size)
        y = y + self.D * x_conv

        y = y * F.silu(z)
        return self.out_proj(y)


# ========================================================= dt-aware s4d ====
class DtAwareS4DBlock(nn.Module):
    """Diagonal SSM (S4D-style): A, B, C are NOT input-dependent (that's the
    whole point of S4D -- a static, learned linear system per channel), but
    the discretization is still Delta-t-aware (Abar=exp(dt*A)) so real
    observation gaps are respected. Because A/B/C don't depend on the input,
    a fixed-dt S4D can use an FFT convolution; with per-step irregular dt
    that global-convolution shortcut isn't directly available (the effective
    kernel itself would need to vary per gap), so this uses the SAME chunked
    linear scan as the Mamba block for a fair, apples-to-apples wall-clock
    comparison instead of an FFT fast path -- documented as a specific
    implementation choice, not an oversight.
    """

    def __init__(self, d_model, d_state=64, chunk_size=256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size

        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, dt_phys):
        batch, L, _ = x.shape
        x_in = F.silu(self.in_proj(x))               # (batch, L, d_model)

        A = -torch.exp(self.A_log)                    # (d_model, N)
        dt_eff = dt_phys.clamp(1e-4, 10.0)             # (batch, L) -- no input-selectivity for S4D
        y = dt_aware_chunked_ssm(x_in, dt_eff, A, self.B, self.C, chunk_size=self.chunk_size)
        y = y + self.D * x_in
        return self.out_proj(y)


# ============================================================ full model ===
class ResidualBlock(nn.Module):
    def __init__(self, ssm_block, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = ssm_block

    def forward(self, x, dt_phys):
        return x + self.ssm(self.norm(x), dt_phys)


class SSMWorldModel(nn.Module):
    """Masked-autoencoder world model over flux time series. `backbone`
    selects 'mamba' or 's4d'. Predicts flux at every timestep; loss is
    computed only on masked positions (see contiguous_block_mask below)."""

    def __init__(self, d_model=192, n_layers=6, d_state=16, backbone="mamba", chunk_size=256,
                 mamba_expand=4, checkpoint_layers=True):
        super().__init__()
        self.d_model = d_model
        self.backbone = backbone
        self.checkpoint_layers = checkpoint_layers
        self.in_proj = nn.Linear(1, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        def make_block():
            if backbone == "mamba":
                # expand=4 (not Mamba's usual default of 2) chosen specifically to land
                # close to ExoVeil's ~3.2M-parameter scale at d_model=192/6 layers -- see
                # results/module2_ssm_design.md for the parameter sweep.
                return DtAwareMambaBlock(d_model, d_state=d_state, expand=mamba_expand, chunk_size=chunk_size)
            elif backbone == "s4d":
                return DtAwareS4DBlock(d_model, d_state=d_state * 4, chunk_size=chunk_size)
            raise ValueError(backbone)

        self.layers = nn.ModuleList([ResidualBlock(make_block(), d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, flux, dt_phys, mask):
        """flux: (batch, L) normalized flux. dt_phys: (batch, L) physical
        gaps in days. mask: (batch, L) bool, True = masked (hidden from the
        model, must be predicted). Per the guidance doc, masked tokens are
        RETAINED in the sequence (not removed) with their value replaced by
        a learnable [MASK] embedding, preserving dt/phase alignment."""
        x = self.in_proj(flux.unsqueeze(-1))              # (batch, L, d_model)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1).expand_as(x), x)
        for layer in self.layers:
            # Layer-level checkpointing, on top of the inner scan-chunk
            # checkpointing inside dt_aware_chunked_ssm. Found empirically to
            # be necessary, not optional: a single block's forward+backward
            # peaks at ~1.6GB at target scale (L~19728, d_model=192,
            # mamba_expand=4), and with 6 layers stacked, autograd must
            # retain EVERY layer's activations simultaneously for backprop
            # through the residual stack unless the layers themselves are
            # also checkpointed -- ~1.6GB x 6 layers additively explained the
            # 8.7GB peak this fixes (down to ~1.6-2GB, see
            # results/module2_ssm_design.md for the before/after numbers).
            if self.checkpoint_layers and torch.is_grad_enabled():
                x = ckpt.checkpoint(layer, x, dt_phys, use_reentrant=True)
            else:
                x = layer(x, dt_phys)
        x = self.norm_f(x)
        return self.out_proj(x).squeeze(-1)                # (batch, L) predicted flux


# =============================================================== masking ===
def contiguous_block_mask(L, mask_ratio, block_min, block_max, device, batch=1, rng=None):
    """Contiguous block masking, NOT random pointwise. Random pointwise
    masking lets the model interpolate a transit trivially from its
    unmasked in-transit neighbors (flux at 2-min cadence is highly
    autocorrelated), reconstructing the dip perfectly and hiding it from
    the residual detector at inference -- this is the exact failure mode
    documented in the guidance doc (Sec 2) and is why this function exists
    instead of a one-line torch.rand()<ratio call.

    Draws blocks of length in [block_min, block_max] (matching physical
    transit durations, ~2-15h -> 100-600 cadences at 2-min TESS cadence)
    until the target mask_ratio of the sequence is covered.
    """
    rng = rng or np.random.default_rng()
    mask = torch.zeros(batch, L, dtype=torch.bool, device=device)
    target = int(L * mask_ratio)
    for b in range(batch):
        covered = 0
        attempts = 0
        while covered < target and attempts < 1000:
            attempts += 1
            block_len = int(rng.integers(block_min, block_max + 1))
            start = int(rng.integers(0, max(1, L - block_len)))
            end = min(L, start + block_len)
            newly = (~mask[b, start:end]).sum().item()
            mask[b, start:end] = True
            covered += newly
    return mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
