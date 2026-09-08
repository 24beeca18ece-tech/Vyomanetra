#!/usr/bin/env python3
"""
Module 2 -- train the right-sized SSM world model on the local Kepler DR25
corpus (results/kepler_corpus/*.npz, from download_kepler_corpus.py).

Local-dev run only: validates the pipeline trains stably at the scale
already confirmed to fit in 8GB (results/module2_ssm_design.md) before any
GCP spend is considered. Reports GPU memory and wall-clock at each stage,
per the task's explicit requirement.

Fixed sequence length L (default 16384, comfortably within the validated
single-TESS-sector/multi-Kepler-quarter scale): curves longer than L are
randomly cropped (a form of augmentation across epochs); curves shorter
than L are skipped for this run rather than padded, to avoid the model
learning to exploit pad tokens (padding-with-mask would conflate "no data"
with "masked for training", polluting the pretext task).

Usage: python train_ssm_world_model.py --backbone mamba --steps 200 --batch-size 2
"""
import os
import sys
import time
import argparse
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules_2_ssm import SSMWorldModel, contiguous_block_mask, count_parameters

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORPUS_DIR = os.path.join(RESULTS, "kepler_corpus")


def load_corpus(min_len):
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.npz")))
    curves = []
    for f in files:
        d = np.load(f)
        if len(d["time"]) >= min_len:
            curves.append((os.path.basename(f), d["time"], d["flux"], d["dt"]))
    return curves


def sample_batch(curves, L, batch_size, rng, device):
    idxs = rng.choice(len(curves), size=batch_size, replace=len(curves) < batch_size)
    flux_batch = np.empty((batch_size, L), dtype=np.float32)
    dt_batch = np.empty((batch_size, L), dtype=np.float32)
    for i, idx in enumerate(idxs):
        _, t, f, dt = curves[idx]
        start = rng.integers(0, len(f) - L + 1)
        flux_batch[i] = f[start:start + L]
        dt_batch[i] = dt[start:start + L]
    return (torch.from_numpy(flux_batch).to(device),
            torch.from_numpy(dt_batch).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["mamba", "s4d"], default="mamba")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--mask-ratio", type=float, default=0.4)
    ap.add_argument("--block-min", type=int, default=100)
    ap.add_argument("--block-max", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    curves = load_corpus(args.seq_len)
    print(f"loaded {len(curves)} usable curves (>= {args.seq_len} points) from {CORPUS_DIR}", flush=True)
    if len(curves) == 0:
        print("ERROR: no usable curves. Run download_kepler_corpus.py first.")
        return

    model = SSMWorldModel(d_model=192, n_layers=6, d_state=16, backbone=args.backbone,
                          chunk_size=256).to(device)
    n_params = count_parameters(model)
    print(f"model: backbone={args.backbone}  params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses = []
    t_start = time.time()
    for step in range(args.steps):
        flux, dt_phys = sample_batch(curves, args.seq_len, args.batch_size, rng, device)
        mask = contiguous_block_mask(args.seq_len, args.mask_ratio, args.block_min, args.block_max,
                                     device=device, batch=args.batch_size, rng=rng)

        t0 = time.time()
        out = model(flux, dt_phys, mask)
        loss = ((out - flux)[mask]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        step_time = time.time() - t0

        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            mem_str = f"  peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == "cuda" else ""
            print(f"step {step:4d}/{args.steps}  loss={loss.item():.6f}  "
                  f"step_time={step_time:.2f}s{mem_str}", flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal wall-clock: {total_time:.1f}s for {args.steps} steps "
          f"({total_time/args.steps:.2f}s/step avg)", flush=True)
    print(f"Loss: first={losses[0]:.6f}  last={losses[-1]:.6f}  "
          f"min={min(losses):.6f}  (first-10-avg={np.mean(losses[:10]):.6f}, "
          f"last-10-avg={np.mean(losses[-10:]):.6f})", flush=True)

    ckpt_path = os.path.join(RESULTS, f"ssm_world_model_{args.backbone}.pt")
    torch.save({"model_state": model.state_dict(), "args": vars(args),
                "losses": losses, "n_params": n_params}, ckpt_path)
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
