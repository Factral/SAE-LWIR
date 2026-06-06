#!/usr/bin/env python3
# analyze_row_distributions.py

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from framework.data import AtmosphericDataset
from framework.models import MultiHeadSetTransformer
from sae.sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE


def build_sae(cfg: dict):
    t = cfg["sae_type"].lower()
    if t == "vanilla":
        return VanillaSAE(cfg)
    if t == "topk":
        return TopKSAE(cfg)
    if t == "batchtopk":
        return BatchTopKSAE(cfg)
    if t == "jumprelu":
        return JumpReLUSAE(cfg)
    raise ValueError(f"Unknown sae_type={cfg['sae_type']}")


@torch.no_grad()
def get_h(model, x: torch.Tensor) -> torch.Tensor:
    # Preferred API: model(x, return_h=True) -> (T,U,D,h)
    try:
        out = model(x, return_h=True)
        return out[-1]  # h
    except TypeError:
        pass
    # Fallback: model.encode(x)
    if hasattr(model, "encode"):
        return model.encode(x)
    raise RuntimeError("Base model must support forward(return_h=True) or .encode(x).")


def cosine_sim_matrix(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # A: (K, D)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    return An @ An.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--sae_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="row_analysis")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    ap.add_argument("--set_size", type=int, default=7)
    ap.add_argument("--dim_input", type=int, default=256)
    ap.add_argument("--dim_hidden", type=int, default=256)

    ap.add_argument("--sae_type", type=str, default="batchtopk", choices=["vanilla", "topk", "batchtopk", "jumprelu"])
    ap.add_argument("--dict_size", type=int, default=256 * 16)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--l1_coeff", type=float, default=0.0)
    ap.add_argument("--input_unit_norm", action="store_true", default=True)
    ap.add_argument("--bandwidth", type=float, default=0.001)  # for jumprelu if needed

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=30000)

    ap.add_argument("--topn_features", type=int, default=25)
    ap.add_argument("--hist_bins", type=int, default=80)
    ap.add_argument("--max_act_samples_per_row", type=int, default=200000)  # for magnitude histogram sampling

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # ---- load base model (frozen) ----
    base = MultiHeadSetTransformer(
        dim_input=args.dim_input,
        set_size=args.set_size,
        dim_hidden=args.dim_hidden,
    ).to(device=device, dtype=dtype)

    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    base.load_state_dict(state, strict=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    # ---- load SAE (frozen) ----
    sae_cfg = {
        "seed": 0,
        "device": str(device),
        "dtype": dtype,
        "act_size": args.dim_hidden,
        "dict_size": args.dict_size,
        "batch_size": 4096,
        "lr": 0.0,
        "sae_type": args.sae_type,
        "top_k": args.top_k,
        "l1_coeff": args.l1_coeff,
        "input_unit_norm": args.input_unit_norm,
        "num_batches_in_buffer": 1,
        "n_batches_to_dead": 5,
        "top_k_aux": 512,
        "aux_penalty": 1 / 32,
        "bandwidth": args.bandwidth,
    }
    sae = build_sae(sae_cfg).to(device=device, dtype=dtype)
    sae_state = torch.load(args.sae_ckpt, map_location="cpu")
    sae.load_state_dict(sae_state, strict=True)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    K = args.set_size
    D = args.dict_size

    # ---- dataset/loader ----
    ds = AtmosphericDataset(data_dir=args.data_dir)
    n_scan = min(len(ds), args.max_samples)
    subset = torch.utils.data.Subset(ds, range(n_scan))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ---- stats accumulators per row ----
    # per row: sum acts (D,), nonzero count (D,), token count (scalar)
    sum_act = [torch.zeros(D, dtype=torch.float64) for _ in range(K)]
    nz_cnt = [torch.zeros(D, dtype=torch.float64) for _ in range(K)]
    tok_cnt = [0 for _ in range(K)]

    # per row l0 samples (for histogram)
    l0_samples = [[] for _ in range(K)]  # list of arrays

    # per row activation magnitudes (sampled) for histogram
    act_val_samples = [[] for _ in range(K)]

    rng = np.random.default_rng(0)

    @torch.no_grad()
    def maybe_sample_vals(vals_1d: np.ndarray, row: int):
        # reservoir-ish: just cap total stored values per row
        if vals_1d.size == 0:
            return
        cur = sum(v.size for v in act_val_samples[row])
        remaining = args.max_act_samples_per_row - cur
        if remaining <= 0:
            return
        if vals_1d.size <= remaining:
            act_val_samples[row].append(vals_1d)
        else:
            idx = rng.choice(vals_1d.size, size=remaining, replace=False)
            act_val_samples[row].append(vals_1d[idx])

    # ---- main loop ----
    for batch in loader:
        x, y1, y2, y3, temp = batch
        x = x.to(device, non_blocking=True)

        h = get_h(base, x)                 # (B,K,H)
        h_flat = h.reshape(-1, h.size(-1)) # (B*K,H)

        out = sae(h_flat)
        acts = out["feature_acts"]         # (B*K, D)
        acts = acts.reshape(x.size(0), K, D).detach().float().cpu()  # to CPU for stats

        # per-row stats
        for r in range(K):
            A = acts[:, r, :]  # (B, D)

            nz = (A > 0)
            sum_act[r] += A.sum(dim=0).double()
            nz_cnt[r] += nz.sum(dim=0).double()
            tok_cnt[r] += A.size(0)

            l0 = nz.sum(dim=1).numpy()  # (B,)
            l0_samples[r].append(l0)

            # magnitude samples
            vals = A[nz].numpy()
            maybe_sample_vals(vals, r)

    # ---- finalize stats ----
    mean_act = np.stack([(sum_act[r] / max(tok_cnt[r], 1)).numpy() for r in range(K)], axis=0)  # (K,D)
    nz_rate = np.stack([(nz_cnt[r] / max(tok_cnt[r], 1)).numpy() for r in range(K)], axis=0)    # (K,D)

    l0_all = [np.concatenate(l0_samples[r]) if len(l0_samples[r]) else np.array([], dtype=np.int64) for r in range(K)]
    act_all = [np.concatenate(act_val_samples[r]) if len(act_val_samples[r]) else np.array([], dtype=np.float32) for r in range(K)]

    # ---- plot 1: L0 hist per row ----
    plt.figure(figsize=(9, 5))
    for r in range(K):
        if l0_all[r].size == 0:
            continue
        plt.hist(l0_all[r], bins=min(args.hist_bins, max(int(l0_all[r].max() + 1), 10)), alpha=0.45, density=True, label=f"row {r}")
    plt.title("L0 distribution (#active features per token) by row")
    plt.xlabel("L0")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "row_l0.png"), dpi=160)
    plt.close()

    # ---- plot 2: activation magnitude hist per row ----
    # Use log-x to see tail better (common in practice)
    plt.figure(figsize=(9, 5))
    for r in range(K):
        if act_all[r].size == 0:
            continue
        vals = act_all[r]
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        # log-spaced bins
        vmin = max(vals.min(), 1e-8)
        vmax = max(vals.max(), vmin * 10)
        bins = np.logspace(np.log10(vmin), np.log10(vmax), args.hist_bins)
        plt.hist(vals, bins=bins, alpha=0.45, density=True, label=f"row {r}")
    plt.xscale("log")
    plt.title("Activation magnitude distribution (nonzero acts) by row")
    plt.xlabel("activation value (log scale)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "row_actvals.png"), dpi=160)
    plt.close()

    # ---- plot 3: top features per row by nonzero-rate ----
    topn = args.topn_features
    fig, axs = plt.subplots(K, 1, figsize=(12, 2.2 * K), sharex=False)
    if K == 1:
        axs = [axs]
    for r in range(K):
        idx = np.argsort(-nz_rate[r])[:topn]
        axs[r].bar(np.arange(topn), nz_rate[r][idx])
        axs[r].set_title(f"row {r}: top-{topn} features by nonzero-rate")
        axs[r].set_ylabel("p(active)")
        axs[r].set_xticks(np.arange(topn))
        axs[r].set_xticklabels([str(int(i)) for i in idx], rotation=60, fontsize=8)
    axs[-1].set_xlabel("feature id")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "row_top_features.png"), dpi=160)
    plt.close(fig)

    # ---- plot 4: similarity between rows (using nz_rate profiles) ----
    sim = cosine_sim_matrix(nz_rate)  # (K,K)
    plt.figure(figsize=(6, 5))
    plt.imshow(sim, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Row similarity (cosine) using feature nonzero-rate profiles")
    plt.xticks(range(K), [f"{r}" for r in range(K)])
    plt.yticks(range(K), [f"{r}" for r in range(K)])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "row_similarity.png"), dpi=160)
    plt.close()

    # ---- save raw arrays for later ----
    np.save(os.path.join(args.out_dir, "mean_act_by_row.npy"), mean_act)
    np.save(os.path.join(args.out_dir, "nz_rate_by_row.npy"), nz_rate)
    np.save(os.path.join(args.out_dir, "l0_samples_by_row.npy"), np.array(l0_all, dtype=object), allow_pickle=True)
    np.save(os.path.join(args.out_dir, "actval_samples_by_row.npy"), np.array(act_all, dtype=object), allow_pickle=True)

    print(f"[OK] Wrote plots + arrays to: {args.out_dir}")


if __name__ == "__main__":
    main()
