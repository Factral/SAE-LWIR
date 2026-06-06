#!/usr/bin/env python3
"""
Find & cluster "interesting" SAE features and plot top-activating training samples per cluster.

Works with your SetTransformer setup:
- Base model returns encoder activations h when called with return_h=True
  OR exposes model.encode(x) returning (B,K,H).
- SAE forward returns dict with "feature_acts" (Ntok, dict_size).

Outputs:
- out_dir/cluster_{id:03d}.png
- out_dir/feature_stats.csv
- out_dir/cluster_assignments.npy
"""

import argparse
import csv
import heapq
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans

from framework.data import AtmosphericDataset  # your dataset
from framework.models import MultiHeadSetTransformer
from sae.sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE


@dataclass
class TopExample:
    act: float
    uniq: int
    feat: int
    global_sample: int
    row: int
    temp_K: float
    x_row: np.ndarray  # (dim_input,) float16


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
    # Prefer your forward(return_h=True) API if present.
    try:
        out = model(x, return_h=True)
        h = out[-1]
        return h
    except TypeError:
        pass
    # Fallback: encode(x)
    if hasattr(model, "encode"):
        return model.encode(x)
    raise RuntimeError("Base model must support forward(return_h=True) or .encode(x).")


def normalize_rows(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / (norms + eps)


def cluster_features(W_dec: np.ndarray, n_clusters: int, method: str) -> np.ndarray:
    Wn = normalize_rows(W_dec)

    method = method.lower()
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        labels = km.fit_predict(Wn)
        return labels.astype(np.int32)

    if method == "agglom":
        # sklearn changed API: affinity->metric in newer versions
        try:
            cl = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        except TypeError:
            cl = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
        labels = cl.fit_predict(Wn)
        return labels.astype(np.int32)

    raise ValueError("method must be one of: kmeans, agglom")


def save_feature_stats_csv(path: str, mean_act: np.ndarray, max_act: np.ndarray, nz_rate: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "mean_act", "max_act", "nonzero_rate"])
        for j in range(mean_act.shape[0]):
            w.writerow([j, float(mean_act[j]), float(max_act[j]), float(nz_rate[j])])


def plot_cluster(
    out_path: str,
    cluster_id: int,
    examples: List[TopExample],
    *,
    ncols: int = 4,
    title_extra: str = "",
):
    if len(examples) == 0:
        return

    n = len(examples)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows))
    axs = np.atleast_1d(axs).reshape(nrows, ncols)

    for i, ex in enumerate(examples):
        r, c = divmod(i, ncols)
        ax = axs[r, c]
        ax.plot(ex.x_row.astype(np.float32))
        ax.set_title(
            f"f{ex.feat} a={ex.act:.3g} s={ex.global_sample} row={ex.row} T={ex.temp_K:.1f}K",
            fontsize=9,
        )
        ax.set_xlabel("band")
        ax.set_ylabel("value")

    # hide unused
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axs[r, c].axis("off")

    fig.suptitle(f"Cluster {cluster_id} {title_extra}".strip(), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--base_ckpt", type=str, required=True, help="Path to trained SetTransformer checkpoint (.pt)")
    ap.add_argument("--sae_ckpt", type=str, required=True, help="Path to SAE checkpoint (sae.pt)")
    ap.add_argument("--out_dir", type=str, default="sae_feature_mining")

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

    ap.add_argument("--batch_size", type=int, default=128, help="Base data loader batch size (samples)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=20000, help="How many dataset samples to scan")

    ap.add_argument("--top_per_feature", type=int, default=10, help="Top examples to keep per feature")
    ap.add_argument("--n_clusters", type=int, default=64)
    ap.add_argument("--cluster_method", type=str, default="agglom", choices=["agglom", "kmeans"])
    ap.add_argument("--top_clusters", type=int, default=24, help="How many clusters to plot (ranked by activity)")
    ap.add_argument("--examples_per_cluster", type=int, default=12)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # ---- load base model ----
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

    # ---- load SAE ----
    cfg = {
        "seed": 0,
        "device": str(device),
        "dtype": dtype,
        "act_size": args.dim_hidden,
        "dict_size": args.dict_size,
        "batch_size": 4096,  # not used here (analysis), but SAE code expects it
        "lr": 0.0,
        "sae_type": args.sae_type,
        "top_k": args.top_k,
        "l1_coeff": args.l1_coeff,
        "input_unit_norm": args.input_unit_norm,
        "num_batches_in_buffer": 1,
        "n_batches_to_dead": 5,
        "top_k_aux": 512,
        "aux_penalty": 1 / 32,
        "bandwidth": 0.001,
    }
    sae = build_sae(cfg).to(device=device, dtype=dtype)
    sae_state = torch.load(args.sae_ckpt, map_location="cpu")
    sae.load_state_dict(sae_state, strict=True)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    # ---- cluster features by decoder direction ----
    W_dec = sae.W_dec.detach().float().cpu().numpy()  # (dict, act)
    labels = cluster_features(W_dec, n_clusters=args.n_clusters, method=args.cluster_method)
    np.save(os.path.join(args.out_dir, "cluster_assignments.npy"), labels)

    # ---- scan dataset for activation stats + top activating examples ----
    ds = AtmosphericDataset(data_dir=args.data_dir)
    n_scan = min(len(ds), args.max_samples)

    # IMPORTANT: shuffle=False so global_sample indexing is stable.
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, range(n_scan)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    dict_size = W_dec.shape[0]
    sum_act = torch.zeros(dict_size, dtype=torch.float64)
    max_act = torch.zeros(dict_size, dtype=torch.float64)
    nz_cnt = torch.zeros(dict_size, dtype=torch.float64)
    total_tok = 0

    heaps: List[List[Tuple[float, int, TopExample]]] = [[] for _ in range(dict_size)]
    uniq = 0

    global_sample_offset = 0
    K = args.set_size

    for batch in loader:
        x, y1, y2, y3, temp = batch
        bsz = x.size(0)

        x = x.to(device, non_blocking=True)
        h = get_h(base, x)                    # (B,K,H)
        h_flat = h.reshape(-1, h.size(-1))    # (B*K,H)

        out = sae(h_flat)
        acts = out["feature_acts"]            # (B*K, dict)
        acts_cpu = acts.detach().float().cpu()  # for stats / heap updates

        # stats
        sum_act += acts_cpu.sum(dim=0).double()
        max_act = torch.maximum(max_act, acts_cpu.max(dim=0).values.double())
        nz_cnt += (acts_cpu > 0).sum(dim=0).double()
        total_tok += acts_cpu.shape[0]

        # top examples per feature (efficient because acts is sparse)
        nz = (acts_cpu > 0).nonzero(as_tuple=False)  # (nnz,2) [tok_idx, feat_idx]
        if nz.numel() > 0:
            tok_idx = nz[:, 0].numpy()
            feat_idx = nz[:, 1].numpy()
            vals = acts_cpu[nz[:, 0], nz[:, 1]].numpy()

            x_cpu = x.detach().cpu()  # (B,K,dim_input)
            temp_cpu = temp.detach().cpu() if torch.is_tensor(temp) else temp
            for t_i, f_i, v in zip(tok_idx, feat_idx, vals):
                if v <= 0:
                    continue
                sample_i = int(t_i // K)
                row_i = int(t_i % K)
                gsample = int(global_sample_offset + sample_i)

                x_row = x_cpu[sample_i, row_i].to(torch.float16).numpy()
                temp_k = float(temp_cpu[sample_i]) if torch.is_tensor(temp_cpu) else float(temp_cpu)

                ex = TopExample(
                    act=float(v),
                    uniq=uniq,
                    feat=int(f_i),
                    global_sample=gsample,
                    row=row_i,
                temp_K=temp_k,
                    x_row=x_row,
                )
                uniq += 1

                heap = heaps[f_i]
                item = (ex.act, ex.uniq, ex)
                if len(heap) < args.top_per_feature:
                    heapq.heappush(heap, item)
                else:
                    if ex.act > heap[0][0]:
                        heapq.heapreplace(heap, item)

        global_sample_offset += bsz

    mean_act = (sum_act / max(total_tok, 1)).numpy()
    max_act_np = max_act.numpy()
    nz_rate = (nz_cnt / max(total_tok, 1)).numpy()

    save_feature_stats_csv(
        os.path.join(args.out_dir, "feature_stats.csv"),
        mean_act=mean_act,
        max_act=max_act_np,
        nz_rate=nz_rate,
    )

    # ---- rank clusters by how "active" they are (max mean activity among members) ----
    clusters: Dict[int, List[int]] = {}
    for f in range(dict_size):
        c = int(labels[f])
        clusters.setdefault(c, []).append(f)

    cluster_score = []
    for c, feats in clusters.items():
        score = float(np.max(mean_act[np.array(feats, dtype=np.int64)]))
        cluster_score.append((score, c))
    cluster_score.sort(reverse=True)

    # ---- plot top clusters ----
    for rank, (score, c) in enumerate(cluster_score[: args.top_clusters]):
        feats = clusters[c]
        feats = sorted(feats, key=lambda f: mean_act[f], reverse=True)

        # gather top examples across the top features in the cluster
        gathered: List[TopExample] = []
        for f in feats[: min(len(feats), 64)]:
            for _, _, ex in heaps[f]:
                gathered.append(ex)

        gathered.sort(key=lambda e: e.act, reverse=True)
        gathered = gathered[: args.examples_per_cluster]

        out_path = os.path.join(args.out_dir, f"cluster_{c:03d}.png")
        plot_cluster(
            out_path,
            cluster_id=c,
            examples=gathered,
            ncols=4,
            title_extra=f"(rank {rank}, score {score:.3g})",
        )

    print(f"Done. Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
