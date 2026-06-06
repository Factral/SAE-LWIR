# scripts/sae_histograms.py
import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from framework.data import make_split_dataloader
from framework.models import MultiHeadSetTransformer

from sae.config import get_default_cfg, post_init_cfg
from sae.sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE


def build_sae(cfg):
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model_ckpt", type=str, required=True, help="Checkpoint del SetTransformer (state_dict).")
    ap.add_argument("--sae_ckpt", type=str, required=True, help="Checkpoint del SAE (state_dict).")

    ap.add_argument("--out_dir", type=str, default="sae_analysis")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch_size", type=int, default=256)      # loader batch (datos)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--split_seed", type=int, default=1312)

    ap.add_argument("--set_size", type=int, default=7)
    ap.add_argument("--dim_input", type=int, default=256)

    # SAE config overrides
    ap.add_argument("--sae_type", type=str, default=None, help="topk|batchtopk|jumprelu|vanilla")
    ap.add_argument("--dict_size", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--input_unit_norm", action="store_true")

    # Analysis controls
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--token_idx", type=int, default=-1, help="-1 = todos los K tokens; si no 0..K-1")
    ap.add_argument("--feature_id", type=int, default=0)

    args = ap.parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = MultiHeadSetTransformer(dim_input=args.dim_input, set_size=args.set_size).to(device)
    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    # Load SAE cfg + model
    cfg = get_default_cfg()
    if args.sae_type is not None:
        cfg["sae_type"] = args.sae_type
    if args.dict_size is not None:
        cfg["dict_size"] = args.dict_size
    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.input_unit_norm:
        cfg["input_unit_norm"] = True
    cfg["device"] = str(device)
    cfg["act_size"] = args.dim_input  # h dim
    cfg = post_init_cfg(cfg)

    sae = build_sae(cfg)
    sae.load_state_dict(torch.load(args.sae_ckpt, map_location="cpu"))
    sae.eval()

    # Data
    train_loader, val_loader, test_loader = make_split_dataloader(
        split="train",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        train_drop_last=True,
        split_seed=args.split_seed,
    )
    loader = test_loader

    # Streaming stats
    N = 0
    l0_all = []

    act_count = torch.zeros(cfg["dict_size"], dtype=torch.long)
    act_sum = torch.zeros(cfg["dict_size"], dtype=torch.float64)

    feat_vals = []

    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        x, y1, y2, y3, temp = batch
        x = x.to(device, non_blocking=True)

        _, _, _, h = model(x, return_h=True)  # (B,K,H)

        if args.token_idx >= 0:
            h_use = h[:, args.token_idx, :]          # (B,H)
        else:
            h_use = h.reshape(-1, h.size(-1))        # (B*K,H)

        out = sae(h_use)
        acts = out["feature_acts"].detach().cpu()    # (Nvec, dict)

        l0 = (acts > 0).sum(dim=-1).cpu()
        l0_all.append(l0)

        nonzero = (acts > 0)
        act_count += nonzero.sum(dim=0).to(torch.long)
        act_sum += acts.sum(dim=0).to(torch.float64)
        N += acts.size(0)

        feat_vals.append(acts[:, args.feature_id].cpu())

    l0_all = torch.cat(l0_all).numpy()
    feat_vals = torch.cat(feat_vals).numpy()

    # Save L0 histogram
    plt.figure()
    plt.hist(l0_all, bins=60)
    plt.title("L0 per token-vector (number of active latents)")
    plt.xlabel("L0")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "l0_hist.png", dpi=200)
    plt.close()

    # Feature histogram (all values + active-only)
    plt.figure()
    plt.hist(feat_vals, bins=80)
    plt.title(f"Feature {args.feature_id} activations (incl zeros)")
    plt.xlabel("activation")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"feature_{args.feature_id:05d}_hist_all.png", dpi=200)
    plt.close()

    active_vals = feat_vals[feat_vals > 0]
    plt.figure()
    plt.hist(active_vals, bins=80)
    plt.title(f"Feature {args.feature_id} activations (active-only)")
    plt.xlabel("activation")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"feature_{args.feature_id:05d}_hist_active.png", dpi=200)
    plt.close()

    # Feature frequency summary
    freq = (act_count.float() / max(N, 1)).numpy()
    mean_act = (act_sum / torch.clamp(act_count, min=1).to(torch.float64)).numpy()

    df = pd.DataFrame({
        "feature_id": list(range(cfg["dict_size"])),
        "activation_freq": freq,
        "mean_when_active": mean_act,
        "active_count": act_count.numpy(),
    }).sort_values("activation_freq", ascending=False)

    df.to_csv(out_dir / "feature_stats.csv", index=False)

    # Plot top-50 frequencies
    top = df.head(50)
    plt.figure(figsize=(10, 5))
    plt.bar(top["feature_id"].astype(int).astype(str), top["activation_freq"])
    plt.xticks(rotation=90)
    plt.title("Top-50 features by activation frequency")
    plt.ylabel("P(active)")
    plt.tight_layout()
    plt.savefig(out_dir / "top50_feature_freq.png", dpi=200)
    plt.close()

    print(f"[OK] Saved analysis to: {out_dir.resolve()}")
    print(f"[INFO] N_vectors={N}, L0_mean={l0_all.mean():.2f}, dead={(act_count==0).sum().item()}")


if __name__ == "__main__":
    main()
