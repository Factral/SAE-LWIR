# scripts/sae_feature_saturation_effect.py
"""
Saturate (force high activation of) one SAE feature on a single sample, then
compare model outputs (T,U,D) before vs after saturation.

Example:
python scripts/sae_feature_saturation_effect.py \
  --data_dir set_transformer/data \
  --model_ckpt results/set_transformer/trial/model.pt \
  --sae_ckpt checkpoints/.../sae.pt \
  --feature_id 2489 \
  --idx 12 \
  --split test \
  --token_idx -1 \
  --sat_mult 10.0 \
  --out_dir sae_saturation_plots
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from framework.data import AtmosphericDataset  # if your data.py doesn't export this, import directly from your module
from torch.utils.data import random_split
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


def _extract_state_dict(obj):
    # supports: raw state_dict, or {"model": state_dict, ...}
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Checkpoint format not understood (expected dict / state_dict).")


def get_split_subset(data_dir: str, split: str, split_seed: int):
    ds = AtmosphericDataset(data_dir=data_dir)
    n = len(ds)
    train_len = int(n * 0.70)
    val_len = int(n * 0.10)
    test_len = n - train_len - val_len

    g = torch.Generator()
    g.manual_seed(split_seed)
    train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len], generator=g)

    split = split.lower()
    if split == "train":
        return train_ds
    if split == "val":
        return val_ds
    if split == "test":
        return test_ds
    raise ValueError("split must be one of: train/val/test")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model_ckpt", type=str, required=True)
    ap.add_argument("--sae_ckpt", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="sae_saturation_plots")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--split_seed", type=int, default=1312)
    ap.add_argument("--idx", type=int, required=True, help="index within the selected split subset")

    ap.add_argument("--set_size", type=int, default=7)
    ap.add_argument("--dim_input", type=int, default=256)

    # SAE config overrides (optional)
    ap.add_argument("--sae_type", type=str, default=None, choices=[None, "vanilla", "topk", "batchtopk", "jumprelu"])
    ap.add_argument("--dict_size", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--input_unit_norm", action="store_true")

    # intervention
    ap.add_argument("--feature_id", type=int, required=True)
    ap.add_argument("--token_idx", type=int, default=-1, help="-1 = saturate for all rows; else 0..K-1")
    ap.add_argument("--sat_value", type=float, default=None, help="explicit value to set; if None uses sat_mult")
    ap.add_argument("--sat_mult", type=float, default=10.0, help="multiplier for auto sat_value")
    ap.add_argument("--mode", type=str, default="set", choices=["set", "add", "mul"])
    ap.add_argument("--drop_others", action="store_true", help="if True, set all other features to 0")

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- load base model ----------------
    model = MultiHeadSetTransformer(dim_input=args.dim_input, set_size=args.set_size).to(device)
    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    model_sd = _extract_state_dict(ckpt)
    model.load_state_dict(model_sd)
    model.eval()

    # ---------------- load SAE ----------------
    sae_ckpt = torch.load(args.sae_ckpt, map_location="cpu")
    sae_sd = _extract_state_dict(sae_ckpt)

    # infer sizes if not provided (helps avoid mismatch)
    if "W_enc" in sae_sd:
        inferred_act_size, inferred_dict_size = sae_sd["W_enc"].shape
    else:
        inferred_act_size = args.dim_input
        inferred_dict_size = args.dict_size

    cfg = get_default_cfg()
    if args.sae_type is not None:
        cfg["sae_type"] = args.sae_type
    cfg["device"] = str(device)
    cfg["act_size"] = int(inferred_act_size)
    if args.dict_size is not None:
        cfg["dict_size"] = int(args.dict_size)
    elif inferred_dict_size is not None:
        cfg["dict_size"] = int(inferred_dict_size)

    if args.top_k is not None:
        cfg["top_k"] = int(args.top_k)
    if args.input_unit_norm:
        cfg["input_unit_norm"] = True

    cfg = post_init_cfg(cfg)
    sae = build_sae(cfg)
    sae.load_state_dict(sae_sd, strict=True)
    sae.eval()

    fid = int(args.feature_id)
    if not (0 <= fid < cfg["dict_size"]):
        raise ValueError(f"feature_id {fid} out of range [0, {cfg['dict_size']-1}]")

    # ---------------- get one sample (by index) ----------------
    subset = get_split_subset(args.data_dir, args.split, args.split_seed)
    if args.idx < 0 or args.idx >= len(subset):
        raise IndexError(f"idx={args.idx} out of range for split={args.split} (len={len(subset)})")

    x, y1, y2, y3, temp = subset[args.idx]
    # add batch dim
    x = x.unsqueeze(0).to(device, non_blocking=True)   # (1,7,256)
    y1 = y1.unsqueeze(0).to(device, non_blocking=True) # (1,7,256)
    y2 = y2.unsqueeze(0).to(device, non_blocking=True) # (1,7,256)
    y3 = y3.unsqueeze(0).to(device, non_blocking=True) # (1,1,256)

    # ---------------- baseline forward ----------------
    T0, U0, D0, h0 = model(x, return_h=True)  # h0: (1,7,256)

    # ---------------- compute SAE acts on h0 ----------------
    B, K, H = h0.shape
    h_flat = h0.reshape(B * K, H)  # (7,256)

    # We need x_mean/x_std for correct postprocess if input_unit_norm=True
    h_proc, h_mean, h_std = sae.preprocess_input(h_flat)

    sae_out = sae(h_flat)  # uses its own preprocess internally too, but we use acts from here
    acts = sae_out["feature_acts"]  # (7,dict)

    # Print features that are activated with this sample
    # We take the max activation across all tokens for each feature
    max_acts, _ = acts.max(dim=0)
    activated_mask = max_acts > 1e-6
    activated_indices = torch.nonzero(activated_mask).flatten()
    activated_values = max_acts[activated_indices]

    # Sort by value descending
    sorted_indices = torch.argsort(activated_values, descending=True)
    sorted_feat_ids = activated_indices[sorted_indices]
    sorted_feat_vals = activated_values[sorted_indices]

    print(f"[INFO] Activated features for this sample (sorted high to low):")
    for fid_idx, fval in zip(sorted_feat_ids.tolist(), sorted_feat_vals.tolist()):
        print(f"  Feature {fid_idx:5d}: {fval:.4f}")

    # ---------------- choose saturation value ----------------
    # default: based on current activations for that feature; if zero everywhere, fallback to global max
    acts_f = acts[:, fid]
    if args.sat_value is not None:
        sat_val = float(args.sat_value)
    else:
        base = float(acts_f.max().item())
        if base <= 0.0:
            base = float(acts.max().item())
        if base <= 0.0:
            base = 1.0
        sat_val = base * float(args.sat_mult)

    # ---------------- apply intervention in latent space ----------------
    acts_mod = acts.clone()

    if args.token_idx >= 0:
        if not (0 <= args.token_idx < K):
            raise ValueError(f"token_idx must be -1 or 0..{K-1}")
        idxs = torch.tensor([args.token_idx], device=acts_mod.device, dtype=torch.long)
    else:
        idxs = torch.arange(K, device=acts_mod.device)

    if args.mode == "set":
        acts_mod[idxs, fid] = sat_val
    elif args.mode == "add":
        acts_mod[idxs, fid] = acts_mod[idxs, fid] + sat_val
    elif args.mode == "mul":
        acts_mod[idxs, fid] = acts_mod[idxs, fid] * sat_val
    else:
        raise ValueError("unknown mode")

    if args.drop_others:
        # Create a mask for others
        others_mask = torch.ones(acts_mod.shape[1], device=acts_mod.device, dtype=torch.bool)
        others_mask[fid] = False
        acts_mod[:, others_mask] = 0.0

    # ---------------- decode modified h from SAE ----------------
    # Reconstruct in normalized space then postprocess back.
    # Note: this mirrors the SAE reconstruction path: x_reconstruct = acts @ W_dec + b_dec
    h_rec_proc = acts_mod @ sae.W_dec + sae.b_dec  # (7,256) in "processed" space
    h_rec = sae.postprocess_output(h_rec_proc, h_mean, h_std)  # (7,256) in original space

    h_sat = h_rec.reshape(B, K, H)  # (1,7,256)

    # ---------------- forward with saturated h ----------------
    T1, U1, D1 = model(x, h_override=h_sat)

    # ---------------- plot (baseline vs saturated + GT) ----------------
    # Similar to your evaluate() plot style, but with both predictions.
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    base_c = "C0"
    sat_c = "C1"

    # T: plot 7 rows (no GT)
    for j in range(T0.size(1)):
        axs[0].plot(
            T0[0, j].detach().cpu().numpy(),
            "--",
            color=base_c,
            alpha=0.70,
            label="baseline" if j == 0 else None,
        )
        axs[0].plot(
            T1[0, j].detach().cpu().numpy(),
            ":",
            color=sat_c,
            alpha=0.85,
            label="saturated" if j == 0 else None,
        )

    # U: plot 7 rows (no GT)
    for j in range(U0.size(1)):
        axs[1].plot(
            U0[0, j].detach().cpu().numpy(),
            "--",
            color=base_c,
            alpha=0.70,
            label="baseline" if j == 0 else None,
        )
        axs[1].plot(
            U1[0, j].detach().cpu().numpy(),
            ":",
            color=sat_c,
            alpha=0.85,
            label="saturated" if j == 0 else None,
        )

    # D: single row
    axs[2].plot(D0[0, 0].detach().cpu().numpy(), "--", alpha=0.80, label="baseline")
    axs[2].plot(D1[0, 0].detach().cpu().numpy(), ":", alpha=0.95, label="saturated")
    axs[2].plot(y3[0, 0].detach().cpu().numpy(), "-", alpha=1.00, label="gt")

    axs[0].set_title(f"T (feature {fid} {args.mode} to {sat_val:.4g}, token_idx={args.token_idx})")
    axs[1].set_title("U")
    axs[2].set_title("D")
    axs[2].set_xlabel("lambda_idx")

    for ax in axs:
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.set_ylabel("value")

    plt.tight_layout()

    out_path = out_dir / f"idx{args.idx:06d}_fid{fid:05d}_tok{args.token_idx}_mode{args.mode}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    # quick numeric summary
    def mse(a, b):
        return float(F.mse_loss(a, b).item())

    print("[OK] saved:", out_path.resolve())
    print("[INFO] ΔMSE(T):", mse(T1, T0), " ΔMSE(U):", mse(U1, U0), " ΔMSE(D):", mse(D1, D0))


if __name__ == "__main__":
    main()
