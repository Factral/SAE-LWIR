# main.py
# %%
import os
import random

import torch

from sae.training import train_sae
from sae.sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from sae.activation_store import ActivationsStoreST
from sae.config import get_default_cfg, post_init_cfg

from framework.data import make_split_dataloader
from framework.models import MultiHeadSetTransformer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_sae(cfg):
    if cfg["sae_type"] == "vanilla":
        return VanillaSAE(cfg)
    if cfg["sae_type"] == "topk":
        return TopKSAE(cfg)
    if cfg["sae_type"] == "batchtopk":
        return BatchTopKSAE(cfg)
    if cfg["sae_type"] == "jumprelu":
        return JumpReLUSAE(cfg)
    raise ValueError(f"Unknown sae_type={cfg['sae_type']}")


def load_base_model(cfg):
    device = torch.device(cfg["device"])
    model = MultiHeadSetTransformer(
        dim_input=cfg["dim_input"],
        set_size=cfg["set_size"],
        dim_hidden=cfg["dim_hidden"],
    ).to(device=device, dtype=cfg["dtype"])

    if not cfg["base_model_ckpt"]:
        raise ValueError("cfg['base_model_ckpt'] is None. Set it to your trained model checkpoint path.")

    ckpt = torch.load(cfg["base_model_ckpt"], map_location="cpu")
    # your training saved {"model": state_dict, ...}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    base_cfg = get_default_cfg()

    # set these once here (or set them inside config.py defaults)
    base_cfg["data_dir"] = base_cfg["data_dir"] or "data"
    base_cfg["base_model_ckpt"] = base_cfg["base_model_ckpt"] or "results/set_transformer/trial/model.pt"
    base_cfg["wandb_project"] = base_cfg.get("wandb_project", "sae-set-transformer")
    
    # ---- single run: TopK SAE with k=16 ----
    base_cfg["sae_type"] = "topk"
    base_cfg["top_k"] = 16
    base_cfg = post_init_cfg(base_cfg)

    set_seed(base_cfg["seed"])

    # ---- data loader for activation extraction ----
    train_loader, _, _ = make_split_dataloader(
        split="train",
        data_dir=base_cfg["data_dir"],
        batch_size=base_cfg["data_batch_size"],
        num_workers=base_cfg["num_workers"],
        pin_memory=base_cfg["pin_memory"],
        train_drop_last=True,
        split_seed=1312,
    )

    # ---- load frozen base model ----
    base_model = load_base_model(base_cfg)

    cfg = dict(base_cfg)
    sae = build_sae(cfg)
    activations_store = ActivationsStoreST(base_model, train_loader, cfg)
    print(f"\n=== Training SAE: {cfg['name']} ===")
    train_sae(sae, activations_store, base_model, cfg)


if __name__ == "__main__":
    main()
