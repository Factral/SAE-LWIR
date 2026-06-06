# config.py
import torch

def get_default_cfg():
    cfg = {
        "seed": 49,

        # device / dtype
        "device": "cuda",
        "dtype": torch.float32,

        # ===== Base model (the frozen model you interpret) =====
        "base_model_ckpt": "results/set_transformer/trial/best_model.pt",      # path to your trained set-transformer checkpoint (.pt)
        "set_size": 7,
        "dim_input": 256,
        "dim_hidden": 256,            # encoder output dim (this is act_size)

        # ===== Data for extracting activations =====
        "data_dir": "data",
        "data_batch_size": 512,       # batch size for x loader (NOT SAE batch size)
        "num_workers": 8,
        "pin_memory": True,

        # ===== SAE training =====
        "act_size": 256,              # must match dim_hidden
        "dict_size": 256 * 16,         # 16x overcomplete to start
        "batch_size": 4096,            # SAE minibatch over TOKENS (B*K)
        "lr": 3e-4,

        "sae_type": "topk",       # "topk" | "batchtopk" | "vanilla" | "jumprelu"
        "top_k": 32,
        "l1_coeff": 0.0,               # used by vanilla / jumprelu (you can keep 0 for topk/batchtopk)
        "input_unit_norm": True,

        "num_batches_in_buffer": 50,
        "n_batches_to_dead": 5,
        "top_k_aux": 512,
        "aux_penalty": 1/32,

        # How long to train
        "num_steps": 20000,            # change as you want

        # logging / ckpt
        "wandb_project": "sae-set-transformer",
        "run_group": "sweep1",
        "checkpoint_freq": 2000,
        "perf_log_freq": 500,

        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 10.0,
    }
    return post_init_cfg(cfg)

def post_init_cfg(cfg):
    cfg["act_size"] = int(cfg["dim_hidden"])  # keep consistent

    # A readable name for wandb/checkpoints
    cfg["name"] = (
        f"ST_encout_H{cfg['act_size']}_D{cfg['dict_size']}_"
        f"{cfg['sae_type']}_k{cfg.get('top_k','NA')}_lr{cfg['lr']}"
    )
    return cfg
