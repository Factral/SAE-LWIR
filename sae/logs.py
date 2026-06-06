import json
import os

import torch
import wandb


def init_wandb(cfg):
    return wandb.init(
        project=cfg["wandb_project"],
        name=cfg["name"],
        config=cfg,
        reinit=True,
    )


def log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = [
        "loss",
        "l2_loss",
        "l1_loss",
        "l0_norm",
        "l1_norm",
        "aux_loss",
        "num_dead_features",
    ]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}

    # how many dictionary features are completely inactive in this minibatch
    if "feature_acts" in output:
        log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch=None):
    """
    Performance proxy for SetTransformer case:
      compare task loss using original encoder activations h vs SAE-reconstructed h_hat,
      plus zero and mean ablations in h-space.

    Assumes your base model supports:
      model(x, return_h=True) -> (T, U, D, h)
      model(x, h_override=...) -> (T, U, D)
    """
    # Get a batch (x, y1, y2, y3, temp) from the same loader used for activations
    if batch is None:
        if not hasattr(activations_store, "_perf_it"):
            activations_store._perf_it = iter(activations_store.loader)
        try:
            batch = next(activations_store._perf_it)
        except StopIteration:
            activations_store._perf_it = iter(activations_store.loader)
            batch = next(activations_store._perf_it)

    x, y1, y2, y3, _temp = batch
    device = sae.cfg.get("device", getattr(activations_store, "device", "cuda"))
    x = x.to(device, non_blocking=True)
    y1 = y1.to(device, non_blocking=True)
    y2 = y2.to(device, non_blocking=True)
    y3 = y3.to(device, non_blocking=True)

    # Baseline
    T, U, D, h = model(x, return_h=True)

    def task_loss(T_, U_, D_):
        return (
            (T_ - y1).pow(2).mean()
            + (U_ - y2).pow(2).mean()
            + (D_ - y3).pow(2).mean()
        )

    base_loss = task_loss(T, U, D)

    # SAE reconstruction in h-space
    h_flat = h.reshape(-1, h.size(-1))
    h_hat_flat = sae(h_flat)["sae_out"]
    h_hat = h_hat_flat.reshape_as(h)

    try:
        T_hat, U_hat, D_hat = model(x, h_override=h_hat)
    except TypeError:
        # fallback if you implemented encode/decode instead of h_override
        T_hat, U_hat, D_hat = model.decode(h_hat)

    sae_loss = task_loss(T_hat, U_hat, D_hat)

    # Zero ablation
    h_zero = torch.zeros_like(h)
    try:
        T0, U0, D0 = model(x, h_override=h_zero)
    except TypeError:
        T0, U0, D0 = model.decode(h_zero)
    zero_loss = task_loss(T0, U0, D0)

    # Mean ablation (mean over batch and set elements)
    h_mean = h.mean(dim=(0, 1), keepdim=True).expand_as(h)
    try:
        Tm, Um, Dm = model(x, h_override=h_mean)
    except TypeError:
        Tm, Um, Dm = model.decode(h_mean)
    mean_loss = task_loss(Tm, Um, Dm)

    eps = 1e-12
    rec_from_zero = (zero_loss - sae_loss) / (zero_loss - base_loss + eps)
    rec_from_mean = (mean_loss - sae_loss) / (mean_loss - base_loss + eps)

    log_dict = {
        "performance/task_loss_base": base_loss.item(),
        "performance/task_loss_sae": sae_loss.item(),
        "performance/task_loss_zero": zero_loss.item(),
        "performance/task_loss_mean": mean_loss.item(),
        "performance/task_degradation": (sae_loss - base_loss).item(),
        "performance/recovery_from_zero": rec_from_zero.item(),
        "performance/recovery_from_mean": rec_from_mean.item(),
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


def save_checkpoint(wandb_run, sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    json_safe_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            json_safe_cfg[k] = v
        elif isinstance(v, (torch.dtype, type)):
            json_safe_cfg[k] = str(v)
        else:
            json_safe_cfg[k] = str(v)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=f"{cfg['name']}_{step}",
            type="model",
            description=f"Model checkpoint at step {step}",
        )
        artifact.add_file(sae_path)
        artifact.add_file(config_path)
        wandb_run.log_artifact(artifact)

    print(f"Model and config saved at step {step}: {save_dir}")
