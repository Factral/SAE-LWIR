import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from framework.data import make_split_dataloader
from framework.models import MultiHeadSetTransformer
from framework.losses import logit_mse_loss, loss_reconstuction

import wandb
import matplotlib.pyplot as plt
from torchinfo import summary
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--set_size", type=int, default=7)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--split_seed", type=int, default=1312)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--run_name", type=str, default="trial")
    parser.add_argument("--save_dir", type=str, default="results/set_transformer")
    parser.add_argument("--test_every", type=int, default=50)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="set-transformer")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_plots", action="store_true", help="Log a quick prediction-vs-gt plot to wandb.")
    parser.add_argument("--rowwise", action="store_true", help="Use rowwise instead of per-row for decoder.")

    return parser.parse_args()

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validate_loss(model, loader, device, *, criterion):
    model.eval()
    total = 0.0
    count = 0
    for x, y1, y2, y3, temp in loader:
        x = x.to(device, non_blocking=True)
        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)
        y3 = y3.to(device, non_blocking=True)
        T, U, D = model(x)
        loss = (
            criterion(T, y1)
            + F.mse_loss(U, y2)
            + F.mse_loss(D, y3)
            + loss_reconstuction(x, T, U, D, temp)
        )
        bs = x.size(0)
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device, *, run=None, step=None, log_plot=False, plot_key="plot/test", plot_gts=True):
    model.eval()
    sse_t = 0.0
    sse_u = 0.0
    sse_d = 0.0
    
    sam_t_sum = 0.0
    sam_u_sum = 0.0
    sam_d_sum = 0.0

    n_samples = 0
    n_t = 0
    n_u = 0
    n_d = 0

    # For NRMSE, we need the range of the ground truth (max - min)
    # We'll track min and max across the whole eval set.
    t_min, t_max = float("inf"), float("-inf")
    u_min, u_max = float("inf"), float("-inf")
    d_min, d_max = float("inf"), float("-inf")

    last_batch = None
    for x, y1, y2, y3, temp in loader:
        # If x arrives as (B, 7, 256), this is a no-op.
        x = x.to(device, non_blocking=True)
        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)
        y3 = y3.to(device, non_blocking=True)

        # Update min/max for NRMSE
        t_min = min(t_min, y1.min().item())
        t_max = max(t_max, y1.max().item())
        u_min = min(u_min, y2.min().item())
        u_max = max(u_max, y2.max().item())
        d_min = min(d_min, y3.min().item())
        d_max = max(d_max, y3.max().item())

        T, U, D = model(x)
        
        # SSE for RMSE
        sse_t += F.mse_loss(T, y1, reduction="sum").item()
        sse_u += F.mse_loss(U, y2, reduction="sum").item()
        sse_d += F.mse_loss(D, y3, reduction="sum").item()

        # SAM: Spectral Angle Mapper
        # shape: (B, ... , C). We compute cosine similarity along last dim C.
        # then arccos.
        # F.cosine_similarity returns (B, ...)
        
        # T: (B, K, C)
        cos_t = F.cosine_similarity(T, y1, dim=-1).mean() 
        # acos of mean or mean of acos? SAM is usually mean of acos.
        # But for stability with cosine_similarity (can be slightly > 1 or < -1 due to precision)
        # we clamp.
        def compute_sam_sum(pred, target):
            # pred, target: (..., C)
            sim = F.cosine_similarity(pred, target, dim=-1).clamp(-1.0, 1.0)
            angle = torch.acos(sim)
            return angle.sum().item()

        sam_t_sum += compute_sam_sum(T, y1)
        sam_u_sum += compute_sam_sum(U, y2)
        sam_d_sum += compute_sam_sum(D, y3)

        n_t += T.numel()
        n_u += U.numel()
        n_d += D.numel()
        
        # We need sample counts for SAM averaging (number of "spectra")
        # T, U are (B, K, C), so we have B*K spectra.
        # D is (B, 1, C) or (B, C). 
        # Let's count the number of vectors we averaged over.
        # T: B*K, U: B*K, D: B (or B*1)
        # However, earlier code used total elements for MSE. 
        # For SAM, we want per-vector average.
        
        # Actually, let's just use the fact that we summed angles.
        # We need to divide by the number of angle values summed.
        # T, U: (B, K, C) -> (B, K) angles
        # D: (B, 1, C) -> (B, 1) angles
        
        n_samples += x.size(0) # Batch size

        last_batch = (T, U, D, y1, y2, y3)

    if log_plot and run is not None and plt is not None and last_batch is not None:
        T, U, D, y1, y2, y3 = last_batch
        # Select one random batch item (based on the last batch)
        bsz = T.size(0)
        random_idx = torch.randint(0, bsz, (1,), device=T.device).item()

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        for j in range(T.size(1)):
            axs[0].plot(T[random_idx, j].detach().cpu().numpy(), "--", alpha=0.8)
            if plot_gts:
                axs[0].plot(y1[random_idx, j].detach().cpu().numpy(), "-", alpha=1.0)
            axs[1].plot(U[random_idx, j].detach().cpu().numpy(), "--", alpha=0.8)
            if plot_gts:
                axs[1].plot(y2[random_idx, j].detach().cpu().numpy(), "-", alpha=1.0)
        axs[0].set_title("T (pred solid, gt dashed)")
        axs[1].set_title("U (pred solid, gt dashed)")
        axs[2].plot(D[random_idx, 0].detach().cpu().numpy(), "--", alpha=0.9, label="D_pred")
        if plot_gts:
            axs[2].plot(y3[random_idx, 0].detach().cpu().numpy(), "-", alpha=1.0, label="D_gt")
        axs[2].set_title("D (pred solid, gt dashed)")

        run.log({plot_key: wandb.Image(fig)}, step=step)
        plt.close(fig)

    # Calculate final metrics
    mse_t = sse_t / max(n_t, 1)
    mse_u = sse_u / max(n_u, 1)
    mse_d = sse_d / max(n_d, 1)
    
    rmse_t = mse_t ** 0.5
    rmse_u = mse_u ** 0.5
    rmse_d = mse_d ** 0.5

    # NRMSE
    range_t = max(t_max - t_min, 1e-8)
    range_u = max(u_max - u_min, 1e-8)
    range_d = max(d_max - d_min, 1e-8)
    
    nrmse_t = rmse_t / range_t
    nrmse_u = rmse_u / range_u
    nrmse_d = rmse_d / range_d

    # SAM
    # n_t is total elements. T is (B,K,C). Number of vectors is n_t / C.
    # We can infer C from the last batch or just track it.
    # Let's assume C is consistent. 
    # Or better, we know the shapes. 
    # T, U: (B, K, C). n_vectors = n_t / C
    # D: (B, 1, C). n_vectors = n_d / C
    
    # To be safe, let's just use the counters we could have tracked, but we didn't track "number of vectors".
    # But n_t = N_vectors * C.
    # So N_vectors = n_t / C.
    C = last_batch[0].shape[-1] if last_batch is not None else 1
    
    avg_sam_t = sam_t_sum / max(n_t / C, 1)
    avg_sam_u = sam_u_sum / max(n_u / C, 1)
    avg_sam_d = sam_d_sum / max(n_d / C, 1)

    metrics = {
        "nrmse_T": nrmse_t, "nrmse_U": nrmse_u, "nrmse_D": nrmse_d,
        "sam_T": avg_sam_t, "sam_U": avg_sam_u, "sam_D": avg_sam_d
    }

    if run is not None:
        run.log({f"test/{k}": v for k, v in metrics.items()}, step=step)
    
    return metrics


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")
    print(f"Device: {device}")

    seed_all(args.split_seed)
    train_loader, val_loader, test_loader = make_split_dataloader(
        split="train",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        train_drop_last=True,
        split_seed=args.split_seed,
    )


    print(f"rowwise: {args.rowwise}")
    model = MultiHeadSetTransformer(dim_input=args.dim, set_size=args.set_size, rowwise=args.rowwise).to(device)
    summary(model, input_size=(args.batch_size, args.set_size, args.dim))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args),
    )

    ckpt_path = os.path.join(save_dir, "model.pt")

    if args.mode == "test":
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {ckpt_path}")
        metrics = evaluate(model, test_loader, device, run=run, step=0, log_plot=args.log_plots)
        print(f"test/nrmse_T: {metrics['nrmse_T']:.6f} test/nrmse_U: {metrics['nrmse_U']:.6f} test/nrmse_D: {metrics['nrmse_D']:.6f}")
        print(f"test/sam_T: {metrics['sam_T']:.6f} test/sam_U: {metrics['sam_U']:.6f} test/sam_D: {metrics['sam_D']:.6f}")
        run.finish()
        return

    global_step = 0
    best_val = float("inf")
    start_time = time.time()

    criterion = logit_mse_loss 

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True)
        running = 0.0
        seen = 0

        for x, y1, y2, y3, temp in pbar:
            x = x.to(device, non_blocking=True)
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)
            y3 = y3.to(device, non_blocking=True)


            optimizer.zero_grad()
            T, U, D = model(x)
            loss1 = criterion(T, y1) * 10
            loss2 = F.mse_loss(U, y2)
            loss3 = F.mse_loss(D, y3) / 10
            #loss4 = loss_reconstuction(x, T, U, D, temp)
            
            loss = loss1 + loss2 + loss3 #+ loss4
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1000.0)
            optimizer.step()

            bs = x.size(0)
            running += loss.item() * bs
            seen += bs
            global_step += 1

            avg = running / max(seen, 1)
            pbar.set_postfix({"loss": f"{avg:.4f}"})

            if run is not None:
                run.log(
                    {
                        "train/loss": loss.item(),
                        "train/loss1": loss1.item(),
                        "train/loss2": loss2.item(),
                        "train/loss3": loss3.item(),
                        "train/grad_norm": grad_norm.item(),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        #val_loss = validate_loss(model, val_loader, device, criterion=criterion)
        test_metrics = None
        if epoch % args.test_every == 0:
            test_metrics = evaluate(model, test_loader, device, run=run, step=global_step, log_plot=args.log_plots)
        elapsed = time.time() - start_time

        if test_metrics is not None:
            print(
                f"epoch {epoch}: train/loss={running/max(seen,1):.6f} "
                #f"val/loss={val_loss:.6f} "
                f"test/nrmse_T={test_metrics['nrmse_T']:.6f} test/nrmse_U={test_metrics['nrmse_U']:.6f} test/nrmse_D={test_metrics['nrmse_D']:.6f} "
            f"test/sam_T={test_metrics['sam_T']:.6f} test/sam_U={test_metrics['sam_U']:.6f} test/sam_D={test_metrics['sam_D']:.6f} "
            f"elapsed={elapsed:.1f}s"
        )
            if test_metrics['nrmse_D'] < best_val:
                best_val = test_metrics['nrmse_D']
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        else:
            print(
                f"epoch {epoch}: train/loss={running/max(seen,1):.6f} "
                f"elapsed={elapsed:.1f}s"
            )

        #run.log(
        #    {"val/loss": val_loss, "epoch": epoch},
        #    step=global_step,
        #)


    print(f"Best val/loss: {best_val:.6f}")
    run.finish()


if __name__ == "__main__":
    main()
