import argparse
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from framework.data import make_split_dataloader
from framework.models import MultiHeadSetTransformer

# Wavelengths for x-axis (length should match args.dim / spectra length)
WAVELENGTHS = np.array([
    8.104948406, 8.124733705, 8.144519003, 8.164304302, 8.1840896, 8.203874899,
    8.223660198, 8.243446609, 8.263234006, 8.283022643, 8.302812392, 8.322603256,
    8.342395232, 8.362188449, 8.381982908, 8.401778493, 8.421575319, 8.44137326,
    8.46117239, 8.480972866, 8.500774532, 8.52057744, 8.54038159, 8.560186856,
    8.579993366, 8.599801118, 8.619610114, 8.639420227, 8.659231583, 8.679044184,
    8.698857902, 8.718672864, 8.738488945, 8.758306251, 8.778124587, 8.79794415,
    8.817764831, 8.837586658, 8.857409588, 8.877233639, 8.897058681, 8.916884915,
    8.936712127, 8.956540495, 8.976369776, 8.996200176, 9.016031444, 9.035863833,
    9.055697235, 9.075531496, 9.095366751, 9.115203001, 9.13504018, 9.154878303,
    9.174717168, 9.194557028, 9.214397756, 9.234239338, 9.254081809, 9.273925037,
    9.293769135, 9.313613974, 9.333459717, 9.353306178, 9.373153436, 9.393001392,
    9.412850091, 9.432699489, 9.452549683, 9.472400469, 9.492252033, 9.512104205,
    9.531957141, 9.551810561, 9.571664749, 9.591519573, 9.611374864, 9.63123095,
    9.651087541, 9.670944577, 9.690802355, 9.710660582, 9.730519444, 9.750378815,
    9.770238709, 9.790099088, 9.809959972, 9.829821359, 9.849683253, 9.8695455,
    9.889408404, 9.909271689, 9.929135298, 9.948999465, 9.968864066, 9.988729033,
    10.00859445, 10.02846025, 10.04832646, 10.06819313, 10.08806008, 10.10792739,
    10.12779522, 10.1476633, 10.16753173, 10.18740057, 10.20726961, 10.22713908,
    10.24700892, 10.26687908, 10.2867495, 10.30662028, 10.32649138, 10.34636275,
    10.36623438, 10.38610637, 10.40597874, 10.42585124, 10.44572412, 10.46559725,
    10.48547064, 10.50534425, 10.52521809, 10.54509241, 10.56496687, 10.58484158,
    10.60471654, 10.62459182, 10.64446735, 10.66434304, 10.68421907, 10.70409536,
    10.72397178, 10.74384858, 10.76372549, 10.78360279, 10.80348022, 10.82335803,
    10.84323597, 10.8631141, 10.8829926, 10.9028713, 10.92275022, 10.94262938,
    10.96250883, 10.98238854, 11.0022684, 11.02214861, 11.04202907, 11.06190966,
    11.08179064, 11.10167172, 11.1215532, 11.14143493, 11.16131691, 11.18119915,
    11.20108158, 11.22096433, 11.24084737, 11.2607307, 11.28061424, 11.30049812,
    11.32038227, 11.34026677, 11.36015154, 11.38003656, 11.39992196, 11.4198076,
    11.43969363, 11.45957991, 11.47946654, 11.49935358, 11.51924104, 11.53912867,
    11.55901673, 11.5789051, 11.59879378, 11.61868298, 11.63857255, 11.65846241,
    11.67835275, 11.69824348, 11.71813469, 11.73802629, 11.75791827, 11.77781075,
    11.79770375, 11.81759712, 11.83749097, 11.85738536, 11.87728021, 11.89717561,
    11.91707145, 11.93696799, 11.95686497, 11.97676252, 11.99666075, 12.01655944,
    12.03645889, 12.05635887, 12.07625948, 12.09616084, 12.11606282, 12.13596545,
    12.15586882, 12.17577296, 12.19567785, 12.21558351, 12.23548991, 12.25539718,
    12.27530521, 12.29521429, 12.31512409, 12.33503495, 12.35494657, 12.37485921,
    12.39477291, 12.41468758, 12.43460331, 12.45452008, 12.47443805, 12.49435719,
    12.51427667, 12.53419616, 12.55411564, 12.57403512, 12.5939546, 12.61387409,
    12.63379357, 12.65371305, 12.67363253, 12.69355202, 12.7134715, 12.73339098,
    12.75331046, 12.77322995, 12.79314943, 12.81306891, 12.83298839, 12.85290787,
    12.87282736, 12.89274684, 12.91266632, 12.9325858, 12.95250529, 12.97242477,
    12.99234425, 13.01226373, 13.03218322, 13.0521027, 13.07202218, 13.09194166,
    13.11186115, 13.13178063, 13.15170011, 13.17161959,
], dtype=float)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--set_size", type=int, default=7)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--split_seed", type=int, default=1312)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to visualize")
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Larger, more readable plot fonts (applies to PNG + SVG outputs)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 16,
        }
    )

    # X-axis values (wavelengths). Fallback to index if length mismatches.
    if len(WAVELENGTHS) != args.dim:
        print(f"[WARN] len(WAVELENGTHS)={len(WAVELENGTHS)} != dim={args.dim}; using index x-axis.")
        x_axis = np.arange(args.dim, dtype=float)
    else:
        x_axis = WAVELENGTHS

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Data
    _, _, test_loader = make_split_dataloader(
        split="train", # Argument name needs to be fixed in make_split_dataloader call if we want test split? 
                       # Looking at data.py, make_split_dataloader returns train, val, test loaders regardless of 'split' arg 
                       # but uses 'split' arg for some validation? 
                       # Actually, looking at data.py:81, it checks split in {"train", "val", "test"}.
                       # But it returns all three loaders.
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        split_seed=args.split_seed,
    )

    # Load Model
    model = MultiHeadSetTransformer(dim_input=args.dim, set_size=args.set_size).to(device)
    
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Handle both full checkpoint (with 'model' key) and state_dict only
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()

    sse_t, sse_u, sse_d = 0.0, 0.0, 0.0
    n_t, n_u, n_d = 0, 0, 0
    
    sample_count = 0
    
    print("Starting evaluation...")
    
    for x, y1, y2, y3, temp in tqdm(test_loader):
        x = x.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        y3 = y3.to(device)

        T, U, D = model(x)
        
        # Metrics accumulation
        sse_t += F.mse_loss(T, y1, reduction="sum").item()
        sse_u += F.mse_loss(U, y2, reduction="sum").item()
        sse_d += F.mse_loss(D, y3, reduction="sum").item()
        n_t += T.numel()
        n_u += U.numel()
        n_d += D.numel()
        
        # Plotting
        bs = x.size(0)
        for i in range(bs):
            if args.limit is not None and sample_count >= args.limit:
                break
                
            zoom_start, zoom_end = 50, 100
            zoom_slice = slice(zoom_start, zoom_end + 1)

            def _ylim_from_range(y_min, y_max):
                if y_max <= y_min:
                    pad = 1e-6 if y_min == 0 else abs(y_min) * 1e-6
                else:
                    pad = 0.03 * (y_max - y_min)
                return (y_min - pad, y_max + pad)

            # Move tensors to numpy once per sample (reused for both full + zoom plots)
            t_pred = [T[i, j].detach().cpu().numpy() for j in range(T.size(1))]
            t_gt = [y1[i, j].detach().cpu().numpy() for j in range(y1.size(1))]
            u_pred = [U[i, j].detach().cpu().numpy() for j in range(U.size(1))]
            u_gt = [y2[i, j].detach().cpu().numpy() for j in range(y2.size(1))]
            d_pred = D[i, 0].detach().cpu().numpy()
            d_gt = y3[i, 0].detach().cpu().numpy()

            # Precompute y-limits for zoom based on values within [zoom_start, zoom_end]
            t_min = min(min(a[zoom_slice].min() for a in t_pred), min(a[zoom_slice].min() for a in t_gt))
            t_max = max(max(a[zoom_slice].max() for a in t_pred), max(a[zoom_slice].max() for a in t_gt))
            u_min = min(min(a[zoom_slice].min() for a in u_pred), min(a[zoom_slice].min() for a in u_gt))
            u_max = max(max(a[zoom_slice].max() for a in u_pred), max(a[zoom_slice].max() for a in u_gt))
            d_min = min(d_pred[zoom_slice].min(), d_gt[zoom_slice].min())
            d_max = max(d_pred[zoom_slice].max(), d_gt[zoom_slice].max())

            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            # Use a fixed color per set element j (matches prior C{j} behavior / sample_243.png)
            cycle_colors = plt.rcParams.get("axes.prop_cycle", None)
            cycle_colors = cycle_colors.by_key().get("color", []) if cycle_colors is not None else []

            # Plot 1: Transmittance (T)
            # T shape: (B, 7, 256)
            for j in range(len(t_pred)):
                c = cycle_colors[j % len(cycle_colors)] if len(cycle_colors) > 0 else None
                axs[0].plot(x_axis, t_pred[j], "--", alpha=0.8, color=c)
                axs[0].plot(x_axis, t_gt[j], "-", alpha=0.6, color=c)
            axs[0].set_title("Transmittance (T): Pred (dashed) vs GT (solid)")
            axs[0].set_ylabel("")
            
            # Plot 2: Upwelling (U)
            # U shape: (B, 7, 256)
            for j in range(len(u_pred)):
                c = cycle_colors[j % len(cycle_colors)] if len(cycle_colors) > 0 else None
                axs[1].plot(x_axis, u_pred[j], "--", alpha=0.8, color=c)
                axs[1].plot(x_axis, u_gt[j], "-", alpha=0.6, color=c)
            axs[1].set_title("Upwelling (U): Pred (dashed) vs GT (solid)")
            axs[1].set_ylabel("Microflicks")
            
            # Plot 3: Downwelling (D)
            # D shape: (B, 1, 256)
            # Match run.py style: let matplotlib assign default colors (typically blue/orange)
            axs[2].plot(x_axis, d_pred, "--", alpha=0.8, label="D_pred")
            axs[2].plot(x_axis, d_gt, "-", alpha=0.6, label="D_gt")
            axs[2].set_title("Downwelling (D): Pred (dashed) vs GT (solid)")
            axs[2].set_ylabel("Microflicks")
            axs[2].set_xlabel("wavelength (µm)")
            axs[2].legend()
            
            plt.tight_layout()
            base = os.path.join(args.output_dir, f"sample_{sample_count}")
            plt.savefig(base + ".png")
            plt.savefig(base + ".svg")
            plt.close(fig)

            # Zoomed-in SVG: bands 70 to 100, y-axis fit to min/max in that window
            fig_z, axs_z = plt.subplots(3, 1, figsize=(10, 12))

            for j in range(len(t_pred)):
                c = cycle_colors[j % len(cycle_colors)] if len(cycle_colors) > 0 else None
                axs_z[0].plot(x_axis, t_pred[j], "--", alpha=0.8, color=c)
                axs_z[0].plot(x_axis, t_gt[j], "-", alpha=0.6, color=c)
            axs_z[0].set_title(f"Transmittance (T): Pred (dashed) vs GT (solid) [zoom {zoom_start}-{zoom_end}]")
            axs_z[0].set_xlim(float(x_axis[zoom_start]), float(x_axis[zoom_end]))
            axs_z[0].set_ylim(*_ylim_from_range(t_min, t_max))
            axs_z[0].set_ylabel("")

            for j in range(len(u_pred)):
                c = cycle_colors[j % len(cycle_colors)] if len(cycle_colors) > 0 else None
                axs_z[1].plot(x_axis, u_pred[j], "--", alpha=0.8, color=c)
                axs_z[1].plot(x_axis, u_gt[j], "-", alpha=0.6, color=c)
            axs_z[1].set_title(f"Upwelling (U): Pred (dashed) vs GT (solid) [zoom {zoom_start}-{zoom_end}]")
            axs_z[1].set_xlim(float(x_axis[zoom_start]), float(x_axis[zoom_end]))
            axs_z[1].set_ylim(*_ylim_from_range(u_min, u_max))
            axs_z[1].set_ylabel("Microflicks")

            axs_z[2].plot(x_axis, d_pred, "--", alpha=0.8, label="D_pred")
            axs_z[2].plot(x_axis, d_gt, "-", alpha=0.6, label="D_gt")
            axs_z[2].set_title(f"Downwelling (D): Pred (dashed) vs GT (solid) [zoom {zoom_start}-{zoom_end}]")
            axs_z[2].set_xlim(float(x_axis[zoom_start]), float(x_axis[zoom_end]))
            axs_z[2].set_ylim(*_ylim_from_range(d_min, d_max))
            axs_z[2].set_ylabel("Microflicks")
            axs_z[2].set_xlabel("wavelength (µm)")
            axs_z[2].legend()

            plt.tight_layout()
            plt.savefig(base + f"_zoom_{zoom_start}_{zoom_end}.svg")
            plt.close(fig_z)
            
            sample_count += 1
        if args.limit is not None and sample_count >= args.limit:
            break

    mse_t = sse_t / max(n_t, 1)
    mse_u = sse_u / max(n_u, 1)
    mse_d = sse_d / max(n_d, 1)
    
    print("-" * 30)
    print("Evaluation Results:")
    print(f"MSE T: {mse_t:.6f}")
    print(f"MSE U: {mse_u:.6f}")
    print(f"MSE D: {mse_d:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
