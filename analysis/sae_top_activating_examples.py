# scripts/sae_top_activating_examples.py
import argparse
import heapq
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
import pandas as pd

from framework.data import make_split_dataloader
from framework.models import MultiHeadSetTransformer
from sae.config import get_default_cfg, post_init_cfg
from sae.sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
import numpy as np
import numpy as np


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


def add_globe_shading(fig, ax, strength=0.32, highlight=0.24,
                      hl_xy=(0.33, 0.68), hl_sigma=0.13,
                      outline=True, outline_alpha=0.45):
    fig.canvas.draw()
    pos = ax.get_position()

    shade = fig.add_axes(pos, frameon=False)
    shade.set_axis_off()
    shade.set_xlim(0, 1)
    shade.set_ylim(0, 1)
    shade.patch.set_alpha(0)
    shade.set_zorder(ax.get_zorder() + 100)

    clip_circle = mpatches.Circle((0.5, 0.5), 0.5, transform=shade.transAxes,
                                  facecolor="none", edgecolor="none")
    shade.add_patch(clip_circle)

    n = 700
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)

    # Dark vignette (reduced)
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.5
    r = np.clip(r, 0, 1)
    alpha_v = strength * (r ** 2.0)  # slightly softer than 2.2

    rgba_v = np.zeros((n, n, 4))      # black overlay
    rgba_v[..., 3] = alpha_v
    im_v = shade.imshow(rgba_v, origin="lower", extent=(0, 1, 0, 1),
                        interpolation="bilinear")
    im_v.set_clip_path(clip_circle)

    # Highlight (a bit stronger)
    hx, hy = hl_xy
    d2 = (xx - hx) ** 2 + (yy - hy) ** 2
    hl = np.exp(-d2 / (2 * hl_sigma ** 2))
    alpha_h = highlight * hl

    rgba_h = np.ones((n, n, 4))       # white overlay
    rgba_h[..., 3] = alpha_h
    im_h = shade.imshow(rgba_h, origin="lower", extent=(0, 1, 0, 1),
                        interpolation="bilinear")
    im_h.set_clip_path(clip_circle)

    if outline:
        shade.add_patch(
            mpatches.Circle((0.5, 0.5), 0.5, transform=shade.transAxes,
                            fill=False, edgecolor="black", linewidth=1.0, alpha=outline_alpha)
        )

    return shade


def make_3d_globe_with_points(
    lon, lat, idx=None,
    central_longitude=-35, central_latitude=25,
    # more vivid defaults
    ocean="#2E7DFF",
    land="#2CCB6D",
    point_color="#A57DB1",
    figsize=(4.2, 4.2), dpi=220,
):
    if idx is None:
        idx = np.arange(len(lon))

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude, central_latitude))
    ax.set_axis_off()
    ax.set_global()

    # Circular boundary
    theta = np.linspace(0, 2*np.pi, 512)
    circle = np.vstack([0.5 + 0.5*np.cos(theta), 0.5 + 0.5*np.sin(theta)]).T
    ax.set_boundary(mpath.Path(circle), transform=ax.transAxes)

    # Basemap (keep clean, vivid)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=ocean, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"),  facecolor=land,  edgecolor="none", zorder=1)
    ax.coastlines(resolution="110m", linewidth=0.6, alpha=0.9, zorder=2)

    # Points
    ax.scatter(
        np.asarray(lon)[idx], np.asarray(lat)[idx],
        transform=ccrs.PlateCarree(),
        s=15,
        color=point_color,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.30,
        zorder=10,
    )

    # Shading (reduced darkness)
    add_globe_shading(
        fig, ax,
        strength=0.4,      # <- less dark edges
        highlight=0.24,     # <- brighter highlight
        hl_xy=(0.33, 0.68),
        hl_sigma=0.13,
        outline=True,
        outline_alpha=0.45
    )

    return fig, ax


def parse_int_list(s):
    if s is None or str(s).strip() == "":
        return None
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model_ckpt", type=str, required=True)
    ap.add_argument("--sae_ckpt", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="sae_top_examples")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--split_seed", type=int, default=1312)

    ap.add_argument("--set_size", type=int, default=7)
    ap.add_argument("--dim_input", type=int, default=256)

    # SAE overrides (optional)
    ap.add_argument("--sae_type", type=str, default=None)
    ap.add_argument("--dict_size", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--input_unit_norm", action="store_true")

    # feature selection
    ap.add_argument("--feature_ids", type=str, default=None, help="comma-separated e.g. 10,55,123")
    ap.add_argument("--feature_stats_csv", type=str, default=None, help="feature_stats.csv from your histogram script")
    ap.add_argument("--min_freq", type=float, default=0.0)
    ap.add_argument("--max_freq", type=float, default=1.0)
    ap.add_argument("--num_features", type=int, default=50, help="how many features to analyze (if using csv)")

    # analysis controls
    ap.add_argument("--max_batches", type=int, default=1000)
    ap.add_argument("--token_idx", type=int, default=-1, help="-1 = all rows; else 0..K-1")
    ap.add_argument("--top_n", type=int, default=16)

    args = ap.parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load base model ----
    model = MultiHeadSetTransformer(dim_input=args.dim_input, set_size=args.set_size).to(device)
    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    # ---- load SAE ----
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
    sae_state = torch.load(args.sae_ckpt, map_location="cpu")
    sae.load_state_dict(sae_state["model"] if isinstance(sae_state, dict) and "model" in sae_state else sae_state)
    sae.eval()

    # ---- loader ----
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

    # ---- load lat/lon metadata ----
    # The user states clear_lat.npy and clear_lon.npy exist in data_dir.
    # These are likely (N_atm,) or similar. The dataset mapping is:
    #   idx -> atm_idx = idx // 7
    # So we need to load these to look up coordinates later.
    lat_path = Path(args.data_dir) / "clear_lat.npy"
    lon_path = Path(args.data_dir) / "clear_lon.npy"
    if not lat_path.exists() or not lon_path.exists():
        print(f"[WARN] lat/lon files not found in {args.data_dir}, map plots will fail or be skipped.")
        lat_data = None
        lon_data = None
    else:
        lat_data = np.load(lat_path, mmap_mode="r")
        lon_data = np.load(lon_path, mmap_mode="r")

    # ---- choose features ----
    feature_ids = parse_int_list(args.feature_ids)

    if feature_ids is None:
        if args.feature_stats_csv is None:
            raise ValueError("Provide --feature_ids or --feature_stats_csv")

        df = pd.read_csv(args.feature_stats_csv)
        df = df[(df["activation_freq"] >= args.min_freq) & (df["activation_freq"] <= args.max_freq)]
        df = df.sort_values(["activation_freq", "mean_when_active"], ascending=[False, False]).head(args.num_features)
        feature_ids = df["feature_id"].astype(int).tolist()

    feature_ids = [f for f in feature_ids if 0 <= f < cfg["dict_size"]]
    if len(feature_ids) == 0:
        raise ValueError("No valid feature_ids selected.")

    feature_ids_t = torch.tensor(feature_ids, dtype=torch.long, device="cpu")

    # ---- heaps: per feature keep top-N examples ----
    # heap item: (activation_value, counter, payload_dict)
    heaps = {fid: [] for fid in feature_ids}
    counter = 0

    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break

        x, y1, y2, y3, temp = batch
        x = x.to(device, non_blocking=True)

        out = model(x, return_h=True)
        if isinstance(out, (tuple, list)) and len(out) >= 4:
            h = out[3]  # (B,K,H)
        else:
            raise RuntimeError("Expected model(x, return_h=True) to return (..., h) as 4th output")

        B, K, H = h.shape
        if args.token_idx >= 0:
            h_use = h[:, args.token_idx, :]         # (B,H)
            vec_to_b = torch.arange(B, device=device)
            vec_to_k = torch.full((B,), args.token_idx, device=device, dtype=torch.long)
        else:
            h_use = h.reshape(B * K, H)             # (B*K,H)
            idx = torch.arange(B * K, device=device)
            vec_to_b = (idx // K)
            vec_to_k = (idx % K)

        sae_out = sae(h_use)
        acts = sae_out["feature_acts"]              # (Nvec, dict)

        # only selected features
        acts_sel = acts[:, feature_ids_t.to(acts.device)]  # (Nvec, Fsel)

        # for each feature, take top candidates from this batch and update global heap
        Nvec = acts_sel.size(0)
        k_take = min(args.top_n, Nvec)

        acts_sel_cpu = acts_sel.detach().cpu()

        for j, fid in enumerate(feature_ids):
            col = acts_sel_cpu[:, j]
            if k_take <= 0:
                continue
            topv, topi = torch.topk(col, k=k_take, largest=True, sorted=False)

            for vv, ii in zip(topv.tolist(), topi.tolist()):
                #print(len(heaps[fid]))
                if vv <= 0:
                    continue  # only care about active examples

                b_idx = int(vec_to_b[ii].item())
                k_idx = int(vec_to_k[ii].item())

                payload = {
                    "feature_id": int(fid),
                    "activation": float(vv),
                    "batch_index": int(bi),
                    "b": b_idx,
                    "k": k_idx,
                    "x_full": x[b_idx].detach().cpu(),       # (K,256)
                    "x_row": x[b_idx, k_idx].detach().cpu(), # (256,)
                }

                # Add lat/lon if available
                if lat_data is not None and lon_data is not None:
                    # Map batch index to original dataset index
                    # bi is batch index, ii is index within flatten batch?
                    # No, vec_to_b[ii] is the batch index within the batch (0..B-1).
                    # We need the index in the loader.
                    
                    # Current batch start index in loader:
                    batch_start_idx = bi * args.batch_size
                    # Index in the batch (0..B-1):
                    in_batch_idx = b_idx # This is the example index in the batch
                    
                    loader_idx = batch_start_idx + in_batch_idx
                    
                    # The loader operates on test_ds which is a Subset
                    # We need to get the index in the underlying AtmosphericDataset
                    if hasattr(loader.dataset, "indices"):
                        # It is a subset
                        if loader_idx < len(loader.dataset.indices):
                            dataset_idx = loader.dataset.indices[loader_idx]
                            atm_idx = dataset_idx // 7
                            payload["lat"] = float(lat_data[atm_idx])
                            payload["lon"] = float(lon_data[atm_idx])
                    else:
                        # Fallback if not a subset (e.g. if validation/sets changed)
                        dataset_idx = loader_idx
                        atm_idx = dataset_idx // 7
                        if atm_idx < len(lat_data):
                            payload["lat"] = float(lat_data[atm_idx])
                            payload["lon"] = float(lon_data[atm_idx])

                heap = heaps[fid]
                item = (float(vv), counter, payload)
                counter += 1

                if len(heap) < args.top_n:
                    heapq.heappush(heap, item)
                else:
                    if vv > heap[0][0]:
                        heapq.heapreplace(heap, item)

    # ---- save results ----
    top_examples = {}
    for fid, heap in heaps.items():
        heap_sorted = sorted(heap, key=lambda t: t[0], reverse=True)
        top_examples[fid] = [it[2] for it in heap_sorted]
        #print(f"Feature {fid}: {len(top_examples[fid])} top examples")

    torch.save(
        {
            "config": {
                "feature_ids": feature_ids,
                "top_n": args.top_n,
                "token_idx": args.token_idx,
                "max_batches": args.max_batches,
            },
            "examples": top_examples,
        },
        out_dir / "top_examples.pt",
    )

    # ---- plots: grid per feature ----
    def plot_feature(fid, exs):
        n = len(exs)
        if n == 0:
            return

        cols = 4
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axs = [axs] if cols == 1 else [*axs]
        axs = (axs if isinstance(axs, list) else axs.flatten())

        for i, ex in enumerate(exs):
            ax = axs[i]
            x_full = ex["x_full"].numpy()  # (K,256)
            k_idx = ex["k"]
            # Plot spectra (one line per row). Highlight the selected row k_idx.
            K = x_full.shape[0]
            for r in range(K):
                if r == k_idx:
                    ax.plot(x_full[r], linewidth=2.0, alpha=1.0, label=f"k={k_idx}")
                else:
                    ax.plot(x_full[r], linewidth=0.8, alpha=0.25, color="gray")
            ax.set_title(f"a={ex['activation']:.4f}  k={k_idx}")
            ax.set_xlabel("lambda_idx")
            ax.set_ylabel("radiance")
            ax.legend(loc="best", fontsize=8, frameon=False)

        for j in range(n, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"feature_{fid:05d}_top.png", dpi=200)
        plt.close(fig)

    for fid in feature_ids:

        plot_feature(fid, top_examples.get(fid, []))

    # ---- map plots ----
    def save_feature_points_csv(fid, exs):
        rows = []
        for ex in exs:
            if "lat" not in ex or "lon" not in ex:
                continue
            rows.append(
                {
                    "feature_id": int(fid),
                    "lat": float(ex["lat"]),
                    "lon": float(ex["lon"]),
                    "activation": float(ex.get("activation", float("nan"))),
                    "batch_index": int(ex.get("batch_index", -1)),
                    "b": int(ex.get("b", -1)),
                    "k": int(ex.get("k", -1)),
                }
            )
        if len(rows) == 0:
            return
        df_pts = pd.DataFrame(rows)
        df_pts.to_csv(out_dir / f"feature_{fid:05d}_points.csv", index=False)

    def plot_feature_map(fid, exs):
        # Extract lat/lons
        lats = [ex["lat"] for ex in exs if "lat" in ex]
        lons = [ex["lon"] for ex in exs if "lon" in ex]
        
        if len(lats) == 0:
            return

        lats = np.array(lats)
        lons = np.array(lons)

        # Calculate central point
        # Simple mean for now. Be aware of date line issues if points are spread across 180/-180.
        cen_lon = float(np.median(lons))
        cen_lat = float(np.median(lats))

        print(cen_lon, cen_lat)
        cen_lon = -79.138286
        cen_lat= -20.991809


        fig, ax = make_3d_globe_with_points(
            lons, lats,
            central_longitude=cen_lon,
            central_latitude=cen_lat
        )
        
        plt.title(f"Feature {fid} Top Locations")
        plt.savefig(out_dir / f"feature_{fid:05d}_map.png", dpi=200)
        plt.close(fig)

    if lat_data is not None:
        for fid in feature_ids:
            save_feature_points_csv(fid, top_examples.get(fid, []))
            plot_feature_map(fid, top_examples.get(fid, []))

    print(f"[OK] Saved: {(out_dir / 'top_examples.pt').resolve()}")
    print(f"[INFO] analyzed {len(feature_ids)} features, top_n={args.top_n}")


if __name__ == "__main__":
    main()
