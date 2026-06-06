uv run python -m analysis.find_sae_clusters \
  --data_dir data \
  --base_ckpt results/set_transformer/trial/best_model.pt \
  --sae_ckpt checkpoints/ST_encout_H256_D4096_topk_k16_lr0.0003_19999/sae.pt \
  --out_dir sae_mining \
  --n_clusters 64 --top_clusters 24 --examples_per_cluster 12
