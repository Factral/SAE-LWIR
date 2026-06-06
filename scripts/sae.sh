uv run python -m analysis.sae_histograms \
  --data_dir data \
  --model_ckpt results/set_transformer/trial/best_model.pt \
  --sae_ckpt checkpoints/ST_encout_H256_D4096_topk_k16_lr0.0003_19999/sae.pt \
  --token_idx 3 \
  --feature_id 556 \
  --out_dir sae_analysis/token3 \
  --sae_type topk
