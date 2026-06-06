uv run python -m analysis.sae_top_activating_examples \
  --data_dir data \
  --model_ckpt results/set_transformer/trial/best_model.pt \
  --sae_ckpt checkpoints/ST_encout_H256_D4096_topk_k16_lr0.0003_19999/sae.pt\
  --feature_stats_csv sae_analysis/token3/feature_stats.csv \
  --min_freq 0.005 --max_freq 0.05 --num_features 50 \
  --top_n 64 --max_batches 5000 --token_idx 3
