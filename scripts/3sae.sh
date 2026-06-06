uv run python -m analysis.analyze_row_distributions \
  --data_dir data \
  --base_ckpt results/set_transformer/trial/best_model.pt\
  --sae_ckpt checkpoints/ST_encout_H256_D4096_topk_k16_lr0.0003_19999/sae.pt  \
  --out_dir row_analysis \
  --sae_type topk --dict_size 4096 --top_k 32
