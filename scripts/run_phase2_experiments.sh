#!/bin/bash
# Run Phase 2 experiments: freeze + fisher-gumbel in parallel
# This script runs ON the VM directly

set -e
cd /mnt/data/noahcylich/MatFormer-OLMo
source /mnt/data/noahcylich/venv/bin/activate

SAVE=/mnt/data/noahcylich/checkpoints/phase2-experiments
mkdir -p $SAVE
BASE=configs/pile-tiny-hmat-gumbel-long.yaml
COMMON="--max_duration=540 --save_interval_unsharded=540 --save_interval=100000 --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 --save_overwrite=true"

echo "Starting Exp A: Phase 2 + Freeze 10%"
torchrun --nproc_per_node=4 scripts/train.py $BASE $COMMON \
  --save_folder=$SAVE/freeze-10pct \
  --hmat.method=gumbel --hmat.gumbel_tau_start=0.5 --hmat.budget_penalty_lambda=0.001 \
  --hmat.gumbel_init_scale=1.1 --hmat.gumbel_freeze_fraction=0.1 \
  > $SAVE/freeze-10pct.log 2>&1 &
PID_A=$!

echo "Starting Exp B: Phase 2.5 Fisher-Guided Gumbel"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29600 scripts/train.py $BASE $COMMON \
  --save_folder=$SAVE/fisher-gumbel \
  --hmat.method=fisher_gumbel --hmat.gumbel_tau_start=0.5 \
  --hmat.gumbel_init_scale=1.1 --hmat.fisher_warmup_frac=0.05 \
  --hmat.fisher_update_interval=50 --hmat.fisher_ema_beta=0.99 \
  > $SAVE/fisher-gumbel.log 2>&1 &
PID_B=$!

echo "Exp A PID=$PID_A, Exp B PID=$PID_B"
echo "Waiting for both to complete..."
wait $PID_A $PID_B
echo "Both experiments complete!"

echo ""
echo "=== RESULTS ==="
echo ""
echo "Baseline reference: 155.9 / 161.8 / 167.2 / 172.9"
echo ""
echo "=== Exp A (freeze 10%) ==="
grep 'eval/pile-val/Perplexity' $SAVE/freeze-10pct.log | tail -4
echo ""
echo "=== Exp B (fisher-gumbel) ==="
grep 'eval/pile-val/Perplexity' $SAVE/fisher-gumbel.log | tail -4
