#!/bin/bash
# Fisher-Gumbel ablation study: run each ablation sequentially on GPUs 0-3
set -e
cd /mnt/data/noahcylich/MatFormer-OLMo
source /mnt/data/noahcylich/venv/bin/activate

BASE=configs/pile-tiny-hmat-gumbel-long.yaml
SAVE=/mnt/data/noahcylich/checkpoints/fisher-ablations
COMMON="--max_duration=540 --save_interval_unsharded=540 --save_interval=100000 --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=100 --save_overwrite=true --hmat.method=fisher_gumbel --hmat.gumbel_tau_start=0.5 --hmat.gumbel_init_scale=1.1 --hmat.fisher_warmup_frac=0.15 --hmat.fisher_ema_beta=0.99"
mkdir -p $SAVE

echo "Baseline reference: 155.9 / 161.8 / 167.2 / 172.9"
echo "Phase 2 best:       157.7 / 158.2 / 162.2 / 167.1"
echo "Fisher 15% base:    166.7 / 166.7 / 170.7 / 175.9"
echo ""

# Ablation A: Option A smooth blending (every step, blend=0.05)
echo "=== Ablation A: Smooth blend (every step, blend=0.05) ==="
torchrun --nproc_per_node=4 scripts/train.py $BASE $COMMON \
  --save_folder=$SAVE/smooth-blend \
  --hmat.fisher_update_interval=0 --hmat.fisher_logit_blend=0.05 \
  > $SAVE/smooth-blend.log 2>&1
echo "  Results:"
grep 'eval/pile-val/Perplexity' $SAVE/smooth-blend.log | tail -4
echo ""

# Ablation B: 30% warmup (#4)
echo "=== Ablation B: 30% warmup ==="
torchrun --nproc_per_node=4 scripts/train.py $BASE $COMMON \
  --save_folder=$SAVE/warmup-30pct \
  --hmat.fisher_warmup_frac=0.30 \
  > $SAVE/warmup-30pct.log 2>&1
echo "  Results:"
grep 'eval/pile-val/Perplexity' $SAVE/warmup-30pct.log | tail -4
echo ""

# Ablation C: Log-scaled Fisher (#5)
echo "=== Ablation C: Log-scaled Fisher logits ==="
torchrun --nproc_per_node=4 scripts/train.py $BASE $COMMON \
  --save_folder=$SAVE/log-scaled \
  --hmat.fisher_logit_mode=log \
  > $SAVE/log-scaled.log 2>&1
echo "  Results:"
grep 'eval/pile-val/Perplexity' $SAVE/log-scaled.log | tail -4
echo ""

# Ablation D: Factor-1 only Fisher (#3)
echo "=== Ablation D: Factor-1 only Fisher accumulation ==="
torchrun --nproc_per_node=4 scripts/train.py $BASE $COMMON \
  --save_folder=$SAVE/factor1-only \
  --hmat.fisher_factor1_only=true \
  > $SAVE/factor1-only.log 2>&1
echo "  Results:"
grep 'eval/pile-val/Perplexity' $SAVE/factor1-only.log | tail -4
echo ""

echo "=== ALL ABLATIONS COMPLETE ==="
echo ""
echo "Summary:"
printf "%-30s %s\n" "Config" "PPL (1/1 / 1/2 / 1/4 / 1/8)"
printf "%-30s %s\n" "------" "----------------------------"
for name in smooth-blend warmup-30pct log-scaled factor1-only; do
  ppls=$(grep 'eval/pile-val/Perplexity' $SAVE/$name.log | tail -4 | grep -oP 'Perplexity=\K[0-9.]+' | tr '\n' ' ')
  printf "%-30s %s\n" "$name" "$ppls"
done
