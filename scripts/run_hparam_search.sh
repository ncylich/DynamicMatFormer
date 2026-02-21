#!/bin/bash
# Hyperparameter search: 6 trials, 1 GPU each, 540 steps
# Grid: tau_start x lambda = {0.5, 1.0, 2.0} x {0.001, 0.005, 0.01}
# Selected 6 combos for max coverage

set -e
cd /mnt/data/noahcylich/MatFormer-OLMo
source /mnt/data/noahcylich/venv/bin/activate

BASE_CONFIG=configs/pile-tiny-hmat-gumbel-long.yaml
SAVE_BASE=/mnt/data/noahcylich/checkpoints/hparam-search

# Trial 1: tau=0.5, lambda=0.001
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 scripts/train.py $BASE_CONFIG \
  --max_duration=540 --save_interval_unsharded=540 --save_interval=100000 \
  --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 \
  --save_folder=$SAVE_BASE/t1-tau0.5-lam0.001 --save_overwrite=true \
  --hmat.gumbel_tau_start=0.5 --hmat.budget_penalty_lambda=0.001 \
  > $SAVE_BASE/t1.log 2>&1 &
PID1=$!
echo "Trial 1 (tau=0.5, lam=0.001) started: PID=$PID1"

# Trial 2: tau=0.5, lambda=0.005
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 scripts/train.py $BASE_CONFIG \
  --max_duration=540 --save_interval_unsharded=540 --save_interval=100000 \
  --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 \
  --save_folder=$SAVE_BASE/t2-tau0.5-lam0.005 --save_overwrite=true \
  --hmat.gumbel_tau_start=0.5 --hmat.budget_penalty_lambda=0.005 \
  > $SAVE_BASE/t2.log 2>&1 &
PID2=$!
echo "Trial 2 (tau=0.5, lam=0.005) started: PID=$PID2"

# Trial 3: tau=0.5, lambda=0.01
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29502 scripts/train.py $BASE_CONFIG \
  --max_duration=540 --save_interval_unsharded=540 --save_interval=100000 \
  --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 \
  --save_folder=$SAVE_BASE/t3-tau0.5-lam0.01 --save_overwrite=true \
  --hmat.gumbel_tau_start=0.5 --hmat.budget_penalty_lambda=0.01 \
  > $SAVE_BASE/t3.log 2>&1 &
PID3=$!
echo "Trial 3 (tau=0.5, lam=0.01) started: PID=$PID3"

# Trial 4: tau=2.0, lambda=0.001 (isolate lambda effect)
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29503 scripts/train.py $BASE_CONFIG \
  --max_duration=540 --save_interval_unsharded=540 --save_interval=100000 \
  --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 \
  --save_folder=$SAVE_BASE/t4-tau2.0-lam0.001 --save_overwrite=true \
  --hmat.gumbel_tau_start=2.0 --hmat.budget_penalty_lambda=0.001 \
  > $SAVE_BASE/t4.log 2>&1 &
PID4=$!
echo "Trial 4 (tau=2.0, lam=0.001) started: PID=$PID4"

# Trial 5: tau=1.0, lambda=0.001
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29504 scripts/train.py $BASE_CONFIG \
  --max_duration=540 --save_interval_unsharded=540 --save_interval=100000 \
  --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 \
  --save_folder=$SAVE_BASE/t5-tau1.0-lam0.001 --save_overwrite=true \
  --hmat.gumbel_tau_start=1.0 --hmat.budget_penalty_lambda=0.001 \
  > $SAVE_BASE/t5.log 2>&1 &
PID5=$!
echo "Trial 5 (tau=1.0, lam=0.001) started: PID=$PID5"

# Trial 6: tau=1.0, lambda=0.005
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29505 scripts/train.py $BASE_CONFIG \
  --max_duration=540 --save_interval_unsharded=540 --save_interval=100000 \
  --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 \
  --save_folder=$SAVE_BASE/t6-tau1.0-lam0.005 --save_overwrite=true \
  --hmat.gumbel_tau_start=1.0 --hmat.budget_penalty_lambda=0.005 \
  > $SAVE_BASE/t6.log 2>&1 &
PID6=$!
echo "Trial 6 (tau=1.0, lam=0.005) started: PID=$PID6"

echo ""
echo "All 6 trials launched. Waiting for completion..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6
echo "All trials complete!"

# Extract final eval results
echo ""
echo "================================================================"
echo "HYPERPARAMETER SEARCH RESULTS"
echo "================================================================"
echo ""
printf "%-35s %-12s %-12s %-15s\n" "Trial" "Eval PPL" "Mask Frac" "Layer Fracs"
printf "%-35s %-12s %-12s %-15s\n" "-----" "--------" "---------" "-----------"

for trial_dir in $SAVE_BASE/t*-tau*/; do
  trial_name=$(basename $trial_dir)
  log_file=$SAVE_BASE/$(echo $trial_name | cut -d'-' -f1).log

  # Get eval PPL (last eval line)
  eval_ppl=$(grep "eval/pile-val/Perplexity" $log_file 2>/dev/null | tail -1 | grep -oP 'Perplexity=\K[0-9.]+' || echo "N/A")

  # Get final mask fractions
  mean_frac=$(grep "gumbel/mean_active_frac" $log_file 2>/dev/null | tail -1 | grep -oP 'mean_active_frac=\K[0-9.]+' || echo "N/A")

  layer_fracs=""
  for l in 0 1 2 3; do
    frac=$(grep "gumbel/layer_${l}_active_frac" $log_file 2>/dev/null | tail -1 | grep -oP "layer_${l}_active_frac=\K[0-9.]+" || echo "?")
    layer_fracs="$layer_fracs $frac"
  done

  printf "%-35s %-12s %-12s %-15s\n" "$trial_name" "$eval_ppl" "$mean_frac" "$layer_fracs"
done
