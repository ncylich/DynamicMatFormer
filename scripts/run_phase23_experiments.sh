#!/bin/bash
# Phase 2.3: Experimental Validation of Full-Width Fix
#
# 15 short screening runs (540 steps) across 3 groups:
#   Group 1 (V1-V3): Method comparison — baseline vs topk vs gumbel_topk
#   Group 2 (G1-G4, T1-T4): Init ablation — zeros, normal, constant, linspace
#   Group 3 (A1-A4): Tau ablation — 0.2, 0.5, 1.0
#
# Batch 1: 8 GPUs (V1, V2, V3, G1, G2, G3, G4, T1)
# Batch 2: 7 GPUs (T2, T3, T4, A1, A2, A3, A4)
# Batch 3: 2-3 GPUs — long runs (2700 steps) of best configs (manual)

set -e
cd /mnt/data/noahcylich/MatFormer-OLMo
source /mnt/data/noahcylich/venv/bin/activate

BASELINE_CONFIG=configs/pile-tiny-matformer.yaml
TOPK_CONFIG=configs/pile-tiny-hmat-topk.yaml
SAVE_BASE=/mnt/data/noahcylich/checkpoints/phase23

mkdir -p $SAVE_BASE

# Common overrides for 540-step screening runs
COMMON="--max_duration=540 --save_interval_unsharded=540 --save_interval=100000 --eval_interval=540 --eval_subset_num_batches=50 --console_log_interval=50 --save_overwrite=true"

# ---- Helper functions ----

extract_results() {
  local log=$1
  local name=$2

  if [ ! -f "$log" ]; then
    printf "%-32s  MISSING LOG\n" "$name"
    return
  fi

  if grep -q "Traceback\|CUDA out of memory" "$log" 2>/dev/null; then
    printf "%-32s  ERROR (check log)\n" "$name"
    return
  fi

  # Last 4 eval/pile-val/Perplexity lines = 1/1, 1/2, 1/4, 1/8
  local ppls
  ppls=$(grep "eval/pile-val/Perplexity" "$log" | tail -4 | grep -oP 'Perplexity=\K[0-9,.]+' | tr -d ',')
  local ppl1 ppl2 ppl4 ppl8
  ppl1=$(echo "$ppls" | sed -n '1p')
  ppl2=$(echo "$ppls" | sed -n '2p')
  ppl4=$(echo "$ppls" | sed -n '3p')
  ppl8=$(echo "$ppls" | sed -n '4p')

  # Layer active fractions (last occurrence of each)
  local fracs=""
  for l in 0 1 2 3; do
    local frac
    frac=$(grep "layer_${l}_active_frac" "$log" 2>/dev/null | tail -1 | grep -oP "active_frac[^=]*=\K[0-9.]+" || echo "-")
    fracs="$fracs $frac"
  done

  printf "%-32s  %-8s %-8s %-8s %-8s  %s\n" \
    "$name" "${ppl1:-N/A}" "${ppl2:-N/A}" "${ppl4:-N/A}" "${ppl8:-N/A}" "$fracs"
}

print_header() {
  echo ""
  printf "%-32s  %-8s %-8s %-8s %-8s  %s\n" "Run" "PPL 1/1" "PPL 1/2" "PPL 1/4" "PPL 1/8" "Layer fracs (0 1 2 3)"
  printf "%-32s  %-8s %-8s %-8s %-8s  %s\n" "---" "-------" "-------" "-------" "-------" "---------------------"
}

######################################################################
# BATCH 1: 8 runs on GPUs 0-7 (~15 min)
######################################################################
echo "================================================================"
echo "Phase 2.3 — Batch 1: 8 runs (V1-V3, G1-G4, T1)"
echo "================================================================"

# V1: Baseline uniform MatFormer (no learnable masks)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  scripts/train.py $BASELINE_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/V1-baseline \
  --hmat.enabled=false \
  > $SAVE_BASE/V1.log 2>&1 &
PID_V1=$!
echo "V1 (baseline, no hmat) started: PID=$PID_V1"

# V2: topk, linspace(1.1) — default topk config
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/V2-topk-linspace \
  > $SAVE_BASE/V2.log 2>&1 &
PID_V2=$!
echo "V2 (topk, linspace 1.1) started: PID=$PID_V2"

# V3: gumbel_topk, linspace(1.1)
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29502 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/V3-gumbel_topk-linspace \
  --hmat.method=gumbel_topk \
  > $SAVE_BASE/V3.log 2>&1 &
PID_V3=$!
echo "V3 (gumbel_topk, linspace 1.1) started: PID=$PID_V3"

# G1: gumbel_topk, zeros init
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29503 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/G1-gumbel_topk-zeros \
  --hmat.method=gumbel_topk --hmat.gumbel_init_mode=zeros \
  > $SAVE_BASE/G1.log 2>&1 &
PID_G1=$!
echo "G1 (gumbel_topk, zeros) started: PID=$PID_G1"

# G2: gumbel_topk, normal(0, 0.3)
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29504 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/G2-gumbel_topk-normal \
  --hmat.method=gumbel_topk --hmat.gumbel_init_mode=normal --hmat.gumbel_init_value=0.3 \
  > $SAVE_BASE/G2.log 2>&1 &
PID_G2=$!
echo "G2 (gumbel_topk, normal 0.3) started: PID=$PID_G2"

# G3: gumbel_topk, constant(+1.5)
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29505 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/G3-gumbel_topk-constant \
  --hmat.method=gumbel_topk --hmat.gumbel_init_mode=constant --hmat.gumbel_init_value=1.5 \
  > $SAVE_BASE/G3.log 2>&1 &
PID_G3=$!
echo "G3 (gumbel_topk, constant 1.5) started: PID=$PID_G3"

# G4: gumbel_topk, linspace(1.1) — control duplicate of V3
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29506 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/G4-gumbel_topk-linspace-dup \
  --hmat.method=gumbel_topk \
  > $SAVE_BASE/G4.log 2>&1 &
PID_G4=$!
echo "G4 (gumbel_topk, linspace dup) started: PID=$PID_G4"

# T1: topk, zeros init
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29507 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/T1-topk-zeros \
  --hmat.gumbel_init_mode=zeros \
  > $SAVE_BASE/T1.log 2>&1 &
PID_T1=$!
echo "T1 (topk, zeros) started: PID=$PID_T1"

echo ""
echo "Waiting for Batch 1 (8 runs)..."
wait $PID_V1 $PID_V2 $PID_V3 $PID_G1 $PID_G2 $PID_G3 $PID_G4 $PID_T1
echo "Batch 1 complete!"

echo ""
echo "================================================================"
echo "BATCH 1 RESULTS"
echo "================================================================"
print_header
extract_results "$SAVE_BASE/V1.log" "V1: baseline"
extract_results "$SAVE_BASE/V2.log" "V2: topk linspace(1.1)"
extract_results "$SAVE_BASE/V3.log" "V3: gumbel_topk linspace(1.1)"
extract_results "$SAVE_BASE/G1.log" "G1: gumbel_topk zeros"
extract_results "$SAVE_BASE/G2.log" "G2: gumbel_topk normal(0.3)"
extract_results "$SAVE_BASE/G3.log" "G3: gumbel_topk constant(1.5)"
extract_results "$SAVE_BASE/G4.log" "G4: gumbel_topk linspace(dup)"
extract_results "$SAVE_BASE/T1.log" "T1: topk zeros"

echo ""
echo "--- Batch 1 Analysis Checklist ---"
echo "1. Sanity: Did all 8 runs complete without errors?"
echo "2. Reproducibility: V3 vs G4 should show similar PPLs (same config, different seed noise)"
echo "3. Method comparison: Compare V1 (baseline) vs V2 (topk) vs V3 (gumbel_topk) at 1/1 and 1/8"
echo "   - Key question: Does topk or gumbel_topk beat baseline at sub-model widths (1/2, 1/4, 1/8)?"
echo "   - Expected: ~1-3% improvement at 1/8, possibly slight regression at 1/1"
echo "4. Init sensitivity (gumbel_topk): Compare G1 (zeros) vs G2 (normal) vs G3 (constant) vs V3 (linspace)"
echo "   - If zeros (G1) crashes or diverges, confirms init matters"
echo "   - If constant(1.5) (G3) matches linspace, init doesn't need to be ordered"
echo "5. Cross-method init: T1 (topk zeros) vs V2 (topk linspace) — is topk more robust to init?"
echo ""

######################################################################
# BATCH 2: 7 runs on GPUs 0-6 (~15 min)
######################################################################
echo ""
echo "================================================================"
echo "Phase 2.3 — Batch 2: 7 runs (T2-T4, A1-A4)"
echo "================================================================"

# T2: topk, normal(0, 0.3)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/T2-topk-normal \
  --hmat.gumbel_init_mode=normal --hmat.gumbel_init_value=0.3 \
  > $SAVE_BASE/T2.log 2>&1 &
PID_T2=$!
echo "T2 (topk, normal 0.3) started: PID=$PID_T2"

# T3: topk, constant(+1.5)
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/T3-topk-constant \
  --hmat.gumbel_init_mode=constant --hmat.gumbel_init_value=1.5 \
  > $SAVE_BASE/T3.log 2>&1 &
PID_T3=$!
echo "T3 (topk, constant 1.5) started: PID=$PID_T3"

# T4: topk, linspace(1.1) — control duplicate of V2
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29502 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/T4-topk-linspace-dup \
  > $SAVE_BASE/T4.log 2>&1 &
PID_T4=$!
echo "T4 (topk, linspace dup) started: PID=$PID_T4"

# A1: topk, linspace(1.1), tau=0.5 (same as V2 default — tau control)
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29503 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/A1-topk-tau0.5 \
  --hmat.gumbel_tau_start=0.5 \
  > $SAVE_BASE/A1.log 2>&1 &
PID_A1=$!
echo "A1 (topk, tau=0.5 control) started: PID=$PID_A1"

# A2: gumbel_topk, linspace(1.1), tau=0.5 (same as V3 default — tau control)
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29504 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/A2-gumbel_topk-tau0.5 \
  --hmat.method=gumbel_topk --hmat.gumbel_tau_start=0.5 \
  > $SAVE_BASE/A2.log 2>&1 &
PID_A2=$!
echo "A2 (gumbel_topk, tau=0.5 control) started: PID=$PID_A2"

# A3: gumbel_topk, linspace(1.1), tau=1.0
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29505 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/A3-gumbel_topk-tau1.0 \
  --hmat.method=gumbel_topk --hmat.gumbel_tau_start=1.0 \
  > $SAVE_BASE/A3.log 2>&1 &
PID_A3=$!
echo "A3 (gumbel_topk, tau=1.0) started: PID=$PID_A3"

# A4: gumbel_topk, linspace(1.1), tau=0.2
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29506 \
  scripts/train.py $TOPK_CONFIG $COMMON \
  --save_folder=$SAVE_BASE/A4-gumbel_topk-tau0.2 \
  --hmat.method=gumbel_topk --hmat.gumbel_tau_start=0.2 \
  > $SAVE_BASE/A4.log 2>&1 &
PID_A4=$!
echo "A4 (gumbel_topk, tau=0.2) started: PID=$PID_A4"

echo ""
echo "Waiting for Batch 2 (7 runs)..."
wait $PID_T2 $PID_T3 $PID_T4 $PID_A1 $PID_A2 $PID_A3 $PID_A4
echo "Batch 2 complete!"

echo ""
echo "================================================================"
echo "BATCH 2 RESULTS"
echo "================================================================"
print_header
extract_results "$SAVE_BASE/T2.log" "T2: topk normal(0.3)"
extract_results "$SAVE_BASE/T3.log" "T3: topk constant(1.5)"
extract_results "$SAVE_BASE/T4.log" "T4: topk linspace(dup)"
extract_results "$SAVE_BASE/A1.log" "A1: topk tau=0.5"
extract_results "$SAVE_BASE/A2.log" "A2: gumbel_topk tau=0.5"
extract_results "$SAVE_BASE/A3.log" "A3: gumbel_topk tau=1.0"
extract_results "$SAVE_BASE/A4.log" "A4: gumbel_topk tau=0.2"

######################################################################
# COMBINED RESULTS
######################################################################
echo ""
echo "================================================================"
echo "ALL SCREENING RESULTS (540 steps)"
echo "================================================================"
print_header
echo "--- Group 1: Method comparison ---"
extract_results "$SAVE_BASE/V1.log" "V1: baseline"
extract_results "$SAVE_BASE/V2.log" "V2: topk linspace(1.1)"
extract_results "$SAVE_BASE/V3.log" "V3: gumbel_topk linspace(1.1)"
echo "--- Group 2a: Init ablation (gumbel_topk) ---"
extract_results "$SAVE_BASE/G1.log" "G1: gumbel_topk zeros"
extract_results "$SAVE_BASE/G2.log" "G2: gumbel_topk normal(0.3)"
extract_results "$SAVE_BASE/G3.log" "G3: gumbel_topk constant(1.5)"
extract_results "$SAVE_BASE/G4.log" "G4: gumbel_topk linspace(dup)"
echo "--- Group 2b: Init ablation (topk) ---"
extract_results "$SAVE_BASE/T1.log" "T1: topk zeros"
extract_results "$SAVE_BASE/T2.log" "T2: topk normal(0.3)"
extract_results "$SAVE_BASE/T3.log" "T3: topk constant(1.5)"
extract_results "$SAVE_BASE/T4.log" "T4: topk linspace(dup)"
echo "--- Group 3: Tau ablation ---"
extract_results "$SAVE_BASE/A1.log" "A1: topk tau=0.5"
extract_results "$SAVE_BASE/A2.log" "A2: gumbel_topk tau=0.5"
extract_results "$SAVE_BASE/A3.log" "A3: gumbel_topk tau=1.0"
extract_results "$SAVE_BASE/A4.log" "A4: gumbel_topk tau=0.2"

echo ""
echo "================================================================"
echo "ANALYSIS GUIDE — Selecting Batch 3 Configs"
echo "================================================================"
echo ""
echo "Answer each question conservatively before selecting long-run configs:"
echo ""
echo "Q1. METHOD: Which method family is best?"
echo "    Compare V1 vs best-topk vs best-gumbel_topk."
echo "    - If topk and gumbel_topk are within noise of each other, prefer topk (simpler)."
echo "    - If one clearly dominates at sub-model widths, use that method."
echo ""
echo "Q2. INIT: Which initialization is best for the winning method?"
echo "    Compare across G1-G4 (gumbel_topk) and T1-T4 (topk)."
echo "    - Rank by geometric mean of PPL across all 4 widths."
echo "    - If linspace wins by >1%, it's a real effect. If <1%, init may not matter."
echo ""
echo "Q3. TAU: Does temperature matter?"
echo "    Compare A2 (tau=0.5) vs A3 (tau=1.0) vs A4 (tau=0.2)."
echo "    - If 0.2 or 1.0 beats 0.5 by >2%, consider using it for Batch 3."
echo "    - If all within noise, tau=0.5 is fine (fewer hyperparams to justify)."
echo ""
echo "Q4. REPRODUCIBILITY: Are control duplicates consistent?"
echo "    V3 vs G4, V2 vs T4, V2 vs A1, V3 vs A2."
echo "    - PPL difference >3% suggests high variance; need more seeds in Batch 3."
echo "    - PPL difference <1% suggests good reproducibility; 1 seed per config OK."
echo ""
echo "BATCH 3 SELECTION:"
echo "  L1 = baseline (always run for reference)"
echo "  L2 = best overall config from screening"
echo "  L3 = best config from the OTHER method family (topk vs gumbel_topk)"
echo "        OR best alternative init/tau if one method dominates"
echo ""
echo "Review results above, then edit Batch 3 section and rerun."

######################################################################
# BATCH 3: Long validation runs (2700 steps, ~75 min)
# Uncomment and fill in best configs after reviewing screening results
######################################################################
# LONG_COMMON="--max_duration=2700 --save_interval_unsharded=2700 --save_interval=100000 --eval_interval=2700 --eval_subset_num_batches=50 --console_log_interval=100 --save_overwrite=true"
#
# # L1: Baseline long run
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
#   scripts/train.py $BASELINE_CONFIG $LONG_COMMON \
#   --save_folder=$SAVE_BASE/L1-baseline-long \
#   --hmat.enabled=false \
#   > $SAVE_BASE/L1.log 2>&1 &
# PID_L1=$!
#
# # L2: Best config from screening (fill in method + init overrides)
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 \
#   scripts/train.py $TOPK_CONFIG $LONG_COMMON \
#   --save_folder=$SAVE_BASE/L2-best-long \
#   > $SAVE_BASE/L2.log 2>&1 &
# PID_L2=$!
#
# # L3: 2nd best config from screening (fill in method + init overrides)
# CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29502 \
#   scripts/train.py $TOPK_CONFIG $LONG_COMMON \
#   --save_folder=$SAVE_BASE/L3-second-long \
#   > $SAVE_BASE/L3.log 2>&1 &
# PID_L3=$!
#
# wait $PID_L1 $PID_L2 $PID_L3
# echo ""
# echo "================================================================"
# echo "BATCH 3 RESULTS (2700 steps)"
# echo "================================================================"
# print_header
# extract_results "$SAVE_BASE/L1.log" "L1: baseline long"
# extract_results "$SAVE_BASE/L2.log" "L2: best long"
# extract_results "$SAVE_BASE/L3.log" "L3: 2nd best long"
