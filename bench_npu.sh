#!/bin/bash
set -euo pipefail
cd /home/pi05/openpi_src

export PALIGEMMA_TOKENIZER_PATH=/home/pi05/models/tokenizer.model
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_IF_BASE_PORT=61000
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

RESULTS=/tmp/bench_results.txt
echo "=== NPU Parameter Sweep Benchmark ===" > "$RESULTS"
echo "Date: $(date)" >> "$RESULTS"
echo "Dataset: 00036 (50440 frames)" >> "$RESULTS"
echo "" >> "$RESULTS"
printf "%-20s %5s %5s %10s %12s %10s %10s\n" CONFIG BS GPUS STEP_S PEAK_MB LOSS ETA_D >> "$RESULTS"
echo "------------------------------------------------------------------------------------------" >> "$RESULTS"

BENCH_STEPS=50
DATASET=/home/pi05/00036_20260804_RM_01_yj_v21
WEIGHTS=/home/pi05/models

run_bench() {
    local bs=$1
    local nproc=$2
    local tag=$3
    local yaml_file=/home/pi05/openpi_src/configs/bench_${tag}.yaml
    local log_file=/tmp/bench_${tag}.log

    # Scale LR linearly with effective batch size (baseline: bs=16 → lr=2.5e-5)
    local peak_lr=$(python3 -c "print(f'{2.5e-5 * $bs * $nproc / 16:.2e}')")
    local warmup=$(python3 -c "print(max(10, int(1000 * $bs * $nproc / 16)))")

    cat > "$yaml_file" <<YAML
config_name: pi05_aloha
framework: pytorch
exp_name: bench_${tag}
project_name: openpi
checkpoint_base_dir: ./checkpoints
assets_base_dir: ./assets
batch_size: ${bs}
num_train_steps: ${BENCH_STEPS}
seed: 42
num_workers: 4
log_interval: 10
save_every_epochs: 999
overwrite: true
resume: false
wandb_enabled: false
fsdp_devices: 1
ema_decay: 0.99
pytorch_training_precision: bfloat16
pytorch_weight_path: ${WEIGHTS}
model:
  action_horizon: 50
lr_schedule:
  warmup_steps: ${warmup}
  peak_lr: ${peak_lr}
  decay_lr: 2.5e-6
optimizer:
  b1: 0.9
  b2: 0.95
  eps: 1.0e-8
  weight_decay: 1.0e-10
  clip_gradient_norm: 1.0
data:
  repo_id: ${DATASET}
  assets:
    assets_dir: ${DATASET}
    asset_id: "."
  default_prompt: "With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box."
  prompt_from_task: true
  prompt_from_subtask: true
  adapt_to_pi: false
  delta_action_dims: [7, 7, -1, -1]
YAML

    echo ">>> [$tag] bs=$bs gpus=$nproc lr=$peak_lr warmup=$warmup"

    # Set visible devices
    local devs=""
    for ((i=0; i<nproc; i++)); do
        [ -n "$devs" ] && devs="$devs,"
        devs="${devs}$i"
    done
    export ASCEND_RT_VISIBLE_DEVICES=$devs

    if [ "$nproc" -eq 1 ]; then
        python scripts/train_from_yaml.py --config "$yaml_file" > "$log_file" 2>&1 || true
    else
        torchrun --standalone --nnodes=1 --nproc_per_node=$nproc \
            scripts/train_from_yaml.py --config "$yaml_file" > "$log_file" 2>&1 || true
    fi

    # Parse results from log
    local last_step_line=$(grep -E '^INFO:root:step=' "$log_file" 2>/dev/null | tail -1 || echo "")
    local loss=$(echo "$last_step_line" | grep -oP 'loss=\K[0-9.]+' || echo "FAIL")

    # Get peak memory
    local peak_reserved=$(grep 'peak_reserved' "$log_file" 2>/dev/null | tail -1 | grep -oP 'peak_reserved: \K[0-9.]+' || echo "0")
    local peak_alloc=$(grep 'peak_allocated' "$log_file" 2>/dev/null | tail -1 | grep -oP 'peak_allocated: \K[0-9.]+' || echo "0")
    local peak_mb="$peak_reserved"
    [ "$peak_mb" = "0" ] && peak_mb="$peak_alloc"

    # Average step time from last 5 tqdm entries (skip first 10 warmup steps)
    local avg_step=$(grep -oP '[0-9]+\.[0-9]+s/it' "$log_file" 2>/dev/null | tail -10 | sed 's/s\/it//' | awk '{s+=$1;n++} END{if(n>0) printf "%.2f",s/n; else print "FAIL"}')
    [ -z "$avg_step" ] && avg_step="FAIL"

    # ETA for 100 epochs (steps_per_epoch = 50440/(bs*nproc), total = spe*100)
    local eta="N/A"
    if [ "$avg_step" != "FAIL" ]; then
        eta=$(python3 -c "spe=50440//($bs*$nproc); total=spe*100; print(f'{total*float(\"$avg_step\")/86400:.1f}')" 2>/dev/null || echo "N/A")
    fi

    printf "%-20s %5d %5d %10s %12s %10s %10s\n" "$tag" "$bs" "$nproc" "${avg_step}" "${peak_mb}" "$loss" "$eta" >> "$RESULTS"

    echo "    => step=${avg_step}s peak_hbm=${peak_mb}MB loss=$loss eta=${eta}d"
    echo ""

    # Clean up NPU between runs
    sleep 5
}

echo ""
echo "=== Phase 1: Single-card batch size sweep ==="
echo ""
run_bench 32  1  "bs32_1g"
run_bench 64  1  "bs64_1g"
run_bench 128 1  "bs128_1g"

echo "=== Phase 2: Multi-card (bs=64 per card) ==="
echo ""
run_bench 64 2  "bs64_2g"
run_bench 64 4  "bs64_4g"
run_bench 64 8  "bs64_8g"

echo "=== Phase 3: Large batch multi-card ==="
echo ""
run_bench 128 4   "bs128_4g"
run_bench 128 8   "bs128_8g"
run_bench 128 16  "bs128_16g"

echo ""
echo "==============================="
echo "  BENCHMARK RESULTS SUMMARY"
echo "==============================="
cat "$RESULTS"
echo ""
echo "Benchmark complete at $(date)"
