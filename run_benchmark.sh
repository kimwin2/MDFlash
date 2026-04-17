#!/usr/bin/env bash

set -u

count_cuda_visible_devices() {
  local value="$1"
  if [[ -z "${value}" || "${value}" == "-1" || "${value}" == "NoDevFiles" ]]; then
    echo 0
    return
  fi

  local count=0
  local entries=()
  IFS=',' read -r -a entries <<< "${value}"
  for entry in "${entries[@]}"; do
    if [[ -n "${entry}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

detect_physical_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  echo 0
}

build_cuda_visible_devices() {
  local count="$1"
  local devices=()
  for ((device_idx = 0; device_idx < count; device_idx++)); do
    devices+=("${device_idx}")
  done
  local IFS=,
  echo "${devices[*]}"
}

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  VISIBLE_GPU_COUNT="$(detect_physical_gpu_count)"
  if [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] && (( VISIBLE_GPU_COUNT > 0 )); then
    export CUDA_VISIBLE_DEVICES="$(build_cuda_visible_devices "${VISIBLE_GPU_COUNT}")"
  fi
else
  VISIBLE_GPU_COUNT="$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${VISIBLE_GPU_COUNT:-1}}"
if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" == "0" ]]; then
  NPROC_PER_NODE=1
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 1
fi
if [[ "${VISIBLE_GPU_COUNT:-0}" =~ ^[0-9]+$ ]] && (( VISIBLE_GPU_COUNT > 0 && NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} is larger than visible GPU count ${VISIBLE_GPU_COUNT} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})." >&2
  echo "Set NPROC_PER_NODE=${VISIBLE_GPU_COUNT} or reduce CUDA_VISIBLE_DEVICES to avoid multiple benchmark workers sharing a GPU." >&2
  exit 1
fi

MASTER_PORT="${MASTER_PORT:-29600}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DIR="${RUN_DIR:-runs}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

TASKS=(
  "gsm8k:128"
  "math500:128"
  "aime24:30"
  "aime25:30"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "swe-bench:128"
  "mt-bench:80"
  "alpaca:128"
)

MODEL_DRAFT_PAIRS=(
  "Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16"
  "Qwen/Qwen3-8B|z-lab/Qwen3-8B-DFlash-b16"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct|z-lab/Qwen3-Coder-30B-A3B-DFlash"
)

TEMPERATURES=(
  "0.0"
  "1.0"
)

MDFLASH_PROPOSAL_TEMPERATURE="${MDFLASH_PROPOSAL_TEMPERATURE:-1.0}"
PEXPRESS_PERTURBATION_TEMPERATURE="${PEXPRESS_PERTURBATION_TEMPERATURE:-0.75}"
PEXPRESS_POSITION_TEMPERATURE_DECAY="${PEXPRESS_POSITION_TEMPERATURE_DECAY:-0.0}"
PFLASH_BRANCH_PRIOR_WEIGHT="${PFLASH_BRANCH_PRIOR_WEIGHT:-0.5}"
PFLASH_MERGE_PREFIX_BRANCHES="${PFLASH_MERGE_PREFIX_BRANCHES:-0}"
PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT="${PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT:-0.0}"
PFLASH_V4_BACKBONE_FRACTION="${PFLASH_V4_BACKBONE_FRACTION:-0.75}"
PFLASH_V4_SUPPORT_BONUS_WEIGHT="${PFLASH_V4_SUPPORT_BONUS_WEIGHT:-0.70}"
PFLASH_V4_BASE_GAP_PENALTY="${PFLASH_V4_BASE_GAP_PENALTY:-0.35}"
PFLASH_V4_GRAFT_SCORE_THRESHOLD="${PFLASH_V4_GRAFT_SCORE_THRESHOLD:-1.0}"
PFLASH_V5_HIGH_AGREEMENT_THRESHOLD="${PFLASH_V5_HIGH_AGREEMENT_THRESHOLD:-0.95}"
PFLASH_V5_MID_AGREEMENT_THRESHOLD="${PFLASH_V5_MID_AGREEMENT_THRESHOLD:-0.90}"
PFLASH_V5_LOW_AGREEMENT_DEPTH="${PFLASH_V5_LOW_AGREEMENT_DEPTH:-5}"
PFLASH_V6_HIGH_ALIGNMENT_THRESHOLD="${PFLASH_V6_HIGH_ALIGNMENT_THRESHOLD:-0.95}"
PFLASH_V6_MID_ALIGNMENT_THRESHOLD="${PFLASH_V6_MID_ALIGNMENT_THRESHOLD:-0.90}"
PFLASH_V6_HIGH_BLOCK_SIZE="${PFLASH_V6_HIGH_BLOCK_SIZE:-16}"
PFLASH_V6_MID_BLOCK_SIZE="${PFLASH_V6_MID_BLOCK_SIZE:-8}"
PFLASH_V6_LOW_BLOCK_SIZE="${PFLASH_V6_LOW_BLOCK_SIZE:-8}"
PFLASH_V6_HIGH_TREE_BUDGET="${PFLASH_V6_HIGH_TREE_BUDGET:-128}"
PFLASH_V6_MID_TREE_BUDGET="${PFLASH_V6_MID_TREE_BUDGET:-64}"
PFLASH_V6_LOW_TREE_BUDGET="${PFLASH_V6_LOW_TREE_BUDGET:-32}"
PFLASH_V7_BUDGET="${PFLASH_V7_BUDGET:-}"
PFLASH_V8_BUDGET="${PFLASH_V8_BUDGET:-}"
PFLASH_V9_BUDGET="${PFLASH_V9_BUDGET:-}"
PFLASH_V10_BUDGET="${PFLASH_V10_BUDGET:-}"
PFLASH_V11_BUDGET="${PFLASH_V11_BUDGET:-}"
EXP_DDTREE_BUDGET="${EXP_DDTREE_BUDGET:-}"
EXP_PREDICTMV="${EXP_PREDICTMV:-0}"
MEASURE_BATCH_AGREEMENT="${MEASURE_BATCH_AGREEMENT:-0}"

PFLASH_EXTRA_BENCHMARK_ARGS=()
if [[ "${PFLASH_MERGE_PREFIX_BRANCHES}" != "0" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--pflash-merge-prefix-branches)
fi
if [[ "${MEASURE_BATCH_AGREEMENT}" != "0" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--measure-batch-agreement)
fi
if [[ -n "${EXP_DDTREE_BUDGET}" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--exp-ddtree-budget "${EXP_DDTREE_BUDGET}")
fi
if [[ "${EXP_PREDICTMV}" != "0" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--exp-predictmv)
fi
if [[ -n "${PFLASH_V7_BUDGET}" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--pflash-v7-budget "${PFLASH_V7_BUDGET}")
fi
if [[ -n "${PFLASH_V8_BUDGET}" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--pflash-v8-budget "${PFLASH_V8_BUDGET}")
fi
if [[ -n "${PFLASH_V9_BUDGET}" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--pflash-v9-budget "${PFLASH_V9_BUDGET}")
fi
if [[ -n "${PFLASH_V10_BUDGET}" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--pflash-v10-budget "${PFLASH_V10_BUDGET}")
fi
if [[ -n "${PFLASH_V11_BUDGET}" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--pflash-v11-budget "${PFLASH_V11_BUDGET}")
fi

COMMON_BENCHMARK_ARGS=(
  --max-new-tokens 2048
  --mdflash-proposal-temperature "${MDFLASH_PROPOSAL_TEMPERATURE}"
  --pexpress-perturbation-temperature "${PEXPRESS_PERTURBATION_TEMPERATURE}"
  --pexpress-position-temperature-decay "${PEXPRESS_POSITION_TEMPERATURE_DECAY}"
  --pflash-branch-prior-weight "${PFLASH_BRANCH_PRIOR_WEIGHT}"
  --pflash-prefix-support-bonus-weight "${PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT}"
  --pflash-v4-backbone-fraction "${PFLASH_V4_BACKBONE_FRACTION}"
  --pflash-v4-support-bonus-weight "${PFLASH_V4_SUPPORT_BONUS_WEIGHT}"
  --pflash-v4-base-gap-penalty "${PFLASH_V4_BASE_GAP_PENALTY}"
  --pflash-v4-graft-score-threshold "${PFLASH_V4_GRAFT_SCORE_THRESHOLD}"
  --pflash-v5-high-agreement-threshold "${PFLASH_V5_HIGH_AGREEMENT_THRESHOLD}"
  --pflash-v5-mid-agreement-threshold "${PFLASH_V5_MID_AGREEMENT_THRESHOLD}"
  --pflash-v5-low-agreement-depth "${PFLASH_V5_LOW_AGREEMENT_DEPTH}"
  --pflash-v6-high-alignment-threshold "${PFLASH_V6_HIGH_ALIGNMENT_THRESHOLD}"
  --pflash-v6-mid-alignment-threshold "${PFLASH_V6_MID_ALIGNMENT_THRESHOLD}"
  --pflash-v6-high-block-size "${PFLASH_V6_HIGH_BLOCK_SIZE}"
  --pflash-v6-mid-block-size "${PFLASH_V6_MID_BLOCK_SIZE}"
  --pflash-v6-low-block-size "${PFLASH_V6_LOW_BLOCK_SIZE}"
  --pflash-v6-high-tree-budget "${PFLASH_V6_HIGH_TREE_BUDGET}"
  --pflash-v6-mid-tree-budget "${PFLASH_V6_MID_TREE_BUDGET}"
  --pflash-v6-low-tree-budget "${PFLASH_V6_LOW_TREE_BUDGET}"
  "${PFLASH_EXTRA_BENCHMARK_ARGS[@]}"
)

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

build_config_slug() {
  local parts=()
  [[ "${MDFLASH_PROPOSAL_TEMPERATURE}" != "1.0" ]] && parts+=("md${MDFLASH_PROPOSAL_TEMPERATURE}")
  [[ "${PEXPRESS_PERTURBATION_TEMPERATURE}" != "0.75" ]] && parts+=("pe${PEXPRESS_PERTURBATION_TEMPERATURE}")
  [[ "${PEXPRESS_POSITION_TEMPERATURE_DECAY}" != "0.0" ]] && parts+=("pd${PEXPRESS_POSITION_TEMPERATURE_DECAY}")
  [[ "${PFLASH_BRANCH_PRIOR_WEIGHT}" != "0.5" ]] && parts+=("pp${PFLASH_BRANCH_PRIOR_WEIGHT}")
  [[ "${PFLASH_MERGE_PREFIX_BRANCHES}" != "0" ]] && parts+=("pm${PFLASH_MERGE_PREFIX_BRANCHES}")
  [[ "${PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT}" != "0.0" ]] && parts+=("ps${PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT}")
  [[ "${PFLASH_V4_BACKBONE_FRACTION}" != "0.75" ]] && parts+=("v4bf${PFLASH_V4_BACKBONE_FRACTION}")
  [[ "${PFLASH_V4_SUPPORT_BONUS_WEIGHT}" != "0.70" ]] && parts+=("v4sb${PFLASH_V4_SUPPORT_BONUS_WEIGHT}")
  [[ "${PFLASH_V4_BASE_GAP_PENALTY}" != "0.35" ]] && parts+=("v4bg${PFLASH_V4_BASE_GAP_PENALTY}")
  [[ "${PFLASH_V4_GRAFT_SCORE_THRESHOLD}" != "1.0" ]] && parts+=("v4gt${PFLASH_V4_GRAFT_SCORE_THRESHOLD}")
  [[ "${PFLASH_V5_HIGH_AGREEMENT_THRESHOLD}" != "0.95" ]] && parts+=("v5hi${PFLASH_V5_HIGH_AGREEMENT_THRESHOLD}")
  [[ "${PFLASH_V5_MID_AGREEMENT_THRESHOLD}" != "0.90" ]] && parts+=("v5mid${PFLASH_V5_MID_AGREEMENT_THRESHOLD}")
  [[ "${PFLASH_V5_LOW_AGREEMENT_DEPTH}" != "5" ]] && parts+=("v5low${PFLASH_V5_LOW_AGREEMENT_DEPTH}")
  [[ "${PFLASH_V6_HIGH_ALIGNMENT_THRESHOLD}" != "0.95" ]] && parts+=("v6hi${PFLASH_V6_HIGH_ALIGNMENT_THRESHOLD}")
  [[ "${PFLASH_V6_MID_ALIGNMENT_THRESHOLD}" != "0.90" ]] && parts+=("v6mid${PFLASH_V6_MID_ALIGNMENT_THRESHOLD}")
  [[ "${PFLASH_V6_HIGH_BLOCK_SIZE}" != "16" ]] && parts+=("v6hb${PFLASH_V6_HIGH_BLOCK_SIZE}")
  [[ "${PFLASH_V6_MID_BLOCK_SIZE}" != "8" ]] && parts+=("v6mb${PFLASH_V6_MID_BLOCK_SIZE}")
  [[ "${PFLASH_V6_LOW_BLOCK_SIZE}" != "8" ]] && parts+=("v6lb${PFLASH_V6_LOW_BLOCK_SIZE}")
  [[ "${PFLASH_V6_HIGH_TREE_BUDGET}" != "128" ]] && parts+=("v6ht${PFLASH_V6_HIGH_TREE_BUDGET}")
  [[ "${PFLASH_V6_MID_TREE_BUDGET}" != "64" ]] && parts+=("v6mt${PFLASH_V6_MID_TREE_BUDGET}")
  [[ "${PFLASH_V6_LOW_TREE_BUDGET}" != "32" ]] && parts+=("v6lt${PFLASH_V6_LOW_TREE_BUDGET}")
  [[ -n "${PFLASH_V7_BUDGET}" ]] && parts+=("v7${PFLASH_V7_BUDGET}")
  [[ -n "${PFLASH_V8_BUDGET}" ]] && parts+=("v8${PFLASH_V8_BUDGET}")
  [[ -n "${PFLASH_V9_BUDGET}" ]] && parts+=("v9${PFLASH_V9_BUDGET}")
  [[ -n "${PFLASH_V10_BUDGET}" ]] && parts+=("v10${PFLASH_V10_BUDGET}")
  [[ -n "${PFLASH_V11_BUDGET}" ]] && parts+=("v11${PFLASH_V11_BUDGET}")
  [[ -n "${EXP_DDTREE_BUDGET}" ]] && parts+=("edt${EXP_DDTREE_BUDGET}")
  [[ "${EXP_PREDICTMV}" != "0" ]] && parts+=("epmv${EXP_PREDICTMV}")
  [[ "${MEASURE_BATCH_AGREEMENT}" != "0" ]] && parts+=("ba${MEASURE_BATCH_AGREEMENT}")

  local IFS="_"
  slugify "${parts[*]}"
}

run_benchmark() {
  local dataset_name="$1"
  local max_samples="$2"
  local model_name="$3"
  local draft_name="$4"
  local mode_name="$5"
  local save_path="$6"
  local log_path="$7"
  shift 7

  echo "========================================================"
  echo "Running Benchmark: dataset=${dataset_name} max_samples=${max_samples} model=${model_name} draft=${draft_name} mode=${mode_name}"
  echo "GPU config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "========================================================"

  if [[ -f "${save_path}" ]]; then
    echo "Skipping existing run: ${save_path}"
    return
  fi

  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    benchmark.py \
    --dataset "${dataset_name}" \
    --max-samples "${max_samples}" \
    --model-name-or-path "${model_name}" \
    --draft-name-or-path "${draft_name}" \
    --save-path "${save_path}" \
    "${COMMON_BENCHMARK_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${log_path}"
}

for task in "${TASKS[@]}"; do
  IFS=':' read -r dataset_name max_samples <<< "${task}"

  for pair in "${MODEL_DRAFT_PAIRS[@]}"; do
    IFS='|' read -r model_name draft_name <<< "${pair}"

    model_slug="$(slugify "${model_name}")"
    draft_slug="$(slugify "${draft_name}")"
    for temperature in "${TEMPERATURES[@]}"; do
      temperature_slug="$(slugify "${temperature}")"
      config_suffix=""
      config_slug="$(build_config_slug)"
      if [[ -n "${config_slug}" ]]; then
        config_suffix="__cfg${config_slug}"
      fi
      run_name="${dataset_name}__${model_slug}__${draft_slug}__temp${temperature_slug}${config_suffix}"

      run_benchmark \
        "${dataset_name}" \
        "${max_samples}" \
        "${model_name}" \
        "${draft_name}" \
        "sdpa" \
        "${RUN_DIR}/${run_name}__sdpa.pt" \
        "${LOG_DIR}/${run_name}__sdpa.log" \
        --temperature "${temperature}"

      run_benchmark \
        "${dataset_name}" \
        "${max_samples}" \
        "${model_name}" \
        "${draft_name}" \
        "flash_attn" \
        "${RUN_DIR}/${run_name}__flash_attn.pt" \
        "${LOG_DIR}/${run_name}__flash_attn.log" \
        --temperature "${temperature}" \
        --flash-attn
    done
  done
done
