#!/usr/bin/env bash

set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
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

PFLASH_EXTRA_BENCHMARK_ARGS=()
if [[ "${PFLASH_MERGE_PREFIX_BRANCHES}" != "0" ]]; then
  PFLASH_EXTRA_BENCHMARK_ARGS+=(--pflash-merge-prefix-branches)
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
  "${PFLASH_EXTRA_BENCHMARK_ARGS[@]}"
)

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
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
      if [[ "${MDFLASH_PROPOSAL_TEMPERATURE}" != "1.0" ]] \
        || [[ "${PEXPRESS_PERTURBATION_TEMPERATURE}" != "0.75" ]] \
        || [[ "${PEXPRESS_POSITION_TEMPERATURE_DECAY}" != "0.0" ]] \
        || [[ "${PFLASH_BRANCH_PRIOR_WEIGHT}" != "0.5" ]] \
        || [[ "${PFLASH_MERGE_PREFIX_BRANCHES}" != "0" ]] \
        || [[ "${PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT}" != "0.0" ]] \
        || [[ "${PFLASH_V4_BACKBONE_FRACTION}" != "0.75" ]] \
        || [[ "${PFLASH_V4_SUPPORT_BONUS_WEIGHT}" != "0.70" ]] \
        || [[ "${PFLASH_V4_BASE_GAP_PENALTY}" != "0.35" ]] \
        || [[ "${PFLASH_V4_GRAFT_SCORE_THRESHOLD}" != "1.0" ]]; then
        config_slug="$(slugify "md${MDFLASH_PROPOSAL_TEMPERATURE}_pe${PEXPRESS_PERTURBATION_TEMPERATURE}_pd${PEXPRESS_POSITION_TEMPERATURE_DECAY}_pp${PFLASH_BRANCH_PRIOR_WEIGHT}_pm${PFLASH_MERGE_PREFIX_BRANCHES}_ps${PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT}_v4bf${PFLASH_V4_BACKBONE_FRACTION}_v4sb${PFLASH_V4_SUPPORT_BONUS_WEIGHT}_v4bg${PFLASH_V4_BASE_GAP_PENALTY}_v4gt${PFLASH_V4_GRAFT_SCORE_THRESHOLD}")"
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
