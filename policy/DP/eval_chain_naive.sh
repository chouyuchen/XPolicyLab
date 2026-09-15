#!/bin/bash
# Naive long-horizon eval: PI1 (open) → PI2 (load) → PI3 (close).
# SWITCH_MODE=flag|mock_vlm|vlm  (default flag = B1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TEST_NUM="${TEST_NUM:-3}"
TASK_CONFIG="${TASK_CONFIG:-cabinet_chain_nominal_dr}"
SEED_START="${SEED_START:-100000}"
POLICY_GPU="${POLICY_GPU:-0}"
ENV_GPU="${ENV_GPU:-0}"
SWITCH_MODE="${SWITCH_MODE:-flag}"
SWITCH_EVERY_N="${SWITCH_EVERY_N:-8}"
MIN_STEPS_BEFORE_SWITCH="${MIN_STEPS_BEFORE_SWITCH:-0}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
VLM_CONDA_ENV="${VLM_CONDA_ENV:-xpl_vlm}"
# Same GPU as DP by default; set VLM_DEVICE=cpu if OOM.
VLM_DEVICE="${VLM_DEVICE:-cuda:${POLICY_GPU}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate RoboTwin

cd "${BENCH_ROOT}"
export CUDA_VISIBLE_DEVICES="${ENV_GPU}"

python scripts/eval_cabinet_chain_naive.py \
  --task-config "${TASK_CONFIG}" \
  --test-num "${TEST_NUM}" \
  --seed-start "${SEED_START}" \
  --policy-gpu "${POLICY_GPU}" \
  --switch "${SWITCH_MODE}" \
  --switch-every-n "${SWITCH_EVERY_N}" \
  --min-steps-before-switch "${MIN_STEPS_BEFORE_SWITCH}" \
  --vlm-model "${VLM_MODEL}" \
  --vlm-conda-env "${VLM_CONDA_ENV}" \
  --vlm-device "${VLM_DEVICE}"

echo "[CHAIN] eval finished (switch=${SWITCH_MODE})"
