#!/bin/bash
set -euo pipefail
# Standalone GT-action replay for π3 close drawer (no DP / no policy server).
# Usage:
#   bash eval_gt_action_replay.sh [env_gpu] [max_steps]
# Examples:
#   bash eval_gt_action_replay.sh 0 0          # full episodes on GPU 0
#   bash eval_gt_action_replay.sh 0 50         # first 50 GT steps only
#   SAVE_VIDEOS=0 bash eval_gt_action_replay.sh 0 0

ENV_GPU_ID=${1:-0}
MAX_STEPS=${2:-0}
EVAL_CONDA_ENV=${3:-RoboTwin}
DEMO_INDICES=${DEMO_INDICES:-27,32,135,146,196}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${EVAL_CONDA_ENV}"
set -u

SAVE_VIDEOS=${SAVE_VIDEOS:-1}
VIDEO_ARGS=()
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
    VIDEO_ARGS+=(--save-videos)
else
    VIDEO_ARGS+=(--no-save-videos)
fi

cd "${BENCH_ROOT}"

echo "[MAIN] GT-action replay demos=${DEMO_INDICES} max_steps=${MAX_STEPS} gpu=${ENV_GPU_ID}"

CUDA_VISIBLE_DEVICES="${ENV_GPU_ID}" \
python "${BENCH_ROOT}/scripts/eval_pi3_gt_action_replay.py" \
    --demo-indices "${DEMO_INDICES}" \
    --max-steps "${MAX_STEPS}" \
    "${VIDEO_ARGS[@]}"

echo "[MAIN] GT-action replay finished"
