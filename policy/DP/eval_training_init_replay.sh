#!/bin/bash
set -euo pipefail
# Training initial-state replay for π3 close drawer.
# Usage:
#   bash eval_training_init_replay.sh [num_demos] [repeats] [policy_gpu] [env_gpu]
# Example (5 demos x 3 repeats = 15 rollouts):
#   bash eval_training_init_replay.sh 5 3 0 0

NUM_DEMOS=${1:-5}
REPEATS=${2:-3}
POLICY_GPU_ID=${3:-0}
ENV_GPU_ID=${4:-0}
POLICY_CONDA_ENV=${5:-xpl_dp}
EVAL_CONDA_ENV=${6:-RoboTwin}

BENCH_NAME=cabinet_chain_nominal_dr
TASK_NAME=skill_close_loaded_drawer
CKPT_NAME=cabinet_chain_nominal_dr-skill_close_loaded_drawer-aloha_agilex-joint-0
ENV_CFG_TYPE=aloha_agilex
ACTION_TYPE=joint
SEED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[MAIN] training init replay: demos=${NUM_DEMOS}, repeats=${REPEATS}, port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${BENCH_NAME}" \
    "${TASK_NAME}" \
    "${CKPT_NAME}" \
    "${ENV_CFG_TYPE}" \
    "${ACTION_TYPE}" \
    "${SEED}" \
    "${POLICY_GPU_ID}" \
    "${POLICY_CONDA_ENV}" \
    "${policy_server_port}" &

SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${EVAL_CONDA_ENV}"
set -u

SAVE_VIDEOS=${SAVE_VIDEOS:-0}
VIDEO_ARGS=()
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
    VIDEO_ARGS+=(--save-videos)
fi

cd "${BENCH_ROOT}"

CUDA_VISIBLE_DEVICES="${ENV_GPU_ID}" \
python "${BENCH_ROOT}/scripts/eval_pi3_training_init_replay.py" \
    --host "${policy_server_ip}" \
    --port "${policy_server_port}" \
    --num-demos "${NUM_DEMOS}" \
    --repeats "${REPEATS}" \
    "${VIDEO_ARGS[@]}"

echo "[MAIN] training init replay finished"
