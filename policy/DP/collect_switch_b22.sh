#!/bin/bash
# B2.2: collect flag-oracle CONTINUE/SWITCH frames (TEST_NUM=3 default).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TEST_NUM="${TEST_NUM:-3}"
SAMPLE_EVERY_N="${SAMPLE_EVERY_N:-8}"
SEED_START="${SEED_START:-100000}"
STAGES="${STAGES:-open,load,close}"
# Default: single-skill open thr (0.50) so π1 yields SWITCH labels.
# Set OPEN_FRAC_THRESHOLD=0.85 for chain handoff semantics.
OPEN_FRAC_THRESHOLD="${OPEN_FRAC_THRESHOLD:-0.5}"
TASK_CONFIG="${TASK_CONFIG:-cabinet_chain_nominal_dr}"
POLICY_GPU="${POLICY_GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate hooks (e.g. cuda-nvcc) touch unset vars under `set -u`
set +u
conda activate RoboTwin
set -u
cd "${BENCH_ROOT}"

extra=()
if [[ -n "${OUTPUT_DIR}" ]]; then
  extra+=(--output-dir "${OUTPUT_DIR}")
fi

python scripts/collect_switch_dataset.py \
  --task-config "${TASK_CONFIG}" \
  --test-num "${TEST_NUM}" \
  --seed-start "${SEED_START}" \
  --sample-every-n "${SAMPLE_EVERY_N}" \
  --policy-gpu "${POLICY_GPU}" \
  --stages "${STAGES}" \
  --open-frac-threshold "${OPEN_FRAC_THRESHOLD}" \
  "${extra[@]}"

echo "[B2.2] collect finished"
