#!/bin/bash
# B2.2: LoRA-train Qwen2.5-VL switcher on collected manifest (xpl_vlm).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATA_DIR="${1:-}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: bash collect_switch_b22.sh  # then
  bash train_switch_b22.sh data/switch_b22/<stamp>"
  exit 1
fi
# Allow relative paths from repo root.
if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${BENCH_ROOT}/${DATA_DIR}"
fi

MAX_STEPS="${MAX_STEPS:-200}"
STAMP="$(basename "${DATA_DIR}")"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCH_ROOT}/checkpoints/switch_b22_lora/${STAMP}}"
DEVICE="${VLM_DEVICE:-cuda:0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate hooks may touch unset vars under `set -u`
set +u
conda activate xpl_vlm
set -u
cd "${BENCH_ROOT}"

python scripts/switchers/train_vlm_switch_lora.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --max-steps "${MAX_STEPS}"

echo "[B2.2] adapter → ${OUTPUT_DIR}"
echo "Eval with: VLM_MODEL=${OUTPUT_DIR} SWITCH_MODE=vlm TEST_NUM=3 bash eval_chain_naive.sh"
