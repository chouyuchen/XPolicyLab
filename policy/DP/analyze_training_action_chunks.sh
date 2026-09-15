#!/bin/bash
set -euo pipefail
# Offline action-chunk analysis: initial window of 5 training episodes × 5 diffusion samples.
# Does not modify training or deployment code.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xpl_dp

python analyze_training_action_chunks.py \
  --ckpt checkpoints/cabinet_chain_nominal_dr-skill_close_loaded_drawer-aloha_agilex-joint-0/150.ckpt \
  --zarr data/cabinet_chain_nominal_dr-skill_close_loaded_drawer-aloha_agilex-joint.zarr \
  --episodes 27,32,135,146,196 \
  --num-samples 5 \
  --device cuda:0 \
  "$@"
