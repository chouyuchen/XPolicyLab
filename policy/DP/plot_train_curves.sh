#!/usr/bin/env bash
# Thin wrapper around plot_train_curves.py
# Usage:
#   bash plot_train_curves.sh --latest
#   bash plot_train_curves.sh --all --compare
#   bash plot_train_curves.sh data/outputs/.../<run_dir>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
python plot_train_curves.py "$@"
