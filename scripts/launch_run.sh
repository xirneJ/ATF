#!/usr/bin/env bash
# Launch a training/eval run whose records live inside the repo:
#   <repo>/runs/<run_name>/
#       config.yaml, metrics.jsonl, run_summary.json, checkpoint_*, images/, run.log
#
# Usage:
#   bash scripts/launch_run.sh <run_name> <config_mode> [extra main.py flags]
#
# Examples:
#   bash scripts/launch_run.sh 20261105_tiny_fd  pMF_tiny_debug
#   bash scripts/launch_run.sh 20261105_real_fd  pMF_B_16_real_debug
#   bash scripts/launch_run.sh 20261105_real_jvp pMF_B_16_real_debug_jvp
#
# Notes:
#   - XLA_PYTHON_CLIENT_PREALLOCATE defaults to false (shared GPUs); override by
#     exporting it before calling this script.
#   - heavy artifacts (checkpoints / images) are git-ignored via .gitignore.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <run_name> <config_mode> [extra main.py flags]" >&2
    echo "example: $0 20261105_tiny_fd pMF_tiny_debug" >&2
    exit 1
fi

RUN_NAME="$1"
CONFIG_MODE="$2"
shift 2

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "[launch_run] run dir : $RUN_DIR"
echo "[launch_run] config  : configs/load_config.py:$CONFIG_MODE"
echo "[launch_run] mem frac: $XLA_PYTHON_CLIENT_MEM_FRACTION (preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE)"

cd "$REPO_DIR"
python main.py \
    --workdir="$RUN_DIR" \
    --config="configs/load_config.py:$CONFIG_MODE" \
    "$@" 2>&1 | tee "$RUN_DIR/run.log"
