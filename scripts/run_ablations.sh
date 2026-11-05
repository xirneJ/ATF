#!/usr/bin/env bash
# Run the ablation suite sequentially. Every ablation config stops at
# training.max_steps=2000 (see configs/ablation/*_config.yml), so one run is
# ~15-20 min on a shared 5090 at batch 2.
#
# Usage:
#   bash scripts/run_ablations.sh            # run names: abl_<axis>
#   bash scripts/run_ablations.sh abl_v2     # custom prefix
#
# Results: runs/<prefix>_<axis>/ with config/metrics/run_summary/README/metadata.
# Summary: python scripts/ablation_summary.py --root runs

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PREFIX="${1:-abl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

names=(delta001 delta005 forward cheb0 cheb1 cheb2 reg1e3)

for n in "${names[@]}"; do
    echo "=================================================================="
    echo "[ablation] ${PREFIX}_${n}  (config: ablation/${n})"
    echo "=================================================================="
    if ! bash scripts/launch_run.sh "${PREFIX}_${n}" "ablation/${n}"; then
        echo "[warn] ablation run '${n}' failed; continuing with the next one."
    fi
done

echo
echo "[ablation] all runs finished."
echo "[ablation] summarize: python scripts/ablation_summary.py --root runs"
