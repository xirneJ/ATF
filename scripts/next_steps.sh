#!/usr/bin/env bash
# One-command continuation after the ablation suite:
#   1) run missing ablations (cheb2, reg1e3)
#   2) summarize the ablation table
#   3) create subset train/ref split + FID reference stats
#   4) FID pipeline smoke test on an existing checkpoint
#   5) traditional-pMF baseline correctness check
#
# Usage:
#   bash scripts/next_steps.sh
#
# Everything is skippable/re-runnable: finished steps are detected and skipped.
# Estimated time on a shared 5090: ~1-1.5 h (mostly the two ablations).

set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_DIR="$(pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

DATA_SUBSET="${DATA_SUBSET:-/home/xyds25/qiangchen/data/imagenet_subset10}"
DATA_SPLIT="${DATA_SPLIT:-/home/xyds25/qiangchen/data/imagenet_subset10_split}"
FID_REF="${FID_REF:-/home/xyds25/qiangchen/data/fid/subset10_ref.npz}"
SMOKE_CKPT="${SMOKE_CKPT:-$REPO_DIR/runs/20261106_real_fd/checkpoint_3250}"

step() {
    echo
    echo "=================================================="
    echo "[next_steps] $*"
    echo "=================================================="
}

# True if runs/<name>/metrics.jsonl has reached step >= 1999.
has_ablation() {
    local f="runs/$1/metrics.jsonl"
    [ -f "$f" ] || return 1
    python - "$f" <<'PY'
import json, sys
last = 0
for line in open(sys.argv[1], errors="ignore"):
    line = line.strip()
    if not line:
        continue
    try:
        last = max(last, int(json.loads(line).get("step", 0)))
    except Exception:
        pass
sys.exit(0 if last >= 1999 else 1)
PY
}

echo "GPU status:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true

# ---------------------------------------------------------------- 1) ablations
for pair in "abl_cheb2 ablation/cheb2" "abl_reg1e3 ablation/reg1e3"; do
    set -- $pair
    if has_ablation "$1"; then
        echo "[skip] $1 already finished"
    else
        step "ablation $1"
        bash scripts/launch_run.sh "$1" "$2" || echo "[warn] $1 failed"
    fi
done

# ---------------------------------------------------------------- 2) summary
step "ablation summary"
python scripts/ablation_summary.py --root runs --steps 2000 \
    --md docs/ablation_table.md --csv docs/ablation_table.csv || true

# ---------------------------------------------------- 3) split + FID ref stats
step "subset split + FID reference stats"
if [ -d "$DATA_SPLIT" ]; then
    echo "[skip] $DATA_SPLIT exists"
else
    python scripts/make_subset_split_for_fid.py --src "$DATA_SUBSET" --dst "$DATA_SPLIT" \
        --train-per-class 1000 --ref-per-class 300 || echo "[warn] split failed"
fi
if [ -f "$FID_REF" ]; then
    echo "[skip] $FID_REF exists"
else
    mkdir -p "$(dirname "$FID_REF")"
    python scripts/compute_fid_ref.py --data-dir "$DATA_SPLIT/ref" --out "$FID_REF" \
        || echo "[warn] fid reference stats failed"
fi

# ------------------------------------------- 4) FID pipeline smoke test
step "FID pipeline smoke test (existing checkpoint)"
if [ -d "$SMOKE_CKPT" ]; then
    cp configs/fid/subset10_eval_cheb_config.yml /tmp/eval_smoke.yml
    sed -i "s|load_from: ''|load_from: $SMOKE_CKPT|" /tmp/eval_smoke.yml
    grep load_from /tmp/eval_smoke.yml
    python main.py --workdir=runs/eval_smoke_cheb --config=/tmp/eval_smoke.yml \
        || echo "[warn] FID smoke test failed"
else
    echo "[skip] checkpoint not found: $SMOKE_CKPT"
fi

# ------------------------------------------- 5) traditional pMF baseline check
step "traditional pMF baseline correctness check"
if has_ablation "chk_direct_jvp"; then
    echo "[skip] chk_direct_jvp already finished"
else
    bash scripts/launch_run.sh chk_direct_jvp direct_jvp_b2 \
        || echo "[warn] baseline check failed"
fi
python scripts/compare_logs.py --a runs/chk_direct_jvp --b runs/abl_cheb0 --smooth 5 || true

step "done"
echo
echo "Send back to the assistant:"
echo "  - docs/ablation_table.md"
echo "  - FID_* / IS_* numbers printed by the smoke test"
echo "  - the compare_logs output above"
echo
echo "Optional next (expensive, 6-15h each):"
echo "  bash scripts/launch_run.sh subset10_pilot_ours   fid/subset10_pilot_b2"
echo "  bash scripts/launch_run.sh subset10_pilot_direct fid/subset10_pilot_direct_jvp_b2"
