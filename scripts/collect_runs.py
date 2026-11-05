#!/usr/bin/env python3
"""Collect run records (run_summary.json) into a table for the paper.

Every training/eval run writes, inside its `--workdir`:
    config.yaml       - full config snapshot
    metrics.jsonl     - one JSON record per logging step (loss curves)
    run_summary.json  - status / duration / git commit / config / last metrics
    checkpoint_*      - checkpoints (count controlled by training.checkpoint_keep)
    images/           - sampled images (when use_wandb=False)

This script scans a root directory and prints a markdown table, optionally
writing it to a file and/or a CSV.

Usage:
    python scripts/collect_runs.py --root $HOME/qiangchen/logs
    python scripts/collect_runs.py --root $HOME/qiangchen/logs \
        --md docs/runs.md --csv docs/runs.csv
"""

import argparse
import csv
import json
from pathlib import Path


def short(v, ndigits=4):
    if isinstance(v, float):
        return f"{v:.{ndigits}g}"
    return "" if v is None else str(v)


def load_runs(root: Path):
    runs = []
    for path in sorted(root.rglob("run_summary.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")
            continue
        runs.append((path.parent, data))
    return runs


def extract(workdir: Path, data: dict) -> dict:
    cfg = data.get("config", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    dataset = cfg.get("dataset", {})
    metrics = data.get("last_metrics") or {}
    return {
        "run": workdir.name,
        "path": str(workdir),
        "status": data.get("status", ""),
        "step": data.get("step", ""),
        "duration_min": round(float(data.get("duration_sec", 0.0)) / 60.0, 1),
        "git": str(data.get("git_commit", ""))[:8],
        "model": model.get("model_str", ""),
        "deriv": model.get("deriv_mode", ""),
        "cheb": model.get("cheb_order", ""),
        "delta": model.get("fd_delta", ""),
        "reg": model.get("reg_lambda", ""),
        "batch": training.get("batch_size", ""),
        "lr": training.get("learning_rate", ""),
        "epochs": training.get("num_epochs", ""),
        "max_steps": training.get("max_steps", 0),
        "data": "synthetic" if dataset.get("synthetic") else dataset.get("root", ""),
        "loss": short(metrics.get("loss")),
        "loss_vel": short(metrics.get("loss_vel")),
        "u_abs": short(metrics.get("u_abs")),
        "du_dt_abs": short(metrics.get("du_dt_abs")),
        "steps_per_s": short(metrics.get("steps_per_second")),
        "jax_backend": data.get("jax_backend", ""),
        "summary_json": str(workdir / "run_summary.json"),
        "metrics_jsonl": str(workdir / "metrics.jsonl"),
    }


COLUMNS = [
    "run", "status", "step", "duration_min", "model", "deriv", "cheb", "delta",
    "batch", "max_steps", "data", "loss", "loss_vel", "u_abs", "du_dt_abs",
    "steps_per_s", "git",
]


def write_markdown(rows, path: Path):
    lines = ["| " + " | ".join(COLUMNS) + " |",
             "|" + "|".join(["---"] * len(COLUMNS)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory to scan (e.g. logs/)")
    ap.add_argument("--md", default=None, help="optional markdown output path")
    ap.add_argument("--csv", default=None, help="optional csv output path")
    args = ap.parse_args()

    runs = load_runs(Path(args.root))
    if not runs:
        print(f"no run_summary.json found under {args.root}")
        return
    rows = [extract(workdir, data) for workdir, data in runs]

    header = " | ".join(COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]) for c in COLUMNS))

    if args.md:
        write_markdown(rows, Path(args.md))
    if args.csv:
        write_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
