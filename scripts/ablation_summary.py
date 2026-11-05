#!/usr/bin/env python3
"""Summarize ablation runs into a markdown/CSV table.

Reads `runs/<name>/run_summary.json` (config) and `runs/<name>/metrics.jsonl`
(metrics), truncates every run to the first `--steps` logging records, and
reports mean/final values so ablations can be compared apples-to-apples.

Usage:
    python scripts/ablation_summary.py --root runs
    python scripts/ablation_summary.py --root runs --steps 2000 \
        --prefix abl_ --include 20261106_real_fd_b2 20261106_real_jvp_b2 \
        20261106_real_fd_b2_seed43

Example output is a pipe table you can paste into the paper.
"""

import argparse
import csv
import json
from pathlib import Path

METRICS = ["loss_vel", "du_dt_abs", "u_abs", "loss_reg"]


def load_run(run_dir: Path):
    cfg, recs = {}, []
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            cfg = json.load(open(summary_path)).get("config", {})
        except Exception:
            cfg = {}
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        for line in open(metrics_path, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return cfg, recs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def collect(root: Path, prefix: str, include, steps: int):
    rows = []
    dirs = [root / n for n in include]
    dirs += [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith(prefix)]
    seen = set()
    for d in dirs:
        if not d.is_dir() or d.name in seen:
            continue
        seen.add(d.name)
        cfg, recs = load_run(d)
        if not recs:
            continue
        # Truncate by *step value* (not by number of records) so that runs with a
        # different amount of logging history are compared at the same horizon.
        if steps > 0:
            recs = [
                r for r in recs
                if r.get("step") is not None and int(r["step"]) <= steps
            ]
        if not recs:
            continue
        model = cfg.get("model", {})
        training = cfg.get("training", {})
        dataset = cfg.get("dataset", {})
        row = {
            "run": d.name,
            "deriv": model.get("deriv_mode", ""),
            "cheb": model.get("cheb_order", ""),
            "delta": model.get("fd_delta", ""),
            "central": model.get("fd_central", ""),
            "reg": model.get("reg_lambda", ""),
            "batch": training.get("batch_size", ""),
            "seed": training.get("seed", ""),
            "synthetic": dataset.get("synthetic", ""),
            "steps": len(recs),
            "last_step": max(int(r["step"]) for r in recs),
        }
        last_n = max(1, int(len(recs) * 0.25))
        last = recs[-last_n:]
        for k in METRICS:
            row[f"mean_{k}"] = mean([r.get(k) for r in recs])
            row[f"last_{k}"] = mean([r.get(k) for r in last])
        rows.append(row)
    return rows


COLUMNS = [
    "run", "deriv", "cheb", "delta", "central", "reg", "batch", "seed",
    "steps", "last_step",
    "mean_loss_vel", "last_loss_vel", "mean_du_dt_abs", "last_du_dt_abs",
    "last_u_abs", "last_loss_reg",
]


def print_table(rows):
    header = " | ".join(COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(fmt(r.get(c)) for c in COLUMNS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--prefix", default="abl_")
    ap.add_argument("--include", nargs="*", default=[
        "20261106_real_fd_b2", "20261106_real_jvp_b2", "20261106_real_fd_b2_seed43",
    ])
    ap.add_argument("--steps", type=int, default=2000,
                    help="truncate every run to records with step <= STEPS (0 = all)")
    ap.add_argument("--md", default=None, help="optional markdown output path")
    ap.add_argument("--csv", default=None, help="optional csv output path")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return
    rows = collect(root, args.prefix, args.include, args.steps)
    if not rows:
        print("No runs found. Check --root / --prefix / --include.")
        return

    print_table(rows)

    if args.md:
        lines = ["| " + " | ".join(COLUMNS) + " |",
                 "|" + "|".join(["---"] * len(COLUMNS)) + "|"]
        for r in rows:
            lines.append("| " + " | ".join(fmt(r.get(c)) for c in COLUMNS) + " |")
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.md}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c) for c in COLUMNS})
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
