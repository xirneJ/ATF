#!/usr/bin/env python3
"""Compare `loss_vel` / `du_dt_abs` / `u_abs` / `loss_reg` between two runs.

Preferred input: a run directory (auto-detected) or a `metrics.jsonl` file.
Fallback input: a raw `.log` file (parses `[step] key=value, ...` lines, tolerant
to the absl logging prefix and to field order).

Usage:
    # recommended: point at the run directories (uses metrics.jsonl)
    python scripts/compare_logs.py --a runs/20261106_real_fd --b runs/20261106_real_jvp

    # or point at files directly
    python scripts/compare_logs.py --a runs/20261106_real_fd/metrics.jsonl \
                                   --b runs/20261106_real_jvp/metrics.jsonl
    python scripts/compare_logs.py --a logs/real_fd.log --b logs/real_jvp.log
"""

import argparse
import json
import re
from pathlib import Path

METRIC_KEYS = ["loss_vel", "du_dt_abs", "u_abs", "loss_reg"]
STEP_RE = re.compile(r"\[(\d+)\]")
KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)="
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_metrics_jsonl(path: Path):
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step is None:
                continue
            out[int(step)] = {k: rec.get(k) for k in METRIC_KEYS}
    return out


def parse_log(path: Path):
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            step_match = STEP_RE.search(line)
            if not step_match or "loss_vel" not in line:
                continue
            step = int(step_match.group(1))
            kv = {}
            for k, v in KV_RE.findall(line):
                if k in METRIC_KEYS:
                    try:
                        kv[k] = float(v)
                    except ValueError:
                        pass
            if "loss_vel" in kv:
                out[step] = kv
    return out


def load(path_str: str):
    path = Path(path_str)
    if path.is_dir():
        jsonl = path / "metrics.jsonl"
        if jsonl.exists():
            return parse_metrics_jsonl(jsonl), str(jsonl)
        log = path / "run.log"
        if log.exists():
            return parse_log(log), str(log)
        raise FileNotFoundError(f"{path} contains neither metrics.jsonl nor run.log")
    if path.suffix == ".jsonl":
        return parse_metrics_jsonl(path), str(path)
    return parse_log(path), str(path)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / (abs(a) + 1e-12)


def moving_average(values, window):
    """Centered moving average over a list (None entries are skipped)."""
    if window <= 1:
        return values
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = [v for v in values[lo:hi] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run directory / metrics.jsonl / log of run A")
    ap.add_argument("--b", required=True, help="run directory / metrics.jsonl / log of run B")
    ap.add_argument("--keys", nargs="*", default=["loss_vel", "du_dt_abs"],
                    help="which metric keys to compare")
    ap.add_argument("--smooth", type=int, default=1,
                    help="moving-average window (in records) for both runs, e.g. 9")
    ap.add_argument("--every", type=int, default=0,
                    help="print only every k-th row (0 = auto: 1 if no smoothing else 10)")
    args = ap.parse_args()

    try:
        a, a_src = load(args.a)
        b, b_src = load(args.b)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"A = {a_src}  ({len(a)} records)")
    print(f"B = {b_src}  ({len(b)} records)")
    if not a or not b:
        print("\nNo records parsed. Check that the run actually logged (see run.log) "
              "or that metrics.jsonl exists.")
        return

    common = sorted(set(a) & set(b))
    if not common:
        print(f"\nNo common steps. A steps: {sorted(a)[:5]}...{sorted(a)[-3:]}, "
              f"B steps: {sorted(b)[:5]}...{sorted(b)[-3:]}")
        return

    keys = [k for k in args.keys if any(k in a[s] for s in common)]
    if not keys:
        print(f"None of the requested keys {args.keys} found in the records.")
        return

    # Optionally smooth both curves (useful when batch is small and curves are noisy).
    series = {}
    for k in keys:
        va = [a[s].get(k) for s in common]
        vb = [b[s].get(k) for s in common]
        if args.smooth > 1:
            va = moving_average(va, args.smooth)
            vb = moving_average(vb, args.smooth)
        series[k] = (va, vb)

    every = args.every if args.every > 0 else (1 if args.smooth <= 1 else 10)
    print(f"\nsmooth window = {args.smooth}, printing every {every} record(s)")

    header = f"{'step':>6} | " + " | ".join(
        f"{'A.' + k:>13} {'B.' + k:>13} {'rel':>9}" for k in keys
    )
    print(header)
    print("-" * len(header))
    rels = {k: [] for k in keys}
    for i, s in enumerate(common):
        cells = []
        for k in keys:
            va, vb = series[k][0][i], series[k][1][i]
            r = rel(va, vb)
            if r is not None:
                rels[k].append(r)
            if i % every == 0 or i == len(common) - 1:
                cells.append(f"{fmt(va):>13} {fmt(vb):>13} {('%.2e' % r) if r is not None else '-':>9}")
        if cells:
            print(f"{s:>6} | " + " | ".join(cells))

    print("-" * len(header))
    print(f"common steps: {len(common)} (smooth={args.smooth})")
    for k in keys:
        if rels[k]:
            print(f"  {k}: max rel diff = {max(rels[k]):.2e}, mean = {sum(rels[k]) / len(rels[k]):.2e}")


if __name__ == "__main__":
    main()
