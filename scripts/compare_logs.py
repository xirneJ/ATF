#!/usr/bin/env python3
"""Compare `loss_vel` (and friends) between two training logs.

The training loop prints lines like:
    [100] aux_loss_convnext=0, du_dt_abs=..., loss=1, loss_reg=..., loss_vel=1.2...

Usage:
    python scripts/compare_logs.py --a logs/fd.log --b logs/jvp.log
"""

import argparse
import re

PATTERN = re.compile(r"^\[(\d+)\].*?loss_vel=([0-9.eE+\-]+).*?du_dt_abs=([0-9.eE+\-]+)")


def parse(path):
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                step = int(m.group(1))
                out[step] = (float(m.group(2)), float(m.group(3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="log of run A (e.g. fd)")
    ap.add_argument("--b", required=True, help="log of run B (e.g. jvp)")
    args = ap.parse_args()

    a, b = parse(args.a), parse(args.b)
    steps = sorted(set(a) & set(b))
    if not steps:
        print("no common steps found; check the logs")
        return
    print(f"{'step':>6} | {'A.loss_vel':>12} {'B.loss_vel':>12} {'rel':>9} | "
          f"{'A.du_dt':>10} {'B.du_dt':>10} {'rel':>9}")
    print("-" * 88)
    for s in steps:
        la, da = a[s]
        lb, db = b[s]
        rel_l = abs(la - lb) / (abs(la) + 1e-12)
        rel_d = abs(da - db) / (abs(da) + 1e-12)
        print(f"{s:>6} | {la:>12.6f} {lb:>12.6f} {rel_l:>9.2e} | "
              f"{da:>10.6f} {db:>10.6f} {rel_d:>9.2e}")

    rels = [abs(a[s][0] - b[s][0]) / (abs(a[s][0]) + 1e-12) for s in steps]
    print("-" * 88)
    print(f"common steps: {len(steps)}, max rel diff (loss_vel): {max(rels):.2e}, "
          f"mean rel diff: {sum(rels) / len(rels):.2e}")


if __name__ == "__main__":
    main()
