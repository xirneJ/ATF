#!/usr/bin/env python3
"""Create a train/ref split of a class-folder dataset (for subset FID).

FID needs reference (real) statistics that are *not* seen during training.
For the 10-class ImageNet subset this script splits each class into:
    <dst>/train/<class>/...   (symlinks, used for training)
    <dst>/ref/<class>/...     (symlinks, used to compute FID reference stats)

Usage:
    python scripts/make_subset_split_for_fid.py \
        --src ~/qiangchen/data/imagenet_subset10 \
        --dst ~/qiangchen/data/imagenet_subset10_split \
        --train-per-class 1000 --ref-per-class 300

(Subset has 1300 images/class; keeping 300/class for the FID reference gives
3000 reference images, enough for a 2048-dim feature covariance estimate.)

Then compute the reference stats:
    python scripts/compute_fid_ref.py \
        --data-dir ~/qiangchen/data/imagenet_subset10_split/ref \
        --out $HOME/qiangchen/data/fid/subset10_ref.npz
"""

import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="dataset root containing train/<class>/ (e.g. imagenet_subset10)")
    ap.add_argument("--dst", required=True, help="output split root")
    ap.add_argument("--train-per-class", type=int, default=1000)
    ap.add_argument("--ref-per-class", type=int, default=300)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_train = os.path.join(args.src, "train")
    if not os.path.isdir(src_train):
        raise FileNotFoundError(f"{src_train} not found")

    if os.path.exists(args.dst):
        if not args.overwrite:
            raise FileExistsError(f"{args.dst} already exists (use --overwrite)")
        shutil.rmtree(args.dst)

    classes = sorted(
        d for d in os.listdir(src_train) if os.path.isdir(os.path.join(src_train, d))
    )
    if not classes:
        raise ValueError(f"no class dirs under {src_train}")

    for split, n in (("train", args.train_per_class), ("ref", args.ref_per_class)):
        for c in classes:
            os.makedirs(os.path.join(args.dst, split, c), exist_ok=True)

    for c in classes:
        files = sorted(
            os.path.join(src_train, c, f) for f in os.listdir(os.path.join(src_train, c))
        )
        if len(files) < args.train_per_class + args.ref_per_class:
            print(f"[warn] class {c}: only {len(files)} images "
                  f"(< train+ref = {args.train_per_class + args.ref_per_class})")
        n_train = min(args.train_per_class, len(files))
        n_ref = min(args.ref_per_class, len(files) - n_train)
        for split, chunk in (("train", files[:n_train]),
                             ("ref", files[n_train:n_train + n_ref])):
            for src in chunk:
                dst = os.path.join(args.dst, split, c, os.path.basename(src))
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(os.path.abspath(src), dst)
        print(f"{c}: train={n_train}, ref={n_ref}")

    print(f"\nsplit written to {args.dst}")
    print(f"  train: {os.path.join(args.dst, 'train')}")
    print(f"  ref  : {os.path.join(args.dst, 'ref')}")


if __name__ == "__main__":
    main()
