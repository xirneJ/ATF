#!/usr/bin/env python3
"""Compute FID reference statistics (ref_mu / ref_sigma) from a real-image folder.

The training/eval code loads these via `config.fid.cache_ref` (see
utils/fid_util.get_reference). Standard ImageNet-256 FID uses 50k real images
sampled across all 1000 classes; pass `--per-class 50` for that.

Usage (full ImageNet-256 reference, 50k images = 50 per class):
    python scripts/compute_fid_ref.py \
        --data-dir /mnt/win/data/imagenet/train \
        --per-class 50 --num-samples 50000 \
        --out $HOME/qiangchen/data/fid/imagenet256_ref.npz

Usage (10-class subset reference, e.g. 200 images per class):
    python scripts/compute_fid_ref.py \
        --data-dir ~/qiangchen/data/imagenet_subset10_split/ref \
        --image-size 256 \
        --out $HOME/qiangchen/data/fid/subset10_ref.npz

Notes:
  - `--data-dir` must contain class sub-folders (ImageFolder layout).
  - Images are center-cropped to `--image-size` and fed to the same JAX
    InceptionV3 used at FID evaluation time, so the statistics are consistent.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader

from utils import fid_util
from utils.input_pipeline import center_crop_arr
from utils.logging_util import log_for_0


class ImageListDataset(Dataset):
    def __init__(self, paths, image_size):
        self.paths = paths
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = pil_loader(self.paths[idx])
        arr = center_crop_arr(img, self.image_size)
        return np.array(arr), 0


def collect_paths(data_dir, per_class, num_samples, seed=0):
    classes = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    if not classes:
        raise ValueError(f"no class sub-folders found under {data_dir}")
    rng = random.Random(seed)
    paths = []
    for c in classes:
        files = sorted(
            os.path.join(data_dir, c, f) for f in os.listdir(os.path.join(data_dir, c))
        )
        n = per_class if per_class > 0 else len(files)
        if len(files) > n:
            files = rng.sample(files, n)
        paths.extend(files)
        log_for_0(f"{c}: {len(files)} images")
    if num_samples > 0 and len(paths) > num_samples:
        paths = rng.sample(paths, num_samples)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="folder with class sub-directories (ImageFolder layout)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--per-class", type=int, default=0,
                    help="max images per class (0 = all)")
    ap.add_argument("--num-samples", type=int, default=50000,
                    help="max images total (0 = all)")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        print(f"reference stats already exist: {args.out} (use --overwrite)")
        return

    paths = collect_paths(args.data_dir, args.per_class, args.num_samples)
    log_for_0(f"computing FID reference stats from {len(paths)} images")

    ds = ImageListDataset(paths, args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, drop_last=False)

    inception_net = fid_util.build_jax_inception(batch_size=args.batch_size)

    feats = []
    n = 0
    for i, (images, _) in enumerate(dl):
        images = np.asarray(images)
        feats.append(fid_util.compute_batch_features(images, inception_net,
                                                     args.batch_size))
        n += images.shape[0]
        if i % 20 == 0:
            log_for_0(f"processed {n}/{len(paths)} images")

    feats = np.concatenate(feats, axis=0).astype(np.float64)
    log_for_0(f"feature matrix shape: {feats.shape}")

    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, ref_mu=mu, ref_sigma=sigma)
    log_for_0(f"saved FID reference stats to {args.out}")
    log_for_0(f"  mu: {mu.shape}, sigma: {sigma.shape}")


if __name__ == "__main__":
    main()
