#!/usr/bin/env python3
"""Create a small ImageFolder subset of ImageNet by symlinking (or copying) the
first N classes.

Example (symlink, no extra disk space):
    python scripts/prepare_imagenet_subset.py \
        --src /mnt/win/data/imagenet --dst ~/qiangchen/data/imagenet_subset10 \
        --num-classes 10

Example (copy to local disk, useful if the source mount is slow):
    python scripts/prepare_imagenet_subset.py \
        --src /mnt/win/data/imagenet --dst ~/qiangchen/data/imagenet_subset10 \
        --num-classes 10 --copy

Notes:
    - `train/` must be a standard ImageFolder tree (class directories).
    - `val/` is linked only if it is also an ImageFolder tree; a flat ImageNet
      val split (ILSVRC2012_val_*.JPEG files) is skipped with a warning --
      training does not need val.
    - output layout: <dst>/train/<class>/..., <dst>/val/<class>/...
"""

import argparse
import shutil
from pathlib import Path


def count_images(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file())


def export_split(src_split: Path, dst_split: Path, classes, copy: bool) -> int:
    dst_split.mkdir(parents=True, exist_ok=True)
    linked = 0
    for cls in classes:
        src_cls = src_split / cls
        if not src_cls.is_dir():
            print(f"[warn] {src_cls} not found, skipping")
            continue
        dst_cls = dst_split / cls
        if dst_cls.is_symlink() or dst_cls.exists():
            if dst_cls.is_symlink() or dst_cls.is_file():
                dst_cls.unlink()
            else:
                shutil.rmtree(dst_cls)
        if copy:
            shutil.copytree(src_cls, dst_cls)
        else:
            dst_cls.symlink_to(src_cls.resolve(), target_is_directory=True)
        print(f"{dst_split.name}/{cls}: {count_images(src_cls)} images")
        linked += 1
    return linked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ImageNet root containing train/ and val/")
    ap.add_argument("--dst", required=True, help="output subset root")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    train_dir = src / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"{train_dir} not found; expected ImageNet layout")

    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if args.num_classes > len(classes):
        raise ValueError(f"requested {args.num_classes} classes, found {len(classes)}")
    classes = classes[: args.num_classes]
    print(f"exporting {len(classes)} classes to {dst} (copy={args.copy})")

    n_train = export_split(train_dir, dst / "train", classes, args.copy)
    if n_train == 0:
        raise RuntimeError("no training classes were found")
    if (src / "val").is_dir():
        n_val = export_split(src / "val", dst / "val", classes, args.copy)
        if n_val == 0:
            print("[warn] val split is not ImageFolder-shaped (flat JPEGs) -- skipped; "
                  "training does not need it")
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
