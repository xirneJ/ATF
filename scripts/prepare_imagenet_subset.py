#!/usr/bin/env python3
"""Create a small ImageFolder subset of ImageNet by symlinking the first N classes.

Example:
    python scripts/prepare_imagenet_subset.py \
        --src /data/imagenet --dst /data/imagenet_subset10 --num-classes 10

The resulting directory can be used directly as `dataset.root`:
    /data/imagenet_subset10/train/<class>/*.JPEG
    /data/imagenet_subset10/val/<class>/*.JPEG

Symlinks are used, so this costs almost no extra disk space. `torchvision`'s
ImageFolder follows symlinked class directories fine.
"""

import argparse
from pathlib import Path


def count_images(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file())


def link_split(src_split: Path, dst_split: Path, classes):
    dst_split.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        src_cls = src_split / cls
        if not src_cls.is_dir():
            raise FileNotFoundError(f"missing class directory: {src_cls}")
        dst_cls = dst_split / cls
        if dst_cls.is_symlink() or dst_cls.exists():
            dst_cls.unlink()
        dst_cls.symlink_to(src_cls.resolve(), target_is_directory=True)
        print(f"{dst_split.name}/{cls}: {count_images(src_cls)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ImageNet root containing train/ and val/")
    ap.add_argument("--dst", required=True, help="output subset root")
    ap.add_argument("--num-classes", type=int, default=10)
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
    print(f"linking {len(classes)} classes")
    link_split(train_dir, dst / "train", classes)
    if (src / "val").is_dir():
        link_split(src / "val", dst / "val", classes)
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
