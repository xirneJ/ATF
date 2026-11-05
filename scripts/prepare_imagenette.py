#!/usr/bin/env python3
"""Download ImageNette (10-class ImageNet subset) and export it as ImageFolder.

ImageNette is a small, public 10-class subset of ImageNet, useful for
debugging / method comparisons on a single GPU. It is downloaded from the
HuggingFace Hub (`frgfm/imagenette`) and written as:

    <dst>/train/<class>/*.jpg
    <dst>/val/<class>/*.jpg

Usage:
    python scripts/prepare_imagenette.py --dst $HOME/qiangchen/data/imagenette
    # in mainland China, if huggingface.co is slow/blocked:
    HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_imagenette.py --dst ...

Notes:
    - images are written as JPEG (quality 95) at the resolution of the chosen
      config (320px by default -> min side 320, good enough for 256x256 training)
    - the output directory can be used directly as `dataset.root`
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def class_names(ds):
    feature = ds.features["label"]
    return list(feature.names)


def export_split(ds, split_dir: Path, names):
    split_dir.mkdir(parents=True, exist_ok=True)
    counters = {n: 0 for n in names}
    for i, example in enumerate(ds):
        label = example["label"]
        cls = names[label]
        out_dir = split_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        counters[cls] += 1
        out_path = out_dir / f"{counters[cls]:06d}.jpg"
        example["image"].convert("RGB").save(out_path, quality=95)
        if (i + 1) % 500 == 0:
            print(f"  {split_dir.name}: {i + 1}/{len(ds)}", flush=True)
    print(f"{split_dir.name}: {len(ds)} images -> {split_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="output ImageFolder root")
    ap.add_argument(
        "--config",
        default="320px",
        choices=["160px", "320px", "full_size"],
        help="ImageNette resolution config",
    )
    args = ap.parse_args()

    dst = Path(args.dst)
    print(f"loading frgfm/imagenette [{args.config}] ...", flush=True)
    train = load_dataset("frgfm/imagenette", args.config, split="train")
    names = class_names(train)
    print(f"classes: {names}")

    export_split(train, dst / "train", names)
    try:
        val = load_dataset("frgfm/imagenette", args.config, split="validation")
        export_split(val, dst / "val", names)
    except Exception as e:  # some configs may only expose a train split
        print(f"[warn] could not load validation split: {e}")

    print(f"done -> {dst}")
    print("Use it as dataset.root, with dataset.num_classes = 10")


if __name__ == "__main__":
    main()
