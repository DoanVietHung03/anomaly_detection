#!/usr/bin/env python3
"""Hybrid supervised anomaly segmentation on top of PatchCore maps.

Workflow:
1. Split VisA into supervised train/val/test folders.
2. Use infer_demo.py + PatchCore to export anomaly maps for each split.
3. Train a small 4-channel U-Net on RGB + anomaly_map -> defect mask.

The image-level good/bad decision is calibrated from validation predictions,
while pixel metrics are reported with the best validation mask threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_SIZE = (384, 384)
DEFAULT_PATCHCORE_LAYERS = ("layer2", "layer3")
DEFAULT_PATCHCORE_CORESET_RATIO = 0.10
DEFAULT_PATCHCORE_NUM_NEIGHBORS = 9
DEFAULT_PATCHCORE_PRECISION = "float16"
SPLITS = ("train", "val", "test")
LABELS = ("good", "bad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hybrid PatchCore + supervised U-Net segmentation workflow for VisA.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("./VisA"), help="Path to the VisA root folder.")
    parser.add_argument("--category", default="candle", help="VisA category, for example candle.")
    parser.add_argument("--split-dir", type=Path, default=None, help="Prepared supervised split folder.")
    parser.add_argument("--maps-dir", type=Path, default=None, help="PatchCore map output root for train/val/test.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Hybrid model output directory.")
    parser.add_argument(
        "--patchcore-results-dir",
        type=Path,
        default=Path("./runs_patchcore"),
        help="Existing PatchCore training output directory searched by infer_demo.py.",
    )
    parser.add_argument(
        "--patchcore-checkpoint",
        type=Path,
        default=None,
        help="Direct PatchCore checkpoint path. Overrides --patchcore-results-dir during map generation.",
    )
    parser.add_argument(
        "--train-patchcore",
        action="store_true",
        help="Train PatchCore stage 1 before generating hybrid maps.",
    )
    parser.add_argument("--patchcore-epochs", type=int, default=1, help="PatchCore epochs when --train-patchcore is set.")
    parser.add_argument("--patchcore-train-batch-size", type=int, default=1, help="PatchCore train batch size.")
    parser.add_argument("--patchcore-eval-batch-size", type=int, default=1, help="PatchCore eval/infer batch size.")
    parser.add_argument("--patchcore-layers", nargs="+", default=list(DEFAULT_PATCHCORE_LAYERS))
    parser.add_argument("--patchcore-coreset-ratio", type=float, default=DEFAULT_PATCHCORE_CORESET_RATIO)
    parser.add_argument("--patchcore-num-neighbors", type=int, default=DEFAULT_PATCHCORE_NUM_NEIGHBORS)
    parser.add_argument("--patchcore-precision", choices=["float16", "float32"], default=DEFAULT_PATCHCORE_PRECISION)
    parser.add_argument("--image-size", type=int, default=None, help="Square image size for PatchCore maps and U-Net.")
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--train-good", type=int, default=800)
    parser.add_argument("--train-bad", type=int, default=60)
    parser.add_argument("--val-good", type=int, default=50)
    parser.add_argument("--val-bad", type=int, default=20)
    parser.add_argument("--test-good", type=int, default=50)
    parser.add_argument("--test-bad", type=int, default=20)
    parser.add_argument("--hybrid-epochs", type=int, default=40)
    parser.add_argument("--hybrid-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=30.0)
    parser.add_argument(
        "--image-score-mode",
        choices=["max", "p99", "area", "blob", "hybrid"],
        default="hybrid",
        help="How to convert predicted masks into image-level anomaly scores.",
    )
    parser.add_argument("--image-area-weight", type=float, default=4.0)
    parser.add_argument("--image-blob-weight", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "cuda"], default="gpu")
    parser.add_argument("--devices", default="1", help="Kept for parity with existing scripts; hybrid uses one visible device.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Clean generated split/maps/output folders before running.")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse an existing --split-dir.")
    parser.add_argument("--skip-map-generation", action="store_true", help="Reuse existing PatchCore maps in --maps-dir.")
    parser.add_argument("--skip-training", action="store_true", help="Stop after data preparation/map generation.")
    parser.add_argument("--no-augment", action="store_true", help="Disable simple flips during hybrid training.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision for U-Net training.")
    parser.add_argument("--save-test-maps", action="store_true", help="Save test predicted masks and overlays.")
    return parser.parse_args()


def resolve_image_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise SystemExit("Pass both --image-height and --image-width, or neither.")
        size = (args.image_height, args.image_width)
    elif args.image_size is not None:
        size = (args.image_size, args.image_size)
    else:
        size = DEFAULT_IMAGE_SIZE
    if size[0] <= 0 or size[1] <= 0:
        raise SystemExit("Image height and width must be positive.")
    return size


def project_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def run_command(command: list[str], project_root: Path) -> None:
    print(f"\n[RUN] {format_command(command)}", flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and not p.stem.lower().endswith("_mask")
    )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def take_paths(paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    paths = list(paths)
    rng.shuffle(paths)
    if count <= 0:
        return []
    return paths[: min(count, len(paths))]


def source_mask_path(dataset_root: Path, category: str, image_path: Path) -> Path | None:
    candidate = dataset_root / category / "Data" / "Masks" / "Anomaly" / f"{image_path.stem}.png"
    return candidate if candidate.exists() else None


def copy_split_records(
    paths: list[Path],
    *,
    out_dir: Path,
    split: str,
    label: str,
    dataset_root: Path,
    category: str,
) -> list[dict[str, str]]:
    target = out_dir / split / label
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for idx, src in enumerate(paths):
        dst = target / f"{idx:03d}_{label}_{src.name}"
        shutil.copy2(src, dst)
        mask = source_mask_path(dataset_root, category, src) if label == "bad" else None
        records.append(
            {
                "split": split,
                "label": label,
                "source": str(src.resolve()),
                "destination": str(dst.resolve()),
                "mask": str(mask.resolve()) if mask else "",
            },
        )
    return records


def visa_source_pools(dataset_root: Path, category: str) -> tuple[str, dict[str, list[Path]]]:
    split_root = dataset_root / "visa_pytorch" / category
    split_train_good = list_images(split_root / "train" / "good")
    split_test_good = list_images(split_root / "test" / "good")
    split_test_bad = list_images(split_root / "test" / "bad")
    if split_train_good and split_test_good and split_test_bad:
        return (
            "visa_pytorch",
            {
                "train_good": split_train_good,
                "eval_good": split_test_good,
                "bad": split_test_bad,
            },
        )

    data_root = dataset_root / category / "Data" / "Images"
    good = list_images(data_root / "Normal")
    bad = list_images(data_root / "Anomaly")
    if not good or not bad:
        raise SystemExit(
            "Could not find VisA images. Expected either visa_pytorch/<category>/... "
            f"or raw folders under {data_root}.",
        )
    return ("raw", {"good": good, "bad": bad})


def prepare_hybrid_split(args: argparse.Namespace, split_dir: Path) -> dict[str, Any]:
    rng = random.Random(args.seed)
    if args.clean:
        reset_dir(split_dir)
    else:
        split_dir.mkdir(parents=True, exist_ok=True)

    layout, pools = visa_source_pools(args.dataset_root, args.category)
    records: list[dict[str, str]] = []
    counts = {
        "train": {"good": args.train_good, "bad": args.train_bad},
        "val": {"good": args.val_good, "bad": args.val_bad},
        "test": {"good": args.test_good, "bad": args.test_bad},
    }

    if layout == "visa_pytorch":
        train_good = take_paths(pools["train_good"], args.train_good, rng)
        eval_good_pool = list(pools["eval_good"])
        rng.shuffle(eval_good_pool)
        val_good = eval_good_pool[: min(args.val_good, len(eval_good_pool))]
        test_good_pool = eval_good_pool[len(val_good) :]
        test_good = test_good_pool[: min(args.test_good, len(test_good_pool))]

        bad_pool = list(pools["bad"])
        rng.shuffle(bad_pool)
        train_bad = bad_pool[: min(args.train_bad, len(bad_pool))]
        val_bad_pool = bad_pool[len(train_bad) :]
        val_bad = val_bad_pool[: min(args.val_bad, len(val_bad_pool))]
        test_bad_pool = val_bad_pool[len(val_bad) :]
        test_bad = test_bad_pool[: min(args.test_bad, len(test_bad_pool))]
    else:
        good_pool = list(pools["good"])
        bad_pool = list(pools["bad"])
        rng.shuffle(good_pool)
        rng.shuffle(bad_pool)
        train_good = good_pool[: min(args.train_good, len(good_pool))]
        val_good = good_pool[len(train_good) : len(train_good) + min(args.val_good, max(0, len(good_pool) - len(train_good)))]
        test_good = good_pool[
            len(train_good) + len(val_good) : len(train_good) + len(val_good) + min(args.test_good, max(0, len(good_pool) - len(train_good) - len(val_good)))
        ]
        train_bad = bad_pool[: min(args.train_bad, len(bad_pool))]
        val_bad = bad_pool[len(train_bad) : len(train_bad) + min(args.val_bad, max(0, len(bad_pool) - len(train_bad)))]
        test_bad = bad_pool[
            len(train_bad) + len(val_bad) : len(train_bad) + len(val_bad) + min(args.test_bad, max(0, len(bad_pool) - len(train_bad) - len(val_bad)))
        ]

    chosen = {
        "train": {"good": train_good, "bad": train_bad},
        "val": {"good": val_good, "bad": val_bad},
        "test": {"good": test_good, "bad": test_bad},
    }

    for split in SPLITS:
        for label in LABELS:
            records.extend(
                copy_split_records(
                    chosen[split][label],
                    out_dir=split_dir,
                    split=split,
                    label=label,
                    dataset_root=args.dataset_root,
                    category=args.category,
                ),
            )

    actual_counts = {split: {label: len(chosen[split][label]) for label in LABELS} for split in SPLITS}
    manifest = {
        "dataset": "visa",
        "layout": layout,
        "dataset_root": str(args.dataset_root.resolve()),
        "category": args.category,
        "seed": args.seed,
        "requested_counts": counts,
        "actual_counts": actual_counts,
        "records": records,
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[INFO] Hybrid split prepared")
    print(f"[INFO] Layout    : {layout}")
    print(f"[INFO] Split dir : {split_dir}")
    print(f"[INFO] Counts    : {actual_counts}")
    print("=" * 80)
    return manifest


def train_patchcore_stage(args: argparse.Namespace, project_root: Path, image_size: tuple[int, int], patchcore_results_dir: Path) -> None:
    command = [
        sys.executable,
        "train_demo.py",
        "--dataset",
        "visa",
        "--dataset-root",
        str(args.dataset_root),
        "--category",
        args.category,
        "--model",
        "patchcore",
        "--results-dir",
        str(patchcore_results_dir),
        "--epochs",
        str(args.patchcore_epochs),
        "--train-batch-size",
        str(args.patchcore_train_batch_size),
        "--eval-batch-size",
        str(args.patchcore_eval_batch_size),
        "--image-height",
        str(image_size[0]),
        "--image-width",
        str(image_size[1]),
        "--num-workers",
        str(args.num_workers),
        "--tiling",
        "off",
        "--patchcore-layers",
        *args.patchcore_layers,
        "--patchcore-coreset-ratio",
        str(args.patchcore_coreset_ratio),
        "--patchcore-num-neighbors",
        str(args.patchcore_num_neighbors),
        "--patchcore-precision",
        args.patchcore_precision,
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
        "--seed",
        str(args.seed),
    ]
    run_command(command, project_root)


def generate_patchcore_maps(
    args: argparse.Namespace,
    project_root: Path,
    image_size: tuple[int, int],
    split_dir: Path,
    maps_dir: Path,
    patchcore_results_dir: Path,
) -> None:
    if args.clean and maps_dir.exists():
        shutil.rmtree(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        output_dir = maps_dir / split
        command = [
            sys.executable,
            "infer_demo.py",
            "--dataset",
            "visa",
            "--input-path",
            str(split_dir / split),
            "--model",
            "patchcore",
            "--dataset-root",
            str(args.dataset_root),
            "--category",
            args.category,
            "--output-dir",
            str(output_dir),
            "--image-height",
            str(image_size[0]),
            "--image-width",
            str(image_size[1]),
            "--tiling",
            "off",
            "--patchcore-layers",
            *args.patchcore_layers,
            "--patchcore-coreset-ratio",
            str(args.patchcore_coreset_ratio),
            "--patchcore-num-neighbors",
            str(args.patchcore_num_neighbors),
            "--patchcore-precision",
            args.patchcore_precision,
            "--roi-mode",
            "foreground",
            "--score-aggregation",
            "pixel-percentile",
            "--score-percentile",
            "99",
            "--accelerator",
            args.accelerator,
            "--devices",
            args.devices,
        ]
        if args.patchcore_checkpoint is not None:
            command.extend(["--checkpoint", str(args.patchcore_checkpoint)])
        else:
            command.extend(["--results-dir", str(patchcore_results_dir)])
        run_command(command, project_root)


@dataclass(frozen=True)
class HybridSample:
    split: str
    image_path: Path
    map_path: Path
    mask_path: Path
    label: int
    name: str


def load_hybrid_samples(output_dir: Path, split: str) -> list[HybridSample]:
    csv_path = output_dir / "predictions.csv"
    if not csv_path.exists():
        raise SystemExit(f"PatchCore predictions.csv not found for {split}: {csv_path}")

    samples: list[HybridSample] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_label = str(row.get("gt_label", "")).lower()
            label = 1 if gt_label == "bad" else 0
            image_rel = row.get("original_rel")
            map_rel = row.get("raw_float_map_rel")
            mask_rel = row.get("gt_mask_rel")
            if not image_rel or not map_rel or not mask_rel:
                continue
            image_path = output_dir / image_rel
            map_path = output_dir / map_rel
            mask_path = output_dir / mask_rel
            if not image_path.exists() or not map_path.exists() or not mask_path.exists():
                raise SystemExit(f"Incomplete PatchCore map row in {csv_path}: {row}")
            samples.append(
                HybridSample(
                    split=split,
                    image_path=image_path,
                    map_path=map_path,
                    mask_path=mask_path,
                    label=label,
                    name=str(row.get("image_name") or image_path.name),
                ),
            )
    if not samples:
        raise SystemExit(f"No usable hybrid samples found in: {csv_path}")
    return samples


def read_rgb(path: Path, image_size: tuple[int, int]) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(path).convert("RGB"))
    target_h, target_w = image_size
    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32) / 255.0


def read_mask(path: Path, image_size: tuple[int, int]) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(path).convert("L"))
    target_h, target_w = image_size
    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.float32)


def read_map(path: Path, image_size: tuple[int, int], map_low: float, map_high: float) -> Any:
    import cv2
    import numpy as np

    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    target_h, target_w = image_size
    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if map_high <= map_low:
        return np.zeros((target_h, target_w), dtype=np.float32)
    arr = (arr - map_low) / (map_high - map_low)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def compute_map_range(samples: list[HybridSample]) -> tuple[float, float]:
    import numpy as np

    lows: list[float] = []
    highs: list[float] = []
    for sample in samples:
        arr = np.load(sample.map_path).astype(np.float32).reshape(-1)
        lows.append(float(np.percentile(arr, 1.0)))
        highs.append(float(np.percentile(arr, 99.5)))
    low = float(np.percentile(lows, 5.0)) if lows else 0.0
    high = float(np.percentile(highs, 95.0)) if highs else 1.0
    if high <= low:
        high = low + 1.0
    return low, high


class HybridMapDataset:
    def __init__(
        self,
        samples: list[HybridSample],
        *,
        image_size: tuple[int, int],
        map_range: tuple[float, float],
        augment: bool,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.map_low, self.map_high = map_range
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        sample = self.samples[index]
        rgb = read_rgb(sample.image_path, self.image_size)
        anomaly = read_map(sample.map_path, self.image_size, self.map_low, self.map_high)
        mask = read_mask(sample.mask_path, self.image_size)

        if self.augment:
            if random.random() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                anomaly = np.ascontiguousarray(anomaly[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.2:
                rgb = np.ascontiguousarray(rgb[::-1])
                anomaly = np.ascontiguousarray(anomaly[::-1])
                mask = np.ascontiguousarray(mask[::-1])

        x = np.concatenate([np.transpose(rgb, (2, 0, 1)), anomaly[None, ...]], axis=0)
        return {
            "x": torch.from_numpy(x).float(),
            "mask": torch.from_numpy(mask[None, ...]).float(),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "name": sample.name,
            "image_path": str(sample.image_path),
        }


def make_dataloader(
    samples: list[HybridSample],
    *,
    image_size: tuple[int, int],
    map_range: tuple[float, float],
    batch_size: int,
    num_workers: int,
    train: bool,
    augment: bool,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = HybridMapDataset(samples, image_size=image_size, map_range=map_range, augment=augment and train)
    sampler = None
    shuffle = train
    if train:
        labels = [sample.label for sample in samples]
        good_count = max(1, labels.count(0))
        bad_count = max(1, labels.count(1))
        weights = [0.5 / bad_count if label == 1 else 0.5 / good_count for label in labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_unet(in_channels: int, base_channels: int) -> Any:
    import torch
    from torch import nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            self.enc1 = ConvBlock(in_channels, c)
            self.enc2 = ConvBlock(c, c * 2)
            self.enc3 = ConvBlock(c * 2, c * 4)
            self.enc4 = ConvBlock(c * 4, c * 8)
            self.pool = nn.MaxPool2d(2)
            self.mid = ConvBlock(c * 8, c * 16)
            self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
            self.dec4 = ConvBlock(c * 16, c * 8)
            self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3 = ConvBlock(c * 8, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = ConvBlock(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = ConvBlock(c * 2, c)
            self.out = nn.Conv2d(c, 1, 1)

        @staticmethod
        def cat_skip(x: Any, skip: Any) -> Any:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([x, skip], dim=1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            mid = self.mid(self.pool(e4))
            d4 = self.dec4(self.cat_skip(self.up4(mid), e4))
            d3 = self.dec3(self.cat_skip(self.up3(d4), e3))
            d2 = self.dec2(self.cat_skip(self.up2(d3), e2))
            d1 = self.dec1(self.cat_skip(self.up1(d2), e1))
            return self.out(d1)

    return SmallUNet()


def dice_loss_from_logits(logits: Any, targets: Any, eps: float = 1e-6) -> Any:
    import torch

    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    denominator = torch.sum(probs, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def estimate_pos_weight(samples: list[HybridSample], image_size: tuple[int, int], max_pos_weight: float) -> float:
    pos = 0.0
    total = 0.0
    for sample in samples:
        mask = read_mask(sample.mask_path, image_size)
        pos += float(mask.sum())
        total += float(mask.size)
    if pos <= 0.0:
        return 1.0
    neg = max(0.0, total - pos)
    return float(min(max_pos_weight, max(1.0, neg / pos)))


def largest_component_fraction(binary: Any) -> float:
    import cv2
    import numpy as np

    binary = binary.astype(np.uint8)
    if binary.max() == 0:
        return 0.0
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if len(stats) <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(largest) / float(binary.size)


def image_score_from_prob(prob: Any, mask_threshold: float, mode: str, area_weight: float, blob_weight: float) -> tuple[float, float, float]:
    import numpy as np

    prob = np.asarray(prob, dtype=np.float32)
    binary = prob >= mask_threshold
    area_fraction = float(binary.mean())
    blob_fraction = largest_component_fraction(binary)
    max_score = float(prob.max())
    p99 = float(np.percentile(prob, 99.0))
    if mode == "max":
        score = max_score
    elif mode == "p99":
        score = p99
    elif mode == "area":
        score = area_fraction
    elif mode == "blob":
        score = blob_fraction
    else:
        score = p99 + area_weight * area_fraction + blob_weight * blob_fraction
    return float(score), area_fraction, blob_fraction


def rank_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def threshold_candidates(scores: list[float]) -> list[float]:
    unique = sorted(set(float(score) for score in scores))
    if not unique:
        return [0.5]
    candidates = set(unique)
    for low, high in zip(unique, unique[1:]):
        candidates.add((low + high) / 2.0)
    span = max(unique) - min(unique)
    eps = max(1e-6, span * 1e-6)
    candidates.add(min(unique) - eps)
    candidates.add(max(unique) + eps)
    return sorted(candidates)


def image_metrics(labels: list[int], scores: list[float], threshold: float | None = None) -> dict[str, Any]:
    if threshold is None:
        best: dict[str, Any] | None = None
        for candidate in threshold_candidates(scores):
            metrics = image_metrics(labels, scores, candidate)
            score = (metrics["f1"], metrics["balanced_accuracy"], metrics["accuracy"])
            if best is None or score > (best["f1"], best["balanced_accuracy"], best["accuracy"]):
                best = metrics
        return best or image_metrics(labels, scores, 0.5)

    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(f1),
        "auc": rank_auc(labels, scores),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def pixel_stats(pred: Any, target: Any) -> tuple[int, int, int]:
    import numpy as np

    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    return tp, fp, fn


def safe_dice(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 1.0


def safe_iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom else 1.0


def evaluate_model(
    model: Any,
    loader: Any,
    *,
    device: Any,
    mask_threshold: float,
    image_threshold: float | None,
    image_score_mode: str,
    image_area_weight: float,
    image_blob_weight: float,
    save_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    rows: list[dict[str, Any]] = []
    global_tp = global_fp = global_fn = 0
    per_image_dice: list[float] = []
    per_image_iou: list[float] = []

    if save_dir is not None:
        (save_dir / "pred_masks").mkdir(parents=True, exist_ok=True)
        (save_dir / "pred_overlays").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            probs = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
            targets = target.cpu().numpy()[:, 0]
            batch_labels = batch["label"].cpu().numpy().astype(int).tolist()
            names = list(batch["name"])
            image_paths = list(batch["image_path"])

            for idx, prob in enumerate(probs):
                pred_mask = prob >= mask_threshold
                gt_mask = targets[idx] >= 0.5
                tp, fp, fn = pixel_stats(pred_mask, gt_mask)
                global_tp += tp
                global_fp += fp
                global_fn += fn
                dice = safe_dice(tp, fp, fn)
                iou = safe_iou(tp, fp, fn)
                score, area_fraction, blob_fraction = image_score_from_prob(
                    prob,
                    mask_threshold,
                    image_score_mode,
                    image_area_weight,
                    image_blob_weight,
                )
                label = int(batch_labels[idx])
                pred_label = "" if image_threshold is None else int(score >= image_threshold)
                labels.append(label)
                scores.append(score)
                per_image_dice.append(dice)
                per_image_iou.append(iou)

                pred_mask_rel = ""
                overlay_rel = ""
                if save_dir is not None:
                    stem = f"{len(rows):04d}_{Path(names[idx]).stem}"
                    mask_path = save_dir / "pred_masks" / f"{stem}.png"
                    overlay_path = save_dir / "pred_overlays" / f"{stem}.png"
                    Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(mask_path)
                    rgb = np.array(Image.open(image_paths[idx]).convert("RGB"))
                    if rgb.shape[:2] != pred_mask.shape:
                        rgb = cv2.resize(rgb, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_AREA)
                    overlay = rgb.copy().astype(np.float32)
                    overlay[pred_mask] = overlay[pred_mask] * 0.35 + np.array([255, 40, 40], dtype=np.float32) * 0.65
                    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)
                    pred_mask_rel = mask_path.relative_to(save_dir).as_posix()
                    overlay_rel = overlay_path.relative_to(save_dir).as_posix()

                rows.append(
                    {
                        "image_name": names[idx],
                        "gt_label": "bad" if label else "good",
                        "pred_label": pred_label,
                        "correct": "" if pred_label == "" else bool(pred_label == label),
                        "image_score": score,
                        "mask_threshold": mask_threshold,
                        "image_threshold": "" if image_threshold is None else image_threshold,
                        "pred_area_fraction": area_fraction,
                        "largest_blob_fraction": blob_fraction,
                        "pixel_dice": dice,
                        "pixel_iou": iou,
                        "pred_mask_rel": pred_mask_rel,
                        "pred_overlay_rel": overlay_rel,
                    },
                )

    image_report = image_metrics(labels, scores, image_threshold) if labels else {}
    report = {
        "num_samples": len(labels),
        "mask_threshold": mask_threshold,
        "image_threshold": image_threshold,
        "pixel_dice_global": safe_dice(global_tp, global_fp, global_fn),
        "pixel_iou_global": safe_iou(global_tp, global_fp, global_fn),
        "pixel_dice_mean_image": float(np.mean(per_image_dice)) if per_image_dice else None,
        "pixel_iou_mean_image": float(np.mean(per_image_iou)) if per_image_iou else None,
        "pixel_tp": global_tp,
        "pixel_fp": global_fp,
        "pixel_fn": global_fn,
        "image": image_report,
    }
    return report, rows


def train_one_epoch(
    model: Any,
    loader: Any,
    *,
    optimizer: Any,
    scaler: Any,
    device: Any,
    bce_loss: Any,
    bce_weight: float,
    dice_weight: float,
    use_amp: bool,
) -> float:
    from contextlib import nullcontext

    import torch

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.amp.autocast("cuda") if use_amp and hasattr(torch, "amp") else nullcontext()
        with autocast_context:
            logits = model(x)
            loss = bce_weight * bce_loss(logits, target) + dice_weight * dice_loss_from_logits(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * x.shape[0]
        total_samples += int(x.shape[0])
    return total_loss / max(1, total_samples)


def choose_device(accelerator: str) -> Any:
    import torch

    if accelerator in {"gpu", "cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_hybrid_model(args: argparse.Namespace, image_size: tuple[int, int], maps_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_samples = load_hybrid_samples(maps_dir / "train", "train")
    val_samples = load_hybrid_samples(maps_dir / "val", "val")
    test_samples = load_hybrid_samples(maps_dir / "test", "test")
    map_range = compute_map_range(train_samples)
    device = choose_device(args.accelerator)
    use_amp = bool(device.type == "cuda" and not args.no_amp)

    train_loader = make_dataloader(
        train_samples,
        image_size=image_size,
        map_range=map_range,
        batch_size=args.hybrid_batch_size,
        num_workers=args.num_workers,
        train=True,
        augment=not args.no_augment,
    )
    train_eval_loader = make_dataloader(
        train_samples,
        image_size=image_size,
        map_range=map_range,
        batch_size=args.hybrid_batch_size,
        num_workers=args.num_workers,
        train=False,
        augment=False,
    )
    val_loader = make_dataloader(
        val_samples,
        image_size=image_size,
        map_range=map_range,
        batch_size=args.hybrid_batch_size,
        num_workers=args.num_workers,
        train=False,
        augment=False,
    )
    test_loader = make_dataloader(
        test_samples,
        image_size=image_size,
        map_range=map_range,
        batch_size=args.hybrid_batch_size,
        num_workers=args.num_workers,
        train=False,
        augment=False,
    )

    model = build_unet(in_channels=4, base_channels=args.base_channels).to(device)
    pos_weight = estimate_pos_weight(train_samples, image_size, args.max_pos_weight)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.hybrid_epochs))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -math.inf
    best_path = output_dir / "hybrid_unet_best.pt"
    history: list[dict[str, Any]] = []

    print("=" * 80)
    print("[INFO] Hybrid U-Net training")
    print(f"[INFO] Device       : {device}")
    print(f"[INFO] AMP          : {use_amp}")
    print(f"[INFO] Image size   : {image_size[0]}x{image_size[1]}")
    print(f"[INFO] Map range    : low={map_range[0]:.6f}, high={map_range[1]:.6f}")
    print(f"[INFO] Train samples: {len(train_samples)}")
    print(f"[INFO] Val samples  : {len(val_samples)}")
    print(f"[INFO] Test samples : {len(test_samples)}")
    print(f"[INFO] Pos weight   : {pos_weight:.3f}")
    print("=" * 80)

    for epoch in range(1, args.hybrid_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            bce_loss=bce_loss,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            use_amp=use_amp,
        )
        scheduler.step()
        val_report, _ = evaluate_model(
            model,
            val_loader,
            device=device,
            mask_threshold=0.5,
            image_threshold=None,
            image_score_mode=args.image_score_mode,
            image_area_weight=args.image_area_weight,
            image_blob_weight=args.image_blob_weight,
        )
        val_image = val_report["image"]
        val_score = float(val_report["pixel_dice_global"]) + float(val_image["f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pixel_dice": val_report["pixel_dice_global"],
            "val_pixel_iou": val_report["pixel_iou_global"],
            "val_image_f1": val_image["f1"],
            "val_image_acc": val_image["accuracy"],
            "val_image_auc": val_image["auc"],
        }
        history.append(row)
        print(
            f"[EPOCH {epoch:03d}] loss={train_loss:.4f} "
            f"val_dice={float(row['val_pixel_dice']):.4f} "
            f"val_img_f1={float(row['val_image_f1']):.4f} "
            f"val_acc={float(row['val_image_acc']):.4f}",
            flush=True,
        )
        if val_score > best_score:
            best_score = val_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": image_size,
                    "map_range": map_range,
                    "base_channels": args.base_channels,
                    "pos_weight": pos_weight,
                    "args": vars(args),
                },
                best_path,
            )

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_mask_threshold = 0.5
    best_mask_score = -math.inf
    for candidate in [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]:
        report, _ = evaluate_model(
            model,
            val_loader,
            device=device,
            mask_threshold=float(candidate),
            image_threshold=None,
            image_score_mode=args.image_score_mode,
            image_area_weight=args.image_area_weight,
            image_blob_weight=args.image_blob_weight,
        )
        score = float(report["pixel_dice_global"])
        if score > best_mask_score:
            best_mask_score = score
            best_mask_threshold = float(candidate)

    val_report_for_threshold, _ = evaluate_model(
        model,
        val_loader,
        device=device,
        mask_threshold=best_mask_threshold,
        image_threshold=None,
        image_score_mode=args.image_score_mode,
        image_area_weight=args.image_area_weight,
        image_blob_weight=args.image_blob_weight,
    )
    image_threshold = float(val_report_for_threshold["image"]["threshold"])

    train_report, train_rows = evaluate_model(
        model,
        train_eval_loader,
        device=device,
        mask_threshold=best_mask_threshold,
        image_threshold=image_threshold,
        image_score_mode=args.image_score_mode,
        image_area_weight=args.image_area_weight,
        image_blob_weight=args.image_blob_weight,
    )
    val_report, val_rows = evaluate_model(
        model,
        val_loader,
        device=device,
        mask_threshold=best_mask_threshold,
        image_threshold=image_threshold,
        image_score_mode=args.image_score_mode,
        image_area_weight=args.image_area_weight,
        image_blob_weight=args.image_blob_weight,
    )
    test_save_dir = output_dir / "test_visuals" if args.save_test_maps else None
    test_report, test_rows = evaluate_model(
        model,
        test_loader,
        device=device,
        mask_threshold=best_mask_threshold,
        image_threshold=image_threshold,
        image_score_mode=args.image_score_mode,
        image_area_weight=args.image_area_weight,
        image_blob_weight=args.image_blob_weight,
        save_dir=test_save_dir,
    )

    write_rows_csv(output_dir / "history.csv", history)
    write_rows_csv(output_dir / "train_predictions.csv", train_rows)
    write_rows_csv(output_dir / "val_predictions.csv", val_rows)
    write_rows_csv(output_dir / "test_predictions.csv", test_rows)

    summary = {
        "category": args.category,
        "image_size": f"{image_size[0]}x{image_size[1]}",
        "map_range": {"low": map_range[0], "high": map_range[1]},
        "model": "hybrid_patchcore_unet",
        "checkpoint": str(best_path),
        "best_mask_threshold": best_mask_threshold,
        "image_threshold": image_threshold,
        "image_score_mode": args.image_score_mode,
        "train": train_report,
        "val": val_report,
        "test": test_report,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[INFO] Hybrid training done")
    print(f"[INFO] Best checkpoint : {best_path}")
    print(f"[INFO] Mask threshold  : {best_mask_threshold:.3f}")
    print(f"[INFO] Image threshold : {image_threshold:.6f}")
    print(f"[INFO] Test image acc  : {test_report['image']['accuracy']:.4f}")
    print(f"[INFO] Test image F1   : {test_report['image']['f1']:.4f}")
    print(f"[INFO] Test image AUC  : {test_report['image']['auc']}")
    print(f"[INFO] Test pixel Dice : {test_report['pixel_dice_global']:.4f}")
    print(f"[INFO] Test pixel IoU  : {test_report['pixel_iou_global']:.4f}")
    print(f"[INFO] Summary JSON    : {output_dir / 'summary.json'}")
    print("=" * 80)
    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    args.dataset_root = project_path(args.dataset_root, project_root)
    image_size = resolve_image_size(args)
    split_dir = project_path(args.split_dir or Path(f"./hybrid_inputs_visa_{args.category}"), project_root)
    maps_dir = project_path(args.maps_dir or Path(f"./hybrid_maps_visa_{args.category}_patchcore"), project_root)
    output_dir = project_path(args.output_dir or Path(f"./hybrid_outputs_visa_{args.category}"), project_root)
    patchcore_results_dir = project_path(args.patchcore_results_dir, project_root)
    if args.patchcore_checkpoint is not None:
        args.patchcore_checkpoint = project_path(args.patchcore_checkpoint, project_root)

    if args.train_bad <= 0 or args.val_bad <= 0 or args.test_bad <= 0:
        raise SystemExit("Hybrid supervised training needs bad samples in train/val/test.")
    if args.val_good <= 0 or args.test_good <= 0:
        raise SystemExit("Hybrid threshold calibration/evaluation needs good samples in val/test.")
    if args.hybrid_epochs <= 0:
        raise SystemExit("--hybrid-epochs must be positive.")
    if args.hybrid_batch_size <= 0:
        raise SystemExit("--hybrid-batch-size must be positive.")
    if args.base_channels <= 0:
        raise SystemExit("--base-channels must be positive.")

    if not args.skip_prepare:
        prepare_hybrid_split(args, split_dir)
    if args.train_patchcore:
        train_patchcore_stage(args, project_root, image_size, patchcore_results_dir)
    if not args.skip_map_generation:
        generate_patchcore_maps(args, project_root, image_size, split_dir, maps_dir, patchcore_results_dir)
    if not args.skip_training:
        train_hybrid_model(args, image_size, maps_dir, output_dir)


if __name__ == "__main__":
    main()
