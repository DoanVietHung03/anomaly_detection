#!/usr/bin/env python3
"""Train an anomaly detection demo on VisA or MVTec AD 2.

This script focuses on a minimal, reproducible training flow for PatchCore and EfficientAD.
It trains on defect-free data and evaluates on the public test split.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from anomalib.data import MVTecAD2 as _MVTecAD2Base
except Exception:  # pragma: no cover - import_dependencies handles the runtime error message.
    _MVTecAD2Base = object


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_VARIANTS = ("regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3")
VARIANT_CHOICES = ("all", *IMAGE_VARIANTS)
DATASET_CHOICES = ("mvtec_ad2", "visa")
MODEL_CHOICES = ("patchcore", "efficientad")
DEFAULT_IMAGE_SIZE = (384, 384)
DEFAULT_TILING = "off"
DEFAULT_PATCHCORE_LAYERS = ("layer2", "layer3")
DEFAULT_PATCHCORE_CORESET_RATIO = 0.05
DEFAULT_PATCHCORE_NUM_NEIGHBORS = 9
DEFAULT_PATCHCORE_PRECISION = "float16"
DEFAULT_EFFICIENTAD_IMAGENET_DIR = Path("./datasets/imagenette")
DEFAULT_EFFICIENTAD_MODEL_SIZE = "small"
DEFAULT_EFFICIENTAD_TEACHER_OUT_CHANNELS = 384
DEFAULT_EFFICIENTAD_LR = 1e-4
DEFAULT_EFFICIENTAD_WEIGHT_DECAY = 1e-5
DEFAULT_EFFICIENTAD_MAX_STEPS = 70_000


class SafeMVTecAD2(_MVTecAD2Base):
    def __init__(self, *args: Any, test_variants: Iterable[str] | None = None, **kwargs: Any) -> None:
        self.test_variants = normalize_variant_selection(test_variants)
        super().__init__(*args, **kwargs)

    def _setup(self, *args: Any, **kwargs: Any) -> None:
        super()._setup(*args, **kwargs)
        for dataset_name in (
            "train_data",
            "val_data",
            "test_data",
            "test_public_data",
            "test_private_data",
            "test_private_mixed_data",
        ):
            normalize_mask_paths(getattr(self, dataset_name, None))
        for dataset_name in (
            "test_data",
            "test_public_data",
            "test_private_data",
            "test_private_mixed_data",
        ):
            filter_dataset_variants(getattr(self, dataset_name, None), self.test_variants)


if __name__ == "__main__":
    sys.modules.setdefault("train_demo", sys.modules[__name__])
SafeMVTecAD2.__module__ = "train_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VisA/MVTec demo anomaly detector.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="visa",
        help="Dataset format. Use visa for the VisA dataset.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("./VisA"), help="Path to VisA or MVTec_AD_2 root.")
    parser.add_argument("--category", type=str, default="candle", help="Dataset category.")
    parser.add_argument(
        "--model",
        type=str,
        default="patchcore",
        choices=MODEL_CHOICES,
        help="Model to train.",
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Override train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Eval/test batch size.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square image size for model preprocessing. Overridden by --image-height/--image-width.",
    )
    parser.add_argument("--image-height", type=int, default=None, help="Model input height. Defaults to 384.")
    parser.add_argument("--image-width", type=int, default=None, help="Model input width. Defaults to 384.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument(
        "--tiling",
        choices=["auto", "on", "off"],
        default=DEFAULT_TILING,
        help="Enable tiled PatchCore processing. off is the default and EfficientAD does not use tiling.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="PatchCore tile size when tiling is enabled.")
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=None,
        help="PatchCore tile stride. Defaults to half of --tile-size.",
    )
    parser.add_argument(
        "--patchcore-layers",
        nargs="+",
        default=list(DEFAULT_PATCHCORE_LAYERS),
        choices=["layer1", "layer2", "layer3", "layer4"],
        help="PatchCore feature layers. layer2+layer3 balances fine texture and object context.",
    )
    parser.add_argument(
        "--patchcore-coreset-ratio",
        type=float,
        default=DEFAULT_PATCHCORE_CORESET_RATIO,
        help="PatchCore coreset sampling ratio. Increase to 0.1-0.15 only if GPU memory allows.",
    )
    parser.add_argument(
        "--patchcore-num-neighbors",
        type=int,
        default=DEFAULT_PATCHCORE_NUM_NEIGHBORS,
        help="PatchCore nearest-neighbor count.",
    )
    parser.add_argument(
        "--patchcore-precision",
        choices=["float16", "float32"],
        default=DEFAULT_PATCHCORE_PRECISION,
        help="PatchCore compute precision. float16 is the memory-safe default; use float32 only if memory allows.",
    )
    parser.add_argument(
        "--efficientad-imagenet-dir",
        type=Path,
        default=DEFAULT_EFFICIENTAD_IMAGENET_DIR,
        help="ImageNet/Imagenette-style folder used by EfficientAD's penalty branch.",
    )
    parser.add_argument(
        "--efficientad-model-size",
        choices=["small", "medium"],
        default=DEFAULT_EFFICIENTAD_MODEL_SIZE,
        help="EfficientAD student/teacher size.",
    )
    parser.add_argument(
        "--efficientad-teacher-out-channels",
        type=int,
        default=DEFAULT_EFFICIENTAD_TEACHER_OUT_CHANNELS,
        help="EfficientAD teacher output channels.",
    )
    parser.add_argument("--efficientad-lr", type=float, default=DEFAULT_EFFICIENTAD_LR, help="EfficientAD learning rate.")
    parser.add_argument(
        "--efficientad-weight-decay",
        type=float,
        default=DEFAULT_EFFICIENTAD_WEIGHT_DECAY,
        help="EfficientAD optimizer weight decay.",
    )
    parser.add_argument("--efficientad-padding", action="store_true", help="Enable padding in EfficientAD convolutions.")
    parser.add_argument(
        "--efficientad-no-pad-maps",
        action="store_true",
        help="Do not pad EfficientAD anomaly maps when convolution padding is disabled.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional Lightning max_steps. EfficientAD defaults to 70000 when omitted; when set, it overrides the epoch limit.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default="public",
        choices=["public", "private", "private_mixed"],
        help="Which MVTec AD 2 test split to use.",
    )
    parser.add_argument(
        "--test-variant",
        nargs="+",
        default=["regular"],
        choices=VARIANT_CHOICES,
        help="Filter evaluation images by capture variant. Use all for the full public split.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Lightning accelerator, e.g. auto/cpu/gpu.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Lightning devices value, e.g. auto/1/[0].",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clean", action="store_true", help="Remove --results-dir before training.")
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from anomalib.callbacks import TilerConfigurationCallback
        from anomalib.data import MVTecAD2, Visa
        from anomalib.data.utils.tiler import ImageUpscaleMode
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd, Patchcore
        from lightning.pytorch import seed_everything
        from lightning.pytorch.loggers import CSVLogger
    except Exception as exc:  # pragma: no cover - runtime safeguard
        raise SystemExit(
            "Failed to import Anomalib stack. Install dependencies first, for example:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc
    return (MVTecAD2, Visa), Engine, (Patchcore, EfficientAd), seed_everything, CSVLogger, TilerConfigurationCallback, ImageUpscaleMode


def resolve_image_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise SystemExit("Pass both --image-height and --image-width, or neither.")
        image_size = (args.image_height, args.image_width)
    elif args.image_size is not None:
        image_size = (args.image_size, args.image_size)
    else:
        image_size = DEFAULT_IMAGE_SIZE

    if image_size[0] <= 0 or image_size[1] <= 0:
        raise SystemExit("Image height and width must be positive integers.")
    return image_size


def format_image_size(image_size: tuple[int, int]) -> str:
    return f"{image_size[0]}x{image_size[1]}"


def should_enable_tiling(model_name: str, tiling: str) -> bool:
    if tiling == "off":
        return False
    if tiling == "on":
        return True
    return model_name == "patchcore"


def resolve_tiling(
    model_name: str,
    tiling: str,
    tile_size: int,
    tile_stride: int | None,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    enabled = should_enable_tiling(model_name, tiling)
    if not enabled:
        return {"enabled": False, "tile_size": None, "tile_stride": None}
    if model_name != "patchcore":
        raise SystemExit("Tiling is only supported for PatchCore in this demo.")
    if tile_size <= 0:
        raise SystemExit("--tile-size must be a positive integer.")

    max_tile = min(image_size)
    effective_tile_size = min(tile_size, max_tile)
    if effective_tile_size != tile_size:
        print(
            f"[WARN] Requested tile size {tile_size} exceeds image size {format_image_size(image_size)}; "
            f"using {effective_tile_size}.",
        )

    stride = tile_stride if tile_stride is not None else max(1, effective_tile_size // 2)
    if stride <= 0:
        raise SystemExit("--tile-stride must be a positive integer.")
    if stride > effective_tile_size:
        print(f"[WARN] Requested tile stride {stride} exceeds tile size {effective_tile_size}; using {effective_tile_size}.")
        stride = effective_tile_size

    return {"enabled": True, "tile_size": effective_tile_size, "tile_stride": stride}


def build_tiling_callbacks(
    tiling_config: dict[str, Any],
    tiler_callback_cls: Any,
    upscale_mode_cls: Any,
) -> list[Any]:
    if not tiling_config["enabled"]:
        return []
    return [
        tiler_callback_cls(
            enable=True,
            tile_size=int(tiling_config["tile_size"]),
            stride=int(tiling_config["tile_stride"]),
            mode=upscale_mode_cls.PADDING,
        ),
    ]


def validate_patchcore_args(args: argparse.Namespace) -> tuple[str, ...]:
    layers = tuple(dict.fromkeys(args.patchcore_layers))
    if not layers:
        raise SystemExit("--patchcore-layers must include at least one layer.")
    if not (0.0 < args.patchcore_coreset_ratio <= 1.0):
        raise SystemExit("--patchcore-coreset-ratio must be in the range (0, 1].")
    if args.patchcore_num_neighbors <= 0:
        raise SystemExit("--patchcore-num-neighbors must be a positive integer.")
    return layers


def validate_efficientad_args(args: argparse.Namespace) -> None:
    if args.efficientad_teacher_out_channels <= 0:
        raise SystemExit("--efficientad-teacher-out-channels must be a positive integer.")
    if args.efficientad_lr <= 0.0:
        raise SystemExit("--efficientad-lr must be positive.")
    if args.efficientad_weight_decay < 0.0:
        raise SystemExit("--efficientad-weight-decay must be non-negative.")
    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None and max_steps <= 0:
        raise SystemExit("--max-steps must be a positive integer.")


def build_model_from_args(
    args: argparse.Namespace,
    patchcore_cls: Any,
    efficientad_cls: Any,
    image_size: tuple[int, int],
) -> Any:
    pre_processor_size = image_size
    if args.model == "patchcore":
        layers = validate_patchcore_args(args)
        return patchcore_cls(
            backbone="wide_resnet50_2",
            layers=layers,
            coreset_sampling_ratio=args.patchcore_coreset_ratio,
            num_neighbors=args.patchcore_num_neighbors,
            precision=args.patchcore_precision,
            pre_processor=patchcore_cls.configure_pre_processor(image_size=pre_processor_size),
        )
    if args.model == "efficientad":
        validate_efficientad_args(args)
        return efficientad_cls(
            imagenet_dir=args.efficientad_imagenet_dir,
            teacher_out_channels=args.efficientad_teacher_out_channels,
            model_size=args.efficientad_model_size,
            lr=args.efficientad_lr,
            weight_decay=args.efficientad_weight_decay,
            padding=args.efficientad_padding,
            pad_maps=not args.efficientad_no_pad_maps,
            pre_processor=efficientad_cls.configure_pre_processor(image_size=pre_processor_size),
        )
    raise ValueError(f"Unsupported model: {args.model}")


def normalize_variant_selection(variants: Iterable[str] | None) -> set[str] | None:
    if variants is None:
        return None
    selected = {str(variant).lower() for variant in variants}
    if not selected or "all" in selected:
        return None
    unknown = selected.difference(IMAGE_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown image variant(s): {', '.join(sorted(unknown))}")
    return selected


def format_variant_selection(variants: Iterable[str] | None) -> str:
    selected = normalize_variant_selection(variants)
    if selected is None:
        return "all"
    return ",".join(sorted(selected))


def image_variant(path: Any) -> str:
    stem = Path(str(path)).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def filter_dataset_variants(dataset: Any, variants: set[str] | None) -> None:
    if variants is None:
        return

    samples = getattr(dataset, "samples", None)
    if samples is None or "image_path" not in samples.columns:
        return

    filtered = samples[samples["image_path"].map(image_variant).isin(variants)].copy()
    dataset.samples = filtered.reset_index(drop=True)


def normalize_mask_paths(dataset: Any) -> None:
    """Convert pandas NaN mask paths from MVTecAD2 into real None values."""
    samples = getattr(dataset, "samples", None)
    if samples is None or "mask_path" not in samples.columns:
        return

    import pandas as pd

    samples = samples.copy()
    samples["mask_path"] = samples["mask_path"].astype("object").where(pd.notna(samples["mask_path"]), None)
    dataset.samples = samples


def resolve_category_root(dataset_root: Path, category: str) -> Path:
    category_root = dataset_root / category
    return category_root if category_root.exists() else dataset_root


def list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and not p.stem.lower().endswith("_mask")
    )


def count_variants(folder: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in list_image_files(folder):
        variant = image_variant(path)
        counts[variant] = counts.get(variant, 0) + 1
    return dict(sorted(counts.items()))


def build_dataset_variant_summary(dataset_root: Path, category: str) -> dict[str, dict[str, int]]:
    category_root = resolve_category_root(dataset_root, category)
    return {
        "train_good": count_variants(category_root / "train" / "good"),
        "validation_good": count_variants(category_root / "validation" / "good"),
        "validation_bad": count_variants(category_root / "validation" / "bad"),
        "test_public_good": count_variants(category_root / "test_public" / "good"),
        "test_public_bad": count_variants(category_root / "test_public" / "bad"),
    }


def build_visa_dataset_summary(dataset_root: Path, category: str) -> dict[str, dict[str, int]]:
    raw_root = dataset_root / category / "Data"
    split_root = dataset_root / "visa_pytorch" / category
    return {
        "raw_normal": {"count": len(list_image_files(raw_root / "Images" / "Normal"))},
        "raw_anomaly": {"count": len(list_image_files(raw_root / "Images" / "Anomaly"))},
        "raw_masks": {"count": len(list_image_files(raw_root / "Masks" / "Anomaly"))},
        "split_train_good": {"count": len(list_image_files(split_root / "train" / "good"))},
        "split_test_good": {"count": len(list_image_files(split_root / "test" / "good"))},
        "split_test_bad": {"count": len(list_image_files(split_root / "test" / "bad"))},
    }


def print_domain_shift_warning(summary: dict[str, dict[str, int]], selected_variants: set[str] | None) -> None:
    train_variants = set(summary.get("train_good", {}))
    test_good_variants = set(summary.get("test_public_good", {}))
    evaluated_variants = test_good_variants if selected_variants is None else selected_variants
    unseen_variants = sorted(evaluated_variants.difference(train_variants))
    if unseen_variants:
        print(
            "[WARN] Evaluation includes normal-image variants not present in train/good: "
            + ", ".join(unseen_variants)
        )
        print("[WARN] One-class models may flag these capture shifts as anomalies. Use --test-variant regular for a same-condition check.")


def print_threshold_warning(summary: dict[str, dict[str, int]]) -> None:
    if not summary.get("validation_bad"):
        print("[WARN] validation/bad is empty. Anomalib's adaptive threshold may classify everything as good.")
        print("[WARN] Use infer_demo.py --calibration-path with enough good and bad images for calibrated labels.")


def print_patchcore_profile_warnings(args: argparse.Namespace, image_size: tuple[int, int], tiling_config: dict[str, Any]) -> None:
    layers = set(validate_patchcore_args(args))
    if "layer3" not in layers:
        print("[WARN] PatchCore is running without layer3; image-level scores may miss broader object context.")
    if args.patchcore_coreset_ratio < 0.05:
        print("[WARN] PatchCore coreset ratio is below 0.05; normal edge cases may be under-represented.")
    if args.dataset != "visa" and image_size[0] == image_size[1]:
        print("[WARN] Square PatchCore input can distort wide images. Prefer an aspect-preserving size for non-VisA datasets.")
    if tiling_config["enabled"] and max(image_size) / min(image_size) >= 1.5:
        print("[WARN] Tiling on wide images creates multiple overlapping tiles and can exhaust GPU memory.")


def make_safe_mvtec_ad2_cls(mvtec_ad2_cls: Any) -> Any:
    if not issubclass(SafeMVTecAD2, mvtec_ad2_cls):
        raise SystemExit("SafeMVTecAD2 base class mismatch. Restart the Python process and try again.")
    return SafeMVTecAD2


def find_best_checkpoint(results_dir: Path) -> str | None:
    checkpoint_candidates = sorted(results_dir.rglob("*.ckpt"))
    if not checkpoint_candidates:
        return None

    preferred = [p for p in checkpoint_candidates if "best" in p.name.lower() or "model" in p.name.lower()]
    if preferred:
        return str(preferred[0].resolve())
    return str(checkpoint_candidates[0].resolve())


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    return str(obj)


def clean_results_dir(results_dir: Path, dataset_root: Path) -> None:
    target = results_dir.resolve()
    cwd = Path.cwd().resolve()
    dataset = dataset_root.resolve()
    if target == cwd:
        raise SystemExit("--clean refuses to remove the current working directory. Use a dedicated --results-dir.")
    if target == dataset:
        raise SystemExit("--clean refuses to remove --dataset-root. Use a dedicated --results-dir.")
    if target.exists():
        shutil.rmtree(target)


def main() -> None:
    args = parse_args()
    (
        datamodule_classes,
        engine_cls,
        model_classes,
        seed_everything,
        csv_logger_cls,
        tiler_callback_cls,
        upscale_mode_cls,
    ) = import_dependencies()
    mvtec_ad2_cls, visa_cls = datamodule_classes
    patchcore_cls, efficientad_cls = model_classes
    safe_mvtec_ad2_cls = make_safe_mvtec_ad2_cls(mvtec_ad2_cls)

    if args.results_dir is None:
        args.results_dir = Path(f"./runs_{args.model}_{args.dataset}_{args.category}")
    if args.clean:
        clean_results_dir(args.results_dir, args.dataset_root)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, workers=True)
    image_size = resolve_image_size(args)
    if args.model == "efficientad" and args.max_steps is None:
        args.max_steps = DEFAULT_EFFICIENTAD_MAX_STEPS
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be a positive integer.")
    tiling_config = resolve_tiling(args.model, args.tiling, args.tile_size, args.tile_stride, image_size)
    selected_test_variants = normalize_variant_selection(args.test_variant)
    if args.dataset == "visa":
        dataset_variant_summary = build_visa_dataset_summary(args.dataset_root, args.category)
    else:
        dataset_variant_summary = build_dataset_variant_summary(args.dataset_root, args.category)

    train_batch_size = args.train_batch_size
    if train_batch_size is None:
        train_batch_size = 1
    if args.model == "efficientad" and train_batch_size != 1:
        raise SystemExit("EfficientAD requires --train-batch-size 1.")

    if args.dataset == "visa":
        datamodule = visa_cls(
            root=args.dataset_root,
            category=args.category,
            train_batch_size=train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    else:
        datamodule = safe_mvtec_ad2_cls(
            root=args.dataset_root,
            category=args.category,
            train_batch_size=train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            test_type=args.test_type,
            test_variants=args.test_variant,
            seed=args.seed,
        )

    model = build_model_from_args(args, patchcore_cls, efficientad_cls, image_size)
    csv_logger = csv_logger_cls(
        save_dir=str(args.results_dir / "logs"),
        name=f"{args.model}_{args.category}",
    )
    callbacks = build_tiling_callbacks(tiling_config, tiler_callback_cls, upscale_mode_cls)

    precision_flag = "16-true" if args.model == "patchcore" and args.patchcore_precision == "float16" else "32"
    engine_kwargs = {
        "callbacks": callbacks,
        "default_root_dir": str(args.results_dir),
        "max_epochs": -1 if args.max_steps is not None else args.epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "logger": csv_logger,
        "precision": precision_flag,
    }
    if args.max_steps is not None:
        engine_kwargs["max_steps"] = args.max_steps

    engine = engine_cls(**engine_kwargs)

    print("=" * 80)
    print("[INFO] Starting training")
    print(f"[INFO] dataset     : {args.dataset}")
    print(f"[INFO] dataset_root: {args.dataset_root}")
    print(f"[INFO] category    : {args.category}")
    print(f"[INFO] model       : {args.model}")
    print(f"[INFO] image_size  : {format_image_size(image_size)}")
    print(f"[INFO] tiling      : {tiling_config}")
    if args.max_steps is not None:
        print(f"[INFO] max_steps   : {args.max_steps}")
    if args.model == "patchcore":
        print(f"[INFO] layers      : {','.join(validate_patchcore_args(args))}")
        print(f"[INFO] coreset     : {args.patchcore_coreset_ratio}")
        print(f"[INFO] precision   : {args.patchcore_precision}")
    else:
        print(f"[INFO] model_size  : {args.efficientad_model_size}")
        print(f"[INFO] imagenet_dir: {args.efficientad_imagenet_dir}")
        print(f"[INFO] lr          : {args.efficientad_lr}")
        print(f"[INFO] weight_decay: {args.efficientad_weight_decay}")
    if args.dataset != "visa":
        print(f"[INFO] test_type   : {args.test_type}")
        print(f"[INFO] test_variant: {format_variant_selection(args.test_variant)}")
    print(f"[INFO] results_dir : {args.results_dir}")
    print(f"[INFO] variants    : {dataset_variant_summary}")
    if args.dataset != "visa":
        print_domain_shift_warning(dataset_variant_summary, selected_test_variants)
        print_threshold_warning(dataset_variant_summary)
    if args.model == "patchcore":
        print_patchcore_profile_warnings(args, image_size, tiling_config)
    print("=" * 80)

    engine.fit(model=model, datamodule=datamodule)

    print("[INFO] Training complete. Running test on selected split...")
    try:
        test_results = engine.test(model=model, datamodule=datamodule, ckpt_path="best")
    except Exception:
        test_results = engine.test(model=model, datamodule=datamodule)

    best_checkpoint = None
    try:
        checkpoint_callback = getattr(engine.trainer, "checkpoint_callback", None)
        if checkpoint_callback is not None:
            best_checkpoint = getattr(checkpoint_callback, "best_model_path", None) or None
    except Exception:
        best_checkpoint = None

    if not best_checkpoint:
        best_checkpoint = find_best_checkpoint(args.results_dir)

    metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"

    metadata = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "category": args.category,
        "model": args.model,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "image_size": format_image_size(image_size),
        "image_height": image_size[0],
        "image_width": image_size[1],
        "tiling": tiling_config,
        "patchcore_layers": list(validate_patchcore_args(args)) if args.model == "patchcore" else None,
        "patchcore_coreset_ratio": args.patchcore_coreset_ratio if args.model == "patchcore" else None,
        "patchcore_num_neighbors": args.patchcore_num_neighbors if args.model == "patchcore" else None,
        "patchcore_precision": args.patchcore_precision if args.model == "patchcore" else None,
        "efficientad_imagenet_dir": args.efficientad_imagenet_dir if args.model == "efficientad" else None,
        "efficientad_model_size": args.efficientad_model_size if args.model == "efficientad" else None,
        "efficientad_teacher_out_channels": args.efficientad_teacher_out_channels if args.model == "efficientad" else None,
        "efficientad_lr": args.efficientad_lr if args.model == "efficientad" else None,
        "efficientad_weight_decay": args.efficientad_weight_decay if args.model == "efficientad" else None,
        "efficientad_padding": args.efficientad_padding if args.model == "efficientad" else None,
        "efficientad_pad_maps": (not args.efficientad_no_pad_maps) if args.model == "efficientad" else None,
        "train_batch_size": train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "test_type": args.test_type,
        "test_variant": format_variant_selection(args.test_variant),
        "dataset_variant_summary": dataset_variant_summary,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "seed": args.seed,
        "best_checkpoint": best_checkpoint,
        "metrics_csv": metrics_csv if metrics_csv.exists() else None,
        "test_results": test_results,
    }

    metadata_path = args.results_dir / f"train_summary_{args.model}_{args.category}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(metadata), f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("[INFO] Done")
    print(f"[INFO] Summary JSON : {metadata_path}")
    print(f"[INFO] Best checkpoint: {best_checkpoint}")
    print(f"[INFO] Metrics CSV    : {metrics_csv if metrics_csv.exists() else 'not written'}")
    print("=" * 80)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
