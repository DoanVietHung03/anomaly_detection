#!/usr/bin/env python3
"""Run the full anomaly detection demo from one cross-platform entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_VARIANTS = ("regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3")
VARIANT_CHOICES = ("all", *IMAGE_VARIANTS)
DEFAULT_IMAGE_SIZE = (384, 837)
DEFAULT_TILING = "off"
DEFAULT_PATCHCORE_LAYERS = ("layer2", "layer3")
DEFAULT_PATCHCORE_CORESET_RATIO = 0.05
DEFAULT_PATCHCORE_NUM_NEIGHBORS = 9
DEFAULT_PATCHCORE_PRECISION = "float16"
DEFAULT_FIXED_ROI = (0.06, 0.10, 0.94, 0.90)
ROI_MODE_CHOICES = ("fixed-foreground", "fixed", "foreground", "off")
SCORE_AGGREGATION_CHOICES = (
    "model",
    "pixel-topk",
    "pixel-percentile",
    "max",
    "percentile",
    "topk-mean",
    "local-contrast",
    "local-ratio",
)
REQUIRED_IMPORTS = {
    "anomalib": "Anomalib",
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "PIL": "Pillow",
    "torch": "PyTorch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train, demo input preparation, and inference.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./can"),
        help="Path to the MVTec AD 2 root. Defaults to ./can for this project layout.",
    )
    parser.add_argument("--category", type=str, default="can", help="MVTec AD 2 category.")
    parser.add_argument(
        "--model",
        type=str,
        default="patchcore",
        choices=["patchcore", "efficientad"],
        help="Model architecture to train and run.",
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="Training output directory.")
    parser.add_argument("--input-dir", type=Path, default=Path("./demo_inputs"), help="Prepared demo input directory.")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("./calibration_inputs"),
        help="Prepared calibration input directory used for threshold selection.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Inference output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs. Defaults by model.")
    parser.add_argument("--num-good", type=int, default=8, help="Number of good demo images to sample.")
    parser.add_argument("--num-bad", type=int, default=8, help="Number of anomalous demo images to sample.")
    parser.add_argument("--num-calibration-good", type=int, default=30, help="Number of good calibration images to sample.")
    parser.add_argument("--num-calibration-bad", type=int, default=30, help="Number of bad calibration images to sample.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Eval/test batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square training/inference image size. Overridden by --image-height/--image-width.",
    )
    parser.add_argument("--image-height", type=int, default=None, help="Training/inference image height. Defaults to 384.")
    parser.add_argument("--image-width", type=int, default=None, help="Training/inference image width. Defaults to 837.")
    parser.add_argument(
        "--tiling",
        choices=["auto", "on", "off"],
        default=DEFAULT_TILING,
        help="Enable tiled PatchCore processing. off is the memory-safe default for wide can images.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="PatchCore tile size when tiling is enabled.")
    parser.add_argument("--tile-stride", type=int, default=None, help="PatchCore tile stride. Defaults to half tile size.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training and sampling.")
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration input preparation and use model/default labels.",
    )
    parser.add_argument(
        "--test-variant",
        nargs="+",
        default=["regular"],
        choices=VARIANT_CHOICES,
        help="Variant filter for the training-time evaluation split.",
    )
    parser.add_argument(
        "--demo-variants",
        nargs="+",
        default=None,
        choices=VARIANT_CHOICES,
        help="Variant filter for sampled demo inputs. Defaults to --test-variant.",
    )
    parser.add_argument(
        "--calibration-variants",
        nargs="+",
        default=None,
        choices=VARIANT_CHOICES,
        help="Variant filter for calibration inputs. Defaults to --demo-variants/--test-variant.",
    )
    parser.add_argument(
        "--exclude-calibration-from-demo",
        action="store_true",
        help="Keep demo samples distinct from calibration samples. With large regular calibration this may leave few demo images.",
    )
    parser.add_argument(
        "--heatmap-normalization",
        choices=["global", "per-image"],
        default="global",
        help="Normalize heatmaps across the inference batch or independently per image.",
    )
    parser.add_argument(
        "--roi-mode",
        choices=ROI_MODE_CHOICES,
        default="fixed-foreground",
        help="Restrict inference score aggregation to a fixed can ROI, foreground mask, both, or the full image.",
    )
    parser.add_argument(
        "--fixed-roi",
        nargs=4,
        type=float,
        default=list(DEFAULT_FIXED_ROI),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Fixed ROI as normalized fractions of width/height. Default keeps the can body and skips borders/glare edges.",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=SCORE_AGGREGATION_CHOICES,
        default="pixel-topk",
        help="Image score aggregation used by infer_demo.py. pixel-topk is the ROI/pixel-level default.",
    )
    parser.add_argument(
        "--calibration-objective",
        choices=["balanced-f1", "f1"],
        default="balanced-f1",
        help="Threshold selection objective for labelled calibration images.",
    )
    parser.add_argument(
        "--score-percentile",
        type=float,
        default=99.95,
        help="Percentile used when --score-aggregation percentile.",
    )
    parser.add_argument(
        "--score-topk-percent",
        type=float,
        default=0.05,
        help="Percentage of highest ROI pixels averaged by top-k based score aggregations.",
    )
    parser.add_argument(
        "--score-local-sigma",
        type=float,
        default=15.0,
        help="Gaussian sigma for local-contrast/local-ratio score aggregation.",
    )
    parser.add_argument("--accelerator", type=str, default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices", type=str, default="1", help="Lightning devices value.")
    parser.add_argument("--check-only", action="store_true", help="Only validate environment and dataset, then exit.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment and dataset preflight checks.")
    return parser.parse_args()


def default_epochs(model: str) -> int:
    return 30 if model == "efficientad" else 1


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


def validate_fixed_roi(values: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    roi = tuple(float(value) for value in values)
    if len(roi) != 4:
        raise SystemExit("--fixed-roi expects four values: X1 Y1 X2 Y2.")
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise SystemExit("--fixed-roi values must satisfy 0 <= X1 < X2 <= 1 and 0 <= Y1 < Y2 <= 1.")
    return roi


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run(command: list[str], project_root: Path) -> None:
    print(f"\n[RUN] {format_command(command)}", flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def project_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def require_python_version() -> list[str]:
    version = sys.version_info
    if (version.major, version.minor) < (3, 10) or (version.major, version.minor) > (3, 12):
        return [
            f"Python {version.major}.{version.minor} is active at {sys.executable}.",
            "Use Python 3.10-3.12 for this Anomalib demo, for example:",
            r"  .\venv\Scripts\python.exe run_demo.py",
        ]
    return []


def require_imports() -> list[str]:
    missing = [label for module, label in REQUIRED_IMPORTS.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return []
    return [
        "Missing Python packages: " + ", ".join(missing),
        "Install dependencies first, for example:",
        r"  .\venv\Scripts\python.exe -m pip install -r requirements.txt",
    ]


def wants_gpu(accelerator: str) -> bool:
    return accelerator.lower() in {"cuda", "gpu"}


def require_cuda(accelerator: str) -> list[str]:
    if not wants_gpu(accelerator) or importlib.util.find_spec("torch") is None:
        return []

    import torch

    if not torch.cuda.is_available():
        return [
            "PyTorch is installed, but CUDA is not available.",
            "Reinstall dependencies from requirements.txt and make sure it installs a CUDA PyTorch build.",
        ]
    return []


def require_dataset(dataset_root: Path, category: str) -> list[str]:
    category_root = dataset_root / category
    required_dirs = [
        category_root / "train" / "good",
        category_root / "validation" / "good",
        category_root / "test_public" / "good",
        category_root / "test_public" / "bad",
    ]

    errors: list[str] = []
    if not category_root.exists():
        return [
            f"Dataset category folder not found: {category_root}",
            "Pass --dataset-root to the folder that contains the category directory.",
        ]

    for path in required_dirs:
        image_count = count_images(path)
        if image_count == 0:
            errors.append(f"No images found under required dataset folder: {path}")

    return errors


def preflight(args: argparse.Namespace) -> None:
    errors = []
    errors.extend(require_python_version())
    errors.extend(require_imports())
    errors.extend(require_cuda(args.accelerator))
    errors.extend(require_dataset(args.dataset_root, args.category))

    if errors:
        raise SystemExit("[PRECHECK FAILED]\n" + "\n".join(f"- {error}" for error in errors))

    if wants_gpu(args.accelerator) and importlib.util.find_spec("torch") is not None:
        import torch

        device = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[PRECHECK OK] Python, dependencies, dataset, and GPU look ready: {device} ({memory_gb:.1f} GB).")
    else:
        print("[PRECHECK OK] Python, dependencies, and dataset look ready.")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    dataset_root = project_path(args.dataset_root, project_root)
    results_dir = project_path(args.results_dir or Path(f"./runs_{args.model}"), project_root)
    input_dir = project_path(args.input_dir, project_root)
    calibration_dir = project_path(args.calibration_dir, project_root)
    output_dir = project_path(args.output_dir or Path(f"./demo_outputs_{args.model}"), project_root)
    image_size = resolve_image_size(args)
    fixed_roi = validate_fixed_roi(args.fixed_roi)
    epochs = args.epochs if args.epochs is not None else default_epochs(args.model)
    default_batch_size = 1
    train_batch_size = args.train_batch_size if args.train_batch_size is not None else default_batch_size
    demo_variants = args.demo_variants if args.demo_variants is not None else args.test_variant
    calibration_variants = args.calibration_variants if args.calibration_variants is not None else demo_variants

    args.dataset_root = dataset_root
    if not args.skip_checks:
        preflight(args)
    if args.check_only:
        return

    python = sys.executable

    train_command = [
        python,
        "train_demo.py",
        "--dataset-root",
        str(dataset_root),
        "--category",
        args.category,
        "--model",
        args.model,
        "--results-dir",
        str(results_dir),
        "--epochs",
        str(epochs),
        "--train-batch-size",
        str(train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--image-height",
        str(image_size[0]),
        "--image-width",
        str(image_size[1]),
        "--num-workers",
        str(args.num_workers),
        "--tiling",
        args.tiling,
        "--tile-size",
        str(args.tile_size),
        "--patchcore-layers",
        *args.patchcore_layers,
        "--patchcore-coreset-ratio",
        str(args.patchcore_coreset_ratio),
        "--patchcore-num-neighbors",
        str(args.patchcore_num_neighbors),
        "--patchcore-precision",
        args.patchcore_precision,
        "--test-variant",
        *args.test_variant,
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
        "--seed",
        str(args.seed),
    ]
    if args.tile_stride is not None:
        train_command.extend(["--tile-stride", str(args.tile_stride)])
    run(train_command, project_root)

    calibration_manifest = calibration_dir / "manifest.json"
    use_calibration = not args.skip_calibration and (args.num_calibration_good > 0 or args.num_calibration_bad > 0)
    if use_calibration:
        calibration_command = [
            python,
            "prepare_demo_inputs.py",
            "--dataset-root",
            str(dataset_root),
            "--category",
            args.category,
            "--output-dir",
            str(calibration_dir),
            "--num-good",
            str(args.num_calibration_good),
            "--num-bad",
            str(args.num_calibration_bad),
            "--seed",
            str(args.seed),
            "--variants",
            *calibration_variants,
            "--manifest-path",
            str(calibration_manifest),
            "--clean",
        ]
        run(calibration_command, project_root)

    prepare_command = [
        python,
        "prepare_demo_inputs.py",
        "--dataset-root",
        str(dataset_root),
        "--category",
        args.category,
        "--output-dir",
        str(input_dir),
        "--num-good",
        str(args.num_good),
        "--num-bad",
        str(args.num_bad),
        "--seed",
        str(args.seed + 1),
        "--variants",
        *demo_variants,
        "--clean",
    ]
    if args.exclude_calibration_from_demo and use_calibration and calibration_manifest.exists():
        prepare_command.extend(["--exclude-manifest", str(calibration_manifest)])
    run(prepare_command, project_root)

    infer_command = [
        python,
        "infer_demo.py",
        "--input-path",
        str(input_dir),
        "--results-dir",
        str(results_dir),
        "--model",
        args.model,
        "--dataset-root",
        str(dataset_root),
        "--category",
        args.category,
        "--output-dir",
        str(output_dir),
        "--image-height",
        str(image_size[0]),
        "--image-width",
        str(image_size[1]),
        "--tiling",
        args.tiling,
        "--tile-size",
        str(args.tile_size),
        "--patchcore-layers",
        *args.patchcore_layers,
        "--patchcore-coreset-ratio",
        str(args.patchcore_coreset_ratio),
        "--patchcore-num-neighbors",
        str(args.patchcore_num_neighbors),
        "--patchcore-precision",
        args.patchcore_precision,
        "--roi-mode",
        args.roi_mode,
        "--fixed-roi",
        *[str(value) for value in fixed_roi],
        "--score-aggregation",
        args.score_aggregation,
        "--score-percentile",
        str(args.score_percentile),
        "--score-topk-percent",
        str(args.score_topk_percent),
        "--score-local-sigma",
        str(args.score_local_sigma),
        "--heatmap-normalization",
        args.heatmap_normalization,
        "--calibration-objective",
        args.calibration_objective,
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
    ]
    if args.tile_stride is not None:
        infer_command.extend(["--tile-stride", str(args.tile_stride)])
    if use_calibration and calibration_manifest.exists():
        infer_command.extend(["--calibration-path", str(calibration_dir)])
    run(infer_command, project_root)

    print(f"\nDone. Open: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
