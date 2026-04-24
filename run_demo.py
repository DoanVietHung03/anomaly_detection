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

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_VARIANTS = ("regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3")
VARIANT_CHOICES = ("all", *IMAGE_VARIANTS)
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
    parser.add_argument("--output-dir", type=Path, default=None, help="Inference output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs. Defaults by model.")
    parser.add_argument("--num-good", type=int, default=8, help="Number of good demo images to sample.")
    parser.add_argument("--num-bad", type=int, default=8, help="Number of anomalous demo images to sample.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Eval/test batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--image-size", type=int, default=256, help="Training and inference image size.")
    parser.add_argument(
        "--test-variant",
        nargs="+",
        default=["all"],
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
        "--heatmap-normalization",
        choices=["global", "per-image"],
        default="global",
        help="Normalize heatmaps across the inference batch or independently per image.",
    )
    parser.add_argument("--accelerator", type=str, default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices", type=str, default="1", help="Lightning devices value.")
    parser.add_argument("--check-only", action="store_true", help="Only validate environment and dataset, then exit.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment and dataset preflight checks.")
    return parser.parse_args()


def default_epochs(model: str) -> int:
    return 30 if model == "efficientad" else 1


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
    output_dir = project_path(args.output_dir or Path(f"./demo_outputs_{args.model}"), project_root)
    epochs = args.epochs if args.epochs is not None else default_epochs(args.model)
    default_batch_size = 1 if args.model == "efficientad" else 8
    train_batch_size = args.train_batch_size if args.train_batch_size is not None else default_batch_size
    demo_variants = args.demo_variants if args.demo_variants is not None else args.test_variant

    args.dataset_root = dataset_root
    if not args.skip_checks:
        preflight(args)
    if args.check_only:
        return

    python = sys.executable

    run(
        [
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
            "--image-size",
            str(args.image_size),
            "--num-workers",
            str(args.num_workers),
            "--test-variant",
            *args.test_variant,
            "--accelerator",
            args.accelerator,
            "--devices",
            args.devices,
        ],
        project_root,
    )

    run(
        [
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
            "--variants",
            *demo_variants,
            "--clean",
        ],
        project_root,
    )

    run(
        [
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
            "--image-size",
            str(args.image_size),
            "--heatmap-normalization",
            args.heatmap_normalization,
            "--accelerator",
            args.accelerator,
            "--devices",
            args.devices,
        ],
        project_root,
    )

    print(f"\nDone. Open: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
