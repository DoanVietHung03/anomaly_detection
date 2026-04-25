#!/usr/bin/env python3
"""Copy a small, mixed set of public test images into a demo input folder."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IGNORED_TEST_DIRS = {"ground_truth"}
IMAGE_VARIANTS = ("regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3")
VARIANT_CHOICES = ("all", *IMAGE_VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small demo input folder from MVTec AD 2 test_public.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to MVTec_AD_2 root.")
    parser.add_argument("--category", type=str, default="can", help="MVTec AD 2 category.")
    parser.add_argument("--output-dir", type=Path, default=Path("./demo_inputs"), help="Output folder.")
    parser.add_argument("--num-good", type=int, default=8, help="Number of good images to sample.")
    parser.add_argument("--num-bad", type=int, default=8, help="Number of anomalous images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["regular"],
        choices=VARIANT_CHOICES,
        help="Capture variants to sample. Use all, or one/more of regular, overexposed, underexposed, shift_1, shift_2, shift_3.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing generated good/bad folders first.")
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
            and p.is_file()
            and not p.stem.lower().endswith("_mask")
        ]
    )


def sample_paths(paths: list[Path], n: int, rng: random.Random) -> list[Path]:
    if n <= 0 or not paths:
        return []
    if len(paths) <= n:
        return paths
    return rng.sample(paths, n)


def normalize_variant_selection(variants: list[str]) -> set[str] | None:
    selected = {variant.lower() for variant in variants}
    if not selected or "all" in selected:
        return None
    return selected


def image_variant(path: Path) -> str:
    stem = path.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def filter_by_variants(paths: list[Path], variants: set[str] | None) -> list[Path]:
    if variants is None:
        return paths
    return [path for path in paths if image_variant(path) in variants]


def format_variant_selection(variants: set[str] | None) -> str:
    if variants is None:
        return "all"
    return ",".join(sorted(variants))


def copy_paths(paths: list[Path], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(paths):
        dst = out_dir / f"{idx:03d}_{prefix}_{src.name}"
        if dst.exists():
            dst.chmod(0o666)
        shutil.copy2(src, dst)
        dst.chmod(0o666)


def reset_generated_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file_path in list_images(path):
        try:
            file_path.chmod(0o666)
            file_path.unlink()
        except PermissionError as exc:
            raise SystemExit(
                f"Could not clean locked demo input file: {file_path}\n"
                "Close any image viewer, Explorer preview, browser report tab, or editor preview using this file, then retry."
            ) from exc
    for child in sorted(path.iterdir(), reverse=True):
        if child.is_dir():
            try:
                child.chmod(0o777)
                os.rmdir(child)
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    selected_variants = normalize_variant_selection(args.variants)

    category_root = args.dataset_root / args.category
    test_public = category_root / "test_public"
    good_dir = test_public / "good"

    if not test_public.exists():
        raise SystemExit(f"Could not find test_public directory: {test_public}")
    if not good_dir.exists():
        raise SystemExit(f"Could not find good directory: {good_dir}")

    good_images = filter_by_variants(list_images(good_dir), selected_variants)

    bad_images: list[Path] = []
    for child in sorted(test_public.iterdir()):
        child_name = child.name.lower()
        if child.is_dir() and child_name != "good" and child_name not in IGNORED_TEST_DIRS:
            bad_images.extend(list_images(child))
    bad_images = filter_by_variants(bad_images, selected_variants)

    if args.num_good > 0 and not good_images:
        raise SystemExit(f"No good images matched --variants {format_variant_selection(selected_variants)}.")
    if args.num_bad > 0 and not bad_images:
        raise SystemExit(f"No bad images matched --variants {format_variant_selection(selected_variants)}.")

    sampled_good = sample_paths(good_images, args.num_good, rng)
    sampled_bad = sample_paths(bad_images, args.num_bad, rng)

    good_out = args.output_dir / "good"
    bad_out = args.output_dir / "bad"
    if args.clean:
        reset_generated_files(good_out)
        reset_generated_files(bad_out)

    copy_paths(sampled_good, good_out, "good")
    copy_paths(sampled_bad, bad_out, "bad")

    print("=" * 80)
    print("[INFO] Demo inputs prepared")
    print(f"[INFO] Source      : {test_public}")
    print(f"[INFO] Variants    : {format_variant_selection(selected_variants)}")
    print(f"[INFO] Output dir  : {args.output_dir}")
    print(f"[INFO] Good copied : {len(sampled_good)}")
    print(f"[INFO] Bad copied  : {len(sampled_bad)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
