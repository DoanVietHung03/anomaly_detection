#!/usr/bin/env python3
"""Train/export an EfficientAD-style native backend for prepared VisA splits.

This script is intentionally separate from train_demo.py. It gives us a small
test bed for the EfficientAD recipe used by the reference implementation:
256px inputs, max anomaly-map score, validation quantile normalization, and an
optional ImageNet/Imagenette penalty branch that can be disabled with "none".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from itertools import cycle
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from infer_demo import (
    infer_ground_truth_from_path,
    load_gt_mask,
    make_heatmap,
    normalize_map,
    overlay_heatmap,
    overlay_mask,
    prediction_diagnostics,
    save_image,
)
from train_demo import (
    DEFAULT_EFFICIENTAD_LR,
    DEFAULT_EFFICIENTAD_TEACHER_OUT_CHANNELS,
    DEFAULT_EFFICIENTAD_WEIGHT_DECAY,
    format_image_size,
)


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
LABELS = ("good", "bad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native EfficientAD trainer/exporter for VisA hybrid splits.")
    parser.add_argument("--dataset-root", type=Path, default=Path("./VisA"))
    parser.add_argument("--category", default="candle")
    parser.add_argument("--split-dir", type=Path, default=None, help="Prepared hybrid split folder.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Native EfficientAD checkpoint/log folder.")
    parser.add_argument("--maps-dir", type=Path, default=None, help="Output root containing train/val/test maps.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Existing native checkpoint for --export-only.")
    parser.add_argument("--model-size", choices=["small", "medium"], default="medium")
    parser.add_argument("--teacher-out-channels", type=int, default=DEFAULT_EFFICIENTAD_TEACHER_OUT_CHANNELS)
    parser.add_argument(
        "--teacher-weights",
        default="auto",
        help="Path to pretrained teacher .pth, or auto to use Anomalib's cached weights.",
    )
    parser.add_argument(
        "--imagenet-train-path",
        default="none",
        help="ImageNet/Imagenette-style folder for penalty branch, or none to disable it.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=70_000)
    parser.add_argument("--lr", type=float, default=DEFAULT_EFFICIENTAD_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_EFFICIENTAD_WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "cuda"], default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Clean results/maps dirs before running.")
    parser.add_argument("--export-only", action="store_true", help="Load --checkpoint and only export maps.")
    parser.add_argument("--skip-export", action="store_true", help="Train only; do not export maps.")
    parser.add_argument("--no-visuals", action="store_true", help="Only write raw maps, masks, originals, and CSV.")
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(accelerator: str) -> torch.device:
    if accelerator in {"gpu", "cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
            ],
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def image_loader(paths: list[Path], image_size: int, *, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ImagePathDataset(paths, image_size),
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def penalty_loader(path_text: str, image_size: int, num_workers: int) -> DataLoader | None:
    if path_text.lower() in {"", "none", "off", "false"}:
        return None
    path = Path(path_text)
    if not path.is_dir():
        raise SystemExit(f"--imagenet-train-path not found: {path}")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size * 2, image_size * 2), antialias=True),
            transforms.RandomGrayscale(p=0.3),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
        ],
    )
    dataset = ImageFolder(path, transform=transform)
    if len(dataset) == 0:
        raise SystemExit(f"No images found in --imagenet-train-path: {path}")
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def import_efficientad_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from anomalib.models.image.efficient_ad.lightning_model import (
            WEIGHTS_DOWNLOAD_INFO,
            download_and_extract,
            get_pretrained_weights_dir,
        )
        from anomalib.models.image.efficient_ad.torch_model import (
            EfficientAdModel,
            EfficientAdModelSize,
            reduce_tensor_elems,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(f"Could not import Anomalib EfficientAD internals: {exc}") from exc
    return EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems, get_pretrained_weights_dir, (download_and_extract, WEIGHTS_DOWNLOAD_INFO)


def resolve_teacher_weights(args: argparse.Namespace, get_pretrained_weights_dir: Any, download_bundle: tuple[Any, Any]) -> Path:
    if args.teacher_weights != "auto":
        path = Path(args.teacher_weights)
        if not path.exists():
            raise SystemExit(f"--teacher-weights not found: {path}")
        return path

    download_and_extract, weights_download_info = download_bundle
    pretrained_dir = get_pretrained_weights_dir()
    weights_dir = pretrained_dir / "efficientad_pretrained_weights"
    if not weights_dir.is_dir():
        print("[INFO] Anomalib EfficientAD teacher weights not found; downloading once...")
        download_and_extract(pretrained_dir, weights_download_info)
    path = weights_dir / f"pretrained_teacher_{args.model_size}.pth"
    if not path.exists():
        raise SystemExit(f"Could not find pretrained teacher weights: {path}")
    return path


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(args: argparse.Namespace, device: torch.device) -> Any:
    EfficientAdModel, EfficientAdModelSize, _, get_pretrained_weights_dir, download_bundle = import_efficientad_stack()
    model_size = EfficientAdModelSize.M if args.model_size == "medium" else EfficientAdModelSize.S
    model = EfficientAdModel(
        teacher_out_channels=args.teacher_out_channels,
        model_size=model_size,
        padding=False,
        pad_maps=True,
    ).to(device)
    teacher_weights = resolve_teacher_weights(args, get_pretrained_weights_dir, download_bundle)
    state_dict = torch_load(teacher_weights, device)
    model.teacher.load_state_dict(state_dict)
    for param in model.teacher.parameters():
        param.requires_grad_(False)
    model.teacher.eval()
    return model


@torch.no_grad()
def set_teacher_mean_std(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    means = []
    for images, _ in tqdm(loader, desc="Computing teacher mean"):
        output = model.teacher(images.to(device, non_blocking=True))
        means.append(torch.mean(output, dim=[0, 2, 3]))
    if not means:
        raise SystemExit("No train/good images found for teacher normalization.")
    channel_mean = torch.mean(torch.stack(means), dim=0)[None, :, None, None]

    variances = []
    for images, _ in tqdm(loader, desc="Computing teacher std"):
        output = model.teacher(images.to(device, non_blocking=True))
        distance = (output - channel_mean) ** 2
        variances.append(torch.mean(distance, dim=[0, 2, 3]))
    channel_std = torch.sqrt(torch.mean(torch.stack(variances), dim=0))[None, :, None, None]
    channel_std = torch.clamp(channel_std, min=1e-6)

    model.mean_std["mean"].data.copy_(channel_mean.to(device))
    model.mean_std["std"].data.copy_(channel_std.to(device))
    return {"teacher_mean_avg": float(channel_mean.mean().cpu()), "teacher_std_avg": float(channel_std.mean().cpu())}


def train_native(
    args: argparse.Namespace,
    model: Any,
    train_loader: DataLoader,
    penalty: DataLoader | None,
    device: torch.device,
) -> list[dict[str, float]]:
    _, _, reduce_tensor_elems, _, _ = import_efficientad_stack()
    optimizer = torch.optim.Adam(
        list(model.student.parameters()) + list(model.ae.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(0.95 * args.train_steps)), gamma=0.1)
    train_iter = cycle(train_loader)
    penalty_iter = cycle(penalty) if penalty is not None else None
    history: list[dict[str, float]] = []

    model.train()
    model.teacher.eval()
    pbar = tqdm(range(args.train_steps), desc="Training native EfficientAD")
    for step in pbar:
        images, _ = next(train_iter)
        images = images.to(device, non_blocking=True)
        penalty_images = None
        if penalty_iter is not None:
            penalty_images, _ = next(penalty_iter)
            penalty_images = penalty_images.to(device, non_blocking=True)

        student_output, distance_st = model.compute_student_teacher_distance(images)
        distance_st = reduce_tensor_elems(distance_st)
        hard_threshold = torch.quantile(distance_st, 0.999)
        loss_hard = torch.mean(distance_st[distance_st >= hard_threshold])
        if penalty_images is not None:
            student_output_penalty = model.student(penalty_images)[:, : args.teacher_out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
        else:
            loss_penalty = torch.zeros((), device=device)
        loss_st = loss_hard + loss_penalty

        aug_images = model.choose_random_aug_image(images)
        ae_output = model.ae(aug_images, images.shape[-2:])
        with torch.no_grad():
            teacher_output = model.teacher(aug_images)
            teacher_output = (teacher_output - model.mean_std["mean"]) / model.mean_std["std"]
        student_output_ae = model.student(aug_images)[:, args.teacher_out_channels :]
        loss_ae = torch.mean((teacher_output - ae_output) ** 2)
        loss_stae = torch.mean((ae_output - student_output_ae) ** 2)
        loss = loss_st + loss_ae + loss_stae

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == args.train_steps - 1:
            row = {
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "loss_st": float(loss_st.detach().cpu()),
                "loss_ae": float(loss_ae.detach().cpu()),
                "loss_stae": float(loss_stae.detach().cpu()),
                "loss_penalty": float(loss_penalty.detach().cpu()),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            history.append(row)
            pbar.set_postfix(loss=f"{row['loss']:.4f}", penalty=f"{row['loss_penalty']:.4f}")
    return history


@torch.no_grad()
def set_map_quantiles(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    maps_st = []
    maps_ae = []
    for images, _ in tqdm(loader, desc="Computing map quantiles"):
        images = images.to(device, non_blocking=True)
        map_st, map_ae = model.get_maps(images, normalize=False)
        maps_st.append(map_st.detach())
        maps_ae.append(map_ae.detach())
    if not maps_st:
        raise SystemExit("No good validation images found for map normalization.")
    flat_st = torch.cat([m.reshape(-1) for m in maps_st])
    flat_ae = torch.cat([m.reshape(-1) for m in maps_ae])
    qa_st = torch.quantile(flat_st, 0.9).to(device)
    qb_st = torch.quantile(flat_st, 0.995).to(device)
    qa_ae = torch.quantile(flat_ae, 0.9).to(device)
    qb_ae = torch.quantile(flat_ae, 0.995).to(device)
    model.quantiles["qa_st"].data.copy_(qa_st)
    model.quantiles["qb_st"].data.copy_(torch.maximum(qb_st, qa_st + 1e-6))
    model.quantiles["qa_ae"].data.copy_(qa_ae)
    model.quantiles["qb_ae"].data.copy_(torch.maximum(qb_ae, qa_ae + 1e-6))
    return {
        "qa_st": float(qa_st.cpu()),
        "qb_st": float(qb_st.cpu()),
        "qa_ae": float(qa_ae.cpu()),
        "qb_ae": float(qb_ae.cpu()),
    }


def save_checkpoint(path: Path, model: Any, args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, model: Any, device: torch.device) -> dict[str, Any]:
    checkpoint = torch_load(path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}


def split_paths(split_dir: Path) -> dict[str, dict[str, list[Path]]]:
    paths = {split: {label: list_images(split_dir / split / label) for label in LABELS} for split in SPLITS}
    if not paths["train"]["good"]:
        raise SystemExit(f"No train/good images found under split dir: {split_dir}")
    return paths


def resized_image_rgb(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.array(image)


@torch.no_grad()
def predict_anomaly_map(model: Any, image_tensor: torch.Tensor, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    image_tensor = image_tensor.to(device, non_blocking=True)
    map_st, map_ae = model.get_maps(image_tensor, normalize=True)
    anomaly_map = 0.5 * map_st + 0.5 * map_ae
    score = float(torch.amax(anomaly_map).detach().cpu())
    return anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32), score


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_split(
    args: argparse.Namespace,
    model: Any,
    split: str,
    paths_by_label: dict[str, list[Path]],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    for name in ("originals", "raw_maps", "raw_float_maps", "gt_masks"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    if not args.no_visuals:
        for name in ("heatmaps", "overlays", "gt_overlays"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)

    all_paths = [(path, label) for label in LABELS for path in paths_by_label[label]]
    dataset = ImagePathDataset([path for path, _ in all_paths], args.image_size)
    rows: list[dict[str, Any]] = []
    for idx, ((image_tensor, _), (image_path, label)) in enumerate(
        tqdm(zip(dataset, all_paths), total=len(all_paths), desc=f"Exporting {split} maps"),
    ):
        anomaly_map, pred_score = predict_anomaly_map(model, image_tensor.unsqueeze(0), device)
        image_rgb = resized_image_rgb(image_path, args.image_size)
        gt_label = label if label in {"good", "bad"} else infer_ground_truth_from_path(image_path)
        gt_mask, gt_mask_source = load_gt_mask(
            dataset_root=args.dataset_root,
            category=args.category,
            test_type="public",
            image_path=image_path,
            image_hw=image_rgb.shape[:2],
            gt_label=gt_label,
            dataset="visa",
        )

        stem = f"{idx:04d}_{image_path.stem}"
        original_path = output_dir / "originals" / f"{stem}.png"
        raw_map_path = output_dir / "raw_maps" / f"{stem}.png"
        raw_float_path = output_dir / "raw_float_maps" / f"{stem}.npy"
        gt_mask_path = output_dir / "gt_masks" / f"{stem}.png"

        save_image(original_path, image_rgb)
        Image.fromarray(normalize_map(anomaly_map)).save(raw_map_path)
        np.save(raw_float_path, anomaly_map.astype(np.float32, copy=False))
        Image.fromarray(gt_mask).save(gt_mask_path)

        heatmap_rel = overlay_rel = gt_overlay_rel = ""
        if not args.no_visuals:
            heatmap = make_heatmap(normalize_map(anomaly_map), target_hw=image_rgb.shape[:2])
            overlay = overlay_heatmap(image_rgb, heatmap, alpha=0.45)
            gt_overlay = overlay_mask(image_rgb, gt_mask)
            heatmap_path = output_dir / "heatmaps" / f"{stem}.png"
            overlay_path = output_dir / "overlays" / f"{stem}.png"
            gt_overlay_path = output_dir / "gt_overlays" / f"{stem}.png"
            save_image(heatmap_path, heatmap)
            save_image(overlay_path, overlay)
            save_image(gt_overlay_path, gt_overlay)
            heatmap_rel = heatmap_path.relative_to(output_dir).as_posix()
            overlay_rel = overlay_path.relative_to(output_dir).as_posix()
            gt_overlay_rel = gt_overlay_path.relative_to(output_dir).as_posix()

        rows.append(
            {
                "index": idx,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "pred_label": "",
                "pred_label_name": "",
                "anomalib_pred_label": "",
                "pred_score": pred_score,
                "raw_pred_score": pred_score,
                "model_pred_score": pred_score,
                "calibrated_threshold": "",
                "score_mode": "efficientad_native",
                "score_aggregation": "max",
                "gt_label": gt_label,
                "correct": "",
                "anomaly_map_min": float(anomaly_map.min()),
                "anomaly_map_max": float(anomaly_map.max()),
                "gt_mask_source": str(gt_mask_source) if gt_mask_source else "",
                "original_rel": original_path.relative_to(output_dir).as_posix(),
                "gt_mask_rel": gt_mask_path.relative_to(output_dir).as_posix(),
                "gt_overlay_rel": gt_overlay_rel,
                "heatmap_rel": heatmap_rel,
                "overlay_rel": overlay_rel,
                "raw_map_rel": raw_map_path.relative_to(output_dir).as_posix(),
                "raw_float_map_rel": raw_float_path.relative_to(output_dir).as_posix(),
            },
        )

    write_rows_csv(output_dir / "predictions.csv", rows)
    diagnostics = prediction_diagnostics(rows)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "efficientad_native",
                "split": split,
                "image_size": format_image_size((args.image_size, args.image_size)),
                "prediction_diagnostics": diagnostics,
                "num_samples": len(rows),
            },
            f,
            indent=2,
        )
    return {"num_samples": len(rows), "prediction_diagnostics": diagnostics}


def write_training_log(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    write_rows_csv(path, history)


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive.")
    if args.train_steps <= 0:
        raise SystemExit("--train-steps must be positive.")
    if args.split_dir is None:
        args.split_dir = Path(f"./hybrid_inputs_visa_{args.category}_native")
    if args.results_dir is None:
        args.results_dir = Path(f"./runs_efficientad_native_visa_{args.category}")
    if args.maps_dir is None:
        args.maps_dir = Path(f"./hybrid_maps_visa_{args.category}_efficientad_native")

    seed_everything(args.seed)
    device = resolve_device(args.accelerator)
    split = split_paths(args.split_dir)

    if args.clean and not args.export_only:
        reset_dir(args.results_dir)
    else:
        args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.clean and not args.skip_export:
        reset_dir(args.maps_dir)
    else:
        args.maps_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    checkpoint_path = args.checkpoint or (args.results_dir / "efficientad_native_model.pt")
    metadata: dict[str, Any] = {
        "model": "efficientad_native",
        "category": args.category,
        "image_size": format_image_size((args.image_size, args.image_size)),
        "model_size": args.model_size,
        "teacher_out_channels": args.teacher_out_channels,
        "imagenet_train_path": args.imagenet_train_path,
        "train_steps": args.train_steps,
        "seed": args.seed,
    }

    if args.export_only:
        if not checkpoint_path.exists():
            raise SystemExit(f"--export-only checkpoint not found: {checkpoint_path}")
        metadata.update(load_checkpoint(checkpoint_path, model, device))
    else:
        train_good_loader = image_loader(split["train"]["good"], args.image_size, shuffle=True, num_workers=args.num_workers)
        norm_loader = image_loader(split["train"]["good"], args.image_size, shuffle=False, num_workers=args.num_workers)
        val_good = split["val"]["good"] or split["train"]["good"]
        quantile_loader = image_loader(val_good, args.image_size, shuffle=False, num_workers=args.num_workers)
        penalty = penalty_loader(args.imagenet_train_path, args.image_size, args.num_workers)

        print("=" * 80)
        print("[INFO] Training native EfficientAD")
        print(f"[INFO] split_dir     : {args.split_dir}")
        print(f"[INFO] results_dir   : {args.results_dir}")
        print(f"[INFO] maps_dir      : {args.maps_dir}")
        print(f"[INFO] device        : {device}")
        print(f"[INFO] image_size    : {args.image_size}x{args.image_size}")
        print(f"[INFO] model_size    : {args.model_size}")
        print(f"[INFO] imagenet path : {args.imagenet_train_path}")
        print(f"[INFO] train/good    : {len(split['train']['good'])}")
        print(f"[INFO] val/good      : {len(split['val']['good'])}")
        print(f"[INFO] test/good,bad : {len(split['test']['good'])}, {len(split['test']['bad'])}")
        print("=" * 80)

        metadata.update(set_teacher_mean_std(model, norm_loader, device))
        history = train_native(args, model, train_good_loader, penalty, device)
        metadata["quantiles"] = set_map_quantiles(model, quantile_loader, device)
        save_checkpoint(checkpoint_path, model, args, metadata)
        write_training_log(args.results_dir / "training_log.csv", history)

    if not args.skip_export:
        export_summary = {}
        for split_name in SPLITS:
            export_summary[split_name] = export_split(
                args,
                model,
                split_name,
                split[split_name],
                args.maps_dir / split_name,
                device,
            )
        metadata["exports"] = export_summary

    summary_path = args.results_dir / f"train_summary_efficientad_native_{args.category}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("=" * 80)
    print("[INFO] Native EfficientAD done")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Summary   : {summary_path}")
    if not args.skip_export:
        test_diag = metadata.get("exports", {}).get("test", {}).get("prediction_diagnostics", {})
        auc = test_diag.get("score_auc")
        if auc is not None:
            print(f"[INFO] Test score AUC: {float(auc):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
