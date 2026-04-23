#!/usr/bin/env python3
"""Run inference for a trained anomaly detection model and build a demo report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference demo and export heatmaps/report.")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to a single image or a folder of images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Direct path to a .ckpt file. If omitted, --results-dir will be searched.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Training output directory to search for checkpoint automatically.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="patchcore",
        choices=["patchcore", "efficientad"],
        help="Model architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./can"),
        help="Path to MVTec AD 2 root. Used only to add GT masks/overlays to reports.",
    )
    parser.add_argument("--category", type=str, default="can", help="MVTec AD 2 category for GT masks.")
    parser.add_argument(
        "--test-type",
        type=str,
        default="public",
        choices=["public", "private", "private_mixed"],
        help="MVTec AD 2 test split used for GT masks.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./demo_outputs"), help="Inference output directory.")
    parser.add_argument("--image-size", type=int, default=256, help="PredictDataset image size.")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices", type=str, default="1", help="Lightning devices.")
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from anomalib.data import PredictDataset
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd, Patchcore
    except Exception as exc:  # pragma: no cover - runtime safeguard
        raise SystemExit(
            "Failed to import Anomalib stack. Install dependencies first, for example:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc
    return PredictDataset, Engine, Patchcore, EfficientAd


def find_checkpoint(results_dir: Path) -> Path | None:
    ckpts = sorted(results_dir.rglob("*.ckpt"))
    if not ckpts:
        return None
    preferred = [p for p in ckpts if "best" in p.name.lower() or "model" in p.name.lower()]
    return preferred[0] if preferred else ckpts[0]


def build_model(model_name: str, patchcore_cls: Any, efficientad_cls: Any) -> Any:
    if model_name == "patchcore":
        return patchcore_cls(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
    if model_name == "efficientad":
        return efficientad_cls()
    raise ValueError(f"Unsupported model: {model_name}")


def flatten_predictions(predictions: Any) -> list[Any]:
    if predictions is None:
        return []
    if isinstance(predictions, list):
        flattened: list[Any] = []
        for item in predictions:
            flattened.extend(flatten_predictions(item))
        return flattened
    return [predictions]


def get_batch_size(prediction: Any) -> int:
    image_path = getattr(prediction, "image_path", None)
    if isinstance(image_path, (list, tuple)):
        return len(image_path)

    for field in ("pred_score", "pred_label", "anomaly_map", "image"):
        value = getattr(prediction, field, None)
        shape = getattr(value, "shape", None)
        if shape and len(shape) > 0:
            return int(shape[0])
    return 1


def is_batch_prediction(prediction: Any) -> bool:
    if isinstance(getattr(prediction, "image_path", None), (list, tuple)):
        return True
    for field in ("pred_score", "pred_label", "anomaly_map", "image"):
        value = getattr(prediction, field, None)
        shape = getattr(value, "shape", None)
        if shape and len(shape) > 0:
            return True
    return False


def select_batch_value(value: Any, index: int, batch_size: int) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index]

    shape = getattr(value, "shape", None)
    if shape and len(shape) > 0 and int(shape[0]) == batch_size:
        return value[index]
    return value


def expand_batch_prediction(prediction: Any) -> list[Any]:
    batch_size = get_batch_size(prediction)
    if not is_batch_prediction(prediction):
        return [prediction]

    return [
        SimpleNamespace(
            image_path=select_batch_value(getattr(prediction, "image_path", None), index, batch_size),
            image=select_batch_value(getattr(prediction, "image", None), index, batch_size),
            anomaly_map=select_batch_value(getattr(prediction, "anomaly_map", None), index, batch_size),
            pred_label=select_batch_value(getattr(prediction, "pred_label", None), index, batch_size),
            pred_score=select_batch_value(getattr(prediction, "pred_score", None), index, batch_size),
        )
        for index in range(batch_size)
    ]


def expand_predictions(predictions: Any) -> list[Any]:
    expanded: list[Any] = []
    for prediction in flatten_predictions(predictions):
        expanded.extend(expand_batch_prediction(prediction))
    return expanded


def to_numpy_image(image_like: Any) -> np.ndarray:
    if image_like is None:
        raise ValueError("Received None instead of image.")

    if isinstance(image_like, np.ndarray):
        arr = image_like
    else:
        try:
            import torch

            if isinstance(image_like, torch.Tensor):
                arr = image_like.detach().cpu().float().numpy()
            else:
                arr = np.asarray(image_like)
        except Exception:
            arr = np.asarray(image_like)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def tensor_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().float().numpy()
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)

    arr = np.squeeze(arr)
    return arr


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32)
    min_v = float(anomaly_map.min())
    max_v = float(anomaly_map.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.uint8)
    norm = (anomaly_map - min_v) / (max_v - min_v)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def make_heatmap(gray_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    heatmap = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    target_h, target_w = target_hw
    if heatmap.shape[:2] != (target_h, target_w):
        heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return heatmap


def overlay_heatmap(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = (image_rgb.astype(np.float32) * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def overlay_mask(image_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Overlay a green GT mask on the image, dilated slightly for visibility."""
    if mask_gray.shape[:2] != image_rgb.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = mask_gray > 0
    if not np.any(mask):
        return image_rgb.copy()

    kernel_size = max(5, int(round(min(image_rgb.shape[:2]) * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    display_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0

    overlay = image_rgb.copy().astype(np.float32)
    gt_color = np.array([0, 255, 80], dtype=np.float32)
    overlay[display_mask] = overlay[display_mask] * (1.0 - alpha) + gt_color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def infer_ground_truth_from_path(path: Path) -> str | None:
    text = str(path).replace("\\", "/").lower()
    if "/good/" in text:
        return "good"
    if "/bad/" in text:
        return "bad"
    if "test_public" in text:
        return "bad"
    return None


def source_image_name_from_demo_name(path: Path) -> str:
    parts = path.name.split("_")
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].lower() in {"good", "bad"}:
        return "_".join(parts[2:])
    return path.name


def category_roots(dataset_root: Path, category: str) -> list[Path]:
    roots = [dataset_root / category, dataset_root]
    return [root for idx, root in enumerate(roots) if root not in roots[:idx]]


def find_gt_mask_path(dataset_root: Path, category: str, test_type: str, image_path: Path, gt_label: str | None) -> Path | None:
    if gt_label != "bad":
        return None

    source_name = source_image_name_from_demo_name(image_path)
    source_stem = Path(source_name).stem
    split_name = f"test_{test_type}"
    mask_name = f"{source_stem}_mask.png"

    for category_root in category_roots(dataset_root, category):
        candidates = [
            category_root / split_name / "ground_truth" / "bad" / mask_name,
            category_root / split_name / "ground_truth" / mask_name,
            category_root / "ground_truth" / "bad" / mask_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def load_gt_mask(
    dataset_root: Path,
    category: str,
    test_type: str,
    image_path: Path,
    image_hw: tuple[int, int],
    gt_label: str | None,
) -> tuple[np.ndarray, Path | None]:
    target_h, target_w = image_hw
    mask_path = find_gt_mask_path(dataset_root, category, test_type, image_path, gt_label)
    if mask_path is None:
        return np.zeros((target_h, target_w), dtype=np.uint8), None

    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return mask, mask_path


def save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def build_html_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <div class=\"card\">
              <div class=\"meta\">
                <div><b>File:</b> {html.escape(row['image_name'])}</div>
                <div><b>Pred:</b> {html.escape(str(row['pred_label']))}</div>
                <div><b>Score:</b> {row['pred_score']:.6f}</div>
                <div><b>GT guess:</b> {html.escape(str(row.get('gt_label', 'unknown')))}</div>
              </div>
              <div class=\"grid\">
                <div><p>Original</p><img src=\"{html.escape(row['original_rel'])}\"></div>
                <div><p>GT Mask</p><img src=\"{html.escape(row['gt_mask_rel'])}\"></div>
                <div><p>GT Overlay</p><img src=\"{html.escape(row['gt_overlay_rel'])}\"></div>
                <div><p>Heatmap</p><img src=\"{html.escape(row['heatmap_rel'])}\"></div>
                <div><p>Pred Overlay</p><img src=\"{html.escape(row['overlay_rel'])}\"></div>
              </div>
            </div>
            """.strip()
        )

    html_text = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anomaly Demo Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    .summary {{ margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
    .meta {{ margin-bottom: 12px; line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: repeat(5, minmax(180px, 1fr)); gap: 12px; }}
    img {{ width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
    p {{ margin: 0 0 8px 0; font-weight: 600; }}
    @media (max-width: 960px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Anomaly Detection Demo Report</h1>
  <div class="summary">
    <div><b>Total samples:</b> {len(rows)}</div>
  </div>
  {'\n'.join(cards)}
</body>
</html>
    """.strip()
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    predict_dataset_cls, engine_cls, patchcore_cls, efficient_ad_cls = import_dependencies()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if args.results_dir is None:
            raise SystemExit("Provide either --checkpoint or --results-dir.")
        ckpt_path = find_checkpoint(args.results_dir)
        if ckpt_path is None:
            raise SystemExit(f"No checkpoint found under: {args.results_dir}")

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if not args.input_path.exists():
        raise SystemExit(f"Input path not found: {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = args.output_dir / "originals"
    heatmap_dir = args.output_dir / "heatmaps"
    overlay_dir = args.output_dir / "overlays"
    rawmap_dir = args.output_dir / "raw_maps"
    gt_mask_dir = args.output_dir / "gt_masks"
    gt_overlay_dir = args.output_dir / "gt_overlays"
    for d in (original_dir, heatmap_dir, overlay_dir, rawmap_dir, gt_mask_dir, gt_overlay_dir):
        d.mkdir(parents=True, exist_ok=True)

    dataset = predict_dataset_cls(path=args.input_path, image_size=(args.image_size, args.image_size))
    model = build_model(args.model, patchcore_cls, efficient_ad_cls)
    engine = engine_cls(accelerator=args.accelerator, devices=args.devices)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
        return_predictions=True,
    )
    flat_predictions = expand_predictions(predictions)
    if not flat_predictions:
        raise SystemExit("No predictions were returned by Engine.predict().")

    rows: list[dict[str, Any]] = []

    for idx, pred in enumerate(flat_predictions):
        image_path = Path(getattr(pred, "image_path", f"sample_{idx:04d}.png"))
        pred_label = int(np.asarray(tensor_to_numpy(getattr(pred, "pred_label", 0))).reshape(-1)[0])
        pred_score = float(np.asarray(tensor_to_numpy(getattr(pred, "pred_score", 0.0))).reshape(-1)[0])
        anomaly_map_like = getattr(pred, "anomaly_map", None)
        if anomaly_map_like is None:
            raise SystemExit("Prediction object does not contain anomaly_map.")
        anomaly_map = tensor_to_numpy(anomaly_map_like)

        image_like = getattr(pred, "image", None)
        if image_path.exists():
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
        elif image_like is not None:
            image_rgb = to_numpy_image(image_like)
        else:
            raise SystemExit(f"Could not load source image for prediction {idx}: {image_path}")

        gray_map = normalize_map(anomaly_map)
        heatmap_rgb = make_heatmap(gray_map, target_hw=image_rgb.shape[:2])
        overlay_rgb = overlay_heatmap(image_rgb, heatmap_rgb, alpha=0.45)
        gt_label = infer_ground_truth_from_path(image_path)
        gt_mask, gt_mask_source = load_gt_mask(
            dataset_root=args.dataset_root,
            category=args.category,
            test_type=args.test_type,
            image_path=image_path,
            image_hw=image_rgb.shape[:2],
            gt_label=gt_label,
        )
        gt_overlay_rgb = overlay_mask(image_rgb, gt_mask)

        stem = f"{idx:04d}_{image_path.stem}"
        original_path = original_dir / f"{stem}.png"
        heatmap_path = heatmap_dir / f"{stem}.png"
        overlay_path = overlay_dir / f"{stem}.png"
        rawmap_path = rawmap_dir / f"{stem}.png"
        gt_mask_path = gt_mask_dir / f"{stem}.png"
        gt_overlay_path = gt_overlay_dir / f"{stem}.png"

        save_image(original_path, image_rgb)
        save_image(heatmap_path, heatmap_rgb)
        save_image(overlay_path, overlay_rgb)
        save_image(gt_overlay_path, gt_overlay_rgb)
        Image.fromarray(gray_map).save(rawmap_path)
        Image.fromarray(gt_mask).save(gt_mask_path)

        row = {
            "index": idx,
            "image_path": str(image_path),
            "image_name": image_path.name,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "gt_label": gt_label,
            "gt_mask_source": str(gt_mask_source) if gt_mask_source else "",
            "original_rel": original_path.relative_to(args.output_dir).as_posix(),
            "gt_mask_rel": gt_mask_path.relative_to(args.output_dir).as_posix(),
            "gt_overlay_rel": gt_overlay_path.relative_to(args.output_dir).as_posix(),
            "heatmap_rel": heatmap_path.relative_to(args.output_dir).as_posix(),
            "overlay_rel": overlay_path.relative_to(args.output_dir).as_posix(),
            "raw_map_rel": rawmap_path.relative_to(args.output_dir).as_posix(),
        }
        rows.append(row)

    csv_path = args.output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    html_path = args.output_dir / "report.html"
    build_html_report(rows, html_path)

    metadata = {
        "checkpoint": str(ckpt_path),
        "input_path": str(args.input_path),
        "dataset_root": str(args.dataset_root),
        "category": args.category,
        "test_type": args.test_type,
        "model": args.model,
        "image_size": args.image_size,
        "num_predictions": len(rows),
        "csv": str(csv_path),
        "report_html": str(html_path),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[INFO] Inference done")
    print(f"[INFO] Checkpoint : {ckpt_path}")
    print(f"[INFO] CSV        : {csv_path}")
    print(f"[INFO] HTML       : {html_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
