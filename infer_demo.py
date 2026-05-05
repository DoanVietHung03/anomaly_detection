#!/usr/bin/env python3
"""Run inference for a trained anomaly detection model and build a demo report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
from PIL import Image

from train_demo import (
    DATASET_CHOICES,
    DEFAULT_FASTFLOW_BACKBONE,
    DEFAULT_FASTFLOW_FLOW_STEPS,
    DEFAULT_FASTFLOW_HIDDEN_RATIO,
    DEFAULT_PADIM_BACKBONE,
    DEFAULT_PADIM_LAYERS,
    DEFAULT_PADIM_N_FEATURES,
    DEFAULT_PATCHCORE_CORESET_RATIO,
    DEFAULT_PATCHCORE_LAYERS,
    DEFAULT_PATCHCORE_NUM_NEIGHBORS,
    DEFAULT_PATCHCORE_PRECISION,
    DEFAULT_TILING,
    build_model_from_args,
    build_tiling_callbacks,
    format_image_size,
    resolve_image_size,
    resolve_tiling,
    validate_fastflow_args,
    validate_padim_args,
    validate_patchcore_args,
)

DEFAULT_FIXED_ROI = (0.06, 0.10, 0.94, 0.90)
ROI_MODE_CHOICES = ("fixed-foreground", "fixed", "foreground", "off")
SCORE_AGGREGATION_CHOICES = (
    "model",
    "pixel-mean",
    "pixel-topk",
    "pixel-percentile",
    "blob-count",
    "blob-score",
    "max",
    "percentile",
    "topk-mean",
    "local-contrast",
    "local-ratio",
)


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
        choices=["patchcore", "padim", "fastflow"],
        help="Model architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./can"),
        help="Path to MVTec AD 2 root. Used only to add GT masks/overlays to reports.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="mvtec_ad2",
        help="Dataset format for GT mask lookup.",
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
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square PredictDataset image size. Overridden by --image-height/--image-width.",
    )
    parser.add_argument("--image-height", type=int, default=None, help="PredictDataset image height. Defaults to 384.")
    parser.add_argument("--image-width", type=int, default=None, help="PredictDataset image width. Defaults to 837.")
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=None,
        help="Folder with labelled good/bad images used to choose a calibrated threshold.",
    )
    parser.add_argument(
        "--calibration-threshold",
        type=float,
        default=None,
        help="Manual threshold to apply to prediction scores. Overrides --calibration-path.",
    )
    parser.add_argument(
        "--calibration-objective",
        choices=["balanced-f1", "f1"],
        default="balanced-f1",
        help="Threshold selection objective for labelled calibration images.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["raw", "anomalib"],
        default="raw",
        help="Use raw model scores for calibration, or Anomalib post-processed scores.",
    )
    parser.add_argument(
        "--roi-mode",
        choices=ROI_MODE_CHOICES,
        default=None,
        help="Restrict score aggregation. Defaults to foreground for PaDiM and fixed-foreground for PatchCore.",
    )
    parser.add_argument(
        "--fixed-roi",
        nargs=4,
        type=float,
        default=list(DEFAULT_FIXED_ROI),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Fixed ROI as normalized fractions of width/height. Default crops the can body and skips borders/glare edges.",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=SCORE_AGGREGATION_CHOICES,
        default=None,
        help="Image score aggregation. Defaults to pixel-percentile for PaDiM and pixel-mean for PatchCore.",
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
    parser.add_argument(
        "--score-blob-threshold",
        type=float,
        default=15.0,
        help="Anomaly-map threshold used to count hot connected components when --score-aggregation blob-*.",
    )
    parser.add_argument(
        "--score-blob-strong-threshold",
        type=float,
        default=18.0,
        help="Higher anomaly-map threshold used for the largest hot component term in blob-score.",
    )
    parser.add_argument(
        "--score-blob-min-area",
        type=int,
        default=1,
        help="Minimum connected-component area in anomaly-map pixels for blob-count/blob-score.",
    )
    parser.add_argument(
        "--score-blob-area-weight",
        type=float,
        default=5000.0,
        help="Weight applied to largest strong blob area fraction in blob-score.",
    )
    parser.add_argument(
        "--tiling",
        choices=["auto", "on", "off"],
        default=DEFAULT_TILING,
        help="Enable tiled PatchCore inference. off is the memory-safe default for wide can images.",
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
        help="PatchCore feature layers. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--patchcore-coreset-ratio",
        type=float,
        default=DEFAULT_PATCHCORE_CORESET_RATIO,
        help="PatchCore coreset sampling ratio. Must match the trained checkpoint.",
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
        help="PatchCore compute precision. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--padim-backbone",
        default=DEFAULT_PADIM_BACKBONE,
        help="PaDiM feature backbone. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--padim-layers",
        nargs="+",
        default=list(DEFAULT_PADIM_LAYERS),
        choices=["layer1", "layer2", "layer3", "layer4"],
        help="PaDiM feature layers. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--padim-n-features",
        type=int,
        default=DEFAULT_PADIM_N_FEATURES,
        help="PaDiM retained feature dimensions. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--fastflow-backbone",
        default=DEFAULT_FASTFLOW_BACKBONE,
        help="FastFlow feature backbone. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--fastflow-flow-steps",
        type=int,
        default=DEFAULT_FASTFLOW_FLOW_STEPS,
        help="FastFlow normalizing-flow steps. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--fastflow-conv3x3-only",
        action="store_true",
        help="Use only 3x3 convolutions in FastFlow coupling blocks. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--fastflow-hidden-ratio",
        type=float,
        default=DEFAULT_FASTFLOW_HIDDEN_RATIO,
        help="FastFlow hidden channel ratio. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--heatmap-normalization",
        choices=["global", "per-image"],
        default="global",
        help="Normalize heatmaps across the inference batch or independently per image.",
    )
    parser.add_argument("--accelerator", type=str, default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices", type=str, default="1", help="Lightning devices.")
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from anomalib.callbacks import TilerConfigurationCallback
        from anomalib.data import PredictDataset
        from anomalib.data.utils.tiler import ImageUpscaleMode
        from anomalib.engine import Engine
        from anomalib.models import Fastflow, Padim, Patchcore
    except Exception as exc:  # pragma: no cover - runtime safeguard
        raise SystemExit(
            "Failed to import Anomalib stack. Install dependencies first, for example:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc
    return PredictDataset, Engine, Patchcore, Padim, Fastflow, TilerConfigurationCallback, ImageUpscaleMode


def install_checkpoint_compatibility_aliases() -> None:
    """Allow old checkpoints to unpickle SafeMVTecAD2 saved from train_demo.py."""
    try:
        from train_demo import SafeMVTecAD2
    except Exception:
        return

    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "SafeMVTecAD2"):
        setattr(main_module, "SafeMVTecAD2", SafeMVTecAD2)


def find_checkpoint(results_dir: Path) -> Path | None:
    ckpts = sorted(results_dir.rglob("*.ckpt"))
    if not ckpts:
        return None
    preferred = [p for p in ckpts if "best" in p.name.lower() or "model" in p.name.lower()]
    return preferred[0] if preferred else ckpts[0]


def default_score_aggregation(model_name: str) -> str:
    if model_name == "padim":
        return "pixel-percentile"
    if model_name == "fastflow":
        return "model"
    return "pixel-mean"


def default_roi_mode(model_name: str) -> str:
    return "foreground" if model_name in {"padim", "fastflow"} else "fixed-foreground"


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

    for field in ("pred_score", "pred_label"):
        value = getattr(prediction, field, None)
        shape = getattr(value, "shape", None)
        if shape and len(shape) > 0:
            return int(shape[0])
    anomaly_map = getattr(prediction, "anomaly_map", None)
    shape = getattr(anomaly_map, "shape", None)
    if shape and len(shape) >= 3:
        return int(shape[0])
    image = getattr(prediction, "image", None)
    shape = getattr(image, "shape", None)
    if shape and len(shape) == 4:
        return int(shape[0])
    return 1


def is_batch_prediction(prediction: Any) -> bool:
    if isinstance(getattr(prediction, "image_path", None), (list, tuple)):
        return True
    for field in ("pred_score", "pred_label"):
        value = getattr(prediction, field, None)
        shape = getattr(value, "shape", None)
        if shape and len(shape) > 0:
            return True
    anomaly_map = getattr(prediction, "anomaly_map", None)
    shape = getattr(anomaly_map, "shape", None)
    if shape and len(shape) >= 3:
        return True
    image = getattr(prediction, "image", None)
    shape = getattr(image, "shape", None)
    if shape and len(shape) == 4:
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


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    arr = tensor_to_numpy(value)
    if arr.size == 0:
        return None
    return int(np.asarray(arr).reshape(-1)[0])


def scalar_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    arr = tensor_to_numpy(value)
    if arr.size == 0:
        return default
    return float(np.asarray(arr).reshape(-1)[0])


def configure_score_mode(model: Any, score_mode: str) -> None:
    if score_mode != "raw":
        return
    post_processor = getattr(model, "post_processor", None)
    if post_processor is None:
        return
    if hasattr(post_processor, "enable_normalization"):
        post_processor.enable_normalization = False
    if hasattr(post_processor, "enable_thresholding"):
        post_processor.enable_thresholding = False


def normalize_map(anomaly_map: np.ndarray, min_v: float | None = None, max_v: float | None = None) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32)
    min_v = float(anomaly_map.min()) if min_v is None else min_v
    max_v = float(anomaly_map.max()) if max_v is None else max_v
    if max_v - min_v < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.uint8)
    norm = (anomaly_map - min_v) / (max_v - min_v)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def foreground_mask_from_image(image_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Estimate the can foreground and resize it to the anomaly-map resolution."""
    if image_rgb.size == 0:
        return np.ones(target_hw, dtype=bool)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mask = ((saturation > 12) | (value < 245)).astype(np.uint8)

    kernel_size = max(7, int(round(min(image_rgb.shape[:2]) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    target_h, target_w = target_hw
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask.astype(bool)
    if mask_bool.mean() < 0.02:
        return np.ones(target_hw, dtype=bool)
    return mask_bool


def validate_fixed_roi(values: Iterable[float]) -> tuple[float, float, float, float]:
    roi = tuple(float(value) for value in values)
    if len(roi) != 4:
        raise SystemExit("--fixed-roi expects four values: X1 Y1 X2 Y2.")
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise SystemExit("--fixed-roi values must satisfy 0 <= X1 < X2 <= 1 and 0 <= Y1 < Y2 <= 1.")
    return roi


def fixed_roi_box(shape_hw: tuple[int, int], fixed_roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    height, width = shape_hw
    x1, y1, x2, y2 = fixed_roi
    left = int(np.floor(x1 * width))
    top = int(np.floor(y1 * height))
    right = int(np.ceil(x2 * width))
    bottom = int(np.ceil(y2 * height))
    left = min(max(left, 0), max(width - 1, 0))
    top = min(max(top, 0), max(height - 1, 0))
    right = min(max(right, left + 1), width)
    bottom = min(max(bottom, top + 1), height)
    return top, bottom, left, right


def fixed_roi_mask(target_hw: tuple[int, int], fixed_roi: tuple[float, float, float, float]) -> np.ndarray:
    mask = np.zeros(target_hw, dtype=bool)
    top, bottom, left, right = fixed_roi_box(target_hw, fixed_roi)
    mask[top:bottom, left:right] = True
    return mask


def crop_fixed_roi(image_rgb: np.ndarray, fixed_roi: tuple[float, float, float, float]) -> np.ndarray:
    top, bottom, left, right = fixed_roi_box(image_rgb.shape[:2], fixed_roi)
    return image_rgb[top:bottom, left:right].copy()


def score_mask_for_image(
    image_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    roi_mode: str,
    fixed_roi: tuple[float, float, float, float],
) -> np.ndarray:
    if roi_mode == "off":
        return np.ones(anomaly_map.shape[:2], dtype=bool)
    if roi_mode == "foreground":
        return foreground_mask_from_image(image_rgb, anomaly_map.shape[:2])

    fixed_mask = fixed_roi_mask(anomaly_map.shape[:2], fixed_roi)
    if roi_mode == "fixed":
        return fixed_mask

    foreground_mask = foreground_mask_from_image(image_rgb, anomaly_map.shape[:2])
    combined_mask = fixed_mask & foreground_mask
    min_pixels = max(8, int(round(float(fixed_mask.sum()) * 0.10)))
    if int(combined_mask.sum()) < min_pixels:
        return fixed_mask
    return combined_mask


def aggregate_anomaly_score(
    anomaly_map: np.ndarray,
    *,
    model_score: float,
    roi_mask: np.ndarray,
    aggregation: str,
    percentile: float,
    topk_percent: float,
    local_sigma: float,
    blob_threshold: float,
    blob_strong_threshold: float,
    blob_min_area: int,
    blob_area_weight: float,
) -> float:
    if aggregation == "model":
        return float(model_score)
    if aggregation == "pixel-mean":
        values = anomaly_map.astype(np.float32, copy=False)[roi_mask]
        if values.size == 0:
            values = anomaly_map.astype(np.float32, copy=False).reshape(-1)
        return float(values.mean())
    if aggregation == "pixel-topk":
        aggregation = "topk-mean"
    elif aggregation == "pixel-percentile":
        aggregation = "percentile"

    score_map = anomaly_map.astype(np.float32, copy=False)
    if aggregation in {"blob-count", "blob-score"}:
        return aggregate_blob_score(
            score_map,
            roi_mask=roi_mask,
            aggregation=aggregation,
            threshold=blob_threshold,
            strong_threshold=blob_strong_threshold,
            min_area=blob_min_area,
            area_weight=blob_area_weight,
        )

    if aggregation == "local-contrast":
        baseline = cv2.GaussianBlur(score_map, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
        score_map = score_map - baseline
        aggregation = "topk-mean"
    elif aggregation == "local-ratio":
        baseline = cv2.GaussianBlur(score_map, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
        score_map = score_map / (baseline + 1e-6)
        aggregation = "topk-mean"

    values = score_map[roi_mask]
    if values.size == 0:
        values = score_map.reshape(-1)

    values = values.astype(np.float32, copy=False)
    if aggregation == "max":
        return float(values.max())
    if aggregation == "percentile":
        return float(np.percentile(values, percentile))
    if aggregation == "topk-mean":
        k = max(1, int(np.ceil(values.size * (topk_percent / 100.0))))
        k = min(k, values.size)
        top_values = np.partition(values, values.size - k)[-k:]
        return float(top_values.mean())
    raise ValueError(f"Unknown score aggregation: {aggregation}")


def connected_component_stats(mask: np.ndarray, min_area: int) -> tuple[int, int]:
    if not np.any(mask):
        return 0, 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [int(area) for area in stats[1:, cv2.CC_STAT_AREA] if int(area) >= min_area]
    if not areas:
        return 0, 0
    return len(areas), max(areas)


def aggregate_blob_score(
    score_map: np.ndarray,
    *,
    roi_mask: np.ndarray,
    aggregation: str,
    threshold: float,
    strong_threshold: float,
    min_area: int,
    area_weight: float,
) -> float:
    min_area = max(1, int(min_area))
    roi_pixels = max(1, int(roi_mask.sum()))
    hot_mask = (score_map >= float(threshold)) & roi_mask
    component_count, _ = connected_component_stats(hot_mask, min_area)
    if aggregation == "blob-count":
        return float(component_count)

    strong_mask = (score_map >= float(strong_threshold)) & roi_mask
    _, largest_strong_area = connected_component_stats(strong_mask, min_area)
    largest_strong_fraction = float(largest_strong_area) / float(roi_pixels)
    return float(component_count + float(area_weight) * largest_strong_fraction)


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


def find_gt_mask_path(
    dataset_root: Path,
    category: str,
    test_type: str,
    image_path: Path,
    gt_label: str | None,
    dataset: str = "mvtec_ad2",
) -> Path | None:
    if gt_label != "bad":
        return None

    source_name = source_image_name_from_demo_name(image_path)
    source_stem = Path(source_name).stem
    if dataset == "visa":
        for candidate in (
            dataset_root / category / "Data" / "Masks" / "Anomaly" / f"{source_stem}.png",
            dataset_root / "visa_pytorch" / category / "ground_truth" / "bad" / f"{source_stem}.png",
        ):
            if candidate.exists():
                return candidate
        return None

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
    dataset: str = "mvtec_ad2",
) -> tuple[np.ndarray, Path | None]:
    target_h, target_w = image_hw
    mask_path = find_gt_mask_path(dataset_root, category, test_type, image_path, gt_label, dataset)
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


def threshold_candidates(scores: list[float]) -> list[float]:
    unique_scores = sorted(set(scores))
    if not unique_scores:
        return []
    candidates = set(unique_scores)
    for low, high in zip(unique_scores, unique_scores[1:]):
        candidates.add((low + high) / 2.0)
    span = max(unique_scores) - min(unique_scores)
    epsilon = max(1e-6, span * 1e-6)
    candidates.add(min(unique_scores) - epsilon)
    candidates.add(max(unique_scores) + epsilon)
    return sorted(candidates, reverse=True)


def best_f1_threshold(labelled_scores: list[tuple[float, int]], objective: str = "balanced-f1") -> dict[str, Any] | None:
    if not labelled_scores:
        return None
    positives = sum(label for _, label in labelled_scores)
    negatives = len(labelled_scores) - positives
    if positives == 0 or negatives == 0:
        return None

    best: dict[str, Any] | None = None
    for threshold in threshold_candidates([score for score, _ in labelled_scores]):
        tp = fp = tn = fn = 0
        for score, label in labelled_scores:
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
        balanced_accuracy = (recall + specificity) / 2.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        candidate = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if best is None:
            best = candidate
            continue
        if objective == "balanced-f1":
            best_key = (
                float(best["balanced_accuracy"]),
                float(best["f1"]),
                float(best["accuracy"]),
                float(best["precision"]),
                float(best["threshold"]),
            )
            candidate_key = (balanced_accuracy, f1, accuracy, precision, threshold)
        else:
            best_key = (
                float(best["f1"]),
                float(best["balanced_accuracy"]),
                float(best["accuracy"]),
                float(best["precision"]),
                float(best["threshold"]),
            )
            candidate_key = (f1, balanced_accuracy, accuracy, precision, threshold)
        if candidate_key > best_key:
            best = candidate
    if best is not None:
        best["objective"] = objective
    return best


def calibration_scores(predictions: list[dict[str, Any]]) -> list[tuple[float, int]]:
    labelled: list[tuple[float, int]] = []
    for item in predictions:
        label = infer_ground_truth_from_path(item["image_path"])
        if label == "bad":
            labelled.append((float(item["pred_score"]), 1))
        elif label == "good":
            labelled.append((float(item["pred_score"]), 0))
    return labelled


def score_auc(labelled_scores: list[tuple[float, int]]) -> float | None:
    positives = [score for score, label in labelled_scores if label == 1]
    negatives = [score for score, label in labelled_scores if label == 0]
    if not positives or not negatives:
        return None
    wins = ties = total = 0
    for positive_score in positives:
        for negative_score in negatives:
            total += 1
            if positive_score > negative_score:
                wins += 1
            elif positive_score == negative_score:
                ties += 1
    return (wins + 0.5 * ties) / total if total else None


def prediction_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    labelled_scores: list[tuple[float, int]] = []
    score_groups: dict[str, list[float]] = {"good": [], "bad": []}

    for row in rows:
        gt_label = row.get("gt_label")
        pred_label = row.get("pred_label_name")
        if gt_label not in {"good", "bad"}:
            continue
        score = float(row["pred_score"])
        score_groups[str(gt_label)].append(score)
        labelled_scores.append((score, 1 if gt_label == "bad" else 0))
        if gt_label == "bad" and pred_label == "bad":
            confusion["tp"] += 1
        elif gt_label == "good" and pred_label == "bad":
            confusion["fp"] += 1
        elif gt_label == "good" and pred_label == "good":
            confusion["tn"] += 1
        elif gt_label == "bad" and pred_label == "good":
            confusion["fn"] += 1

    score_summary: dict[str, dict[str, float | int | None]] = {}
    for label, scores in score_groups.items():
        score_summary[label] = {
            "count": len(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": float(np.mean(scores)) if scores else None,
        }

    total = sum(confusion.values())
    accuracy = (confusion["tp"] + confusion["tn"]) / total if total else None
    return {
        "confusion": confusion,
        "accuracy": accuracy,
        "score_auc": score_auc(labelled_scores),
        "score_summary": score_summary,
    }


def predict_path(
    *,
    path: Path,
    predict_dataset_cls: Any,
    engine: Any,
    model: Any,
    ckpt_path: Path,
    image_size: tuple[int, int],
    roi_mode: str,
    fixed_roi: tuple[float, float, float, float],
    score_aggregation: str,
    score_percentile: float,
    score_topk_percent: float,
    score_local_sigma: float,
    score_blob_threshold: float,
    score_blob_strong_threshold: float,
    score_blob_min_area: int,
    score_blob_area_weight: float,
) -> list[dict[str, Any]]:
    dataset = predict_dataset_cls(path=path, image_size=image_size)
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
        return_predictions=True,
    )
    flat_predictions = expand_predictions(predictions)
    if not flat_predictions:
        raise SystemExit(f"No predictions were returned for: {path}")

    prepared: list[dict[str, Any]] = []
    for idx, pred in enumerate(flat_predictions):
        image_path = Path(getattr(pred, "image_path", f"sample_{idx:04d}.png"))
        anomaly_map_like = getattr(pred, "anomaly_map", None)
        if anomaly_map_like is None:
            raise SystemExit("Prediction object does not contain anomaly_map.")

        image_like = getattr(pred, "image", None)
        if image_path.exists():
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
        elif image_like is not None:
            image_rgb = to_numpy_image(image_like)
        else:
            raise SystemExit(f"Could not load source image for prediction {idx}: {image_path}")

        model_pred_score = scalar_float(getattr(pred, "pred_score", None))
        pred_label = optional_int(getattr(pred, "pred_label", None))
        anomaly_map = tensor_to_numpy(anomaly_map_like).astype(np.float32)
        roi_mask = score_mask_for_image(image_rgb, anomaly_map, roi_mode, fixed_roi)
        pred_score = aggregate_anomaly_score(
            anomaly_map,
            model_score=model_pred_score,
            roi_mask=roi_mask,
            aggregation=score_aggregation,
            percentile=score_percentile,
            topk_percent=score_topk_percent,
            local_sigma=score_local_sigma,
            blob_threshold=score_blob_threshold,
            blob_strong_threshold=score_blob_strong_threshold,
            blob_min_area=score_blob_min_area,
            blob_area_weight=score_blob_area_weight,
        )
        prepared.append(
            {
                "index": idx,
                "image_path": image_path,
                "pred_label": pred_label,
                "pred_score": pred_score,
                "raw_pred_score": pred_score,
                "model_pred_score": model_pred_score,
                "anomaly_map": anomaly_map,
                "anomaly_map_min": float(anomaly_map.min()),
                "anomaly_map_max": float(anomaly_map.max()),
                "roi_mask": roi_mask,
                "roi_coverage": float(roi_mask.mean()),
                "fixed_roi_box": fixed_roi_box(image_rgb.shape[:2], fixed_roi),
                "score_aggregation": score_aggregation,
                "score_local_sigma": score_local_sigma,
                "score_blob_threshold": score_blob_threshold,
                "score_blob_strong_threshold": score_blob_strong_threshold,
                "score_blob_min_area": score_blob_min_area,
                "score_blob_area_weight": score_blob_area_weight,
                "roi_mode": roi_mode,
                "image_rgb": image_rgb,
            },
        )
    return prepared


def build_html_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <div class=\"card\">
              <div class=\"meta\">
                <div><b>File:</b> {html.escape(row['image_name'])}</div>
                <div><b>Pred:</b> {html.escape(str(row.get('pred_label_name', row['pred_label'])))}</div>
                <div><b>Score:</b> {row['pred_score']:.6f}</div>
                <div><b>Model score:</b> {float(row.get('model_pred_score', row['pred_score'])):.6f}</div>
                <div><b>Aggregation:</b> {html.escape(str(row.get('score_aggregation', 'model')))} / {html.escape(str(row.get('roi_mode', 'off')))}</div>
                <div><b>Threshold:</b> {html.escape(str(row.get('calibrated_threshold') or 'model/default'))}</div>
                <div><b>GT guess:</b> {html.escape(str(row.get('gt_label', 'unknown')))}</div>
              </div>
              <div class=\"grid\">
                <div><p>Original</p><img src=\"{html.escape(row['original_rel'])}\"></div>
                <div><p>Fixed ROI Crop</p><img src=\"{html.escape(row['roi_crop_rel'])}\"></div>
                <div><p>GT Mask</p><img src=\"{html.escape(row['gt_mask_rel'])}\"></div>
                <div><p>GT Overlay</p><img src=\"{html.escape(row['gt_overlay_rel'])}\"></div>
                <div><p>ROI Mask</p><img src=\"{html.escape(row['roi_mask_rel'])}\"></div>
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
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
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
    (
        predict_dataset_cls,
        engine_cls,
        patchcore_cls,
        padim_cls,
        fastflow_cls,
        tiler_callback_cls,
        upscale_mode_cls,
    ) = import_dependencies()
    install_checkpoint_compatibility_aliases()
    image_size = resolve_image_size(args)
    if args.score_aggregation is None:
        args.score_aggregation = default_score_aggregation(args.model)
    args.roi_mode = args.roi_mode or default_roi_mode(args.model)
    fixed_roi = validate_fixed_roi(args.fixed_roi)
    tiling_config = resolve_tiling(args.model, args.tiling, args.tile_size, args.tile_stride, image_size)

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
    if args.calibration_path is not None and not args.calibration_path.exists():
        raise SystemExit(f"Calibration path not found: {args.calibration_path}")
    if args.calibration_threshold is not None and not np.isfinite(args.calibration_threshold):
        raise SystemExit("--calibration-threshold must be finite.")
    if not (0.0 < args.score_percentile <= 100.0):
        raise SystemExit("--score-percentile must be in the range (0, 100].")
    if not (0.0 < args.score_topk_percent <= 100.0):
        raise SystemExit("--score-topk-percent must be in the range (0, 100].")
    if args.score_local_sigma <= 0.0:
        raise SystemExit("--score-local-sigma must be positive.")
    if not np.isfinite(args.score_blob_threshold):
        raise SystemExit("--score-blob-threshold must be finite.")
    if not np.isfinite(args.score_blob_strong_threshold):
        raise SystemExit("--score-blob-strong-threshold must be finite.")
    if args.score_blob_min_area <= 0:
        raise SystemExit("--score-blob-min-area must be a positive integer.")
    if args.score_blob_area_weight < 0.0:
        raise SystemExit("--score-blob-area-weight must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = args.output_dir / "originals"
    heatmap_dir = args.output_dir / "heatmaps"
    overlay_dir = args.output_dir / "overlays"
    rawmap_dir = args.output_dir / "raw_maps"
    raw_float_dir = args.output_dir / "raw_float_maps"
    roi_mask_dir = args.output_dir / "roi_masks"
    roi_crop_dir = args.output_dir / "roi_crops"
    gt_mask_dir = args.output_dir / "gt_masks"
    gt_overlay_dir = args.output_dir / "gt_overlays"
    for d in (
        original_dir,
        heatmap_dir,
        overlay_dir,
        rawmap_dir,
        raw_float_dir,
        roi_mask_dir,
        roi_crop_dir,
        gt_mask_dir,
        gt_overlay_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    model = build_model_from_args(args, patchcore_cls, padim_cls, fastflow_cls, image_size)
    configure_score_mode(model, args.score_mode)
    callbacks = build_tiling_callbacks(tiling_config, tiler_callback_cls, upscale_mode_cls)
    precision_flag = "16-true" if (args.model == "patchcore" and getattr(args, "patchcore_precision", "") == "float16") else "32"
    engine = engine_cls(
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision_flag,
    )

    calibration_result: dict[str, Any] | None = None
    calibration_score_rows: list[dict[str, Any]] | None = None
    calibrated_threshold = args.calibration_threshold
    if calibrated_threshold is None and args.calibration_path is not None:
        calibration_predictions = predict_path(
            path=args.calibration_path,
            predict_dataset_cls=predict_dataset_cls,
            engine=engine,
            model=model,
            ckpt_path=ckpt_path,
            image_size=image_size,
            roi_mode=args.roi_mode,
            fixed_roi=fixed_roi,
            score_aggregation=args.score_aggregation,
            score_percentile=args.score_percentile,
            score_topk_percent=args.score_topk_percent,
            score_local_sigma=args.score_local_sigma,
            score_blob_threshold=args.score_blob_threshold,
            score_blob_strong_threshold=args.score_blob_strong_threshold,
            score_blob_min_area=args.score_blob_min_area,
            score_blob_area_weight=args.score_blob_area_weight,
        )
        calibration_score_rows = [
            {
                "image_path": str(item["image_path"]),
                "gt_label": infer_ground_truth_from_path(item["image_path"]),
                "pred_score": float(item["pred_score"]),
                "model_pred_score": float(item["model_pred_score"]),
                "roi_mode": str(item["roi_mode"]),
                "roi_coverage": float(item["roi_coverage"]),
                "score_aggregation": str(item["score_aggregation"]),
                "score_blob_threshold": float(item["score_blob_threshold"]),
                "score_blob_strong_threshold": float(item["score_blob_strong_threshold"]),
                "score_blob_min_area": int(item["score_blob_min_area"]),
                "score_blob_area_weight": float(item["score_blob_area_weight"]),
            }
            for item in calibration_predictions
        ]
        calibration_result = best_f1_threshold(calibration_scores(calibration_predictions), args.calibration_objective)
        if calibration_result is not None:
            calibrated_threshold = float(calibration_result["threshold"])
            if float(calibration_result.get("specificity", 0.0)) == 0.0:
                print("[WARN] Calibration could not keep any labelled good images below threshold; predictions may over-call bad.")
        else:
            print("[WARN] Calibration path did not contain both good and bad labelled samples; using model labels if available.")
    elif calibrated_threshold is not None:
        calibration_result = {"threshold": calibrated_threshold, "source": "manual"}

    prepared_predictions = predict_path(
        path=args.input_path,
        predict_dataset_cls=predict_dataset_cls,
        engine=engine,
        model=model,
        ckpt_path=ckpt_path,
        image_size=image_size,
        roi_mode=args.roi_mode,
        fixed_roi=fixed_roi,
        score_aggregation=args.score_aggregation,
            score_percentile=args.score_percentile,
            score_topk_percent=args.score_topk_percent,
            score_local_sigma=args.score_local_sigma,
            score_blob_threshold=args.score_blob_threshold,
            score_blob_strong_threshold=args.score_blob_strong_threshold,
            score_blob_min_area=args.score_blob_min_area,
            score_blob_area_weight=args.score_blob_area_weight,
        )

    map_mins = [float(item["anomaly_map_min"]) for item in prepared_predictions]
    map_maxs = [float(item["anomaly_map_max"]) for item in prepared_predictions]

    global_map_min = min(map_mins) if map_mins else None
    global_map_max = max(map_maxs) if map_maxs else None
    rows: list[dict[str, Any]] = []

    for item in prepared_predictions:
        idx = int(item["index"])
        image_path = item["image_path"]
        image_rgb = item["image_rgb"]
        model_pred_label = item["pred_label"]
        pred_score = float(item["pred_score"])
        anomaly_map = item["anomaly_map"]
        roi_mask = item["roi_mask"]

        if args.heatmap_normalization == "global":
            gray_map = normalize_map(anomaly_map, global_map_min, global_map_max)
        else:
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
            dataset=args.dataset,
        )
        gt_overlay_rgb = overlay_mask(image_rgb, gt_mask)

        stem = f"{idx:04d}_{image_path.stem}"
        original_path = original_dir / f"{stem}.png"
        heatmap_path = heatmap_dir / f"{stem}.png"
        overlay_path = overlay_dir / f"{stem}.png"
        rawmap_path = rawmap_dir / f"{stem}.png"
        raw_float_path = raw_float_dir / f"{stem}.npy"
        roi_mask_path = roi_mask_dir / f"{stem}.png"
        roi_crop_path = roi_crop_dir / f"{stem}.png"
        gt_mask_path = gt_mask_dir / f"{stem}.png"
        gt_overlay_path = gt_overlay_dir / f"{stem}.png"

        save_image(original_path, image_rgb)
        save_image(roi_crop_path, crop_fixed_roi(image_rgb, fixed_roi))
        save_image(heatmap_path, heatmap_rgb)
        save_image(overlay_path, overlay_rgb)
        save_image(gt_overlay_path, gt_overlay_rgb)
        Image.fromarray(gray_map).save(rawmap_path)
        np.save(raw_float_path, anomaly_map.astype(np.float32, copy=False))
        Image.fromarray((roi_mask.astype(np.uint8) * 255)).save(roi_mask_path)
        Image.fromarray(gt_mask).save(gt_mask_path)

        if calibrated_threshold is not None:
            pred_label = 1 if pred_score >= calibrated_threshold else 0
        elif model_pred_label is not None:
            pred_label = int(model_pred_label)
        else:
            pred_label = 1 if pred_score >= 0.5 else 0

        pred_label_name = "bad" if pred_label == 1 else "good"
        correct = pred_label_name == gt_label if gt_label in {"good", "bad"} else ""
        row = {
            "index": idx,
            "image_path": str(image_path),
            "image_name": image_path.name,
            "pred_label": pred_label,
            "pred_label_name": pred_label_name,
            "anomalib_pred_label": "" if model_pred_label is None else int(model_pred_label),
            "pred_score": pred_score,
            "raw_pred_score": item["raw_pred_score"],
            "model_pred_score": item["model_pred_score"],
            "calibrated_threshold": "" if calibrated_threshold is None else calibrated_threshold,
            "score_mode": args.score_mode,
            "score_aggregation": item["score_aggregation"],
            "score_local_sigma": item["score_local_sigma"],
            "score_blob_threshold": item["score_blob_threshold"],
            "score_blob_strong_threshold": item["score_blob_strong_threshold"],
            "score_blob_min_area": item["score_blob_min_area"],
            "score_blob_area_weight": item["score_blob_area_weight"],
            "roi_mode": item["roi_mode"],
            "roi_coverage": item["roi_coverage"],
            "fixed_roi": ",".join(f"{value:.4f}" for value in fixed_roi),
            "fixed_roi_box": ",".join(str(value) for value in item["fixed_roi_box"]),
            "gt_label": gt_label,
            "correct": correct,
            "anomaly_map_min": item["anomaly_map_min"],
            "anomaly_map_max": item["anomaly_map_max"],
            "gt_mask_source": str(gt_mask_source) if gt_mask_source else "",
            "original_rel": original_path.relative_to(args.output_dir).as_posix(),
            "gt_mask_rel": gt_mask_path.relative_to(args.output_dir).as_posix(),
            "gt_overlay_rel": gt_overlay_path.relative_to(args.output_dir).as_posix(),
            "roi_crop_rel": roi_crop_path.relative_to(args.output_dir).as_posix(),
            "heatmap_rel": heatmap_path.relative_to(args.output_dir).as_posix(),
            "overlay_rel": overlay_path.relative_to(args.output_dir).as_posix(),
            "raw_map_rel": rawmap_path.relative_to(args.output_dir).as_posix(),
            "raw_float_map_rel": raw_float_path.relative_to(args.output_dir).as_posix(),
            "roi_mask_rel": roi_mask_path.relative_to(args.output_dir).as_posix(),
        }
        rows.append(row)

    diagnostics = prediction_diagnostics(rows)
    score_auc_value = diagnostics.get("score_auc")
    if score_auc_value is not None and float(score_auc_value) < 0.6:
        print(
            "[WARN] Prediction scores have weak good/bad separation "
            f"(AUC={float(score_auc_value):.3f}). Try a larger aspect-preserving image size or a different model.",
        )

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
        "dataset": args.dataset,
        "category": args.category,
        "test_type": args.test_type,
        "model": args.model,
        "image_size": format_image_size(image_size),
        "image_height": image_size[0],
        "image_width": image_size[1],
        "tiling": tiling_config,
        "patchcore_layers": list(validate_patchcore_args(args)) if args.model == "patchcore" else None,
        "patchcore_coreset_ratio": args.patchcore_coreset_ratio if args.model == "patchcore" else None,
        "patchcore_num_neighbors": args.patchcore_num_neighbors if args.model == "patchcore" else None,
        "patchcore_precision": args.patchcore_precision if args.model == "patchcore" else None,
        "padim_backbone": args.padim_backbone if args.model == "padim" else None,
        "padim_layers": list(validate_padim_args(args)) if args.model == "padim" else None,
        "padim_n_features": args.padim_n_features if args.model == "padim" else None,
        "fastflow_backbone": args.fastflow_backbone if args.model == "fastflow" else None,
        "fastflow_flow_steps": args.fastflow_flow_steps if args.model == "fastflow" else None,
        "fastflow_conv3x3_only": args.fastflow_conv3x3_only if args.model == "fastflow" else None,
        "fastflow_hidden_ratio": args.fastflow_hidden_ratio if args.model == "fastflow" else None,
        "score_mode": args.score_mode,
        "score_aggregation": args.score_aggregation,
        "score_percentile": args.score_percentile,
        "score_topk_percent": args.score_topk_percent,
        "score_local_sigma": args.score_local_sigma,
        "score_blob_threshold": args.score_blob_threshold,
        "score_blob_strong_threshold": args.score_blob_strong_threshold,
        "score_blob_min_area": args.score_blob_min_area,
        "score_blob_area_weight": args.score_blob_area_weight,
        "roi_mode": args.roi_mode,
        "fixed_roi": list(fixed_roi),
        "calibration_path": str(args.calibration_path) if args.calibration_path else None,
        "calibration_objective": args.calibration_objective,
        "calibrated_threshold": calibrated_threshold,
        "calibration_result": calibration_result,
        "calibration_scores": calibration_score_rows,
        "heatmap_normalization": args.heatmap_normalization,
        "global_anomaly_map_min": global_map_min,
        "global_anomaly_map_max": global_map_max,
        "prediction_diagnostics": diagnostics,
        "num_predictions": len(rows),
        "csv": str(csv_path),
        "report_html": str(html_path),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[INFO] Inference done")
    print(f"[INFO] Checkpoint : {ckpt_path}")
    print(f"[INFO] Image size : {format_image_size(image_size)}")
    print(f"[INFO] Tiling     : {tiling_config}")
    if args.model == "patchcore":
        print(f"[INFO] Layers     : {','.join(validate_patchcore_args(args))}")
        print(f"[INFO] Coreset    : {args.patchcore_coreset_ratio}")
        print(f"[INFO] Precision  : {args.patchcore_precision}")
    if args.model == "padim":
        print(f"[INFO] Backbone   : {args.padim_backbone}")
        print(f"[INFO] Layers     : {','.join(validate_padim_args(args))}")
        print(f"[INFO] Features   : {args.padim_n_features}")
    if args.model == "fastflow":
        validate_fastflow_args(args)
        print(f"[INFO] Backbone   : {args.fastflow_backbone}")
        print(f"[INFO] Flow steps : {args.fastflow_flow_steps}")
        print(f"[INFO] Conv3x3    : {args.fastflow_conv3x3_only}")
        print(f"[INFO] Hidden     : {args.fastflow_hidden_ratio}")
    print(f"[INFO] Score mode : {args.score_mode}")
    print(f"[INFO] Aggregator : {args.score_aggregation} ({args.roi_mode})")
    if args.score_aggregation in {"blob-count", "blob-score"}:
        print(
            "[INFO] Blob score : "
            f"thr={args.score_blob_threshold}, strong={args.score_blob_strong_threshold}, "
            f"min_area={args.score_blob_min_area}, area_weight={args.score_blob_area_weight}",
        )
    print(f"[INFO] Fixed ROI  : {','.join(f'{value:.4f}' for value in fixed_roi)}")
    print(f"[INFO] Threshold  : {calibrated_threshold if calibrated_threshold is not None else 'model/default'}")
    if diagnostics["accuracy"] is not None:
        print(f"[INFO] Accuracy   : {float(diagnostics['accuracy']):.4f}")
    if diagnostics["score_auc"] is not None:
        print(f"[INFO] Score AUC  : {float(diagnostics['score_auc']):.4f}")
    print(f"[INFO] CSV        : {csv_path}")
    print(f"[INFO] HTML       : {html_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
