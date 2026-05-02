#!/usr/bin/env python3
"""Build a static dashboard from existing training and inference artifacts.

The dashboard intentionally does not require a new training run. It reads:
- runs_*/train_summary_*.json for final test metrics
- demo_outputs_*/predictions.csv for score distributions and confusion matrices
- hybrid_outputs_*/summary.json + test_predictions.csv for hybrid PatchCore/U-Net runs
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


METRIC_KEYS = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")
TRAIN_METRIC_KEYS = (
    "train_loss",
    "train_st",
    "train_ae",
    "train_stae",
    "val_pixel_dice",
    "val_image_f1",
    "val_image_acc",
)
MODEL_LABELS = {
    "patchcore": "PatchCore",
    "padim": "PaDiM",
    "fastflow": "FastFlow",
    "hybrid_patchcore_unet": "Hybrid PatchCore + U-Net",
}
MODEL_COLORS = ("#0f766e", "#c2410c", "#2563eb", "#7c2d12", "#4338ca", "#15803d")
GOOD_COLOR = "#0f766e"
BAD_COLOR = "#c2410c"
UNKNOWN_COLOR = "#64748b"


@dataclass
class PredictionRow:
    image_name: str
    gt_label: str | None
    pred_label: int | None
    pred_score: float
    raw_pred_score: float | None = None
    model_pred_score: float | None = None
    calibrated_threshold: float | None = None
    score_mode: str | None = None
    score_aggregation: str | None = None
    score_local_sigma: float | None = None
    roi_mode: str | None = None
    roi_coverage: float | None = None
    original_rel: str | None = None
    gt_mask_rel: str | None = None
    gt_overlay_rel: str | None = None
    heatmap_rel: str | None = None
    overlay_rel: str | None = None
    raw_map_rel: str | None = None
    raw_float_map_rel: str | None = None
    roi_mask_rel: str | None = None
    pred_mask_rel: str | None = None
    pred_overlay_rel: str | None = None
    pixel_dice: float | None = None
    pixel_iou: float | None = None
    pred_area_fraction: float | None = None
    largest_blob_fraction: float | None = None


@dataclass
class ModelArtifacts:
    model: str
    category: str | None = None
    summary_path: Path | None = None
    predictions_path: Path | None = None
    metrics_csv_path: Path | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    training_metrics: list[dict[str, float]] = field(default_factory=list)
    predictions: list[PredictionRow] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a static HTML dashboard from existing artifacts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing runs_*, demo_outputs_*, or hybrid_outputs_* directories.",
    )
    parser.add_argument("--runs-glob", default="runs_*", help="Glob for directories containing train summaries.")
    parser.add_argument("--demo-glob", default="demo_outputs_*", help="Glob for directories containing predictions.csv.")
    parser.add_argument("--hybrid-glob", default="**/hybrid_outputs_*", help="Glob for hybrid output directories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dashboard_outputs"),
        help="Directory where the dashboard HTML will be written.",
    )
    parser.add_argument("--output-name", default="artifact_dashboard.html", help="Dashboard HTML filename.")
    parser.add_argument(
        "--max-gallery-items",
        type=int,
        default=48,
        help="Maximum heatmap gallery rows per model. Use 0 or a negative value to include all rows.",
    )
    return parser.parse_args()


def project_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def normalize_model_name(model: str | None) -> str:
    if not model:
        return "unknown"
    return model.strip().lower().replace("-", "_")


def display_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def extract_metrics(summary: dict[str, Any]) -> dict[str, float]:
    if summary.get("model") == "hybrid_patchcore_unet" or "test" in summary:
        test = summary.get("test") if isinstance(summary.get("test"), dict) else {}
        image = test.get("image") if isinstance(test.get("image"), dict) else {}
        metrics: dict[str, float] = {}
        image_auc = safe_float(image.get("auc"))
        image_f1 = safe_float(image.get("f1"))
        pixel_dice = safe_float(test.get("pixel_dice_global"))
        if image_auc is not None:
            metrics["image_AUROC"] = image_auc
        if image_f1 is not None:
            metrics["image_F1Score"] = image_f1
        if pixel_dice is not None:
            metrics["pixel_F1Score"] = pixel_dice
        return metrics

    test_results = summary.get("test_results")
    if isinstance(test_results, list) and test_results and isinstance(test_results[0], dict):
        source = test_results[0]
    elif isinstance(test_results, dict):
        source = test_results
    else:
        source = {}

    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = safe_float(source.get(key))
        if value is not None:
            metrics[key] = value
    return metrics


def read_training_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            parsed: dict[str, float] = {}
            for key, value in raw.items():
                number = safe_float(value)
                if key and number is not None:
                    parsed[key] = number
            if parsed:
                rows.append(parsed)
    return rows


def resolve_summary_metrics_csv(summary: dict[str, Any], project_root: Path) -> Path | None:
    metrics_csv = summary.get("metrics_csv")
    if not metrics_csv:
        return None
    path = Path(str(metrics_csv))
    if not path.is_absolute():
        path = project_root / path
    return path if path.exists() else None


def find_metrics_csv(record: ModelArtifacts, project_root: Path) -> Path | None:
    path = resolve_summary_metrics_csv(record.summary, project_root)
    if path is not None:
        return path
    if record.summary_path is None:
        return None

    candidates = sorted(
        record.summary_path.parent.rglob("metrics.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    model_token = record.model.lower()
    preferred = [candidate for candidate in candidates if model_token in str(candidate).lower()]
    return preferred[0] if preferred else candidates[0]


def load_train_summaries(project_root: Path, runs_glob: str) -> dict[str, ModelArtifacts]:
    artifacts: dict[str, ModelArtifacts] = {}
    summary_paths: list[Path] = []
    for directory in sorted(project_root.glob(runs_glob)):
        if directory.is_dir():
            summary_paths.extend(sorted(directory.glob("train_summary_*.json")))

    for path in summary_paths:
        summary = read_json(path)
        stem_parts = path.stem.split("_")
        fallback_model = stem_parts[2] if len(stem_parts) >= 3 else None
        model = normalize_model_name(summary.get("model") or fallback_model)
        record = artifacts.setdefault(model, ModelArtifacts(model=model))
        record.summary = summary
        record.summary_path = path
        record.category = str(summary.get("category") or record.category or "")
        record.metrics = extract_metrics(summary)
        record.metrics_csv_path = find_metrics_csv(record, project_root)
        if record.metrics_csv_path is not None:
            record.training_metrics = read_training_metrics(record.metrics_csv_path)
    return artifacts


def load_hybrid_outputs(project_root: Path, hybrid_glob: str, artifacts: dict[str, ModelArtifacts]) -> None:
    for directory in sorted(project_root.glob(hybrid_glob)):
        if not directory.is_dir():
            continue
        summary_path = directory / "summary.json"
        predictions_path = directory / "test_predictions.csv"
        if not summary_path.exists() and not predictions_path.exists():
            continue

        summary = read_json(summary_path) if summary_path.exists() else {}
        model = normalize_model_name(str(summary.get("model") or "hybrid_patchcore_unet"))
        record = artifacts.setdefault(model, ModelArtifacts(model=model))
        record.summary = summary
        record.summary_path = summary_path if summary_path.exists() else None
        record.category = str(summary.get("category") or record.category or "")
        record.metrics = extract_metrics(summary)
        history_path = directory / "history.csv"
        if history_path.exists():
            record.metrics_csv_path = history_path
            record.training_metrics = read_training_metrics(history_path)
        if predictions_path.exists():
            record.predictions_path = predictions_path
            record.predictions = read_predictions(predictions_path)


def model_from_demo_dir(path: Path) -> str:
    metadata_path = path / "run_metadata.json"
    if metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
            return normalize_model_name(str(metadata.get("model") or ""))
        except Exception:
            pass
    prefix = "demo_outputs_"
    if path.name.lower().startswith(prefix):
        return normalize_model_name(path.name[len(prefix) :])
    return normalize_model_name(path.name)


def read_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            score = safe_float(raw.get("pred_score"))
            if score is None:
                score = safe_float(raw.get("image_score"))
            if score is None:
                continue
            gt_label = (raw.get("gt_label") or "").strip().lower() or None
            rows.append(
                PredictionRow(
                    image_name=raw.get("image_name") or Path(raw.get("image_path") or "").name,
                    gt_label=gt_label,
                    pred_label=safe_int(raw.get("pred_label")),
                    pred_score=score,
                    raw_pred_score=safe_float(raw.get("raw_pred_score")),
                    model_pred_score=safe_float(raw.get("model_pred_score")),
                    calibrated_threshold=safe_float(raw.get("calibrated_threshold")),
                    score_mode=raw.get("score_mode") or None,
                    score_aggregation=raw.get("score_aggregation") or None,
                    score_local_sigma=safe_float(raw.get("score_local_sigma")),
                    roi_mode=raw.get("roi_mode") or None,
                    roi_coverage=safe_float(raw.get("roi_coverage")),
                    original_rel=raw.get("original_rel") or None,
                    gt_mask_rel=raw.get("gt_mask_rel") or None,
                    gt_overlay_rel=raw.get("gt_overlay_rel") or None,
                    heatmap_rel=raw.get("heatmap_rel") or None,
                    overlay_rel=raw.get("overlay_rel") or None,
                    raw_map_rel=raw.get("raw_map_rel") or None,
                    raw_float_map_rel=raw.get("raw_float_map_rel") or None,
                    roi_mask_rel=raw.get("roi_mask_rel") or None,
                    pred_mask_rel=raw.get("pred_mask_rel") or None,
                    pred_overlay_rel=raw.get("pred_overlay_rel") or None,
                    pixel_dice=safe_float(raw.get("pixel_dice")),
                    pixel_iou=safe_float(raw.get("pixel_iou")),
                    pred_area_fraction=safe_float(raw.get("pred_area_fraction")),
                    largest_blob_fraction=safe_float(raw.get("largest_blob_fraction")),
                ),
            )
    return rows


def load_predictions(project_root: Path, demo_glob: str, artifacts: dict[str, ModelArtifacts]) -> None:
    for directory in sorted(project_root.glob(demo_glob)):
        if not directory.is_dir():
            continue
        predictions_path = directory / "predictions.csv"
        if not predictions_path.exists():
            continue

        model = model_from_demo_dir(directory)
        record = artifacts.setdefault(model, ModelArtifacts(model=model))
        record.predictions_path = predictions_path
        record.predictions = read_predictions(predictions_path)


def format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def relpath(path: Path | None, project_root: Path) -> str:
    if path is None:
        return "-"
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def metric_label(key: str) -> str:
    return key.replace("_", " ")


def prediction_label(pred_label: int | None) -> str:
    if pred_label == 0:
        return "good"
    if pred_label == 1:
        return "bad"
    return "unknown"


def confusion_counts(rows: list[PredictionRow]) -> dict[str, int]:
    counts = {
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "unknown": 0,
    }
    for row in rows:
        gt = row.gt_label
        pred = prediction_label(row.pred_label)
        if gt == "good" and pred == "good":
            counts["tn"] += 1
        elif gt == "good" and pred == "bad":
            counts["fp"] += 1
        elif gt == "bad" and pred == "good":
            counts["fn"] += 1
        elif gt == "bad" and pred == "bad":
            counts["tp"] += 1
        else:
            counts["unknown"] += 1
    return counts


def binary_stats(rows: list[PredictionRow]) -> dict[str, float | None]:
    counts = confusion_counts(rows)
    total = counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"]
    if total == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    accuracy = (counts["tp"] + counts["tn"]) / total
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def labelled_predictions(rows: list[PredictionRow]) -> list[tuple[float, int]]:
    labelled: list[tuple[float, int]] = []
    for row in rows:
        if row.gt_label == "bad":
            labelled.append((row.pred_score, 1))
        elif row.gt_label == "good":
            labelled.append((row.pred_score, 0))
    return labelled


def threshold_counts(labelled: list[tuple[float, int]], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for score, label in labelled:
        pred = 1 if score >= threshold else 0
        if label == 1 and pred == 1:
            counts["tp"] += 1
        elif label == 0 and pred == 1:
            counts["fp"] += 1
        elif label == 0 and pred == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def threshold_metrics(labelled: list[tuple[float, int]], threshold: float) -> dict[str, float | int]:
    counts = threshold_counts(labelled, threshold)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        **counts,
    }


def curve_auc(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    sorted_points = sorted(points)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(sorted_points, sorted_points[1:], strict=False):
        area += (x1 - x0) * (y0 + y1) / 2
    return max(0.0, min(1.0, area))


def roc_pr_data(rows: list[PredictionRow]) -> dict[str, Any]:
    labelled = labelled_predictions(rows)
    positives = sum(label for _, label in labelled)
    negatives = len(labelled) - positives
    if not labelled or positives == 0 or negatives == 0:
        return {
            "roc": [],
            "pr": [],
            "roc_auc": None,
            "pr_auc": None,
            "best_threshold": None,
        }

    thresholds = [float("inf"), *sorted({score for score, _ in labelled}, reverse=True)]
    roc: list[tuple[float, float]] = []
    pr: list[tuple[float, float]] = []
    threshold_rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = threshold_metrics(labelled, threshold)
        tp, fp = int(metrics["tp"]), int(metrics["fp"])
        fpr = fp / negatives
        tpr = float(metrics["recall"])
        roc.append((fpr, tpr))
        pr.append((float(metrics["recall"]), float(metrics["precision"])))
        if math.isfinite(threshold):
            threshold_rows.append(metrics)

    if roc[-1] != (1.0, 1.0):
        roc.append((1.0, 1.0))
    if pr[-1][0] < 1.0:
        last_precision = positives / len(labelled)
        pr.append((1.0, last_precision))

    best_threshold = max(
        threshold_rows,
        key=lambda item: (float(item["f1"]), float(item["accuracy"]), float(item["threshold"])),
        default=None,
    )
    return {
        "roc": roc,
        "pr": pr,
        "roc_auc": curve_auc(roc),
        "pr_auc": curve_auc(pr),
        "best_threshold": best_threshold,
    }


def grouped_scores(rows: list[PredictionRow]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = row.gt_label if row.gt_label in {"good", "bad"} else "unknown"
        groups[label].append(row.pred_score)
    return groups


def score_summary(rows: list[PredictionRow], label: str) -> dict[str, float | int | None]:
    scores = grouped_scores(rows).get(label, [])
    if not scores:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(scores),
        "mean": mean(scores),
        "median": median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def make_bins(scores: list[float], bin_count: int = 10) -> tuple[float, float, list[tuple[float, float]]]:
    if not scores:
        return 0.0, 1.0, [(idx / bin_count, (idx + 1) / bin_count) for idx in range(bin_count)]

    min_score = min(scores)
    max_score = max(scores)
    if min_score >= 0.0 and max_score <= 1.0:
        start, end = 0.0, 1.0
    elif abs(max_score - min_score) < 1e-12:
        start, end = min_score - 0.5, max_score + 0.5
    else:
        padding = (max_score - min_score) * 0.05
        start, end = min_score - padding, max_score + padding

    width = (end - start) / bin_count
    bins = [(start + idx * width, start + (idx + 1) * width) for idx in range(bin_count)]
    return start, end, bins


def count_bins(scores: list[float], bins: list[tuple[float, float]]) -> list[int]:
    counts = [0 for _ in bins]
    if not bins:
        return counts
    for score in scores:
        for idx, (left, right) in enumerate(bins):
            is_last = idx == len(bins) - 1
            if left <= score < right or (is_last and left <= score <= right):
                counts[idx] += 1
                break
    return counts


def svg_metric_chart(records: list[ModelArtifacts]) -> str:
    records = [record for record in records if record.metrics]
    if not records:
        return "<div class=\"empty\">No final test metrics found.</div>"

    width, height = 920, 340
    left, right, top, bottom = 74, 28, 34, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(METRIC_KEYS)
    bar_w = min(28.0, group_w / max(len(records), 1) * 0.56)
    color_by_model = {record.model: MODEL_COLORS[idx % len(MODEL_COLORS)] for idx, record in enumerate(records)}

    parts = [
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Final metrics chart\">",
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - tick * plot_h
        parts.append(f"<line class=\"gridline\" x1=\"{left}\" y1=\"{y:.1f}\" x2=\"{width - right}\" y2=\"{y:.1f}\" />")
        parts.append(f"<text class=\"axis-label\" x=\"{left - 12}\" y=\"{y + 4:.1f}\" text-anchor=\"end\">{tick:.2f}</text>")

    for metric_index, metric in enumerate(METRIC_KEYS):
        group_left = left + metric_index * group_w
        center = group_left + group_w / 2
        total_bar_w = bar_w * len(records)
        for record_index, record in enumerate(records):
            value = max(0.0, min(1.0, record.metrics.get(metric, 0.0)))
            x = center - total_bar_w / 2 + record_index * bar_w
            bar_h = value * plot_h
            y = top + plot_h - bar_h
            color = color_by_model[record.model]
            parts.append(
                f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_w - 3:.1f}\" height=\"{bar_h:.1f}\" "
                f"rx=\"3\" fill=\"{color}\"><title>{html.escape(display_model_name(record.model))} "
                f"{html.escape(metric_label(metric))}: {value:.4f}</title></rect>",
            )
        parts.append(
            f"<text class=\"x-label\" x=\"{center:.1f}\" y=\"{height - 48}\" text-anchor=\"middle\">"
            f"{html.escape(metric_label(metric))}</text>",
        )

    legend_x = left
    legend_y = height - 20
    for idx, record in enumerate(records):
        x = legend_x + idx * 156
        color = color_by_model[record.model]
        parts.append(f"<rect x=\"{x}\" y=\"{legend_y - 10}\" width=\"12\" height=\"12\" rx=\"2\" fill=\"{color}\" />")
        parts.append(f"<text class=\"legend-label\" x=\"{x + 18}\" y=\"{legend_y}\">{html.escape(display_model_name(record.model))}</text>")

    parts.append("</svg>")
    return "\n".join(parts)


def svg_histogram(record: ModelArtifacts) -> str:
    rows = record.predictions
    if not rows:
        return "<div class=\"empty\">No prediction scores found.</div>"

    groups = grouped_scores(rows)
    good_scores = groups.get("good", [])
    bad_scores = groups.get("bad", [])
    unknown_scores = groups.get("unknown", [])
    all_scores = [row.pred_score for row in rows]
    _, _, bins = make_bins(all_scores, bin_count=10)
    good_counts = count_bins(good_scores, bins)
    bad_counts = count_bins(bad_scores, bins)
    unknown_counts = count_bins(unknown_scores, bins)
    max_count = max(good_counts + bad_counts + unknown_counts + [1])

    width, height = 760, 310
    left, right, top, bottom = 64, 24, 28, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    slot_w = plot_w / len(bins)
    bar_w = slot_w * 0.24

    parts = [
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" "
        f"aria-label=\"Score histogram for {html.escape(display_model_name(record.model))}\">",
    ]
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        count = max_count * ratio
        y = top + plot_h - ratio * plot_h
        parts.append(f"<line class=\"gridline\" x1=\"{left}\" y1=\"{y:.1f}\" x2=\"{width - right}\" y2=\"{y:.1f}\" />")
        parts.append(f"<text class=\"axis-label\" x=\"{left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\">{count:.0f}</text>")

    series = [
        ("good", good_counts, GOOD_COLOR, 0.16),
        ("bad", bad_counts, BAD_COLOR, 0.43),
    ]
    if unknown_scores:
        series.append(("unknown", unknown_counts, UNKNOWN_COLOR, 0.70))

    for idx, (left_edge, right_edge) in enumerate(bins):
        x0 = left + idx * slot_w
        for label, counts, color, offset in series:
            bar_h = counts[idx] / max_count * plot_h
            x = x0 + slot_w * offset
            y = top + plot_h - bar_h
            parts.append(
                f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_w:.1f}\" height=\"{bar_h:.1f}\" "
                f"rx=\"2\" fill=\"{color}\"><title>{html.escape(label)} scores "
                f"{left_edge:.3f}-{right_edge:.3f}: {counts[idx]}</title></rect>",
            )

    first_label = f"{bins[0][0]:.2f}"
    last_label = f"{bins[-1][1]:.2f}"
    parts.append(f"<text class=\"axis-label\" x=\"{left}\" y=\"{height - 44}\" text-anchor=\"middle\">{first_label}</text>")
    parts.append(f"<text class=\"axis-label\" x=\"{width - right}\" y=\"{height - 44}\" text-anchor=\"middle\">{last_label}</text>")
    parts.append(f"<text class=\"axis-label\" x=\"{left + plot_w / 2:.1f}\" y=\"{height - 20}\" text-anchor=\"middle\">Anomaly score</text>")

    legend_y = height - 48
    legend_x = width - right - 230
    for idx, (label, _, color, _) in enumerate(series):
        x = legend_x + idx * 86
        parts.append(f"<rect x=\"{x}\" y=\"{legend_y - 10}\" width=\"12\" height=\"12\" rx=\"2\" fill=\"{color}\" />")
        parts.append(f"<text class=\"legend-label\" x=\"{x + 18}\" y=\"{legend_y}\">{html.escape(label)}</text>")

    parts.append("</svg>")
    return "\n".join(parts)


def svg_line_chart(
    points: list[tuple[float, float]],
    title: str,
    x_label: str,
    y_label: str,
    color: str,
    diagonal: bool = False,
) -> str:
    if len(points) < 2:
        return "<div class=\"empty\">Not enough labelled prediction rows for this curve.</div>"

    width, height = 420, 300
    left, right, top, bottom = 54, 18, 28, 58
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + max(0.0, min(1.0, value)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - max(0.0, min(1.0, value)) * plot_h

    sorted_points = sorted(points)
    path = " ".join(
        f"{'M' if idx == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}"
        for idx, (x, y) in enumerate(sorted_points)
    )
    parts = [
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{html.escape(title)}\">",
        f"<text class=\"chart-title\" x=\"{left}\" y=\"18\">{html.escape(title)}</text>",
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = sy(tick)
        x = sx(tick)
        parts.append(f"<line class=\"gridline\" x1=\"{left}\" y1=\"{y:.1f}\" x2=\"{width - right}\" y2=\"{y:.1f}\" />")
        parts.append(f"<line class=\"gridline\" x1=\"{x:.1f}\" y1=\"{top}\" x2=\"{x:.1f}\" y2=\"{top + plot_h}\" />")
        parts.append(f"<text class=\"axis-label\" x=\"{left - 8}\" y=\"{y + 4:.1f}\" text-anchor=\"end\">{tick:.2f}</text>")
        parts.append(f"<text class=\"axis-label\" x=\"{x:.1f}\" y=\"{height - 36}\" text-anchor=\"middle\">{tick:.2f}</text>")
    if diagonal:
        parts.append(
            f"<line x1=\"{sx(0):.1f}\" y1=\"{sy(0):.1f}\" x2=\"{sx(1):.1f}\" y2=\"{sy(1):.1f}\" "
            "stroke=\"#94a3b8\" stroke-dasharray=\"5 5\" stroke-width=\"1.4\" />",
        )
    parts.append(f"<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\" stroke-linejoin=\"round\" />")
    for x, y in sorted_points:
        parts.append(
            f"<circle cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" r=\"2.6\" fill=\"{color}\">"
            f"<title>{x_label}: {x:.4f}, {y_label}: {y:.4f}</title></circle>",
        )
    parts.append(f"<text class=\"axis-label\" x=\"{left + plot_w / 2:.1f}\" y=\"{height - 12}\" text-anchor=\"middle\">{html.escape(x_label)}</text>")
    parts.append(f"<text class=\"axis-label\" transform=\"translate(16 {top + plot_h / 2:.1f}) rotate(-90)\" text-anchor=\"middle\">{html.escape(y_label)}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def svg_training_curves(record: ModelArtifacts) -> str:
    series: dict[str, list[tuple[float, float]]] = {}
    for row_index, row in enumerate(record.training_metrics):
        x = row.get("epoch", row.get("step", float(row_index)))
        for key in TRAIN_METRIC_KEYS:
            if key in row:
                series.setdefault(key, []).append((x, row[key]))

    series = {key: points for key, points in series.items() if points}
    if not series:
        return "<div class=\"empty\">No train_loss metrics found yet. Run training again after the CSVLogger change.</div>"

    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if abs(max_x - min_x) < 1e-12:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-12:
        max_y = min_y + 1.0

    width, height = 760, 320
    left, right, top, bottom = 70, 24, 28, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - min_y) / (max_y - min_y) * plot_h

    parts = [
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Training metric curves\">",
    ]
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = min_y + (max_y - min_y) * ratio
        y = sy(value)
        parts.append(f"<line class=\"gridline\" x1=\"{left}\" y1=\"{y:.1f}\" x2=\"{width - right}\" y2=\"{y:.1f}\" />")
        parts.append(f"<text class=\"axis-label\" x=\"{left - 8}\" y=\"{y + 4:.1f}\" text-anchor=\"end\">{value:.4f}</text>")

    for idx, (key, points) in enumerate(series.items()):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        path = " ".join(
            f"{'M' if point_idx == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}"
            for point_idx, (x, y) in enumerate(points)
        )
        parts.append(f"<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.8\" stroke-linejoin=\"round\" />")
        legend_x = left + idx * 130
        legend_y = height - 24
        parts.append(f"<rect x=\"{legend_x}\" y=\"{legend_y - 10}\" width=\"12\" height=\"12\" rx=\"2\" fill=\"{color}\" />")
        parts.append(f"<text class=\"legend-label\" x=\"{legend_x + 18}\" y=\"{legend_y}\">{html.escape(key)}</text>")

    parts.append(f"<text class=\"axis-label\" x=\"{left + plot_w / 2:.1f}\" y=\"{height - 44}\" text-anchor=\"middle\">epoch or step</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def render_threshold_curves(record: ModelArtifacts) -> str:
    data = roc_pr_data(record.predictions)
    best = data["best_threshold"]
    best_text = "No threshold recommendation available."
    if best is not None:
        best_text = (
            f"Best F1 threshold: <strong>{format_number(float(best['threshold']))}</strong> "
            f"(F1 {format_percent(float(best['f1']))}, precision {format_percent(float(best['precision']))}, "
            f"recall {format_percent(float(best['recall']))})"
        )
    return f"""
<div class="curve-summary">{best_text}</div>
<div class="curves">
  <div>
    {svg_line_chart(data['roc'], f"ROC AUC {format_number(data['roc_auc'])}", "FPR", "TPR", "#2563eb", diagonal=True)}
  </div>
  <div>
    {svg_line_chart(data['pr'], f"PR AUC {format_number(data['pr_auc'])}", "Recall", "Precision", "#c2410c")}
  </div>
</div>
""".strip()


def prediction_artifact_href(record: ModelArtifacts, row: PredictionRow, field_name: str, output_dir: Path) -> str | None:
    relative_value = getattr(row, field_name)
    if not relative_value or record.predictions_path is None:
        return None
    path = record.predictions_path.parent / relative_value
    if not path.exists() and field_name in {"pred_mask_rel", "pred_overlay_rel"}:
        path = record.predictions_path.parent / "test_visuals" / relative_value
    if not path.exists():
        return None
    return os.path.relpath(path, output_dir).replace("\\", "/")


def render_image_gallery(record: ModelArtifacts, output_dir: Path, max_items: int) -> str:
    rows = sorted(record.predictions, key=lambda row: row.pred_score, reverse=True)
    if max_items > 0:
        rows = rows[:max_items]
    cards = []
    for row in rows:
        original = prediction_artifact_href(record, row, "original_rel", output_dir)
        gt_mask = prediction_artifact_href(record, row, "gt_mask_rel", output_dir)
        gt_overlay = prediction_artifact_href(record, row, "gt_overlay_rel", output_dir)
        roi_mask = prediction_artifact_href(record, row, "roi_mask_rel", output_dir)
        heatmap = prediction_artifact_href(record, row, "heatmap_rel", output_dir)
        overlay = prediction_artifact_href(record, row, "overlay_rel", output_dir)
        pred_mask = prediction_artifact_href(record, row, "pred_mask_rel", output_dir)
        pred_overlay = prediction_artifact_href(record, row, "pred_overlay_rel", output_dir)
        if not any((original, gt_mask, gt_overlay, roi_mask, heatmap, overlay, pred_mask, pred_overlay)):
            continue
        image_blocks = []
        for label, href in (
            ("Original", original),
            ("GT Mask", gt_mask),
            ("GT Overlay", gt_overlay),
            ("ROI Mask", roi_mask),
            ("Heatmap", heatmap),
            ("Pred Overlay", overlay),
            ("Pred Mask", pred_mask),
            ("Hybrid Overlay", pred_overlay),
        ):
            if href:
                image_blocks.append(
                    f"<figure><img src=\"{html.escape(href)}\" alt=\"{html.escape(label)} "
                    f"{html.escape(row.image_name)}\"><figcaption>{html.escape(label)}</figcaption></figure>",
                )
        cards.append(
            "<article class=\"sample\">"
            "<div class=\"sample-meta\">"
            f"<strong>{html.escape(row.image_name)}</strong>"
            f"<span>GT {html.escape(str(row.gt_label or 'unknown'))} | "
            f"Pred {html.escape(prediction_label(row.pred_label))} | Score {row.pred_score:.4f}"
            f"{' | Model ' + format_number(row.model_pred_score) if row.model_pred_score is not None else ''}"
            f"{' | ' + html.escape(row.score_aggregation) + '/' + html.escape(row.roi_mode or 'off') if row.score_aggregation else ''}"
            f"{' | Threshold ' + format_number(row.calibrated_threshold) if row.calibrated_threshold is not None else ''}</span>"
            f"{'<span>Pixel Dice ' + format_number(row.pixel_dice) + ' | IoU ' + format_number(row.pixel_iou) + ' | Area ' + format_percent(row.pred_area_fraction) + ' | Blob ' + format_percent(row.largest_blob_fraction) + '</span>' if row.pixel_dice is not None else ''}"
            "</div>"
            f"<div class=\"sample-images\">{''.join(image_blocks)}</div>"
            "</article>",
        )

    if not cards:
        return "<div class=\"empty\">No generated original/GT/heatmap/overlay images were found.</div>"
    return f"<div class=\"gallery\">{''.join(cards)}</div>"


def render_metric_table(records: list[ModelArtifacts], project_root: Path) -> str:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(display_model_name(record.model))}</td>"
            f"<td>{html.escape(record.category or '-')}</td>"
            f"<td>{format_number(record.metrics.get('image_AUROC'))}</td>"
            f"<td>{format_number(record.metrics.get('image_F1Score'))}</td>"
            f"<td>{format_number(record.metrics.get('pixel_AUROC'))}</td>"
            f"<td>{format_number(record.metrics.get('pixel_F1Score'))}</td>"
            f"<td><code>{html.escape(relpath(record.summary_path, project_root))}</code></td>"
            "</tr>",
        )
    return (
        "<table>"
        "<thead><tr><th>Model</th><th>Category</th><th>Image AUROC</th><th>Image F1</th>"
        "<th>Pixel AUROC</th><th>Pixel F1</th><th>Summary</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_score_stats(record: ModelArtifacts) -> str:
    good = score_summary(record.predictions, "good")
    bad = score_summary(record.predictions, "bad")
    unknown = score_summary(record.predictions, "unknown")
    rows = []
    for label, summary in (("good", good), ("bad", bad), ("unknown", unknown)):
        if summary["count"] == 0 and label == "unknown":
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{summary['count']}</td>"
            f"<td>{format_number(summary['mean'])}</td>"
            f"<td>{format_number(summary['median'])}</td>"
            f"<td>{format_number(summary['min'])}</td>"
            f"<td>{format_number(summary['max'])}</td>"
            "</tr>",
        )
    return (
        "<table>"
        "<thead><tr><th>GT label</th><th>Count</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_confusion_matrix(record: ModelArtifacts) -> str:
    counts = confusion_counts(record.predictions)
    return (
        "<table class=\"matrix\">"
        "<thead><tr><th>Ground truth</th><th>Pred good</th><th>Pred bad</th></tr></thead>"
        "<tbody>"
        f"<tr><th>good</th><td>{counts['tn']}</td><td>{counts['fp']}</td></tr>"
        f"<tr><th>bad</th><td>{counts['fn']}</td><td>{counts['tp']}</td></tr>"
        "</tbody></table>"
    )


def render_model_panel(record: ModelArtifacts, project_root: Path, output_dir: Path, max_gallery_items: int) -> str:
    stats = binary_stats(record.predictions)
    counts = confusion_counts(record.predictions)
    prediction_path = relpath(record.predictions_path, project_root)
    metrics_path = relpath(record.metrics_csv_path, project_root)
    metric_tiles = [
        ("Samples", str(len(record.predictions))),
        ("Accuracy", format_percent(stats["accuracy"])),
        ("Precision", format_percent(stats["precision"])),
        ("Recall", format_percent(stats["recall"])),
        ("F1", format_percent(stats["f1"])),
    ]
    if counts["unknown"]:
        metric_tiles.append(("Unknown rows", str(counts["unknown"])))

    tiles = "".join(
        f"<div class=\"tile\"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in metric_tiles
    )
    return f"""
<section class="panel">
  <div class="panel-head">
    <div>
      <h3>{html.escape(display_model_name(record.model))}</h3>
      <p>Predictions: <code>{html.escape(prediction_path)}</code></p>
      <p>Training metrics: <code>{html.escape(metrics_path)}</code></p>
    </div>
  </div>
  <div class="tiles">{tiles}</div>
  <h4>Training Curves</h4>
  {svg_training_curves(record)}
  <h4>Threshold Curves</h4>
  {render_threshold_curves(record)}
  <div class="two-col">
    <div>
      <h4>Score Distribution</h4>
      {svg_histogram(record)}
    </div>
    <div>
      <h4>Score Summary</h4>
      {render_score_stats(record)}
      <h4>Confusion Matrix</h4>
      {render_confusion_matrix(record)}
    </div>
  </div>
  <h4>Image Evidence</h4>
  {render_image_gallery(record, output_dir, max_gallery_items)}
</section>
""".strip()


def render_html(records: list[ModelArtifacts], project_root: Path, output_dir: Path, max_gallery_items: int) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_records = [record for record in records if record.metrics]
    prediction_records = [record for record in records if record.predictions]
    training_records = [record for record in records if record.training_metrics]
    total_predictions = sum(len(record.predictions) for record in prediction_records)

    panels = "\n".join(
        render_model_panel(record, project_root, output_dir, max_gallery_items)
        for record in prediction_records
    )
    if not panels:
        panels = "<div class=\"empty\">No prediction CSV files were found.</div>"

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anomaly Combined Report</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --text: #172033;
      --muted: #5c667a;
      --line: #d9dee8;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-2: #c2410c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: #172033;
      color: #ffffff;
      padding: 28px 32px;
      border-bottom: 5px solid var(--accent);
    }}
    header h1 {{ margin: 0 0 8px 0; font-size: 28px; font-weight: 700; letter-spacing: 0; }}
    header p {{ margin: 0; color: #d8dee9; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
    .overview {{
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .tile, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .tile {{ padding: 14px 16px; min-height: 78px; }}
    .tile span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .tile strong {{ display: block; font-size: 24px; line-height: 1.1; }}
    .panel {{ padding: 18px; margin-bottom: 18px; }}
    .panel-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 16px;
    }}
    h2, h3, h4 {{ margin: 0; letter-spacing: 0; }}
    h2 {{ font-size: 20px; margin-bottom: 14px; }}
    h3 {{ font-size: 20px; }}
    h4 {{ font-size: 15px; margin: 16px 0 10px 0; }}
    p {{ margin: 6px 0 0 0; color: var(--muted); }}
    code {{
      color: #253048;
      background: #eef1f6;
      border-radius: 4px;
      padding: 2px 5px;
      white-space: normal;
      word-break: break-word;
    }}
    .tiles {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
      gap: 18px;
      align-items: start;
    }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .chart-title {{ fill: #253048; font-size: 14px; font-weight: 700; }}
    .gridline {{ stroke: #d9dee8; stroke-width: 1; }}
    .axis-label, .legend-label, .x-label {{ fill: #5c667a; font-size: 12px; }}
    .curves {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
      margin-bottom: 8px;
    }}
    .curve-summary {{
      color: #253048;
      background: #eef7f4;
      border: 1px solid #b8d8d1;
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .sample {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }}
    .sample-meta {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }}
    .sample-meta strong, .sample-meta span {{ display: block; }}
    .sample-meta span {{ color: var(--muted); font-size: 13px; margin-top: 3px; }}
    .sample-images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 1px;
      background: var(--line);
    }}
    figure {{ margin: 0; background: #ffffff; }}
    figure img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #f8fafc;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 12px;
      text-align: center;
      padding: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 11px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ color: #253048; background: #eef1f6; font-weight: 700; }}
    tr:last-child td, tr:last-child th {{ border-bottom: none; }}
    .matrix th, .matrix td {{ text-align: center; }}
    .matrix th:first-child {{ text-align: left; }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
      background: #ffffff;
    }}
    @media (max-width: 980px) {{
      main {{ padding: 16px; }}
      .overview, .tiles, .two-col, .curves {{ grid-template-columns: 1fr; }}
      .sample-images {{ grid-template-columns: 1fr; }}
      header {{ padding: 22px 18px; }}
      header h1 {{ font-size: 24px; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Anomaly Combined Report</h1>
    <p>Training summaries, threshold curves, score distributions, and image heatmap evidence. Generated at {html.escape(generated)}.</p>
  </header>
  <main>
    <section class="overview">
      <div class="tile"><span>Models</span><strong>{len(records)}</strong></div>
      <div class="tile"><span>Training summaries</span><strong>{len(metric_records)}</strong></div>
      <div class="tile"><span>Training CSVs</span><strong>{len(training_records)}</strong></div>
      <div class="tile"><span>Prediction CSVs</span><strong>{len(prediction_records)}</strong></div>
      <div class="tile"><span>Prediction rows</span><strong>{total_predictions}</strong></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>Final Test Metrics</h2>
          <p>Values come from train_summary JSON files after engine.test.</p>
        </div>
      </div>
      {svg_metric_chart(metric_records)}
      {render_metric_table(metric_records, project_root) if metric_records else '<div class="empty">No train summaries found.</div>'}
    </section>

    {panels}
  </main>
</body>
</html>
""".strip()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    artifacts = load_train_summaries(project_root, args.runs_glob)
    load_predictions(project_root, args.demo_glob, artifacts)
    load_hybrid_outputs(project_root, args.hybrid_glob, artifacts)

    records = sorted(
        artifacts.values(),
        key=lambda record: (display_model_name(record.model).lower(), record.model),
    )
    if not records:
        raise SystemExit(
            "No artifacts found. Expected train_summary_*.json under runs_*, "
            "predictions.csv under demo_outputs_*, or summary.json/test_predictions.csv under hybrid_outputs_*.",
        )

    output_path.write_text(
        render_html(records, project_root, output_dir, args.max_gallery_items),
        encoding="utf-8",
    )

    print("=" * 80)
    print("[INFO] Dashboard generated")
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Output      : {output_path}")
    print(f"[INFO] Models      : {', '.join(display_model_name(record.model) for record in records)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
