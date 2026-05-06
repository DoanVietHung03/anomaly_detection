# Hybrid PatchCore + Supervised Segmentation on VisA

`hybrid_demo.py` adds a supervised stage on top of the existing PatchCore workflow.

Workflow:

1. Split VisA into supervised `train/val/test` folders.
2. Use PatchCore via `infer_demo.py` to export anomaly maps and image scores for each split.
3. Train a 4-channel segmentation model on `RGB + anomaly_map`.
4. Calibrate PatchCore image-level good/bad threshold on validation predictions.
5. Use U-Net only for localization/masks, with mask threshold selected on validation Dice.
6. Report final image accuracy/F1/AUC from PatchCore and pixel Dice/IoU from the segmentation model.

This is the recommended product-style flow for small defects: PatchCore decides whether an image is good/bad, and
the segmentation model explains where the defect is. To compare against the older behavior where mask scores decide image-level
labels, pass `--image-decision-source unet`.

The default hybrid settings are now the best-fast profile:

- `512x512` model input.
- FPN segmentation head with an `efficientnet-b0` ImageNet encoder.
- Defect-centered crop training directly on native-resolution images, plus small-defect oversampling, focal loss, and Tversky loss.
- Validation/test segmentation uses tiled native-resolution inference and stitches the probability map before thresholding.
- PatchCore keeps the image-level good/bad decision; segmentation is used for localization.
- Base-map fallback is enabled when the segmentation mask is too small.

Install the segmentation dependency once before running this profile:

```bash
python3 -m pip install segmentation-models-pytorch timm
```

Run the full workflow on the server GPU:

```bash
python3 hybrid_demo.py \
  --train-base-model \
  --save-test-maps \
  --clean
```

If PatchCore is already trained, omit `--train-patchcore` and point to the existing result folder:

```bash
python3 hybrid_demo.py \
  --save-test-maps
```

## Pretrained Segmentation Training

The default pretrained segmentation model is FPN with an EfficientNet-B0 ImageNet encoder. Reuse existing prepared
splits and maps with a short command:

```bash
python3 hybrid_demo.py \
  --split-dir ./hybrid_result_visa_candle/hybrid_inputs_visa_candle \
  --patchcore-results-dir ./hybrid_result_visa_candle/runs_visa_candle_patchcore \
  --maps-dir ./hybrid_result_visa_candle/hybrid_maps_visa_candle_patchcore \
  --output-dir ./hybrid_result_visa_candle/hybrid_outputs_visa_candle_best_fast \
  --skip-prepare \
  --skip-map-generation \
  --save-test-maps
```

If the server has no internet for ImageNet weights, use `--seg-encoder-weights none`. U-Net++ and larger encoders remain
available for experiments, but they are no longer the recommended default because they are much slower.

## Small-defect Segmentation Training

If image-level post-processing still misses tiny defects, retrain only the U-Net stage with defect-centered crops,
small-defect oversampling, and focal/Tversky losses. Reuse the existing PatchCore maps:

```bash
python3 hybrid_demo.py \
  --dataset-root ./VisA \
  --category candle \
  --maps-dir ./hybrid_maps_visa_candle_patchcore \
  --output-dir ./hybrid_outputs_visa_candle_small_defect \
  --skip-prepare \
  --skip-map-generation \
  --image-size 384 \
  --hybrid-epochs 60 \
  --hybrid-batch-size 16 \
  --num-workers 8 \
  --accelerator gpu \
  --devices 1 \
  --hybrid-train-crop-size 256 \
  --defect-crop-prob 0.85 \
  --small-defect-oversample 3.0 \
  --small-defect-area-threshold 600 \
  --bce-weight 0.5 \
  --dice-weight 1.0 \
  --focal-weight 0.75 \
  --focal-alpha 0.8 \
  --focal-gamma 2.0 \
  --tversky-weight 0.75 \
  --tversky-alpha 0.3 \
  --tversky-beta 0.7 \
  --postprocess-min-component-area 60 \
  --postprocess-object-edge-ignore-px 0 \
  --image-decision-source patchcore \
  --patchcore-score-column pred_score \
  --patchcore-threshold-min-recall 0.90 \
  --patchcore-threshold-min-specificity 0.92 \
  --save-test-maps
```

For an RTX A4000 16 GB, the default `512x512`, FPN EfficientNet-B0, batch size `4`, native-resolution crop training, and
tiled native-resolution validation/test profile is the recommended starting point.

Important outputs:

- `hybrid_inputs_visa_<category>/manifest.json`: supervised split manifest.
- `hybrid_maps_visa_<category>_patchcore/{train,val,test}`: PatchCore maps and masks.
- `hybrid_outputs_visa_<category>/summary.json`: final metrics.
- `hybrid_outputs_visa_<category>/test_predictions.csv`: per-image decisions and pixel metrics.
- `hybrid_outputs_visa_<category>/test_visuals/`: predicted masks and overlays when `--save-test-maps` is used.

Important scoring fields:

- `patchcore_image_score`: image-level score used by default for final good/bad.
- `unet_image_score`: optional mask-derived image score for comparison/debugging.
- `image_score` / `final_image_score`: the score from the active `--image-decision-source`.
- `patchcore_image_threshold` and `unet_image_threshold`: thresholds selected from validation.
- `mask_source` / `fallback_used`: whether the final visualization mask came from the segmentation model or PatchCore fallback.

## Post-processing sweep

After a hybrid model is trained, you can re-run only scoring/post-processing without retraining U-Net or regenerating PatchCore maps:

```bash
python3 hybrid_demo.py \
  --dataset-root ./VisA \
  --category candle \
  --maps-dir ./hybrid_maps_visa_candle_patchcore \
  --output-dir ./hybrid_outputs_visa_candle_pp \
  --hybrid-checkpoint ./hybrid_outputs_visa_candle/hybrid_unet_best.pt \
  --eval-only \
  --skip-prepare \
  --skip-map-generation \
  --image-size 384 \
  --hybrid-batch-size 8 \
  --num-workers 4 \
  --accelerator gpu \
  --devices 1 \
  --postprocess-min-component-area 150 \
  --postprocess-object-edge-ignore-px 8 \
  --image-decision-source patchcore \
  --patchcore-score-column pred_score \
  --patchcore-threshold-min-recall 0.90 \
  --patchcore-threshold-min-specificity 0.92 \
  --save-test-maps
```

Use a separate `--output-dir` for sweeps so the previous summary and visualizations stay intact.
