# Hybrid PatchCore + U-Net on VisA

`hybrid_demo.py` adds a supervised stage without changing the existing PatchCore, PaDiM, or FastFlow scripts.

Workflow:

1. Split VisA into supervised `train/val/test` folders.
2. Use PatchCore via `infer_demo.py` to export anomaly maps for each split.
3. Train a small 4-channel U-Net on `RGB + anomaly_map`.
4. Calibrate image-level good/bad threshold on validation predictions.
5. Report image accuracy/F1/AUC and pixel Dice/IoU.

Run the full workflow on the server GPU:

```bash
python3 hybrid_demo.py \
  --dataset-root ./VisA \
  --category candle \
  --train-patchcore \
  --patchcore-results-dir ./runs_visa_candle_patchcore \
  --patchcore-coreset-ratio 0.10 \
  --patchcore-layers layer2 layer3 \
  --patchcore-num-neighbors 9 \
  --patchcore-precision float16 \
  --image-size 384 \
  --train-good 800 \
  --train-bad 60 \
  --val-good 50 \
  --val-bad 20 \
  --test-good 50 \
  --test-bad 20 \
  --hybrid-epochs 40 \
  --hybrid-batch-size 8 \
  --num-workers 4 \
  --accelerator gpu \
  --devices 1 \
  --save-test-maps \
  --clean
```

If PatchCore is already trained, omit `--train-patchcore` and point to the existing result folder:

```bash
python3 hybrid_demo.py \
  --dataset-root ./VisA \
  --category candle \
  --patchcore-results-dir ./runs_visa_candle_patchcore \
  --image-size 384 \
  --hybrid-epochs 40 \
  --hybrid-batch-size 8 \
  --num-workers 4 \
  --accelerator gpu \
  --devices 1 \
  --save-test-maps
```

## Small-defect segmentation training

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
  --image-p99-weight 0.25 \
  --image-threshold-min 0.003 \
  --image-threshold-max 0.006 \
  --save-test-maps
```

If this uses too much VRAM, lower `--hybrid-batch-size` to `8`. The validation and test passes still use the full
image size; crops affect only training batches.

Important outputs:

- `hybrid_inputs_visa_<category>/manifest.json`: supervised split manifest.
- `hybrid_maps_visa_<category>_patchcore/{train,val,test}`: PatchCore maps and masks.
- `hybrid_outputs_visa_<category>/summary.json`: final metrics.
- `hybrid_outputs_visa_<category>/test_predictions.csv`: per-image decisions and pixel metrics.
- `hybrid_outputs_visa_<category>/test_visuals/`: predicted masks and overlays when `--save-test-maps` is used.

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
  --image-p99-weight 0.25 \
  --image-threshold-min 0.006 \
  --image-threshold-max 0.008 \
  --save-test-maps
```

Use a separate `--output-dir` for sweeps so the previous summary and visualizations stay intact.
