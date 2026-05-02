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

Important outputs:

- `hybrid_inputs_visa_<category>/manifest.json`: supervised split manifest.
- `hybrid_maps_visa_<category>_patchcore/{train,val,test}`: PatchCore maps and masks.
- `hybrid_outputs_visa_<category>/summary.json`: final metrics.
- `hybrid_outputs_visa_<category>/test_predictions.csv`: per-image decisions and pixel metrics.
- `hybrid_outputs_visa_<category>/test_visuals/`: predicted masks and overlays when `--save-test-maps` is used.
