# Artifact Dashboard

Use this dashboard when you already have training and inference artifacts and do not want to train again.

Run:

```powershell
.\venv\Scripts\python.exe visualize_results.py
```

Default output:

```text
./dashboard_outputs/artifact_dashboard.html
```

The script reads:

- `runs_*/train_summary_*.json` for final test metrics
- `demo_outputs_*/predictions.csv` for score distributions and confusion matrices
- `demo_outputs_*` generated image folders for original, heatmap, and overlay evidence
- future `runs_*/logs/*/version_*/metrics.csv` files for training curves

The report includes final metrics, ROC/PR threshold curves, score histograms, confusion matrices, training curves when a CSV log exists, and the image/heatmap/overlay gallery from the current inference report.

Current historical runs may show an empty training-curve section. Run training again with the updated `train_demo.py` to create CSVLogger output for future curves.
