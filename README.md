# VisA Anomaly Detection Demo

Project demo anomaly detection bằng Anomalib, mặc định dùng **VisA / `candle`**. Pipeline hiện hỗ trợ 2 model stage 1:

- `patchcore`
- `efficientad`

Bạn có thể chạy riêng lẻ từng model qua `train_demo.py`, `infer_demo.py`, `run_demo.py`, hoặc dùng hybrid segmentation qua `hybrid_demo.py`.

## Dataset

Mặc định các script tìm dataset ở:

```text
./VisA/
├── candle/
│   └── Data/
└── split_csv/
```

Nếu VisA nằm chỗ khác, truyền `--dataset-root`. Category mặc định là `candle`, đổi bằng `--category`.

## Cài đặt

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trên Linux/macOS:

```bash
./venv/bin/python -m pip install -r requirements.txt
```

## Chạy nhanh

PatchCore:

```powershell
.\venv\Scripts\python.exe run_demo.py --model patchcore
```

EfficientAD:

```powershell
.\venv\Scripts\python.exe run_demo.py --model efficientad --max-steps 70000 --efficientad-imagenet-dir .\datasets\imagenette
```

`run_demo.py` sẽ train, chuẩn bị ảnh demo/calibration từ VisA, chạy inference và xuất report HTML.

## Train riêng lẻ

PatchCore:

```powershell
.\venv\Scripts\python.exe train_demo.py --dataset visa --dataset-root .\VisA --category candle --model patchcore --results-dir .\runs_patchcore_visa_candle --epochs 1
```

EfficientAD:

```powershell
.\venv\Scripts\python.exe train_demo.py --dataset visa --dataset-root .\VisA --category candle --model efficientad --results-dir .\runs_efficientad_visa_candle --max-steps 70000 --efficientad-imagenet-dir .\datasets\imagenette
```

EfficientAD cần folder ImageNet/Imagenette-style cho penalty branch. Nếu máy có mạng, Anomalib có thể tự tải một số weights; nếu chạy offline, chuẩn bị sẵn `--efficientad-imagenet-dir`.

## Inference riêng lẻ

```powershell
.\venv\Scripts\python.exe infer_demo.py --model patchcore --dataset visa --dataset-root .\VisA --category candle --input-path .\demo_inputs_visa_candle --results-dir .\runs_patchcore_visa_candle
```

Đổi `--model efficientad` và `--results-dir .\runs_efficientad_visa_candle` để dùng EfficientAD.

## Hybrid

Hybrid dùng anomaly map từ stage-1 model cộng với RGB để train U-Net segmentation.

PatchCore hybrid:

```powershell
.\venv\Scripts\python.exe hybrid_demo.py --base-model patchcore --train-base-model
```

EfficientAD hybrid:

```powershell
.\venv\Scripts\python.exe hybrid_demo.py --base-model efficientad --train-base-model --efficientad-max-steps 70000 --efficientad-imagenet-dir .\datasets\imagenette
```

Các tên cột cũ kiểu `patchcore_*` vẫn được giữ trong CSV/JSON để không phá file phân tích cũ, nhưng khi `--base-model efficientad` thì chúng đại diện cho score/map của EfficientAD.

## Output chính

- `runs_<model>_visa_<category>/`: checkpoint và summary train
- `demo_inputs_visa_<category>/`: ảnh demo được sample từ VisA
- `calibration_inputs_visa_<category>/`: ảnh calibration
- `demo_outputs_<model>_visa_<category>/`: `report.html`, `predictions.csv`, heatmap, overlay, raw maps
- `hybrid_outputs_visa_<category>_<base-model>/`: checkpoint U-Net, CSV prediction, `summary.json`
