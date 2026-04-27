# MVTec AD 2 `can` Demo Project

Project tối thiểu để bạn train và chạy demo anomaly detection trên **MVTec AD 2 / class `can`** bằng **Anomalib**.

Mục tiêu của bộ này là:
- train nhanh một model anomaly detection trên `can`
- test trên `test_public`
- chạy inference trên ảnh hoặc thư mục ảnh
- xuất **heatmap**, **overlay**, **CSV summary** và **HTML report** để demo

> Lưu ý: bộ `can` chỉ là **proxy** để dựng demo anomaly detection trên vật thể hình trụ/kim loại. Nó **không đại diện trực tiếp** cho lỗi label thùng sơn thật.

## 1) Cấu trúc file

```text
mvtec_can_demo/
├── README.md
├── requirements.txt
├── train_demo.py
├── infer_demo.py
├── prepare_demo_inputs.py
└── run_demo.py
```

## 2) Chuẩn bị môi trường

### Python
Khuyến nghị Python 3.10 - 3.12.

### Cài thư viện
Project này mặc định dùng NVIDIA GPU. Profile PatchCore mặc định đang nhắm tới RTX A4000 16 GB VRAM: `image-size 512`, `layer2 layer3`, `coreset 0.1`, `float16`, và `tiling auto`.

Tạo môi trường ảo nếu chưa có:

```bash
python -m venv venv
```

Cài thư viện bằng Python trong `venv`:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trên Linux/macOS:

```bash
./venv/bin/python -m pip install -r requirements.txt
```

Nếu nâng driver NVIDIA mới hơn, bạn có thể đổi CUDA extra trong `requirements.txt` theo version Anomalib đang dùng, ví dụ `anomalib[cu126]` hoặc `anomalib[cu130]`.

## 3) Chuẩn bị dataset

Giả sử bạn đã giải nén dataset ở:

```text
./can/
└── can/
    ├── train/
    ├── validation/
    ├── test_public/
    ├── test_private/
    └── test_private_mixed/
```

Project này mặc định dùng **`test_public`** để đánh giá local.

Nếu dataset nằm ở chỗ khác, truyền `--dataset-root` khi chạy script.

Thư mục `can/` trong project hiện là dataset local. Source code không bắt buộc phải có đúng thư mục này, nhưng train/demo cần một MVTec AD 2 dataset root có chứa category `can`.

## 4) Train model

### PatchCore (khuyến nghị chạy đầu tiên)

```bash
python train_demo.py \
  --dataset-root ./can \
  --category can \
  --model patchcore \
  --results-dir ./runs_patchcore \
  --epochs 1 \
  --image-size 512 \
  --tiling auto \
  --patchcore-layers layer2 layer3 \
  --patchcore-coreset-ratio 0.1 \
  --patchcore-precision float16 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --accelerator gpu \
  --devices 1
```

### EfficientAD (để so tốc độ)

```bash
python train_demo.py \
  --dataset-root ./can \
  --category can \
  --model efficientad \
  --results-dir ./runs_efficientad \
  --epochs 20 \
  --train-batch-size 1 \
  --eval-batch-size 8 \
  --accelerator gpu \
  --devices 1
```

> Ghi chú: EfficientAD thường chạy với `train_batch_size=1`.

### Chẩn đoán theo biến thể ảnh

Trong bộ `can` này, `train/good` và `validation/good` chỉ có ảnh `regular`, nhưng `test_public/good` có thêm `overexposed`, `underexposed`, `shift_1`, `shift_2`, `shift_3`. Với one-class anomaly detection, các biến thể này có thể bị xem là anomaly dù nhãn là good.

Để kiểm tra cùng điều kiện chụp trước:

```powershell
.\venv\Scripts\python.exe train_demo.py --dataset-root .\can --category can --model patchcore --results-dir .\runs_patchcore_regular --epochs 1 --image-size 512 --patchcore-layers layer2 layer3 --patchcore-coreset-ratio 0.1 --patchcore-precision float16 --tiling auto --test-variant regular
```

Để đo toàn bộ public split:

```powershell
.\venv\Scripts\python.exe train_demo.py --dataset-root .\can --category can --model patchcore --results-dir .\runs_patchcore_all --epochs 1 --image-size 512 --patchcore-layers layer2 layer3 --patchcore-coreset-ratio 0.1 --patchcore-precision float16 --tiling auto --test-variant all
```

## 5) Chuẩn bị ảnh input demo

Bạn có thể lấy sẵn một ít ảnh từ `test_public` để demo:

```bash
python prepare_demo_inputs.py \
  --dataset-root ./can \
  --category can \
  --output-dir ./demo_inputs \
  --num-good 8 \
  --num-bad 8
```

Nếu muốn demo cùng điều kiện chụp với train, thêm:

```powershell
--variants regular
```

## 6) Chạy inference và xuất report

### Dùng checkpoint cụ thể

```bash
python infer_demo.py \
  --input-path ./demo_inputs \
  --checkpoint /path/to/model.ckpt \
  --model patchcore \
  --output-dir ./demo_outputs
```

### Hoặc tự tìm checkpoint tốt nhất trong thư mục results

```bash
python infer_demo.py \
  --input-path ./demo_inputs \
  --results-dir ./runs_patchcore \
  --model patchcore \
  --output-dir ./demo_outputs
```

Heatmap mặc định được chuẩn hóa `global` trên cả batch inference để tránh ảnh good bị tô đỏ chỉ vì từng ảnh được stretch riêng. Nếu muốn hành vi cũ, dùng `--heatmap-normalization per-image`.

Kết quả sẽ có:
- `predictions.csv`
- `report.html`
- `heatmaps/`
- `overlays/`
- `raw_maps/`

## 7) Quy trình demo nhanh nhất

Chạy toàn bộ pipeline bằng một lệnh chung cho Windows, Linux và macOS:

```powershell
.\venv\Scripts\python.exe run_demo.py --check-only
.\venv\Scripts\python.exe run_demo.py
```

```bash
./venv/bin/python run_demo.py --check-only
./venv/bin/python run_demo.py
```

Nếu dataset không nằm trong `./can`, chỉ rõ đường dẫn:

```powershell
.\venv\Scripts\python.exe run_demo.py --dataset-root /data/MVTec_AD_2
```

Muốn chạy EfficientAD:

```powershell
.\venv\Scripts\python.exe run_demo.py --model efficientad
```

## 8) Kết quả mong đợi

Sau khi chạy xong bạn có thể mở:

```text
./demo_outputs/report.html
```

để xem demo dạng trực quan: ảnh gốc, heatmap, overlay, score và nhãn dự đoán.

## 9) Khi chuyển sang dữ liệu thùng sơn thật

Bạn nên giữ lại logic project này và chỉ thay:
- nguồn ảnh đầu vào
- datamodule / folder dataset riêng
- metric/threshold
- phần sinh report

Tức là bộ file này có thể dùng như skeleton cho bước sau.
