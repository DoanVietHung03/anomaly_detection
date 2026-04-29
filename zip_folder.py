import os
import zipfile
from pathlib import Path
from glob import glob


# ====== CẤU HÌNH ======
FOLDERS_TO_ZIP = [
    "calibration_*",
    "demo_*",
    "results",
    "runs_*",
    "dashboard_*",
]

BASE_DIR = "."  # thư mục hiện tại
OUTPUT_DIR = "zipped_output"
# ======================


def expand_folders(patterns, base_dir):
    all_folders = []

    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        matches = glob(full_pattern)

        for m in matches:
            if os.path.isdir(m):
                all_folders.append(m)

    return list(set(all_folders))  # remove duplicate


def zip_folder(folder_path: str, output_dir: str):
    folder = Path(folder_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / f"{folder.name}.zip"

    print(f"[ZIP] {folder} -> {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(folder.parent)
                zipf.write(file_path, arcname)

    print(f"[DONE] {zip_path}")


def main():
    folders = expand_folders(FOLDERS_TO_ZIP, BASE_DIR)

    print("Folders tìm được:")
    for f in folders:
        print(" -", f)

    for folder in folders:
        zip_folder(folder, OUTPUT_DIR)

    print("\nHoàn tất.")


if __name__ == "__main__":
    main()