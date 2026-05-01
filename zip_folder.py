#!/usr/bin/env python3
"""Zip generated server artifacts into one archive.

Typical workflow:
1. Run this script on the server.
2. Download zipped_output/server_artifacts.zip to your local machine.
3. Delete the server artifact folders after you have confirmed the zip is safe.
"""

from __future__ import annotations

import shutil
import zipfile
from glob import glob
from pathlib import Path


BASE_DIR = Path(".")
OUTPUT_DIR = Path("zipped_output")
OUTPUT_ZIP = OUTPUT_DIR / "server_artifacts.zip"

FOLDER_PATTERNS = [
    "calibration_inputs*",
    "demo_inputs*",
    "demo_outputs*",
    "dashboard_outputs*",
    "results*",
    "runs*",
]

# Keep this False until you have confirmed the zip file exists and is downloadable.
DELETE_SOURCE_FOLDERS_AFTER_ZIP = True


def find_folders() -> list[Path]:
    base_dir = BASE_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    folders: set[Path] = set()

    for pattern in FOLDER_PATTERNS:
        for match in glob(str(base_dir / pattern)):
            path = Path(match).resolve()
            if not path.is_dir():
                continue
            if path == output_dir or output_dir in path.parents:
                continue
            folders.add(path)

    return sorted(folders)


def make_zip(folders: list[Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for file_path in sorted(folder.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.relative_to(BASE_DIR.resolve()))


def delete_folders(folders: list[Path]) -> None:
    for folder in folders:
        print(f"[DELETE] {folder}")
        shutil.rmtree(folder)


def main() -> None:
    folders = find_folders()
    if not folders:
        print("[INFO] No artifact folders found.")
        return

    print("[INFO] Folders to zip:")
    for folder in folders:
        print(f" - {folder}")

    make_zip(folders)
    print(f"[DONE] Created: {OUTPUT_ZIP.resolve()}")

    if DELETE_SOURCE_FOLDERS_AFTER_ZIP:
        delete_folders(folders)
        print("[DONE] Source folders deleted.")
    else:
        print("[INFO] Source folders kept. Set DELETE_SOURCE_FOLDERS_AFTER_ZIP = True to delete after zipping.")


if __name__ == "__main__":
    main()
