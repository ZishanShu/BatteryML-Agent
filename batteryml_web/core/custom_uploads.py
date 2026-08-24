from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ALLOWED_CUSTOM_SOURCE_FORMATS = ["ARBIN", "BIOLOGIC", "LANDT", "MACCOR", "NEWARE"]


@dataclass
class UploadValidationResult:
    ok: bool
    message: str
    details: dict


def _is_archive(path: Path) -> bool:
    suffixes = [s.lower() for s in path.suffixes]
    return path.suffix.lower() == ".zip" or suffixes[-2:] == [".tar", ".gz"] or path.suffix.lower() in {".tgz", ".tar"}


def save_uploaded_files(uploaded_files, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in uploaded_files or []:
        dst = target_dir / uploaded.name
        dst.write_bytes(uploaded.getbuffer())
        saved.append(dst)
    return saved


def extract_if_archive(path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not _is_archive(path):
        return []
    extracted = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target_dir)
            extracted = [target_dir / n for n in zf.namelist()]
    else:
        mode = "r:gz" if path.suffixes[-1].lower() == ".gz" else "r"
        with tarfile.open(path, mode) as tf:
            tf.extractall(target_dir)
            extracted = [target_dir / m.name for m in tf.getmembers()]
    return extracted


def collect_raw_files(target_dir: Path):
    if not target_dir.exists():
        return []
    return [p for p in target_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]


def validate_custom_upload_layout(train_dir: Path, valid_dir: Path):
    train_files = collect_raw_files(train_dir)
    valid_files = collect_raw_files(valid_dir)
    ok = bool(train_files) and bool(valid_files)
    return UploadValidationResult(
        ok=ok,
        message="ok" if ok else "Training raw files are required." if not train_files else "Validation/test raw files are required.",
        details={"train_file_count": len(train_files), "valid_file_count": len(valid_files)},
    )


def build_uploaded_dataset_manifest(project_dir: Path, source_format: str, dataset_name: str, task_type: str):
    uploads_dir = project_dir / "uploads"
    train_dir = uploads_dir / "train_raw"
    valid_dir = uploads_dir / "valid_raw"
    manifest = {
        "dataset_name": dataset_name,
        "source_format": source_format,
        "task_type": task_type,
        "train_raw_files": [str(p.relative_to(project_dir)) for p in collect_raw_files(train_dir)],
        "valid_raw_files": [str(p.relative_to(project_dir)) for p in collect_raw_files(valid_dir)],
    }
    out = uploads_dir / "uploaded_dataset_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_batteryml_preprocess_for_split(source_format: str, raw_dir: Path, output_dir: Path, optional_config_path: Path | None = None):
    from subprocess import run
    import sys

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "batteryml_web/core/run_batteryml_cli.py", "preprocess", source_format, str(raw_dir), str(output_dir)]
    if optional_config_path is not None:
        command.extend(["--config", str(optional_config_path)])
    proc = run(command, capture_output=True, text=True)
    report = {
        "source_format": source_format,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "optional_config_path": str(optional_config_path) if optional_config_path else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0,
    }
    return report


def validate_processed_output(output_dir: Path):
    files = collect_raw_files(output_dir)
    batterydata_files = [p for p in files if p.suffix.lower() in {".pkl", ".pickle", ".parquet", ".npz", ".npy"}]
    ok = bool(files) and bool(batterydata_files)
    return UploadValidationResult(
        ok=ok,
        message="ok" if ok else "Processed output is missing or incomplete.",
        details={"file_count": len(files), "batterydata_file_count": len(batterydata_files)},
    )
