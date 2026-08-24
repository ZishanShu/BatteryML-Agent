from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


_TEXT_PREVIEW_EXTS = {".txt", ".log", ".md", ".json", ".yaml", ".yml", ".csv", ".tsv"}
_TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}


def detect_file_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".tsv"}:
        return "tsv"
    if suffix in {".xlsx", ".xls"}:
        return "xlsx"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".json"}:
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix in {".txt", ".log", ".md"}:
        return "text"
    return suffix.lstrip(".") or "file"


def _safe_file_info(path: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "full_path": str(path),
        "size": int(path.stat().st_size),
        "size_human": f"{path.stat().st_size / 1024:.1f} KB",
        "file_type": detect_file_type(path),
        "modified_time": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
        "can_preview": path.suffix.lower() in _TEXT_PREVIEW_EXTS or path.suffix.lower() in _TABLE_EXTS,
    }


def list_dataset_files(dataset_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    return [_safe_file_info(path) for path in sorted(root.rglob("*")) if path.is_file()]


def _inspect_dir(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    files = list_dataset_files(path) if exists else []
    return {
        "path": str(path),
        "exists": exists,
        "file_count": len(files),
        "files": files,
    }


def inspect_dataset(dataset: str, current_root: str | Path | None = None) -> Dict[str, Any]:
    dataset = dataset.strip()
    root = Path(current_root) if current_root else Path(".")
    raw = _inspect_dir(Path("data/raw") / dataset)
    processed = _inspect_dir(Path("data/processed") / dataset)
    cache = _inspect_dir(root / "cache" / dataset)
    runs = _inspect_dir(root / "runs" / dataset)

    overall_status = "Missing"
    if processed["exists"]:
        overall_status = "Ready"
    elif raw["exists"]:
        overall_status = "Need Preprocess"

    return {
        "dataset": dataset,
        "raw": raw,
        "processed": processed,
        "cache": cache,
        "runs": runs,
        "overall_status": overall_status,
        "can_run": processed["exists"],
    }


def preview_file(file_path: str | Path, max_rows: int = 20) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
            return {
                "preview_type": "csv",
                "shape": tuple(df.shape),
                "columns": list(df.columns),
                "preview": df.head(max_rows).to_dict(orient="records"),
            }
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
            return {
                "preview_type": "xlsx",
                "shape": tuple(df.shape),
                "columns": list(df.columns),
                "preview": df.head(max_rows).to_dict(orient="records"),
            }
        if suffix == ".parquet":
            df = pd.read_parquet(path)
            return {
                "preview_type": "parquet",
                "shape": tuple(df.shape),
                "columns": list(df.columns),
                "preview": df.head(max_rows).to_dict(orient="records"),
            }
        if suffix == ".json":
            data = pd.read_json(path, orient="records") if path.read_text(encoding="utf-8", errors="ignore").strip().startswith("[") else yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
            return {"preview_type": "json", "preview": data}
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
            return {"preview_type": "yaml", "preview": data}
        if suffix in {".txt", ".log", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            return {"preview_type": "txt", "preview": "\n".join(lines[:max_rows]), "total_lines": len(lines)}
        return {"preview_type": "binary", "preview": {"note": f"Preview not implemented for {suffix or 'unknown'} files."}}
    except Exception as exc:
        return {"error": str(exc)}


def format_file_table(files: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for file_info in files:
        rows.append(
            {
                "File": file_info.get("name", ""),
                "Type": file_info.get("file_type", ""),
                "Size": file_info.get("size_human", ""),
                "Preview": "✓" if file_info.get("can_preview") else "—",
            }
        )
    return pd.DataFrame(rows)
