from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def _load_raw_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def standardize_dataset(raw_file: str | Path, schema_mapping: Dict[str, str], output_dir: str | Path) -> Tuple[Path, Dict[str, Any]]:
    raw_file = Path(raw_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_raw_file(raw_file)
    provenance = {col: col for col in df.columns}
    renamed = df.copy()

    for schema_field, raw_col in schema_mapping.items():
        if raw_col in renamed.columns and schema_field not in renamed.columns:
            renamed = renamed.rename(columns={raw_col: schema_field})

    out_csv = output_dir / "standardized_dataset.csv"
    renamed.to_csv(out_csv, index=False)

    provenance_path = output_dir / "column_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    profile = {
        "raw_file": str(raw_file),
        "rows": int(len(renamed)),
        "columns": list(renamed.columns),
        "column_count": int(len(renamed.columns)),
        "raw_provenance": provenance,
        "schema_mapping": schema_mapping,
        "standardized_csv": str(out_csv),
        "note": "This standardized dataset is ready for validation. Additional dataset adapter may be required for full BatteryML training depending on task/model requirements.",
    }
    (output_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_csv, profile
