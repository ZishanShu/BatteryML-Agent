from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular file type: {path.suffix}")


def inspect_tabular_file(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = _read_tabular_file(path)
    preview_df = df.head(10).copy()

    return {
        "path": str(path),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "preview": preview_df.to_dict(orient="records"),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }
