from __future__ import annotations

import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_PATTERNS = ["*metric*.csv", "*metrics*.csv", "*metric*.json", "*metrics*.json"]
PREDICTION_PATTERNS = [
    "*prediction*.csv",
    "*predictions*.csv",
    "*prediction*.pkl",
    "*predictions*.pkl",
    "manual_predictions_seed_*.pkl",
    "predictions_seed_*.pkl",
]
LOG_PATTERNS = ["training_log.txt", "*.log", "*log*.txt", "preprocess*.log"]
REPORT_PATTERNS = ["report.html", "report.pdf", "*.html", "*.pdf"]
FIGURE_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.svg"]
CHECKPOINT_PATTERNS = ["*.ckpt", "*.pt", "*.pth", "*.pkl"]


def _glob_many(root: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    if not root.exists():
        return found
    for pattern in patterns:
        found.extend(sorted(root.rglob(pattern)))
    uniq = []
    seen = set()
    for p in found:
        if p.is_file() and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def find_result_files(project_dir: Path, workspace_dir: Path) -> dict:
    roots = [project_dir, workspace_dir]
    result = {
        "metrics_files": [],
        "prediction_files": [],
        "log_files": [],
        "report_files": [],
        "figure_files": [],
        "checkpoint_files": [],
        "all_files": [],
        "searched_roots": roots,
    }
    for root in roots:
        if not root.exists():
            continue
        result["metrics_files"].extend(_glob_many(root, METRIC_PATTERNS))
        result["prediction_files"].extend(_glob_many(root, PREDICTION_PATTERNS))
        result["log_files"].extend(_glob_many(root, LOG_PATTERNS))
        result["report_files"].extend(_glob_many(root, REPORT_PATTERNS))
        result["figure_files"].extend(_glob_many(root, FIGURE_PATTERNS))
        result["checkpoint_files"].extend(_glob_many(root, CHECKPOINT_PATTERNS))
    seen = set()
    for key in ["metrics_files", "prediction_files", "log_files", "report_files", "figure_files", "checkpoint_files"]:
        deduped = []
        for p in result[key]:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        result[key] = deduped
    result["all_files"] = sorted(seen)
    return result


def _normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    if "metric" in cols and "value" in cols:
        return df[[cols["metric"], cols["value"]]].rename(columns={cols["metric"]: "Metric", cols["value"]: "Value"})
    records = []
    for metric in ["RMSE", "MAE", "MAPE"]:
        for c in df.columns:
            if metric.lower() in c.lower():
                records.append({"Metric": metric, "Value": df.iloc[0][c]})
                break
    return pd.DataFrame(records) if records else df


def load_metrics(metrics_files: list[Path]) -> pd.DataFrame | None:
    for path in metrics_files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                return _normalize_metrics_df(df)
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(payload, dict):
                    return _normalize_metrics_df(pd.DataFrame([payload]))
                if isinstance(payload, list):
                    return _normalize_metrics_df(pd.DataFrame(payload))
        except Exception:
            continue
    return None


def _parse_metrics_from_text(text: str) -> pd.DataFrame | None:
    patterns = [
        r"\{[^\n\r]*RMSE[^\n\r]*MAE[^\n\r]*MAPE[^\n\r]*\}",
        r"RMSE\s*[:=]\s*([0-9.eE+-]+).{0,80}?MAE\s*[:=]\s*([0-9.eE+-]+).{0,80}?MAPE\s*[:=]\s*([0-9.eE+-]+)",
    ]
    m = re.search(patterns[1], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return pd.DataFrame([
            {"Metric": "RMSE", "Value": float(m.group(1))},
            {"Metric": "MAE", "Value": float(m.group(2))},
            {"Metric": "MAPE", "Value": float(m.group(3))},
        ])
    m = re.search(patterns[0], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        try:
            payload = ast.literal_eval(m.group(0))
            if isinstance(payload, dict):
                rows = []
                for key in ["RMSE", "MAE", "MAPE"]:
                    if key in payload:
                        rows.append({"Metric": key, "Value": payload[key]})
                if rows:
                    return pd.DataFrame(rows)
        except Exception:
            return None
    return None


def load_metrics_from_logs(log_files: list[Path]) -> pd.DataFrame | None:
    for path in log_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        parsed = _parse_metrics_from_text(text)
        if parsed is not None and not parsed.empty:
            return parsed
    return None


def _extract_series_like(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return list(payload)
    if isinstance(payload, dict):
        for key in ["y_true", "true", "gt", "labels", "label", "predictions", "prediction", "y_pred", "pred"]:
            if key in payload:
                return _extract_series_like(payload[key])
    return None


def load_predictions(prediction_files: list[Path]) -> pd.DataFrame | None:
    for path in prediction_files:
        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            elif path.suffix.lower() == ".pkl":
                with path.open("rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, pd.DataFrame):
                    return payload
                if isinstance(payload, list):
                    return pd.DataFrame(payload)
                if isinstance(payload, dict):
                    y_true = None
                    y_pred = None
                    for key in ["y_true", "true", "gt", "labels", "label"]:
                        if key in payload:
                            y_true = _extract_series_like(payload[key])
                            break
                    for key in ["y_pred", "pred", "prediction", "predictions"]:
                        if key in payload:
                            y_pred = _extract_series_like(payload[key])
                            break
                    if y_true is not None and y_pred is not None:
                        n = min(len(y_true), len(y_pred))
                        return pd.DataFrame({"y_true": list(y_true)[:n], "y_pred": list(y_pred)[:n]})
                    if "scores" in payload and isinstance(payload["scores"], dict):
                        return pd.DataFrame([payload["scores"]])
        except Exception:
            continue
    return None
