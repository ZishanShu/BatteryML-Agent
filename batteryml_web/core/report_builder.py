from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .results_reader import list_figures, read_log, read_metrics, read_predictions


def _config_summary(config_text: str) -> Dict[str, Any]:
    try:
        import yaml

        cfg = yaml.safe_load(config_text) or {}
    except Exception:
        return {}

    return {
        "dataset": cfg.get("dataset") or cfg.get("data") or cfg.get("train_test_split", {}).get("cell_data_path"),
        "task": cfg.get("task") or cfg.get("label", {}).get("name"),
        "model": cfg.get("model", {}).get("name") if isinstance(cfg.get("model"), dict) else cfg.get("model"),
        "splitter": cfg.get("train_test_split", {}).get("name"),
        "label": cfg.get("label", {}).get("name") if isinstance(cfg.get("label"), dict) else cfg.get("label"),
        "feature_extractor": cfg.get("feature", {}).get("name") if isinstance(cfg.get("feature"), dict) else cfg.get("feature"),
        "workspace": cfg.get("workspace") or cfg.get("output_dir"),
    }


def generate_html_report(project_dir: str | Path) -> str:
    project_dir = Path(project_dir)
    state_path = project_dir / "project_state.json"
    config_candidates = [project_dir / "generated_config.yaml", project_dir / "quickstart_config.yaml", project_dir / "config.yaml"]
    schema_path = project_dir / "schema_mapping.json"

    project_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    config_text = ""
    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            config_text = candidate.read_text(encoding="utf-8")
            break
    schema_mapping = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    metrics = read_metrics(project_dir)
    preds = read_predictions(project_dir)
    log_text = read_log(project_dir)
    figures = list_figures(project_dir)
    summary = _config_summary(config_text) if config_text else {}

    html = [
        "<html><head><meta charset='utf-8'><title>BatteryML Report</title></head><body>",
        "<h1>BatteryML Report</h1>",
        f"<p><strong>Project:</strong> {project_dir}</p>",
        "<h2>Config Summary</h2>",
        f"<pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>",
        "<h2>Project State</h2>",
        f"<pre>{json.dumps(project_state, indent=2, ensure_ascii=False)}</pre>",
        "<h2>Config</h2>",
        f"<pre>{config_text}</pre>",
        "<h2>Schema Mapping</h2>",
        f"<pre>{json.dumps(schema_mapping, indent=2, ensure_ascii=False)}</pre>",
        "<h2>Metrics Table</h2>",
        f"<pre>{json.dumps(metrics.get('metrics', []), indent=2, ensure_ascii=False)}</pre>",
        "<h2>Prediction Plot</h2>",
        f"<pre>{json.dumps(preds.get('parsed', {}), default=str, indent=2, ensure_ascii=False)}</pre>",
        "<h2>File Paths</h2>",
        f"<pre>{json.dumps({'project_dir': str(project_dir), 'config_path': str(config_path) if config_path else None, 'training_log': str(project_dir / 'training_log.txt')}, indent=2, ensure_ascii=False)}</pre>",
        "<h2>Figures</h2>",
        "<ul>",
    ]
    for fig in figures:
        html.append(f"<li>{fig}</li>")
    html.extend([
        "</ul>",
        "<h2>Training Log</h2>",
        f"<p>{project_dir / 'training_log.txt'}</p>",
        f"<pre>{log_text}</pre>",
        "</body></html>",
    ])
    return "\n".join(html)
