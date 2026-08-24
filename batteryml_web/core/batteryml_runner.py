from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .results_reader import ensure_metrics_csv_from_log


def run_batteryml(config_path, output_dir, device="cpu", extra_args=None, cli_args=None):
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.txt"

    cmd = [
        sys.executable,
        "batteryml_web/core/run_batteryml_cli.py",
        "run",
        str(config_path),
        "--workspace",
        str(output_dir),
        "--device",
        device,
    ]

    # Add extra args (like --train, --eval, --seed)
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    log_path.write_text(
        "COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + (stdout or "") + "\n\nSTDERR:\n" + (stderr or ""),
        encoding="utf-8",
    )
    ensure_metrics_csv_from_log(output_dir)
    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "log_path": str(log_path),
        "command": cmd,
    }


def run_dry_run(config_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    repo_root = Path(__file__).resolve().parents[2]
    report_path = output_dir / "dry_run_report.txt"
    cmd = [sys.executable, "batteryml_web/core/run_batteryml_cli.py", "run", str(config_path), "--workspace", str(output_dir), "--device", "cpu"]
    report_path.write_text(
        "DRY RUN COMMAND:\n" + " ".join(cmd) + "\n\nNOTE:\nBatteryML native dry-run is not exposed in this repository, so this report records a preflight command only.\n",
        encoding="utf-8",
    )
    return {"success": True, "report_path": str(report_path), "command": cmd, "cwd": str(repo_root)}
