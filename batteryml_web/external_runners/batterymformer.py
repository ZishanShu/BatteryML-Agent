"""Readiness and command adapter for BatteryMFormer trajectory forecasting."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


class BatteryMFormerRunner:
    model = "BatteryMFormer"
    task = "Degradation trajectory forecasting"

    def __init__(self, repo_root: str | Path | None = None, config: Mapping[str, Any] | None = None):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        self.config = dict(config or {})
        self.source_root = self.repo_root / "third_party" / "BatteryMFormer"

    def readiness_check(self) -> dict[str, Any]:
        missing: list[str] = []
        if not (self.source_root / "models" / "BatteryMFormer.py").is_file():
            missing.append("third_party/BatteryMFormer/models/BatteryMFormer.py")
        if not (self.source_root / "run_main.py").is_file():
            missing.append("third_party/BatteryMFormer/run_main.py")
        if shutil.which("python") is None:
            missing.append("python executable")
        paths = self._paths()
        for label, path in paths.items():
            if not path or not Path(path).exists():
                missing.append(f"{label}: {path or '<unset>'}")
        status = "ready" if not missing else "missing_data" if any("data" in x.lower() or "soh" in x.lower() for x in missing) else "missing_asset"
        return {"model": self.model, "task": self.task, "ready": not missing, "status": status, "missing": missing}

    def _paths(self) -> dict[str, str | None]:
        return {
            "BatteryLife_v11": self.config.get("root_path"),
            "processed_SOH": self.config.get("processed_SOH_path"),
            "prompt_embeddings": self.config.get("prompt_embeddings_path") or str(self.source_root / "data_provider" / "prompt_embeddings" / "Qwen3_total.pkl"),
            "split_json": self.config.get("split_json_path"),
        }

    def _command(self, extra_args: Sequence[str] = ()) -> list[str]:
        readiness = self.readiness_check()
        if not readiness["ready"]:
            raise RuntimeError(f"BatteryMFormer is not executable ({readiness['status']}): {json.dumps(readiness['missing'])}")
        args = self.config.get("args", {})
        command = ["python", str(self.source_root / "run_main.py"), "--model", self.model]
        for key, value in args.items():
            command.extend((f"--{key}", str(value)))
        return [*command, *extra_args]

    def build_train_command(self, extra_args: Sequence[str] = ()) -> list[str]:
        return self._command(extra_args)

    def build_eval_command(self, extra_args: Sequence[str] = ()) -> list[str]:
        return self._command(extra_args)

    def normalize_results(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "predictions": [dict(row) for row in rows],
            "prediction_schema": ["cell_id", "cycle_index", "observed_soh", "predicted_soh"],
            "metrics": {"trajectory_rmse": None, "trajectory_mae": None},
        }
