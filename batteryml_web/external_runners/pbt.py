"""Readiness and command adapter for the official PBT implementation."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


class PBTRunner:
    """Run PBT in its isolated conda environment without importing it here."""

    model = "PBT"
    task = "RUL prediction"
    environment = "pbt311"

    def __init__(self, repo_root: str | Path | None = None, config: Mapping[str, Any] | None = None):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        self.config = dict(config or {})
        self.source_root = self.repo_root / "third_party" / "PBT"

    def readiness_check(self) -> dict[str, Any]:
        missing: list[str] = []
        source = self.source_root / "models" / "PBT.py"
        if not source.is_file():
            missing.append(str(source))
        if shutil.which("conda") is None:
            return self._report("dependency_unavailable", ["conda executable (required for pbt311)"] + missing)
        probe = subprocess.run(
            ["conda", "run", "-n", self.environment, "python", "-c", "from models import PBT"],
            cwd=self.source_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode:
            missing.append(f"import PBT in conda environment {self.environment}: {probe.stderr.strip()}")
        paths = self._asset_paths()
        for label, path in paths.items():
            if not path:
                missing.append(label)
            elif not Path(path).exists():
                missing.append(f"{label}: {path}")
        dataset = self.config.get("dataset_path")
        if not dataset or not Path(str(dataset)).exists():
            missing.append(f"dataset_path: {dataset or '<unset>'}")
        status = "ready" if not missing else "missing_checkpoint" if any("checkpoint" in x.lower() for x in missing) else "missing_asset"
        return self._report(status, missing)

    def _asset_paths(self) -> dict[str, str | None]:
        assets = self.config.get("assets", {})
        return {
            "pretrained_checkpoint": assets.get("pretrained_checkpoint"),
            "aging_condition_embeddings": assets.get("aging_condition_embeddings"),
            "pca_path": assets.get("pca_path"),
            "args_path": assets.get("args_path"),
            "LLM_path": assets.get("LLM_path"),
        }

    def _report(self, status: str, missing: Sequence[str]) -> dict[str, Any]:
        return {"model": self.model, "task": self.task, "ready": not missing, "status": status, "missing": list(missing)}

    def build_command(self, script: str = "finetune_model.py", extra_args: Sequence[str] = ()) -> list[str]:
        readiness = self.readiness_check()
        if not readiness["ready"]:
            raise RuntimeError(f"PBT is not executable ({readiness['status']}): {json.dumps(readiness['missing'])}")
        return ["conda", "run", "-n", self.environment, "python", str(self.source_root / script), *extra_args]

    def run_train_or_finetune(self, extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
        return subprocess.run(self.build_command(extra_args=extra_args), cwd=self.source_root, check=True, text=True)

    def run_evaluation(self, extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
        return subprocess.run(self.build_command("evaluate_model.py", extra_args), cwd=self.source_root, check=True, text=True)

    @staticmethod
    def normalize_results(result: Mapping[str, Any]) -> dict[str, Any]:
        return {"model": "PBT", "task": "RUL prediction", "predictions": result.get("predictions"), "metrics": result.get("metrics", {})}
