"""Hardening tests for the result reader: CPU-safe, multi-run, provenance.

Backend-only: no training, no subprocess, no real LLM calls. Synthetic
workspaces use small pickled payloads; the real smoke workspace is only
inspected read-only. These tests extend (not replace)
``tests/test_agent_results.py``.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_results_hardening -v
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.results import (  # noqa: E402
    _load_prediction_payload,
    build_result_context,
)

SMOKE_WORKSPACE = REPO_ROOT / "runs" / "agent_smoke_cptransformer"
SMOKE_PREDICTIONS = SMOKE_WORKSPACE / "predictions_seed_0_20260914003852.pkl"
SMOKE_SCORES = {
    "RMSE": 455.5853271484375,
    "MAE": 393.8516540527344,
    "MAPE": 0.6621854901313782,
}

MATR_CONFIG_COPY = """\
model:
    name: 'CPTransformerRULPredictor'
train_test_split:
    name: 'MATRPrimaryTestTrainTestSplitter'
    cell_data_path: 'data/processed/MATR'
feature:
    name: 'BatteryLifeCycleCurveFeatureExtractor'
label:
    name: 'RULLabelAnnotator'
"""


def _write_predictions(path: Path, payload: dict) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _payload(scores: dict, n: int = 3) -> dict:
    return {
        "prediction": [float(i) for i in range(n)],
        "scores": dict(scores),
        "seed": 0,
    }


def _provenance_kwargs(config_name: str = "configs/external/cptransformer/matr_1.yaml") -> dict:
    # NOTE: config_path is the repo-level source config when the workspace
    # holds the *executed copy* (config_<timestamp>.yaml), so provenance
    # tests pass the caller config only where the task demands it. Tests
    # that check dataset/model content matching omit config_path to avoid
    # conflating "source path != copy path" with a real contradiction;
    # the dedicated execution-config-mismatch test covers that rule.
    return {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": "cptransformer",
        "config_path": config_name,
    }


def _content_kwargs() -> dict:
    return {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": "cptransformer",
    }


def _workspace_with_config(
    base: Path, *, name: str, scores: dict, config_text: str = MATR_CONFIG_COPY
) -> Path:
    workspace = base / name
    workspace.mkdir(parents=True)
    _write_predictions(
        workspace / "predictions_seed_0_20250101.pkl", _payload(scores)
    )
    (workspace / "config_20250101.yaml").write_text(config_text, encoding="utf-8")
    (workspace / "latest.ckpt").write_bytes(b"x")
    return workspace


class TestCpuSafeLoading(unittest.TestCase):
    """CUDA-origin artifacts must load without a visible GPU."""

    @unittest.skipUnless(SMOKE_PREDICTIONS.is_file(), "real smoke workspace not present")
    def test_real_cuda_artifact_loads_with_no_visible_gpu(self) -> None:
        script = (
            "import sys; sys.path.insert(0, '.');"
            "from pathlib import Path;"
            "from batteryml_web.agent.results import _load_prediction_payload;"
            "obj = _load_prediction_payload(Path('"
            + SMOKE_PREDICTIONS.as_posix()
            + "'));"
            "assert obj is not None, 'payload failed to load';"
            "assert sorted(obj['scores'].keys()) == ['MAE', 'MAPE', 'RMSE'];"
            "print('device=' + str(obj['prediction'].device));"
            "print('scores=' + repr(obj['scores']))"
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        self.assertEqual(
            proc.returncode, 0, f"CPU-only load failed:\n{proc.stdout}\n{proc.stderr}"
        )
        self.assertIn("device=cpu", proc.stdout)
        for key, expected in SMOKE_SCORES.items():
            self.assertIn(repr(expected), proc.stdout)

    def test_cpu_only_payload_needs_no_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.pkl"
            _write_predictions(path, _payload(SMOKE_SCORES))
            payload = _load_prediction_payload(path)

        assert payload is not None
        self.assertEqual(payload["scores"], SMOKE_SCORES)

    def test_no_torch_global_modified_by_load(self) -> None:
        import torch.serialization as serialization
        import torch.storage as storage

        before_validate = serialization._validate_device
        before_loader = storage._load_from_bytes
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.pkl"
            _write_predictions(path, _payload(SMOKE_SCORES))
            payload = _load_prediction_payload(path)
            second = _load_prediction_payload(path)
        assert payload is not None and second is not None
        self.assertEqual(payload["scores"], SMOKE_SCORES)
        self.assertEqual(second["scores"], SMOKE_SCORES)
        self.assertIs(serialization._validate_device, before_validate)
        self.assertIs(storage._load_from_bytes, before_loader)

    def test_two_consecutive_loads_behave_normally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.pkl"
            _write_predictions(path, _payload(SMOKE_SCORES))
            first = _load_prediction_payload(path)
            second = _load_prediction_payload(path)
        assert first is not None and second is not None
        self.assertEqual(first["scores"], second["scores"])

    def test_saved_artifact_not_modified(self) -> None:
        if not SMOKE_PREDICTIONS.is_file():
            self.skipTest("real smoke workspace not present")
        digest_before = __import__("hashlib").sha256(
            SMOKE_PREDICTIONS.read_bytes()
        ).hexdigest()
        _load_prediction_payload(SMOKE_PREDICTIONS)
        digest_after = __import__("hashlib").sha256(
            SMOKE_PREDICTIONS.read_bytes()
        ).hexdigest()
        self.assertEqual(digest_before, digest_after)


class TestRepeatedWorkspaceSelection(unittest.TestCase):
    """Two prediction files must never resolve by glob order."""

    def _two_file_workspace(self, base: Path) -> tuple[Path, Path, Path]:
        workspace = base / "agent_repeat"
        workspace.mkdir(parents=True)
        old = workspace / "predictions_seed_0_20250101.pkl"
        new = workspace / "predictions_seed_0_20250202.pkl"
        _write_predictions(old, _payload({"RMSE": 100.0, "MAE": 90.0, "MAPE": 0.9}))
        _write_predictions(new, _payload({"RMSE": 1.0, "MAE": 0.9, "MAPE": 0.09}))
        (workspace / "config_20250101.yaml").write_text(
            MATR_CONFIG_COPY, encoding="utf-8"
        )
        return workspace, old, new

    def test_two_files_without_execution_result_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, _, _ = self._two_file_workspace(Path(tmp))
            context = build_result_context(workspace, **_provenance_kwargs())

        self.assertFalse(context["success"])
        self.assertEqual(context["metrics"], {})
        self.assertIsNone(context["predictions_file"])
        self.assertTrue(
            any("Multiple predictions" in w for w in context["warnings"]),
            context["warnings"],
        )

    def test_execution_artifacts_select_exact_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, _, new = self._two_file_workspace(Path(tmp))
            context = build_result_context(
                workspace,
                execution_result={
                    "success": True,
                    "returncode": 0,
                    "artifacts": {"prediction_files": [str(new)]},
                },
                **_provenance_kwargs(),
            )

        self.assertTrue(context["success"], context["warnings"])
        self.assertEqual(context["predictions_file"], str(new))
        self.assertEqual(context["metrics"]["RMSE"], 1.0)

    def test_execution_artifacts_naming_absent_file_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            context = build_result_context(
                workspace,
                execution_result={
                    "success": True,
                    "returncode": 0,
                    "artifacts": {
                        "prediction_files": [str(workspace / "predictions_seed_0_20990101.pkl")]
                    },
                },
                **_provenance_kwargs(),
            )

        self.assertTrue(context["success"], context["warnings"])
        self.assertTrue(
            any("no predictions file" in w.lower() for w in context["warnings"]),
            context["warnings"],
        )

    def test_single_file_fallback_selects_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            context = build_result_context(workspace, **_provenance_kwargs())

        self.assertTrue(context["success"], context["warnings"])
        self.assertTrue(str(context["predictions_file"]).endswith(".pkl"))
        self.assertEqual(context["metrics"], SMOKE_SCORES)


class TestProvenanceVerification(unittest.TestCase):
    """Caller claims must match workspace evidence, never filenames."""

    def test_matching_provenance_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            context = build_result_context(workspace, **_provenance_kwargs())

        self.assertTrue(context["provenance_verified"])
        self.assertEqual(context["provenance_warnings"], [])
        self.assertTrue(context["success"])

    def test_dataset_mismatch_surfaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            context = build_result_context(
                workspace,
                dataset="HUST",
                task="RUL prediction",
                model="cptransformer",
            )

        self.assertFalse(context["provenance_verified"])
        self.assertTrue(
            any("HUST" in w and "MATR" in w for w in context["provenance_warnings"]),
            context["provenance_warnings"],
        )
        # Contradiction: must not present as a clean run for HUST.
        self.assertFalse(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)

    def test_model_mismatch_surfaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            context = build_result_context(
                workspace,
                dataset="MATR",
                task="RUL prediction",
                model="batlinet",
            )

        self.assertFalse(context["provenance_verified"])
        # Contradiction: must not present as a clean verified run.
        self.assertFalse(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)

    def test_plan_config_mismatch_surfaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            context = build_result_context(
                workspace,
                dataset="MATR",
                task="RUL prediction",
                model="cptransformer",
                config_path="configs/external/batlinet/matr_1.yaml",
                execution_result={
                    "success": True,
                    "returncode": 0,
                    "command": [
                        "python",
                        "batteryml_web/core/run_batteryml_cli.py",
                        "run",
                        "configs/external/cptransformer/matr_1.yaml",
                        "--train",
                    ],
                },
            )

        self.assertFalse(context["provenance_verified"])
        self.assertFalse(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)

    def test_missing_config_copy_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "agent_nocfg"
            workspace.mkdir()
            _write_predictions(
                workspace / "predictions_seed_0_x.pkl", _payload(SMOKE_SCORES)
            )
            context = build_result_context(workspace, **_provenance_kwargs())

        self.assertFalse(context["provenance_verified"])
        # No config evidence: metrics readable but not a clean verified run.
        self.assertTrue(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)
        self.assertTrue(
            any("config copy" in w for w in context["provenance_warnings"]),
            context["provenance_warnings"],
        )

    def test_two_configs_without_execution_result_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "agent_multi"
            workspace.mkdir()
            _write_predictions(
                workspace / "predictions_seed_0_a.pkl", _payload(SMOKE_SCORES)
            )
            (workspace / "config_20250101.yaml").write_text(
                MATR_CONFIG_COPY, encoding="utf-8"
            )
            (workspace / "config_20250202.yaml").write_text(
                MATR_CONFIG_COPY, encoding="utf-8"
            )
            context = build_result_context(workspace, **_provenance_kwargs())

        self.assertFalse(context["provenance_verified"])
        self.assertTrue(
            any("config" in w.lower() for w in context["provenance_warnings"]),
            context["provenance_warnings"],
        )
        # Metrics are still read; only clean verified provenance is withheld.
        self.assertEqual(context["metrics"], SMOKE_SCORES)
        self.assertTrue(context["success"])

    def test_execution_command_config_selects_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "agent_multi"
            workspace.mkdir()
            _write_predictions(
                workspace / "predictions_seed_0_a.pkl", _payload(SMOKE_SCORES)
            )
            (workspace / "config_20250101.yaml").write_text(
                MATR_CONFIG_COPY.replace("data/processed/MATR", "data/processed/HUST"),
                encoding="utf-8",
            )
            (workspace / "config_20250202.yaml").write_text(
                MATR_CONFIG_COPY, encoding="utf-8"
            )
            second = str(workspace / "config_20250202.yaml")
            pred = str(workspace / "predictions_seed_0_a.pkl")
            context = build_result_context(
                workspace,
                execution_result={
                    "success": True,
                    "returncode": 0,
                    "command": [
                        "python",
                        "batteryml_web/core/run_batteryml_cli.py",
                        "run",
                        "configs/external/cptransformer/matr_1.yaml",
                    ],
                    "artifacts": {
                        "prediction_files": [pred],
                        "config_files": [second],
                    },
                },
                **_provenance_kwargs(),
            )

        self.assertTrue(context["provenance_verified"], context["provenance_warnings"])
        self.assertTrue(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)

    def test_execution_config_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            context = build_result_context(
                workspace,
                dataset="MATR",
                task="RUL prediction",
                model="cptransformer",
                config_path="configs/external/batlinet/matr_1.yaml",
                execution_result={
                    "success": True,
                    "returncode": 0,
                    "command": [
                        "python",
                        "batteryml_web/core/run_batteryml_cli.py",
                        "run",
                        "configs/external/cptransformer/matr_1.yaml",
                    ],
                },
            )

        self.assertFalse(context["provenance_verified"])
        self.assertFalse(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)

    def test_matching_execution_config_and_prediction_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_config(
                Path(tmp), name="agent_ok", scores=SMOKE_SCORES
            )
            pred = str(workspace / "predictions_seed_0_20250101.pkl")
            cfg = str(workspace / "config_20250101.yaml")
            context = build_result_context(
                workspace,
                execution_result={
                    "success": True,
                    "returncode": 0,
                    "command": [
                        "python",
                        "batteryml_web/core/run_batteryml_cli.py",
                        "run",
                        "configs/external/cptransformer/matr_1.yaml",
                    ],
                    "artifacts": {
                        "prediction_files": [pred],
                        "config_files": [cfg],
                    },
                },
                **_provenance_kwargs(),
            )

        self.assertTrue(context["provenance_verified"], context["provenance_warnings"])
        self.assertTrue(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)

    def test_ambiguous_config_with_tied_prediction_reads_metrics_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "agent_multi"
            workspace.mkdir()
            _write_predictions(
                workspace / "predictions_seed_0_a.pkl", _payload(SMOKE_SCORES)
            )
            (workspace / "config_20250101.yaml").write_text(
                MATR_CONFIG_COPY, encoding="utf-8"
            )
            (workspace / "config_20250202.yaml").write_text(
                MATR_CONFIG_COPY, encoding="utf-8"
            )
            context = build_result_context(
                workspace,
                execution_result={
                    "success": True,
                    "returncode": 0,
                    "artifacts": {
                        "prediction_files": [
                            str(workspace / "predictions_seed_0_a.pkl")
                        ]
                    },
                },
                **_provenance_kwargs(),
            )

        self.assertFalse(context["provenance_verified"])
        self.assertTrue(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)

    @unittest.skipUnless(SMOKE_PREDICTIONS.is_file(), "real smoke workspace not present")
    def test_real_smoke_metrics_still_exact(self) -> None:
        context = build_result_context(
            SMOKE_WORKSPACE,
            execution_result={"success": True, "returncode": 0},
            smoke_run=True,
            **_provenance_kwargs(),
        )

        self.assertTrue(context["provenance_verified"], context["provenance_warnings"])
        self.assertTrue(context["success"])
        self.assertEqual(context["metrics"], SMOKE_SCORES)
        self.assertEqual(context["num_predictions"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
