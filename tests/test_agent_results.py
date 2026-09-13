"""Backend-only tests for the deterministic result-reading layer.

No training, no subprocess, no real LLM calls: synthetic workspaces use
small pickled ``{'prediction', 'scores', 'seed'}`` payloads plus stub
files, while the real 1-epoch smoke workspace
(``runs/agent_smoke_cptransformer``) is only inspected read-only for its
canonical saved scores when present.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_results -v
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.result_agent import explain_run_result  # noqa: E402
from batteryml_web.agent.results import (  # noqa: E402
    SMOKE_RUN_WARNING,
    build_result_context,
    build_result_interpretation_prompt,
)

SMOKE_WORKSPACE = REPO_ROOT / "runs" / "agent_smoke_cptransformer"
SMOKE_PREDICTIONS = SMOKE_WORKSPACE / "predictions_seed_0_20260914003852.pkl"
SMOKE_SCORES = {
    "RMSE": 455.5853271484375,
    "MAE": 393.8516540527344,
    "MAPE": 0.6621854901313782,
}


def _write_predictions(path: Path, payload: dict) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


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


def _synthetic_workspace(
    base: Path, *, scores: dict | None, name: str = "ws"
) -> Path:
    workspace = base / name
    workspace.mkdir(parents=True)
    payload: dict = {"prediction": [1.0, 2.0, 3.0], "seed": 0}
    if scores is not None:
        payload["scores"] = scores
    _write_predictions(workspace / "predictions_seed_0_test.pkl", payload)
    (workspace / "latest.ckpt").write_bytes(b"x")
    (workspace / "config_test.yaml").write_text(MATR_CONFIG_COPY, encoding="utf-8")
    return workspace


class TestStructuredResultContext(unittest.TestCase):
    """Canonical metrics only; nothing fabricated."""

    def test_successful_workspace_reports_canonical_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _synthetic_workspace(
                Path(tmp),
                scores={"RMSE": 10.0, "MAE": 8.0, "MAPE": 0.5},
                name="agent_ok",
            )
            context = build_result_context(
                workspace,
                dataset="MATR",
                task="RUL prediction",
                model="cptransformer",
                config_path="configs/external/cptransformer/matr_1.yaml",
            )

            self.assertTrue(context["success"])
            self.assertEqual(context["run_status"], "SUCCESS")
            self.assertEqual(
                context["metrics"], {"RMSE": 10.0, "MAE": 8.0, "MAPE": 0.5}
            )
            self.assertEqual(context["num_predictions"], 3)
            self.assertTrue(str(context["predictions_file"]).endswith(".pkl"))
            self.assertTrue(context["checkpoint_files"])
            self.assertTrue(context["config_files"])
            for paths in (
                context["figure_files"],
                context["checkpoint_files"],
                context["log_files"],
                context["config_files"],
            ):
                for path in paths:
                    self.assertTrue(Path(path).is_file(), path)

    def test_metrics_come_only_from_saved_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _synthetic_workspace(Path(tmp), scores=None)
            context = build_result_context(workspace, smoke_run=False)

        self.assertTrue(context["success"])
        self.assertEqual(context["metrics"], {})
        self.assertTrue(
            any("scores" in warning for warning in context["warnings"])
        )
        prompt = build_result_interpretation_prompt(context)
        self.assertIn("unavailable", prompt)

    def test_no_prediction_file_means_no_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "empty_ws"
            workspace.mkdir()
            (workspace / "latest.ckpt").write_bytes(b"x")
            context = build_result_context(workspace)

        self.assertEqual(context["metrics"], {})
        self.assertIsNone(context["predictions_file"])
        self.assertTrue(
            any("No predictions artifact" in w for w in context["warnings"])
        )

    def test_failed_run_never_successful(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _synthetic_workspace(
                Path(tmp), scores={"RMSE": 1.0, "MAE": 1.0, "MAPE": 0.1}
            )
            failed = build_result_context(
                workspace,
                run_status="FAILED",
                execution_result={"success": False, "returncode": 1},
            )
            mismatch = build_result_context(
                workspace, execution_result={"success": False, "returncode": 1}
            )

        for context in (failed, mismatch):
            self.assertFalse(context["success"])
            self.assertEqual(context["run_status"], "FAILED")
            self.assertEqual(context["metrics"], {})

    def test_missing_workspace_safe(self) -> None:
        context = build_result_context("runs/__definitely_no_such_workspace__")

        self.assertFalse(context["success"])
        self.assertEqual(context["metrics"], {})
        self.assertTrue(context["warnings"])

    def test_mape_ratio_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _synthetic_workspace(
                Path(tmp), scores={"RMSE": 1.0, "MAE": 1.0, "MAPE": 0.662}
            )
            context = build_result_context(workspace)

        self.assertEqual(context["metrics"]["MAPE"], 0.662)
        prompt = build_result_interpretation_prompt(context)
        self.assertIn("0.662", prompt)
        self.assertIn("66.20%", prompt)

    def test_smoke_warning_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _synthetic_workspace(
                Path(tmp), scores={"RMSE": 1.0, "MAE": 1.0, "MAPE": 0.1},
                name="agent_smoke_x",
            )
            context = build_result_context(
                workspace, dataset="MATR", model="cptransformer", smoke_run=True
            )

        self.assertIn(SMOKE_RUN_WARNING, context["warnings"])
        prompt = build_result_interpretation_prompt(context)
        self.assertIn("smoke run", prompt.lower())
        self.assertIn("not a performance benchmark", prompt.lower())


class TestInterpretationBoundary(unittest.TestCase):
    """The LLM explains; it never supplies the numbers."""

    def test_stubbed_interpretation_keeps_metrics(self) -> None:
        seen: dict = {}

        def stub(prompt: str, *, system_prompt: str = "", **kwargs: object) -> str:
            seen["prompt"] = prompt
            seen["system"] = system_prompt
            return json.dumps({"explanation": "smoke output only", "caveats": "n/a"})

        with tempfile.TemporaryDirectory() as tmp:
            workspace = _synthetic_workspace(
                Path(tmp), scores={"RMSE": 2.0, "MAE": 1.5, "MAPE": 0.2}
            )
            context = build_result_context(workspace, smoke_run=True)
            parsed = explain_run_result(
                context,
                base_url="http://stub",
                api_key="stub",
                model="stub",
                llm_call=stub,
            )

        self.assertEqual(parsed["metrics"], {"RMSE": 2.0, "MAE": 1.5, "MAPE": 0.2})
        self.assertIs(parsed["result_context"], context)
        self.assertIn("2.0", seen["prompt"])
        self.assertIn("smoke", seen["prompt"].lower())

    def test_empty_metrics_interpreted_honestly(self) -> None:
        def stub(prompt: str, *, system_prompt: str = "", **kwargs: object) -> str:
            self.assertIn("unavailable", prompt)
            return json.dumps({"explanation": "no metrics saved", "caveats": "n/a"})

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            context = build_result_context(workspace)
            parsed = explain_run_result(
                context,
                base_url="http://stub",
                api_key="stub",
                model="stub",
                llm_call=stub,
            )

        self.assertEqual(parsed["metrics"], {})

    def test_failed_run_with_metrics_refused(self) -> None:
        context = {
            "success": False,
            "metrics": {"RMSE": 1.0},
            "warnings": [],
        }
        with self.assertRaises(ValueError):
            explain_run_result(
                context,
                base_url="http://stub",
                api_key="stub",
                model="stub",
                llm_call=lambda prompt, **kwargs: "{}",
            )


class TestRealSmokeWorkspace(unittest.TestCase):
    """Read-only inspection of the real 1-epoch smoke artifacts."""

    @unittest.skipUnless(
        SMOKE_PREDICTIONS.is_file(), "real smoke workspace not present"
    )
    def test_real_smoke_scores_recovered(self) -> None:
        context = build_result_context(
            SMOKE_WORKSPACE,
            dataset="MATR",
            task="RUL prediction",
            model="cptransformer",
            config_path="configs/external/cptransformer/matr_1.yaml",
            execution_result={"success": True, "returncode": 0},
            smoke_run=True,
        )

        self.assertTrue(context["success"])
        for key, expected in SMOKE_SCORES.items():
            self.assertAlmostEqual(context["metrics"][key], expected, places=9)
        self.assertEqual(context["metrics"]["MAPE"], SMOKE_SCORES["MAPE"])
        self.assertEqual(context["num_predictions"], 42)
        self.assertTrue(str(context["predictions_file"]).endswith(".pkl"))
        self.assertTrue(context["checkpoint_files"])
        self.assertIn(SMOKE_RUN_WARNING, context["warnings"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
