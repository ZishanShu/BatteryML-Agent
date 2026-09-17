"""Resume is reachable from the web, and only where it is真 supported.

The backend could already resume; the button was disabled. These cover the
wiring: what gates the control, what the click builds, and that a resumed run
is an ordinary run — new id, own workspace, own manifest, stoppable — that
happens to start from a checkpoint.

No Streamlit runtime, no GPU, no training.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_resume_launch -v
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from batteryml_web.agent.execution import build_execution_plan  # noqa: E402
from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.training_checkpoint import (  # noqa: E402
    build_checkpoint,
    classify_checkpoint,
    save_checkpoint,
    validate_resume,
)
from batteryml_web.ui import run_history as history  # noqa: E402


def _identity(**overrides):
    base = {
        "model_key": "cnn",
        "dataset": "MATR",
        "task": "RUL prediction",
        "seed": "0",
    }
    base.update(overrides)
    return base


class _ResumeFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def _resumable(self, epoch: int = 40, target: int = 100, **identity) -> str:
        payload = build_checkpoint(
            model_state=nn.Linear(2, 1).state_dict(),
            optimizer_state={"state": {}, "param_groups": []},
            epoch=epoch,
            target_total_epochs=target,
            training_identity=_identity(**identity),
        )
        return str(save_checkpoint(payload, self.directory / f"e{epoch}.resume.ckpt"))

    def _weights_only(self) -> str:
        path = self.directory / "legacy.ckpt"
        torch.save(nn.Linear(2, 1).state_dict(), str(path))
        return str(path)

    def _manifest(self, model_key="cnn", family="nn_model", checkpoints=(), **extra):
        manifest = {
            "run_id": "run_A",
            "project": "alpha",
            "model_key": model_key,
            "model_family": family,
            "dataset": "MATR",
            "task": "RUL prediction",
            "seed": "0",
            "device": "cpu",
            "artifacts": {"checkpoint_files": list(checkpoints)},
        }
        manifest.update(extra)
        return manifest


class TestResumeGate(unittest.TestCase):
    """§1: every condition is backend-derived and all must hold."""

    def test_the_ui_names_no_model_when_deciding(self) -> None:
        source = (REPO_ROOT / "batteryml_web/ui/run_history.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree)
        for name in ("cnn", "lstm", "batlinet", "BatLiNet", "xgb", "XGBoost",
                     "cptransformer", "CPTransformer"):
            self.assertNotIn(name, code)
        self.assertIn("resume_capability", code)


class TestResumeOffer(_ResumeFixture):
    """What the panel decides, without rendering anything."""

    def test_a_supported_model_with_a_resumable_checkpoint_is_offered(self) -> None:
        offer = history.resume_offer(
            self._manifest(checkpoints=[self._resumable(40, 100)])
        )
        self.assertTrue(offer["available"])
        self.assertEqual(offer["completed_epochs"], 40)
        self.assertEqual(offer["suggested_target"], 100)

    def test_an_unsupported_model_is_never_offered(self) -> None:
        for model, family in (
            ("batlinet", "nn_model"),   # its own loop persists no state
            ("xgb", ""),                # not epoch-trained
            ("cptransformer", ""),      # unaudited source
        ):
            with self.subTest(model=model):
                offer = history.resume_offer(
                    self._manifest(model, family, [self._resumable()])
                )
                self.assertFalse(offer["available"])

    def test_a_weights_only_checkpoint_is_not_enough(self) -> None:
        offer = history.resume_offer(
            self._manifest(checkpoints=[self._weights_only()])
        )
        self.assertFalse(offer["available"])

    def test_a_corrupt_checkpoint_is_not_enough(self) -> None:
        broken = self.directory / "broken.ckpt"
        broken.write_bytes(b"not a checkpoint")
        offer = history.resume_offer(self._manifest(checkpoints=[str(broken)]))
        self.assertFalse(offer["available"])

    def test_a_run_with_no_checkpoint_is_not_offered(self) -> None:
        self.assertFalse(history.resume_offer(self._manifest())["available"])

    def test_the_latest_resumable_checkpoint_wins(self) -> None:
        offer = history.resume_offer(
            self._manifest(
                checkpoints=[self._resumable(10, 100), self._resumable(40, 100)]
            )
        )
        self.assertEqual(offer["completed_epochs"], 40)

    def test_a_filename_alone_never_implies_resumability(self) -> None:
        """§9: only the schema decides."""
        misleading = self.directory / "resume_epoch_40.resume.ckpt"
        torch.save(nn.Linear(2, 1).state_dict(), str(misleading))
        self.assertFalse(
            history.resume_offer(self._manifest(checkpoints=[str(misleading)]))["available"]
        )


class TestResumeEpochSemantics(_ResumeFixture):
    """§4: 40 of 100 continues 41..100 — not 0..100, not 41..140."""

    def test_the_start_epoch_is_the_checkpoint_plus_one(self) -> None:
        verdict = classify_checkpoint(self._resumable(40, 100))
        result = validate_resume(verdict, _identity(), target_total_epochs=100)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["start_epoch"], 41)
        self.assertEqual(result["remaining"], 60)
        self.assertEqual(result["target_total_epochs"], 100)

    def test_a_target_at_or_below_the_checkpoint_is_blocked(self) -> None:
        verdict = classify_checkpoint(self._resumable(40, 100))
        for target in (40, 39, 1):
            with self.subTest(target=target):
                self.assertFalse(
                    validate_resume(verdict, _identity(), target_total_epochs=target)["ok"]
                )

    def test_the_plan_passes_the_total_target_not_additional_epochs(self) -> None:
        """The CLI receives --epochs 100, meaning stop at 100."""
        plan = build_execution_plan(
            "MATR", "RUL prediction", "cnn",
            workflow_context={
                "dataset": "MATR", "dataset_status": "READY",
                "task": "RUL prediction",
                "runnable_models": [
                    {"model": "cnn", "status": "READY",
                     "validated_config": "configs/x.yaml"}
                ],
                "blocked_models": [],
            },
            epochs=100,
            resume_from="/tmp/e40.resume.ckpt",
        )
        command = plan.get("command") or []
        if not command:
            self.skipTest(f"planner refused in this environment: {plan.get('reason')}")
        self.assertIn("--epochs", command)
        self.assertEqual(command[command.index("--epochs") + 1], "100")
        self.assertIn("--ckpt-to-resume", command)
        self.assertEqual(
            command[command.index("--ckpt-to-resume") + 1], "/tmp/e40.resume.ckpt"
        )

    def test_an_ordinary_plan_carries_no_resume_flag(self) -> None:
        source = (REPO_ROOT / "batteryml_web/agent/execution.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("if resume_from:", source)
        self.assertIn('"--ckpt-to-resume"', source)


class TestResumedRunIsAnOrdinaryRun(_ResumeFixture):
    """§3, §5: new run, own workspace, lineage, source untouched."""

    def test_the_lineage_block_records_where_it_continued_from(self) -> None:
        lineage = rm.build_resume_lineage(
            source_run_id="run_A",
            checkpoint_path="/runs/e40.resume.ckpt",
            starting_epoch=41,
            target_total_epochs=100,
            completed_epochs=40,
        )
        manifest = rm.build_manifest(
            run_id="run_B", project="alpha", workspace="w", dataset="MATR",
            task="RUL prediction", model_key="cnn", seed=0, resumed_from=lineage,
        )
        self.assertEqual(manifest["resumed_from"]["run_id"], "run_A")
        self.assertEqual(manifest["resumed_from"]["starting_epoch"], 41)
        self.assertEqual(manifest["resumed_from"]["target_total_epochs"], 100)
        self.assertEqual(manifest["resumed_from"]["completed_epochs"], 40)

    def test_the_source_run_is_never_modified(self) -> None:
        source_dir = self.directory / "run_A"
        rm.write_manifest(
            source_dir,
            rm.build_manifest(
                run_id="run_A", project="alpha", workspace=str(source_dir),
                dataset="MATR", task="RUL prediction", model_key="cnn", seed=0,
            ),
        )
        rm.update_manifest(source_dir, {}, status=rm.STATUS_RUNNING)
        rm.update_manifest(source_dir, {}, status=rm.STATUS_COMPLETED)
        before = rm.read_manifest(source_dir)

        resumed_dir = self.directory / "run_B"
        rm.write_manifest(
            resumed_dir,
            rm.build_manifest(
                run_id="run_B", project="alpha", workspace=str(resumed_dir),
                dataset="MATR", task="RUL prediction", model_key="cnn", seed=0,
                resumed_from=rm.build_resume_lineage(
                    source_run_id="run_A", checkpoint_path="c",
                    starting_epoch=41, target_total_epochs=100,
                    completed_epochs=40,
                ),
            ),
        )
        self.assertEqual(rm.read_manifest(source_dir), before)
        self.assertIsNone(before["resumed_from"])

    def test_an_ordinary_run_has_no_lineage(self) -> None:
        manifest = rm.build_manifest(
            run_id="run_1", project="p", workspace="w", dataset="d",
            task="t", model_key="m", seed=0,
        )
        self.assertIsNone(manifest["resumed_from"])

    def test_a_resumed_run_gets_its_own_workspace_in_the_current_project(self) -> None:
        from batteryml_web.core.project_context import resolve_project, run_workspace
        from batteryml_web.core.run_identity import is_run_id, new_run_id

        project = resolve_project(project_name="alpha")
        first = new_run_id(dataset="MATR", model="cnn", seed=0)
        second = new_run_id(dataset="MATR", model="cnn", seed=0)
        self.assertNotEqual(first, second)
        self.assertTrue(is_run_id(second))
        workspace = run_workspace(project, "MATR", "cnn", second)
        self.assertTrue(workspace.startswith("runs/web_projects/alpha/"))
        self.assertNotIn(first, workspace)

    def test_resume_launches_through_the_existing_run_manager(self) -> None:
        """§5: no second execution system."""
        source = (REPO_ROOT / "batteryml_web/ui/run_history.py").read_text(
            encoding="utf-8"
        )
        block = source.split("def _start_resumed_run")[1].split("\ndef ")[0]
        self.assertIn("_launch_run", block)
        self.assertIn("build_execution_plan", block)
        for forbidden in ("Popen", "subprocess", "execute_plan_streaming", "torchrun"):
            self.assertNotIn(forbidden, block)

    def test_a_resumed_run_can_be_stopped_like_any_other(self) -> None:
        """Stop is independent of how a run started."""
        self.assertTrue(rm.can_transition(rm.STATUS_RUNNING, rm.STATUS_CANCELLED))
        run_dir = self.directory / "run_B"
        rm.write_manifest(
            run_dir,
            rm.build_manifest(
                run_id="run_B", project="alpha", workspace=str(run_dir),
                dataset="MATR", task="RUL prediction", model_key="cnn", seed=0,
                resumed_from=rm.build_resume_lineage(
                    source_run_id="run_A", checkpoint_path="c", starting_epoch=41,
                    target_total_epochs=100, completed_epochs=40,
                ),
            ),
        )
        rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        rm.update_manifest(run_dir, {"metrics": {}}, status=rm.STATUS_CANCELLED)
        stored = rm.read_manifest(run_dir)
        self.assertEqual(stored["status"], rm.STATUS_CANCELLED)
        self.assertEqual(stored["metrics"], {})
        # Lineage survives a cancellation.
        self.assertEqual(stored["resumed_from"]["run_id"], "run_A")


class TestUpstreamResumeWiring(unittest.TestCase):
    """The pipeline distinguishes a resumable checkpoint from a warm start."""

    def _pipeline_source(self) -> str:
        return (REPO_ROOT / "batteryml/pipeline.py").read_text(encoding="utf-8")

    def test_a_resumable_checkpoint_reaches_fit_as_resume_state(self) -> None:
        source = self._pipeline_source()
        self.assertIn("resume_state = _load_resume_state(ckpt_to_resume)", source)
        self.assertIn("resume_state=resume_state", source)

    def test_a_weights_only_checkpoint_keeps_the_old_warm_start_path(self) -> None:
        source = self._pipeline_source()
        self.assertIn(
            "None if resume_state is not None else ckpt_to_resume", source
        )

    def test_the_rng_state_is_restored_before_training_continues(self) -> None:
        source = self._pipeline_source()
        restore = source.index("_restore_rng_state(resume_state")
        fit = source.index("model.fit(dataset")
        self.assertLess(restore, fit, "RNG must be restored before training")

    def test_models_with_their_own_fit_are_called_unchanged(self) -> None:
        """BatLiNet must not gain resume kwargs by accident."""
        source = self._pipeline_source()
        self.assertIn("_supported_fit_kwargs", source)

        class _OwnLoop:
            def fit(self, dataset, timestamp=None):
                return None

        sys.path.insert(0, str(REPO_ROOT))
        from batteryml.pipeline import _supported_fit_kwargs

        self.assertEqual(
            _supported_fit_kwargs(_OwnLoop(), resume_state={"a": 1}, on_epoch_end=print),
            {},
        )

    def test_compiled_and_wrapped_models_are_unwrapped_before_saving(self) -> None:
        from batteryml.pipeline import _unwrap

        plain = nn.Linear(2, 1)

        class _Compiled:
            def __init__(self, inner):
                self._orig_mod = inner

        self.assertIs(_unwrap(_Compiled(plain)), plain)
        self.assertIs(_unwrap(plain), plain)
        for key in _unwrap(_Compiled(plain)).state_dict():
            self.assertFalse(key.startswith("_orig_mod."))

    def test_the_pipeline_does_not_need_the_web_layer(self) -> None:
        """Standalone BatteryML must still work with no web layer present.

        This used to be satisfied by importing
        ``batteryml_web.core.training_checkpoint`` inside a ``try`` and
        degrading to a no-op. The schema now lives in the training layer
        (``batteryml.training_checkpoint``), so the pipeline needs no web
        import at all — a stronger property, and the one asserted here.
        ``tests/test_core_independence.py`` enforces it for the whole core
        package by AST; this keeps the claim next to the resume wiring.
        """
        from batteryml import pipeline, training_checkpoint

        self.assertIs(pipeline._resume_schema_module, training_checkpoint)
        self.assertNotIn("batteryml_web", self._pipeline_source())


if __name__ == "__main__":
    unittest.main()
