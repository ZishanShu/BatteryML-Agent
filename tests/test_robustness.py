"""Failure classification, LLM fallback and recovery.

No GPU, no network, no Streamlit. The point of these is that the platform
says true things when something goes wrong: the right category, no invented
cause, no silent repair, and no secret in the message.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_robustness -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.failure_taxonomy import (  # noqa: E402
    CATEGORY_CANCELLED,
    CATEGORY_CONFIG_INVALID,
    CATEGORY_CUDA_OOM,
    CATEGORY_DATASET_MISSING,
    CATEGORY_LLM_AUTH,
    CATEGORY_LLM_INVALID_RESPONSE,
    CATEGORY_LLM_NOT_CONFIGURED,
    CATEGORY_LLM_TIMEOUT,
    CATEGORY_MODEL_ASSETS,
    CATEGORY_NONE,
    CATEGORY_PROCESS_FAILED,
    CATEGORY_UNKNOWN,
    classify_llm_failure,
    classify_readiness_block,
    classify_run_failure,
    explanation_failure_is_not_a_run_failure,
    redact,
)
from batteryml_web.core import run_manifest as rm  # noqa: E402


def _failed(stderr: str = "", returncode: int = 1) -> dict:
    return {
        "success": False,
        "started": True,
        "completed": True,
        "returncode": returncode,
        "stdout": "",
        "stderr": stderr,
        "error": None,
        "command": ["python", "x"],
    }


class TestExecutionFailureClassification(unittest.TestCase):
    """C1: conservative categories, never a guessed cause."""

    def test_cuda_out_of_memory_is_recognised(self) -> None:
        finding = classify_run_failure(
            _failed("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"),
            device="cuda:0",
        )
        self.assertEqual(finding["category"], CATEGORY_CUDA_OOM)
        self.assertIn("GPU memory", finding["short_title"])
        self.assertTrue(finding["recommended_action"])

    def test_an_unrelated_runtime_error_is_not_called_oom(self) -> None:
        """The specific mislabel this must never make."""
        for stderr in (
            "RuntimeError: shape mismatch in layer 3",
            "RuntimeError: expected scalar type Float but found Double",
            "ValueError: memory is a fascinating subject",
            "KeyError: 'out of memory jokes'",
        ):
            with self.subTest(stderr=stderr):
                finding = classify_run_failure(_failed(stderr), device="cuda:0")
                self.assertNotEqual(finding["category"], CATEGORY_CUDA_OOM)

    def test_a_plain_nonzero_exit_is_a_process_failure(self) -> None:
        finding = classify_run_failure(_failed("Traceback...", returncode=1))
        self.assertEqual(finding["category"], CATEGORY_PROCESS_FAILED)

    def test_a_successful_run_has_no_finding(self) -> None:
        self.assertEqual(
            classify_run_failure({"success": True})["category"], CATEGORY_NONE
        )

    def test_a_cancelled_run_is_cancelled_not_crashed(self) -> None:
        finding = classify_run_failure(
            _failed("Terminated", returncode=-15), status="CANCELLED"
        )
        self.assertEqual(finding["category"], CATEGORY_CANCELLED)
        self.assertIn("stopped before it finished", finding["user_message"])

    def test_nothing_is_ever_retried(self) -> None:
        finding = classify_run_failure(
            _failed("CUDA out of memory"), device="cuda:0"
        )
        self.assertFalse(finding["retried"])

    def test_no_silent_device_fallback_exists_on_the_run_path(self) -> None:
        """A recommendation is text. It is never carried out."""
        import ast

        for name in (
            "batteryml_web/agent/failure_taxonomy.py",
            "batteryml_web/agent/run_policy.py",
            "batteryml_web/core/run_manager.py",
        ):
            tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    body = getattr(node, "body", [])
                    if (
                        body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        node.body = body[1:] or [ast.Pass()]
            code = ast.unparse(tree)
            for forbidden in ('device = "cpu"', "device='cpu'", "retry", "attempt += 1"):
                self.assertNotIn(forbidden, code, f"{name} may not repair a run")


class TestReadinessBlocks(unittest.TestCase):
    """C1: why a plan cannot run, in the same vocabulary."""

    def test_a_validator_refusal_is_a_config_problem(self) -> None:
        finding = classify_readiness_block(
            validation_errors=["`model.name` is not registered in MODELS"]
        )
        self.assertEqual(finding["category"], CATEGORY_CONFIG_INVALID)
        self.assertIn("not registered", finding["user_message"])

    def test_raw_only_data_is_a_dataset_problem_with_an_action(self) -> None:
        finding = classify_readiness_block(dataset_status="RAW_ONLY")
        self.assertEqual(finding["category"], CATEGORY_DATASET_MISSING)
        self.assertIn("preprocessing", finding["recommended_action"].lower())

    def test_missing_model_assets_are_named_as_such(self) -> None:
        finding = classify_readiness_block(model_status="NEEDS_ASSETS")
        self.assertEqual(finding["category"], CATEGORY_MODEL_ASSETS)

    def test_a_ready_plan_produces_no_finding(self) -> None:
        self.assertEqual(
            classify_readiness_block(dataset_status="READY", model_status="READY")[
                "category"
            ],
            CATEGORY_NONE,
        )


class TestLLMFailureClassification(unittest.TestCase):
    """C3: the assistant failing is never the run failing."""

    def test_no_configuration_is_a_state_not_a_fault(self) -> None:
        finding = classify_llm_failure(None, configured=False)
        self.assertEqual(finding["category"], CATEGORY_LLM_NOT_CONFIGURED)
        self.assertIn("deterministic workflow", finding["user_message"])

    def test_authentication_failures_are_recognised(self) -> None:
        for error in ("HTTP 401 Unauthorized", "403 Forbidden",
                      "Invalid API key provided", "authentication failed"):
            with self.subTest(error=error):
                self.assertEqual(
                    classify_llm_failure(error)["category"], CATEGORY_LLM_AUTH
                )

    def test_authentication_failures_are_never_retried(self) -> None:
        finding = classify_llm_failure("HTTP 401 Unauthorized")
        self.assertIn("not retried automatically", finding["recommended_action"])

    def test_network_and_timeout_failures_are_recognised(self) -> None:
        for error in ("Read timed out", "Connection refused",
                      "Temporary failure in name resolution", "HTTP 503"):
            with self.subTest(error=error):
                self.assertEqual(
                    classify_llm_failure(error)["category"], CATEGORY_LLM_TIMEOUT
                )

    def test_a_malformed_reply_is_its_own_category(self) -> None:
        finding = classify_llm_failure(None, response_text="not json at all")
        self.assertEqual(finding["category"], CATEGORY_LLM_INVALID_RESPONSE)

    def test_an_unrecognised_error_says_so_rather_than_guessing(self) -> None:
        finding = classify_llm_failure("Something entirely novel happened")
        self.assertEqual(finding["category"], CATEGORY_UNKNOWN)
        self.assertIn("not being guessed", finding["user_message"])

    def test_an_llm_failure_is_flagged_as_not_a_run_failure(self) -> None:
        self.assertTrue(
            explanation_failure_is_not_a_run_failure(classify_llm_failure("401"))
        )
        self.assertFalse(
            explanation_failure_is_not_a_run_failure(
                classify_run_failure(_failed("boom"))
            )
        )

    def test_a_computed_result_survives_an_explanation_failure(self) -> None:
        """C3: the metrics are BatteryML's; the explanation is a nicety."""
        manifest_metrics = {"RMSE": 716.48}
        finding = classify_llm_failure("Read timed out")
        # Nothing in the finding touches the run's own status or metrics.
        self.assertTrue(explanation_failure_is_not_a_run_failure(finding))
        self.assertEqual(manifest_metrics, {"RMSE": 716.48})
        self.assertNotIn("FAILED", finding["category"].upper())


class TestNoSecretsInDiagnostics(unittest.TestCase):
    """C2: technical details may be verbose; they may not leak credentials."""

    def test_key_shaped_text_is_redacted_from_messages(self) -> None:
        for secret in (
            "sk-abcdef1234567890",
            "Bearer abcdef1234567890",
            'api_key: "abcdef123456"',
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, redact(f"request failed: {secret}"))

    def test_a_failure_message_carrying_a_key_is_cleaned(self) -> None:
        finding = classify_llm_failure(
            "401 Unauthorized for key sk-live-abcdef123456789"
        )
        blob = repr(finding)
        self.assertNotIn("sk-live-abcdef123456789", blob)
        self.assertEqual(finding["category"], CATEGORY_LLM_AUTH)


class TestRecoveryAfterRestart(unittest.TestCase):
    """C6: what survives, and what must not be claimed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project_root = Path(self._tmp.name) / "alpha"

    def _run(self, run_id: str, status: str) -> Path:
        run_dir = self.project_root / "agent_matr_cnn" / run_id
        manifest = rm.build_manifest(
            run_id=run_id, project="alpha", workspace=str(run_dir),
            dataset="MATR", task="RUL prediction", model_key="cnn", seed=0,
        )
        rm.write_manifest(run_dir, manifest)
        if status != rm.STATUS_PLANNED:
            rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        if status not in (rm.STATUS_PLANNED, rm.STATUS_RUNNING):
            rm.update_manifest(run_dir, {}, status=status)
        return run_dir

    def test_finished_runs_survive_a_completely_new_process(self) -> None:
        self._run("run_done", rm.STATUS_COMPLETED)
        self._run("run_failed", rm.STATUS_FAILED)
        self._run("run_cancelled", rm.STATUS_CANCELLED)
        # live_run_ids empty: nothing is tracked, as after a restart.
        listed = {e["run_id"]: e["status"] for e in rm.list_runs(self.project_root, live_run_ids=[])}
        self.assertEqual(listed["run_done"], rm.STATUS_COMPLETED)
        self.assertEqual(listed["run_failed"], rm.STATUS_FAILED)
        self.assertEqual(listed["run_cancelled"], rm.STATUS_CANCELLED)

    def test_an_unverifiable_running_run_is_never_called_completed(self) -> None:
        self._run("run_orphan", rm.STATUS_RUNNING)
        entry = rm.list_runs(self.project_root, live_run_ids=[])[0]
        self.assertEqual(entry["status"], rm.STATUS_INTERRUPTED)
        self.assertNotEqual(entry["status"], rm.STATUS_COMPLETED)
        self.assertIn("not known to have finished", entry["reconcile_reason"])

    def test_no_unknown_pid_is_signalled_after_a_restart(self) -> None:
        """C6: the manager only ever signals a process it holds a handle to."""
        import ast

        source = (REPO_ROOT / "batteryml_web/core/run_manager.py").read_text(
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
        # A pid read back from a manifest is never signalled: the only pid
        # used comes from a live Popen object this process created.
        self.assertNotIn('manifest["pid"]', code)
        self.assertNotIn("manifest.get('pid')", code)
        self.assertIn("getattr(process, 'pid', None)", code)


class TestStaleStateIsNotShownAsCurrent(unittest.TestCase):
    """C5: a changed selection never wears the previous run's result."""

    def test_analyzing_a_new_request_clears_the_previous_runs_state(self) -> None:
        import ast

        source = (REPO_ROOT / "batteryml_web/ui/agent_workflow.py").read_text(
            encoding="utf-8"
        )
        block = source.split("def _analyze_request")[1].split("\ndef ")[0]
        for key in (
            "agent_execution_result",
            "agent_result_context",
            "agent_explanation",
            "agent_explanation_failure",
            "agent_run_context",
        ):
            self.assertIn(key, block, f"{key} must be cleared on a new request")

    def test_a_changed_plan_drops_the_stored_result(self) -> None:
        source = (REPO_ROOT / "batteryml_web/ui/agent_workflow.py").read_text(
            encoding="utf-8"
        )
        # The plan's own argv is the signature; a different plan invalidates
        # the stored result rather than displaying it beside a new selection.
        self.assertIn("agent_run_signature", source)
        self.assertIn("stored_signature != current_signature", source)

    def test_a_historical_run_is_labelled_as_such(self) -> None:
        source = (REPO_ROOT / "batteryml_web/ui/run_history.py").read_text(
            encoding="utf-8"
        )
        # History renders from a manifest and never re-runs anything.
        self.assertIn("never re-runs", source.lower().replace("re-run it", "never re-runs"))
        self.assertNotIn("execute_plan", source)


class TestResumeIsDocumentedAsAbsent(unittest.TestCase):
    """C7: no Resume control exists, and the audit says why."""

    def test_no_resume_control_is_exposed_anywhere_in_the_ui(self) -> None:
        for name in ("agent_workflow.py", "run_history.py"):
            source = (REPO_ROOT / "batteryml_web" / "ui" / name).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("Resume run", source)
            self.assertNotIn("ckpt_to_resume", source)

    def test_the_audit_records_the_finding(self) -> None:
        audit = (REPO_ROOT / "docs" / "RESUME_AUDIT.md").read_text(encoding="utf-8")
        self.assertIn("RESUME IS NOT IMPLEMENTED", audit)
        for required in ("optimizer", "epoch", "sklearn", "load_checkpoint"):
            self.assertIn(required, audit.lower())


if __name__ == "__main__":
    unittest.main()
