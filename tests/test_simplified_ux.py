"""The product is CPU / one GPU, Resume only where it works, Stop always.

Three simplifications this pins:

* the normal UI offers CPU and a single GPU. The DDP infrastructure still
  exists and is still tested (tests/test_distributed.py), but nothing about
  it reaches an ordinary user;
* Resume appears only where the backend says the model can resume *and* a
  resumable checkpoint exists. Otherwise the control is absent — not
  disabled with a paragraph of reasons;
* Stop is offered for any running job, whatever the model, and never depends
  on whether that run could later be resumed.

No Streamlit runtime, no GPU.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_simplified_ux -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.catalog import (  # noqa: E402
    REQUEST_TEMPLATE_STRUCTURE,
    build_request_template,
    task_phrase_for,
)
from batteryml_web.agent.run_policy import (  # noqa: E402
    gpu_status_line,
    resume_capability,
)
from batteryml_web.core.readiness import TASK_RUL, TASK_SOH, TASK_TRAJECTORY  # noqa: E402
from batteryml_web.ui import agent_workflow as ui  # noqa: E402
from batteryml_web.ui import run_history as history  # noqa: E402
from batteryml_web.ui import settings_panel as settings  # noqa: E402

UI_DIR = REPO_ROOT / "batteryml_web" / "ui"


def _code_without_prose(path: Path) -> str:
    """Module code with docstrings dropped; comments go with ast.unparse.

    The modules explain *why* multi-GPU is no longer offered, and reading
    that prose as behaviour would fail these tests for documenting
    themselves.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _inventory(*specs) -> dict:
    """An inventory from (index, free_mib, total_mib) triples."""
    return {
        "source": "nvidia-smi",
        "available": bool(specs),
        "device_count": len(specs),
        "devices": [
            {
                "index": index,
                "name": "Test GPU",
                "memory_total_mib": total,
                "memory_used_mib": total - free,
                "memory_free_mib": free,
                "utilization_pct": 0,
                "device": f"cuda:{index}",
            }
            for index, free, total in specs
        ],
        "error": None,
    }


class TestNormalUiIsCpuOrOneGpu(unittest.TestCase):
    """§1: no multi-GPU affordance survives in the page."""

    def test_the_page_renders_no_multi_select_for_devices(self) -> None:
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        self.assertNotIn("st.multiselect", code)
        self.assertIn("GPU_SELECT_LABEL", code)

    def test_the_page_consults_no_multi_gpu_capability(self) -> None:
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        for gone in (
            "multi_gpu_capability",
            "execution_blockers",
            "selection_blockers",
            "MULTI_GPU_NOTICE",
            "MULTI_GPU_UNSUPPORTED_MESSAGE",
        ):
            self.assertNotIn(gone, code, f"{gone} still reachable from the page")

    def test_no_distributed_vocabulary_reaches_the_page(self) -> None:
        """§9: ranks, sharding and the contract doc are not user-facing."""
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        for jargon in ("DDP", "torchrun", "nproc_per_node", "world_size",
                       "rank_0", "DDP_CONTRACT"):
            self.assertNotIn(jargon, code)

    def test_the_ddp_infrastructure_is_retained(self) -> None:
        """§2: simplifying the UI must not delete the backend work."""
        for path in (
            "batteryml_web/core/distributed_plan.py",
            "batteryml_web/core/distributed_runner.py",
            "docs/DDP_CONTRACT.md",
            "tests/test_distributed.py",
        ):
            self.assertTrue((REPO_ROOT / path).is_file(), f"{path} was deleted")

    def test_multi_gpu_capability_metadata_still_exists_for_future_work(self) -> None:
        from batteryml_web.agent.run_policy import multi_gpu_capability

        capability = multi_gpu_capability("cnn")
        self.assertIn("supports_multi_gpu", capability)
        self.assertFalse(capability["supports_multi_gpu"])


class TestGpuStatusIsCompact(unittest.TestCase):
    """§10: a short line, and a warning only when it is warranted."""

    def test_a_healthy_gpu_is_a_caption(self) -> None:
        status = gpu_status_line("cuda:7", _inventory((7, 24109, 24576)))
        self.assertEqual(status["severity"], "caption")
        self.assertIn("cuda:7", status["text"])
        self.assertIn("23.5 GB free", status["text"])
        self.assertIn("availability may change", status["text"])
        # Compact: one line, not a paragraph.
        self.assertLess(len(status["text"]), 130)
        self.assertNotIn("snapshot of a shared machine; another process", status["text"])

    def test_a_nearly_full_gpu_warns(self) -> None:
        status = gpu_status_line("cuda:0", _inventory((0, 500, 24576)))
        self.assertEqual(status["severity"], "warning")

    def test_an_unenumerated_gpu_warns_rather_than_inventing_a_figure(self) -> None:
        status = gpu_status_line("cuda:3", _inventory((0, 1000, 2000)))
        self.assertEqual(status["severity"], "warning")
        self.assertIn("not enumerated", status["text"])
        self.assertNotIn("GB free", status["text"])

    def test_cpu_has_no_status_line(self) -> None:
        self.assertEqual(gpu_status_line("cpu", _inventory((0, 1, 2)))["text"], "")

    def test_no_absolute_memory_floor_is_invented(self) -> None:
        """The threshold is relative to the card, not a model requirement."""
        # A small card that is mostly free must not warn just for being small.
        self.assertEqual(
            gpu_status_line("cuda:0", _inventory((0, 3500, 4096)))["severity"],
            "caption",
        )


class TestResumeIsCapabilityDriven(unittest.TestCase):
    """§3: shown where it works, absent where it does not."""

    def _manifest(self, model_key: str, family: str = "", checkpoints=()) -> dict:
        return {
            "run_id": "run_1",
            "model_key": model_key,
            "model_family": family,
            "artifacts": {"checkpoint_files": list(checkpoints)},
        }

    def test_capability_comes_from_the_backend_not_a_model_name_in_the_ui(self) -> None:
        code = _code_without_prose(UI_DIR / "run_history.py")
        self.assertIn("resume_capability", code)
        for name in ("batlinet", "BatLiNet", "cptransformer", "CPTransformer",
                     "xgb", "XGBoost", "cnn"):
            self.assertNotIn(name, code)

    def test_an_unsupported_model_reports_unsupported(self) -> None:
        for model, family in (("batlinet", "nn_model"), ("xgb", ""), ("cptransformer", "")):
            with self.subTest(model=model):
                self.assertFalse(resume_capability(model, family=family)["supports_resume"])

    def test_the_audited_family_reports_supported(self) -> None:
        for model in ("cnn", "lstm", "mlp", "transformer"):
            with self.subTest(model=model):
                self.assertTrue(
                    resume_capability(model, family="nn_model")["supports_resume"]
                )

    def test_the_offer_returns_without_rendering_when_unsupported(self) -> None:
        """Absent, not disabled-with-an-essay."""
        from tests.test_agent_web_ui import FakeStreamlit, _fake_streamlit

        fake = FakeStreamlit({})
        with _fake_streamlit(fake):
            history._render_resume_offer(
                self._manifest("batlinet", "nn_model", ["/tmp/x.ckpt"])
            )
        self.assertEqual(fake.rec["expanders"], [])
        self.assertEqual(fake.rec["buttons"], [])
        self.assertEqual(fake.rec["captions"], [])

    def test_a_supported_model_without_a_resumable_checkpoint_shows_nothing(self) -> None:
        from tests.test_agent_web_ui import FakeStreamlit, _fake_streamlit

        fake = FakeStreamlit({})
        with _fake_streamlit(fake):
            # No checkpoint files at all.
            history._render_resume_offer(self._manifest("cnn", "nn_model"))
        self.assertEqual(fake.rec["expanders"], [])

    def test_the_panel_exposes_no_training_internals(self) -> None:
        """§5: optimizer, RNG and scheduler belong in docs, not the page."""
        code = _code_without_prose(UI_DIR / "run_history.py")
        block = code.split("def _render_resume_offer")[1].split("\ndef ")[0]
        for internal in ("optimizer", "rng", "scheduler", "scaler", "schema_version"):
            self.assertNotIn(internal, block.lower())


class TestStopIsAlwaysAvailable(unittest.TestCase):
    """§4: any running job can be stopped, whatever it is."""

    def test_the_control_is_called_stop_run(self) -> None:
        self.assertEqual(ui.CANCEL_BUTTON_LABEL, "Stop run")

    def test_stop_does_not_consult_resume_capability(self) -> None:
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        block = code.split("def _render_active_run")[1].split("\ndef ")[0]
        self.assertIn("CANCEL_BUTTON_LABEL", block)
        self.assertNotIn("resume", block.lower())

    def test_stop_reuses_the_existing_cancellation(self) -> None:
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        block = code.split("def _render_active_run")[1].split("\ndef ")[0]
        self.assertIn("manager.cancel(run_id)", block)

    def test_the_backend_state_is_still_cancelled(self) -> None:
        from batteryml_web.core import run_manifest as rm

        self.assertIn(rm.STATUS_CANCELLED, rm.ALL_STATUSES)
        self.assertTrue(rm.can_transition(rm.STATUS_RUNNING, rm.STATUS_CANCELLED))


class TestShortenedCopy(unittest.TestCase):
    """§6: helper copy is short; failure messages are not touched."""

    def test_the_long_protocol_paragraph_is_gone(self) -> None:
        note = settings.SUPPORTED_API_NOTE
        self.assertIn("OpenAI-compatible", note)
        self.assertLess(len(note), 120, note)
        self.assertNotIn("self-hosted gateways", note)
        self.assertNotIn("chat completions are not implemented", note)

    def test_the_compatibility_list_survives_the_shortening(self) -> None:
        note = settings.SUPPORTED_API_NOTE.lower()
        for provider in ("openai", "gemini", "claude", "kimi", "glm", "custom"):
            self.assertIn(provider, note)

    def test_the_long_key_storage_paragraph_is_gone(self) -> None:
        code = _code_without_prose(UI_DIR / "settings_panel.py")
        self.assertNotIn("is never written to a file, a log or the debug view", code)
        self.assertIn("API keys stay in this session only", code)

    def test_the_security_behaviour_is_unchanged(self) -> None:
        code = _code_without_prose(UI_DIR / "settings_panel.py")
        self.assertIn("type='password'", code)
        # The summary still reduces the key to a boolean.
        self.assertIn("api_key_set", code)

    def test_failure_messages_were_not_shortened_away(self) -> None:
        """Only helper copy was cut; actionable failures keep their detail."""
        from batteryml_web.agent.failure_taxonomy import classify_llm_failure

        finding = classify_llm_failure("Network is unreachable")
        self.assertIn("outbound internet", finding["recommended_action"])
        self.assertTrue(len(finding["recommended_action"]) > 80)


class TestGenericRequestTemplate(unittest.TestCase):
    """§7-8: the visible placeholder names nothing in particular."""

    def test_the_placeholder_is_the_generic_shape(self) -> None:
        self.assertEqual(
            REQUEST_TEMPLATE_STRUCTURE, "Use [Model] on [dataset] to predict [task]."
        )

    def test_the_placeholder_names_no_model_dataset_or_task(self) -> None:
        for name in ("XGBoost", "MATR", "HUST", "CPTransformer", "BatLiNet",
                     "RUL", "sklearn", "xgb"):
            self.assertNotIn(name, REQUEST_TEMPLATE_STRUCTURE)

    def test_an_empty_catalog_keeps_the_slots(self) -> None:
        filled = build_request_template({})
        self.assertIn("[Model]", filled)
        self.assertIn("[dataset]", filled)

    def test_use_template_fills_from_the_backend(self) -> None:
        filled = build_request_template(
            {
                "datasets": [{"dataset": "MATR", "status": "READY"}],
                "models": [
                    {"model_key": "xgb", "model_display": "XGBoost — Boosted",
                     "datasets": ["MATR"], "external": False}
                ],
            }
        )
        self.assertEqual(filled, "Use XGBoost on MATR to predict battery lifetime.")

    def test_the_canonical_tasks_are_preserved(self) -> None:
        """§8: not every request is a lifetime prediction."""
        phrases = {
            TASK_RUL: task_phrase_for(TASK_RUL),
            TASK_SOH: task_phrase_for(TASK_SOH),
            TASK_TRAJECTORY: task_phrase_for(TASK_TRAJECTORY),
        }
        self.assertEqual(len(set(phrases.values())), 3, phrases)
        self.assertEqual(phrases[TASK_SOH], "state of health")

    def test_an_unknown_task_keeps_its_canonical_label(self) -> None:
        self.assertEqual(task_phrase_for("Some new task"), "Some new task")

    def test_the_page_reads_the_task_from_the_resolved_context(self) -> None:
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        block = code.split("def request_template")[1].split("\ndef ")[0]
        self.assertIn("task=task", block)
        self.assertIn("context.get('task')", block)

    def test_filling_the_template_does_not_analyse_or_run(self) -> None:
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        block = code.split("if st.button(USE_TEMPLATE_LABEL")[1].split("if st.button")[0]
        self.assertIn("USE_TEMPLATE_KEY", block)
        for forbidden in ("_analyze_request", "execute", "get_run_manager"):
            self.assertNotIn(forbidden, block)


if __name__ == "__main__":
    unittest.main()
