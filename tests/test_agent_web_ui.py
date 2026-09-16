"""UI/backend boundary tests for the Streamlit Agent Workflow integration.

These tests never train, never launch a subprocess and never call an LLM.
They check the contract the Streamlit layer must honour:

* structured state comes only from the backend workflow APIs;
* the Run button is the only path to ``execute_plan_streaming`` and is
  enabled only when the backend planner says ``can_execute``;
* smoke mode maps only to the planner's ``epochs=1`` override;
* blocked datasets/models never surface commands, selectable models or an
  enabled Run;
* result display consumes ``build_result_context`` output verbatim, shows
  the smoke and provenance warnings, and shows no metrics for failed runs.

The render pass runs against a fake ``st`` object, so the logic is asserted
without a Streamlit runtime.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_web_ui -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import (  # noqa: E402
    REGISTRIES_AVAILABLE,
    REGISTRY_SKIP_MESSAGE,
)
from batteryml_web.agent.results import SMOKE_RUN_WARNING  # noqa: E402
from batteryml_web.agent.run_policy import SMOKE_NOTICE  # noqa: E402
from batteryml_web.core import device_inventory as di  # noqa: E402
from batteryml_web.ui import agent_workflow as ui  # noqa: E402
from batteryml_web.ui import feedback as ui_feedback  # noqa: E402
from batteryml_web.ui import llm_state as ui_llm_state  # noqa: E402

#: A fixed two-GPU inventory. Seeded into session state by every render test
#: so the UI never shells out to nvidia-smi during the suite and the device
#: control is deterministic wherever the tests run.
_FAKE_GPU_INVENTORY = {
    "source": "nvidia-smi",
    "available": True,
    "device_count": 2,
    "devices": di.parse_nvidia_smi_gpu_csv(
        "0, NVIDIA GeForce RTX 3090, 24576, 22500, 2076, 97\n"
        "1, NVIDIA GeForce RTX 3090, 24576, 14000, 10576, 12\n"
    ),
    "error": None,
}


class _Rerun(Exception):
    """Sentinel for st.rerun()."""


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _ExpanderCtx:
    """A container whose contents are recorded separately from the page body.

    The real page hides the full model catalogues inside expanders, so a test
    that asks "is the catalogue in the primary view?" needs the fake to keep
    the two apart.
    """

    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        self._fake.in_expander = True
        return self

    def __exit__(self, *exc_info):
        self._fake.in_expander = False
        return False


class _Col:
    def __init__(self, rec: dict):
        self._rec = rec

    def metric(self, label, value, **kwargs):
        self._rec["metrics"].append((str(label), str(value)))

    def caption(self, text, **kwargs):
        self._rec["captions"].append(str(text))


class _Slot:
    """Stand-in for st.empty(): records whatever the live progress draws."""

    def __init__(self, rec: dict):
        self._rec = rec

    def info(self, text, **kwargs):
        self._rec["infos"].append(str(text))

    def error(self, text, **kwargs):
        self._rec["errors"].append(str(text))

    def caption(self, text, **kwargs):
        self._rec["captions"].append(str(text))

    def code(self, text, **kwargs):
        self._rec["codes"].append(str(text))

    def text(self, text, **kwargs):
        self._rec["writes"].append(str(text))

    def progress(self, value=0.0, **kwargs):
        self._rec["progress"].append(float(value))


class FakeStreamlit:
    """Minimal stand-in for the streamlit surface the UI module uses."""

    def __init__(
        self,
        session: dict,
        clicked_buttons: set[str] | None = None,
        checkbox_values: dict[str, bool] | None = None,
        multiselect_values: dict[str, list] | None = None,
    ):
        self.session_state = session
        self._clicked = set(clicked_buttons or ())
        self._checkbox_values = dict(checkbox_values or {})
        self._multiselect_values = dict(multiselect_values or {})
        self.rec: dict[str, list] = {
            "markdown": [],
            "captions": [],
            "codes": [],
            "warnings": [],
            "errors": [],
            "infos": [],
            "writes": [],
            "dataframes": [],
            "metrics": [],
            "buttons": [],
            "selectboxes": [],
            "progress": [],
            "tabs": [],
            "toasts": [],
            "expanders": [],
            "expander_writes": [],
            "multiselects": [],
            "headings": [],
        }
        self.in_expander = False

    # --- text/display -------------------------------------------------
    def markdown(self, text, **kwargs):
        self.rec["markdown"].append(str(text))

    def header(self, text, **kwargs):
        self.rec["headings"].append(str(text))

    def subheader(self, text, **kwargs):
        self.rec["headings"].append(str(text))

    def caption(self, text, **kwargs):
        self.rec["captions"].append(str(text))

    def code(self, text, **kwargs):
        self.rec["codes"].append(str(text))

    def text(self, text, **kwargs):
        self.rec["writes"].append(str(text))

    def write(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        bucket = "expander_writes" if self.in_expander else "writes"
        self.rec[bucket].append(text)

    def info(self, text, **kwargs):
        self.rec["infos"].append(str(text))

    def warning(self, text, **kwargs):
        self.rec["warnings"].append(str(text))

    def error(self, text, **kwargs):
        self.rec["errors"].append(str(text))

    def dataframe(self, data, **kwargs):
        self.rec["dataframes"].append(data)

    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Col(self.rec) for _ in range(count)]

    # --- widgets -------------------------------------------------------
    def text_area(self, label, value="", **kwargs):
        return str(value or "")

    def selectbox(self, label, options, index=0, **kwargs):
        opts = list(options)
        self.rec["selectboxes"].append({"label": str(label), "options": [str(o) for o in opts]})
        return opts[index if index is not None else 0]

    def checkbox(self, label, **kwargs):
        return bool(self._checkbox_values.get(str(label), False))

    def multiselect(self, label, options, default=None, **kwargs):
        self.rec["multiselects"].append(
            {"label": str(label), "options": [str(o) for o in options]}
        )
        if str(label) in self._multiselect_values:
            return list(self._multiselect_values[str(label)])
        return list(default or [])

    def button(self, label, **kwargs):
        disabled = bool(kwargs.get("disabled", False))
        clicked = (not disabled) and (str(label) in self._clicked)
        self.rec["buttons"].append(
            {"label": str(label), "disabled": disabled, "clicked": clicked}
        )
        return clicked

    # --- containers ------------------------------------------------------
    def tabs(self, labels, **kwargs):
        self.rec["tabs"].append([str(label) for label in labels])
        return [_Ctx() for _ in labels]

    def toast(self, message, **kwargs):
        self.rec["toasts"].append(str(message))

    def expander(self, label, **kwargs):
        self.rec["expanders"].append(str(label))
        return _ExpanderCtx(self)

    def spinner(self, label, **kwargs):
        return _Ctx()

    def empty(self):
        return _Slot(self.rec)

    def progress(self, value=0.0, **kwargs):
        self.rec["progress"].append(float(value))
        return _Slot(self.rec)

    def rerun(self):
        raise _Rerun()


#: Example requests as the backend would build them for this stub catalog.
_FAKE_EXAMPLES = (
    "Use CPTransformer on MATR to predict battery lifetime.",
    "Recommend a model to predict battery lifetime on MATR.",
    "帮我用 MATR 做电池寿命预测",
)

#: A catalog stub large enough for the model-request resolution the UI does.
_FAKE_CATALOG = {
    "datasets": [{"dataset": "MATR", "aliases": ["MATR", "matr"], "status": "READY",
                  "preprocessing_supported": True}],
    "tasks": [{"task": "RUL prediction", "aliases": ["RUL"]}],
    "models": [
        {"model_key": "cptransformer", "model_display": "CPTransformer", "aliases": ["cptransformer"],
         "datasets": ["MATR"], "tasks": ["RUL prediction"], "has_validated_config": True, "external": False},
        {"model_key": "batlinet", "model_display": "BatLiNet", "aliases": ["batlinet"],
         "datasets": ["MATR"], "tasks": ["RUL prediction"], "has_validated_config": True, "external": False},
        {"model_key": "PBT", "model_display": "PBT", "aliases": ["PBT", "pbt"],
         "datasets": [], "tasks": ["RUL prediction"], "has_validated_config": False, "external": True},
    ],
}

_MATR_CONTEXT = {
    "dataset": "MATR",
    "dataset_status": "READY",
    "task": "RUL prediction",
    "preprocessing_supported": False,
    "runnable_models": [
        {
            "model": "cptransformer",
            "model_display": "CPTransformer — CyclePatch Transformer",
            "status": "READY",
            "validated_config": "configs/external/cptransformer/matr_1.yaml",
        },
        {
            "model": "batlinet",
            "model_display": "BatLiNet — Inter-cell Lifetime Prediction",
            "status": "READY",
            "validated_config": "configs/external/batlinet/matr_1.yaml",
        },
    ],
    "blocked_models": [
        {
            "model": "PBT",
            "status": "NEEDS_ASSETS",
            "missing": ["pretrained checkpoint"],
            "recommended_action": "Provide the pretrained assets.",
        }
    ],
    "recommended_next_steps": ["Pick a runnable model."],
}

_TRAJ_CONTEXT = {
    "dataset": "MATR",
    "dataset_status": "READY",
    "task": "Degradation trajectory forecasting",
    "preprocessing_supported": False,
    "runnable_models": [],
    "blocked_models": [
        {
            "model": "BatteryMFormer",
            "status": "NEEDS_DATA",
            "missing": ["prepared trajectory inputs"],
            "recommended_action": "Prepare trajectory data via the BatteryMFormer runner.",
        }
    ],
    "recommended_next_steps": [
        "Degradation trajectory forecasting is served by external models only "
        "(BatteryMFormer); it is not a scalar RUL task and must not run through RULLabelAnnotator."
    ],
}

_MICH_CONTEXT = {
    "dataset": "MICH",
    "dataset_status": "RAW_ONLY",
    "task": "RUL prediction",
    "preprocessing_supported": False,
    "runnable_models": [],
    "blocked_models": [
        {"model": "cptransformer", "status": "NEEDS_DATA", "missing": [], "recommended_action": "Preprocess MICH first."}
    ],
    "recommended_next_steps": [
        "Dataset 'MICH' is RAW_ONLY: BatteryLife v11 MICH data are present locally "
        "(data/raw/BatteryLife/v11/MICH.zip), but this archive is not yet wired to the "
        "BatteryML preprocessing route. Prepare/unpack an approved BatteryML-compatible "
        "MICH input before model execution."
    ],
}


def _ok_plan(model: str = "cptransformer", epochs: int | None = None) -> dict:
    command = [
        "python",
        "batteryml_web/core/run_batteryml_cli.py",
        "run",
        "configs/external/cptransformer/matr_1.yaml",
        "--workspace",
        "runs/agent_matr_cptransformer",
        "--train",
        "--eval",
        "--device",
        "cpu",
        "--seed",
        "0",
    ]
    if epochs is not None:
        command += ["--epochs", str(epochs)]
    return {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": model,
        "model_display": "CPTransformer — CyclePatch Transformer",
        "model_status": "READY",
        "dataset_status": "READY",
        "config_path": "configs/external/cptransformer/matr_1.yaml",
        "config_valid": True,
        "can_execute": True,
        "validation_errors": [],
        "command": command,
        "epochs": epochs,
        "reason": "plan only, nothing executed.",
    }


def _failed_plan(model: str = "PBT") -> dict:
    return {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": model,
        "model_display": None,
        "model_status": "NEEDS_ASSETS",
        "dataset_status": "READY",
        "config_path": None,
        "config_valid": False,
        "can_execute": False,
        "validation_errors": [],
        "command": None,
        "epochs": None,
        "reason": f"Model {model!r} is NEEDS_ASSETS and must not execute.",
    }


def _render(
    session: dict,
    clicked: set[str] | None = None,
    checkboxes: dict[str, bool] | None = None,
    multiselect_values: dict[str, list] | None = None,
) -> FakeStreamlit:
    fake = FakeStreamlit(session, clicked, checkboxes, multiselect_values)
    with _fake_streamlit(fake):
        try:
            ui.render_agent_workflow()
        except _Rerun:
            pass
    return fake


class _fake_streamlit:
    """Point every UI module that reads Streamlit at the same fake.

    Three modules hold their own ``st``: the page, the feedback helper
    (which reads ``st.session_state`` to deduplicate toasts) and the LLM
    settings helper (which reads the credentials). Missing one does not
    raise — it silently reads the *real* Streamlit's empty session state,
    which is how ``is_llm_configured()`` quietly returned False and made a
    test look like a production bug.
    """

    def __init__(self, fake: FakeStreamlit):
        self._fake = fake
        self._saved: list = []

    def __enter__(self):
        for module in (ui, ui_feedback, ui_llm_state):
            self._saved.append((module, module.st))
            module.st = self._fake
        return self._fake

    def __exit__(self, *exc_info):
        for module, original in self._saved:
            module.st = original
        return False


class TestPureBoundaryHelpers(unittest.TestCase):
    """The small helpers the UI leans on, checked without Streamlit."""

    def test_ready_context_offers_only_runnable_canonical_keys(self) -> None:
        self.assertEqual(
            ui.runnable_model_keys(_MATR_CONTEXT), ["cptransformer", "batlinet"]
        )
        self.assertNotIn("PBT", ui.runnable_model_keys(_MATR_CONTEXT))
        self.assertTrue(ui.has_runnable_model(_MATR_CONTEXT))

    def test_blocked_or_missing_context_offers_nothing(self) -> None:
        for context in (_TRAJ_CONTEXT, _MICH_CONTEXT):
            with self.subTest(task=context["task"], dataset=context["dataset"]):
                self.assertEqual(ui.runnable_model_keys(context), [])
                self.assertFalse(ui.has_runnable_model(context))

    def test_run_gate_requires_backend_can_execute(self) -> None:
        self.assertTrue(ui.run_is_enabled(_ok_plan()))
        self.assertFalse(ui.run_is_enabled(_failed_plan()))
        self.assertFalse(ui.run_is_enabled({"can_execute": "yes"}))
        self.assertFalse(ui.run_is_enabled(None))

    def test_smoke_mode_maps_only_to_epochs_one(self) -> None:
        self.assertEqual(ui.smoke_epochs_value(True), 1)
        self.assertIsNone(ui.smoke_epochs_value(False))
        # And only epochs=1 appears in the resulting planner command:
        plan = _ok_plan(epochs=1)
        self.assertEqual(plan["command"][plan["command"].index("--epochs") + 1], "1")
        self.assertNotIn("--lr", " ".join(plan["command"]))

    def test_mich_notice_is_the_backend_text_not_a_fake_command(self) -> None:
        notice = ui.raw_only_notice(_MICH_CONTEXT)
        self.assertIsNotNone(notice)
        self.assertIn("not yet wired", notice)
        self.assertNotIn("python", notice)
        self.assertNotIn("data/raw/MICH", notice)
        self.assertIsNone(ui.raw_only_notice(_MATR_CONTEXT))

    def test_failed_run_never_surfaces_metrics(self) -> None:
        context = {
            "success": False,
            "metrics": {"RMSE": 1.0, "MAE": 2.0, "MAPE": 0.5},
        }
        self.assertEqual(ui.display_metrics(context), {})
        self.assertEqual(ui.display_metrics(None), {})

    def test_successful_metrics_pass_through_verbatim(self) -> None:
        context = {
            "success": True,
            "metrics": {
                "RMSE": 455.5853271484375,
                "MAE": 393.8516540527344,
                "MAPE": 0.6621854901313782,
            },
        }
        metrics = ui.display_metrics(context)
        self.assertEqual(metrics["RMSE"], 455.5853271484375)
        self.assertEqual(metrics["MAPE"], 0.6621854901313782)
        self.assertEqual(ui.mape_percent(metrics["MAPE"]), "66.22%")

    def test_smoke_warning_is_recognised(self) -> None:
        context = {"success": True, "warnings": [SMOKE_RUN_WARNING]}
        surfaced = ui.smoke_warnings(context)
        self.assertEqual(surfaced, [SMOKE_RUN_WARNING])
        self.assertEqual(ui.smoke_warnings({"success": True, "warnings": ["cache miss"]}), [])

    def test_provenance_warning_is_recognised(self) -> None:
        context = {
            "provenance_verified": False,
            "provenance_warnings": ["Caller dataset 'MICH' differs from workspace 'MATR'."],
        }
        warnings = ui.provenance_warnings(context)
        self.assertEqual(len(warnings), 1)
        self.assertIn("differs", warnings[0])
        self.assertEqual(ui.provenance_warnings({"provenance_verified": True, "provenance_warnings": []}), [])

    def test_result_rows_come_only_from_build_result_context_shape(self) -> None:
        context = {
            "success": True,
            "dataset": "MATR",
            "task": "RUL prediction",
            "model": "cptransformer",
            "metrics": {"RMSE": 1.0},
            "num_predictions": 42,
            "predictions_file": "ws/predictions_matr.pkl",
            "figure_files": ["ws/fig1.png"],
        }
        rows = dict(ui.result_summary_rows(context))
        self.assertEqual(rows["Dataset"], "MATR")
        self.assertEqual(rows["Number of predictions"], 42)
        self.assertEqual(rows["Predictions artifact"], "ws/predictions_matr.pkl")
        self.assertEqual(rows["Figures"], "ws/fig1.png")

    def test_workspace_stays_inside_runs_area(self) -> None:
        ws = ui.workspace_for_request("MATR", "cptransformer", True)
        self.assertTrue(ws.startswith("runs/"))
        self.assertTrue(ws.endswith("_smoke"))
        self.assertFalse(ui.workspace_for_request("MATR", "cptransformer", False).endswith("_smoke"))
        escaped = ui.workspace_for_request("MATR", "cptransformer", False, project_root="/tmp/elsewhere")
        self.assertTrue(escaped.startswith("runs/"))


class TestRenderBoundaries(unittest.TestCase):
    """render_agent_workflow() must only orchestrate the backend APIs."""

    def setUp(self) -> None:
        # Neutralise the backend calls; these tests check orchestration only.
        self._plan_calls: list[dict[str, Any]] = []
        self._exec_calls: list[dict] = []
        self._real_build = ui.build_execution_plan
        self._real_execute = ui.execute_plan_streaming
        self._real_result_ctx = ui.build_result_context
        self._real_llm = ui.is_llm_configured

        def stub_build(dataset, task, model, **kwargs):
            self._plan_calls.append({"dataset": dataset, "task": task, "model": model, **kwargs})
            return kwargs.get("__plan__") or _ok_plan(model, epochs=kwargs.get("epochs"))

        def stub_execute(plan, **kwargs):
            self._exec_calls.append(plan)
            return {
                "started": True,
                "completed": True,
                "success": True,
                "returncode": 0,
                "command": plan["command"],
                "workspace": "runs/agent_matr_cptransformer",
                "stdout": "x" * 10000,
                "stderr": "",
                "error": None,
                "artifacts": {"prediction_files": []},
                "dry_run": False,
            }

        def stub_result_context(workspace, **kwargs):
            return {
                "workspace": workspace,
                "run_status": "SUCCESS",
                "success": True,
                "dataset": kwargs.get("dataset"),
                "task": kwargs.get("task"),
                "model": kwargs.get("model"),
                "metrics": {"RMSE": 455.5853271484375, "MAE": 393.8516540527344, "MAPE": 0.6621854901313782},
                "num_predictions": 42,
                "predictions_file": "ws/predictions.pkl",
                "figure_files": [],
                "warnings": [SMOKE_RUN_WARNING],
                "provenance_verified": False,
                "provenance_warnings": ["unverified"],
            }

        ui.build_execution_plan = stub_build
        ui.execute_plan_streaming = stub_execute
        ui.build_result_context = stub_result_context
        ui.is_llm_configured = lambda: False

    def tearDown(self) -> None:
        ui.build_execution_plan = self._real_build
        ui.execute_plan_streaming = self._real_execute
        ui.build_result_context = self._real_result_ctx
        ui.is_llm_configured = self._real_llm

    def _session(self, context: dict) -> dict:
        return {
            "agent_workflow_context": context,
            "agent_original_request": "Use CPTransformer on MATR to predict battery lifetime.",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
        }

    def test_analyze_never_executes_and_plan_is_per_selection(self) -> None:
        session = self._session(_MATR_CONTEXT)
        fake = _render(session)
        # No execution without an explicit Run click.
        self.assertEqual(self._exec_calls, [])
        # A plan was built for the selected runnable model only.
        self.assertEqual(len(self._plan_calls), 1)
        self.assertEqual(self._plan_calls[0]["model"], "cptransformer")
        self.assertEqual(self._plan_calls[0]["epochs"], None)
        # Selection is restricted to runnable models; blocked models stay display-only.
        sel = fake.rec["selectboxes"][0]["options"]
        self.assertIn("cptransformer", sel)
        self.assertNotIn("PBT", sel)
        run_buttons = [b for b in fake.rec["buttons"] if b["label"] == ui.RUN_BUTTON_LABEL]
        self.assertTrue(run_buttons and run_buttons[0]["disabled"] is False)

    def test_run_requires_explicit_button_and_validated_plan(self) -> None:
        # Button not clicked: nothing executed.
        _render(self._session(_MATR_CONTEXT))
        self.assertEqual(self._exec_calls, [])
        # Button clicked with a can_execute plan: executed exactly once.
        _render(self._session(_MATR_CONTEXT), clicked={"Run"})
        self.assertEqual(len(self._exec_calls), 1)
        # Blocked plan (can_execute False): Run disabled, executor untouched.
        self._exec_calls.clear()
        self._real_build  # noqa: B018 (documented seam)
        ui.build_execution_plan = lambda dataset, task, model, **kw: _failed_plan(model)
        fake = _render(self._session(_MATR_CONTEXT), clicked={"Run"})
        self.assertEqual(self._exec_calls, [])
        run_buttons = [b for b in fake.rec["buttons"] if b["label"] == ui.RUN_BUTTON_LABEL]
        self.assertTrue(run_buttons and run_buttons[0]["disabled"] is True)

    def test_smoke_checkbox_maps_only_to_epochs_one(self) -> None:
        _render(self._session(_MATR_CONTEXT), checkboxes={ui.SMOKE_CHECKBOX_LABEL: True})
        self.assertEqual(self._plan_calls[-1]["epochs"], 1)
        self.assertIn("_smoke", self._plan_calls[-1]["workspace"])
        _render(self._session(_MATR_CONTEXT), checkboxes={ui.SMOKE_CHECKBOX_LABEL: False})
        self.assertIsNone(self._plan_calls[-1]["epochs"])

    def test_blocked_dataset_shows_nothing_selectable_or_executable(self) -> None:
        fake = _render(self._session(_MICH_CONTEXT), clicked={"Run"})
        self.assertEqual(fake.rec["selectboxes"], [])
        self.assertEqual([b for b in fake.rec["buttons"] if b["label"] == ui.RUN_BUTTON_LABEL], [])
        self.assertEqual(self._exec_calls, [])
        self.assertEqual(fake.rec["codes"], [])  # no fake preprocessing command
        joined = " ".join(fake.rec["warnings"] + fake.rec["writes"])
        self.assertIn("not yet wired", joined)
        self.assertNotIn("data/raw/MICH", joined)

    def test_trajectory_workflow_has_no_rul_run_path(self) -> None:
        fake = _render(self._session(_TRAJ_CONTEXT), clicked={"Run"})
        self.assertEqual(fake.rec["selectboxes"], [])
        self.assertEqual([b for b in fake.rec["buttons"] if b["label"] == ui.RUN_BUTTON_LABEL], [])
        self.assertEqual(self._exec_calls, [])
        # Nothing is runnable, so the blocked model that would have served the
        # request is named in the primary view rather than only in an expander.
        shown = " ".join(fake.rec["writes"] + fake.rec["expander_writes"])
        self.assertIn("BatteryMFormer", shown)
        self.assertEqual(self._plan_calls, [])

    def test_failed_execution_displays_no_metrics(self) -> None:
        ui.execute_plan_streaming = lambda plan, **kw: {
            "started": True,
            "completed": True,
            "success": False,
            "returncode": 1,
            "command": plan["command"],
            "workspace": "runs/agent_matr_cptransformer",
            "stdout": "",
            "stderr": "boom",
            "error": "wrapper exited with returncode 1",
            "artifacts": {},
            "dry_run": False,
        }
        session = self._session(_MATR_CONTEXT)
        fake = _render(session, clicked={"Run"})
        self.assertEqual(self._exec_calls, [])  # streaming stub replaced above
        rendered = session.get("agent_result_context")
        self.assertIsNone(rendered)  # failure never fabricates a result context
        self.assertFalse(any(label in {"RMSE", "MAE", "MAPE"} for label, _ in fake.rec["metrics"]))

    def test_successful_result_display_uses_backend_fields(self) -> None:
        session = self._session(_MATR_CONTEXT)
        session["agent_result_context"] = {
            "workspace": "runs/x",
            "run_status": "SUCCESS",
            "success": True,
            "dataset": "MATR",
            "task": "RUL prediction",
            "model": "cptransformer",
            "metrics": {"RMSE": 455.5853271484375, "MAE": 393.8516540527344, "MAPE": 0.6621854901313782},
            "num_predictions": 42,
            "predictions_file": "ws/predictions.pkl",
            "figure_files": [],
            "warnings": [SMOKE_RUN_WARNING],
            "provenance_verified": False,
            "provenance_warnings": ["Provenance could not be verified"],
        }
        fake = _render(session)
        labels = dict(fake.rec["metrics"])
        self.assertEqual(labels["RMSE"], "455.5853271484375")
        self.assertEqual(labels["MAE"], "393.8516540527344")
        self.assertEqual(labels["MAPE"], "0.6621854901313782")  # raw ratio preserved
        self.assertTrue(any("66.22%" in c for c in fake.rec["captions"]))
        self.assertTrue(any("smoke run" in w.lower() for w in fake.rec["warnings"]))
        self.assertTrue(any("Provenance" in w for w in fake.rec["warnings"]))

    def test_stdout_is_tail_only_in_session_state(self) -> None:
        session = self._session(_MATR_CONTEXT)
        fake = _render(session, clicked={"Run"})
        stored = session.get("agent_execution_result")
        if stored is not None:  # rerun happens before store in some flows
            self.assertLessEqual(len(stored.get("stdout", "")), ui.OUTPUT_TAIL_CHARS + 64)


class TestSmokeAndDeviceSelection(unittest.TestCase):
    """Milestone A: smoke wording, device inventory, and the CPU default."""

    def test_smoke_label_is_the_required_sentence(self) -> None:
        self.assertEqual(ui.SMOKE_CHECKBOX_LABEL, SMOKE_NOTICE)
        self.assertEqual(
            ui.SMOKE_CHECKBOX_LABEL,
            "1-epoch smoke test — workflow validation only; not benchmark performance.",
        )

    def test_device_options_follow_the_inventory_not_a_constant(self) -> None:
        self.assertEqual(
            ui.device_options(_FAKE_GPU_INVENTORY), ["cpu", "cuda:0", "cuda:1"]
        )
        eight = {
            "devices": di.parse_nvidia_smi_gpu_csv(
                "".join(f"{i}, RTX 3090, 24576, 22000, 2576, 50\n" for i in range(8))
            )
        }
        self.assertEqual(len(ui.device_options(eight)), 9)
        # Unknown inventory still allows an explicit GPU choice.
        self.assertEqual(
            ui.device_options(di.empty_inventory("no nvidia-smi")), ["cpu", "cuda:0"]
        )

    def test_batlinet_smoke_defaults_to_cpu_and_explains_why(self) -> None:
        # CPU is the empty GPU selection, so the control opens with none ticked.
        self.assertEqual(ui.default_gpu_selection("batlinet", True), [])
        self.assertIn("CPU", ui.smoke_device_note("batlinet", True))
        # No recommendation text for the ordinary cases.
        self.assertEqual(ui.smoke_device_note("batlinet", False), "")
        self.assertEqual(ui.smoke_device_note("cptransformer", True), "")

    def test_the_gpu_control_lists_gpus_only(self) -> None:
        self.assertEqual(ui.gpu_device_options(_FAKE_GPU_INVENTORY), ["cuda:0", "cuda:1"])
        self.assertNotIn("cpu", ui.gpu_device_options(_FAKE_GPU_INVENTORY))

    def test_no_hardcoded_gpu_count_in_the_ui_module(self) -> None:
        source = Path(ui.__file__).read_text(encoding="utf-8")
        for forbidden in ("cuda:1", "cuda:2", "cuda:3", "cuda:7", "range(8)"):
            self.assertNotIn(forbidden, source)


class TestBatLiNetSmokeRender(unittest.TestCase):
    """The rendered BatLiNet smoke flow: CPU default, epochs=1, no auto-run."""

    def setUp(self) -> None:
        self._plan_calls: list[dict[str, Any]] = []
        self._exec_calls: list[dict] = []
        self._real_build = ui.build_execution_plan
        self._real_execute = ui.execute_plan_streaming
        self._real_llm = ui.is_llm_configured

        def stub_build(dataset, task, model, **kwargs):
            self._plan_calls.append({"dataset": dataset, "task": task, "model": model, **kwargs})
            return _ok_plan(model, epochs=kwargs.get("epochs"))

        def stub_execute(plan, **kwargs):
            self._exec_calls.append(plan)
            raise AssertionError("render must not execute without an explicit Run click")

        ui.build_execution_plan = stub_build
        ui.execute_plan_streaming = stub_execute
        ui.is_llm_configured = lambda: False

    def tearDown(self) -> None:
        ui.build_execution_plan = self._real_build
        ui.execute_plan_streaming = self._real_execute
        ui.is_llm_configured = self._real_llm

    def _batlinet_session(self) -> dict:
        return {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_original_request": "Use BatLiNet on MATR to predict battery lifetime.",
            "agent_selected_model": "batlinet",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
        }

    def test_smoke_render_uses_cpu_and_epochs_one(self) -> None:
        session = self._batlinet_session()
        fake = _render(session, checkboxes={ui.SMOKE_CHECKBOX_LABEL: True})
        self.assertEqual(self._exec_calls, [])
        call = self._plan_calls[-1]
        self.assertEqual(call["model"], "batlinet")
        self.assertEqual(call["epochs"], 1)
        self.assertEqual(call["device"], "cpu")
        # The compute control lists every enumerated GPU; none ticked means CPU.
        gpu_box = [
            box
            for box in fake.rec["multiselects"]
            if box["label"].startswith("GPUs")
        ]
        self.assertTrue(gpu_box, "the compute control is missing")
        self.assertEqual(gpu_box[0]["options"], ["cuda:0", "cuda:1"])
        self.assertEqual(
            session["agent_device_selection"], {"mode": "cpu", "gpu_ids": []}
        )
        joined = " ".join(fake.rec["infos"])
        self.assertIn(SMOKE_NOTICE, joined)

    def test_non_smoke_render_leaves_the_config_schedule_alone(self) -> None:
        session = self._batlinet_session()
        _render(session, checkboxes={ui.SMOKE_CHECKBOX_LABEL: False})
        self.assertIsNone(self._plan_calls[-1]["epochs"])

    def test_gpu_inventory_is_displayed_not_invented(self) -> None:
        fake = _render(self._batlinet_session())
        rows = [row for frame in fake.rec["dataframes"] if isinstance(frame, list) for row in frame]
        gpu_rows = [row for row in rows if isinstance(row, dict) and "GPU" in row]
        self.assertEqual([row["Device"] for row in gpu_rows], ["cuda:0", "cuda:1"])
        self.assertEqual(gpu_rows[0]["Free (MiB)"], "2076")

    def test_cuda_oom_result_shows_actionable_message_and_keeps_stderr(self) -> None:
        session = self._batlinet_session()
        session["agent_execution_result"] = {
            "started": True,
            "completed": True,
            "success": False,
            "returncode": 1,
            "command": ["python", "batteryml_web/core/run_batteryml_cli.py", "run"],
            "workspace": "runs/agent_matr_batlinet_smoke",
            "stdout": "",
            "stderr": "Traceback...\ntorch.cuda.OutOfMemoryError: CUDA out of memory.",
            "error": "wrapper exited with returncode 1",
            "device": "cuda:0",
            "smoke": True,
            "artifacts": {},
            "dry_run": False,
        }
        fake = _render(session)
        errors = " ".join(fake.rec["errors"])
        self.assertIn("CUDA out of memory on cuda:0", errors)
        actions = " ".join(fake.rec["writes"])
        self.assertIn("CPU", actions)
        self.assertIn("cuda:1", actions)  # only the GPU with more free memory
        self.assertIn("never moves a failed run", actions)
        # The raw traceback is still displayed verbatim.
        self.assertTrue(any("OutOfMemoryError" in w for w in fake.rec["writes"]))
        # No metrics for a failed run.
        self.assertFalse(any(label in {"RMSE", "MAE", "MAPE"} for label, _ in fake.rec["metrics"]))


class TestLiveProgressRendering(unittest.TestCase):
    """Milestone B: progress is shown while running, and never invented."""

    def _slots(self) -> tuple[FakeStreamlit, Any]:
        fake = FakeStreamlit({})
        progress = ui._LiveProgress(fake.empty(), fake.empty(), fake.empty())
        return fake, progress

    def test_caption_shows_counter_only_when_the_run_reported_one(self) -> None:
        with_counter = ui.progress_caption("Training", 3, 10, 65.0)
        self.assertIn("Training", with_counter)
        self.assertIn("step 3/10", with_counter)
        self.assertIn("00:01:05", with_counter)
        without = ui.progress_caption("Loading data", None, None, 5.0)
        self.assertIn("Loading data", without)
        self.assertNotIn("step", without)
        self.assertNotIn("%", without)

    def test_bar_is_drawn_only_for_a_reported_fraction(self) -> None:
        fake, progress = self._slots()
        progress.on_event(
            {"type": "stage", "stage": "Loading data", "current": None,
             "total": None, "fraction": None, "message": "Seed is set to 0."}
        )
        self.assertEqual(fake.rec["progress"], [])
        progress.on_event(
            {"type": "training_progress", "stage": "Training", "current": 1,
             "total": 4, "fraction": 0.25, "message": "Training BatLiNet:  25%|#| 1/4"}
        )
        self.assertEqual(fake.rec["progress"], [0.25])

    def test_stage_changes_are_always_drawn_and_failure_is_an_error(self) -> None:
        fake, progress = self._slots()
        progress.on_event(
            {"type": "stage", "stage": "Loading data", "current": None,
             "total": None, "fraction": None, "message": "Seed is set to 0."}
        )
        progress.on_event(
            {"type": "completed", "stage": ui.STAGE_FAILED, "current": None,
             "total": None, "fraction": None, "message": "wrapper exited with returncode 1"}
        )
        self.assertTrue(any(ui.STAGE_FAILED in text for text in fake.rec["errors"]))

    def test_heartbeats_do_not_force_a_repaint(self) -> None:
        fake, progress = self._slots()
        first = {"type": "stage", "stage": "Training", "current": None,
                 "total": None, "fraction": None, "message": "Training BatLiNet:  0%|"}
        progress.on_event(first)
        drawn = len(fake.rec["infos"])
        for _ in range(20):
            progress.on_event(
                {"type": "stage", "stage": "Training", "current": None,
                 "total": None, "fraction": None, "message": ui.HEARTBEAT_MESSAGE}
            )
        self.assertEqual(len(fake.rec["infos"]), drawn)

    def test_progress_is_drawn_while_the_run_is_still_going(self) -> None:
        """R8: the page consumes callbacks during the run, not a final status.

        The stub records what the page had drawn at the moment each event was
        delivered. If the UI were blocking until the subprocess exited, every
        event would arrive with nothing drawn and the result already stored.
        """
        drawn_at_event: list[int] = []
        result_stored_at_event: list[bool] = []

        def stub_stream(plan, on_event=None, **kwargs):
            for event in (
                {"type": "stage", "stage": "Loading data", "current": None,
                 "total": None, "fraction": None, "message": "Seed is set to 0."},
                {"type": "training_progress", "stage": "Training", "current": 1,
                 "total": 4, "fraction": 0.25, "message": "Training X:  25%|#| 1/4"},
                {"type": "training_progress", "stage": "Training", "current": 4,
                 "total": 4, "fraction": 1.0, "message": "Training X: 100%|#| 4/4"},
            ):
                if on_event is not None:
                    on_event(event)
                drawn_at_event.append(len(fake_holder[0].rec["infos"]))
                result_stored_at_event.append(
                    "agent_result_context" in fake_holder[0].session_state
                )
            return {
                "started": True, "completed": True, "success": True, "returncode": 0,
                "command": plan["command"], "workspace": "runs/w",
                "stdout": "", "stderr": "", "error": None,
                "artifacts": {}, "dry_run": False,
                "final_stage": "Complete", "elapsed_s": 1.0,
            }

        real_stream = ui.execute_plan_streaming
        real_build = ui.build_execution_plan
        real_ctx = ui.build_result_context
        real_llm = ui.is_llm_configured
        ui.execute_plan_streaming = stub_stream
        ui.build_execution_plan = lambda d, t_, m, **kw: _ok_plan(m, epochs=kw.get("epochs"))
        ui.build_result_context = lambda ws, **kw: {
            "workspace": ws, "success": True, "metrics": {}, "warnings": []
        }
        ui.is_llm_configured = lambda: False
        fake_holder: list = []
        try:
            session = {
                "agent_workflow_context": _MATR_CONTEXT,
                "gpu_inventory": _FAKE_GPU_INVENTORY,
            }
            fake = FakeStreamlit(session, {"Run"})
            fake_holder.append(fake)
            with _fake_streamlit(fake):
                try:
                    ui.render_agent_workflow()
                except _Rerun:
                    pass
        finally:
            ui.execute_plan_streaming = real_stream
            ui.build_execution_plan = real_build
            ui.build_result_context = real_ctx
            ui.is_llm_configured = real_llm

        self.assertEqual(len(drawn_at_event), 3, "the page did not consume the events")
        # Something was drawn by the time the first event had been handled...
        self.assertGreater(drawn_at_event[0], 0)
        # ...and the drawing grew as later events arrived, which cannot happen
        # if the page waits for the process to exit before rendering.
        self.assertGreater(drawn_at_event[-1], 0)
        # The result context did not exist yet while progress was being drawn.
        self.assertEqual(result_stored_at_event, [False, False, False])

    def test_run_click_streams_and_stores_no_fabricated_metrics(self) -> None:
        emitted: list[dict] = []

        def stub_stream(plan, on_event=None, **kwargs):
            for event in (
                {"type": "stage", "stage": "Loading data", "current": None,
                 "total": None, "fraction": None, "message": "Seed is set to 0."},
                {"type": "training_progress", "stage": "Training", "current": 1,
                 "total": 1, "fraction": 1.0, "message": "Training BatLiNet: 100%|##| 1/1"},
                {"type": "completed", "stage": "Complete", "current": None,
                 "total": None, "fraction": 1.0, "message": "Finished in 6.8s."},
            ):
                emitted.append(event)
                if on_event is not None:
                    on_event(event)
            return {
                "started": True, "completed": True, "success": True, "returncode": 0,
                "command": plan["command"], "workspace": "runs/agent_matr_batlinet_smoke",
                "stdout": "{'RMSE': 794.32}", "stderr": "", "error": None,
                "artifacts": {"prediction_files": []}, "dry_run": False,
                "final_stage": "Complete", "elapsed_s": 6.8,
            }

        real_stream = ui.execute_plan_streaming
        real_build = ui.build_execution_plan
        real_ctx = ui.build_result_context
        real_llm = ui.is_llm_configured
        seen: list[dict] = []

        def stub_ctx(workspace, **kwargs):
            seen.append(kwargs)
            return {"workspace": workspace, "success": True, "metrics": {}, "warnings": []}

        ui.execute_plan_streaming = stub_stream
        ui.build_execution_plan = lambda d, t_, m, **kw: _ok_plan(m, epochs=kw.get("epochs"))
        ui.build_result_context = stub_ctx
        ui.is_llm_configured = lambda: False
        try:
            session = {
                "agent_workflow_context": _MATR_CONTEXT,
                "agent_original_request": "Use BatLiNet on MATR to predict battery lifetime.",
                "gpu_inventory": _FAKE_GPU_INVENTORY,
            }
            fake = _render(session, clicked={"Run"}, checkboxes={ui.SMOKE_CHECKBOX_LABEL: True})
        finally:
            ui.execute_plan_streaming = real_stream
            ui.build_execution_plan = real_build
            ui.build_result_context = real_ctx
            ui.is_llm_configured = real_llm

        self.assertEqual(len(emitted), 3)
        # Every fraction drawn came from an event that carried one; the UI
        # never manufactures a position of its own.
        carried = {e["fraction"] for e in emitted if e.get("fraction") is not None}
        self.assertTrue(fake.rec["progress"])
        self.assertTrue(set(fake.rec["progress"]) <= carried)
        # Metrics come from build_result_context, never from the streamed stdout.
        self.assertTrue(seen)
        self.assertEqual(seen[0]["run_status"], "SUCCESS")
        self.assertFalse(any("794.32" in text for text in fake.rec["infos"]))


class TestHandoverFromHome(unittest.TestCase):
    """Milestone C: a request handed over from Home is analysed, not executed."""

    def setUp(self) -> None:
        self._real_resolve = ui.resolve_workflow_intent_with_llm
        self._real_context = ui.build_agent_workflow_context
        self._real_catalog = ui.build_intent_catalog
        self._real_build = ui.build_execution_plan
        self._real_stream = ui.execute_plan_streaming
        self._real_llm = ui.is_llm_configured
        self.preview_calls: list[str] = []
        self.exec_calls: list[dict] = []

        def stub_resolve(request, **kwargs):
            self.preview_calls.append(request)
            return {
                "source": "deterministic",
                "llm_available": False,
                "dataset": "MATR",
                "task": "RUL prediction",
                "requested_model": "batlinet",
                "requested_model_name": None,
                "needs_clarification": False,
                "clarification_question": None,
                "suggested_model_keys": [],
                "user_facing_rationale": "",
                "rejected": [],
                "warnings": [],
                "message": "Resolved dataset=MATR, task=RUL prediction.",
            }

        def stub_stream(plan, **kwargs):
            self.exec_calls.append(plan)
            raise AssertionError("handover must never execute")

        ui.resolve_workflow_intent_with_llm = stub_resolve
        ui.build_agent_workflow_context = lambda dataset, task, **kw: _MATR_CONTEXT
        ui.build_intent_catalog = lambda **kw: _FAKE_CATALOG
        ui.build_execution_plan = lambda d, t_, m, **kw: _ok_plan(m, epochs=kw.get("epochs"))
        ui.execute_plan_streaming = stub_stream
        ui.is_llm_configured = lambda: False

    def tearDown(self) -> None:
        ui.resolve_workflow_intent_with_llm = self._real_resolve
        ui.build_agent_workflow_context = self._real_context
        ui.build_intent_catalog = self._real_catalog
        ui.build_execution_plan = self._real_build
        ui.execute_plan_streaming = self._real_stream
        ui.is_llm_configured = self._real_llm

    def test_pending_request_is_analysed_once_and_never_run(self) -> None:
        session = {
            "agent_pending_request": "Use BatLiNet on MATR to predict battery lifetime.",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
        }
        _render(session)
        self.assertEqual(
            self.preview_calls, ["Use BatLiNet on MATR to predict battery lifetime."]
        )
        self.assertEqual(self.exec_calls, [])
        # Consumed, so a rerun cannot re-analyse it.
        self.assertNotIn("agent_pending_request", session)
        self.assertEqual(session["agent_workflow_context"]["dataset"], "MATR")
        self.assertEqual(
            session["agent_original_request"],
            "Use BatLiNet on MATR to predict battery lifetime.",
        )

    def test_no_pending_request_means_no_analysis(self) -> None:
        session = {
            "agent_original_request": "Use BatLiNet on MATR to predict battery lifetime.",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
        }
        _render(session)
        self.assertEqual(self.preview_calls, [])
        self.assertEqual(self.exec_calls, [])


class TestLlmAssistedIntentInTheUi(unittest.TestCase):
    """Milestone D: the page shows what was understood and what was refused."""

    def test_offline_shortfall_is_stated_plainly(self) -> None:
        fake = FakeStreamlit({})
        with _fake_streamlit(fake):
            ui._render_intent(
                {
                    "source": "deterministic_fallback",
                    "llm_available": False,
                    "user_facing_rationale": "",
                    "warnings": [],
                    "rejected": [],
                    "needs_clarification": False,
                    "clarification_question": None,
                }
            )
        joined = " ".join(fake.rec["captions"])
        self.assertIn("advanced natural-language understanding is unavailable", joined)

    def test_rejected_values_are_shown_not_hidden(self) -> None:
        fake = FakeStreamlit({})
        with _fake_streamlit(fake):
            ui._render_intent(
                {
                    "source": "llm",
                    "llm_available": True,
                    "user_facing_rationale": "MATR has validated RUL configs.",
                    "warnings": ["Backend-owned field(s) discarded."],
                    "rejected": [
                        {"field": "config_path", "value": "configs/fake.yaml",
                         "kind": "forbidden_field"}
                    ],
                    "needs_clarification": True,
                    "clarification_question": "Which dataset should I use?",
                }
            )
        self.assertIn("MATR has validated RUL configs.", " ".join(fake.rec["writes"]))
        self.assertIn("Backend-owned field(s) discarded.", " ".join(fake.rec["warnings"]))
        # The refused values are shown; they sit inside the "Values the
        # platform refused" expander rather than the page body.
        shown = " ".join(fake.rec["writes"] + fake.rec["expander_writes"])
        self.assertIn("config_path", shown)
        self.assertIn("configs/fake.yaml", shown)
        self.assertIn("Which dataset should I use?", " ".join(fake.rec["infos"]))

    def test_blocked_model_is_reported_and_not_substituted(self) -> None:
        fake = FakeStreamlit({})
        presentation = ui.build_model_presentation(
            resolution={"requested_model": "PBT"},
            workflow_context=_MATR_CONTEXT,
            model_request={
                "requested": "PBT",
                "status": "blocked",
                "model_key": "PBT",
                "model_display": "PBT",
                "reason": "PBT is NEEDS_ASSETS for this dataset and task: provide assets.",
                "alternatives": ["cptransformer", "batlinet"],
                "requires_confirmation": True,
            },
        )
        with _fake_streamlit(fake):
            ui._render_model_presentation(presentation)
        joined = " ".join(fake.rec["writes"] + fake.rec["warnings"] + fake.rec["captions"])
        # Semantics, not punctuation: the blocker is stated, runnable
        # alternatives are offered, and nothing was chosen for the user.
        self.assertIn("NEEDS_ASSETS", joined)
        self.assertIn("Nothing was substituted", joined)
        self.assertEqual(
            [entry["model"] for entry in presentation["blocked_focus"]], ["PBT"]
        )
        alternatives = [entry["model"] for entry in presentation["primary"]]
        self.assertTrue(alternatives)
        self.assertTrue(set(alternatives) <= set(ui.runnable_model_keys(_MATR_CONTEXT)))
        # Each alternative is named on the page, by whatever display name the
        # backend gave it.
        display = ui.runnable_display_names(_MATR_CONTEXT)
        for key in alternatives:
            self.assertIn(display[key], joined)
        # The blocked model is never offered as a choice, and nothing is preselected.
        self.assertNotIn("PBT", ui.selectbox_options(presentation))
        self.assertEqual(ui.selectbox_options(presentation)[0], ui.NO_SELECTION)
        self.assertIsNone(presentation["default_model"])

    def test_a_requested_ready_model_is_shown_alone(self) -> None:
        fake = FakeStreamlit({})
        presentation = ui.build_model_presentation(
            resolution={"requested_model": "cptransformer"},
            workflow_context=_MATR_CONTEXT,
            model_request={
                "requested": "cptransformer",
                "status": "runnable",
                "model_key": "cptransformer",
                "model_display": "CPTransformer",
                "reason": "",
                "alternatives": ["batlinet"],
                "requires_confirmation": False,
            },
        )
        with _fake_streamlit(fake):
            ui._render_model_presentation(presentation)
        named = " ".join(fake.rec["writes"])
        self.assertIn("CPTransformer", named)
        # The other runnable model is not dumped into the primary view.
        self.assertNotIn("BatLiNet", named)
        self.assertIn("BatLiNet", " ".join(fake.rec["expander_writes"]))
        self.assertEqual(presentation["default_model"], "cptransformer")

    def test_the_full_list_appears_only_behind_an_expander(self) -> None:
        fake = FakeStreamlit({})
        presentation = ui.build_model_presentation(
            resolution={"requested_model": "cptransformer"},
            workflow_context=_MATR_CONTEXT,
            model_request={
                "requested": "cptransformer", "status": "runnable",
                "model_key": "cptransformer", "model_display": "CPTransformer",
                "reason": "", "alternatives": [], "requires_confirmation": False,
            },
        )
        with _fake_streamlit(fake):
            ui._render_model_presentation(presentation)
        self.assertTrue(
            any("Show all compatible models" in label for label in fake.rec["expanders"])
        )

    def test_blocked_suggestions_are_never_offered_as_runnable(self) -> None:
        fake = FakeStreamlit({})
        presentation = ui.build_model_presentation(
            resolution={"recommendation_requested": True},
            workflow_context=_MATR_CONTEXT,
            suggestions={"runnable": ["cptransformer"], "blocked": ["PBT"], "unknown": ["MegaNet"]},
        )
        with _fake_streamlit(fake):
            ui._render_model_presentation(presentation)
        self.assertEqual([e["model"] for e in presentation["primary"]], ["cptransformer"])
        self.assertEqual([e["model"] for e in presentation["blocked_focus"]], ["PBT"])
        self.assertNotIn("PBT", presentation["selectable"])

    def test_analysis_stores_canonical_state_and_never_executes(self) -> None:
        real_resolve = ui.resolve_workflow_intent_with_llm
        real_context = ui.build_agent_workflow_context
        real_catalog = ui.build_intent_catalog
        real_llm = ui.is_llm_configured
        real_stream = ui.execute_plan_streaming
        executed: list = []

        ui.resolve_workflow_intent_with_llm = lambda request, **kw: {
            "source": "llm",
            "llm_available": True,
            "dataset": "MATR",
            "task": "RUL prediction",
            "requested_model": "PBT",
            "requested_model_name": None,
            "needs_clarification": False,
            "clarification_question": None,
            "suggested_model_keys": ["cptransformer", "PBT"],
            "user_facing_rationale": "PBT was asked for but is blocked.",
            "rejected": [],
            "warnings": [],
            "message": "ok",
        }
        ui.build_agent_workflow_context = lambda dataset, task, **kw: _MATR_CONTEXT
        ui.build_intent_catalog = lambda **kw: _FAKE_CATALOG
        ui.is_llm_configured = lambda: False
        ui.execute_plan_streaming = lambda plan, **kw: executed.append(plan)

        session: dict = {}
        fake = FakeStreamlit(session)
        try:
            with _fake_streamlit(fake):
                ui._analyze_request("Use PBT on MATR to predict battery lifetime.")
        finally:
            ui.resolve_workflow_intent_with_llm = real_resolve
            ui.build_agent_workflow_context = real_context
            ui.build_intent_catalog = real_catalog
            ui.is_llm_configured = real_llm
            ui.execute_plan_streaming = real_stream

        self.assertEqual(executed, [])
        self.assertEqual(session["agent_workflow_context"]["dataset"], "MATR")
        self.assertEqual(session["agent_model_request"]["status"], "blocked")
        # A blocked request must not preselect anything.
        self.assertNotIn("agent_selected_model", session)
        self.assertEqual(session["agent_suggestions"]["runnable"], ["cptransformer"])
        self.assertEqual(session["agent_suggestions"]["blocked"], ["PBT"])


class TestRequestTabsAndExamples(unittest.TestCase):
    """B: two ways to supply a request; examples fill the box and stop there."""

    def setUp(self) -> None:
        self._real_resolve = ui.resolve_workflow_intent_with_llm
        self._real_catalog = ui.build_intent_catalog
        self._real_examples = ui.build_example_requests
        self._real_stream = ui.execute_plan_streaming
        self._real_llm = ui.is_llm_configured
        self.analysed: list[str] = []
        self.executed: list = []

        def stub_resolve(request, **kwargs):
            self.analysed.append(request)
            raise AssertionError("clicking an example must not analyse")

        def stub_stream(plan, **kwargs):
            self.executed.append(plan)
            raise AssertionError("clicking an example must not execute")

        ui.resolve_workflow_intent_with_llm = stub_resolve
        ui.execute_plan_streaming = stub_stream
        ui.build_intent_catalog = lambda **kw: _FAKE_CATALOG
        ui.build_example_requests = lambda catalog, **kw: list(_FAKE_EXAMPLES)
        ui.is_llm_configured = lambda: False

    def tearDown(self) -> None:
        ui.resolve_workflow_intent_with_llm = self._real_resolve
        ui.build_intent_catalog = self._real_catalog
        ui.build_example_requests = self._real_examples
        ui.execute_plan_streaming = self._real_stream
        ui.is_llm_configured = self._real_llm

    def test_the_two_tabs_are_rendered_in_order(self) -> None:
        fake = _render({"gpu_inventory": _FAKE_GPU_INVENTORY})
        self.assertIn(["Describe request", "Examples"], fake.rec["tabs"])
        self.assertEqual(ui.REQUEST_TABS, ("Describe request", "Examples"))

    def test_examples_are_offered_as_buttons(self) -> None:
        fake = _render({"gpu_inventory": _FAKE_GPU_INVENTORY})
        labels = [button["label"] for button in fake.rec["buttons"]]
        for example in _FAKE_EXAMPLES:
            self.assertIn(example, labels)

    def test_clicking_an_example_fills_the_box_and_does_nothing_else(self) -> None:
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY, "agent_request_input": "stale text"}
        chosen = _FAKE_EXAMPLES[0]
        _render(session, clicked={chosen})
        # Written into the same field the Describe request tab reads...
        self.assertEqual(session["agent_original_request"], chosen)
        # ...the stale widget value is cleared so the box shows it...
        self.assertNotIn("agent_request_input", session)
        # ...and nothing was analysed or executed.
        self.assertEqual(self.analysed, [])
        self.assertEqual(self.executed, [])
        self.assertNotIn("agent_workflow_context", session)

    def test_examples_are_computed_once_and_cached(self) -> None:
        calls: list = []

        def counting(catalog, **kwargs):
            calls.append(kwargs)
            return list(_FAKE_EXAMPLES)

        ui.build_example_requests = counting
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY}
        _render(session)
        _render(session)
        self.assertEqual(len(calls), 1)
        self.assertEqual(session["agent_example_requests"], list(_FAKE_EXAMPLES))

    def test_an_empty_catalog_offers_no_example_rather_than_a_fake_one(self) -> None:
        ui.build_example_requests = lambda catalog, **kw: []
        fake = _render({"gpu_inventory": _FAKE_GPU_INVENTORY})
        self.assertTrue(any("No example can be offered" in text for text in fake.rec["infos"]))

    def test_no_model_name_is_written_into_the_example_ui(self) -> None:
        """The examples come from the backend; the page names no model."""
        source = Path(ui.__file__).read_text(encoding="utf-8")
        render_examples = source.split("def _render_examples")[1].split("\ndef ")[0]
        for name in ("CPTransformer", "BatLiNet", "XGBoost", "PBT", "MATR"):
            self.assertNotIn(name, render_examples)


class TestGpuMultiSelectRender(unittest.TestCase):
    """G: CPU / one GPU / several, with Run gated on backend capability."""

    def setUp(self) -> None:
        self._real_build = ui.build_execution_plan
        self._real_stream = ui.execute_plan_streaming
        self._real_llm = ui.is_llm_configured
        self.plans: list[dict] = []
        self.executed: list = []

        def stub_build(dataset, task, model, **kwargs):
            self.plans.append({"model": model, **kwargs})
            return _ok_plan(model, epochs=kwargs.get("epochs"))

        ui.build_execution_plan = stub_build
        ui.execute_plan_streaming = lambda plan, **kw: self.executed.append(plan)
        ui.is_llm_configured = lambda: False

    def tearDown(self) -> None:
        ui.build_execution_plan = self._real_build
        ui.execute_plan_streaming = self._real_stream
        ui.is_llm_configured = self._real_llm

    def _session(self) -> dict:
        return {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_original_request": "Use BatLiNet on MATR to predict battery lifetime.",
            "agent_selected_model": "batlinet",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
        }

    def test_no_gpu_selected_runs_on_cpu(self) -> None:
        session = self._session()
        fake = _render(session, multiselect_values={"GPUs (leave empty to run on CPU)": []})
        self.assertEqual(session["agent_device_selection"], {"mode": "cpu", "gpu_ids": []})
        self.assertEqual(self.plans[-1]["device"], "cpu")
        run_buttons = [b for b in fake.rec["buttons"] if b["label"] == ui.RUN_BUTTON_LABEL]
        self.assertTrue(run_buttons and run_buttons[0]["disabled"] is False)

    def test_one_gpu_selected_stays_executable(self) -> None:
        session = self._session()
        fake = _render(
            session, multiselect_values={"GPUs (leave empty to run on CPU)": ["cuda:1"]}
        )
        self.assertEqual(
            session["agent_device_selection"], {"mode": "single_gpu", "gpu_ids": [1]}
        )
        self.assertEqual(self.plans[-1]["device"], "cuda:1")
        run_buttons = [b for b in fake.rec["buttons"] if b["label"] == ui.RUN_BUTTON_LABEL]
        self.assertTrue(run_buttons and run_buttons[0]["disabled"] is False)

    def test_several_gpus_disable_run_with_the_exact_message(self) -> None:
        session = self._session()
        fake = _render(
            session,
            multiselect_values={"GPUs (leave empty to run on CPU)": ["cuda:0", "cuda:1"]},
        )
        # The selection is kept exactly as chosen, not reduced to one GPU.
        self.assertEqual(
            session["agent_device_selection"], {"mode": "multi_gpu", "gpu_ids": [0, 1]}
        )
        # No plan was built, so no device string was needed.
        self.assertEqual(self.plans, [])
        self.assertEqual(self.executed, [])
        errors = " ".join(fake.rec["errors"])
        self.assertIn(
            "This model/runtime does not yet support multi-GPU execution.", errors
        )
        run_buttons = [b for b in fake.rec["buttons"] if b["label"] == ui.RUN_BUTTON_LABEL]
        self.assertTrue(run_buttons)
        self.assertTrue(all(button["disabled"] for button in run_buttons))

    def test_a_multi_gpu_selection_cannot_be_executed_by_clicking(self) -> None:
        session = self._session()
        _render(
            session,
            clicked={ui.RUN_BUTTON_LABEL},
            multiselect_values={"GPUs (leave empty to run on CPU)": ["cuda:0", "cuda:1"]},
        )
        self.assertEqual(self.executed, [])

    def test_the_capability_fields_are_shown(self) -> None:
        session = self._session()
        fake = _render(
            session,
            multiselect_values={"GPUs (leave empty to run on CPU)": ["cuda:0", "cuda:1"]},
        )
        captions = " ".join(fake.rec["captions"])
        self.assertIn("supports_multi_gpu: False", captions)
        self.assertIn("multi_gpu_strategy: None", captions)

    def test_the_inventory_table_shows_utilization(self) -> None:
        fake = _render(self._session(), multiselect_values={})
        rows = [
            row
            for frame in fake.rec["dataframes"]
            if isinstance(frame, list)
            for row in frame
            if isinstance(row, dict) and "GPU" in row
        ]
        self.assertTrue(rows)
        self.assertIn("Utilization (%)", rows[0])


class TestTheFakeCoversTheRealSurface(unittest.TestCase):
    """The fake must offer exactly what the page actually calls.

    The regression this guards against really happened: the page moved to
    ``st.header`` for the Agent Workflow title (a native primitive, so no
    heading-anchor markup leaks into the UI), the fake was never told, and
    every render test in this file died with ``FakeStreamlit has no
    attribute 'header'``. Because those tests could not even import in the
    authoring environment, the breakage was invisible until the server ran
    them.

    Comparing the two surfaces mechanically means the next primitive the
    page adopts fails here, loudly, instead of on someone else's machine.
    """

    def _production_surface(self) -> set[str]:
        import ast

        used: set[str] = set()
        for module in (ui, ui_feedback, ui_llm_state):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "st"
                ):
                    used.add(node.attr)
        return used

    def test_the_fake_provides_every_primitive_the_page_uses(self) -> None:
        fake = FakeStreamlit({})
        missing = sorted(
            name for name in self._production_surface() if not hasattr(fake, name)
        )
        self.assertEqual(
            missing,
            [],
            "the page calls Streamlit primitives the fake does not provide, so "
            "its render tests cannot run",
        )

    def test_headings_use_native_primitives_not_markdown(self) -> None:
        """UI-03: markdown headings carry an anchor; native ones can suppress it."""
        source = Path(ui.__file__).read_text(encoding="utf-8")
        for bad in ('st.markdown("# ', 'st.markdown("## ', 'st.markdown("### '):
            self.assertNotIn(bad, source)

    def test_the_page_title_is_the_undecorated_label(self) -> None:
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY}
        fake = _render(session)
        self.assertIn("Agent Workflow", fake.rec["headings"])
        for decorated in ("🤖 Agent Workflow", "▶ Agent Workflow", "▶ 🤖 Agent Workflow"):
            self.assertNotIn(decorated, fake.rec["headings"])


class TestTheLlmResolverIsOnTheRealPath(unittest.TestCase):
    """R6: the page really calls the LLM-assisted resolver, with credentials.

    Asserting that llm_intent.py exists proves nothing. These drive the
    actual Analyze path and check what reached the resolver.
    """

    def setUp(self) -> None:
        self._real_resolve = ui.resolve_workflow_intent_with_llm
        self._real_context = ui.build_agent_workflow_context
        self._real_catalog = ui.build_intent_catalog
        self._real_guide = ui.guide_workflow
        self.calls: list[dict] = []

        def stub_resolve(request, **kwargs):
            self.calls.append({"request": request, **kwargs})
            return {
                "source": "llm", "llm_available": True,
                "dataset": "MATR", "task": "RUL prediction",
                "requested_model": None, "requested_model_name": None,
                "recommendation_requested": True, "needs_clarification": False,
                "clarification_question": None,
                "suggested_model_keys": ["cptransformer"],
                "user_facing_rationale": "ok", "rejected": [], "warnings": [],
                "message": "ok",
            }

        ui.resolve_workflow_intent_with_llm = stub_resolve
        ui.build_agent_workflow_context = lambda dataset, task, **kw: _MATR_CONTEXT
        ui.build_intent_catalog = lambda **kw: _FAKE_CATALOG
        ui.guide_workflow = lambda *a, **kw: {"guidance": "explained"}

    def tearDown(self) -> None:
        ui.resolve_workflow_intent_with_llm = self._real_resolve
        ui.build_agent_workflow_context = self._real_context
        ui.build_intent_catalog = self._real_catalog
        ui.guide_workflow = self._real_guide

    def _analyze(self, session: dict, request: str) -> None:
        fake = FakeStreamlit(session)
        with _fake_streamlit(fake):
            ui._analyze_request(request)

    def test_configured_credentials_reach_the_resolver(self) -> None:
        session = {
            "llm_base_url": "https://example.invalid/v1",
            "llm_api_key": "secret",
            "llm_model": "some-model",
        }
        self._analyze(session, "Can you recommend something for cycle aging on MATR?")
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["request"], "Can you recommend something for cycle aging on MATR?")
        self.assertEqual(call["base_url"], "https://example.invalid/v1")
        self.assertEqual(call["api_key"], "secret")
        self.assertEqual(call["model"], "some-model")
        self.assertIs(call["catalog"], _FAKE_CATALOG)

    def test_without_credentials_the_resolver_is_still_used_offline(self) -> None:
        """The deterministic path runs through the same entry point."""
        session: dict = {}
        self._analyze(session, "Use CPTransformer on MATR to predict battery lifetime.")
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("api_key", self.calls[0])

    def test_the_structured_result_drives_the_page_not_prose(self) -> None:
        session = {
            "llm_base_url": "u", "llm_api_key": "k", "llm_model": "m",
        }
        self._analyze(session, "Which model should I use for MATR lifetime?")
        # Canonical identifiers, taken from the validated structure.
        self.assertEqual(session["agent_workflow_context"]["dataset"], "MATR")
        self.assertEqual(session["agent_intent"]["task"], "RUL prediction")
        # Suggestions were split by readiness before reaching the page.
        self.assertEqual(session["agent_suggestions"]["runnable"], ["cptransformer"])
        # The presentation is a recommendation, capped and readiness-filtered.
        presentation = ui.build_model_presentation(
            resolution=session["agent_intent"],
            workflow_context=_MATR_CONTEXT,
            model_request=session["agent_model_request"],
            suggestions=session["agent_suggestions"],
        )
        self.assertEqual(presentation["mode"], "recommendation")
        self.assertEqual([e["model"] for e in presentation["primary"]], ["cptransformer"])


class TestNoFrontendDuplication(unittest.TestCase):
    """The UI module must not reimplement what the backend owns."""

    def setUp(self) -> None:
        self.source = Path(ui.__file__).read_text(encoding="utf-8")

    def test_no_subprocess_or_shell_execution_in_ui(self) -> None:
        self.assertNotIn("import subprocess", self.source)
        self.assertNotIn("shell=True", self.source)
        self.assertNotIn("os.system", self.source)
        self.assertNotIn("popen", self.source.lower())

    def test_no_duplicated_readiness_or_validator_or_metric_logic(self) -> None:
        for forbidden in (
            "get_workflow_context",
            "load_and_validate_config_file",
            "assess_dataset",
            "assess_model",
        ):
            self.assertNotIn(forbidden, self.source, f"UI must not duplicate {forbidden}")

    def test_no_metric_computation_in_ui(self) -> None:
        lowered = self.source.lower()
        # Metrics may be *mentioned* in docstrings, but never computed.
        for token in ("sqrt(", "mean_absolute", "np.mean(", "torch.mean("):
            self.assertNotIn(token, lowered)
        self.assertNotIn("def rmse", lowered)
        self.assertNotIn("def mae", lowered)
        self.assertNotIn("def mape(", lowered)


class TestBackendPlanContractsUsedByTheUI(unittest.TestCase):
    """Registry-backed checks that the UI gate (can_execute) is correct."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_ready_models_produce_executable_plans(self) -> None:
        plan = self._plan("MATR", "RUL prediction", "cptransformer")
        self.assertTrue(ui.run_is_enabled(plan), plan.get("reason"))
        self.assertEqual(plan["model"], "cptransformer")
        self.assertTrue(plan["config_path"].endswith("configs/external/cptransformer/matr_1.yaml"))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_blocked_pbt_plan_disabled(self) -> None:
        plan = self._plan("MATR", "RUL prediction", "PBT")
        self.assertFalse(ui.run_is_enabled(plan))
        self.assertTrue(plan["command"] in (None, []))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_trajectory_batterymformer_never_rul_executable(self) -> None:
        plan = self._plan("MATR", "Degradation trajectory forecasting", "BatteryMFormer")
        self.assertFalse(ui.run_is_enabled(plan))
        self.assertTrue(plan["command"] in (None, []))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_mich_raw_only_plan_disabled_and_without_command(self) -> None:
        plan = self._plan("MICH", "RUL prediction", "cptransformer")
        self.assertEqual(plan["dataset_status"], "RAW_ONLY")
        self.assertFalse(ui.run_is_enabled(plan))
        self.assertIsNone(plan["command"])
        self.assertNotIn("data/raw/MICH", str(plan))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_smoke_plan_command_carries_only_epochs_override(self) -> None:
        plan = self._plan("MATR", "RUL prediction", "cptransformer", epochs=1)
        self.assertTrue(ui.run_is_enabled(plan))
        self.assertIn("--epochs", plan["command"])
        self.assertEqual(plan["command"][plan["command"].index("--epochs") + 1], "1")
        # The verified CLI contract is intact: wrapper, run, config, train, eval, device, seed.
        self.assertEqual(plan["command"][1], "batteryml_web/core/run_batteryml_cli.py")
        self.assertEqual(plan["command"][2], "run")
        self.assertIn("--train", plan["command"])
        self.assertIn("--eval", plan["command"])
        self.assertIn("--seed", plan["command"])
        self.assertIn("--device", plan["command"])

    @staticmethod
    def _plan(dataset: str, task: str, model: str, epochs: int | None = None) -> dict:
        return ui.build_execution_plan(
            dataset, task, model, epochs=epochs, workspace="runs/ui_boundary_check"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
