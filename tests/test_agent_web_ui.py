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

import json
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import (  # noqa: E402
    DEMO_ENVIRONMENT_AVAILABLE,
    DEMO_ENVIRONMENT_SKIP_MESSAGE,
)
from batteryml_web.agent.results import SMOKE_RUN_WARNING  # noqa: E402
from batteryml_web.agent.run_policy import SMOKE_NOTICE  # noqa: E402
from batteryml_web.core import device_inventory as di  # noqa: E402
from batteryml_web.agent.catalog import (  # noqa: E402
    REQUEST_TEMPLATE_STRUCTURE,
)
from batteryml_web.core.run_identity import is_run_id  # noqa: E402
from batteryml_web.core.project_context import (  # noqa: E402
    agent_workspace,
    resolve_project,
)
from batteryml_web.ui import agent_workflow as ui  # noqa: E402
from batteryml_web.ui import feedback as ui_feedback  # noqa: E402
from batteryml_web.ui import llm_state as ui_llm_state  # noqa: E402
from batteryml_web.ui import settings_panel as ui_settings  # noqa: E402

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


def _code_without_prose(module) -> str:
    """A module's code with docstrings and comments dropped.

    Static scans in this file assert what the code *does*. A docstring or a
    ``#`` comment that quotes an old label, an example metric or a GPU name
    while explaining why it is gone is documentation, and reading it as
    behaviour makes a test that punishes the codebase for being explained.
    """
    import ast

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
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
    # ast.unparse drops ``#`` comments as a side effect of re-rendering
    # from the tree, which is the other half of what this needs.
    return ast.unparse(tree)


class _FakeRunManager:
    """Records what the page asks of the manager; starts nothing.

    The page no longer calls the streaming executor itself — it hands the
    validated plan to a RunManager that executes on its own thread. These
    tests assert the page's half of that contract: which run it started, that
    it started exactly one, and that it started none for a refused plan.
    """

    def __init__(self, snapshots=None):
        self.started: list[dict] = []
        self.cancelled: list[str] = []
        self._snapshots = dict(snapshots or {})

    def start(self, *, run_id, run_dir, plan, repo_root, env=None, on_finished=None):
        self.started.append({"run_id": run_id, "run_dir": str(run_dir), "plan": plan})
        return None

    def snapshot(self, run_id):
        return self._snapshots.get(str(run_id))

    def get(self, run_id):
        return None

    def cancel(self, run_id, **kwargs):
        self.cancelled.append(str(run_id))
        return True

    def forget(self, run_id):
        pass

    def live_run_ids(self):
        return list(self._snapshots)


class _Col:
    """Stand-in for one st.columns() cell.

    It holds the fake itself, not just its recorder, because a widget in a
    column is a widget: a button here must honour the same clicked set as one
    rendered at top level, or a two-button row (Save / Connect) could never
    be exercised.
    """

    def __init__(self, fake: "FakeStreamlit"):
        self._fake = fake
        self._rec = fake.rec

    def metric(self, label, value, **kwargs):
        self._rec["metrics"].append((str(label), str(value)))

    def caption(self, text, **kwargs):
        self._rec["captions"].append(str(text))

    def button(self, label, **kwargs):
        return self._fake.button(label, **kwargs)

    def markdown(self, text, **kwargs):
        self._rec["markdown"].append(str(text))

    def write(self, *args, **kwargs):
        self._fake.write(*args, **kwargs)


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
        if kwargs.get("text") is not None:
            self._rec["captions"].append(str(kwargs["text"]))

    def container(self, **kwargs):
        return _Ctx()


class FakeStreamlit:
    """Minimal stand-in for the streamlit surface the UI module uses."""

    def __init__(
        self,
        session: dict,
        clicked_buttons: set[str] | None = None,
        checkbox_values: dict[str, bool] | None = None,
        multiselect_values: dict[str, list] | None = None,
        radio_values: dict[str, str] | None = None,
        selectbox_values: dict[str, str] | None = None,
    ):
        self.session_state = session
        self._clicked = set(clicked_buttons or ())
        self._checkbox_values = dict(checkbox_values or {})
        self._multiselect_values = dict(multiselect_values or {})
        self._radio_values = dict(radio_values or {})
        self._selectbox_values = dict(selectbox_values or {})
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
            "statuses": [],
            "tabs": [],
            "toasts": [],
            "expanders": [],
            "expander_states": [],
            "expander_writes": [],
            "multiselects": [],
            "headings": [],
            "radios": [],
            "text_inputs": [],
            "text_areas": [],
            "dialogs": [],
            "fragments": [],
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
        # Routed like write(): a code block inside a disclosure is not in the
        # primary view, and "is the raw log the main display?" is exactly the
        # question this fake has to be able to answer.
        bucket = "expander_writes" if self.in_expander else "codes"
        self.rec[bucket].append(str(text))

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
        return [_Col(self) for _ in range(count)]

    # --- widgets -------------------------------------------------------
    def text_area(self, label, value="", **kwargs):
        # The placeholder is recorded because the request suggestion *is*
        # the placeholder: grey, and not the user's request until accepted.
        # "value" is what the box shows: Streamlit takes a keyed widget's
        # content from its session key when that key exists. "default" is
        # the raw ``value=`` argument, recorded separately because a
        # non-empty default on a key written through Session State makes
        # Streamlit log a warning.
        key = kwargs.get("key")
        if key is not None and str(key) in self.session_state:
            shown = str(self.session_state[str(key)] or "")
        else:
            shown = str(value or "")
        self.rec["text_areas"].append(
            {
                "label": str(label),
                "value": shown,
                "default": str(value or ""),
                "placeholder": str(kwargs.get("placeholder", "")),
            }
        )
        return shown

    def selectbox(self, label, options, index=0, **kwargs):
        opts = list(options)
        self.rec["selectboxes"].append({"label": str(label), "options": [str(o) for o in opts]})
        if str(label) in self._selectbox_values:
            return self._selectbox_values[str(label)]
        return opts[index if index is not None else 0] if opts else None

    def checkbox(self, label, **kwargs):
        return bool(self._checkbox_values.get(str(label), False))

    def text_input(self, label, value="", **kwargs):
        # ``type`` is recorded because the API key field must be masked; the
        # *value* of a password field is deliberately not echoed anywhere a
        # test recorder could leak it beyond this dict.
        self.rec["text_inputs"].append(
            {"label": str(label), "type": str(kwargs.get("type", "default"))}
        )
        key = kwargs.get("key")
        if key is not None and str(key) in self.session_state:
            return str(self.session_state[str(key)] or "")
        return str(value or "")

    def dialog(self, title, **kwargs):
        """Stand-in for ``st.dialog``: records the title, runs the body.

        The real decorator needs a live script run; here it simply returns
        the function, so calling it executes the body inline and the widgets
        inside land in the same recorder as everything else.
        """
        self.rec["dialogs"].append(str(title))

        def decorator(func):
            return func

        return decorator

    def radio(self, label, options, index=0, **kwargs):
        opts = [str(o) for o in options]
        self.rec["radios"].append({"label": str(label), "options": opts})
        if str(label) in self._radio_values:
            return self._radio_values[str(label)]
        return opts[index if index is not None else 0]

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
        # Recorded separately from the label list, which other tests compare
        # against plain strings. "Does this panel open by itself?" is a
        # question about ``expanded``, and it had no answer here before.
        self.rec["expander_states"].append(
            (str(label), bool(kwargs.get("expanded", False)))
        )
        return _ExpanderCtx(self)

    def spinner(self, label, **kwargs):
        return _Ctx()

    def status(self, label, **kwargs):
        """st.status: the indeterminate counterpart of st.progress.

        Recorded with its state so a test can tell "working, amount unknown"
        from a bar. Its label goes into the same caption stream as
        st.progress's ``text``, because to a reader it is the same line.
        """
        self.rec["statuses"].append((str(label), str(kwargs.get("state", "running"))))
        self.rec["captions"].append(str(label))
        return _Ctx()

    def empty(self):
        return _Slot(self.rec)

    def progress(self, value=0.0, **kwargs):
        self.rec["progress"].append(float(value))
        if kwargs.get("text") is not None:
            self.rec["captions"].append(str(kwargs["text"]))
        return _Slot(self.rec)

    def rerun(self):
        raise _Rerun()

    def fragment(self, func=None, **kwargs):
        """Stand-in for st.fragment(run_every=...).

        The real decorator re-executes only its own block on a timer. Here it
        returns the function unchanged, so calling it runs the body once —
        which is what a render test wants: one deterministic draw, not a
        timer.
        """
        self.rec["fragments"].append(dict(kwargs))

        def decorator(inner):
            return inner

        return decorator if func is None else func


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


def _ok_plan(
    model: str = "cptransformer",
    epochs: int | None = None,
    workspace: str = "runs/agent_matr_cptransformer",
) -> dict:
    """A planner stub shaped like a real plan.

    ``workspace`` is threaded into the argv because that is where the real
    planner puts it and where provenance reads it back from: a fixture that
    ignored the caller's workspace would have the stub, not the page, decide
    which project a run belongs to — exactly the bug these tests exist to
    catch.
    """
    command = [
        "python",
        "batteryml_web/core/run_batteryml_cli.py",
        "run",
        "configs/external/cptransformer/matr_1.yaml",
        "--workspace",
        str(workspace),
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
    radio_values: dict[str, str] | None = None,
    selectbox_values: dict[str, str] | None = None,
) -> FakeStreamlit:
    fake = FakeStreamlit(
        session, clicked, checkboxes, multiselect_values, radio_values,
        selectbox_values,
    )
    with _fake_streamlit(fake):
        try:
            ui.render_agent_workflow()
        except _Rerun:
            pass
    return fake


class _fake_streamlit:
    """Point every UI module that reads Streamlit at the same fake.

    Four modules hold their own ``st``: the page, the feedback helper
    (which reads ``st.session_state`` to deduplicate toasts), the LLM
    credential store and the settings panel (which opens the configuration
    form and writes those credentials). Missing one does not raise — it
    silently reads the *real* Streamlit's empty session state, which is how
    ``is_llm_configured()`` quietly returned False and made a test look like
    a production bug, and how the settings panel reached the real
    ``st.dialog`` and raised ``open() is not a valid Streamlit command``.
    """

    def __init__(self, fake: FakeStreamlit):
        self._fake = fake
        self._saved: list = []

    def __enter__(self):
        for module in (ui, ui_feedback, ui_llm_state, ui_settings):
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
        """The UI owns no workspace rule of its own.

        There used to be a second copy of this logic in the page, with its
        own literal default path. Workspaces now come from
        ``core.project_context`` only, so this asserts that module's
        guarantees: always under ``runs/``, smoke runs kept apart, and a
        project root that tries to leave the runs area brought back into it.
        """
        smoke = agent_workspace(
            resolve_project(project_root=None), "MATR", "cptransformer", smoke=True
        )
        self.assertTrue(smoke.startswith("runs/"))
        self.assertTrue(smoke.endswith("_smoke"))
        full = agent_workspace(
            resolve_project(project_root=None), "MATR", "cptransformer", smoke=False
        )
        self.assertFalse(full.endswith("_smoke"))
        escaped = agent_workspace(
            resolve_project(project_root="/tmp/elsewhere"),
            "MATR",
            "cptransformer",
            smoke=False,
        )
        self.assertTrue(escaped.startswith("runs/"))
        self.assertNotIn("/tmp/", escaped)

    def test_the_page_defines_no_workspace_of_its_own(self) -> None:
        source = Path(ui.__file__).read_text(encoding="utf-8")
        self.assertNotIn("def workspace_for_request", source)


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

        self.manager = _FakeRunManager()
        self._real_manager = ui.get_run_manager
        self._real_write = ui.write_manifest
        ui.get_run_manager = lambda: self.manager
        # Nothing in a render test may touch the real runs/ directory.
        self._manifests: list = []
        ui.write_manifest = lambda run_dir, manifest: self._manifests.append(
            (str(run_dir), manifest)
        )

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
        ui.get_run_manager = self._real_manager
        ui.write_manifest = self._real_write

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
        """Run hands one job to the manager; nothing else starts anything.

        Execution moved off the script's thread, so what the page is
        responsible for is starting exactly one run, exactly when the button
        is pressed, and never for a plan the backend refused.
        """
        # Button not clicked: nothing started.
        _render(self._session(_MATR_CONTEXT))
        self.assertEqual(self.manager.started, [])
        self.assertEqual(self._exec_calls, [])
        # Button clicked with a can_execute plan: started exactly once, with
        # a real run id, and recorded on disk before anything ran.
        _render(self._session(_MATR_CONTEXT), clicked={"Run"})
        self.assertEqual(len(self.manager.started), 1)
        self.assertTrue(is_run_id(self.manager.started[0]["run_id"]))
        self.assertEqual(len(self._manifests), 1)
        self.assertEqual(
            self._manifests[0][1]["status"], "PLANNED",
            "the manifest must exist before execution, not after",
        )
        # The page itself never calls the blocking executor any more.
        self.assertEqual(self._exec_calls, [])
        # Blocked plan (can_execute False): Run disabled, manager untouched.
        self.manager.started.clear()
        ui.build_execution_plan = lambda dataset, task, model, **kw: _failed_plan(model)
        fake = _render(self._session(_MATR_CONTEXT), clicked={"Run"})
        self.assertEqual(self.manager.started, [])
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
        # Compact row: short enough to read, canonical value preserved.
        compact = " ".join(fake.rec["markdown"])
        self.assertIn("455.585", compact)
        self.assertIn("393.852", compact)
        self.assertIn("0.662", compact)
        self.assertIn("66.22%", compact)
        # Full precision remains available in the details expander.
        details = " ".join(fake.rec["captions"] + fake.rec["expander_writes"])
        self.assertIn("455.5853271484375", details)
        self.assertIn("0.6621854901313782", details)
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
        # CPU is the default compute mode, chosen explicitly.
        modes = [box for box in fake.rec["radios"] if box["label"] == ui.COMPUTE_LABEL]
        self.assertTrue(modes, "the compute-device control is missing")
        self.assertEqual(modes[0]["options"], ["CPU", "GPU"])
        self.assertEqual(
            session["agent_device_selection"], {"mode": "cpu", "gpu_ids": []}
        )
        joined = " ".join(fake.rec["infos"])
        self.assertIn(SMOKE_NOTICE, joined)

    def test_non_smoke_render_leaves_the_config_schedule_alone(self) -> None:
        session = self._batlinet_session()
        _render(session, checkboxes={ui.SMOKE_CHECKBOX_LABEL: False})
        self.assertIsNone(self._plan_calls[-1]["epochs"])

    def test_gpu_information_is_displayed_not_invented(self) -> None:
        """Runtime facts still reach the user — now in the selector itself.

        The separate inventory table was dropped when the device UX was
        simplified to CPU / one GPU: the option labels already carry index,
        name, free memory and utilization, and a second table repeated them.
        What must remain true is that every figure comes from the enumerated
        device rather than from a constant.
        """
        fake = _render(
            self._batlinet_session(),
            radio_values={ui.COMPUTE_LABEL: "GPU"},
            selectbox_values={ui.GPU_SELECT_LABEL: "cuda:0"},
        )
        boxes = [s for s in fake.rec["selectboxes"] if s["label"] == ui.GPU_SELECT_LABEL]
        self.assertEqual(boxes[0]["options"], ["cuda:0", "cuda:1"])

        inventory = _FAKE_GPU_INVENTORY["devices"]
        for device in inventory:
            label = ui.gpu_option_label(device["device"], _FAKE_GPU_INVENTORY)
            self.assertIn(str(device["name"]), label)
            self.assertIn(f"{device['memory_free_mib'] / 1024:.1f} GB free", label)
            self.assertIn(f"{device['utilization_pct']}% util", label)

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
        # The plain answer is the error line; the detail and the suggested
        # actions follow it. Asserted across everything the page rendered,
        # so the layout can change without the substance being lost.
        errors = " ".join(fake.rec["errors"])
        self.assertIn("GPU memory", errors)
        visible = " ".join(
            fake.rec["errors"] + fake.rec["writes"] + fake.rec["markdown"]
            + fake.rec["captions"]
        )
        self.assertIn("CUDA out of memory on cuda:0", visible)
        self.assertIn("CPU", visible)
        self.assertIn("cuda:1", visible)  # only the GPU with more free memory
        self.assertIn("never moves a failed run", visible)
        # ...and the page says outright that nothing was retried or swapped.
        self.assertIn("Nothing was retried", visible)
        # The raw traceback is still displayed verbatim, behind the details.
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
        self.assertIn("3 / 10", with_counter)
        self.assertIn("01:05", with_counter)
        without = ui.progress_caption("Loading data", None, None, 5.0)
        self.assertIn("Loading data", without)
        self.assertIn("elapsed 00:05", without)
        # No counter, no ETA, no invented countdown.
        self.assertNotIn("/", without)
        self.assertNotIn("%", without)
        self.assertNotIn("ETA", without)

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

    def test_the_page_never_blocks_on_the_executor(self) -> None:
        """The structural guarantee behind a working Cancel button.

        Progress used to be proof that the page consumed events during the
        run. It is now stronger than that: the page does not call the
        blocking executor at all. It hands the plan to the RunManager, which
        owns the thread, and returns — which is the only reason a Cancel
        button drawn afterwards can ever receive a click.
        """
        import ast

        source = Path(ui.__file__).read_text(encoding="utf-8")
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("execute_plan_streaming", called)
        self.assertIn("get_run_manager", called)

    def test_live_progress_is_drawn_from_the_managers_real_state(self) -> None:
        """What is drawn is what the run reported — nothing is invented."""
        snapshot = {
            "run_id": "run_1",
            "status": "RUNNING",
            "finished": False,
            "elapsed_s": 65.0,
            "event": {
                "type": "training_progress",
                "stage": "Training",
                "current": 3,
                "total": 10,
                "fraction": 0.3,
                "message": "Training X:  30%|###| 3/10",
            },
            "tail": ["Training X:  30%|###| 3/10"],
        }
        manager = _FakeRunManager({"run_1": snapshot})
        real_manager = ui.get_run_manager
        ui.get_run_manager = lambda: manager
        try:
            fake = FakeStreamlit({})
            with _fake_streamlit(fake):
                ui._render_active_run("run_1", plan={}, smoke=False, device="cpu")
        finally:
            ui.get_run_manager = real_manager

        captions = " ".join(fake.rec["captions"])
        self.assertIn("Training", captions)
        self.assertIn("3 / 10", captions)
        self.assertIn("01:05", captions)
        # The bar is drawn only because the run reported a fraction.
        self.assertEqual(fake.rec["progress"], [0.3])
        # The refresh is a timed fragment, not a rerun loop.
        self.assertTrue(fake.rec["fragments"])
        self.assertEqual(
            fake.rec["fragments"][0]["run_every"], ui.PROGRESS_POLL_SECONDS
        )

    def test_a_running_run_offers_a_cancel_button_that_reaches_the_manager(self) -> None:
        snapshot = {
            "run_id": "run_1", "status": "RUNNING", "finished": False,
            "elapsed_s": 5.0, "event": {"stage": "Training"}, "tail": [],
        }
        manager = _FakeRunManager({"run_1": snapshot})
        real_manager = ui.get_run_manager
        ui.get_run_manager = lambda: manager
        try:
            fake = FakeStreamlit({}, {ui.CANCEL_BUTTON_LABEL})
            with _fake_streamlit(fake):
                try:
                    ui._render_active_run("run_1", plan={}, smoke=False, device="cpu")
                except _Rerun:
                    pass
        finally:
            ui.get_run_manager = real_manager
        self.assertEqual(manager.cancelled, ["run_1"])

    def test_no_metric_is_ever_read_out_of_the_runs_output(self) -> None:
        """stdout may contain something that looks like a metric. It is not one."""
        captured: list = []
        real_ctx = ui.build_result_context
        real_manager = ui.get_run_manager

        class _Handle:
            run_dir = Path("/tmp/does-not-matter")
            result = {
                "success": True,
                "workspace": "runs/web_projects/alpha/agent_matr_batlinet/run_1",
                "returncode": 0,
                # A number in the output that must never become a metric.
                "stdout": "{'RMSE': 794.32}",
                "stderr": "",
                "command": [],
            }

        class _Manager(_FakeRunManager):
            def get(self, run_id):
                return _Handle()

        manager = _Manager()
        ui.get_run_manager = lambda: manager
        ui.build_result_context = lambda ws, **kw: (
            captured.append(ws) or {"workspace": ws, "success": True,
                                    "metrics": {}, "warnings": []}
        )
        try:
            session = {"agent_execution_plan": _ok_plan("batlinet")}
            fake = FakeStreamlit(session)
            with _fake_streamlit(fake):
                ui._finish_active_run(
                    "run_1", {"status": "COMPLETED"}, smoke=False
                )
        finally:
            ui.build_result_context = real_ctx
            ui.get_run_manager = real_manager

        # Metrics came from the canonical backend reader, given the run's own
        # workspace — never parsed from the text above.
        self.assertEqual(captured, ["runs/web_projects/alpha/agent_matr_batlinet/run_1"])
        self.assertEqual(session["agent_result_context"]["metrics"], {})
        self.assertNotIn("794.32", json.dumps(session["agent_result_context"]))

    def test_a_cancelled_run_never_produces_a_result_context(self) -> None:
        """A killed run has measured nothing, so it offers no metrics."""
        real_ctx = ui.build_result_context
        real_manager = ui.get_run_manager
        built: list = []

        class _Handle:
            run_dir = Path("/tmp/does-not-matter")
            result = {"success": False, "workspace": "w", "returncode": -15,
                      "stdout": "partial", "stderr": "", "command": []}

        class _Manager(_FakeRunManager):
            def get(self, run_id):
                return _Handle()

        ui.get_run_manager = lambda: _Manager()
        ui.build_result_context = lambda ws, **kw: built.append(ws) or {}
        try:
            session = {"agent_execution_plan": _ok_plan("batlinet")}
            fake = FakeStreamlit(session)
            with _fake_streamlit(fake):
                ui._finish_active_run(
                    "run_1", {"status": "CANCELLED"}, smoke=False
                )
        finally:
            ui.build_result_context = real_ctx
            ui.get_run_manager = real_manager

        self.assertEqual(built, [], "no result context for a cancelled run")
        # The key is removed rather than set to None: there is no result to
        # show, so nothing stale can be rendered from it either.
        self.assertNotIn("agent_result_context", session)
        self.assertEqual(session["last_run_status"], "cancelled")


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


class TestRequestAreaIsOneControl(unittest.TestCase):
    """One box, one template helper: no tabs, no gallery, no key hooks."""

    TEMPLATE = "Use CPTransformer on MATR to predict battery lifetime."

    def setUp(self) -> None:
        self._real_resolve = ui.resolve_workflow_intent_with_llm
        self._real_catalog = ui.build_intent_catalog
        self._real_template = ui.build_request_template
        self._real_stream = ui.execute_plan_streaming
        self._real_llm = ui.is_llm_configured
        self.analysed: list[str] = []
        self.executed: list = []

        def stub_resolve(request, **kwargs):
            self.analysed.append(request)
            raise AssertionError("using the template must not analyse")

        def stub_stream(plan, **kwargs):
            self.executed.append(plan)
            raise AssertionError("using the template must not execute")

        ui.resolve_workflow_intent_with_llm = stub_resolve
        ui.execute_plan_streaming = stub_stream
        ui.build_intent_catalog = lambda **kw: _FAKE_CATALOG
        ui.build_request_template = lambda catalog, **kw: self.TEMPLATE
        ui.is_llm_configured = lambda: False

    def tearDown(self) -> None:
        ui.resolve_workflow_intent_with_llm = self._real_resolve
        ui.build_intent_catalog = self._real_catalog
        ui.build_request_template = self._real_template
        ui.execute_plan_streaming = self._real_stream
        ui.is_llm_configured = self._real_llm

    # --- the gallery is gone ------------------------------------------
    def test_no_tabs_are_rendered_for_the_request(self) -> None:
        fake = _render({"gpu_inventory": _FAKE_GPU_INVENTORY})
        for group in fake.rec["tabs"]:
            self.assertNotIn("Examples", group)
            self.assertNotIn("Describe request", group)
        self.assertFalse(hasattr(ui, "REQUEST_TABS"))

    def test_the_example_gallery_is_gone_from_the_source(self) -> None:
        source = Path(ui.__file__).read_text(encoding="utf-8")
        for gone in (
            "REQUEST_TABS",
            "_render_examples",
            "build_example_requests",
            "agent_example_requests",
            "build_request_suggestion",
        ):
            self.assertNotIn(gone, source, f"{gone} still present in the page")

    def test_no_gallery_of_example_buttons_is_rendered(self) -> None:
        fake = _render({"gpu_inventory": _FAKE_GPU_INVENTORY})
        labels = [b["label"] for b in fake.rec["buttons"]]
        # Exactly one suggestion affordance, not a list of requests.
        self.assertEqual(labels.count(ui.USE_TEMPLATE_LABEL), 1)
        for label in labels:
            self.assertNotIn("帮我用", label)

    # --- one suggestion ------------------------------------------------
    def test_the_empty_box_shows_the_generic_template_shape(self) -> None:
        """The placeholder teaches the pattern, not a curated example."""
        fake = _render({"gpu_inventory": _FAKE_GPU_INVENTORY})
        areas = [a for a in fake.rec["text_areas"] if a["label"] == ui.REQUEST_LABEL]
        self.assertEqual(len(areas), 1)
        self.assertEqual(areas[0]["placeholder"], REQUEST_TEMPLATE_STRUCTURE)
        # The slots are visible in the hint; the filled values arrive only
        # when the user asks for them.
        for slot in ("[Model]", "[dataset]", "[task]"):
            self.assertIn(slot, areas[0]["placeholder"])
        self.assertEqual(areas[0]["value"], "")

    def test_the_suggestion_is_computed_once_and_cached(self) -> None:
        calls: list = []

        def counting(catalog, **kwargs):
            calls.append(kwargs)
            return self.TEMPLATE

        ui.build_request_template = counting
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY}
        _render(session)
        _render(session)
        self.assertEqual(len(calls), 1)
        self.assertEqual(session["agent_request_template"], self.TEMPLATE)

    def test_the_page_names_no_model_or_dataset_of_its_own(self) -> None:
        source = Path(ui.__file__).read_text(encoding="utf-8")
        block = source.split("def request_template")[1].split("\ndef ")[0]
        for name in ("CPTransformer", "BatLiNet", "XGBoost", "PBT", "MATR"):
            self.assertNotIn(name, block)

    # --- accepting it ---------------------------------------------------
    def test_accepting_writes_the_canonical_request_state(self) -> None:
        """Two runs, because that is how a keyed widget is re-seeded.

        The click records the intent and reruns; the next run writes the
        widget key *before* the text area is built. A browser check of the
        first attempt — which popped the key and leaned on ``value=`` —
        showed the box still empty, so this asserts the text really lands in
        the widget, not only in the canonical state.
        """
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY}
        _render(session, clicked={ui.USE_TEMPLATE_LABEL})
        self.assertTrue(session[ui.USE_TEMPLATE_KEY])

        fake = _render(session)
        # The same key manual typing writes — one state, not three.
        self.assertEqual(session[ui.REQUEST_STATE_KEY], self.TEMPLATE)
        self.assertEqual(ui.REQUEST_STATE_KEY, "agent_original_request")
        # ...and the box itself now holds it.
        self.assertEqual(session[ui.REQUEST_WIDGET_KEY], self.TEMPLATE)
        areas = [a for a in fake.rec["text_areas"] if a["label"] == ui.REQUEST_LABEL]
        self.assertEqual(areas[0]["value"], self.TEMPLATE)
        # The one-shot flag is consumed, so a later rerun cannot re-accept
        # over text the user has since typed.
        self.assertNotIn(ui.USE_TEMPLATE_KEY, session)

    def test_accepting_does_not_analyse_call_the_llm_or_execute(self) -> None:
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY}
        _render(session, clicked={ui.USE_TEMPLATE_LABEL})
        _render(session)
        self.assertEqual(self.analysed, [])
        self.assertEqual(self.executed, [])
        self.assertNotIn("agent_workflow_context", session)
        self.assertNotIn("agent_execution_plan", session)

    def test_typed_text_is_never_overwritten_by_the_suggestion(self) -> None:
        typed = "Run BatLiNet on HUST"
        session = {
            "gpu_inventory": _FAKE_GPU_INVENTORY,
            ui.REQUEST_STATE_KEY: typed,
            "agent_request_input": typed,
        }
        fake = _render(session, clicked={ui.USE_TEMPLATE_LABEL})
        # The accept control is not even offered once the field has content.
        labels = [b["label"] for b in fake.rec["buttons"]]
        self.assertNotIn(ui.USE_TEMPLATE_LABEL, labels)
        self.assertEqual(session[ui.REQUEST_STATE_KEY], typed)

    def test_text_typed_before_the_click_lands_is_not_overwritten(self) -> None:
        """The race a browser check exposed.

        Streamlit commits a text area on blur, so the accept control can
        still be on screen while the user types. Clicking it blurs the box,
        which commits their text on the very run that applies the
        acceptance. Their text must win.
        """
        typed = "Run BatLiNet on MATR"
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY}
        _render(session, clicked={ui.USE_TEMPLATE_LABEL})
        self.assertTrue(session[ui.USE_TEMPLATE_KEY])
        # The blur commits what they typed, then the accepting run happens.
        session[ui.REQUEST_WIDGET_KEY] = typed
        _render(session)
        self.assertEqual(session[ui.REQUEST_WIDGET_KEY], typed)
        self.assertNotEqual(session.get(ui.REQUEST_STATE_KEY), self.TEMPLATE)

    def test_the_accept_rule_is_pure_and_refuses_a_non_empty_field(self) -> None:
        self.assertTrue(ui.accept_template(self.TEMPLATE, ""))
        self.assertTrue(ui.accept_template(self.TEMPLATE, "   "))
        self.assertFalse(ui.accept_template(self.TEMPLATE, "typed"))
        self.assertFalse(ui.accept_template("", ""))

    def test_analyze_remains_the_explicit_action(self) -> None:
        fake = _render({"gpu_inventory": _FAKE_GPU_INVENTORY})
        self.assertIn("Analyze", [b["label"] for b in fake.rec["buttons"]])
        self.assertEqual(self.analysed, [])

    def test_no_keyboard_interception_is_claimed_or_injected(self) -> None:
        """Section 4: do not fake Tab completion.

        The page must not inject a key handler, and must not tell the user
        that Tab does something it does not do.
        """
        source = Path(ui.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "keydown",
            "addEventListener",
            "preventDefault",
            "window.parent",
            "unsafe_allow_html",
            "declare_component",
            "_bidi_component",
        ):
            self.assertNotIn(forbidden, source)
        rendered = repr(_render({"gpu_inventory": _FAKE_GPU_INVENTORY}).rec)
        self.assertNotIn("Press Tab", rendered)
        self.assertNotIn("press Tab", rendered)

    def test_the_box_never_combines_a_default_with_a_state_write(self) -> None:
        """No Streamlit Session State warning on the accepting run.

        Streamlit 1.64 logs "The widget with key ... was created with a
        default value but also had its value set via the Session State API"
        (``check_session_state_rules``) when ``value=`` is non-empty on a run
        that also wrote the key. Accepting the template writes the key, so
        the box must take its content from the key alone — on every run.
        """
        session = {"gpu_inventory": _FAKE_GPU_INVENTORY}
        runs = [
            _render(session, clicked={ui.USE_TEMPLATE_LABEL}),
            _render(session),  # the accepting run: the key is written here
            _render(session),
        ]
        for fake in runs:
            for area in fake.rec["text_areas"]:
                if area["label"] == ui.REQUEST_LABEL:
                    self.assertEqual(area["default"], "")
        # ...and the content still arrives, through the key.
        self.assertEqual(runs[1].rec["text_areas"][0]["value"], self.TEMPLATE)

    def test_a_request_handed_over_from_home_still_fills_the_box(self) -> None:
        """Home pops the widget key; the box must then show the new request."""
        session = {
            "gpu_inventory": _FAKE_GPU_INVENTORY,
            ui.REQUEST_STATE_KEY: "Use XGBoost on MATR to predict RUL.",
        }
        fake = _render(session)
        areas = [a for a in fake.rec["text_areas"] if a["label"] == ui.REQUEST_LABEL]
        self.assertEqual(areas[0]["value"], "Use XGBoost on MATR to predict RUL.")
        self.assertEqual(areas[0]["default"], "")


class TestStructuredProgressIsThePrimaryView(unittest.TestCase):
    """Milestone B: one structured state, and the raw log one click away.

    The failure this replaces: the run page's main area was a code box
    filling with hundreds of identical ``Running.`` lines — the heartbeat,
    logged as if it were output — above a caption that said "Loading data"
    for every one of the four data stages.
    """

    def _snapshot(self, event: dict, tail: list[str] | None = None,
                  elapsed: float = 5.0) -> dict:
        return {
            "run_id": "run_1",
            "status": "RUNNING",
            "finished": False,
            "elapsed_s": elapsed,
            "event": event,
            "tail": list(tail or []),
        }

    def _render_active(self, snapshot: dict) -> FakeStreamlit:
        manager = _FakeRunManager({"run_1": snapshot})
        real_manager = ui.get_run_manager
        ui.get_run_manager = lambda: manager
        try:
            fake = FakeStreamlit({})
            with _fake_streamlit(fake):
                ui._render_active_run("run_1", plan={}, smoke=False, device="cpu")
            return fake
        finally:
            ui.get_run_manager = real_manager

    def test_a_real_counter_draws_a_bar_at_the_parsed_fraction(self) -> None:
        fake = self._render_active(self._snapshot({
            "type": "stage",
            "stage": "Extracting train samples",
            "current": 20, "total": 41, "fraction": 20 / 41,
            "heartbeat": False, "indeterminate": False,
            "message": "Extracting train cycle samples:  49%|####| 20/41",
        }))
        self.assertEqual(len(fake.rec["progress"]), 1)
        self.assertAlmostEqual(fake.rec["progress"][0], 20 / 41)
        captions = " ".join(fake.rec["captions"])
        self.assertIn("Extracting train samples", captions)
        self.assertIn("20 / 41", captions)
        # A bar, not a spinner: the denominator is real.
        self.assertEqual(fake.rec["statuses"], [])

    def test_an_indeterminate_stage_gets_a_spinner_and_no_percentage(self) -> None:
        fake = self._render_active(self._snapshot({
            "type": "stage", "stage": "Evaluating",
            "current": None, "total": None, "fraction": None,
            "heartbeat": False, "indeterminate": True,
            "message": "Evaluating",
        }, elapsed=4.0))
        self.assertEqual(
            fake.rec["progress"], [],
            "a bar was drawn for a stage with no denominator",
        )
        self.assertTrue(fake.rec["statuses"])
        label, state = fake.rec["statuses"][0]
        self.assertIn("Evaluating", label)
        self.assertIn("elapsed 00:04", label)
        self.assertEqual(state, "running")
        self.assertNotIn("%", label)
        self.assertNotIn("ETA", label)

    def test_repeated_running_lines_are_not_the_primary_display(self) -> None:
        """The regression test for the ``Running.`` spam.

        Even if a tail somehow still contained heartbeat lines, they may only
        appear inside the technical-log disclosure — never as the page's own
        status, and never as a bare code block in the main area.
        """
        spam = [ui.HEARTBEAT_MESSAGE] * 40
        fake = self._render_active(self._snapshot(
            {
                "type": "stage", "stage": "Training",
                "current": None, "total": None, "fraction": None,
                "heartbeat": True, "indeterminate": False,
                "message": ui.HEARTBEAT_MESSAGE,
            },
            tail=spam,
        ))
        status_labels = " ".join(label for label, _ in fake.rec["statuses"])
        self.assertNotIn(ui.HEARTBEAT_MESSAGE, status_labels)
        self.assertNotIn(ui.HEARTBEAT_MESSAGE, " ".join(fake.rec["captions"]))
        # The raw text is behind the disclosure, and only there.
        self.assertIn(ui.TECHNICAL_LOG_LABEL, fake.rec["expanders"])
        outside = [
            text for text in fake.rec["codes"]
            if ui.HEARTBEAT_MESSAGE in text
        ]
        self.assertEqual(
            outside, [],
            "heartbeat text was drawn as a code block outside the disclosure",
        )

    def test_the_raw_log_is_still_reachable(self) -> None:
        """Hidden by default, removed never."""
        fake = self._render_active(self._snapshot(
            {
                "type": "stage", "stage": "Training",
                "current": None, "total": None, "fraction": None,
                "heartbeat": False, "indeterminate": False,
                "message": "Traning:   4%|#| 4/100",
            },
            tail=["Seed is set to 0.", "Traning:   4%|#| 4/100"],
        ))
        self.assertIn(ui.TECHNICAL_LOG_LABEL, fake.rec["expanders"])
        self.assertTrue(
            any("Traning" in text for text in fake.rec["expander_writes"] + fake.rec["codes"]),
            "the run's real output is no longer reachable at all",
        )

    def test_the_run_manager_never_logs_a_heartbeat(self) -> None:
        """Cut off at the source, not filtered at the display.

        The tail and run.log are what a failed run is diagnosed from; a
        heartbeat in either is noise that pushes real output out of the
        window the tail keeps.
        """
        from batteryml_web.core import run_manager as rm_module

        source = Path(rm_module.__file__).read_text(encoding="utf-8")
        self.assertIn('event.get("heartbeat")', source)

    def test_a_heartbeat_is_recognised_by_its_flag_and_by_its_text(self) -> None:
        """The flag is authoritative; the text is the compatibility fallback.

        A RunManager snapshot can outlive the code that wrote it — a run in
        flight across a restart — so an event with no flag but the heartbeat
        message must still not be redrawn as a real stage change.
        """
        self.assertTrue(ui.is_heartbeat({"heartbeat": True, "message": "anything"}))
        self.assertTrue(ui.is_heartbeat({"message": ui.HEARTBEAT_MESSAGE}))
        self.assertFalse(ui.is_heartbeat({"message": "Reading train data: 20/41"}))
        self.assertFalse(ui.is_heartbeat({}))

    def test_an_unflagged_heartbeat_does_not_reach_the_live_log(self) -> None:
        fake = FakeStreamlit({})
        progress = ui._LiveProgress(fake.empty(), fake.empty(), fake.empty())
        progress.on_event({
            "type": "output", "stage": "Training", "current": None,
            "total": None, "fraction": None, "message": ui.HEARTBEAT_MESSAGE,
        })
        drawn = fake.rec["codes"] + fake.rec["expander_writes"]
        self.assertEqual(
            [text for text in drawn if ui.HEARTBEAT_MESSAGE in text], [],
            "an unflagged heartbeat was appended to the live log",
        )

    def test_no_percentage_is_drawn_without_a_counter(self) -> None:
        """A fraction with no counter behind it is refused, not displayed."""
        timer = ui.EpochTimer()
        view = ui.progress_view(
            {"stage": "Evaluating", "current": None, "total": None,
             "fraction": 0.7, "message": "Evaluating"},
            elapsed_s=3.0,
            timer=timer,
        )
        self.assertIsNone(view["fraction"])
        self.assertFalse(view["determinate"])
        self.assertNotIn("%", view["caption"])

    def test_a_true_fraction_is_never_read_as_one_hundred_percent(self) -> None:
        timer = ui.EpochTimer()
        view = ui.progress_view(
            {"stage": "Training", "current": True, "total": True,
             "fraction": True, "message": "x"},
            elapsed_s=1.0,
            timer=timer,
        )
        self.assertIsNone(view["fraction"])
        self.assertIsNone(view["counter"])

    def test_an_eta_appears_only_after_enough_of_a_counter(self) -> None:
        timer = ui.EpochTimer()
        first = ui.progress_view(
            {"stage": "Reading train data", "current": 1, "total": 41,
             "fraction": 1 / 41, "message": "x"}, 1.0, timer)
        self.assertIn("estimating", first["caption"])
        self.assertNotIn("ETA", first["caption"])
        for step in range(2, 6):
            view = ui.progress_view(
                {"stage": "Reading train data", "current": step, "total": 41,
                 "fraction": step / 41, "message": "x"}, float(step), timer)
        self.assertIn("ETA", view["caption"])
        self.assertNotIn("estimating", view["caption"])

    def test_a_new_stage_does_not_inherit_the_previous_stages_rate(self) -> None:
        """Reading 41 cells and extracting from 41 cells are different work."""
        timer = ui.EpochTimer()
        for step in range(1, 12):
            ui.progress_view(
                {"stage": "Reading train data", "current": step, "total": 41,
                 "fraction": step / 41, "message": "x"}, float(step), timer)
        moved = ui.progress_view(
            {"stage": "Extracting train samples", "current": 1, "total": 41,
             "fraction": 1 / 41, "message": "x"}, 12.0, timer)
        self.assertIn(
            "estimating", moved["caption"],
            "the new stage inherited an ETA measured on the previous one",
        )


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
        for module in (ui, ui_feedback, ui_llm_state, ui_settings):
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


def _inventory_with(count: int) -> dict:
    """An inventory of exactly ``count`` GPUs, built like the real one."""
    csv = "".join(
        f"{i}, NVIDIA GeForce RTX 3090, 24576, 22500, 2076, {40 + i}\n"
        for i in range(count)
    )
    devices = di.parse_nvidia_smi_gpu_csv(csv)
    return {
        "source": "nvidia-smi",
        "available": bool(devices),
        "device_count": len(devices),
        "devices": devices,
        "error": None if devices else "nvidia-smi reported no GPU.",
    }


class TestComputeDeviceSelector(unittest.TestCase):
    """The product supports CPU and one GPU. Nothing else is offered.

    Multi-GPU selection was removed from the normal UI: no model can execute
    a distributed run (docs/DDP_CONTRACT.md), so offering the choice put an
    error box in front of anyone who tried it. The DDP infrastructure and its
    tests remain; only the user-facing offer is gone.
    """

    def setUp(self) -> None:
        self._real_build = ui.build_execution_plan
        self._real_stream = ui.execute_plan_streaming
        self._real_llm = ui.is_llm_configured
        self._real_manager = ui.get_run_manager
        self._real_write = ui.write_manifest
        self.plans: list[dict] = []
        ui.build_execution_plan = lambda d, t_, m, **kw: (
            self.plans.append({"model": m, **kw})
            or _ok_plan(m, epochs=kw.get("epochs"), workspace=kw["workspace"])
        )
        ui.execute_plan_streaming = lambda plan, **kw: None
        ui.is_llm_configured = lambda: False
        ui.get_run_manager = lambda: _FakeRunManager()
        ui.write_manifest = lambda run_dir, manifest: None

    def tearDown(self) -> None:
        ui.build_execution_plan = self._real_build
        ui.execute_plan_streaming = self._real_stream
        ui.is_llm_configured = self._real_llm
        ui.get_run_manager = self._real_manager
        ui.write_manifest = self._real_write

    def _session(self, inventory: dict) -> dict:
        return {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_selected_model": "batlinet",
            "gpu_inventory": inventory,
            "project_root": "runs/web_projects/alpha",
        }

    def test_the_only_compute_choices_are_cpu_and_gpu(self) -> None:
        fake = _render(self._session(_inventory_with(2)))
        modes = [b for b in fake.rec["radios"] if b["label"] == ui.COMPUTE_LABEL]
        self.assertEqual(modes[0]["options"], ["CPU", "GPU"])

    def test_cpu_needs_no_gpu_interaction(self) -> None:
        session = self._session(_inventory_with(2))
        fake = _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        self.assertEqual(
            [s for s in fake.rec["selectboxes"] if s["label"] == ui.GPU_SELECT_LABEL], []
        )
        self.assertEqual(session["agent_device_selection"], {"mode": "cpu", "gpu_ids": []})
        self.assertEqual(self.plans[-1]["device"], "cpu")

    def test_gpu_mode_offers_exactly_the_discovered_gpus(self) -> None:
        for count in (1, 4, 8):
            with self.subTest(gpus=count):
                self.plans.clear()
                fake = _render(
                    self._session(_inventory_with(count)),
                    radio_values={ui.COMPUTE_LABEL: "GPU"},
                )
                boxes = [
                    s for s in fake.rec["selectboxes"] if s["label"] == ui.GPU_SELECT_LABEL
                ]
                self.assertEqual(len(boxes), 1, "exactly one GPU control")
                self.assertEqual(boxes[0]["options"], [f"cuda:{i}" for i in range(count)])

    def test_no_multi_gpu_widget_is_rendered_at_all(self) -> None:
        fake = _render(
            self._session(_inventory_with(4)), radio_values={ui.COMPUTE_LABEL: "GPU"}
        )
        self.assertEqual(fake.rec["multiselects"], [])

    def test_exactly_one_gpu_is_selected_and_becomes_cuda_n(self) -> None:
        session = self._session(_inventory_with(4))
        _render(
            session,
            radio_values={ui.COMPUTE_LABEL: "GPU"},
            selectbox_values={ui.GPU_SELECT_LABEL: "cuda:2"},
        )
        self.assertEqual(
            session["agent_device_selection"], {"mode": "single_gpu", "gpu_ids": [2]}
        )
        self.assertEqual(self.plans[-1]["device"], "cuda:2")

    def test_no_multi_device_string_can_be_built(self) -> None:
        session = self._session(_inventory_with(8))
        _render(
            session,
            radio_values={ui.COMPUTE_LABEL: "GPU"},
            selectbox_values={ui.GPU_SELECT_LABEL: "cuda:3"},
        )
        blob = repr(session) + repr(self.plans)
        self.assertNotIn("cuda:0,1", blob)
        self.assertNotIn(",cuda:", blob)

    def test_no_ddp_warning_reaches_an_ordinary_user(self) -> None:
        """§9: the distributed internals belong in docs, not on the page."""
        fake = _render(
            self._session(_inventory_with(4)),
            radio_values={ui.COMPUTE_LABEL: "GPU"},
            selectbox_values={ui.GPU_SELECT_LABEL: "cuda:1"},
        )
        rendered = repr(fake.rec)
        for leak in (
            "multi-GPU", "multi_gpu", "DDP_CONTRACT", "supports_multi_gpu",
            "rank", "sharding", "torchrun",
        ):
            self.assertNotIn(leak, rendered)

    def test_a_machine_with_no_gpu_defaults_to_cpu(self) -> None:
        session = self._session(di.empty_inventory("nvidia-smi was not found"))
        _render(session)
        self.assertEqual(session["agent_device_selection"], {"mode": "cpu", "gpu_ids": []})
        self.assertEqual(self.plans[-1]["device"], "cpu")

    def test_gpu_labels_come_from_the_runtime_inventory(self) -> None:
        inventory = _inventory_with(2)
        label = ui.gpu_option_label("cuda:1", inventory)
        self.assertTrue(label.startswith("cuda:1"))
        self.assertIn(str(inventory["devices"][1]["name"]), label)
        self.assertIn("free", label)
        self.assertIn("util", label)

    def test_a_healthy_gpu_gets_a_caption_not_a_warning_box(self) -> None:
        """§10: a full-width warning for an idle GPU trains people to ignore them."""
        # _FAKE_GPU_INVENTORY's cuda:1 reports ~10 GB free of 24 GB.
        session = dict(self._session(_FAKE_GPU_INVENTORY))
        fake = _render(
            session,
            radio_values={ui.COMPUTE_LABEL: "GPU"},
            selectbox_values={ui.GPU_SELECT_LABEL: "cuda:1"},
        )
        captions = " ".join(fake.rec["captions"])
        self.assertIn("cuda:1", captions)
        self.assertIn("GB free", captions)
        self.assertIn("availability may change", captions)
        self.assertNotIn("cuda:1", " ".join(fake.rec["warnings"]))

    def test_a_nearly_full_gpu_does_warn(self) -> None:
        # ...while its cuda:0 reports ~2 GB free of 24 GB, which does.
        fake = _render(
            dict(self._session(_FAKE_GPU_INVENTORY)),
            radio_values={ui.COMPUTE_LABEL: "GPU"},
            selectbox_values={ui.GPU_SELECT_LABEL: "cuda:0"},
        )
        self.assertIn("cuda:0", " ".join(fake.rec["warnings"]))
        self.assertNotIn("cuda:0", " ".join(fake.rec["captions"]))


class TestProjectDeterminesTheWorkspaceInTheUi(unittest.TestCase):
    """The page's project is the project the plan is built for."""

    def setUp(self) -> None:
        self._real_build = ui.build_execution_plan
        self._real_llm = ui.is_llm_configured
        self.plans: list[dict] = []
        ui.build_execution_plan = lambda d, t_, m, **kw: (
            self.plans.append({"model": m, **kw})
            or _ok_plan(m, epochs=kw.get("epochs"), workspace=kw["workspace"])
        )
        ui.is_llm_configured = lambda: False

    def tearDown(self) -> None:
        ui.build_execution_plan = self._real_build
        ui.is_llm_configured = self._real_llm

    def _render_for(self, project_root: str | None) -> dict:
        session = {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_selected_model": "batlinet",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
        }
        if project_root is not None:
            session["project_root"] = project_root
        _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        return session

    def test_the_loaded_project_owns_the_workspace(self) -> None:
        """Still the project's directory — now with a run of its own inside.

        The plan points at ``<project>/agent_<dataset>_<model>/<run_id>``, so
        two runs of the same model cannot overwrite each other, and the plan
        the user reviews is the plan that executes.
        """
        self._render_for("runs/web_projects/alpha")
        workspace = self.plans[-1]["workspace"]
        base = "runs/web_projects/alpha/agent_matr_batlinet"
        self.assertTrue(workspace.startswith(base + "/"), workspace)
        self.assertTrue(is_run_id(workspace.rsplit("/", 1)[-1]), workspace)

    def test_switching_project_switches_the_workspace(self) -> None:
        self._render_for("runs/web_projects/alpha")
        alpha = self.plans[-1]["workspace"]
        self._render_for("runs/web_projects/beta")
        beta = self.plans[-1]["workspace"]
        self.assertNotEqual(alpha, beta)
        self.assertIn("/alpha/", alpha)
        self.assertIn("/beta/", beta)

    def test_the_run_context_names_the_same_project_and_directory(self) -> None:
        session = self._render_for("runs/web_projects/gamma")
        run_context = session["agent_run_context"]
        self.assertEqual(run_context["project"], "gamma")
        self.assertEqual(run_context["workspace"], self.plans[-1]["workspace"])

    def test_no_project_is_reported_as_the_default_not_hidden(self) -> None:
        session = self._render_for(None)
        self.assertTrue(session["agent_run_context"]["project_is_default"])
        self.assertIn("default", self.plans[-1]["workspace"])

    def test_run_context_is_shown_before_execution(self) -> None:
        session = {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_selected_model": "batlinet",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
            "project_root": "runs/web_projects/alpha",
        }
        fake = _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        # The provenance block is the frame carrying "Project"; the execution
        # plan renders its own table, and merging the two would let one
        # table's "Device" answer for the other's.
        frames = [
            f
            for f in fake.rec["dataframes"]
            if isinstance(f, list)
            and any(isinstance(r, dict) and r.get("Field") == "Project" for r in f)
        ]
        self.assertEqual(len(frames), 1, "expected exactly one run-context table")
        fields = {r["Field"]: r["Value"] for r in frames[0]}
        for required in ("Project", "Run directory", "Dataset", "Task", "Model",
                         "Config", "Seed", "Epochs", "Device"):
            self.assertIn(required, fields)
        self.assertEqual(fields["Project"], "alpha")
        # The readable label, not the argv value the plan table already shows.
        self.assertEqual(fields["Device"], "CPU")
        self.assertTrue(
            fields["Run directory"].startswith(
                "runs/web_projects/alpha/agent_matr_batlinet/"
            ),
            fields["Run directory"],
        )
        # The run this page is about to create is named before it starts.
        self.assertTrue(is_run_id(fields["Run ID"]), fields["Run ID"])
        self.assertTrue(fields["Run directory"].endswith(fields["Run ID"]))


class TestExplainOpensConfigurationInsteadOfDeadEnding(unittest.TestCase):
    """The Explain button is always actionable."""

    def setUp(self) -> None:
        self._real_explain = ui.explain_run_result
        self._real_llm = ui.is_llm_configured
        self.explain_calls: list = []
        ui.explain_run_result = lambda ctx, **kw: (
            self.explain_calls.append(kw) or {"summary": "explained"}
        )

    def tearDown(self) -> None:
        ui.explain_run_result = self._real_explain
        ui.is_llm_configured = self._real_llm

    def _session(self) -> dict:
        return {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_selected_model": "batlinet",
            "gpu_inventory": _FAKE_GPU_INVENTORY,
            "project_root": "runs/web_projects/alpha",
            "agent_result_context": {
                "workspace": "runs/web_projects/alpha/agent_matr_batlinet",
                "success": True,
                "metrics": {"RMSE": 1.0, "MAE": 2.0, "MAPE": 0.25},
                "warnings": [],
                "predictions_file": "runs/web_projects/alpha/agent_matr_batlinet/p.pkl",
            },
        }

    def test_the_button_is_never_disabled(self) -> None:
        ui.is_llm_configured = lambda: False
        fake = _render(self._session(), radio_values={ui.COMPUTE_LABEL: "CPU"})
        buttons = [
            b for b in fake.rec["buttons"] if b["label"] == ui.EXPLAIN_BUTTON_LABEL
        ]
        self.assertTrue(buttons)
        self.assertFalse(any(b["disabled"] for b in buttons))

    def test_clicking_without_configuration_points_at_settings_and_calls_nothing(
        self,
    ) -> None:
        """The click is never a dead end, and never a second settings screen.

        It used to open the full provider / Base URL / API key / model form
        in a dialog here. That form wrote the same store as the sidebar's,
        so nothing ever diverged — but being asked for an API key on the
        page you came to read a result on is what made the app feel like it
        had forgotten, which is the complaint this replaces. One sentence
        naming where the setting lives, and still no call.
        """
        ui.is_llm_configured = lambda: False
        session = self._session()
        fake = _render(
            session,
            clicked={ui.EXPLAIN_BUTTON_LABEL},
            radio_values={ui.COMPUTE_LABEL: "CPU"},
        )
        self.assertEqual(self.explain_calls, [])
        self.assertTrue(session.get(ui.NEEDS_CONFIGURATION_KEY))
        # The click reruns, so the notice is drawn on the following run.
        fake = _render(
            session, radio_values={ui.COMPUTE_LABEL: "CPU"}
        )
        self.assertIn(
            ui_settings.CONFIGURE_IN_SETTINGS_MESSAGE, fake.rec["captions"]
        )

    def test_clicking_with_configuration_explains_directly(self) -> None:
        ui.is_llm_configured = lambda: True
        session = self._session()
        session.update(
            {"llm_base_url": "u", "llm_api_key": "k", "llm_model": "m"}
        )
        _render(
            session,
            clicked={ui.EXPLAIN_BUTTON_LABEL},
            radio_values={ui.COMPUTE_LABEL: "CPU"},
        )
        self.assertEqual(len(self.explain_calls), 1)

    def test_the_api_key_never_reaches_the_rendered_page(self) -> None:
        ui.is_llm_configured = lambda: True
        session = self._session()
        secret = "sk-do-not-leak-me"
        session.update({"llm_base_url": "u", "llm_api_key": secret, "llm_model": "m"})
        fake = _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        rendered = repr(fake.rec)
        self.assertNotIn(secret, rendered)

    def test_the_settings_summary_never_carries_the_key(self) -> None:
        fake = FakeStreamlit({"llm_base_url": "u", "llm_api_key": "sk-secret", "llm_model": "m"})
        with _fake_streamlit(fake):
            summary = ui_settings.llm_settings_summary()
        self.assertTrue(summary["api_key_set"])
        self.assertNotIn("sk-secret", repr(summary))

    # --- Save and Connect -------------------------------------------
    def test_the_form_offers_save_and_connect_and_no_cancel_pair(self) -> None:
        fake = FakeStreamlit({})
        with _fake_streamlit(fake):
            ui_settings.render_llm_settings_form(key_prefix="t")
        labels = [b["label"] for b in fake.rec["buttons"]]
        self.assertIn(ui_settings.SAVE_LABEL, labels)
        self.assertIn(ui_settings.CONNECT_LABEL, labels)
        self.assertEqual(labels, ["Save", "Connect"])
        # The old primary pair must not survive anywhere.
        self.assertNotIn("Cancel", labels)
        self.assertNotIn("Save settings", labels)
        # ...and the old pair is gone from the code too. Scanned with
        # docstrings and comments removed: the module explains in prose why
        # "Cancel / Save settings" was replaced, and that prose is
        # documentation, not a button.
        code = _code_without_prose(ui_settings)
        self.assertNotIn("Save settings", code)
        self.assertNotIn("Cancel", code)

    def test_save_stores_settings_without_testing_the_connection(self) -> None:
        calls: list = []
        real = ui_settings.test_llm_connection
        ui_settings.test_llm_connection = lambda **kw: calls.append(kw) or (True, "ok")
        try:
            fake = FakeStreamlit({}, clicked_buttons={ui_settings.SAVE_LABEL})
            fake.session_state.update(
                {"t_base_url": "u", "t_api_key": "k", "t_model": "m"}
            )
            with _fake_streamlit(fake):
                action = ui_settings.render_llm_settings_form(key_prefix="t")
        finally:
            ui_settings.test_llm_connection = real
        self.assertEqual(action, "save")
        self.assertEqual(calls, [], "Save must not call the connection test")
        self.assertEqual(fake.session_state["llm_base_url"], "u")
        self.assertEqual(fake.session_state["llm_model"], "m")
        self.assertIn(ui_settings.SAVED_MESSAGE, " ".join(fake.rec["toasts"]))

    def test_connect_invokes_the_existing_connection_test(self) -> None:
        calls: list = []
        real = ui_settings.test_llm_connection
        ui_settings.test_llm_connection = lambda **kw: (
            calls.append(kw) or (True, "LLM connection verified.")
        )
        try:
            fake = FakeStreamlit({}, clicked_buttons={ui_settings.CONNECT_LABEL})
            fake.session_state.update(
                {"t_base_url": "u", "t_api_key": "k", "t_model": "m"}
            )
            with _fake_streamlit(fake):
                action = ui_settings.render_llm_settings_form(key_prefix="t")
        finally:
            ui_settings.test_llm_connection = real
        self.assertEqual(action, "connect")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            fake.session_state["llm_status"], ui_llm_state.LLM_STATUS_CONNECTED
        )
        self.assertIn(ui_settings.CONNECTED_MESSAGE, " ".join(fake.rec["toasts"]))

    def test_connection_testing_is_not_reimplemented(self) -> None:
        """One connection implementation, imported from the agent client."""
        source = Path(ui_settings.__file__).read_text(encoding="utf-8")
        self.assertIn("from ..agent.llm_client import test_llm_connection", source)
        for forbidden in ("requests.post", "urlopen", "httpx", "chat/completions"):
            self.assertNotIn(forbidden, source)
        app_source = (Path(ui.__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("test_llm_connection(base_url=", app_source)

    def test_a_failed_connect_is_persistent_and_leaks_no_key(self) -> None:
        secret = "sk-never-show-me"
        real = ui_settings.test_llm_connection
        ui_settings.test_llm_connection = lambda **kw: (
            False,
            "LLM connection failed: Connection refused",
        )
        try:
            fake = FakeStreamlit({}, clicked_buttons={ui_settings.CONNECT_LABEL})
            fake.session_state.update(
                {"t_base_url": "u", "t_api_key": secret, "t_model": "m"}
            )
            with _fake_streamlit(fake):
                action = ui_settings.render_llm_settings_form(key_prefix="t")
        finally:
            ui_settings.test_llm_connection = real
        self.assertEqual(action, "connect_failed")
        self.assertEqual(
            fake.session_state["llm_status"], ui_llm_state.LLM_STATUS_FAILED
        )
        # Persistent, not a toast that vanishes before it can be acted on.
        self.assertTrue(fake.rec["errors"])
        self.assertEqual(fake.rec["toasts"], [])
        self.assertNotIn(secret, repr(fake.rec))

    def test_connect_with_a_missing_field_never_reaches_the_network(self) -> None:
        calls: list = []
        real = ui_settings.test_llm_connection
        ui_settings.test_llm_connection = lambda **kw: calls.append(kw) or (True, "ok")
        try:
            fake = FakeStreamlit({}, clicked_buttons={ui_settings.CONNECT_LABEL})
            fake.session_state.update({"t_base_url": "u", "t_api_key": "", "t_model": "m"})
            with _fake_streamlit(fake):
                ui_settings.render_llm_settings_form(key_prefix="t")
        finally:
            ui_settings.test_llm_connection = real
        self.assertEqual(calls, [])
        self.assertIn(ui_settings.INCOMPLETE_MESSAGE, " ".join(fake.rec["errors"]))

    def test_the_signature_used_for_toasts_never_carries_the_key(self) -> None:
        fake = FakeStreamlit(
            {"llm_base_url": "u", "llm_api_key": "sk-secret", "llm_model": "m"}
        )
        with _fake_streamlit(fake):
            signature = ui_settings.llm_settings_signature()
        self.assertNotIn("sk-secret", signature)
        self.assertIn("key", signature)

    def test_the_agent_page_renders_no_credential_fields(self) -> None:
        """Whatever the state, this page never asks for the credentials.

        Asserted against what was actually drawn rather than against the
        source, because the form used to arrive here indirectly — the page
        called a helper, which opened a dialog, which rendered it. A check
        on the rendered widgets cannot be satisfied by moving the call.
        """
        ui.is_llm_configured = lambda: False
        session = self._session()
        session[ui.NEEDS_CONFIGURATION_KEY] = True
        fake = _render(
            session,
            clicked={ui.EXPLAIN_BUTTON_LABEL},
            radio_values={ui.COMPUTE_LABEL: "CPU"},
        )
        labels = [
            str(entry.get("label", ""))
            for entry in fake.rec.get("text_inputs", [])
        ]
        for field in ui_settings.FIELD_LABELS.values():
            self.assertNotIn(field, labels)
        button_labels = [str(b["label"]) for b in fake.rec["buttons"]]
        self.assertNotIn(ui_settings.SAVE_LABEL, button_labels)
        self.assertNotIn(ui_settings.CONNECT_LABEL, button_labels)

    def test_one_configuration_source_backs_both_entry_points(self) -> None:
        source = Path(ui_settings.__file__).read_text(encoding="utf-8")
        # Both the sidebar and the dialog go through the same form, which
        # writes the existing llm_state keys.
        self.assertIn("render_llm_settings_form", source)
        self.assertIn("set_llm_settings", source)
        self.assertNotIn("st.secrets", source)
        for key in ("llm_base_url", "llm_api_key", "llm_model"):
            self.assertIn(key, source)


class TestNothingRuntimeIsHardcoded(unittest.TestCase):
    """Designed to fail if someone later writes a runtime value into the UI."""

    UI_SOURCES = ("agent_workflow.py", "workflow_sidebar.py", "settings_panel.py",
                  "formatting.py", "feedback.py")

    def _ui_text(self) -> str:
        """The UI layer's *code*, with docstrings removed.

        Scanned via the AST rather than as raw text, for the same reason the
        other static scans in this suite are: a docstring is documentation,
        not behaviour. ``formatting.py`` explains itself with a worked
        example of a long metric, and a raw-text scan read that prose as a
        hardcoded metric value. Stripping docstrings keeps the scan pointed
        at code, where a hardcoded runtime value would actually do harm.

        Non-vacuity: temporarily writing ``RMSE = 455.58`` into any of these
        modules makes both value scans below fail.
        """
        import ast

        base = Path(ui.__file__).parent
        chunks: list[str] = []
        for name in self.UI_SOURCES:
            tree = ast.parse((base / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    body = getattr(node, "body", [])
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        node.body = body[1:] or [ast.Pass()]
            chunks.append(ast.unparse(tree))
        return "\n".join(chunks)

    def test_no_gpu_hardware_is_named(self) -> None:
        text = self._ui_text()
        for forbidden in ("RTX 3090", "Tesla", "A100", "24576", "cuda:1", "cuda:7"):
            self.assertNotIn(forbidden, text)

    def test_no_fixed_gpu_count(self) -> None:
        text = self._ui_text()
        for forbidden in ("range(8)", "range(4)", "8 GPUs", "GPU_COUNT ="):
            self.assertNotIn(forbidden, text)

    def test_no_fixed_project_or_workspace(self) -> None:
        text = self._ui_text()
        for forbidden in ("web_projects/default", "web_projects/1", "/root/share"):
            self.assertNotIn(forbidden, text)

    def test_no_metric_values_are_written_into_the_ui(self) -> None:
        text = self._ui_text()
        for forbidden in ("794.32", "455.58", "0.1819", "RMSE ="):
            self.assertNotIn(forbidden, text)

    def test_no_fixed_model_or_blocked_counts(self) -> None:
        text = self._ui_text()
        for forbidden in ("23 models", "= 23", "blocked_count = 1"):
            self.assertNotIn(forbidden, text)

    def test_counts_come_from_the_context_not_a_constant(self) -> None:
        """Changing the context changes what the page reports."""
        small = dict(_MATR_CONTEXT, runnable_models=_MATR_CONTEXT["runnable_models"][:1])
        fake_small = FakeStreamlit({})
        with _fake_streamlit(fake_small):
            ui._render_context(small)
        fake_full = FakeStreamlit({})
        with _fake_streamlit(fake_full):
            ui._render_context(_MATR_CONTEXT)
        self.assertIn("1 model(s) runnable", " ".join(fake_small.rec["captions"]))
        self.assertIn("2 model(s) runnable", " ".join(fake_full.rec["captions"]))


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

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_ready_models_produce_executable_plans(self) -> None:
        plan = self._plan("MATR", "RUL prediction", "cptransformer")
        self.assertTrue(ui.run_is_enabled(plan), plan.get("reason"))
        self.assertEqual(plan["model"], "cptransformer")
        self.assertTrue(plan["config_path"].endswith("configs/external/cptransformer/matr_1.yaml"))

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_blocked_pbt_plan_disabled(self) -> None:
        plan = self._plan("MATR", "RUL prediction", "PBT")
        self.assertFalse(ui.run_is_enabled(plan))
        self.assertTrue(plan["command"] in (None, []))

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_trajectory_batterymformer_never_rul_executable(self) -> None:
        plan = self._plan("MATR", "Degradation trajectory forecasting", "BatteryMFormer")
        self.assertFalse(ui.run_is_enabled(plan))
        self.assertTrue(plan["command"] in (None, []))

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_mich_raw_only_plan_disabled_and_without_command(self) -> None:
        plan = self._plan("MICH", "RUL prediction", "cptransformer")
        self.assertEqual(plan["dataset_status"], "RAW_ONLY")
        self.assertFalse(ui.run_is_enabled(plan))
        self.assertIsNone(plan["command"])
        self.assertNotIn("data/raw/MICH", str(plan))

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
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
