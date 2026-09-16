"""Request-aware model presentation: show the answer, hide the catalogue.

Milestone E. Pure: the presentation layer takes a resolved intent and a
readiness context and returns a structure. No Streamlit, no LLM, no
readiness call — the context is supplied, because readiness is the caller's
job and the whole point is that this module cannot overrule it.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_model_presentation -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.recommendation import (  # noqa: E402
    MODE_BROWSE,
    MODE_RECOMMENDATION,
    MODE_REQUESTED_BLOCKED,
    MODE_REQUESTED_READY,
    MODE_REQUESTED_UNKNOWN,
    NO_SELECTION,
    PRIMARY_LIMIT,
    build_model_presentation,
    default_option_index,
    hidden_blocked_count,
    hidden_runnable_count,
    selectbox_options,
)

RECOMMENDATION_PATH = REPO_ROOT / "batteryml_web" / "agent" / "recommendation.py"

#: Five runnable and two blocked models, so "show at most three" is testable.
#: The keys are deliberately opaque: the module under test must not care.
CONTEXT = {
    "dataset": "DS",
    "dataset_status": "READY",
    "task": "RUL prediction",
    "runnable_models": [
        {"model": f"model_{i}", "model_display": f"Model {i}", "status": "READY",
         "validated_config": f"configs/{i}.yaml"}
        for i in range(5)
    ],
    "blocked_models": [
        {"model": "blocked_a", "status": "NEEDS_ASSETS", "missing": ["checkpoint"],
         "recommended_action": "Provide the pretrained assets."},
        {"model": "blocked_b", "status": "CONFIG_INVALID", "missing": ["bad field"],
         "recommended_action": "Fix the config."},
    ],
    "recommended_next_steps": [],
}

EMPTY_CONTEXT = {
    "dataset": "DS",
    "dataset_status": "RAW_ONLY",
    "task": "RUL prediction",
    "runnable_models": [],
    "blocked_models": list(CONTEXT["blocked_models"]),
    "recommended_next_steps": [],
}


def _keys(entries) -> list[str]:
    return [str(entry.get("model")) for entry in entries]


class TestRequestedReadyModel(unittest.TestCase):
    """1. A named, runnable model is the answer; nothing competes with it."""

    def setUp(self) -> None:
        self.presentation = build_model_presentation(
            resolution={"requested_model": "model_3"},
            workflow_context=CONTEXT,
            model_request={
                "status": "runnable", "model_key": "model_3", "requested": "model_3",
                "reason": "", "alternatives": [], "requires_confirmation": False,
            },
        )

    def test_only_the_requested_model_is_named(self) -> None:
        self.assertEqual(self.presentation["mode"], MODE_REQUESTED_READY)
        self.assertEqual(_keys(self.presentation["primary"]), ["model_3"])

    def test_it_is_preselected_and_needs_no_confirmation(self) -> None:
        self.assertEqual(self.presentation["default_model"], "model_3")
        self.assertFalse(self.presentation["requires_confirmation"])
        options = selectbox_options(self.presentation)
        self.assertNotIn(NO_SELECTION, options)
        self.assertEqual(options[default_option_index(self.presentation)], "model_3")

    def test_the_others_stay_available_but_unnamed(self) -> None:
        self.assertEqual(hidden_runnable_count(self.presentation), 4)
        self.assertEqual(len(self.presentation["selectable"]), 5)


class TestRequestedBlockedModel(unittest.TestCase):
    """2. Exact blockers, then alternatives — and never a substitution."""

    def setUp(self) -> None:
        self.presentation = build_model_presentation(
            resolution={"requested_model": "blocked_a"},
            workflow_context=CONTEXT,
            model_request={
                "status": "blocked", "model_key": "blocked_a", "requested": "blocked_a",
                "reason": "blocked_a is NEEDS_ASSETS for this dataset and task.",
                "alternatives": _keys(CONTEXT["runnable_models"]),
                "requires_confirmation": True,
            },
        )

    def test_the_blocker_is_shown_verbatim(self) -> None:
        self.assertEqual(self.presentation["mode"], MODE_REQUESTED_BLOCKED)
        focus = self.presentation["blocked_focus"]
        self.assertEqual(_keys(focus), ["blocked_a"])
        self.assertEqual(focus[0]["status"], "NEEDS_ASSETS")
        self.assertEqual(focus[0]["missing"], ["checkpoint"])
        self.assertEqual(focus[0]["recommended_action"], "Provide the pretrained assets.")

    def test_a_small_set_of_ready_alternatives_is_offered(self) -> None:
        self.assertEqual(len(self.presentation["primary"]), PRIMARY_LIMIT)
        self.assertTrue(
            set(self.presentation["alternatives"]) <= set(_keys(CONTEXT["runnable_models"]))
        )

    def test_nothing_is_substituted_and_confirmation_is_required(self) -> None:
        self.assertTrue(self.presentation["requires_confirmation"])
        self.assertIsNone(self.presentation["default_model"])
        options = selectbox_options(self.presentation)
        self.assertEqual(options[0], NO_SELECTION)
        self.assertEqual(default_option_index(self.presentation), 0)
        self.assertTrue(
            any("Nothing was substituted" in note for note in self.presentation["notes"])
        )

    def test_the_blocked_model_never_becomes_selectable(self) -> None:
        self.assertNotIn("blocked_a", self.presentation["selectable"])


class TestRecommendation(unittest.TestCase):
    """3. The agent proposes; readiness decides; at most three are shown."""

    def test_agent_suggestions_are_kept_only_where_runnable(self) -> None:
        presentation = build_model_presentation(
            resolution={"recommendation_requested": True},
            workflow_context=CONTEXT,
            suggestions={
                "runnable": ["model_4", "model_1"],
                "blocked": ["blocked_a"],
                "unknown": ["not_a_model"],
            },
        )
        self.assertEqual(presentation["mode"], MODE_RECOMMENDATION)
        self.assertEqual(_keys(presentation["primary"]), ["model_4", "model_1"])
        self.assertEqual(presentation["default_model"], "model_4")
        self.assertEqual(_keys(presentation["blocked_focus"]), ["blocked_a"])

    def test_a_suggestion_that_is_not_runnable_is_dropped(self) -> None:
        presentation = build_model_presentation(
            resolution={"recommendation_requested": True},
            workflow_context=CONTEXT,
            suggestions={"runnable": ["model_0", "blocked_a", "invented"], "blocked": [], "unknown": []},
        )
        self.assertEqual(_keys(presentation["primary"]), ["model_0"])

    def test_at_most_three_primary_recommendations(self) -> None:
        presentation = build_model_presentation(
            resolution={"recommendation_requested": True},
            workflow_context=CONTEXT,
            suggestions={"runnable": _keys(CONTEXT["runnable_models"]), "blocked": [], "unknown": []},
        )
        self.assertEqual(len(presentation["primary"]), PRIMARY_LIMIT)

    def test_without_agent_suggestions_the_backend_order_is_used(self) -> None:
        presentation = build_model_presentation(
            resolution={"recommendation_requested": True},
            workflow_context=CONTEXT,
            suggestions={"runnable": [], "blocked": [], "unknown": []},
        )
        self.assertEqual(_keys(presentation["primary"]), ["model_0", "model_1", "model_2"])
        self.assertTrue(
            any("readiness reports as ready" in note for note in presentation["notes"])
        )

    def test_nothing_runnable_recommends_nothing(self) -> None:
        presentation = build_model_presentation(
            resolution={"recommendation_requested": True},
            workflow_context=EMPTY_CONTEXT,
            suggestions={"runnable": ["model_0"], "blocked": [], "unknown": []},
        )
        self.assertEqual(presentation["primary"], [])
        self.assertIsNone(presentation["default_model"])
        self.assertIn("No model can run", presentation["headline"])


class TestUnknownModel(unittest.TestCase):
    """4. Not registered: say so, suggest, and require a deliberate choice."""

    def setUp(self) -> None:
        self.presentation = build_model_presentation(
            resolution={"requested_model_name": "SomeNewModel"},
            workflow_context=CONTEXT,
            model_request={
                "status": "unknown", "model_key": None, "requested": "SomeNewModel",
                "reason": "'SomeNewModel' is not currently registered on this platform.",
                "alternatives": ["model_2", "model_0"],
                "requires_confirmation": True,
            },
        )

    def test_it_is_reported_as_unregistered(self) -> None:
        self.assertEqual(self.presentation["mode"], MODE_REQUESTED_UNKNOWN)
        self.assertIn("SomeNewModel", self.presentation["headline"])
        self.assertIn("not registered", self.presentation["headline"])

    def test_suggestions_are_runnable_and_not_preselected(self) -> None:
        self.assertEqual(_keys(self.presentation["primary"]), ["model_2", "model_0"])
        self.assertIsNone(self.presentation["default_model"])
        self.assertTrue(self.presentation["requires_confirmation"])
        self.assertEqual(selectbox_options(self.presentation)[0], NO_SELECTION)

    def test_an_alternative_that_is_not_runnable_is_not_suggested(self) -> None:
        presentation = build_model_presentation(
            resolution={"requested_model_name": "X"},
            workflow_context=CONTEXT,
            model_request={
                "status": "unknown", "model_key": None, "requested": "X",
                "reason": "", "alternatives": ["blocked_a", "model_1"],
                "requires_confirmation": True,
            },
        )
        self.assertEqual(_keys(presentation["primary"]), ["model_1"])


class TestBrowseAndExpanders(unittest.TestCase):
    """5 and 6. The full catalogues live behind an optional expansion."""

    def test_browse_names_at_most_three(self) -> None:
        presentation = build_model_presentation(resolution={}, workflow_context=CONTEXT)
        self.assertEqual(presentation["mode"], MODE_BROWSE)
        self.assertEqual(len(presentation["primary"]), PRIMARY_LIMIT)

    def test_the_remainder_is_counted_for_an_expander(self) -> None:
        presentation = build_model_presentation(resolution={}, workflow_context=CONTEXT)
        self.assertEqual(hidden_runnable_count(presentation), 2)
        self.assertEqual(len(presentation["all_runnable"]), 5)

    def test_blocked_catalogue_is_hidden_unless_it_is_the_subject(self) -> None:
        browse = build_model_presentation(resolution={}, workflow_context=CONTEXT)
        self.assertEqual(browse["blocked_focus"], [])
        self.assertEqual(hidden_blocked_count(browse), 2)

        blocked = build_model_presentation(
            resolution={"requested_model": "blocked_a"},
            workflow_context=CONTEXT,
            model_request={
                "status": "blocked", "model_key": "blocked_a", "requested": "blocked_a",
                "reason": "", "alternatives": [], "requires_confirmation": True,
            },
        )
        self.assertEqual(_keys(blocked["blocked_focus"]), ["blocked_a"])
        self.assertEqual(hidden_blocked_count(blocked), 1)

    def test_the_presentation_never_invents_an_entry(self) -> None:
        presentation = build_model_presentation(
            resolution={"requested_model": "ghost"},
            workflow_context=CONTEXT,
            model_request={
                "status": "runnable", "model_key": "ghost", "requested": "ghost",
                "reason": "", "alternatives": [], "requires_confirmation": False,
            },
        )
        # readiness never listed "ghost", so nothing is shown for it.
        self.assertEqual(presentation["primary"], [])


class TestNoHardcodedModelNames(unittest.TestCase):
    """The absolute rule: recommendation logic names no model."""

    FORBIDDEN = (
        "CPTransformer", "cptransformer",
        "BatLiNet", "batlinet",
        "PBT", "pbt",
        "XGBoost", "xgboost", "xgb",
        "BatteryMFormer",
    )

    def test_the_recommendation_module_names_no_model(self) -> None:
        source = RECOMMENDATION_PATH.read_text(encoding="utf-8")
        for name in self.FORBIDDEN:
            self.assertNotIn(name, source, f"recommendation.py names {name}")

    def test_the_ui_recommendation_renderer_names_no_model(self) -> None:
        source = (
            REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        renderer = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_render_model_presentation"
        )
        body = ast.get_source_segment(source, renderer) or ""
        for name in self.FORBIDDEN:
            self.assertNotIn(name, body, f"the renderer names {name}")

    def test_the_module_reads_only_the_supplied_context(self) -> None:
        """It cannot consult readiness itself, so it cannot overrule it."""
        source = RECOMMENDATION_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "get_workflow_context",
            "build_agent_workflow_context",
            "assess_model",
            "assess_dataset",
            "load_and_validate_config_file",
            "import streamlit",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
