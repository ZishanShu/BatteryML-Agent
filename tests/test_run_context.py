"""Project -> workspace, run provenance, ETA honesty, and metric display.

The contract these protect: the project the page shows is the project the
run is written to, an ETA exists only where it was measured, and formatting
never becomes the value.

All pure — no Streamlit, no filesystem, no clock.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_run_context -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.progress_estimate import (  # noqa: E402
    ESTIMATING_LABEL,
    MIN_OBSERVATIONS,
    EpochTimer,
    format_duration,
    historical_eta_seconds,
    progress_line,
    stage_evidence_key,
)
from batteryml_web.agent.provenance import (  # noqa: E402
    FIELD_ORDER,
    UNKNOWN,
    artifact_display,
    build_run_context,
    context_rows,
    matches_run,
)
from batteryml_web.core.project_context import (  # noqa: E402
    DEFAULT_PROJECT_NAME,
    PROJECTS_ROOT,
    agent_workspace,
    display_path,
    project_root_for,
    resolve_project,
    safe_segment,
)
from batteryml_web.ui.formatting import (  # noqa: E402
    METRIC_DECIMALS,
    compact_metric,
    mape_percent,
    metric_display,
)


def _plan(workspace: str, **overrides) -> dict:
    plan = {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": "cnn",
        "model_display": "CNN — Convolutional Neural Network",
        "config_path": f"{REPO_ROOT}/configs/baselines/nn/cnn/matr_1.yaml",
        "epochs": None,
        "command": [
            "python", "batteryml_web/core/run_batteryml_cli.py", "run",
            f"{REPO_ROOT}/configs/baselines/nn/cnn/matr_1.yaml",
            "--workspace", workspace, "--train", "--eval",
            "--device", "cpu", "--seed", "0",
        ],
    }
    plan.update(overrides)
    return plan


class TestProjectDecidesTheWorkspace(unittest.TestCase):
    """The requirement: project alpha writes under alpha, beta under beta."""

    def test_named_projects_get_their_own_workspace(self) -> None:
        for name in ("alpha", "beta", "1", "my_project"):
            with self.subTest(project=name):
                project = resolve_project(project_root=f"{PROJECTS_ROOT}/{name}")
                self.assertEqual(project["name"], name)
                self.assertEqual(
                    agent_workspace(project, "MATR", "cnn"),
                    f"{PROJECTS_ROOT}/{name}/agent_matr_cnn",
                )

    def test_switching_project_switches_workspace(self) -> None:
        alpha = agent_workspace(
            resolve_project(project_root=f"{PROJECTS_ROOT}/alpha"), "MATR", "cnn"
        )
        beta = agent_workspace(
            resolve_project(project_root=f"{PROJECTS_ROOT}/beta"), "MATR", "cnn"
        )
        self.assertNotEqual(alpha, beta)
        self.assertTrue(alpha.startswith(f"{PROJECTS_ROOT}/alpha/"))
        self.assertTrue(beta.startswith(f"{PROJECTS_ROOT}/beta/"))

    def test_a_project_root_outside_the_projects_area_is_not_silently_default(
        self,
    ) -> None:
        """The old bug: display "beta", execute under "default"."""
        project = resolve_project(project_root="/somewhere/else/beta")
        self.assertEqual(project["name"], "beta")
        self.assertEqual(project["root"], f"{PROJECTS_ROOT}/beta")
        self.assertEqual(project["source"], "root")
        self.assertNotIn(DEFAULT_PROJECT_NAME, agent_workspace(project, "MATR", "cnn"))

    def test_no_project_resolves_to_a_named_default_not_a_hidden_one(self) -> None:
        project = resolve_project()
        self.assertEqual(project["name"], DEFAULT_PROJECT_NAME)
        self.assertTrue(project["is_default"])
        self.assertEqual(project["source"], "default")

    def test_name_is_used_when_no_root_is_stored(self) -> None:
        project = resolve_project(project_name="gamma")
        self.assertEqual(project["root"], f"{PROJECTS_ROOT}/gamma")
        self.assertEqual(project["source"], "name")

    def test_smoke_runs_get_their_own_directory(self) -> None:
        project = resolve_project(project_root=f"{PROJECTS_ROOT}/alpha")
        self.assertTrue(
            agent_workspace(project, "MATR", "cnn", smoke=True).endswith("_smoke")
        )

    def test_a_workspace_can_never_escape_the_projects_area(self) -> None:
        for hostile in ("..", ".", "../../etc", "/", ""):
            with self.subTest(name=hostile):
                project = resolve_project(project_root=hostile)
                workspace = agent_workspace(project, "MATR", "cnn")
                self.assertTrue(workspace.startswith(PROJECTS_ROOT + "/"))
                self.assertNotIn("..", workspace)

    def test_unusable_components_raise_rather_than_guess(self) -> None:
        project = resolve_project(project_root=f"{PROJECTS_ROOT}/alpha")
        for dataset, model in (("", "cnn"), ("MATR", ""), ("..", "cnn")):
            with self.subTest(dataset=dataset, model=model):
                with self.assertRaises(ValueError):
                    agent_workspace(project, dataset, model)

    def test_project_root_for_rejects_an_unnameable_project(self) -> None:
        with self.assertRaises(ValueError):
            project_root_for("...")
        self.assertEqual(project_root_for("alpha"), f"{PROJECTS_ROOT}/alpha")

    def test_safe_segment_never_yields_a_traversal(self) -> None:
        for hostile in ("..", ".", "../..", "   "):
            self.assertEqual(safe_segment(hostile), "")


class TestRunProvenance(unittest.TestCase):
    def setUp(self) -> None:
        self.project = resolve_project(project_root=f"{PROJECTS_ROOT}/1")
        self.workspace = agent_workspace(self.project, "MATR", "cnn")
        self.context = build_run_context(
            project=self.project,
            plan=_plan(self.workspace),
            device_label="CPU",
            seed=0,
        )

    def test_every_required_field_is_present(self) -> None:
        labels = [label for label, _ in context_rows(self.context)]
        self.assertEqual(
            labels,
            ["Project", "Run directory", "Dataset", "Task", "Model", "Config",
             "Seed", "Epochs", "Device"],
        )
        self.assertEqual(len(FIELD_ORDER), 9)

    def test_the_workspace_comes_from_the_command_that_will_execute(self) -> None:
        self.assertEqual(self.context["workspace"], self.workspace)
        rows = dict(context_rows(self.context))
        self.assertEqual(rows["Run directory"], self.workspace)
        self.assertEqual(rows["Project"], "1")

    def test_paths_are_shown_repo_relative_but_kept_absolute(self) -> None:
        rows = dict(context_rows(self.context))
        self.assertFalse(rows["Config"].startswith("/"))
        self.assertTrue(self.context["config"].startswith("/"))

    def test_absent_values_say_so_rather_than_being_invented(self) -> None:
        context = build_run_context(project=self.project, plan=None)
        rows = dict(context_rows(context))
        self.assertEqual(rows["Dataset"], UNKNOWN)
        self.assertEqual(rows["Seed"], UNKNOWN)
        self.assertEqual(rows["Run directory"], UNKNOWN)

    def test_epochs_reports_the_config_default_when_there_is_no_override(self) -> None:
        self.assertEqual(dict(context_rows(self.context))["Epochs"], "Config default")
        smoke = build_run_context(
            project=self.project, plan=_plan(self.workspace, epochs=1), seed=0
        )
        self.assertEqual(dict(context_rows(smoke))["Epochs"], "1")

    def test_a_result_from_another_workspace_is_not_our_run(self) -> None:
        self.assertTrue(matches_run(self.context, {"workspace": self.workspace}))
        self.assertFalse(
            matches_run(self.context, {"workspace": f"{PROJECTS_ROOT}/other/agent_x"})
        )
        self.assertFalse(matches_run(self.context, {}))

    def test_artifact_is_offered_in_both_forms(self) -> None:
        absolute = f"{REPO_ROOT}/{self.workspace}/predictions_seed_0_2026.pkl"
        artifact = artifact_display({"predictions_file": absolute})
        self.assertEqual(artifact["absolute"], absolute)
        self.assertEqual(
            artifact["display"], f"{self.workspace}/predictions_seed_0_2026.pkl"
        )

    def test_a_path_outside_the_repository_is_not_shortened(self) -> None:
        self.assertEqual(display_path("/elsewhere/x.pkl"), "/elsewhere/x.pkl")

    def test_the_default_project_is_flagged_so_the_page_can_say_so(self) -> None:
        context = build_run_context(project=resolve_project(), plan=None)
        self.assertTrue(context["project_is_default"])


class TestEtaIsEarnedNotInvented(unittest.TestCase):
    def test_no_eta_before_enough_observations(self) -> None:
        timer = EpochTimer()
        timer.observe(0, 100, 0.0)
        self.assertIsNone(timer.eta_seconds())
        timer.observe(1, 100, 5.0)
        self.assertEqual(timer.observations, 1)
        self.assertLess(timer.observations, MIN_OBSERVATIONS)
        self.assertIsNone(timer.eta_seconds())

    def test_an_eta_appears_once_epochs_have_been_timed(self) -> None:
        timer = EpochTimer()
        timer.observe(0, 100, 0.0)
        timer.observe(1, 100, 5.0)
        timer.observe(2, 100, 10.0)
        self.assertAlmostEqual(timer.eta_seconds(), 98 * 5.0)

    def test_the_eta_tracks_observed_timing_changes(self) -> None:
        fast = EpochTimer()
        slow = EpochTimer()
        for timer, step in ((fast, 1.0), (slow, 10.0)):
            timer.observe(0, 10, 0.0)
            timer.observe(1, 10, step)
            timer.observe(2, 10, step * 2)
        self.assertLess(fast.eta_seconds(), slow.eta_seconds())

    def test_a_stage_with_no_counter_gets_no_eta(self) -> None:
        timer = EpochTimer()
        for elapsed in (1.0, 2.0, 3.0, 4.0):
            timer.observe(None, None, elapsed)
        self.assertEqual(timer.observations, 0)
        self.assertIsNone(timer.eta_seconds())

    def test_repeated_and_backwards_counters_time_nothing(self) -> None:
        timer = EpochTimer()
        timer.observe(5, 10, 0.0)
        timer.observe(5, 10, 1.0)
        timer.observe(5, 10, 2.0)
        self.assertEqual(timer.observations, 0)
        timer.observe(1, 10, 3.0)  # a new stage reusing the bar
        self.assertEqual(timer.observations, 0)

    def test_a_finished_counter_reports_zero_not_a_negative(self) -> None:
        timer = EpochTimer()
        timer.observe(0, 2, 0.0)
        timer.observe(1, 2, 1.0)
        timer.observe(2, 2, 2.0)
        self.assertEqual(timer.eta_seconds(), 0.0)

    def test_loading_data_never_shows_a_countdown_without_evidence(self) -> None:
        line = progress_line("Loading data", elapsed_s=12.0)
        self.assertIn("Loading data", line)
        self.assertIn("elapsed 00:12", line)
        self.assertNotIn("ETA", line)
        self.assertNotIn("left", line)

    def test_estimating_is_the_honest_middle_state(self) -> None:
        line = progress_line("Training", current=1, total=100, elapsed_s=5.0, estimating=True)
        self.assertIn(ESTIMATING_LABEL, line)
        self.assertNotIn("ETA", line)

    def test_a_real_eta_is_labelled_as_an_estimate(self) -> None:
        line = progress_line("Training", current=12, total=100, elapsed_s=84.0, eta_s=616.0)
        self.assertIn("12 / 100", line)
        self.assertIn("ETA ~10:16", line)

    def test_historical_evidence_must_match_on_every_axis(self) -> None:
        key = stage_evidence_key(
            dataset="MATR", config_path="a.yaml", device="cuda:3", stage="Loading data"
        )
        other = stage_evidence_key(
            dataset="HUST", config_path="a.yaml", device="cuda:3", stage="Loading data"
        )
        history = [{"key": key, "duration_s": 20.0}, {"key": key, "duration_s": 24.0}]
        self.assertEqual(historical_eta_seconds(history, key, 4.0), 18.0)
        self.assertIsNone(historical_eta_seconds(history, other, 4.0))
        self.assertIsNone(historical_eta_seconds([], key, 4.0))

    def test_gpu_index_does_not_fragment_the_evidence(self) -> None:
        self.assertEqual(
            stage_evidence_key(dataset="M", config_path="c", device="cuda:0", stage="s"),
            stage_evidence_key(dataset="M", config_path="c", device="cuda:7", stage="s"),
        )

    def test_history_stops_predicting_once_the_run_outlives_it(self) -> None:
        history = [{"key": "k", "duration_s": 20.0}]
        self.assertIsNone(historical_eta_seconds(history, "k", 30.0))

    def test_durations_render_readably(self) -> None:
        self.assertEqual(format_duration(0), "00:00")
        self.assertEqual(format_duration(84), "01:24")
        self.assertEqual(format_duration(3700), "1:01:40")
        self.assertEqual(format_duration(None), "")


class TestMetricDisplayNeverBecomesTheValue(unittest.TestCase):
    CANONICAL = {
        "RMSE": 455.5853271484375,
        "MAE": 393.8516540527344,
        "MAPE": 0.1819471241235733,
    }

    def test_compact_display_is_short_and_unclipped(self) -> None:
        self.assertEqual(compact_metric(455.5853271484375), "455.585")
        self.assertEqual(len("455.585".split(".")[1]), METRIC_DECIMALS)

    def test_the_canonical_value_is_untouched(self) -> None:
        before = dict(self.CANONICAL)
        for key in self.CANONICAL:
            metric_display(self.CANONICAL, key)
        self.assertEqual(self.CANONICAL, before)

    def test_full_precision_stays_available(self) -> None:
        cell = metric_display(self.CANONICAL, "RMSE")
        self.assertEqual(cell["exact"], repr(455.5853271484375))
        self.assertNotEqual(cell["exact"], cell["compact"])

    def test_mape_percent_is_display_only(self) -> None:
        cell = metric_display(self.CANONICAL, "MAPE")
        self.assertEqual(cell["percent"], "18.19%")
        # The canonical ratio, not the percentage, is what was stored.
        self.assertEqual(cell["exact"], repr(0.1819471241235733))
        self.assertIsNone(metric_display(self.CANONICAL, "RMSE")["percent"])

    def test_a_missing_metric_is_absent_not_zero(self) -> None:
        cell = metric_display(self.CANONICAL, "NOPE")
        self.assertFalse(cell["present"])
        self.assertIsNone(cell["compact"])

    def test_non_numbers_are_shown_as_they_are(self) -> None:
        self.assertIsNone(mape_percent("n/a"))
        self.assertIsNone(mape_percent(True))
        self.assertEqual(compact_metric("n/a"), "n/a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
