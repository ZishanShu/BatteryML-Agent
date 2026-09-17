"""SOH in the Agent and in the result views.

Two things are established here.

**Intent.** "Estimate SOH" resolves to SOH estimation; "predict battery
lifetime", "remaining useful life" and "forecast future degradation" resolve
to something else. These are different scientific questions, and a resolver
that blurs them would run the wrong task and label the result correctly
anyway — the failure would be invisible in the output. Chinese phrasings are
covered because the platform accepts them.

**Results.** An SOH view must read the saved artifact: the observed and
predicted values it plots, and the cell each point belongs to, come from
``sample_index``, not from row order. The figure is asserted to refuse a
whole-cell artifact rather than plotting cycle numbers it invented.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_soh_agent_results -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from batteryml_web.agent.intent import resolve_workflow_intent  # noqa: E402
from batteryml_web.core.readiness import (  # noqa: E402
    TASK_RUL,
    TASK_SOH,
    TASK_TRAJECTORY,
)


def _task_of(request: str) -> str:
    resolved = resolve_workflow_intent(request)
    if isinstance(resolved, dict):
        return str(resolved.get("task") or "")
    return str(getattr(resolved, "task", "") or "")


class TestSOHIntent(unittest.TestCase):
    """SOH language resolves to SOH."""

    SOH_REQUESTS = (
        "Estimate SOH on MATR",
        "estimate SOH",
        "Estimate state of health",
        "predict state of health",
        "What is the battery health now?",
        "battery health now",
        "Estimate health at cycle 300",
        "Estimate SOH for this battery",
        "帮我预测电池 SOH",
        "估计当前电池健康状态",
        "电池健康状态",
        "估计 SOH",
    )

    def test_soh_phrases_resolve_to_soh(self) -> None:
        for request in self.SOH_REQUESTS:
            with self.subTest(request=request):
                self.assertEqual(_task_of(request), TASK_SOH)


class TestNonSOHIntent(unittest.TestCase):
    """Lifetime, RUL and trajectory are not SOH."""

    def test_lifetime_and_rul_are_not_soh(self) -> None:
        for request in (
            "predict battery lifetime",
            "Predict battery lifetime on MATR",
            "remaining useful life",
            "How many cycles are left?",
            "预测电池寿命",
            "剩余使用寿命",
        ):
            with self.subTest(request=request):
                task = _task_of(request)
                self.assertNotEqual(
                    task, TASK_SOH,
                    f"{request!r} is a lifetime question and must not run SOH",
                )
                self.assertEqual(task, TASK_RUL)

    def test_trajectory_is_not_soh(self) -> None:
        for request in (
            "forecast future degradation",
            "How will capacity change over the next 100 cycles?",
            "predict the degradation trajectory",
        ):
            with self.subTest(request=request):
                task = _task_of(request)
                self.assertNotEqual(task, TASK_SOH)
                self.assertIn(task, (TASK_TRAJECTORY, TASK_RUL))

    def test_eol_soh_wording_does_not_drag_a_lifetime_request_into_soh(self) -> None:
        """The config-level decoy, in natural language.

        "End of life at 80% SOH" names the *threshold* an RUL label uses,
        exactly as ``eol_soh`` does in a config. The question is still how
        long the cell lasts, and answering it with an SOH run would be the
        same category error at the other end of the platform.
        """
        for request in (
            "predict the remaining cycles until end of life at 80% SOH",
            "how many cycles until EOL, defined as 80% SOH?",
            "remaining useful life with end-of-life at 0.8 SOH",
        ):
            with self.subTest(request=request):
                self.assertEqual(_task_of(request), TASK_RUL)

    def test_a_real_soh_request_still_resolves_when_eol_is_mentioned(self) -> None:
        """Stripping the decoy must not swallow a genuine SOH question."""
        self.assertEqual(
            _task_of("estimate SOH now; end of life is a separate question"),
            TASK_SOH,
        )


class TestSOHMetrics(unittest.TestCase):
    """MAE and RMSE, not the RUL metric set."""

    def test_soh_asks_for_mae_and_rmse_only(self) -> None:
        from batteryml_web.agent.execution import metrics_for_task

        self.assertEqual(set(metrics_for_task(TASK_SOH)), {"MAE", "RMSE"})
        self.assertNotIn("MAPE", metrics_for_task(TASK_SOH))

    def test_other_tasks_keep_their_metrics(self) -> None:
        from batteryml_web.agent.execution import DEFAULT_METRICS, metrics_for_task

        self.assertEqual(metrics_for_task(TASK_RUL), DEFAULT_METRICS)
        self.assertIn("MAPE", metrics_for_task(TASK_RUL))

    def test_the_plan_passes_the_task_metrics(self) -> None:
        import batteryml_web.agent.execution as execution
        import inspect

        source = inspect.getsource(execution.build_execution_plan)
        self.assertIn("metrics_for_task(resolved_task)", source)
        self.assertIn('"--metric"', source)


class TestSOHResultViews(unittest.TestCase):
    """The views read the saved artifact, not row order."""

    def setUp(self) -> None:
        from batteryml_web.viz import figures, style

        self.figures = figures
        self.theme = style.THEMES["light"] if hasattr(style, "THEMES") else style.light_theme()

    def _draw(self, cells, cycles, observed, predicted):
        return self.figures.soh_vs_cycle_figure(
            cells, np.asarray(cycles), np.asarray(observed), np.asarray(predicted),
            theme=self.theme, title="t", subtitle="s", dpi=72.0,
        )

    def test_it_groups_by_real_cell_id(self) -> None:
        cells = ["cell-a", "cell-b", "cell-a", "cell-b"]
        fig = self._draw(cells, [1, 1, 2, 2], [1.0, 0.9, 0.8, 0.7], [0.99, 0.91, 0.82, 0.69])
        ax = fig.axes[0]
        drawn = {line.get_label() for line in ax.get_lines()}
        self.assertIn("cell-a", drawn)
        self.assertIn("cell-b", drawn)

    def test_the_plotted_values_are_the_given_ones(self) -> None:
        cells = ["cell-a", "cell-a", "cell-a"]
        observed = [1.0, 0.9, 0.8]
        fig = self._draw(cells, [5, 10, 15], observed, [0.99, 0.92, 0.79])
        line = fig.axes[0].get_lines()[0]
        self.assertEqual(list(line.get_xdata()), [5.0, 10.0, 15.0])
        self.assertEqual(list(line.get_ydata()), observed)

    def test_points_are_ordered_by_cycle_not_by_row(self) -> None:
        cells = ["cell-a"] * 3
        fig = self._draw(cells, [30, 10, 20], [0.8, 1.0, 0.9], [0.81, 0.99, 0.91])
        line = fig.axes[0].get_lines()[0]
        self.assertEqual(list(line.get_xdata()), [10.0, 20.0, 30.0])
        self.assertEqual(list(line.get_ydata()), [1.0, 0.9, 0.8])

    def test_mismatched_lengths_are_refused(self) -> None:
        with self.assertRaises(self.figures.VizFigureError) as ctx:
            self._draw(["a", "b"], [1], [1.0], [1.0])
        self.assertIn("same length", str(ctx.exception))

    def test_an_empty_artifact_is_refused_rather_than_drawn_empty(self) -> None:
        with self.assertRaises(self.figures.VizFigureError) as ctx:
            self._draw([], [], [], [])
        self.assertIn("cycle-level", str(ctx.exception))

    def test_it_says_how_many_cells_it_is_showing(self) -> None:
        cells = [f"cell-{i}" for i in range(12) for _ in range(3)]
        cycles = [c for _ in range(12) for c in (1, 2, 3)]
        values = [1.0, 0.9, 0.8] * 12
        fig = self._draw(cells, cycles, values, values)
        text = " ".join(t.get_text() for t in fig.axes[0].texts)
        self.assertIn("of 12 test cell", text)

    def test_it_is_not_called_a_forecast(self) -> None:
        import inspect

        source = inspect.getsource(self.figures.soh_vs_cycle_figure)
        code = "\n".join(
            line for line in source.split('"""')[2].splitlines()
        )
        self.assertNotIn("forecast", code.lower())
        self.assertNotIn("trajectory", code.lower())
        # And the docstring says why, rather than leaving it implied.
        self.assertIn("not a trajectory forecast", source.lower())


class TestSOHArtifactContract(unittest.TestCase):
    """The loader validates the recorded identity before anything plots it."""

    def test_a_length_mismatch_is_caught(self) -> None:
        from batteryml_web.viz.loaders import VizDataError, _sample_index_of

        payload = {"sample_index": [{"cell_id": "a", "cycle_index": 1}]}
        with self.assertRaises(VizDataError) as ctx:
            _sample_index_of(payload, Path("x.pkl"), 3)
        self.assertIn("mislabel every point", str(ctx.exception))

    def test_a_malformed_record_is_caught(self) -> None:
        from batteryml_web.viz.loaders import VizDataError, _sample_index_of

        with self.assertRaises(VizDataError):
            _sample_index_of({"sample_index": [{"cell_id": "a"}]}, Path("x.pkl"), 1)

    def test_a_whole_cell_artifact_simply_has_none(self) -> None:
        from batteryml_web.viz.loaders import _sample_index_of

        self.assertIsNone(_sample_index_of({"prediction": [1, 2]}, Path("x.pkl"), 2))

    def test_the_soh_axis_label_does_not_assert_a_convention(self) -> None:
        """The reference is whatever the run configured, recorded per artifact."""
        from batteryml_web.viz.loaders import LABEL_UNITS

        for annotator in ("SOHLabelAnnotator", "CycleSOHLabelAnnotator"):
            with self.subTest(annotator=annotator):
                unit = LABEL_UNITS[annotator]
                self.assertIn("ratio", unit)
                self.assertNotIn("nominal", unit)
                self.assertNotIn("%", unit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
