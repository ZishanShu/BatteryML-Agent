"""Review Results and Run History: readable, without touching what is recorded.

Regression tests B13–B18 of the global UI/UX unification, plus the Run
History half of A7 (a run status is read, never pressed). The history table
used to print the raw run id (``run_20261012_172911_elastic_net_matr_seed0_
f9b0a1``) and the raw ISO timestamp in eleven columns; it now shows a short
id, a zoned timestamp and one ``model · dataset · task`` cell. These tests
pin that the shortening is display-only: the selector's value, the session
state, the manifest on disk and every section below still use the full id.

Every run here is a synthetic manifest in a temporary project; nothing is
trained, and the timestamps and metrics are made up for the test.

    python -m unittest tests.test_review_results_presentation -v
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.ui import run_history as rh  # noqa: E402
from tests.test_quick_experience_fixes import AppTest, _PageHarness  # noqa: E402

APP = REPO_ROOT / "batteryml_web" / "app.py"

#: Three synthetic runs, deliberately written to disk out of order.
RUNS = (
    ("run_20261012_172911_elastic_net_matr_seed0_f9b0a1", "2026-10-12T17:29:11+00:00", "Elastic Net", 1.5),
    ("run_20261012_090102_xgb_matr_seed1_0c4d2e", "2026-10-12T09:01:02+00:00", "XGBoost", 2.5),
    ("run_20261013_080000_ridge_matr_seed2_77aa10", "2026-10-13T08:00:00+00:00", "Ridge", 3.5),
)


def _manifest(run_id: str, created: str, model: str, rmse: float, run_dir: Path) -> Dict:
    return {
        "run_id": run_id, "status": rm.STATUS_COMPLETED, "created_at": created,
        "started_at": created, "dataset": "MATR", "task": "RUL", "model_display": model,
        "model_key": model.lower().replace(" ", "_"), "seed": int(run_id.split("_seed")[1][0]),
        "device": "cpu", "elapsed_seconds": 75, "metrics": {"RMSE": rmse},
        "run_dir": str(run_dir), "return_code": 0,
    }


def _seed_project(project: Path) -> Dict[str, Path]:
    dirs = {}
    for run_id, created, model, rmse in RUNS:
        run_dir = project / "MATR" / "model" / run_id
        run_dir.mkdir(parents=True)
        rm.write_manifest(run_dir, _manifest(run_id, created, model, rmse, run_dir))
        dirs[run_id] = run_dir
    return dirs


NEWEST_FIRST = [RUNS[2][0], RUNS[0][0], RUNS[1][0]]


class TestReadableValues(unittest.TestCase):
    def test_short_ids_and_zoned_timestamps(self) -> None:
        self.assertEqual(rh.short_run_id(RUNS[0][0]), "…f9b0a1")
        self.assertEqual(rh.short_run_id("legacy-run"), "legacy-run")
        self.assertEqual(rh.short_run_id("a_very_long_legacy_identifier_123456"), "a_very_l…123456")
        self.assertEqual(rh.short_run_id(""), rh.NOT_AVAILABLE)
        self.assertEqual(rh.format_timestamp("2026-10-12T17:29:11+00:00"), "2026-10-12 17:29 UTC")
        self.assertEqual(rh.format_timestamp("2026-10-12T17:29:11Z"), "2026-10-12 17:29 UTC")
        self.assertEqual(rh.format_timestamp("2026-10-12T17:29:11+08:00"), "2026-10-12 17:29 UTC+08:00")
        self.assertEqual(rh.format_timestamp("2026-10-12T17:29:11-05:30"), "2026-10-12 17:29 UTC-05:30")
        self.assertIn("time zone not recorded", rh.format_timestamp("2026-10-12T17:29:11"))
        self.assertEqual(rh.format_timestamp("yesterday"), "yesterday")   # shown, not guessed
        self.assertEqual(rh.format_timestamp(None), rh.NOT_AVAILABLE)

    def test_the_table_is_compact(self) -> None:
        row = rh.history_row(_manifest(*RUNS[0], Path("/tmp/x")))
        self.assertEqual(tuple(row), rh.COLUMNS)
        self.assertLessEqual(len(rh.COLUMNS), 8)
        self.assertEqual(row["Experiment"], "Elastic Net · MATR · RUL")
        self.assertEqual(row["Seed"], "0")                       # seed 0 is a seed, not "missing"
        self.assertLessEqual(max(len(v) for v in row.values()), 32)


class TestB14DisplayNeverRewritesTheRecord(unittest.TestCase):
    def test_rows_and_labels_do_not_mutate_the_manifest(self) -> None:
        manifest = _manifest(*RUNS[0], Path("/tmp/x"))
        before = copy.deepcopy(manifest)
        rh.history_row(manifest)
        rh.run_option_label(manifest)
        rh.status_badge(manifest["status"])
        self.assertEqual(manifest, before)


class TestA07HistoryStatusIsPassive(unittest.TestCase):
    def test_the_badge_is_a_span_with_the_word(self) -> None:
        for status, kind in (("COMPLETED", "completed"), ("FAILED", "failed"),
                             ("INTERRUPTED", "failed"), ("CANCELLED", "cancelled"), ("RUNNING", "running")):
            with self.subTest(status=status):
                badge = rh.status_badge(status)
                self.assertTrue(badge.startswith("<span ") and badge.endswith("</span>"))
                self.assertIn(f"bml-status--{kind}", badge)
                self.assertIn(f">{status}<", badge)
                for interactive in ("<button", "onclick", "href=", "tabindex", 'role="button"'):
                    self.assertNotIn(interactive, badge)
        self.assertNotIn("bml-status--", rh.status_badge("SOMETHING_NEW"))
        self.assertIn("&lt;b&gt;", rh.status_badge("<b>"))                # escaped


class TestB18BenchmarkGatingUnchanged(unittest.TestCase):
    def test_difference_only_when_comparable(self) -> None:
        """Source-level: the panel computes nothing and shows a difference only when the registry allows."""
        source = APP.read_text(encoding="utf-8")
        body = source[source.index("def render_benchmark_panel"):source.index("def _render_selected_result_header")]
        self.assertIn('if block.get("comparable"):', body)
        self.assertIn('Comparison: not directly comparable', body)
        self.assertIn('difference = block.get("difference")', body)
        # The panel reads the registry's numbers; it does no arithmetic of its own.
        for arithmetic in (" - record", "record[\"value\"] -", "abs("):
            self.assertNotIn(arithmetic, body)
        self.assertIn('st.markdown("#### Benchmark reference")', body)


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestReviewResultsSelection(_PageHarness, unittest.TestCase):
    """The real page, on a synthetic project in a temporary projects area.

    Review Results resolves every project under ``project_context.
    PROJECTS_ROOT``; that root is pointed at a temporary directory here, so
    nothing is read from or written to the repository's own ``runs/``.
    """

    def setUp(self) -> None:
        super().setUp()
        from unittest import mock

        from batteryml_web.core import project_context

        area = self.tmp / "web_projects"
        self.stack.enter_context(mock.patch.object(project_context, "PROJECTS_ROOT", str(area)))
        self.review_project = area / "review_project"
        self.dirs = _seed_project(self.review_project)
        self.on_disk = {run_id: (d / rm.MANIFEST_NAME).read_bytes() for run_id, d in self.dirs.items()}

    def open(self) -> "AppTest":
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.session_state["project_root"] = str(self.review_project)
        at.session_state["project_name"] = self.review_project.name
        at.session_state["current_step"] = "review_results"
        at.run()
        self.assertHealthy(at)
        return at

    def choose(self, at, run_id: str) -> None:
        selector = next(s for s in at.selectbox if s.label == "Open a run")
        selector.set_value(run_id).run()
        self.assertHealthy(at)

    def test_b13_b14_b16_b17_the_selected_run_is_the_one_shown_everywhere(self) -> None:
        target = RUNS[1][0]                                   # the oldest, not the default
        at = self.open()
        self.choose(at, target)
        # B13: the selection is the canonical id, and it survives a rerun.
        self.assertEqual(at.session_state["run_history_selection"], target)
        at.run()
        self.assertEqual(at.session_state["run_history_selection"], target)
        # B16: the full id is on the page to copy, and the recorded timestamp in full.
        self.assertIn(target, [c.value for c in at.code])
        detail = next(d.value for d in at.dataframe if "Field" in d.value.columns
                      and "Run ID" in list(d.value["Field"]))
        fields = dict(zip(detail["Field"], detail["Value"]))
        self.assertEqual(fields["Run ID"], target)
        self.assertEqual(fields["Started (as recorded)"], RUNS[1][1])
        # B17: the metric shown is this run's own, and the caption names it.
        metric_tables = [d.value for d in at.dataframe if "Metric" in d.value.columns]
        self.assertTrue(metric_tables)
        for table in metric_tables:
            self.assertEqual(list(table["Metric"]), ["RMSE"])
            self.assertTrue(all("2.5" in str(v) for v in table["Value"]), table)
        self.assertTrue(any(f"Canonical metrics recorded by run {target}" in c.value for c in at.caption))
        self.assertTrue(any(rh.short_run_id(target) in e.label for e in at.expander))
        # A7: the status is a pill in markdown; no control carries a status word.
        self.assertTrue(any("bml-status--completed" in m.value for m in at.markdown))
        statuses = {rm.STATUS_COMPLETED, rm.STATUS_FAILED, rm.STATUS_CANCELLED, rm.STATUS_RUNNING}
        self.assertFalse([b.label for b in at.button if b.label.strip().upper() in statuses])
        # B14: nothing on disk was rewritten by reading it.
        for run_id, directory in self.dirs.items():
            self.assertEqual((directory / rm.MANIFEST_NAME).read_bytes(), self.on_disk[run_id], run_id)

    def test_b15_order_and_default(self) -> None:
        at = self.open()
        table = next(d.value for d in at.dataframe if list(d.value.columns) == list(rh.COLUMNS))
        self.assertEqual(list(table["Run"]), [rh.short_run_id(r) for r in NEWEST_FIRST])
        selector = next(s for s in at.selectbox if s.label == "Open a run")
        self.assertEqual(list(selector.options),
                         [rh.run_option_label(json.loads(self.on_disk[r])) for r in NEWEST_FIRST])
        self.assertEqual(selector.value, NEWEST_FIRST[0])
        self.assertEqual(at.session_state["run_history_selection"], NEWEST_FIRST[0])

    def test_page_sections_follow_the_shared_hierarchy(self) -> None:
        at = self.open()
        self.assertEqual([h.value for h in at.header], ["📈 Review Results"])
        subheaders = [h.value for h in at.subheader]
        for section in ("Run history", "Selected result", "Optional LLM interpretation"):
            self.assertIn(section, subheaders)
        self.assertLess(subheaders.index("Selected result"), subheaders.index("Optional LLM interpretation"))
        self.assertTrue(any(m.value == "#### Benchmark reference" for m in at.markdown))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
