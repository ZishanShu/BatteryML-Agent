"""What the result page draws, what it must not, and what it must not lose.

Three narrow guards, each pinning a defect or an audit conclusion:

**1. A scalar RUL result must not announce a cycle-wise curve.** The real
MATR/CNN/matr_1 run produced 42 predictions — one lifetime estimate per test
cell. The schema layer classified that correctly as ``RUL_SCHEMA`` and the
cycle view refused itself, but the *renderer* was called anyway and drew the
refusal inside a disclosure headed "Cycle-wise RUL curve". A reader scanning
headings saw a cycle-wise curve announced on a run that has no such thing.

**2. Varying predictions stay varying.** An audit of
``build_rul_view`` -> ``prediction_table`` found the path positional
throughout: no broadcast, no sort, no rounding. These tests hold that,
including the honest opposite — a genuinely constant artifact must report
one distinct value rather than manufacture variation.

**3. A finished run is terminal on disk before it is terminal in memory.**
The STEP 4.5 contract, re-asserted end to end: the manifest and Run History
must agree, and a run that exited 0 must never surface as INTERRUPTED.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent import result_views as rv  # noqa: E402
from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.run_identity import new_run_id  # noqa: E402
from batteryml_web.core.run_manager import RunManager  # noqa: E402
from tests.fixtures import synthetic_result_artifacts as fx  # noqa: E402

CYCLE_HEADING = "Cycle-wise RUL curve"


# --------------------------------------------------------------------------
# a Streamlit stand-in that records headings
# --------------------------------------------------------------------------


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeStreamlit:
    """Records every heading, caption and table the renderer draws.

    The record lists are named in the plural and the methods in the
    singular on purpose: a list called ``markdown`` beside a method called
    ``markdown`` shadows the method, and the renderer then calls a list.
    """

    def __init__(self) -> None:
        self.expanders: List[str] = []
        self.markdowns: List[str] = []
        self.captions: List[str] = []
        self.dataframes: List[Any] = []
        self.warnings: List[str] = []
        self.infos: List[str] = []
        self.pyplots: int = 0

    # The fake is itself a context manager, because a column returns one.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def expander(self, label, **_kwargs):
        self.expanders.append(str(label))
        return _Ctx()

    def markdown(self, body, **_kwargs):
        self.markdowns.append(str(body))

    def caption(self, text, **_kwargs):
        self.captions.append(str(text))

    def dataframe(self, rows, **_kwargs):
        self.dataframes.append(rows)

    def warning(self, text, **_kwargs):
        self.warnings.append(str(text))

    def info(self, text, **_kwargs):
        self.infos.append(str(text))

    def pyplot(self, *_args, **_kwargs):
        self.pyplots += 1

    def columns(self, spec, **_kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(count)]

    def __getattr__(self, name):  # pragma: no cover - anything else is a no-op
        def _noop(*_args, **_kwargs):
            return _Ctx()

        return _noop


def _render(payload: Dict[str, Any]) -> _FakeStreamlit:
    """Drive the **production** figure dispatch over one artifact payload.

    ``_render_scientific_views`` is the real branch the page runs, and it is
    what these tests must exercise. A first version of this helper
    reimplemented that branch inline; a mutation run then showed both
    heading tests still passing with the production dispatch reverted,
    because they were testing the copy. So the real function is called, and
    only its file read is replaced: ``load_view`` is imported inside the
    function, so patching it on ``result_views`` is picked up at call time
    and the dispatch below it is untouched.
    """
    from batteryml_web.ui import run_history as rh

    views = {
        "schema": rv.detect_schema(payload),
        "rul": rv.build_rul_view(payload),
        "rul_cycles": rv.build_rul_cycle_view(payload),
        "soh": rv.build_soh_view(payload),
        "trajectory": rv.build_trajectory_view(payload),
        "summary": {},
    }
    manifest = {"artifacts": {"prediction_files": ["synthetic.pkl"]}}

    fake = _FakeStreamlit()
    saved_st, saved_loader = rh.st, rv.load_view
    try:
        rh.st = fake
        rv.load_view = lambda _path: views
        rh._render_scientific_views(manifest)
    finally:
        rh.st = saved_st
        rv.load_view = saved_loader
    return fake


# --------------------------------------------------------------------------
# 1. scalar RUL draws no cycle-wise heading
# --------------------------------------------------------------------------


class TestScalarRulDrawsNoCycleHeading(unittest.TestCase):
    """The reported defect, at the size the real run had."""

    def setUp(self) -> None:
        # 42 cells, 42 predictions: the shape of the real MATR/CNN run.
        self.payload = fx.scalar_rul_payload(n_cells=42)

    def test_the_real_shape_classifies_as_scalar_rul(self) -> None:
        self.assertEqual(rv.detect_schema(self.payload), rv.RUL_SCHEMA)
        view = rv.build_rul_view(self.payload)
        self.assertEqual(view["count"], 42)
        self.assertFalse(view["supports_cycle_curve"])

    def test_no_cycle_wise_heading_is_drawn(self) -> None:
        """The fix. Not "the curve is empty" — the heading is absent.

        Before, the renderer was called for the scalar schema too; the view
        refused itself and the refusal was drawn inside a disclosure headed
        "Cycle-wise RUL curve", which is the misleading part.
        """
        fake = _render(self.payload)
        self.assertNotIn(CYCLE_HEADING, fake.expanders)
        everything = " ".join(
            fake.expanders + fake.markdowns + fake.captions
            + fake.warnings + fake.infos
        )
        self.assertNotIn(CYCLE_HEADING, everything)

    def test_the_views_that_do_apply_are_still_drawn(self) -> None:
        """Removing the heading must not remove the result.

        Observed-vs-predicted, residuals and the prediction table all apply
        to a scalar result and must survive.
        """
        fake = _render(self.payload)
        self.assertGreaterEqual(fake.pyplots, 1, "parity/residual figures missing")
        self.assertIn("**Predictions**", fake.markdowns)
        self.assertTrue(fake.dataframes, "the prediction table is missing")
        self.assertEqual(len(fake.dataframes[-1]), 42)

    def test_the_view_layer_still_states_its_reason(self) -> None:
        """The refusal is not deleted, only not drawn as a heading.

        A caller that asks the view whether a curve exists still gets a
        reason; the page simply stops turning that reason into a title.
        """
        curve = rv.build_rul_cycle_view(self.payload)
        self.assertFalse(curve["available"])
        self.assertEqual(curve["reason"], rv.RUL_SCALAR_CURVE_NOTE)


class TestCyclewiseRulKeepsItsCurve(unittest.TestCase):
    """The other half: a genuine curve must not be lost to the fix."""

    def setUp(self) -> None:
        self.payload = fx.cyclewise_rul_payload()

    def test_it_is_classified_as_cycle_wise(self) -> None:
        self.assertEqual(rv.detect_schema(self.payload), rv.RUL_CYCLEWISE_SCHEMA)
        self.assertTrue(rv.build_rul_cycle_view(self.payload)["available"])

    def test_the_curve_is_still_rendered(self) -> None:
        """The curve's own figure is counted, not just the parity pair.

        ``>= 2`` was the first form of this assertion and it was vacuous:
        the parity and residual figures alone satisfy it, so the test
        passed with the cycle branch disabled outright. The counts below
        are measured from the fixtures (scalar 2, cycle-wise 3); the
        difference of exactly one *is* the curve.
        """
        fake = _render(self.payload)
        # Available, so it draws a figure rather than a refusal disclosure.
        self.assertNotIn(CYCLE_HEADING, fake.expanders)
        scalar_figures = _render(fx.scalar_rul_payload(n_cells=42)).pyplots
        self.assertEqual(scalar_figures, 2, "scalar baseline moved")
        self.assertEqual(
            fake.pyplots,
            3,
            "expected parity, residual and the cycle curve",
        )
        self.assertGreater(
            fake.pyplots,
            scalar_figures,
            "a cycle-wise result must draw strictly more than a scalar one",
        )

    def test_repeated_rows_without_cycle_identity_get_no_curve(self) -> None:
        """Row count is not evidence, and must not become evidence."""
        payload = fx.repeated_rows_without_cycle_identity_payload()
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)
        self.assertNotIn(CYCLE_HEADING, _render(payload).expanders)


def _render_agent(payload: Dict[str, Any]) -> _FakeStreamlit:
    """Drive the Agent page's own figure dispatch over one payload.

    ``agent_workflow._render_result_figures`` carries a second copy of the
    same schema dispatch, so the same defect could return there alone. It
    calls ``st.warning`` through its own module global but draws through the
    renderers it imports from ``run_history``, so both modules' ``st`` are
    replaced with the one fake.
    """
    from batteryml_web.ui import agent_workflow as aw
    from batteryml_web.ui import run_history as rh

    views = {
        "schema": rv.detect_schema(payload),
        "rul": rv.build_rul_view(payload),
        "rul_cycles": rv.build_rul_cycle_view(payload),
        "soh": rv.build_soh_view(payload),
        "trajectory": rv.build_trajectory_view(payload),
        "summary": {},
    }
    context = {"success": True, "predictions_file": "synthetic.pkl"}

    fake = _FakeStreamlit()
    saved_aw, saved_rh, saved_loader = aw.st, rh.st, rv.load_view
    try:
        aw.st = fake
        rh.st = fake
        rv.load_view = lambda _path: views
        aw._render_result_figures(context)
    finally:
        aw.st = saved_aw
        rh.st = saved_rh
        rv.load_view = saved_loader
    return fake


class TestAgentPageDispatchMatches(unittest.TestCase):
    """The Agent page must not keep the heading the Results page dropped."""

    def test_scalar_rul_draws_no_cycle_heading_on_the_agent_page(self) -> None:
        fake = _render_agent(fx.scalar_rul_payload(n_cells=42))
        self.assertNotIn(CYCLE_HEADING, fake.expanders)
        everything = " ".join(
            fake.expanders + fake.markdowns + fake.captions
            + fake.warnings + fake.infos
        )
        self.assertNotIn(CYCLE_HEADING, everything)
        self.assertEqual(fake.pyplots, 2, "parity and residual still expected")
        self.assertTrue(fake.dataframes, "the prediction table is missing")

    def test_a_genuine_curve_survives_on_the_agent_page(self) -> None:
        fake = _render_agent(fx.cyclewise_rul_payload())
        self.assertEqual(
            fake.pyplots, 3, "parity, residual and the cycle curve expected"
        )


class TestSohIsNeverRulCyclewise(unittest.TestCase):
    def test_a_cycle_level_soh_artifact_stays_soh(self) -> None:
        payload = fx.soh_cyclewise_payload()
        self.assertEqual(rv.detect_schema(payload), rv.SOH_SCHEMA)
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])

    def test_it_draws_no_rul_cycle_heading(self) -> None:
        fake = _render(fx.soh_cyclewise_payload())
        self.assertNotIn(CYCLE_HEADING, fake.expanders)


# --------------------------------------------------------------------------
# 2. predictions survive the path to the table
# --------------------------------------------------------------------------


class TestPredictionsReachTheTableIntact(unittest.TestCase):
    """Audit conclusion, held: the path is positional and lossless."""

    def _payload(self, predicted: List[float], observed: List[float]) -> Dict[str, Any]:
        payload = fx.scalar_rul_payload(n_cells=len(predicted))
        payload["prediction"] = list(predicted)
        payload["data"].test_data.label = list(observed)
        return payload

    def test_four_varying_predictions_all_reach_the_table(self) -> None:
        """The headline case: [100, 200, 300, 400] stays four values."""
        payload = self._payload([100.0, 200.0, 300.0, 400.0],
                                [110.0, 190.0, 310.0, 390.0])
        view = rv.build_rul_view(payload)
        self.assertEqual(view["predicted"], [100.0, 200.0, 300.0, 400.0])

        rows = rv.prediction_table(view)
        self.assertEqual([row["Predicted"] for row in rows],
                         [100.0, 200.0, 300.0, 400.0])
        self.assertEqual(len({row["Predicted"] for row in rows}), 4)

    def test_prediction_zero_is_not_broadcast_across_the_rows(self) -> None:
        """The specific failure the report would imply, named.

        A broadcast would make every row equal the first; asserting the
        *set* catches it however it happened.
        """
        payload = self._payload([100.0, 200.0, 300.0, 400.0],
                                [110.0, 190.0, 310.0, 390.0])
        rows = rv.prediction_table(rv.build_rul_view(payload))
        self.assertNotEqual(
            [row["Predicted"] for row in rows], [100.0] * 4,
            "prediction[0] was broadcast across every row",
        )

    def test_plot_series_and_table_are_the_same_aligned_arrays(self) -> None:
        """One array, read twice — never two arrays that could disagree."""
        payload = self._payload([100.0, 200.0, 300.0, 400.0],
                                [110.0, 190.0, 310.0, 390.0])
        view = rv.build_rul_view(payload)
        rows = rv.prediction_table(view)
        for index, row in enumerate(rows):
            with self.subTest(row=index):
                self.assertEqual(row["Predicted"], view["predicted"][index])
                self.assertEqual(row["Observed"], view["observed"][index])
                self.assertEqual(row["Residual"], view["residuals"][index])

    def test_residuals_pair_each_prediction_with_its_own_label(self) -> None:
        """Identity travels together: residual[i] is from row i, not row 0."""
        payload = self._payload([100.0, 200.0, 300.0, 400.0],
                                [110.0, 190.0, 310.0, 390.0])
        view = rv.build_rul_view(payload)
        self.assertEqual(view["residuals"], [-10.0, 10.0, -10.0, 10.0])

    def test_full_precision_survives_to_the_table(self) -> None:
        """Display rounding must not reach the canonical value."""
        exact = [123.456789012345, 99.87654321, 0.000123456789, 4567.891011]
        payload = self._payload(exact, [1.0, 2.0, 3.0, 4.0])
        rows = rv.prediction_table(rv.build_rul_view(payload))
        self.assertEqual([row["Predicted"] for row in rows], exact)

    def test_a_genuinely_constant_artifact_reports_one_distinct_value(self) -> None:
        """The honest opposite. No variation is manufactured.

        If a run really did predict the same number for every cell, the page
        must say so — that is a finding about the run, not something for the
        display layer to smooth over.
        """
        payload = self._payload([200.0, 200.0, 200.0, 200.0],
                                [180.0, 190.0, 210.0, 220.0])
        view = rv.build_rul_view(payload)
        self.assertEqual(view["predicted"], [200.0] * 4)
        rows = rv.prediction_table(view)
        self.assertEqual(len({row["Predicted"] for row in rows}), 1)
        # And the observed values it is compared against still vary, so the
        # residuals do too: the constancy is reported, not propagated.
        self.assertEqual(len({row["Residual"] for row in rows}), 4)

    def test_diagnostics_report_the_distinct_count_either_way(self) -> None:
        """What the diagnostics panel is for: count, spread, first values."""
        from batteryml_web.agent.prediction_diagnostics import describe_series

        varying = describe_series([100.0, 200.0, 300.0, 400.0], "prediction")
        self.assertEqual(varying["count"], 4)
        self.assertEqual(varying["distinct"], 4)
        self.assertFalse(varying["constant"])
        self.assertEqual(varying["min"], 100.0)
        self.assertEqual(varying["max"], 400.0)
        self.assertEqual(varying["non_finite"], 0)

        constant = describe_series([200.0] * 4, "prediction")
        self.assertEqual(constant["distinct"], 1)
        self.assertTrue(constant["constant"])
        self.assertEqual(constant["min"], 200.0)
        self.assertEqual(constant["max"], 200.0)


# --------------------------------------------------------------------------
# 3. run.json and Run History agree
# --------------------------------------------------------------------------


class TestTerminalStateIsConsistent(unittest.TestCase):
    """STEP 4.5's contract, read from disk and through reconciliation."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="terminal-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.manager = RunManager()

    def _run(self, child_source: str, *, cancel: bool = False) -> Dict[str, Any]:
        from tests.test_execution_lifecycle import _fake_repo, _plan

        config = _fake_repo(self.tmp, child_source)
        run_id = new_run_id(dataset="MATR", model="cnn", seed=0)
        workspace = f"runs/agent_matr_cnn/{run_id}"
        run_dir = self.tmp / workspace
        project_root = self.tmp / "runs" / "agent_matr_cnn"
        plan = _plan(config, workspace)
        rm.write_manifest(
            run_dir,
            rm.build_manifest(
                run_id=run_id, project="alpha", workspace=workspace,
                dataset="MATR", task="RUL prediction", model_key="cnn",
                config_path=str(config), seed=0, device="cpu",
                command=plan["command"],
            ),
        )
        self.manager.start(
            run_id=run_id, run_dir=run_dir, plan=plan, repo_root=self.tmp
        )

        deadline = time.monotonic() + 40.0
        snapshot = self.manager.snapshot(run_id) or {}
        while not snapshot.get("finished"):
            if time.monotonic() > deadline:
                self.fail(f"run {run_id} never left {snapshot.get('status')!r}")
            if cancel and snapshot.get("status") == rm.STATUS_RUNNING:
                handle = self.manager.get(run_id)
                if handle is not None and handle._process is not None:
                    self.manager.cancel(run_id)
                    cancel = False
            time.sleep(0)
            snapshot = self.manager.snapshot(run_id) or {}

        manifest = json.loads(
            (run_dir / rm.MANIFEST_NAME).read_text(encoding="utf-8")
        )
        # Exactly what the page does next: drop the handle, then list.
        self.manager.forget(run_id)
        entry = next(
            (
                item for item in rm.list_runs(
                    project_root, live_run_ids=self.manager.live_run_ids()
                )
                if str(item.get("run_id")) == run_id
            ),
            {},
        )
        return {"snapshot": snapshot, "manifest": manifest, "history": entry}

    def test_a_successful_run_is_completed_on_disk_and_in_history(self) -> None:
        """exit 0 -> COMPLETED in both places, and never INTERRUPTED."""
        outcome = self._run("import sys\nprint('ok')\nsys.exit(0)")
        manifest, history = outcome["manifest"], outcome["history"]

        self.assertEqual(manifest["status"], rm.STATUS_COMPLETED)
        self.assertEqual(manifest["return_code"], 0)
        self.assertTrue(manifest.get("finished_at"), "finished_at is missing")

        self.assertEqual(history.get("status"), rm.STATUS_COMPLETED)
        self.assertNotEqual(history.get("status"), rm.STATUS_INTERRUPTED)
        self.assertNotEqual(history.get("status"), rm.STATUS_RUNNING)
        # Not reconciled from anything: the manifest was already terminal.
        self.assertIsNone(history.get("reconciled_from"))

    def test_a_failing_run_is_failed_in_both_places(self) -> None:
        outcome = self._run("import sys\nsys.exit(4)")
        self.assertEqual(outcome["manifest"]["status"], rm.STATUS_FAILED)
        self.assertEqual(outcome["manifest"]["return_code"], 4)
        self.assertEqual(outcome["history"].get("status"), rm.STATUS_FAILED)

    def test_a_cancelled_run_is_cancelled_in_both_places(self) -> None:
        outcome = self._run(
            "import sys, time\nprint('started', flush=True)\ntime.sleep(120)",
            cancel=True,
        )
        self.assertEqual(outcome["manifest"]["status"], rm.STATUS_CANCELLED)
        self.assertEqual(outcome["history"].get("status"), rm.STATUS_CANCELLED)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
