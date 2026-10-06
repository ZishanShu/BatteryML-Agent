"""Scientific result figures: one data path for the page and the export.

Covers the visualization contract end to end on synthetic fixtures
(``CLAUDE.md`` §7: built in a temporary directory, ``synthetic_`` cell ids,
never under ``runs/`` or ``data/``). Real BatteryML label transformations are
used where the inverse matters, so the inverse path under test is the real
one. Nothing here trains, evaluates or reads a dataset.

Run from the repository root::

    python -m unittest tests.test_result_figures -v
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from batteryml.data.transformation.log_scale import LogScaleDataTransformation  # noqa: E402
from batteryml.data.transformation.sequential import SequentialDataTransformation  # noqa: E402
from batteryml.data.transformation.z_score import ZScoreDataTransformation  # noqa: E402
from batteryml_web.agent import result_views as rv  # noqa: E402
from batteryml_web.agent.artifact_provenance import artifact_provenance, smoke_evidence  # noqa: E402
from batteryml_web.viz import result_figures as rf  # noqa: E402
from batteryml_web.viz.result_export import ExportError, build_export, write_export  # noqa: E402
from tests.fixtures import synthetic_result_artifacts as fx  # noqa: E402

SOH_META = {
    "task": "SOH estimation", "label_annotator": "CycleSOHLabelAnnotator",
    "reference_capacity_mode": "first_valid_discharge_capacity",
    "min_cycle": 10, "cycle_stride": 5,
    "feature_extractor": "SOHVoltageCurveFeatureExtractor",
}
RUL_META = {"task": "RUL prediction", "label_annotator": "RULLabelAnnotator"}


def _chain(*members) -> SequentialDataTransformation:
    chain = SequentialDataTransformation.__new__(SequentialDataTransformation)
    chain.transformations = list(members)
    return chain


def _bundle(stored_labels: torch.Tensor, transformation: Any) -> Any:
    bundle = fx._Bundle([], transformation)
    bundle.test_data.label = stored_labels  # a tensor, as DataBundle keeps it
    return bundle


def _lifetime_payload(observed: List[float], predicted: List[float]) -> Dict[str, Any]:
    """Physical lifetimes, stored LogScale -> ZScore as the MATR configs do."""
    t = _chain(LogScaleDataTransformation(), ZScoreDataTransformation())
    t.fit(torch.tensor([[300.0], [500.0], [900.0], [1200.0], [700.0]]))
    cells = [f"synthetic_b1c{i}" for i in range(len(observed))]
    payload = fx._payload(
        prediction=t.transform(torch.tensor(predicted).reshape(-1, 1)),
        observed=None,
        sample_index=fx._records([(c, fx.WHOLE_CELL_CYCLE) for c in cells]),
        task_metadata=dict(RUL_META),
        scores={"RMSE": 40.9, "MAE": 40.0, "MAPE": 0.06},
    )
    payload["data"] = _bundle(t.transform(torch.tensor(observed).reshape(-1, 1)), t)
    return payload


def _soh_payload(pairs, observed, predicted) -> Dict[str, Any]:
    z = ZScoreDataTransformation()
    z.fit(torch.tensor([[0.95], [0.9], [0.85], [0.8], [1.0]]))
    payload = fx._payload(
        prediction=z.transform(torch.tensor(predicted).reshape(-1, 1)),
        observed=None, sample_index=fx._records(pairs),
        task_metadata=dict(SOH_META),
        scores={"RMSE": 0.1234, "MAE": 0.0456, "MAPE": 0.05},
    )
    payload["data"] = _bundle(z.transform(torch.tensor(observed).reshape(-1, 1)), z)
    return payload


# Storage order deliberately not sorted by cell or cycle.
SOH_PAIRS = [
    ("synthetic_b2c1", 20), ("synthetic_b1c0", 15), ("synthetic_b2c1", 10),
    ("synthetic_b1c0", 10), ("synthetic_b1c0", 20), ("synthetic_b2c1", 15),
]
SOH_OBSERVED = [0.93, 0.97, 0.95, 0.99, 0.95, 0.94]
SOH_PREDICTED = [0.92, 0.96, -4.34, 1.00, 2.06, 0.95]


class _Counting:
    """A transformation that counts inverse calls and doubles."""

    def __init__(self) -> None:
        self.calls = 0

    def inverse_transform(self, values):
        self.calls += 1
        return [v * 2 for v in values]


# --------------------------------------------------------------------------
# label space
# --------------------------------------------------------------------------


class TestLabelSpace(unittest.TestCase):
    def test_known_transformed_values_recover_cycles(self) -> None:
        observed = [400.0, 800.0, 1100.0]
        view = rv.build_rul_view(_lifetime_payload(observed, [450.0, 760.0, 1180.0]))
        self.assertEqual(view["transform_status"], rv.TRANSFORM_INVERTED)
        for got, want in zip(view["observed"], observed):
            self.assertAlmostEqual(got, want, delta=want * 1e-5)
        self.assertEqual(view["transform_chain"],
                         ["LogScaleDataTransformation", "ZScoreDataTransformation"])
        self.assertEqual(view["target"]["quantity"], "Battery lifetime")
        self.assertEqual(view["target"]["unit"], "cycles")

    def test_inverse_is_applied_exactly_once(self) -> None:
        counting = _Counting()
        payload = fx._payload(prediction=[1.0, 2.0], observed=[3.0, 4.0],
                              sample_index=None, task_metadata=dict(RUL_META))
        payload["data"].label_transformation = counting
        view = rv.build_rul_view(payload)
        self.assertEqual(counting.calls, 2)  # labels once, predictions once
        self.assertEqual(view["observed"], [6.0, 8.0])
        self.assertEqual(view["predicted"], [2.0, 4.0])
        # Building panels from the view never inverts again.
        rf.build_panels(view, include_zoom=True)
        self.assertEqual(counting.calls, 2)

    def test_a_failing_inverse_is_refused_not_mislabelled(self) -> None:
        class _Broken:
            def inverse_transform(self, values):
                raise RuntimeError("synthetic failure")

        payload = fx._payload(prediction=[0.1, 0.2], observed=[0.3, 0.4],
                              sample_index=None, task_metadata=dict(RUL_META))
        payload["data"].label_transformation = _Broken()
        view = rv.build_rul_view(payload)
        self.assertFalse(view["available"])
        self.assertIn("could not be inverted", view["reason"])

    def test_missing_transformation_metadata_is_refused(self) -> None:
        class _NoRecord:
            def __init__(self) -> None:
                self.test_data = fx._TestData([1.0, 2.0])

        payload = fx._payload(prediction=[1.0, 2.0], observed=None,
                              sample_index=None, task_metadata=dict(RUL_META))
        payload["data"] = _NoRecord()
        view = rv.build_rul_view(payload)
        self.assertFalse(view["available"])
        self.assertIn("does not record whether its labels were transformed", view["reason"])

    def test_no_recorded_transformation_means_values_are_plotted_as_stored(self) -> None:
        view = rv.build_rul_view(fx.scalar_rul_payload(4))
        self.assertEqual(view["transform_status"], rv.TRANSFORM_NONE)
        self.assertFalse(view["inverse_transformed"])


class TestTargetSemantics(unittest.TestCase):
    def test_soh_and_cyclewise_rul_stay_distinct(self) -> None:
        soh = rv.build_soh_view(fx.soh_cyclewise_payload())
        rul = rv.build_rul_view(fx.cyclewise_rul_payload())
        self.assertEqual(soh["target"]["kind"], rv.TARGET_SOH)
        self.assertEqual(soh["target"]["unit"], "ratio")
        self.assertEqual(rul["target"]["kind"], rv.TARGET_RUL)
        self.assertEqual(rul["schema"], rv.RUL_CYCLEWISE_SCHEMA)

    def test_whole_cell_rul_is_named_lifetime_not_rul(self) -> None:
        target = rv.build_rul_view(fx.scalar_rul_payload(3))["target"]
        self.assertEqual(target["quantity"], "Battery lifetime")
        self.assertIn("cycle life", target["definition"])

    def test_axis_labels_keep_acronyms(self) -> None:
        labels = rf._labels({"quantity": "RUL/lifetime target", "unit": "cycles", "kind": "lifetime"})
        self.assertEqual(labels["x"], "Observed RUL/lifetime target (cycles)")
        labels = rf._labels({"quantity": "Battery lifetime", "unit": "cycles", "kind": "lifetime"})
        self.assertEqual(labels["y"], "Predicted battery lifetime (cycles)")
        labels = rf._labels({"quantity": "SOH", "unit": "ratio", "kind": "soh"})
        self.assertEqual(labels["y"], "Estimated SOH (ratio)")

    def test_rul_stays_rul_in_rendered_and_exported_labels(self) -> None:
        """Regression: generic lower-casing once turned "RUL" into "rUL"."""
        payload = fx._payload(
            prediction=[500.0, 700.0], observed=[520.0, 680.0],
            sample_index=fx._records([("synthetic_a", fx.WHOLE_CELL_CYCLE),
                                      ("synthetic_b", fx.WHOLE_CELL_CYCLE)]),
            task_metadata={"task": "RUL prediction"},
        )
        view = rv.build_rul_view(payload)
        self.assertEqual(view["target"]["quantity"], "RUL/lifetime target")
        axes = rf.build_panels(view, include_zoom=False)["parity"]["figure"].axes[0]
        rendered = [axes.get_xlabel(), axes.get_ylabel(), axes.get_title(loc="left")]
        self.assertEqual(rendered[0], "Observed RUL/lifetime target (cycles)")
        self.assertEqual(rendered[1], "Predicted RUL/lifetime target (cycles)")
        for text in rendered:
            self.assertIn("RUL", text)
            self.assertNotIn("rUL", text)
        with tempfile.TemporaryDirectory(prefix="synthetic_rul_label_") as tmp:
            artifact = Path(tmp) / "predictions_seed_0_1.pkl"
            artifact.write_bytes(pickle.dumps(payload))
            notes = build_export(artifact, dpi=60).files["figure_notes.md"].decode("utf-8")
        self.assertIn("**RUL/lifetime target**", notes)
        self.assertNotIn("rUL", notes)

    def test_soh_residual_note_says_percentage_points(self) -> None:
        target = rv.build_soh_view(fx.soh_cyclewise_payload())["target"]
        self.assertIn("percentage points", target["residual_note"])

    def test_the_range_note_no_longer_calls_above_one_impossible(self) -> None:
        self.assertIn("not by itself invalid", rv.RANGE_NOTE)
        self.assertIn("nothing is clipped", rv.RANGE_NOTE.lower())


# --------------------------------------------------------------------------
# values, alignment, non-finite
# --------------------------------------------------------------------------


class TestValuesAndAlignment(unittest.TestCase):
    def test_varying_predictions_stay_varying_and_constant_stay_constant(self) -> None:
        varying = rv.build_rul_view(_lifetime_payload([400.0, 600.0, 800.0], [410.0, 640.0, 790.0]))
        panels = rf.build_panels(varying, include_zoom=False)
        self.assertEqual(len({round(v, 3) for v in panels["parity"]["data"]["predicted"]}), 3)
        constant = rv.build_rul_view(_lifetime_payload([400.0, 600.0, 800.0], [500.0, 500.0, 500.0]))
        panels = rf.build_panels(constant, include_zoom=False)
        self.assertEqual(len({round(v, 3) for v in panels["parity"]["data"]["predicted"]}), 1)

    def test_sorting_moves_complete_records_together(self) -> None:
        payload = _soh_payload(SOH_PAIRS, SOH_OBSERVED, SOH_PREDICTED)
        view = rv.build_soh_view(payload)
        series = rf.soh_cell_series(view, "synthetic_b2c1")
        self.assertEqual(series["cycles"], [10, 15, 20])
        self.assertEqual(series["artifact_rows"], [2, 5, 0])
        for cycle, row, o, p in zip(series["cycles"], series["artifact_rows"],
                                    series["observed"], series["predicted"]):
            self.assertEqual(SOH_PAIRS[row], ("synthetic_b2c1", cycle))
            self.assertAlmostEqual(o, SOH_OBSERVED[row], places=5)
            self.assertAlmostEqual(p, SOH_PREDICTED[row], places=4)

    def test_selecting_another_cell_changes_the_line_data(self) -> None:
        view = rv.build_soh_view(_soh_payload(SOH_PAIRS, SOH_OBSERVED, SOH_PREDICTED))
        first = rf.build_panels(view, cell_id="synthetic_b1c0", include_zoom=False)["soh_cycle"]["data"]
        second = rf.build_panels(view, cell_id="synthetic_b2c1", include_zoom=False)["soh_cycle"]["data"]
        self.assertEqual(first["artifact_rows"], [3, 1, 4])
        self.assertEqual(second["artifact_rows"], [2, 5, 0])
        self.assertNotEqual(first["observed"], second["observed"])

    def test_default_cell_is_first_sorted_and_outcome_blind(self) -> None:
        view = rv.build_soh_view(_soh_payload(SOH_PAIRS, SOH_OBSERVED, SOH_PREDICTED))
        meta = rf.build_panels(view, include_zoom=False)["soh_cycle"]["meta"]
        self.assertEqual(meta["cell_id"], "synthetic_b1c0")
        self.assertEqual(meta["cell_rule"], rf.DEFAULT_CELL_RULE)
        # The worst cell (holding -4.34) is not preferred or avoided.
        with self.assertRaises(rf.FigureDataError):
            rf.build_panels(view, cell_id="synthetic_not_there", include_zoom=False)

    def test_duplicate_cell_cycle_records_are_flagged_not_averaged(self) -> None:
        pairs = [("synthetic_c", 10), ("synthetic_c", 10), ("synthetic_c", 15)]
        view = rv.build_soh_view(_soh_payload(pairs, [0.9, 0.91, 0.89], [0.88, 0.92, 0.9]))
        series = rf.soh_cell_series(view, "synthetic_c")
        self.assertEqual(len(series["cycles"]), 3)
        self.assertEqual(series["duplicates"], [{"cycle_index": 10, "count": 2}])

    def test_non_finite_rows_are_omitted_with_one_mask_and_counted(self) -> None:
        predicted = list(SOH_PREDICTED)
        predicted[1] = float("nan")
        view = rv.build_soh_view(_soh_payload(SOH_PAIRS, SOH_OBSERVED, predicted))
        self.assertTrue(view["available"])
        self.assertEqual(view["schema"], rv.SOH_SCHEMA)
        self.assertEqual(view["total_rows"], 6)
        self.assertEqual(view["omitted_nonfinite"], 1)
        self.assertEqual(view["row_positions"], [0, 2, 3, 4, 5])
        self.assertEqual(len(view["cell_ids"]), 5)
        self.assertIn("1 of 6 row(s)", view["note"])
        self.assertEqual(view["range"]["count"], 5)

    def test_a_single_nan_no_longer_turns_the_result_unknown(self) -> None:
        payload = fx.scalar_rul_payload(4)
        payload["prediction"] = [900.0, float("nan"), 840.0, 810.0]
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)
        view = rv.build_rul_view(payload)
        self.assertEqual(view["count"], 3)
        self.assertEqual(view["cell_ids"],
                         ["synthetic_cell_00", "synthetic_cell_02", "synthetic_cell_03"])


# --------------------------------------------------------------------------
# full range, zoom, per-cell
# --------------------------------------------------------------------------


class TestRangesAndDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = _soh_payload(SOH_PAIRS, SOH_OBSERVED, SOH_PREDICTED)
        self.view = rv.build_soh_view(self.payload)

    def test_full_range_keeps_the_extremes(self) -> None:
        panels = rf.build_panels(self.view, include_zoom=False)
        lo, hi = panels["parity"]["meta"]["limits"]
        self.assertLess(lo, -4.34)
        self.assertGreater(hi, 2.06)
        ylo, yhi = panels["residuals"]["meta"]["ylim"]
        biggest = max(abs(r) for r in self.view["residuals"])
        self.assertLessEqual(ylo, -biggest)
        self.assertGreaterEqual(yhi, biggest)

    def test_zoom_changes_only_the_viewport(self) -> None:
        before = copy.deepcopy({k: self.view[k] for k in ("observed", "predicted", "residuals")})
        saved = {"RMSE": 0.1234, "MAE": 0.0456}
        snapshot = dict(saved)
        panels = rf.build_panels(self.view, saved_metrics=saved, include_zoom=True)
        for key, values in before.items():
            self.assertEqual(self.view[key], values)
        self.assertEqual(saved, snapshot)
        zoom = panels["parity_zoom"]["meta"]
        self.assertEqual(len(panels["parity_zoom"]["data"]["predicted"]), 6)
        expected_outside = sum(
            1 for o, p in zip(self.view["observed"], self.view["predicted"])
            if not (zoom["limits"][0] <= o <= zoom["limits"][1]
                    and zoom["limits"][0] <= p <= zoom["limits"][1])
        )
        self.assertEqual(zoom["outside"], expected_outside)
        self.assertGreaterEqual(zoom["outside"], 1)

    def test_per_cell_diagnostics_group_correctly_and_replace_nothing(self) -> None:
        rows = {row["cell_id"]: row for row in rf.per_cell_errors(self.view)}
        self.assertEqual(sorted(rows), ["synthetic_b1c0", "synthetic_b2c1"])
        residuals = [p - o for o, p in zip(SOH_OBSERVED, SOH_PREDICTED)]
        for cell in rows:
            mine = [r for (c, _), r in zip(SOH_PAIRS, residuals) if c == cell]
            self.assertEqual(rows[cell]["samples"], 3)
            self.assertAlmostEqual(rows[cell]["mean_abs_residual"],
                                   sum(abs(r) for r in mine) / 3, places=4)
            self.assertAlmostEqual(rows[cell]["rms_residual"],
                                   math.sqrt(sum(r * r for r in mine) / 3), places=4)
        self.assertEqual(self.payload["scores"]["RMSE"], 0.1234)


# --------------------------------------------------------------------------
# page state: run A -> run B
# --------------------------------------------------------------------------


class _RecordingSt:
    def __init__(self) -> None:
        self.selectboxes: List[Dict[str, Any]] = []
        self.pyplots = 0
        self.expanders: List[str] = []

    def selectbox(self, label, options, index=0, key=None, **_):
        self.selectboxes.append({"label": label, "options": list(options), "key": key})
        return list(options)[index] if options else None

    def radio(self, label, options, index=0, **_):
        return list(options)[index]

    def pyplot(self, *_a, **_k):
        self.pyplots += 1

    def columns(self, spec, **_):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def expander(self, label, **_):
        self.expanders.append(str(label))
        return self

    def button(self, *_a, **_k):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, _name):
        return lambda *a, **k: None


class TestPageState(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="synthetic_viz_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _render(self, view: Dict[str, Any]) -> _RecordingSt:
        from batteryml_web.ui import run_history as rh

        fake = _RecordingSt()
        saved = rh.st
        rh.st = fake
        try:
            rh.render_soh_figures(view, placement="predictions")
        finally:
            rh.st = saved
        return fake

    def test_cell_options_and_widget_key_follow_the_artifact(self) -> None:
        views = []
        for name, pairs in (("a", SOH_PAIRS), ("b", [("synthetic_zz9", 10), ("synthetic_zz9", 15)])):
            path = self.tmp / name / "predictions_seed_0_1.pkl"
            path.parent.mkdir()
            observed = SOH_OBSERVED[: len(pairs)]
            path.write_bytes(pickle.dumps(_soh_payload(pairs, observed, observed)))
            views.append(rv.load_view(path)["soh"])
        first, second = self._render(views[0]), self._render(views[1])
        a, b = first.selectboxes[0], second.selectboxes[0]
        self.assertEqual(a["options"], ["synthetic_b1c0", "synthetic_b2c1"])
        self.assertEqual(b["options"], ["synthetic_zz9"])
        self.assertNotEqual(a["key"], b["key"])
        self.assertIn(str(views[0]["artifact_path"]), a["key"])
        self.assertEqual(first.pyplots, 3)  # one cell, parity, residuals

    def test_scalar_rul_draws_no_cell_selector_and_no_cycle_panel(self) -> None:
        view = rv.build_rul_view(fx.scalar_rul_payload(5))
        panels = rf.build_panels(view, include_zoom=True)
        self.assertNotIn("soh_cycle", panels)
        from batteryml_web.ui import run_history as rh

        fake = _RecordingSt()
        saved = rh.st
        rh.st = fake
        try:
            rh.render_rul_figures(view)
        finally:
            rh.st = saved
        self.assertEqual(fake.selectboxes, [])
        self.assertFalse(any("cycle" in label.lower() for label in fake.expanders))


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def _read_csv(data: bytes) -> List[Dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


class TestExport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="synthetic_export_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.soh = self.tmp / "soh_smoke_named_dir" / "predictions_seed_0_1.pkl"
        self.soh.parent.mkdir()
        predicted = list(SOH_PREDICTED)
        predicted[1] = float("nan")
        self.soh.write_bytes(pickle.dumps(_soh_payload(SOH_PAIRS, SOH_OBSERVED, predicted)))
        # A sibling run that must never be referenced.
        self.other = self.tmp / "synthetic_sibling_run" / "predictions_seed_0_2.pkl"
        self.other.parent.mkdir()
        self.other.write_bytes(pickle.dumps(fx.scalar_rul_payload(3)))

    def test_csv_values_are_the_arrays_the_figures_received(self) -> None:
        result = build_export(self.soh, cell_id="synthetic_b2c1", dpi=60)
        view = rv.build_soh_view(pickle.loads(self.soh.read_bytes()))
        panels = rf.build_panels(view, cell_id="synthetic_b2c1", include_zoom=False)
        rows = _read_csv(result.files["plotted_samples.csv"])
        self.assertEqual([float(r["observed"]) for r in rows], panels["parity"]["data"]["observed"])
        self.assertEqual([float(r["predicted"]) for r in rows], panels["parity"]["data"]["predicted"])
        self.assertEqual([float(r["residual_predicted_minus_observed"]) for r in rows],
                         panels["residuals"]["data"]["residuals"])
        cell_rows = _read_csv(result.files["soh_cycle_synthetic_b2c1.csv"])
        series = panels["soh_cycle"]["data"]
        self.assertEqual([int(r["cycle_index"]) for r in cell_rows], series["cycles"])
        self.assertEqual([float(r["predicted"]) for r in cell_rows], series["predicted"])
        self.assertEqual([int(r["artifact_row"]) for r in cell_rows], series["artifact_rows"])
        for name in ("parity.svg", "parity.png", "residuals.svg", "soh_cycle_synthetic_b2c1.svg",
                     "parity_zoom.svg", "per_cell_errors.csv", "figure_notes.md"):
            self.assertIn(name, result.files)

    def test_provenance_names_only_the_selected_artifact(self) -> None:
        result = build_export(self.soh, dpi=60)
        sources = json.loads(result.files["figure_sources.json"])
        self.assertEqual(sources["source"]["artifact_path"], str(self.soh))
        self.assertEqual(sources["source"]["artifact_sha256"],
                         hashlib.sha256(self.soh.read_bytes()).hexdigest())
        dumped = json.dumps(sources)
        self.assertNotIn("synthetic_sibling_run", dumped)
        self.assertNotIn(str(self.other), dumped)
        self.assertEqual(sources["counts"]["omitted_nonfinite_rows"], 1)
        self.assertEqual(sources["counts"]["plotted_rows"], 5)
        self.assertEqual(sources["selected_cell"]["selection_rule"], rf.DEFAULT_CELL_RULE)
        self.assertEqual(sources["saved_metrics"]["values"]["RMSE"], 0.1234)
        self.assertTrue(sources["transformation"]["inverse_applied_once"])

    def test_a_directory_named_smoke_is_not_smoke_evidence(self) -> None:
        record = artifact_provenance(self.soh)
        self.assertEqual(record["smoke"], "not recorded")
        self.assertIn("not recorded", smoke_evidence(record))

    def test_run_json_smoke_and_the_config_copy_trap_are_recorded(self) -> None:
        directory = self.soh.parent
        (directory / "run.json").write_text(json.dumps(
            {"run_id": "run_x", "smoke": True, "epochs": "1", "seed": "0"}))
        (directory / "config_1.yaml").write_text("model:\n  name: CNNRULPredictor\n  epochs: 1000\n")
        record = artifact_provenance(self.soh)
        self.assertIs(record["smoke"], True)
        self.assertEqual(record["config_copy_epochs"], 1000)
        self.assertTrue(any("--epochs" in note for note in record["notes"]))
        self.assertIn("smoke=true", smoke_evidence(record))

    def test_refusals(self) -> None:
        with self.assertRaises(ExportError):
            build_export(self.soh, cell_id="synthetic_missing", dpi=60)
        result = build_export(self.soh, dpi=60)
        out = self.tmp / "export"
        write_export(result, out)
        with self.assertRaises(ExportError):
            write_export(result, out)  # never overwrites
        trajectory = self.tmp / "traj" / "predictions_seed_0_3.pkl"
        trajectory.parent.mkdir()
        trajectory.write_bytes(pickle.dumps(fx.trajectory_payload()))
        with self.assertRaises(ExportError):
            build_export(trajectory, dpi=60)

    def test_website_export_records_structured_provenance_not_a_command(self) -> None:
        """Regression: the website once wrote '(website export) <path>', then a
        CLI line rebuilt with a placeholder output directory. Neither ran."""
        run_command = ["python", "batteryml_web/core/run_batteryml_cli.py", "run",
                       "configs/soh/ridge/matr.yaml", "--workspace", "runs/x",
                       "--train", "--eval", "--device", "cpu", "--seed", "0"]
        (self.soh.parent / "run.json").write_text(json.dumps(
            {"run_id": "run_x", "command": run_command}))
        invocation = {"interface": "website", "action": "Prepare figure package",
                      "artifact": str(self.soh), "cell_id": None, "dpi": 60.0}
        result = build_export(self.soh, dpi=60, invocation=invocation)
        sources = json.loads(result.files["figure_sources.json"])
        self.assertEqual(sources["export"]["command"], [])
        self.assertEqual(sources["export"]["invocation"], invocation)
        self.assertEqual(sources["execution_provenance"],
                         {"source": "run.json", "command": run_command})
        notes = result.files["figure_notes.md"].decode("utf-8")
        self.assertIn(" ".join(run_command), notes)
        self.assertIn('"interface": "website"', notes)
        for fabricated in ("(website export)", "<NEW_DIRECTORY>", "scripts.export_result_figures"):
            self.assertNotIn(fabricated, notes)
            self.assertNotIn(fabricated, json.dumps(sources["export"]))

    def test_without_run_json_the_execution_command_is_not_recorded(self) -> None:
        result = build_export(self.soh, dpi=60,
                              invocation={"interface": "website", "artifact": str(self.soh)})
        sources = json.loads(result.files["figure_sources.json"])
        self.assertEqual(sources["execution_provenance"],
                         {"source": "not recorded", "command": None})
        self.assertIn("Command that produced the artifact: not recorded",
                      result.files["figure_notes.md"].decode("utf-8"))

    def test_the_website_button_passes_no_command(self) -> None:
        from batteryml_web.ui import run_history as rh
        from batteryml_web.viz import result_export as rx

        calls: List[Dict[str, Any]] = []
        real = rx.build_export

        def capture(path, **kwargs):
            calls.append(kwargs)
            return real(path, **kwargs)

        class _ClickSt(_RecordingSt):
            def button(self, *_a, **_k):
                return True

        saved_st, rx.build_export = rh.st, capture
        rh.st = _ClickSt()
        try:
            rh._render_export_action({"artifact_path": str(self.soh)}, "scope", cell_id=None)
        finally:
            rh.st, rx.build_export = saved_st, real
        self.assertEqual(len(calls), 1)
        self.assertNotIn("command", calls[0])
        self.assertEqual(calls[0]["invocation"]["interface"], "website")
        self.assertEqual(calls[0]["invocation"]["artifact"], str(self.soh))

    def test_a_cli_export_records_its_literal_argv(self) -> None:
        import contextlib

        from scripts import export_result_figures as cli

        out = self.tmp / "cli_argv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--artifact", str(self.soh), "--output", str(out),
                                       "--dpi", "60"]), 0)
        export = json.loads((out / "figure_sources.json").read_text())["export"]
        self.assertEqual(export["command"][:3], ["python", "-m", "scripts.export_result_figures"])
        self.assertIn(str(out), export["command"])
        self.assertEqual(export["invocation"]["interface"], "cli")

    def test_the_cli_writes_into_a_new_directory_only(self) -> None:
        import contextlib

        from scripts import export_result_figures as cli

        out = self.tmp / "cli_out"
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            self.assertEqual(cli.main(["--artifact", str(self.soh), "--output", str(out), "--dpi", "60"]), 0)
            self.assertTrue((out / "figure_sources.json").is_file())
            self.assertEqual(cli.main(["--artifact", str(self.soh), "--output", str(out)]), 1)
            self.assertEqual(cli.main(["--artifact", str(self.soh), "--list-cells"]), 0)
        self.assertIn("synthetic_b1c0\nsynthetic_b2c1", sink.getvalue())


if __name__ == "__main__":
    unittest.main()
