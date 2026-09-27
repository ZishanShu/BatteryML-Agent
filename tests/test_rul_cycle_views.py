"""The result-schema layer, and the one condition a RUL curve requires.

A "remaining useful life vs cycle" curve is a scientific claim: that the run
estimated remaining life for the same cell at several *observed* cycles. Most
BatteryML lifetime models do not — they produce one number per cell from its
early cycles — and for those, a curve would be a picture of something that
was never measured.

So the rule this file enforces is:

    Cycle-wise RUL curves are shown only for artifacts that contain genuine
    per-cell, per-cycle RUL estimates.

and, just as importantly, the ways of getting that wrong:

* more rows than cells is **not** evidence (a whole-cell run writes one index
  record per cell, and repeated rows can exist without cycle identity);
* the presence of a ``sample_index`` is **not** evidence (every run writes
  one now, with ``cycle_index = -1`` for a whole-cell row);
* the model's class name is **not** evidence (``RidgeRULPredictor`` trains
  the SOH route);
* an SOH artifact is structurally identical and must never be re-read as RUL.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_rul_cycle_views -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent import result_views as rv  # noqa: E402
from tests.fixtures import synthetic_result_artifacts as fx  # noqa: E402


class TestCaseOneScalarRul(unittest.TestCase):
    """10 cells, 10 predictions -> RUL_SCALAR, no cycle curve."""

    def test_it_is_classified_as_scalar_rul(self) -> None:
        self.assertEqual(rv.detect_schema(fx.scalar_rul_payload()), rv.RUL_SCHEMA)

    def test_the_whole_cell_sentinel_is_not_read_as_a_cycle(self) -> None:
        evidence = rv.schema_evidence(fx.scalar_rul_payload())
        self.assertTrue(evidence["has_index"], "the fixture must carry an index")
        self.assertTrue(evidence["whole_cell_index"])
        self.assertEqual(evidence["max_cycles_per_cell"], 0)
        self.assertFalse(evidence["cycle_level"])

    def test_it_does_not_support_a_cycle_curve(self) -> None:
        view = rv.build_rul_view(fx.scalar_rul_payload())
        self.assertTrue(view["available"])
        self.assertFalse(view["supports_cycle_curve"])
        self.assertEqual(view["count"], 10)

    def test_the_cycle_view_refuses_it_and_says_why(self) -> None:
        curve = rv.build_rul_cycle_view(fx.scalar_rul_payload())
        self.assertFalse(curve["available"])
        self.assertEqual(curve["cells"], [])
        self.assertEqual(curve["reason"], rv.RUL_SCALAR_CURVE_NOTE)

    def test_the_parity_residual_and_table_views_all_work(self) -> None:
        """A scalar RUL run loses nothing: only the curve is absent."""
        view = rv.build_rul_view(fx.scalar_rul_payload())
        self.assertEqual(len(view["observed"]), 10)
        self.assertEqual(len(view["predicted"]), 10)
        self.assertEqual(len(view["residuals"]), 10)
        self.assertEqual(len(rv.prediction_table(view)), 10)

    def test_an_artifact_with_no_metadata_at_all_is_still_scalar_rul(self) -> None:
        payload = fx.scalar_rul_payload_without_metadata()
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])


class TestCaseTwoCyclewiseRul(unittest.TestCase):
    """Two cells at cycles 50/100/150 -> RUL_CYCLEWISE, curve available."""

    def setUp(self) -> None:
        self.payload = fx.cyclewise_rul_payload()

    def test_it_is_classified_as_cycle_wise_rul(self) -> None:
        self.assertEqual(rv.detect_schema(self.payload), rv.RUL_CYCLEWISE_SCHEMA)

    def test_the_evidence_is_the_cycle_identity_not_the_row_count(self) -> None:
        evidence = rv.schema_evidence(self.payload)
        self.assertEqual(evidence["task_family"], "RUL")
        self.assertTrue(evidence["has_index"])
        self.assertFalse(evidence["whole_cell_index"])
        self.assertTrue(evidence["repeated_cells"])
        self.assertEqual(evidence["max_cycles_per_cell"], 3)
        self.assertTrue(evidence["cycle_level"])

    def test_the_curve_is_available_with_one_entry_per_cell(self) -> None:
        curve = rv.build_rul_cycle_view(self.payload)
        self.assertTrue(curve["available"], curve["reason"])
        self.assertEqual(curve["n_cells"], 2)
        self.assertEqual(len(curve["cells"]), 2)
        self.assertEqual(curve["count"], 6)
        self.assertEqual(
            [cell["cell_id"] for cell in curve["cells"]],
            ["synthetic_cell_A", "synthetic_cell_B"],
        )

    def test_cycles_are_sorted_ascending_within_each_cell(self) -> None:
        """The fixture stores cell_B out of order on purpose."""
        curve = rv.build_rul_cycle_view(self.payload)
        for cell in curve["cells"]:
            with self.subTest(cell=cell["cell_id"]):
                self.assertEqual(cell["cycles"], sorted(cell["cycles"]))
                self.assertEqual(cell["cycles"], [50, 100, 150])

    def test_reordering_carries_the_values_with_the_cycles(self) -> None:
        """Sorting must move observed and predicted too, not just the x axis."""
        curve = rv.build_rul_cycle_view(self.payload)
        by_cell = {cell["cell_id"]: cell for cell in curve["cells"]}

        rows = list(zip(fx.CYCLEWISE_PAIRS, fx.CYCLEWISE_OBSERVED, fx.CYCLEWISE_PREDICTED))
        for cell_id, cell in by_cell.items():
            expected = sorted(
                [(cycle, obs, pred) for (name, cycle), obs, pred in rows if name == cell_id]
            )
            with self.subTest(cell=cell_id):
                self.assertEqual(cell["cycles"], [row[0] for row in expected])
                self.assertEqual(cell["observed"], [row[1] for row in expected])
                self.assertEqual(cell["predicted"], [row[2] for row in expected])

    def test_no_series_spans_two_cells(self) -> None:
        """Structural: there is no flat series a caller could draw across cells.

        Each cell's points live in their own entry, so joining two cells would
        take deliberate effort rather than being the default.
        """
        curve = rv.build_rul_cycle_view(self.payload)
        self.assertNotIn("cycles", curve, "a flat cycle series would invite one line")
        self.assertNotIn("observed", curve)
        self.assertNotIn("predicted", curve)
        for cell in curve["cells"]:
            with self.subTest(cell=cell["cell_id"]):
                self.assertEqual(len(cell["cycles"]), len(cell["observed"]))
                self.assertEqual(len(cell["cycles"]), len(cell["predicted"]))

    def test_the_values_are_the_saved_arrays_and_nothing_is_reconstructed(self) -> None:
        """D6: no ``true_rul = final_lifetime - cycle`` anywhere.

        Every observed value in the curve appears in the artifact's own saved
        labels, and every predicted value in its saved predictions.
        """
        curve = rv.build_rul_cycle_view(self.payload)
        stored_observed = set(fx.CYCLEWISE_OBSERVED)
        stored_predicted = set(fx.CYCLEWISE_PREDICTED)
        for cell in curve["cells"]:
            for value in cell["observed"]:
                self.assertIn(value, stored_observed)
            for value in cell["predicted"]:
                self.assertIn(value, stored_predicted)

    def test_an_artifact_with_no_observed_labels_draws_nothing(self) -> None:
        payload = fx.cyclewise_rul_payload()
        payload.pop("data")
        curve = rv.build_rul_cycle_view(payload)
        self.assertFalse(curve["available"])
        self.assertIn("never reconstructed", curve["reason"])
        self.assertEqual(curve["cells"], [])

    def test_parity_and_table_still_apply(self) -> None:
        view = rv.build_rul_view(self.payload)
        self.assertTrue(view["available"])
        self.assertTrue(view["supports_cycle_curve"])
        self.assertEqual(view["count"], 6)
        # A cycle-wise table names the cell and the cycle: a bare cell id
        # would repeat three times and identify nothing.
        self.assertEqual(view["sample_ids"][0], "synthetic_cell_A @ cycle 50")


class TestCaseThreeRepeatedRowsWithoutCycleIdentity(unittest.TestCase):
    """D4: more rows than cells is not evidence of anything."""

    def test_repeated_whole_cell_rows_are_not_cycle_wise(self) -> None:
        payload = fx.repeated_rows_without_cycle_identity_payload()
        self.assertNotEqual(rv.detect_schema(payload), rv.RUL_CYCLEWISE_SCHEMA)
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)

    def test_the_shape_alone_would_have_fooled_a_shape_only_test(self) -> None:
        """Non-vacuity: the rejected fixture really does look cycle-wise."""
        payload = fx.repeated_rows_without_cycle_identity_payload()
        evidence = rv.schema_evidence(payload)
        self.assertTrue(evidence["has_index"], "an index exists")
        self.assertTrue(evidence["repeated_cells"], "cell ids repeat")
        self.assertGreater(evidence["rows"], 2, "there are more rows than cells")
        # ...and yet:
        self.assertFalse(evidence["cycle_level"])
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])

    def test_one_cycle_per_cell_is_not_a_curve(self) -> None:
        payload = fx.single_cycle_per_cell_payload()
        evidence = rv.schema_evidence(payload)
        self.assertFalse(evidence["whole_cell_index"], "the cycles are real")
        self.assertEqual(evidence["max_cycles_per_cell"], 1)
        self.assertFalse(evidence["cycle_level"])
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])

    def test_a_malformed_index_record_makes_the_index_unusable(self) -> None:
        payload = fx.cyclewise_rul_payload()
        payload["sample_index"][2] = {"cell_id": "synthetic_cell_A"}
        self.assertIsNone(rv.cycle_sample_index(payload))
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)

    def test_an_index_of_the_wrong_length_is_refused(self) -> None:
        payload = fx.cyclewise_rul_payload()
        payload["sample_index"] = payload["sample_index"][:3]
        self.assertFalse(rv.schema_evidence(payload)["has_index"])
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)


class TestCaseFourSohIsNeverRul(unittest.TestCase):
    """An SOH cycle-level artifact must never become RUL_CYCLEWISE."""

    def test_a_cycle_level_soh_artifact_is_soh(self) -> None:
        self.assertEqual(rv.detect_schema(fx.soh_cyclewise_payload()), rv.SOH_SCHEMA)

    def test_it_is_structurally_identical_to_the_cycle_wise_rul_case(self) -> None:
        """Non-vacuity: only the recorded task separates them.

        Both have two cells, three real cycles each, repeated ids and one
        scalar per row. A classifier reading anything but the task would put
        them in the same box.
        """
        soh = rv.schema_evidence(fx.soh_cyclewise_payload())
        rul = rv.schema_evidence(fx.cyclewise_rul_payload())
        for key in ("has_index", "whole_cell_index", "repeated_cells",
                    "max_cycles_per_cell", "cycle_level", "rows"):
            with self.subTest(key=key):
                self.assertEqual(soh[key], rul[key])
        self.assertNotEqual(soh["task_family"], rul["task_family"])

    def test_the_rul_cycle_view_refuses_an_soh_artifact(self) -> None:
        curve = rv.build_rul_cycle_view(fx.soh_cyclewise_payload())
        self.assertFalse(curve["available"])
        self.assertEqual(curve["cells"], [])
        self.assertIn("remaining-useful-life", curve["reason"])

    def test_the_annotator_alone_identifies_soh_when_no_task_is_declared(self) -> None:
        payload = fx.soh_cyclewise_payload_without_declared_task()
        self.assertNotIn("task", payload["task_metadata"])
        self.assertEqual(rv.detect_schema(payload), rv.SOH_SCHEMA)
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])

    def test_a_whole_cell_soh_result_is_neither_cycle_wise_nor_rul(self) -> None:
        payload = fx.soh_scalar_payload()
        self.assertEqual(rv.detect_schema(payload), rv.SOH_SCALAR_SCHEMA)
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])
        # Parity, residual and table still apply: it is one scalar per row.
        view = rv.build_rul_view(payload)
        self.assertTrue(view["available"])
        self.assertFalse(view["supports_cycle_curve"])

    def test_the_model_class_name_is_never_consulted(self) -> None:
        """``RidgeRULPredictor`` trains the SOH route; the RUL is naming only."""
        payload = fx.soh_cyclewise_payload()
        payload["model_name"] = "RidgeRULPredictor"
        payload["task_metadata"]["model"] = "XGBoostRULPredictor"
        self.assertEqual(rv.detect_schema(payload), rv.SOH_SCHEMA)


class TestCaseFiveTrajectoryIsNeverRulCyclewise(unittest.TestCase):
    def test_a_sequence_per_cell_is_a_trajectory(self) -> None:
        self.assertEqual(rv.detect_schema(fx.trajectory_payload()), rv.TRAJECTORY_SCHEMA)

    def test_rul_semantics_plus_a_cycle_index_do_not_make_it_cycle_wise(self) -> None:
        """The adversarial case: every non-shape signal points the wrong way."""
        payload = fx.trajectory_payload_with_cycle_index()
        self.assertEqual(payload["task_metadata"]["task"], fx.TASK_RUL)
        self.assertIn("sample_index", payload)
        self.assertEqual(rv.detect_schema(payload), rv.TRAJECTORY_SCHEMA)
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])

    def test_a_trajectory_gets_no_parity_view_either(self) -> None:
        view = rv.build_rul_view(fx.trajectory_payload())
        self.assertFalse(view["available"])
        self.assertEqual(view["observed"], [])


class TestTheCategoriesStaySeparate(unittest.TestCase):
    """Every category is reachable, and no two share a value."""

    ALL = (
        rv.RUL_SCHEMA,
        rv.RUL_CYCLEWISE_SCHEMA,
        rv.SOH_SCHEMA,
        rv.SOH_SCALAR_SCHEMA,
        rv.TRAJECTORY_SCHEMA,
        rv.UNKNOWN_SCHEMA,
    )

    def test_the_names_are_distinct(self) -> None:
        self.assertEqual(len(set(self.ALL)), len(self.ALL))

    def test_each_category_is_produced_by_some_artifact(self) -> None:
        produced = {
            rv.detect_schema(fx.scalar_rul_payload()),
            rv.detect_schema(fx.cyclewise_rul_payload()),
            rv.detect_schema(fx.soh_cyclewise_payload()),
            rv.detect_schema(fx.soh_scalar_payload()),
            rv.detect_schema(fx.trajectory_payload()),
            rv.detect_schema({"prediction": "not a series"}),
        }
        self.assertEqual(produced, set(self.ALL))

    def test_an_empty_or_broken_payload_is_unknown(self) -> None:
        for payload in (None, {}, {"prediction": None}, {"prediction": object()}, []):
            with self.subTest(payload=payload):
                self.assertEqual(rv.detect_schema(payload), rv.UNKNOWN_SCHEMA)

    def test_an_unknown_artifact_gets_no_view_at_all(self) -> None:
        payload = {"prediction": "not a series"}
        self.assertFalse(rv.build_rul_view(payload)["available"])
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])
        self.assertFalse(rv.build_soh_view(payload)["available"])
        self.assertFalse(rv.build_trajectory_view(payload)["available"])


class TestTheProductionRulArtifactAudit(unittest.TestCase):
    """Which real RUL models emit which schema — checked, not assumed.

    The answer today is that **none** of them emits per-cell, per-cycle RUL
    estimates, so the cycle-wise view is implemented but dormant against real
    artifacts. That is a fact about the code, and it is asserted here so that
    the day someone adds a cycle-level RUL annotator, this test fails and
    says the feature can come out of dormancy — rather than the claim
    quietly going stale in a document.
    """

    def test_every_rul_config_uses_the_whole_cell_annotator(self) -> None:
        from batteryml_web.core.config_registry import get_config_registry

        registry = get_config_registry(str(REPO_ROOT))
        entries = registry.get_entries(task="RUL prediction")
        self.assertGreater(len(entries), 20, "the RUL catalog is missing")
        annotators = set()
        for entry in entries:
            data = (entry.metadata or {}).get("yaml") or {}
            annotators.add(str((data.get("label") or {}).get("name")))
        self.assertEqual(
            annotators, {"RULLabelAnnotator"},
            "a RUL config uses an annotator this audit has not accounted for",
        )

    def test_the_rul_annotator_is_not_cycle_level(self) -> None:
        """The structural reason, one level below the configs.

        ``Task.is_cycle_level`` is ``isinstance(annotator,
        BaseCycleLabelAnnotator)``. A whole-cell annotator therefore takes the
        ``SampleIndex.for_whole_cells`` branch, which stamps every row with
        ``cycle_index = -1`` — so a RUL run cannot produce a cycle-level index
        even in principle.
        """
        from batteryml.label.base import BaseCycleLabelAnnotator
        from batteryml.label.rul import RULLabelAnnotator

        self.assertFalse(issubclass(RULLabelAnnotator, BaseCycleLabelAnnotator))

    def test_the_only_cycle_level_annotator_is_the_soh_one(self) -> None:
        import batteryml.label  # noqa: F401  (registers the annotators)
        from batteryml.builders import LABEL_ANNOTATORS
        from batteryml.label.base import BaseCycleLabelAnnotator

        cycle_level = sorted(
            name
            for name, cls in LABEL_ANNOTATORS.class_mapping.items()
            if isinstance(cls, type) and issubclass(cls, BaseCycleLabelAnnotator)
        )
        self.assertEqual(cycle_level, ["CycleSOHLabelAnnotator"])

    def test_the_whole_cell_sentinel_is_the_value_this_module_expects(self) -> None:
        """The one number the schema layer mirrors from the training stack."""
        from batteryml.data.sample_index import WHOLE_CELL_CYCLE

        self.assertEqual(rv.WHOLE_CELL_CYCLE, WHOLE_CELL_CYCLE)
        self.assertEqual(fx.WHOLE_CELL_CYCLE, WHOLE_CELL_CYCLE)

    def test_a_real_whole_cell_index_classifies_as_scalar_rul(self) -> None:
        """Built through the training stack's own SampleIndex, not by hand."""
        from batteryml.data.sample_index import SampleIndex

        cells = [f"synthetic_cell_{i:02d}" for i in range(5)]
        payload = {
            "prediction": [900.0, 870.0, 840.0, 810.0, 780.0],
            "sample_index": SampleIndex.for_whole_cells(cells).to_records(),
            "task_metadata": {"label_annotator": "RULLabelAnnotator"},
        }
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)
        self.assertFalse(rv.build_rul_cycle_view(payload)["available"])


class TestEveryDispatchSiteKnowsTheNewCategories(unittest.TestCase):
    """A module that branches on a schema must know all of them.

    A schema no dispatch site recognises falls through to "cannot be
    visualized", which is how a working result silently loses its figures.
    """

    SITES = (
        Path("batteryml_web") / "ui" / "run_history.py",
        Path("batteryml_web") / "ui" / "agent_workflow.py",
        Path("batteryml_web") / "agent" / "artifact_preview.py",
    )

    def test_each_site_handles_the_scalar_family_and_the_cycle_wise_case(self) -> None:
        for site in self.SITES:
            text = (REPO_ROOT / site).read_text(encoding="utf-8")
            with self.subTest(module=site.name):
                self.assertIn("SCALAR_SCHEMAS", text)
                self.assertIn("RUL_CYCLEWISE_SCHEMA", text)
                self.assertIn("SOH_SCHEMA", text)

    def test_the_preview_describes_every_category_it_can_produce(self) -> None:
        from batteryml_web.agent import artifact_preview as ap

        for schema in (rv.RUL_SCHEMA, rv.RUL_CYCLEWISE_SCHEMA,
                       rv.SOH_SCHEMA, rv.SOH_SCALAR_SCHEMA,
                       rv.TRAJECTORY_SCHEMA):
            with self.subTest(schema=schema):
                self.assertIn(schema, ap._OUTPUT_KIND)
                self.assertTrue(ap._OUTPUT_KIND[schema])

    def test_the_preview_accepts_each_fixture_and_labels_it(self) -> None:
        from batteryml_web.agent import artifact_preview as ap

        cases = {
            rv.RUL_SCHEMA: fx.scalar_rul_payload(),
            rv.RUL_CYCLEWISE_SCHEMA: fx.cyclewise_rul_payload(),
            rv.SOH_SCHEMA: fx.soh_cyclewise_payload(),
            rv.SOH_SCALAR_SCHEMA: fx.soh_scalar_payload(),
            rv.TRAJECTORY_SCHEMA: fx.trajectory_payload(),
        }
        for schema, payload in cases.items():
            with self.subTest(schema=schema):
                self.assertTrue(ap._is_prediction_payload(payload))
                self.assertEqual(rv.detect_schema(payload), schema)

    def test_load_view_offers_the_curve_key_for_every_artifact(self) -> None:
        """The key always exists; only its ``available`` flag varies."""
        import pickle
        import tempfile

        for payload, expected in (
            (fx.scalar_rul_payload(), False),
            (fx.cyclewise_rul_payload(), True),
            (fx.soh_cyclewise_payload(), False),
        ):
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
                pickle.dump(payload, handle)
                path = handle.name
            try:
                views = rv.load_view(path)
                with self.subTest(schema=views["schema"]):
                    self.assertIn("rul_cycles", views)
                    self.assertEqual(views["rul_cycles"]["available"], expected)
            finally:
                Path(path).unlink(missing_ok=True)


class TestNoModelNameDecidesAnything(unittest.TestCase):
    """D4: the artifact schema decides, never the model that wrote it."""

    def test_the_schema_module_names_no_model(self) -> None:
        source = (
            REPO_ROOT / "batteryml_web" / "agent" / "result_views.py"
        ).read_text(encoding="utf-8")
        # Prose may discuss a model; a comparison against one is the defect.
        for model in ("CPTransformer", "BatLiNet", "XGBoost", "Ridge", "BatteryLife"):
            for comparison in (f'== "{model}"', f"== '{model}'",
                               f'in ("{model}"', f"in ('{model}'"):
                with self.subTest(model=model, comparison=comparison):
                    self.assertNotIn(comparison, source)

    def test_the_same_numbers_classify_by_their_recorded_task(self) -> None:
        """One payload, two recorded tasks, two categories."""
        as_rul = fx.cyclewise_rul_payload()
        as_soh = fx.cyclewise_rul_payload()
        as_soh["task_metadata"] = {
            "task": fx.TASK_SOH, "label_annotator": "CycleSOHLabelAnnotator",
        }
        self.assertEqual(as_rul["prediction"], as_soh["prediction"])
        self.assertEqual(rv.detect_schema(as_rul), rv.RUL_CYCLEWISE_SCHEMA)
        self.assertEqual(rv.detect_schema(as_soh), rv.SOH_SCHEMA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
