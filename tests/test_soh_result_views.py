"""The SOH result view: cycle-level identity, and an honest range check.

Two things are being pinned here.

**A cycle-level result must not be read as a per-cell one.** An SOH artifact
holds one scalar per *cycle sample*, which is structurally identical to an
RUL artifact's one scalar per cell. The only thing that tells them apart is
the ``sample_index`` the evaluation step records beside the predictions, so
the schema is decided from that and never from the numbers. Read as RUL, an
SOH run's table labelled 6,116 rows by position and its figures said
"Observed" where they meant "Observed SOH (ratio)".

**The range check must not become a correction.** The first real MATR SOH run
estimated values from -4.34 to 2.06 — impossible for a ratio. Clipping them
would have made the figures look reasonable while the reported metrics stayed
what they were, so the platform counts them and says so, and the arrays it
plots are the arrays the metrics came from, value for value.

No Streamlit, no torch, no data: these are pure functions over a payload of
the shape ``Pipeline.evaluate`` writes.

Run from the repository root::

    conda run -n batteryml python -m unittest tests.test_soh_result_views -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent import result_views as rv  # noqa: E402


class _TestData:
    def __init__(self, label):
        self.label = label


class _Bundle:
    """The saved DataBundle, as much of it as the view layer touches."""

    def __init__(self, label, transformation=None, cell_ids=None):
        self.test_data = _TestData(label)
        self.label_transformation = transformation
        if cell_ids is not None:
            self.test_cell_ids = cell_ids


def _payload(
    *,
    predicted,
    observed,
    index=None,
    transformation=None,
    cell_ids=None,
):
    payload = {
        "prediction": list(predicted),
        "data": _Bundle(list(observed), transformation, cell_ids),
    }
    if index is not None:
        payload["sample_index"] = list(index)
    return payload


def _cycle_index(pairs):
    return [{"cell_id": cell, "cycle_index": cycle} for cell, cycle in pairs]


#: Two cells, three observed cycles each. Deliberately includes one estimate
#: below 0 and one above 1.1, because a clean series cannot show that the
#: diagnostic notices anything.
_PAIRS = [("b1c0", 10), ("b1c0", 15), ("b1c0", 20),
          ("b1c1", 10), ("b1c1", 15), ("b1c1", 20)]
_OBSERVED = [1.00, 0.97, 0.94, 0.99, 0.95, 0.90]
_PREDICTED = [0.98, 0.99, -0.20, 1.02, 1.30, 0.88]


def _soh_payload():
    return _payload(
        predicted=_PREDICTED, observed=_OBSERVED, index=_cycle_index(_PAIRS)
    )


class TestSchemaDetection(unittest.TestCase):
    """Structural, from what the artifact records about itself."""

    def test_a_cycle_indexed_scalar_result_is_soh(self) -> None:
        self.assertEqual(rv.detect_schema(_soh_payload()), rv.SOH_SCHEMA)

    def test_the_same_numbers_without_an_index_stay_rul(self) -> None:
        """Non-vacuity: the index is what decides, not the values."""
        payload = _payload(predicted=_PREDICTED, observed=_OBSERVED)
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)

    def test_an_index_of_the_wrong_length_does_not_claim_a_cycle_result(self) -> None:
        payload = _payload(
            predicted=_PREDICTED,
            observed=_OBSERVED,
            index=_cycle_index(_PAIRS[:3]),
        )
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)

    def test_a_malformed_index_record_makes_the_whole_index_unusable(self) -> None:
        payload = _soh_payload()
        payload["sample_index"][2] = {"cell_id": "b1c0"}  # no cycle_index
        self.assertIsNone(rv.cycle_sample_index(payload))
        self.assertEqual(rv.detect_schema(payload), rv.RUL_SCHEMA)

    def test_the_rul_view_refuses_a_cycle_level_artifact(self) -> None:
        view = rv.build_rul_view(_soh_payload())
        self.assertFalse(view["available"])
        self.assertEqual(view["observed"], [])


class TestSohView(unittest.TestCase):
    def setUp(self) -> None:
        self.view = rv.build_soh_view(_soh_payload())

    def test_every_row_names_its_cell_and_its_cycle(self) -> None:
        self.assertTrue(self.view["available"])
        self.assertEqual(self.view["count"], 6)
        self.assertEqual(self.view["n_cells"], 2)
        self.assertEqual(self.view["cell_ids"][:3], ["b1c0", "b1c0", "b1c0"])
        self.assertEqual(self.view["cycle_indices"][:3], [10, 15, 20])
        self.assertEqual(self.view["sample_ids"][0], "b1c0 @ cycle 10")

    def test_the_stored_values_are_passed_through_untouched(self) -> None:
        """The whole point: the view plots what the metrics were computed on."""
        self.assertEqual(self.view["observed"], _OBSERVED)
        self.assertEqual(self.view["predicted"], _PREDICTED)
        # Including the impossible ones.
        self.assertIn(-0.20, self.view["predicted"])
        self.assertIn(1.30, self.view["predicted"])

    def test_residuals_are_a_display_of_two_stored_values(self) -> None:
        for observed, predicted, residual in zip(
            self.view["observed"], self.view["predicted"], self.view["residuals"]
        ):
            self.assertAlmostEqual(residual, predicted - observed)

    def test_a_per_cell_artifact_is_refused_rather_than_reinterpreted(self) -> None:
        view = rv.build_soh_view(
            _payload(predicted=_PREDICTED, observed=_OBSERVED)
        )
        self.assertFalse(view["available"])
        self.assertIn("no per-cycle sample index", view["reason"])

    def test_a_missing_ground_truth_draws_nothing(self) -> None:
        payload = _soh_payload()
        payload["data"].test_data.label = None
        view = rv.build_soh_view(payload)
        self.assertFalse(view["available"])
        self.assertIn("no observed SOH labels", view["reason"])
        self.assertEqual(view["predicted"], [])

    def test_the_label_transformation_is_inverted_and_said_so(self) -> None:
        """Axes must be in the same space as the reported metrics."""

        class _Doubling:
            def inverse_transform(self, values):
                return [value * 2 for value in values]

        payload = _payload(
            predicted=_PREDICTED,
            observed=_OBSERVED,
            index=_cycle_index(_PAIRS),
            transformation=_Doubling(),
        )
        view = rv.build_soh_view(payload)
        self.assertTrue(view["inverse_transformed"])
        self.assertEqual(view["note"], rv.TRANSFORM_NOTE)
        self.assertEqual(view["observed"][0], _OBSERVED[0] * 2)


class TestRangeDiagnostics(unittest.TestCase):
    """Display only. It counts; it never changes anything."""

    def test_impossible_estimates_are_counted_not_removed(self) -> None:
        view = rv.build_soh_view(_soh_payload())
        summary = view["range"]
        self.assertEqual(summary["count"], 6)
        self.assertEqual(summary["below_zero"], 1)       # -0.20
        self.assertEqual(summary["above_one"], 2)        # 1.02 and 1.30
        self.assertEqual(summary["above_implausible"], 1)  # 1.30 only
        self.assertEqual(summary["out_of_range"], 3)
        self.assertAlmostEqual(summary["min"], -0.20)
        self.assertAlmostEqual(summary["max"], 1.30)
        self.assertAlmostEqual(summary["fraction_out_of_range"], 0.5)
        # The counted values are still all there, in their original order.
        self.assertEqual(view["predicted"], _PREDICTED)

    def test_a_clean_run_reports_zero_rather_than_nothing(self) -> None:
        summary = rv.soh_range_diagnostics([1.0, 0.95, 0.5, 0.0001])
        self.assertEqual(summary["out_of_range"], 0)
        self.assertEqual(summary["below_zero"], 0)
        self.assertEqual(summary["above_one"], 0)
        self.assertEqual(summary["fraction_out_of_range"], 0.0)

    def test_the_boundaries_are_the_documented_ones(self) -> None:
        """1.0 is healthy, not "above one"; 1.1 is the implausibility line."""
        summary = rv.soh_range_diagnostics(
            [rv.SOH_HEALTHY, rv.SOH_IMPLAUSIBLE_HIGH, 0.0]
        )
        self.assertEqual(summary["above_one"], 1)          # 1.1 only
        self.assertEqual(summary["above_implausible"], 0)  # 1.1 is not above 1.1
        self.assertEqual(summary["below_zero"], 0)         # 0.0 is not below 0

    def test_an_empty_series_has_no_extremes_rather_than_a_placeholder(self) -> None:
        summary = rv.soh_range_diagnostics([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["min"])
        self.assertIsNone(summary["max"])

    def test_the_note_says_nothing_was_altered(self) -> None:
        summary = rv.soh_range_diagnostics(_PREDICTED)
        self.assertIn("nothing is clipped", summary["note"].lower())


class TestLargestResiduals(unittest.TestCase):
    def test_the_worst_misses_come_first_and_name_their_sample(self) -> None:
        view = rv.build_soh_view(_soh_payload())
        rows = rv.largest_residuals(view, limit=3)
        self.assertEqual(len(rows), 3)
        magnitudes = [abs(row["Residual"]) for row in rows]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        # The -0.20 estimate against an observed 0.94 is the worst of them.
        self.assertEqual(rows[0]["Sample"], "b1c0 @ cycle 20")
        self.assertAlmostEqual(rows[0]["Observed"], 0.94)
        self.assertAlmostEqual(rows[0]["Predicted"], -0.20)

    def test_the_rows_are_stored_values_not_recomputed_ones(self) -> None:
        view = rv.build_soh_view(_soh_payload())
        for row in rv.largest_residuals(view, limit=6):
            position = view["sample_ids"].index(row["Sample"])
            self.assertEqual(row["Observed"], view["observed"][position])
            self.assertEqual(row["Predicted"], view["predicted"][position])
            self.assertEqual(row["Residual"], view["residuals"][position])

    def test_an_unavailable_view_yields_no_rows(self) -> None:
        view = rv.build_soh_view(_payload(predicted=[1.0], observed=[1.0]))
        self.assertEqual(rv.largest_residuals(view), [])


class TestSchemaIsHandledEverywhereItIsProduced(unittest.TestCase):
    """A new schema that only half the dispatch sites know is a regression."""

    def test_load_view_exposes_an_soh_view(self) -> None:
        import inspect

        source = inspect.getsource(rv.load_view)
        self.assertIn('"soh"', source)

    def test_every_schema_dispatch_site_handles_soh(self) -> None:
        sites = [
            REPO_ROOT / "batteryml_web" / "ui" / "run_history.py",
            REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py",
            REPO_ROOT / "batteryml_web" / "agent" / "artifact_preview.py",
        ]
        # The RUL family may be named directly or through the tuple that
        # contains it. Both are the same guarantee, and this proves it
        # rather than assuming it: a site that dispatches on SCALAR_SCHEMAS
        # is handling RUL_SCHEMA.
        self.assertIn(rv.RUL_SCHEMA, rv.SCALAR_SCHEMAS)
        rul_family = ("RUL_SCHEMA", "SCALAR_SCHEMAS")

        for path in sites:
            text = path.read_text(encoding="utf-8")
            with self.subTest(module=path.name):
                # Every module that branches on the RUL family must also
                # know what an SOH artifact is, or an SOH run silently
                # falls through to "cannot be visualized".
                self.assertTrue(
                    any(name in text for name in rul_family),
                    f"{path.name} names neither {' nor '.join(rul_family)}",
                )
                self.assertIn("SOH_SCHEMA", text)

    def test_the_artifact_preview_recognises_an_soh_payload(self) -> None:
        from batteryml_web.agent import artifact_preview as ap

        self.assertTrue(ap._is_prediction_payload(_soh_payload()))
        self.assertEqual(ap._OUTPUT_KIND[rv.SOH_SCHEMA],
                         "Scalar prediction per cycle of each cell (SOH ratio)")


if __name__ == "__main__":
    unittest.main()
