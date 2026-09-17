"""The SOH label, the cycle-level sample contract, and the leakage invariant.

These are the scientific tests for the SOH route's core. They are about what
the numbers *mean*, not about whether the code runs: every SOH ratio asserted
here is an exact value computed by hand from a synthetic cell whose capacities
are halves and quarters, so an equality check is meaningful.

What is established:

* **Reference capacity** is one convention for a whole run, applied to every
  cell. The behaviour it replaces — nominal capacity when the cell carried
  one, otherwise that cell's first cycle — is proved gone, because it labelled
  two cells in the same dataset under two different definitions and recorded
  neither.
* **Cycle-level samples** carry their own identity: many rows per cell, each
  tagged with the cell and the observed cycle, in deterministic order, with
  feature rows, label rows and identity rows the same length.
* **Cells are split before they are expanded into cycles.** The test builds
  the leakage-prone alternative on purpose and proves the validator rejects
  it, rather than trusting the pipeline's current ordering to stay correct.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_soh_core -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from batteryml.data.sample_index import (  # noqa: E402
    WHOLE_CELL_CYCLE,
    SampleIndex,
    SampleIndexError,
    assert_disjoint_cells,
)
from batteryml.label.soh import (  # noqa: E402
    DEFAULT_REFERENCE_CAPACITY_MODE,
    EXCLUSION_INVALID_CAPACITY,
    EXCLUSION_NO_REFERENCE,
    REFERENCE_CAPACITY_MODES,
    CycleSOHLabelAnnotator,
    SOHLabelAnnotator,
    SOHReferenceError,
    cycle_discharge_capacity,
    reference_capacity,
    sampled_cycle_numbers,
)
from tests.fixtures.synthetic_soh_cells import (  # noqa: E402
    SYNTHETIC_CLEAN_CAPACITIES,
    cell_ids,
    cell_with_invalid_cycles,
    cell_without_reference,
    clean_cell,
    cohort,
    make_cell,
    nominal_cell,
)


class TestReferenceCapacity(unittest.TestCase):
    """One convention per run, named in the config, applied to every cell."""

    def test_the_default_is_the_first_valid_discharge_capacity(self) -> None:
        self.assertEqual(
            DEFAULT_REFERENCE_CAPACITY_MODE, "first_valid_discharge_capacity"
        )
        q_ref, reason = reference_capacity(clean_cell())
        self.assertEqual(q_ref, 1.0)
        self.assertIsNone(reason)

    def test_the_first_valid_capacity_skips_invalid_leading_cycles(self) -> None:
        cell = make_cell("synthetic_late_start", [None, 0.0, -1.0, 0.8, 0.4])
        q_ref, reason = reference_capacity(cell, "first_valid_discharge_capacity")
        self.assertAlmostEqual(q_ref, 0.8)
        self.assertIsNone(reason)

    def test_nominal_capacity_is_a_different_convention(self) -> None:
        cell = nominal_cell(nominal=2.0)
        first_valid, _ = reference_capacity(cell, "first_valid_discharge_capacity")
        nominal, _ = reference_capacity(cell, "nominal_capacity")
        self.assertEqual(first_valid, 1.0)
        self.assertEqual(nominal, 2.0)
        # The same cell, two conventions, labels differing by a factor of two.
        # This is exactly why the mode may not be chosen per cell.
        self.assertNotEqual(first_valid, nominal)

    def test_the_old_per_cell_fallback_is_gone(self) -> None:
        """The defect this milestone removes, stated as a test.

        The previous annotator used nominal capacity when a cell carried one
        and the first cycle otherwise. Under the default mode, a cell that
        carries a nominal capacity and one that does not must now be labelled
        the same way.
        """
        with_nominal = nominal_cell(nominal=2.0)
        without_nominal = clean_cell()
        a, _ = reference_capacity(with_nominal, DEFAULT_REFERENCE_CAPACITY_MODE)
        b, _ = reference_capacity(without_nominal, DEFAULT_REFERENCE_CAPACITY_MODE)
        self.assertEqual(a, b, "the nominal-capacity cell was treated differently")

        annotator = CycleSOHLabelAnnotator()
        labels_with, _ = annotator.process_cell_cycles(with_nominal)
        labels_without, _ = annotator.process_cell_cycles(without_nominal)
        self.assertEqual(labels_with.tolist(), labels_without.tolist())

    def test_nominal_mode_excludes_rather_than_switching(self) -> None:
        """A cell that cannot satisfy the mode is excluded, not relabelled."""
        q_ref, reason = reference_capacity(clean_cell(), "nominal_capacity")
        self.assertNotEqual(q_ref, q_ref, "expected NaN for a missing nominal capacity")
        self.assertEqual(reason, EXCLUSION_NO_REFERENCE)

    def test_first_n_valid_mean(self) -> None:
        cell = make_cell("synthetic_mean", [1.0, 0.5, 0.25])
        q_ref, _ = reference_capacity(cell, "first_n_valid_mean", reference_n_cycles=2)
        self.assertAlmostEqual(q_ref, 0.75)

    def test_a_cell_with_no_usable_capacity_is_reported(self) -> None:
        q_ref, reason = reference_capacity(cell_without_reference())
        self.assertNotEqual(q_ref, q_ref)
        self.assertEqual(reason, EXCLUSION_NO_REFERENCE)

    def test_an_unknown_mode_is_refused_loudly(self) -> None:
        with self.assertRaises(SOHReferenceError) as ctx:
            reference_capacity(clean_cell(), "whatever_is_convenient")
        self.assertIn("reference_capacity_mode", str(ctx.exception))
        for mode in REFERENCE_CAPACITY_MODES:
            self.assertIn(mode, str(ctx.exception))

    def test_capacity_validity_rules(self) -> None:
        from tests.fixtures.synthetic_soh_cells import make_cycle

        self.assertEqual(cycle_discharge_capacity(make_cycle(1, 0.8)), 0.8)
        for bad in (None, 0.0, -1.0):
            with self.subTest(capacity=bad):
                value = cycle_discharge_capacity(make_cycle(1, bad))
                self.assertNotEqual(value, value, "expected NaN")


class TestCycleSOHLabels(unittest.TestCase):
    """SOH(cell, t) = Q_discharge(t) / Q_ref, as an exact ratio."""

    def test_exact_label_values(self) -> None:
        labels, cycles = CycleSOHLabelAnnotator().process_cell_cycles(clean_cell())
        self.assertEqual(list(cycles), [1, 2, 3, 4])
        self.assertEqual(labels.tolist(), SYNTHETIC_CLEAN_CAPACITIES)
        self.assertEqual(labels.tolist(), [1.0, 0.75, 0.5, 0.25])

    def test_the_canonical_label_is_a_ratio_not_a_percentage(self) -> None:
        labels, _ = CycleSOHLabelAnnotator().process_cell_cycles(clean_cell())
        self.assertLessEqual(float(labels.max()), 1.0)
        self.assertGreater(float(labels.min()), 0.0)

    def test_a_ratio_above_one_is_kept_not_clipped(self) -> None:
        """Early-life capacity above the reference is a real measurement."""
        cell = make_cell("synthetic_rising", [1.0, 1.25, 0.5])
        labels, _ = CycleSOHLabelAnnotator().process_cell_cycles(cell)
        self.assertEqual(labels.tolist(), [1.0, 1.25, 0.5])

    def test_invalid_cycles_are_excluded_and_counted(self) -> None:
        annotator = CycleSOHLabelAnnotator()
        labels, cycles = annotator.process_cell_cycles(cell_with_invalid_cycles())
        # capacities were [1.0, None, 0.0, -0.5, 0.5]: three unusable.
        self.assertEqual(list(cycles), [1, 5])
        self.assertEqual(labels.tolist(), [1.0, 0.5])
        self.assertEqual(annotator.exclusions[EXCLUSION_INVALID_CAPACITY], 3)

    def test_nothing_is_interpolated_for_a_missing_cycle(self) -> None:
        annotator = CycleSOHLabelAnnotator()
        _, cycles = annotator.process_cell_cycles(cell_with_invalid_cycles())
        self.assertNotIn(2, cycles, "a missing cycle must be dropped, not filled in")

    def test_a_cell_without_a_reference_contributes_nothing(self) -> None:
        annotator = CycleSOHLabelAnnotator()
        labels, cycles = annotator.process_cell_cycles(cell_without_reference())
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(cycles), 0)
        self.assertEqual(annotator.exclusions[EXCLUSION_NO_REFERENCE], 1)

    def test_the_sample_policy_is_configuration_not_a_constant(self) -> None:
        cell = make_cell("synthetic_policy", [1.0 - 0.05 * i for i in range(20)])
        annotator = CycleSOHLabelAnnotator(min_cycle=5, max_cycle=12, cycle_stride=3)
        _, cycles = annotator.process_cell_cycles(cell)
        self.assertEqual(list(cycles), [5, 8, 11])
        self.assertEqual(annotator.sample_policy()["min_cycle"], 5)
        self.assertEqual(annotator.sample_policy()["cycle_stride"], 3)
        self.assertEqual(
            annotator.sample_policy()["reference_capacity_mode"],
            DEFAULT_REFERENCE_CAPACITY_MODE,
        )

    def test_sample_policy_bounds(self) -> None:
        cell = make_cell("synthetic_bounds", [1.0] * 10)
        self.assertEqual(sampled_cycle_numbers(cell, min_cycle=1, max_cycle=None), list(range(1, 11)))
        self.assertEqual(sampled_cycle_numbers(cell, min_cycle=9), [9, 10])
        self.assertEqual(sampled_cycle_numbers(cell, max_cycle=2), [1, 2])
        with self.assertRaises(ValueError):
            sampled_cycle_numbers(cell, cycle_stride=0)

    def test_generation_is_deterministic(self) -> None:
        first = CycleSOHLabelAnnotator().process_cell_cycles(clean_cell())
        second = CycleSOHLabelAnnotator().process_cell_cycles(clean_cell())
        self.assertEqual(first[0].tolist(), second[0].tolist())
        self.assertEqual(list(first[1]), list(second[1]))

    def test_the_whole_cell_annotator_no_longer_mixes_conventions(self) -> None:
        annotator = SOHLabelAnnotator(cycle_index=3)
        self.assertEqual(
            annotator.reference_capacity_mode, DEFAULT_REFERENCE_CAPACITY_MODE
        )
        self.assertAlmostEqual(float(annotator.process_cell(clean_cell())), 0.5)
        # Same value for the cell that carries a nominal capacity: the mode,
        # not the cell, decides.
        self.assertAlmostEqual(float(annotator.process_cell(nominal_cell())), 0.5)

    def test_the_absolute_mode_is_gone(self) -> None:
        """It returned Ah, not a ratio, under a parameter whose other value did."""
        with self.assertRaises(TypeError):
            SOHLabelAnnotator(cycle_index=3, mode="absolute")


class TestSampleIdentity(unittest.TestCase):
    """Many rows per cell, each traceable to its cell and observed cycle."""

    def test_many_samples_per_cell_with_identity(self) -> None:
        index = SampleIndex.for_cell("synthetic_clean", [1, 2, 3, 4])
        self.assertEqual(len(index), 4)
        self.assertEqual(index[2], {"cell_id": "synthetic_clean", "cycle_index": 3})
        self.assertEqual(index.unique_cell_ids, ["synthetic_clean"])

    def test_whole_cell_samples_are_marked_as_such(self) -> None:
        index = SampleIndex.for_whole_cells(["a", "b"])
        self.assertEqual(index.cycle_indices, (WHOLE_CELL_CYCLE, WHOLE_CELL_CYCLE))

    def test_concat_preserves_order(self) -> None:
        index = SampleIndex.concat([
            SampleIndex.for_cell("a", [1, 2]),
            SampleIndex.for_cell("b", [5]),
        ])
        self.assertEqual(index.cell_ids, ("a", "a", "b"))
        self.assertEqual(index.cycle_indices, (1, 2, 5))

    def test_select_keeps_identity_aligned_with_a_mask(self) -> None:
        index = SampleIndex.for_cell("a", [1, 2, 3])
        kept = index.select([True, False, True])
        self.assertEqual(kept.cycle_indices, (1, 3))

    def test_a_length_mismatch_is_refused(self) -> None:
        index = SampleIndex.for_cell("a", [1, 2, 3])
        with self.assertRaises(SampleIndexError) as ctx:
            index.check_matches(2, "test")
        self.assertIn("disagreed about how many samples", str(ctx.exception))

    def test_cardinality_matches_features_and_labels(self) -> None:
        annotator = CycleSOHLabelAnnotator()
        labels, cycles = annotator.process_cell_cycles(clean_cell())
        index = SampleIndex.for_cell("synthetic_clean", cycles)
        self.assertEqual(len(index), len(labels))
        index.check_matches(len(labels), "cardinality")


class TestCellLevelSplitInvariant(unittest.TestCase):
    """The hard rule: split cells, then expand cycles. Never the reverse."""

    def test_a_clean_split_passes(self) -> None:
        cells = cohort(n_cells=6)
        ids = cell_ids(cells)
        assert_disjoint_cells({"train": ids[:3], "val": ids[3:5], "test": ids[5:]})

    def test_the_leakage_prone_alternative_is_caught(self) -> None:
        """Build the wrong pipeline on purpose and prove it is rejected.

        The tempting shortcut is to expand every cell into cycle rows and then
        split rows at random. Here that is simulated exactly: one cell's rows
        land on both sides, and the validator must refuse it.
        """
        cells = cohort(n_cells=4, n_cycles=10)
        annotator = CycleSOHLabelAnnotator()

        rows: list[tuple[str, int]] = []
        for cell in cells:
            _, cycles = annotator.process_cell_cycles(cell)
            rows.extend((cell.cell_id, cycle) for cycle in cycles)

        # A deterministic stand-in for a random row split: every other row.
        train_rows = rows[0::2]
        test_rows = rows[1::2]
        train_ids = [cell_id for cell_id, _ in train_rows]
        test_ids = [cell_id for cell_id, _ in test_rows]
        self.assertTrue(set(train_ids) & set(test_ids), "the fixture is not leaky")

        with self.assertRaises(SampleIndexError) as ctx:
            assert_disjoint_cells({"train": train_ids, "test": test_ids})
        message = str(ctx.exception)
        self.assertIn("appear in more than one split", message)
        self.assertIn("synthetic_cell_0", message)

    def test_every_overlapping_cell_is_named(self) -> None:
        with self.assertRaises(SampleIndexError) as ctx:
            assert_disjoint_cells({"train": ["a", "b", "c"], "test": ["b", "c"]})
        message = str(ctx.exception)
        self.assertIn("2 cell(s)", message)
        self.assertIn("b in train+test", message)
        self.assertIn("c in train+test", message)

    def test_three_way_overlap_is_caught(self) -> None:
        with self.assertRaises(SampleIndexError):
            assert_disjoint_cells({"train": ["a"], "val": ["b"], "test": ["a"]})

    def test_cycles_of_one_cell_never_span_splits_when_split_by_cell(self) -> None:
        """Splitting cells first cannot produce the leak, by construction."""
        cells = cohort(n_cells=6, n_cycles=8)
        train_cells, test_cells = cells[:4], cells[4:]
        annotator = CycleSOHLabelAnnotator()

        def expand(group):
            out = []
            for cell in group:
                _, cycles = annotator.process_cell_cycles(cell)
                out.extend((cell.cell_id, cycle) for cycle in cycles)
            return out

        train_rows, test_rows = expand(train_cells), expand(test_cells)
        self.assertEqual(
            set(c for c, _ in train_rows) & set(c for c, _ in test_rows), set()
        )
        assert_disjoint_cells({
            "train": [c for c, _ in train_rows],
            "test": [c for c, _ in test_rows],
        })
        # And the sample count is the cycles, not the cells.
        self.assertEqual(len(train_rows), 4 * 8)
        self.assertEqual(len(test_rows), 2 * 8)


class TestNoRULLabelInvolved(unittest.TestCase):
    """An SOH run must not reach for the RUL annotator."""

    def test_the_soh_annotator_does_not_use_rul(self) -> None:
        from batteryml.label import soh as soh_module

        source = Path(soh_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("RULLabelAnnotator", source)
        self.assertNotIn("eol_soh", source)

    def test_soh_labels_are_ratios_not_cycle_counts(self) -> None:
        labels, _ = CycleSOHLabelAnnotator().process_cell_cycles(clean_cell())
        self.assertTrue(torch.all(labels <= 2.0), "these look like cycle counts, not ratios")


if __name__ == "__main__":
    unittest.main(verbosity=2)
