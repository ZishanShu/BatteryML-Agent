"""SOH eligibility is decided by opening the processed cells.

A dataset being listed in a README, or having an SOH column in a results
table, says nothing about whether its processed files can produce SOH labels
under a given reference-capacity convention. These tests build small
processed datasets on disk and check that the verdict follows the data — and
in particular that a dataset which is fine for RUL can still be ineligible for
SOH, because the two questions are different.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_soh_readiness -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core.soh_readiness import (  # noqa: E402
    SOH_INSUFFICIENT,
    SOH_NOT_ELIGIBLE,
    SOH_NOT_PROCESSED,
    SOH_READY,
    assess_soh_dataset,
)
from tests.fixtures.synthetic_soh_cells import cohort, make_cell  # noqa: E402


class SOHReadinessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dataset(self, name: str, cells) -> Path:
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            cell.dump(str(directory / f"{cell.cell_id}.pkl"))
        return directory


class TestEligibleDataset(SOHReadinessTestCase):
    def test_a_usable_dataset_is_ready_with_real_counts(self) -> None:
        self._dataset("SYNTH", cohort(n_cells=8, n_cycles=60))
        report = assess_soh_dataset("SYNTH", self.root, min_samples_per_cell=20,
                                    min_eligible_cells=5)
        self.assertEqual(report.status, SOH_READY)
        self.assertEqual(report.cells_inspected, 8)
        self.assertEqual(report.cells_eligible, 8)
        # The count is the real number of cycle-level samples, not the cells.
        self.assertEqual(report.total_samples, 8 * 60)
        self.assertIn("first_valid_discharge_capacity", report.reference_capacity_mode)

    def test_the_sample_policy_changes_the_count(self) -> None:
        self._dataset("SYNTH", cohort(n_cells=8, n_cycles=60))
        dense = assess_soh_dataset("SYNTH", self.root, min_samples_per_cell=1)
        sparse = assess_soh_dataset("SYNTH", self.root, min_samples_per_cell=1,
                                    cycle_stride=5)
        self.assertEqual(dense.total_samples, 8 * 60)
        self.assertEqual(sparse.total_samples, 8 * 12)
        self.assertEqual(sparse.sample_policy["cycle_stride"], 5)

    def test_the_summary_row_is_the_server_table(self) -> None:
        self._dataset("SYNTH", cohort(n_cells=6, n_cycles=40))
        row = assess_soh_dataset("SYNTH", self.root).summary_row()
        self.assertEqual(
            set(row), {"Dataset", "Cells", "Eligible cells", "SOH samples", "Status", "Reason"}
        )
        self.assertEqual(row["Dataset"], "SYNTH")
        self.assertEqual(row["Status"], SOH_READY)


class TestIneligibleDatasets(SOHReadinessTestCase):
    def test_a_dataset_without_reference_capacity_is_not_eligible(self) -> None:
        cells = [make_cell(f"synthetic_dead_{k}", [None] * 40) for k in range(6)]
        self._dataset("NOCAP", cells)
        report = assess_soh_dataset("NOCAP", self.root)
        self.assertEqual(report.status, SOH_NOT_ELIGIBLE)
        self.assertEqual(report.cells_eligible, 0)
        self.assertEqual(report.total_samples, 0)
        self.assertIn("no_valid_reference_capacity", report.exclusions)

    def test_a_dataset_ok_for_rul_can_be_ineligible_for_soh(self) -> None:
        """The distinction that matters.

        These cells run to end of life and have a full cycle history, so an
        RUL label is available for each of them. They carry no usable
        discharge capacity, so no SOH label exists at any cycle.
        """
        cells = [make_cell(f"synthetic_rul_only_{k}", [None] * 200) for k in range(10)]
        self._dataset("RULONLY", cells)
        report = assess_soh_dataset("RULONLY", self.root)
        self.assertEqual(report.status, SOH_NOT_ELIGIBLE)
        self.assertEqual(report.cells_inspected, 10)
        self.assertIn("reference_capacity_mode", report.reason)

    def test_too_few_cells_is_insufficient_not_ready(self) -> None:
        self._dataset("TINY", cohort(n_cells=2, n_cycles=60))
        report = assess_soh_dataset("TINY", self.root, min_eligible_cells=5)
        self.assertEqual(report.status, SOH_INSUFFICIENT)
        self.assertEqual(report.cells_eligible, 2)
        self.assertIn("cell-level split", report.reason)

    def test_short_cells_are_excluded_and_counted(self) -> None:
        self._dataset("SHORT", cohort(n_cells=6, n_cycles=5))
        report = assess_soh_dataset("SHORT", self.root, min_samples_per_cell=20)
        self.assertEqual(report.status, SOH_NOT_ELIGIBLE)
        self.assertEqual(report.exclusions["too_few_valid_samples"], 6)

    def test_a_cell_without_a_voltage_trace_cannot_be_featurised(self) -> None:
        cells = cohort(n_cells=6, n_cycles=40)
        for cell in cells:
            for cycle in cell.cycle_data:
                cycle.voltage_in_V = None
        self._dataset("NOVOLT", cells)
        report = assess_soh_dataset("NOVOLT", self.root)
        self.assertEqual(report.status, SOH_NOT_ELIGIBLE)
        self.assertIn("no_usable_voltage_trace", report.exclusions)


class TestNothingIsFabricated(SOHReadinessTestCase):
    def test_a_missing_dataset_reports_zero_and_says_why(self) -> None:
        report = assess_soh_dataset("ABSENT", self.root)
        self.assertEqual(report.status, SOH_NOT_PROCESSED)
        self.assertEqual(report.cells_inspected, 0)
        self.assertEqual(report.total_samples, 0)
        self.assertIn("does not exist", report.reason)
        self.assertIn("never from a README", report.reason)

    def test_an_empty_directory_is_not_a_verdict_about_the_dataset(self) -> None:
        (self.root / "EMPTY").mkdir(parents=True)
        report = assess_soh_dataset("EMPTY", self.root)
        self.assertEqual(report.status, SOH_NOT_PROCESSED)
        self.assertEqual(report.total_samples, 0)
        self.assertIn("no .pkl cell", report.reason)

    def test_the_reference_mode_is_recorded_in_the_report(self) -> None:
        self._dataset("SYNTH", cohort(n_cells=6, n_cycles=40))
        report = assess_soh_dataset("SYNTH", self.root,
                                    reference_capacity_mode="first_n_valid_mean")
        self.assertEqual(report.reference_capacity_mode, "first_n_valid_mean")
        self.assertIn("first_n_valid_mean", report.to_dict()["reference_capacity_mode"])

    def test_nominal_mode_excludes_cells_that_lack_one(self) -> None:
        """Excluded under the configured mode, never relabelled under another."""
        self._dataset("SYNTH", cohort(n_cells=6, n_cycles=40))
        report = assess_soh_dataset("SYNTH", self.root,
                                    reference_capacity_mode="nominal_capacity")
        self.assertEqual(report.status, SOH_NOT_ELIGIBLE)
        self.assertIn("no_valid_reference_capacity", report.exclusions)

    def test_a_partial_audit_says_how_many_it_looked_at(self) -> None:
        self._dataset("SYNTH", cohort(n_cells=10, n_cycles=40))
        report = assess_soh_dataset("SYNTH", self.root, max_cells=3,
                                    min_eligible_cells=1)
        self.assertEqual(report.cells_inspected, 3)
        self.assertEqual(report.cells_eligible, 3)


class TestAuditCommand(unittest.TestCase):
    def test_it_refuses_to_guess_without_data(self) -> None:
        from scripts.audit_soh_readiness import main

        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--processed-root", str(Path(tmp) / "nothing")])
        self.assertEqual(code, 2, "an absent processed root must not report a verdict")

    def test_it_reports_ready_datasets_with_a_zero_exit(self) -> None:
        from scripts.audit_soh_readiness import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "SYNTH"
            directory.mkdir()
            for cell in cohort(n_cells=6, n_cycles=40):
                cell.dump(str(directory / f"{cell.cell_id}.pkl"))
            code = main(["--processed-root", str(root)])
        self.assertEqual(code, 0)

    def test_it_exits_nonzero_when_nothing_is_eligible(self) -> None:
        from scripts.audit_soh_readiness import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "NOCAP"
            directory.mkdir()
            for k in range(4):
                make_cell(f"synthetic_dead_{k}", [None] * 30).dump(
                    str(directory / f"synthetic_dead_{k}.pkl"))
            code = main(["--processed-root", str(root)])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
