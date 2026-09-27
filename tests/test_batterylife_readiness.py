"""Downloaded data must not report MISSING — and must not report READY.

The reported symptom: datasets downloaded on the server still showed
``Status: MISSING``. Reproduced with filesystem fixtures, three independent
causes, all in ``assess_dataset``'s alternate-raw branch:

1. **Only an archive counted.** The branch accepted
   ``data/raw/BatteryLife/v11/<NAME>.zip`` and nothing else. An *extracted*
   dataset — the normal end state, and the state the external-model
   readiness path already requires of ``root_path`` — was invisible, as was
   a dataset whose archive had been removed after extraction.
2. **Only three datasets were mapped.** ``MICH``, ``MICH_EXP`` and
   ``Stanford_2`` were hardcoded. ``ISU_ILCC``, ``SDU``, ``Tongji`` and
   ``XJTU`` are declared in ``PLATFORM_DATASETS`` and distributed the same
   way, and could never be anything but MISSING however much was downloaded.
3. **It was skipped whenever a caller named its own raw root.** The branch
   was gated on ``raw_root is None``, but the BatteryLife distribution lives
   at a fixed repository path and has nothing to do with that argument.

The opposite error matters as much. Detected is not runnable: no BatteryLife
layout has a BatteryML preprocessor, so these report RAW_ONLY with
``preprocessing_supported`` False, and nothing may offer a preprocess
command for them. An empty directory, a truncated archive and a file wearing
a directory's name all stay MISSING.

Run from the repository root::

    conda run -n batteryml python -m unittest tests.test_batterylife_readiness -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import batteryml_web.core.readiness as rd  # noqa: E402
from batteryml_web.core.readiness import DatasetStatus  # noqa: E402

#: Declared by the platform and distributed through BatteryLife v11 — none
#: has a BatteryML preprocessor.
BATTERYLIFE_DATASETS = (
    "ISU_ILCC", "MICH", "MICH_EXP", "SDU", "Stanford_2", "Tongji", "XJTU",
)

#: Declared by the platform and preprocessable by BatteryML itself.
NATIVE_DATASETS = ("CALCE", "HNEI", "HUST", "MATR", "RWTH", "SNL", "UL_PUR")


class _Fixture(unittest.TestCase):
    """A temporary repository root, so nothing touches the real tree."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="readiness-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._real_root = rd._repo_root
        rd._repo_root = lambda: self.tmp
        self.addCleanup(self._restore)
        self.raw = self.tmp / "data" / "raw"
        self.processed = self.tmp / "data" / "processed"
        self.v11 = self.raw / "BatteryLife" / "v11"
        self.v11.mkdir(parents=True)
        self.processed.mkdir(parents=True)

    def _restore(self) -> None:
        rd._repo_root = self._real_root

    # --- fixture builders ------------------------------------------------
    def native_raw(self, name: str, *, files: int = 1) -> Path:
        directory = self.raw / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(files):
            (directory / f"cell_{index}.mat").write_text("x", encoding="utf-8")
        return directory

    def v11_archive(self, name: str) -> Path:
        target = self.v11 / f"{name}.zip"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(f"{name}/cell_0.pkl", "x")
        return target

    def v11_extracted(self, name: str, *, files: int = 2) -> Path:
        directory = self.v11 / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(files):
            (directory / f"cell_{index}.pkl").write_text("x", encoding="utf-8")
        return directory

    def processed_dir(self, name: str, *, files: int = 0) -> Path:
        directory = self.processed / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(files):
            (directory / f"cell_{index}.pkl").write_text("x", encoding="utf-8")
        return directory

    def status(self, name: str) -> DatasetStatus:
        return rd.assess_dataset(name).status


class TestCaseMatrix(_Fixture):
    """§10 A-I, over every layout the platform recognises."""

    def test_a_nothing_on_disk_is_missing(self) -> None:
        for name in BATTERYLIFE_DATASETS + NATIVE_DATASETS:
            with self.subTest(dataset=name):
                self.assertIs(self.status(name), DatasetStatus.MISSING)

    def test_b_native_raw_without_processed_is_raw_only(self) -> None:
        for name in NATIVE_DATASETS:
            with self.subTest(dataset=name):
                self.native_raw(name)
                report = rd.assess_dataset(name)
                self.assertIs(report.status, DatasetStatus.RAW_ONLY)
                self.assertTrue(report.raw_available)
                # BatteryML can preprocess these, so the action is real.
                self.assertTrue(report.preprocessing_supported)

    def test_c_batterylife_raw_is_raw_only_archived_or_extracted(self) -> None:
        """The reported bug. Both shapes, every declared dataset."""
        for name in BATTERYLIFE_DATASETS:
            with self.subTest(dataset=name, shape="archive"):
                archive = self.v11_archive(name)
                self.assertIs(self.status(name), DatasetStatus.RAW_ONLY)
                archive.unlink()
            with self.subTest(dataset=name, shape="extracted"):
                extracted = self.v11_extracted(name)
                self.assertIs(self.status(name), DatasetStatus.RAW_ONLY)
                shutil.rmtree(extracted)

    def test_c_extracted_survives_the_archive_being_deleted(self) -> None:
        """Unzip, delete the zip — the usual thing. This reported MISSING."""
        self.v11_extracted("Stanford_2")
        self.assertFalse((self.v11 / "Stanford_2.zip").exists())
        self.assertIs(self.status("Stanford_2"), DatasetStatus.RAW_ONLY)

    def test_d_an_empty_raw_directory_is_missing(self) -> None:
        (self.raw / "MATR").mkdir(parents=True)
        (self.v11 / "XJTU").mkdir()
        self.assertIs(self.status("MATR"), DatasetStatus.MISSING)
        self.assertIs(self.status("XJTU"), DatasetStatus.MISSING)

    def test_d_a_tree_of_empty_subdirectories_is_missing(self) -> None:
        """Non-empty means files, not directories."""
        (self.v11 / "MICH" / "sub" / "deeper").mkdir(parents=True)
        self.assertIs(self.status("MICH"), DatasetStatus.MISSING)

    def test_e_a_file_wearing_the_directory_name_is_missing(self) -> None:
        (self.v11 / "XJTU").write_text("not a directory", encoding="utf-8")
        self.assertIs(self.status("XJTU"), DatasetStatus.MISSING)

    def test_e_a_truncated_archive_is_missing_and_says_so(self) -> None:
        """A half-downloaded file has the name but not the data."""
        (self.v11 / "SDU.zip").write_text("not a zip", encoding="utf-8")
        report = rd.assess_dataset("SDU")
        self.assertIs(report.status, DatasetStatus.MISSING)
        # Distinguished from "nothing was ever downloaded".
        self.assertIn("not a readable ZIP", report.reason)

    def test_e_an_unrelated_neighbour_does_not_make_a_dataset_present(self) -> None:
        self.v11_extracted("MICH_EXP")
        self.assertIs(self.status("MICH"), DatasetStatus.MISSING)
        self.assertIs(self.status("MICH_EXP"), DatasetStatus.RAW_ONLY)

    def test_f_an_empty_processed_directory_is_incomplete_not_ready(self) -> None:
        self.processed_dir("SNL", files=0)
        report = rd.assess_dataset("SNL")
        self.assertIs(report.status, DatasetStatus.INCOMPLETE_PROCESSED)
        self.assertFalse(report.processed_available)

    def test_g_a_populated_processed_directory_is_ready(self) -> None:
        self.processed_dir("CALCE", files=3)
        report = rd.assess_dataset("CALCE")
        self.assertIs(report.status, DatasetStatus.READY)
        self.assertEqual(report.processed_sample_count, 3)

    def test_g_processed_outranks_raw(self) -> None:
        self.native_raw("MATR")
        self.processed_dir("MATR", files=2)
        self.assertIs(self.status("MATR"), DatasetStatus.READY)

    def test_h_status_follows_the_filesystem_without_a_restart(self) -> None:
        """Readiness is not cached; a download shows up on the next read."""
        self.assertIs(self.status("XJTU"), DatasetStatus.MISSING)
        self.v11_extracted("XJTU")
        self.assertIs(self.status("XJTU"), DatasetStatus.RAW_ONLY)
        self.processed_dir("XJTU", files=1)
        self.assertIs(self.status("XJTU"), DatasetStatus.READY)

    def test_i_one_dataset_does_not_affect_another(self) -> None:
        self.v11_extracted("Tongji")
        self.native_raw("HUST")
        self.processed_dir("MATR", files=1)
        self.assertIs(self.status("Tongji"), DatasetStatus.RAW_ONLY)
        self.assertIs(self.status("HUST"), DatasetStatus.RAW_ONLY)
        self.assertIs(self.status("MATR"), DatasetStatus.READY)
        for untouched in ("SDU", "ISU_ILCC", "CALCE", "SNL"):
            with self.subTest(dataset=untouched):
                self.assertIs(self.status(untouched), DatasetStatus.MISSING)


class TestDetectedIsNotRunnable(_Fixture):
    """§5: RAW_ONLY must never imply a preprocessing route exists."""

    def test_batterylife_raw_is_never_preprocessable(self) -> None:
        for name in BATTERYLIFE_DATASETS:
            for build in (self.v11_archive, self.v11_extracted):
                with self.subTest(dataset=name, shape=build.__name__):
                    made = build(name)
                    report = rd.assess_dataset(name)
                    self.assertIs(report.status, DatasetStatus.RAW_ONLY)
                    self.assertFalse(
                        report.preprocessing_supported,
                        "no BatteryML preprocessor consumes a BatteryLife "
                        "layout; offering one would fabricate a command",
                    )
                    if made.is_dir():
                        shutil.rmtree(made)
                    else:
                        made.unlink()

    def test_a_native_dataset_without_a_preprocessor_is_not_preprocessable(self) -> None:
        """The distinction is the layout's, not the directory's.

        ``data/raw/XJTU`` is the native shape, but BatteryML registers no
        XJTU preprocessor, so raw data there is detected and still not
        preprocessable.
        """
        self.native_raw("XJTU")
        report = rd.assess_dataset("XJTU")
        self.assertIs(report.status, DatasetStatus.RAW_ONLY)
        self.assertFalse(report.preprocessing_supported)

    def test_the_preprocessable_set_comes_from_batteryml_itself(self) -> None:
        """Non-vacuity: it is read, not listed here."""
        from batteryml.preprocess import SUPPORTED_SOURCES

        self.assertEqual(
            rd._native_preprocessor_datasets(),
            frozenset(n.upper() for n in SUPPORTED_SOURCES["DATASETS"]),
        )
        for name in NATIVE_DATASETS:
            self.assertIn(name.upper(), rd._native_preprocessor_datasets())


class TestLayoutResolution(_Fixture):
    """The declarative table, checked directly."""

    def test_the_native_directory_is_preferred_over_the_v11_copy(self) -> None:
        self.native_raw("MICH")
        self.v11_extracted("MICH")
        layouts = rd._recognised_raw_layouts("MICH", root=self.tmp)
        self.assertEqual(layouts[0].kind, "native")

    def test_an_extracted_directory_is_preferred_over_its_archive(self) -> None:
        self.v11_archive("XJTU")
        self.v11_extracted("XJTU")
        layouts = rd._recognised_raw_layouts("XJTU", root=self.tmp)
        self.assertEqual([layout.kind for layout in layouts], ["batterylife"] * 2)
        self.assertTrue(layouts[0].path.is_dir())
        self.assertEqual(layouts[0].path.name, "XJTU")

    def test_batterylife_detection_ignores_a_caller_supplied_raw_root(self) -> None:
        """It was gated on ``raw_root is None`` and vanished for such callers.

        The v11 distribution lives at a fixed repository path; a caller
        naming its own raw directory says nothing about it.
        """
        self.v11_extracted("Tongji")
        elsewhere = self.tmp / "somewhere" / "else"
        elsewhere.mkdir(parents=True)
        report = rd.assess_dataset("Tongji", raw_root=elsewhere)
        self.assertIs(report.status, DatasetStatus.RAW_ONLY)

    def test_a_named_raw_root_still_wins_when_it_holds_the_data(self) -> None:
        elsewhere = self.tmp / "elsewhere"
        (elsewhere / "MATR").mkdir(parents=True)
        (elsewhere / "MATR" / "cell.mat").write_text("x", encoding="utf-8")
        report = rd.assess_dataset("MATR", raw_root=elsewhere)
        self.assertIs(report.status, DatasetStatus.RAW_ONLY)
        self.assertIn("elsewhere", report.raw_path)

    def test_no_layout_is_recognised_for_an_absent_dataset(self) -> None:
        self.assertEqual(rd._recognised_raw_layouts("MATR", root=self.tmp), [])

    def test_the_reported_path_is_the_layout_that_was_found(self) -> None:
        self.v11_extracted("SDU")
        report = rd.assess_dataset("SDU")
        self.assertIn("BatteryLife", report.raw_path)
        self.assertTrue(report.raw_path.endswith("SDU"))


class TestNativeBehaviourDoesNotRegress(_Fixture):
    """MATR / HUST / SNL keep behaving exactly as before."""

    def test_the_documented_matr_lifecycle(self) -> None:
        self.assertIs(self.status("MATR"), DatasetStatus.MISSING)
        self.native_raw("MATR", files=5)
        report = rd.assess_dataset("MATR")
        self.assertIs(report.status, DatasetStatus.RAW_ONLY)
        self.assertTrue(report.preprocessing_supported)
        self.processed_dir("MATR", files=180)
        report = rd.assess_dataset("MATR")
        self.assertIs(report.status, DatasetStatus.READY)
        self.assertEqual(report.processed_sample_count, 180)

    def test_list_dataset_readiness_covers_every_declared_dataset(self) -> None:
        reports = rd.list_dataset_readiness()
        names = {report.dataset for report in reports}
        for declared in rd.PLATFORM_DATASETS:
            self.assertIn(declared, names)

    def test_list_dataset_readiness_reflects_the_batterylife_layouts(self) -> None:
        self.v11_extracted("XJTU")
        by_name = {r.dataset: r for r in rd.list_dataset_readiness()}
        self.assertIs(by_name["XJTU"].status, DatasetStatus.RAW_ONLY)
        self.assertIs(by_name["SDU"].status, DatasetStatus.MISSING)


if __name__ == "__main__":
    unittest.main()
