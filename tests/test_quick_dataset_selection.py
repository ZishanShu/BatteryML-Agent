"""Home → Quick experience → "Raw dataset": real choices, truthful readiness.

The selector used to be the hardcoded list ``["MATR"]`` — an enabled control
with one option, which read as "this platform supports only MATR". It now
offers every dataset the readiness layer knows, marks the ones the config
registry has no protocol config for, and keeps every dependent selection
(task, model, config, upload, execution plan) consistent with the dataset on
screen.

Readiness is faked per dataset with real ``DatasetReadiness`` objects over a
temporary directory (``synthetic_`` names only); nothing is downloaded,
preprocessed or trained — every data action and run goes through recorders.

Run from the repository root::

    python -m unittest tests.test_quick_dataset_selection -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from batteryml_web.core import config_task as ct  # noqa: E402
from batteryml_web.core import quick_experience as qx  # noqa: E402
from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.config_registry import get_config_registry  # noqa: E402
from batteryml_web.core.config_semantics import infer_config_semantics  # noqa: E402
from tests.test_quick_experience_fixes import AppTest, _PageHarness  # noqa: E402

READY, RAW_ONLY, INCOMPLETE, MISSING = "READY", "RAW_ONLY", "INCOMPLETE_PROCESSED", "MISSING"


def _registry_models(dataset: str, task: str) -> List[str]:
    return sorted({e.model_key for e in get_config_registry(str(REPO_ROOT)).get_entries(dataset, task)})


def _config_dataset(path: str) -> Any:
    return infer_config_semantics(yaml.safe_load(Path(path).read_text(encoding="utf-8"))).dataset


# --------------------------------------------------------------------------
# pure decisions
# --------------------------------------------------------------------------


class TestDatasetChoices(unittest.TestCase):
    PAIRS = [("MATR", "RUL prediction"), ("MATR", "SOH estimation"), ("HUST", "RUL prediction"),
             ("SNL", "RUL prediction"), (None, "RUL prediction")]

    def test_default_first_then_runnable_then_known_rest(self) -> None:
        statuses = {"CALCE": MISSING, "HUST": RAW_ONLY, "MATR": READY, "SNL": MISSING, "ABC": MISSING}
        choices = qx.dataset_choices(statuses, self.PAIRS, default="MATR")
        self.assertEqual([c.name for c in choices], ["MATR", "HUST", "SNL", "ABC", "CALCE"])
        by_name = {c.name: c for c in choices}
        self.assertEqual(by_name["MATR"].tasks, (ct.TASK_RUL, ct.TASK_SOH))      # canonical order
        self.assertEqual(by_name["HUST"].tasks, (ct.TASK_RUL,))
        self.assertFalse(by_name["CALCE"].runnable)
        self.assertEqual(by_name["HUST"].status, RAW_ONLY)                        # readiness kept apart

    def test_a_registry_dataset_without_a_verdict_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "HUST"):
            qx.dataset_choices({"MATR": READY}, self.PAIRS[:3], default="MATR")

    def test_task_options_and_limitation(self) -> None:
        hust = qx.DatasetChoice("HUST", RAW_ONLY, (ct.TASK_RUL,))
        calce = qx.DatasetChoice("CALCE", MISSING, ())
        self.assertEqual(qx.task_options(hust), [ct.TASK_RUL])
        self.assertEqual(qx.task_options(calce), list(ct.SUPPORTED_TASK_LABELS))
        self.assertIsNone(qx.quick_limitation(hust))
        self.assertIn("no protocol configuration is registered", qx.quick_limitation(calce))
        self.assertIn("Prepare Data", qx.quick_limitation(calce))

    def test_reconcile_selection_is_the_config_rule(self) -> None:
        self.assertIs(qx.reconcile_config, qx.reconcile_selection)
        self.assertEqual(qx.reconcile_selection(ct.TASK_SOH, [ct.TASK_RUL], default=ct.TASK_RUL),
                         (ct.TASK_RUL, ct.TASK_SOH))
        self.assertEqual(qx.reconcile_selection("xgb", ["cnn", "xgb"]), ("xgb", None))
        self.assertEqual(qx.reconcile_selection("xgb", []), (None, "xgb"))


class TestActionStates(unittest.TestCase):
    def test_download(self) -> None:
        done = qx.download_action(dataset="MATR", downloadable=True, raw_complete=True)
        self.assertEqual(done.state, qx.ACTION_COMPLETED)
        partial = qx.download_action(dataset="MATR", downloadable=True, raw_complete=False,
                                     raw_count=2, raw_expected=4)
        self.assertEqual((partial.state, "2 of 4" in partial.note), (qx.ACTION_AVAILABLE, True))
        manual = qx.download_action(dataset="SNL", downloadable=False, raw_complete=False)
        self.assertEqual(manual.state, qx.ACTION_UNAVAILABLE)
        self.assertIn("data/raw/SNL/", manual.note)

    def test_process(self) -> None:
        self.assertEqual(qx.process_action(status=READY, raw_ready=True, preprocessing_supported=False).state,
                         qx.ACTION_COMPLETED)
        self.assertEqual(qx.process_action(status=RAW_ONLY, raw_ready=True, preprocessing_supported=True).state,
                         qx.ACTION_AVAILABLE)
        self.assertEqual(qx.process_action(status=INCOMPLETE, raw_ready=True, preprocessing_supported=True).state,
                         qx.ACTION_AVAILABLE)
        no_route = qx.process_action(status=RAW_ONLY, raw_ready=True, preprocessing_supported=False)
        self.assertEqual(no_route.state, qx.ACTION_UNAVAILABLE)
        self.assertIn("automatic preprocessing route", no_route.note)
        missing = qx.process_action(status=MISSING, raw_ready=False, preprocessing_supported=False)
        self.assertEqual((missing.state, "raw files" in missing.note), (qx.ACTION_UNAVAILABLE, True))

    def test_a_plan_for_another_dataset_never_runs(self) -> None:
        """MATR + RUL + CNN + matr_1 is not executable once HUST is the dataset."""
        matr_cnn = REPO_ROOT / "configs" / "baselines" / "nn_models" / "cnn" / "matr_1.yaml"
        plan = qx.build_quick_plan(task=ct.TASK_RUL, config_path=matr_cnn, device="cpu", repo_root=REPO_ROOT,
                                   allowed_datasets=["HUST"], cuda_ok=False, require_data=False)
        self.assertFalse(plan.valid)
        self.assertTrue(any("reads dataset 'MATR'" in e and "HUST" in e for e in plan.errors), plan.errors)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


class _Readiness:
    """Real ``DatasetReadiness`` objects for chosen verdicts, over a temp root."""

    def __init__(self, root: Path, verdicts: Dict[str, str]) -> None:
        self.root, self.verdicts = root, verdicts
        from batteryml.preprocess import DOWNLOAD_LINKS

        for name, verdict in verdicts.items():
            if verdict in (READY, RAW_ONLY, INCOMPLETE):
                raw = root / "raw" / name
                raw.mkdir(parents=True, exist_ok=True)
                names = [item[1] for item in DOWNLOAD_LINKS.get(name, [])] or ["synthetic_raw.csv"]
                for file_name in names:
                    (raw / file_name).write_bytes(b"synthetic raw bytes")

    def __call__(self, dataset: str, **_: Any) -> Any:
        from batteryml_web.core.readiness import DatasetReadiness, DatasetStatus

        verdict = self.verdicts.get(str(dataset), MISSING)
        return DatasetReadiness(
            dataset=str(dataset), status=DatasetStatus(verdict),
            raw_available=verdict != MISSING, processed_available=verdict == READY,
            processed_sample_count=3 if verdict == READY else 0,
            reason=f"synthetic {verdict} in *.pkl terms", recommended_action="synthetic",
            raw_path=str(self.root / "raw" / str(dataset)),
            processed_path=str(self.root / "processed" / str(dataset)),
            preprocessing_supported=verdict in (RAW_ONLY, INCOMPLETE),
        )


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestQuickDatasetPage(_PageHarness, unittest.TestCase):
    VERDICTS = {"MATR": READY, "HUST": RAW_ONLY, "SNL": MISSING, "CALCE": MISSING}

    def setUp(self) -> None:
        super().setUp()
        from batteryml_web.core import readiness

        self.data_calls: List[List[str]] = []
        self.readiness = _Readiness(self.tmp / "data", dict(self.VERDICTS))
        self.stack.enter_context(mock.patch.object(readiness, "assess_dataset", self.readiness))
        # MATR's real completeness check opens HDF5 headers; synthetic bytes are
        # "complete" here exactly as any other non-empty raw file is.
        self.stack.enter_context(mock.patch(
            "batteryml.preprocess.is_download_complete",
            lambda f: Path(f).is_file() and Path(f).stat().st_size > 0))
        self.stack.enter_context(mock.patch(
            "batteryml_web.ui.progress_utils.run_data_command_with_progress", self.fake_data_action))
        self.stack.enter_context(mock.patch.object(qx, "cuda_available", lambda probe=None: (False, "")))

    def fake_data_action(self, *, command, **_: Any) -> Dict[str, Any]:
        self.data_calls.append(list(command))
        return {"success": True, "return_code": 0, "elapsed": 0.0, "log_path": "", "output_tail": ""}

    def open(self) -> "AppTest":
        at = self.app()
        at.run()
        self.assertHealthy(at)
        return at

    def text(self, at) -> str:
        parts = [str(e.value) for kind in ("caption", "markdown", "info", "success", "warning", "error")
                 for e in getattr(at, kind)]
        return "\n".join(parts)

    def buttons(self, at) -> Dict[str, Any]:
        return {b.key: b for b in at.button if b.key}

    # 1 + 2
    def test_matr_is_the_default_and_the_options_are_the_known_datasets(self) -> None:
        at = self.open()
        box = at.selectbox(key="quick_dataset_selector")
        self.assertEqual(box.value, "MATR")
        runnable = {e.dataset for e in get_config_registry(str(REPO_ROOT)).entries if e.dataset}
        values = [v for v in box.options]                          # formatted labels
        self.assertEqual(values[0], "MATR")
        for name in runnable:
            self.assertIn(name, values)                          # plain label: runnable here
        self.assertIn("CALCE · no run configuration", values)    # known, not hidden, marked
        self.assertGreater(len(values), 1)

    # 3 + 8
    def test_a_second_dataset_can_be_selected_and_drives_task_model_config(self) -> None:
        at = self.open()
        self.select(at, "quick_dataset_selector", "HUST")
        self.assertEqual(at.session_state["selected_dataset"], "HUST")
        self.assertEqual(list(at.selectbox(key="quick_task_selector").options), [ct.TASK_RUL])
        models = at.selectbox(key="quick_model_selector")             # options are display labels
        self.assertEqual(len(models.options), len(_registry_models("HUST", ct.TASK_RUL)))
        self.assertIn(models.value, _registry_models("HUST", ct.TASK_RUL))
        self.assertEqual(_config_dataset(self.value(at, "quick_config_selector")), "HUST")

    # 4
    def test_ready_is_reported_as_ready_and_completed_actions_are_not_buttons(self) -> None:
        at = self.open()
        text = self.text(at)
        self.assertIn("Readiness: **READY**", text)
        self.assertIn("synthetic READY in \\*.pkl terms", text)  # reason verbatim, escaped
        self.assertTrue(any(s.value == "Available" for s in at.success))
        self.assertTrue(any(s.value == "Ready" for s in at.success))
        self.assertIn("✓] Raw files present", text)
        self.assertIn("✓] Processed data ready", text)
        keys = self.buttons(at)
        self.assertNotIn("home_quick_download_data", keys)       # done: indicator, no button
        self.assertNotIn("home_quick_process_data", keys)
        self.assertFalse(keys["home_quick_run_experiment"].disabled)

    # 5
    def test_raw_only_never_becomes_ready(self) -> None:
        at = self.open()
        self.select(at, "quick_dataset_selector", "HUST")
        text = self.text(at)
        self.assertIn("Readiness: **RAW_ONLY**", text)
        self.assertFalse(any(s.value == "Ready" for s in at.success))
        self.assertTrue(any(w.value == "Needs processing" for w in at.warning))
        keys = self.buttons(at)
        self.assertFalse(keys["home_quick_process_data"].disabled)   # the real next action
        self.assertTrue(keys["home_quick_run_experiment"].disabled)
        self.assertTrue(any("Process the dataset in step 1" in w.value for w in at.warning))

    # 6
    def test_missing_cannot_execute_and_says_what_is_possible(self) -> None:
        at = self.open()
        self.select(at, "quick_dataset_selector", "SNL")
        text = self.text(at)
        self.assertIn("Readiness: **MISSING**", text)
        self.assertTrue(any(i.value == "No raw files found" for i in at.info))   # no download link
        keys = self.buttons(at)
        self.assertTrue(keys["home_quick_download_data"].disabled)
        self.assertIn("no automatic download for SNL", text)
        self.assertTrue(keys["home_quick_process_data"].disabled)
        self.assertIn("Available once the raw files are present.", text)
        self.assertTrue(keys["home_quick_run_experiment"].disabled)

    # 7
    def test_switching_dataset_clears_the_previous_datasets_config(self) -> None:
        at = self.open()
        self.select(at, "quick_model_selector", "cnn")
        matr_cnn = str(REPO_ROOT / "configs" / "baselines" / "nn_models" / "cnn" / "matr_1.yaml")
        self.select(at, "quick_config_selector", matr_cnn)
        self.select(at, "quick_dataset_selector", "HUST")
        config = self.value(at, "quick_config_selector")
        self.assertNotEqual(config, matr_cnn)
        self.assertEqual(_config_dataset(config), "HUST")
        self.assertEqual(self.value(at, "quick_model_selector"), "cnn")       # HUST has a CNN config
        self.assertTrue(any("matr_1.yaml is not a HUST · RUL prediction configuration" in c.value
                            and "no longer selected" in c.value for c in at.caption))
        self.assertEqual(at.session_state["selected_config"], config)

    # 8
    def test_a_task_the_new_dataset_lacks_is_cleared_and_explained(self) -> None:
        at = self.open()
        self.select(at, "quick_task_selector", ct.TASK_SOH)
        self.select(at, "quick_model_selector", "ridge")
        self.select(at, "quick_dataset_selector", "HUST")
        self.assertEqual(self.value(at, "quick_task_selector"), ct.TASK_RUL)
        self.assertTrue(any("HUST has no registered SOH estimation configuration" in c.value for c in at.caption))
        self.assertEqual(at.session_state["selected_task"], ct.TASK_RUL)
        self.assertEqual(_config_dataset(self.value(at, "quick_config_selector")), "HUST")
        # ... and back: MATR offers SOH again, the RUL selection stays a RUL one.
        self.select(at, "quick_dataset_selector", "MATR")
        self.assertEqual(list(at.selectbox(key="quick_task_selector").options), [ct.TASK_RUL, ct.TASK_SOH])
        self.assertEqual(self.value(at, "quick_task_selector"), ct.TASK_RUL)

    # 9
    def test_selecting_datasets_never_downloads_processes_or_runs(self) -> None:
        at = self.open()
        for name in ("HUST", "SNL", "CALCE", "MATR", "HUST"):
            self.select(at, "quick_dataset_selector", name)
        self.assertEqual(self.data_calls, [])
        self.assertEqual(self.calls, [])

    # 10
    def test_an_unsupported_dataset_explains_and_offers_prepare_data(self) -> None:
        at = self.open()
        self.select(at, "quick_dataset_selector", "CALCE")
        self.assertTrue(any("CALCE is known to the platform" in i.value
                            and "no protocol configuration is registered" in i.value for i in at.info))
        keys = self.buttons(at)
        self.assertNotIn("home_quick_download_data", keys)
        self.assertNotIn("home_quick_process_data", keys)
        self.assertTrue(keys["home_quick_run_experiment"].disabled)
        at.button(key="home_quick_open_prepare").click().run()
        self.assertHealthy(at)
        self.assertEqual(at.session_state["current_step"], "prepare_data")
        self.assertEqual(self.data_calls, [])

    def test_the_only_data_action_runs_on_click_for_the_selected_dataset(self) -> None:
        at = self.open()
        self.select(at, "quick_dataset_selector", "HUST")
        with mock.patch("batteryml_web.core.quick_experience.sync_manifests", lambda *a, **k: (True, "")):
            at.button(key="home_quick_process_data").click().run()
        self.assertHealthy(at)
        self.assertEqual(len(self.data_calls), 1)
        self.assertIn("preprocess", self.data_calls[0])
        self.assertIn("HUST", self.data_calls[0])
        self.assertEqual(self.calls, [])                          # preprocessing is not training

    def test_a_second_ready_dataset_runs_its_own_config(self) -> None:
        self.readiness.verdicts["HUST"] = READY
        at = self.open()
        self.select(at, "quick_dataset_selector", "HUST")
        self.select(at, "quick_model_selector", "ridge")
        at.button(key="home_quick_run_experiment").click().run()
        self.assertHealthy(at)
        self.assertEqual(len(self.calls), 1)
        command = self.calls[0]
        config = command[command.index("run") + 1]
        self.assertEqual(_config_dataset(config), "HUST")
        manifest = rm.find_run(self.project, at.session_state["last_run_id"])
        self.assertEqual((manifest["dataset"], manifest["task"]), ("HUST", ct.TASK_RUL))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
