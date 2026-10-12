"""Regression tests for the Home / Quick experience fixes on PR #13.

Covers the review findings (P0-1 … P2-3) with temporary directories and
synthetic artifacts only. No BatteryML model is trained: the page's run is
driven through a fake ``run_command_with_progress`` that writes a synthetic
prediction artifact, and data readiness is faked so no ``data/processed`` is
needed (``CLAUDE.md`` §7).

Run from the repository root::

    python -m unittest tests.test_quick_experience_fixes -v
"""

from __future__ import annotations

import ast
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests  # noqa: E402
import yaml  # noqa: E402

from batteryml_web.core import config_task as ct  # noqa: E402
from batteryml_web.core import config_upload as cu  # noqa: E402
from batteryml_web.core import quick_experience as qx  # noqa: E402
from batteryml_web.core import run_manifest as rm  # noqa: E402

APP = REPO_ROOT / "batteryml_web" / "app.py"
RUL_CNN = REPO_ROOT / "configs" / "baselines" / "nn_models" / "cnn" / "matr_1.yaml"
RUL_XGB = REPO_ROOT / "configs" / "baselines" / "sklearn" / "xgb" / "matr_1.yaml"
SOH_XGB = REPO_ROOT / "configs" / "soh" / "xgb" / "matr.yaml"
SOH_ENET = REPO_ROOT / "configs" / "soh" / "elastic_net" / "matr.yaml"
TASKS = (ct.TASK_RUL, ct.TASK_SOH)

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - streamlit is a declared dependency
    AppTest = None


def _yaml_bytes(path: Path, **changes: Any) -> bytes:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for dotted, value in changes.items():
        node = data
        keys = dotted.split("__")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        if value is None:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = value
    return yaml.safe_dump(data, sort_keys=False).encode("utf-8")


# --- P2-2: one task rule -----------------------------------------------------------

class TestTaskResolution(unittest.TestCase):
    def test_annotator_decides_and_eol_soh_does_not(self) -> None:
        rul = ct.resolve_config_task({"label": {"name": "RULLabelAnnotator", "eol_soh": 0.8}})
        self.assertEqual((rul.task, rul.structural), (ct.TASK_RUL, True))
        for annotator in ("CycleSOHLabelAnnotator", "SOHLabelAnnotator"):
            with self.subTest(annotator=annotator):
                res = ct.resolve_config_task({"label": {"name": annotator}})
                self.assertEqual((res.task, res.structural), (ct.TASK_SOH, True))

    def test_uploaded_soh_without_task_field_is_soh(self) -> None:
        config = yaml.safe_load(_yaml_bytes(SOH_XGB, task=None))
        self.assertNotIn("task", config)
        self.assertEqual(ct.resolve_config_task(config).task, ct.TASK_SOH)

    def test_conflicts_and_unknowns_are_errors_not_guesses(self) -> None:
        conflict = ct.resolve_config_task({"task": "SOH estimation", "label": {"name": "RULLabelAnnotator"}})
        self.assertIsNone(conflict.task)
        self.assertIn("never picks", conflict.error)
        for config in ({"task": "sohish", "label": {"name": "CycleSOHLabelAnnotator"}},
                       {"label": {"name": "SohLikeAnnotator"}},
                       {"label": {"name": "soh_annotator"}},
                       {}):
            with self.subTest(config=config):
                res = ct.resolve_config_task(config)
                self.assertIsNone(res.task)
                self.assertTrue(res.error)

    def test_one_rule_shared_by_validator_registry_and_semantics(self) -> None:
        from batteryml_web.core import config_validator, readiness
        from batteryml_web.core.config_registry import get_config_registry
        from batteryml_web.core.config_semantics import load_config_semantics

        self.assertEqual((readiness.TASK_RUL, readiness.TASK_SOH), (ct.TASK_RUL, ct.TASK_SOH))
        self.assertEqual(config_validator.TASK_BY_ANNOTATOR["CycleSOHLabelAnnotator"], "SOH")
        report = config_validator.load_and_validate_config_file(SOH_XGB, require_data=False)
        self.assertEqual(report["task"], "SOH")
        self.assertEqual(load_config_semantics(SOH_XGB).task, ct.TASK_SOH)
        self.assertEqual(load_config_semantics(RUL_CNN).task, ct.TASK_RUL)
        entries = {Path(e.config_path).as_posix().split("configs/", 1)[-1]: e.task
                   for e in get_config_registry(str(REPO_ROOT)).get_entries("MATR")}
        self.assertEqual(entries["soh/xgb/matr.yaml"], ct.TASK_SOH)
        self.assertEqual(entries["baselines/nn_models/cnn/matr_1.yaml"], ct.TASK_RUL)
        source = (REPO_ROOT / "batteryml_web" / "core" / "config_semantics.py").read_text()
        self.assertNotIn('startswith("soh")', source)


# --- P0-1: the plan / execution gate ------------------------------------------------

def _data_root() -> Path:
    """A temporary repository root with a non-empty data/processed/MATR."""
    root = Path(tempfile.mkdtemp(prefix="synthetic_quick_root_"))
    processed = root / "data" / "processed" / "MATR"
    processed.mkdir(parents=True)
    (processed / "synthetic_cell.pkl").write_bytes(b"synthetic")
    return root


class TestPlanGate(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _data_root()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def plan(self, task: str, config: Path, device: str = "cpu", **kw) -> qx.QuickPlan:
        return qx.build_quick_plan(task=task, config_path=config, device=device, repo_root=self.root,
                                   allowed_datasets=["MATR"], cuda_ok=kw.pop("cuda_ok", False), **kw)

    def test_rul_config_cannot_run_as_soh_and_back(self) -> None:
        wrong = self.plan(ct.TASK_SOH, RUL_CNN)
        self.assertFalse(wrong.valid)
        self.assertEqual(wrong.config_task, ct.TASK_RUL)
        with self.assertRaises(ValueError):
            qx.plan_command(wrong, workspace="runs/x")
        reverse = self.plan(ct.TASK_RUL, SOH_XGB)
        self.assertFalse(reverse.valid)
        right = self.plan(ct.TASK_SOH, SOH_XGB)
        self.assertTrue(right.valid, right.errors)
        command = qx.plan_command(right, workspace="runs/x")
        self.assertIn(str(SOH_XGB), command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(right.task, ct.TASK_SOH)
        self.assertTrue(self.plan(ct.TASK_RUL, RUL_CNN).valid)

    def test_missing_data_and_missing_config_block(self) -> None:
        empty = Path(tempfile.mkdtemp(prefix="synthetic_empty_root_"))
        try:
            plan = qx.build_quick_plan(task=ct.TASK_SOH, config_path=SOH_XGB, device="cpu",
                                       repo_root=empty, allowed_datasets=["MATR"], cuda_ok=False)
            self.assertFalse(plan.valid)
            self.assertTrue(any("cell_data_path" in e for e in plan.errors))
        finally:
            shutil.rmtree(empty)
        self.assertFalse(self.plan(ct.TASK_SOH, None).valid)
        self.assertFalse(self.plan(ct.TASK_SOH, self.root / "missing.yaml").valid)

    def test_config_choices_follow_task_and_model(self) -> None:
        options = [qx.ConfigOption("rul/cnn/matr_1.yaml", ct.TASK_RUL, "cnn"),
                   qx.ConfigOption("rul/xgb/matr_1.yaml", ct.TASK_RUL, "xgb"),
                   qx.ConfigOption("soh/xgb/matr.yaml", ct.TASK_SOH, "xgb")]
        soh = qx.config_choices(options, task=ct.TASK_SOH, model_key="xgb")
        self.assertEqual(soh, ["soh/xgb/matr.yaml"])
        selected, cleared = qx.reconcile_config("rul/cnn/matr_1.yaml", soh)
        self.assertEqual((selected, cleared), ("soh/xgb/matr.yaml", "rul/cnn/matr_1.yaml"))
        self.assertEqual(qx.reconcile_config("soh/xgb/matr.yaml", soh), ("soh/xgb/matr.yaml", None))
        self.assertEqual(qx.reconcile_config("rul/cnn/matr_1.yaml", []), (None, "rul/cnn/matr_1.yaml"))


# --- P0-3: devices ----------------------------------------------------------------

class TestDevice(unittest.TestCase):
    def test_cuda_probe_failure_means_unavailable_never_a_crash(self) -> None:
        def boom() -> bool:
            raise RuntimeError("no driver")
        self.assertEqual(qx.cuda_available(boom)[0], False)
        self.assertEqual(qx.cuda_available(lambda: False)[0], False)
        self.assertEqual(qx.cuda_available(lambda: True), (True, ""))

    def test_unavailable_gpu_is_explained_not_replaced(self) -> None:
        problem = qx.device_problem("cuda", cuda_ok=False)
        self.assertIn("Choose cpu explicitly", problem)
        self.assertIsNone(qx.device_problem("cpu", cuda_ok=False))
        self.assertIsNone(qx.device_problem("cuda", cuda_ok=True))
        self.assertTrue(qx.device_problem("cuda:0,1", cuda_ok=True))
        root = _data_root()
        try:
            plan = qx.build_quick_plan(task=ct.TASK_SOH, config_path=SOH_XGB, device="cuda",
                                       repo_root=root, allowed_datasets=["MATR"], cuda_ok=False)
            self.assertFalse(plan.valid)
            self.assertEqual(plan.device, "cuda")
        finally:
            shutil.rmtree(root)


# --- P0-2: uploads ----------------------------------------------------------------

class TestUpload(unittest.TestCase):
    def setUp(self) -> None:
        self.managed = Path(tempfile.mkdtemp(prefix="synthetic_upload_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.managed, ignore_errors=True)

    def stage(self, raw: bytes, name: str = "my_config.yaml", tasks=TASKS) -> cu.UploadOutcome:
        return cu.stage_uploaded_config(raw, name, managed_root=self.managed, repo_root=REPO_ROOT,
                                        allowed_datasets=["MATR"], allowed_tasks=tasks)

    def stored(self) -> List[Path]:
        return sorted((self.managed / cu.MANAGED_SUBDIR).glob("*.yaml"))

    def test_valid_soh_and_rul_uploads_are_stored_under_new_ids(self) -> None:
        soh = self.stage(_yaml_bytes(SOH_XGB, task=None), "matr.yaml")
        self.assertTrue(soh.ok, soh.errors)
        self.assertEqual((soh.task, soh.model_key, soh.dataset), (ct.TASK_SOH, "xgb", "MATR"))
        rul = self.stage(RUL_CNN.read_bytes(), "matr.yaml")
        self.assertTrue(rul.ok, rul.errors)
        self.assertEqual(rul.task, ct.TASK_RUL)
        self.assertNotEqual(soh.path, rul.path)                      # same name twice
        self.assertEqual(len(self.stored()), 2)
        for path in (soh.path, rul.path):
            self.assertTrue(cu.is_managed_upload(path, self.managed))
            self.assertNotIn("matr", path.stem)                       # name never chooses the path
        self.assertEqual(yaml.safe_load(soh.path.read_text())["label"]["name"], "CycleSOHLabelAnnotator")
        self.assertEqual(SOH_XGB.read_bytes(), SOH_XGB.read_bytes())

    def test_invalid_uploads_store_nothing(self) -> None:
        cases = {
            "malformed": (b"model: [unclosed\n", "a.yaml"),
            "python tag": (b"model: !!python/object/apply:os.system ['echo hi']\n", "a.yaml"),
            "not a mapping": (b"- 1\n- 2\n", "a.yaml"),
            "empty": (b"   \n", "a.yaml"),
            "traversal name": (RUL_CNN.read_bytes(), "../../configs/x.yaml"),
            "absolute name": (RUL_CNN.read_bytes(), "/etc/passwd.yaml"),
            "wrong suffix": (RUL_CNN.read_bytes(), "config.txt"),
            "oversize": (b"#" * (cu.MAX_UPLOAD_BYTES + 1), "a.yaml"),
            "task conflict": (_yaml_bytes(RUL_CNN, task="SOH estimation"), "a.yaml"),
            "unregistered model": (_yaml_bytes(RUL_CNN, model__name="EvilPredictor"), "a.yaml"),
            "unregistered annotator": (_yaml_bytes(RUL_CNN, label__name="MyLabel"), "a.yaml"),
            "data outside processed": (_yaml_bytes(RUL_CNN, train_test_split__cell_data_path="/tmp/evil"), "a.yaml"),
            "data via ..": (_yaml_bytes(RUL_CNN, train_test_split__cell_data_path="data/processed/../../etc"), "a.yaml"),
            "workspace override": (_yaml_bytes(RUL_CNN, model__workspace="/tmp/elsewhere"), "a.yaml"),
            "extra file option": (_yaml_bytes(RUL_CNN, label__soh_filepath="x.csv"), "a.yaml"),
            "device option": (_yaml_bytes(RUL_CNN, model__device="cuda"), "a.yaml"),
            "other dataset": (_yaml_bytes(RUL_CNN, train_test_split__cell_data_path="data/processed/HUST"), "a.yaml"),
        }
        for label, (raw, name) in cases.items():
            with self.subTest(label):
                outcome = self.stage(raw, name)
                self.assertFalse(outcome.ok)
                self.assertTrue(outcome.errors)
                self.assertIsNone(outcome.path)
        self.assertEqual(self.stored(), [])
        self.assertFalse(self.stage(SOH_XGB.read_bytes(), tasks=(ct.TASK_RUL,)).ok)  # task not offered

    def test_symlinked_upload_directory_is_refused(self) -> None:
        elsewhere = Path(tempfile.mkdtemp(prefix="synthetic_elsewhere_"))
        try:
            (self.managed / "uploads").symlink_to(elsewhere, target_is_directory=True)
            outcome = self.stage(RUL_CNN.read_bytes())
            self.assertFalse(outcome.ok)
            self.assertIn("symbolic link", outcome.errors[0])
            self.assertEqual(list(elsewhere.rglob("*")), [])
        finally:
            shutil.rmtree(elsewhere)

    def test_uploaded_config_gate_rechecks_options_and_device(self) -> None:
        outcome = self.stage(_yaml_bytes(SOH_XGB, task=None))
        root = _data_root()
        try:
            ok = qx.build_quick_plan(task=ct.TASK_SOH, config_path=outcome.path, device="cpu", repo_root=root,
                                     allowed_datasets=["MATR"], cuda_ok=False, uploaded=True)
            self.assertTrue(ok.valid, ok.errors)
            cuda = qx.build_quick_plan(task=ct.TASK_SOH, config_path=outcome.path, device="cuda", repo_root=root,
                                       allowed_datasets=["MATR"], cuda_ok=False, uploaded=True)
            self.assertFalse(cuda.valid)
            tampered = yaml.safe_load(outcome.path.read_text())
            tampered["model"]["workspace"] = "/tmp/elsewhere"
            outcome.path.write_text(yaml.safe_dump(tampered))
            again = qx.build_quick_plan(task=ct.TASK_SOH, config_path=outcome.path, device="cpu", repo_root=root,
                                        allowed_datasets=["MATR"], cuda_ok=False, uploaded=True)
            self.assertFalse(again.valid)
        finally:
            shutil.rmtree(root)

    def test_apply_branch_never_runs_anything(self) -> None:
        """Static: the Apply handler validates and selects; it launches nothing."""
        tree = ast.parse(APP.read_text(encoding="utf-8"))
        panel = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "_render_home_quickstart_panel")
        apply_if = next(n for n in ast.walk(panel) if isinstance(n, ast.If)
                        and "quick_apply_config" in ast.unparse(n.test))
        called = {ast.unparse(c.func) for c in ast.walk(apply_if) if isinstance(c, ast.Call)}
        self.assertIn("config_upload.stage_uploaded_config", called)
        for forbidden in ("_run_home_experiment", "run_command_with_progress", "_begin_page_run"):
            self.assertNotIn(forbidden, called)


# --- P1-3, P2-1, P2-3: backend helpers ----------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_run_metrics_come_from_the_runs_own_manifest(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="synthetic_run_"))
        try:
            rm.write_manifest(tmp, {**rm.build_manifest(run_id="r", project="p", workspace=str(tmp),
                                                        dataset="MATR", task="SOH", model_key="xgb"),
                                    "metrics": {"MAE": 0.01, "RMSE": 0.02, "flag": True}})
            self.assertEqual(qx.run_metrics(tmp), ({"MAE": 0.01, "RMSE": 0.02}, None))
            rm.update_manifest(tmp, {"metrics_error": "ValueError: unreadable"})
            self.assertEqual(qx.run_metrics(tmp)[1], "ValueError: unreadable")
            self.assertEqual(qx.run_metrics(tmp / "missing")[0], {})
        finally:
            shutil.rmtree(tmp)
        source = APP.read_text(encoding="utf-8")
        self.assertNotIn("pickle.load", source)
        self.assertNotIn("_read_run_metrics", source)

    def test_manifest_sync_failure_is_reported(self) -> None:
        def ok(command, cwd):
            self.assertEqual(Path(command[1]), REPO_ROOT / qx.MANIFEST_SYNC_TOOL)
            return {"returncode": 0, "stdout": "", "stderr": ""}
        self.assertEqual(qx.sync_manifests(REPO_ROOT, python="python", run=ok), (True, ""))
        failed = qx.sync_manifests(REPO_ROOT, python="python",
                                   run=lambda c, cwd: {"returncode": 3, "stdout": "", "stderr": "boom\nPermissionError: x"})
        self.assertEqual(failed[0], False)
        self.assertIn("exited 3", failed[1])
        self.assertIn("PermissionError", failed[1])

        def missing(command, cwd):
            raise FileNotFoundError("python")
        self.assertFalse(qx.sync_manifests(REPO_ROOT, python="python", run=missing)[0])

    def test_download_failure_names_the_host_that_failed(self) -> None:
        line = ("error: Could not download data/raw/HUST/x.zip from https://zenodo.org/record/1/x.zip: "
                "ConnectionError: HTTPSConnectionPool(host='zenodo.org', port=443): Max retries exceeded")
        self.assertEqual(qx.download_failure_host("[INFO] data.matr.io file ok\n" + line), "zenodo.org")
        self.assertEqual(qx.download_failure_host("HTTPSConnectionPool(host='a.b', port=443)"), "a.b")
        self.assertIsNone(qx.download_failure_host("Traceback: something else"))

    def test_cli_network_error_names_the_real_host_and_hides_credentials(self) -> None:
        from batteryml_web.core import run_batteryml_cli

        request = requests.Request("GET", "https://user:secret@files.example.org:8443/a?token=abc").prepare()
        error = requests.ConnectionError("refused", request=request)
        stderr = io.StringIO()
        with mock.patch("runpy.run_path", side_effect=error), redirect_stderr(stderr), \
                mock.patch.object(sys, "argv", ["run_batteryml_cli.py", "run", "x.yaml"]):
            with self.assertRaises(SystemExit) as ctx:
                run_batteryml_cli.main()
        self.assertEqual(ctx.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn("files.example.org:8443", message)
        self.assertNotIn("data.matr.io", message)
        self.assertNotIn("secret", message)

    def test_download_messages_redact_credentials(self) -> None:
        from batteryml_web.core.safe_download import redact

        text = redact("https://u:p@h.org/f?token=t0k&x=1 and url: /f?signature=s1g")
        self.assertNotIn("p@", text)
        self.assertNotIn("t0k", text)
        self.assertNotIn("s1g", text)
        self.assertIn("h.org", text)


# --- page behaviour (Streamlit AppTest; fake run, fake readiness) -------------------

class _ReadyMATR:
    """The real readiness report for MATR, reported as downloaded and processed.

    Wraps the real report (so every other reader of it keeps working) and
    changes only what Quick experience reads to enable its Run button.
    """

    def __init__(self, report: Any, root: Path) -> None:
        self._report = report
        self._root = root

    def __getattr__(self, name: str) -> Any:
        return getattr(self._report, name)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self._report.to_dict())
        if str(getattr(self._report, "dataset", "")).upper() == "MATR":
            data.update(raw_available=True, processed_available=True, preprocessing_supported=True,
                        raw_path=str(self._root / "raw"), processed_path=str(self._root / "processed"))
        return data


SOH_PAIRS = [(f"synthetic_s{c}", cycle) for c in range(2) for cycle in (10, 15, 20)]
SOH_OBS = [0.99, 0.98, 0.97, 0.995, 0.985, 0.975]


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestQuickExperiencePage(unittest.TestCase):
    """The mentor's Quick experience panel, driven like a user would."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="synthetic_quick_page_"))
        self.project = self.tmp / "quick_project"
        self.project.mkdir()
        self.calls: List[List[str]] = []
        self.fail_next = False
        self.stack = ExitStack()
        original_plan = qx.build_quick_plan

        def plan_without_data(**kwargs):
            kwargs["require_data"] = False  # no data/processed in the test environment
            return original_plan(**kwargs)

        self.stack.enter_context(mock.patch.object(qx, "build_quick_plan", plan_without_data))
        from batteryml_web.core import readiness

        real_assess = readiness.assess_dataset
        self.stack.enter_context(mock.patch.object(
            readiness, "assess_dataset", lambda *a, **k: _ReadyMATR(real_assess(*a, **k), self.tmp)))
        self.stack.enter_context(mock.patch("batteryml_web.ui.progress_utils.run_command_with_progress",
                                            self.fake_run))

    def tearDown(self) -> None:
        self.stack.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_run(self, *, command, cwd, log_path, steps, container):
        self.calls.append(list(command))
        workspace = Path(cwd) / command[command.index("--workspace") + 1]
        seed = int(command[command.index("--seed") + 1])
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if self.fail_next:
            Path(log_path).write_text("Traceback ...\nValueError: synthetic failure\n")
            return {"success": False, "return_code": 1, "elapsed": 0.1, "log_path": str(log_path),
                    "output": "ValueError: synthetic failure", "output_tail": "ValueError: synthetic failure"}
        from tests.test_paper_figures import soh_payload

        shutil.copyfile(command[command.index("run") + 1], workspace / "config_20260101000000.yaml")
        pred = [o + 0.01 for o in SOH_OBS]
        payload = soh_payload(SOH_PAIRS, SOH_OBS, pred, {"RMSE": 0.0123, "MAE": 0.0101}, seed)
        (workspace / f"predictions_seed_{seed}_20260101000000.pkl").write_bytes(pickle.dumps(payload))
        Path(log_path).write_text("synthetic run\n")
        return {"success": True, "return_code": 0, "elapsed": 0.1, "log_path": str(log_path),
                "output": "synthetic run", "output_tail": "synthetic run"}

    def app(self) -> "AppTest":
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.session_state["project_root"] = str(self.project)
        at.session_state["project_name"] = self.project.name
        return at

    def assertHealthy(self, at) -> None:
        """No exception and no error-boundary page ("Something went wrong")."""
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        boundary = [e.value for e in at.error if "Something went wrong" in str(e.value)]
        self.assertEqual(boundary, [])

    def select(self, at, key: str, value: Any) -> None:
        at.selectbox(key=key).set_value(value).run()
        self.assertHealthy(at)

    def value(self, at, key: str) -> Any:
        return at.selectbox(key=key).value

    def test_task_switch_clears_stale_config_and_runs_soh(self) -> None:
        at = self.app()
        at.run()
        self.assertHealthy(at)
        self.select(at, "quick_model_selector", "cnn")
        self.assertEqual(self.value(at, "quick_model_selector"), "cnn")      # model choice is honoured
        self.select(at, "quick_config_selector", str(RUL_CNN))
        self.select(at, "quick_task_selector", ct.TASK_SOH)
        config = self.value(at, "quick_config_selector")
        self.assertNotEqual(config, str(RUL_CNN))
        self.assertEqual(ct.resolve_config_task(yaml.safe_load(Path(config).read_text())).task, ct.TASK_SOH)
        self.assertTrue(any("no longer selected" in c.value for c in at.caption))
        self.assertEqual(at.session_state["selected_task"], ct.TASK_SOH)
        self.select(at, "quick_model_selector", "xgb")
        self.assertEqual(self.value(at, "quick_config_selector"), str(SOH_XGB))
        at.button(key="home_quick_run_experiment").click().run()
        self.assertHealthy(at)
        self.assertEqual(len(self.calls), 1)
        command = self.calls[0]
        self.assertEqual(command[command.index("run") + 1], str(SOH_XGB))
        # ...and back: SOH -> RUL leaves no SOH config selected.
        self.select(at, "quick_task_selector", ct.TASK_RUL)
        rul_config = self.value(at, "quick_config_selector")
        self.assertEqual(ct.resolve_config_task(yaml.safe_load(Path(rul_config).read_text())).task, ct.TASK_RUL)

    def test_completed_and_failed_runs_are_recorded_and_selectable(self) -> None:
        from batteryml_web.core.selected_result import resolve_selected_result

        at = self.app()
        at.run()
        self.select(at, "quick_task_selector", ct.TASK_SOH)
        self.select(at, "quick_model_selector", "xgb")
        at.button(key="home_quick_run_experiment").click().run()
        self.assertHealthy(at)
        self.assertEqual(at.session_state["last_run_status"], "success")
        first_id = at.session_state["last_run_id"]
        manifest_a = rm.find_run(self.project, first_id)
        self.assertEqual((manifest_a["status"], manifest_a["return_code"]), ("COMPLETED", 0))
        self.assertEqual(manifest_a["task"], ct.TASK_SOH)
        self.assertEqual(manifest_a["device"], "cpu")
        self.assertEqual(manifest_a["seed"], "0")
        files_a = manifest_a["artifacts"]["prediction_files"]
        self.assertEqual(len(files_a), 1)
        self.assertAlmostEqual(manifest_a["metrics"]["MAE"], 0.0101)
        # Home shows the run's own recorded metrics.
        self.assertEqual(at.session_state["last_run_metrics"], qx.run_metrics(Path(files_a[0]).parent)[0])

        self.select(at, "quick_model_selector", "ridge")
        at.button(key="home_quick_run_experiment").click().run()
        second_id = at.session_state["last_run_id"]
        self.assertNotEqual(first_id, second_id)
        self.fail_next = True
        at.button(key="home_quick_run_experiment").click().run()
        failed = rm.find_run(self.project, at.session_state["last_run_id"])
        self.assertEqual((failed["status"], failed["return_code"]), ("FAILED", 1))
        self.assertEqual(failed["metrics"], {})
        self.assertIn("synthetic failure", failed["failure"]["message"])
        self.assertEqual(at.session_state["last_run_status"], "failed")

        runs = {run["run_id"]: run for run in rm.list_runs(self.project)}
        self.assertEqual(len(runs), 3)
        project = {"name": self.project.name, "root": str(self.project)}
        for run_id in (first_id, second_id):
            selected = resolve_selected_result(project, rm.find_run(self.project, run_id), repo_root=REPO_ROOT)
            self.assertTrue(selected.available, selected.reason)
            self.assertIn(run_id, selected.predictions_file)          # exactly this run's artifact
        at.session_state["current_step"] = "review_results"
        at.run()
        self.assertHealthy(at)

    def test_no_cuda_renders_and_explains(self) -> None:
        at = self.app()
        with mock.patch.object(qx, "cuda_available", lambda probe=None: (False, "No CUDA here.")):
            at.run()
            self.assertHealthy(at)
            self.assertEqual(self.value(at, "quick_device_selector"), "cpu")
            self.select(at, "quick_device_selector", "cuda")
            self.assertTrue(any("not available on this machine" in w.value for w in at.warning))
            self.assertTrue(at.button(key="home_quick_run_experiment").disabled)
            self.assertEqual(self.value(at, "quick_device_selector"), "cuda")  # never silently replaced
            self.select(at, "quick_device_selector", "cpu")
            self.assertFalse(at.button(key="home_quick_run_experiment").disabled)
        self.assertEqual(self.calls, [])

    def test_validated_upload_is_applied_without_running_and_cleared_on_task_switch(self) -> None:
        outcome = cu.stage_uploaded_config(
            _yaml_bytes(SOH_XGB, task=None), "my_soh.yaml", managed_root=self.project,
            repo_root=REPO_ROOT, allowed_datasets=["MATR"], allowed_tasks=TASKS)
        self.assertTrue(outcome.ok, outcome.errors)
        record = {"path": str(outcome.path), "task": outcome.task, "model_key": outcome.model_key}
        at = self.app()
        at.session_state["quick_uploaded_configs"] = [record]
        at.session_state["quick_pending_upload"] = record
        at.run()
        self.assertHealthy(at)
        self.assertEqual(self.value(at, "quick_task_selector"), ct.TASK_SOH)
        self.assertEqual(self.value(at, "quick_config_selector"), str(outcome.path))
        self.assertFalse(at.button(key="home_quick_run_experiment").disabled)
        self.assertEqual(self.calls, [])                                     # applying ran nothing
        self.select(at, "quick_task_selector", ct.TASK_RUL)
        self.assertNotEqual(self.value(at, "quick_config_selector"), str(outcome.path))

    def test_upload_error_is_shown_and_selection_kept(self) -> None:
        at = self.app()
        at.run()
        before = self.value(at, "quick_config_selector")
        at.session_state["quick_upload_error"] = ["The file is not valid YAML: boom"]
        at.run()
        self.assertTrue(any("was not applied" in e.value and "boom" in e.value for e in at.error))
        self.assertEqual(self.value(at, "quick_config_selector"), before)

    def test_manifest_sync_error_is_visible(self) -> None:
        at = self.app()
        at.session_state["quick_manifest_sync_error"] = "`python tools/sync...` exited 3. PermissionError"
        at.run()
        self.assertTrue(any("could not be refreshed" in e.value for e in at.error))

    def test_choose_another_start_path(self) -> None:
        at = self.app()
        at.session_state["llm_provider"] = "kept"
        at.run()
        self.assertEqual(at.session_state["home_entry"], "quick")
        self.select(at, "quick_task_selector", ct.TASK_SOH)
        at.button(key="home_choose_another").click().run()
        self.assertHealthy(at)
        self.assertNotIn("home_entry", at.session_state)
        self.assertFalse(any(h.value == "Quick experience" for h in at.subheader))
        self.assertTrue(any(h.value == "How would you like to start?" for h in at.subheader))
        self.assertEqual(at.session_state["llm_provider"], "kept")
        at.run()                                                             # stays on the chooser
        self.assertNotIn("home_entry", at.session_state)
        at.button(key="home_choice_button_create").click().run()
        self.assertEqual(at.session_state["home_entry"], "create")
        at.button(key="home_choose_another").click().run()
        at.button(key="home_choice_button_quick").click().run()
        self.assertTrue(any(h.value == "Quick experience" for h in at.subheader))
        self.assertEqual(self.calls, [])


class TestMentorDesignKept(unittest.TestCase):
    """The mentor's Quick experience labels and entry points are all still drawn."""

    def test_labels_and_entry_points(self) -> None:
        source = APP.read_text(encoding="utf-8")
        for text in ("Quick experience", "#### 1 · Prepare data", "#### 2 · Configure model",
                     "#### 3 · Run training", "#### 4 · Review results", '"Prediction task"',
                     '"Prediction model"', '"Protocol configuration"', '"Compute device"',
                     '"Upload YAML configuration (optional)"', '"Apply uploaded configuration"',
                     '"Run experiment"', '"Open Review Results"', "← Choose another start path",
                     "How would you like to start?", "Create new project", "Existing project",
                     "Your BatteryML workflow"):
            with self.subTest(text=text):
                self.assertIn(text, source)


if __name__ == "__main__":
    unittest.main()
