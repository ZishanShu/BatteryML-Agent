"""Blockers from the real-server acceptance of PR #13 (ad9ef9c).

1. GPU policy. The server has eight RTX 3090s (sm_86) and PyTorch 1.6 /
   CUDA 10.2, which cannot run kernels on them. ``torch.cuda.is_available()``
   is True there, so cuda used to be offered as usable. It is now refused,
   with the reason, on every route (Quick experience, Run Model, Agent
   Workflow) — and never replaced by the CPU.
2. Review Results opened on "No run is selected": a Quick experience run made
   without a project was written to ``runs/web_projects/quick_experience``,
   which no other page reads; Review Results reads the default project.

Everything here is synthetic: nvidia-smi output and torch attributes are
faked; no GPU, model or dataset is touched.

    python -m unittest tests.test_server_acceptance_fixes -v
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import device_inventory as di  # noqa: E402
from batteryml_web.core import quick_experience as qx  # noqa: E402
from batteryml_web.core import run_manifest as rm  # noqa: E402
from tests.test_quick_experience_fixes import AppTest, _PageHarness  # noqa: E402

RTX3090 = [{"index": i, "name": "NVIDIA GeForce RTX 3090", "capability": (8, 6)} for i in range(8)]
V100 = [{"index": 0, "name": "Tesla V100", "capability": (7, 0)}]


class TestBuildCompatibilityRule(unittest.TestCase):
    def test_pytorchs_own_rule_when_it_lists_its_architectures(self) -> None:
        built_for_turing = ["sm_37", "sm_50", "sm_60", "sm_70", "sm_75", "compute_75"]
        self.assertEqual(di.gpu_build_incompatibilities(RTX3090[:1], arch_list=built_for_turing, torch_cuda="11.0"),
                         ["NVIDIA GeForce RTX 3090 (sm_86)"])
        self.assertEqual(di.gpu_build_incompatibilities(RTX3090[:1], arch_list=["sm_70", "sm_80"], torch_cuda="11.1"), [])
        self.assertEqual(di.gpu_build_incompatibilities(V100, arch_list=built_for_turing, torch_cuda="10.2"), [])

    def test_pytorch_16_without_an_architecture_list(self) -> None:
        """The server: PyTorch 1.6 has no get_arch_list; CUDA 10.2 predates Ampere."""
        self.assertEqual(len(di.gpu_build_incompatibilities(RTX3090, arch_list=None, torch_cuda="10.2")), 8)
        self.assertEqual(di.gpu_build_incompatibilities(RTX3090, arch_list=None, torch_cuda="11.1"), [])
        self.assertEqual(di.gpu_build_incompatibilities(V100, arch_list=None, torch_cuda="10.2"), [])

    def test_undecidable_is_not_a_verdict(self) -> None:
        self.assertIsNone(di.gpu_build_incompatibilities(RTX3090, arch_list=None, torch_cuda=None))


class TestComputeCapabilityQuery(unittest.TestCase):
    @staticmethod
    def runner(returncode=0, stdout="", stderr="", raises=None):
        def run(argv, **kwargs):
            assert kwargs.get("shell") is False and argv[0] == "nvidia-smi"
            if raises:
                raise raises
            return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
        return run

    def test_parsed_from_nvidia_smi(self) -> None:
        devices, why = di.query_compute_capabilities(
            runner=self.runner(stdout="0, NVIDIA GeForce RTX 3090, 8.6\n1, NVIDIA GeForce RTX 3090, 8.6\n"))
        self.assertIsNone(why)
        self.assertEqual([d["capability"] for d in devices], [(8, 6), (8, 6)])

    def test_an_old_driver_or_no_driver_means_unknown(self) -> None:
        devices, why = di.query_compute_capabilities(runner=self.runner(
            returncode=2, stdout='Field "compute_cap" is not a valid field to query.'))
        self.assertIsNone(devices)
        self.assertIn("compute_cap", why)
        devices, why = di.query_compute_capabilities(runner=self.runner(raises=FileNotFoundError("nvidia-smi")))
        self.assertIsNone(devices)
        devices, why = di.query_compute_capabilities(
            runner=self.runner(raises=subprocess.TimeoutExpired("nvidia-smi", 1)))
        self.assertIsNone(devices)
        with self.assertRaises(ValueError):
            di.parse_compute_capabilities("0, RTX, eight-point-six")


class TestCudaAvailable(unittest.TestCase):
    def setUp(self) -> None:
        qx._torch_cuda_status.cache_clear()
        self.addCleanup(qx._torch_cuda_status.cache_clear)

    def _status(self, *, torch_attrs: dict, devices) -> tuple:
        import torch

        with mock.patch.object(torch, "cuda", types.SimpleNamespace(**torch_attrs)), \
                mock.patch.object(torch, "_C", types.SimpleNamespace()), \
                mock.patch.object(torch.version, "cuda", torch_attrs.pop("_cuda", "10.2"), create=True), \
                mock.patch.object(torch, "__version__", "1.6.0"), \
                mock.patch.object(di, "query_compute_capabilities", lambda **_: (devices, None)):
            return qx.cuda_available()

    def test_the_server_build_is_refused_with_the_reason(self) -> None:
        ok, detail = self._status(torch_attrs={"is_available": lambda: True}, devices=RTX3090)
        self.assertFalse(ok)
        self.assertIn("NVIDIA GeForce RTX 3090 (sm_86)", detail)
        self.assertIn("PyTorch 1.6.0", detail)
        problem = qx.device_problem("cuda", cuda_ok=ok, cuda_detail=detail)
        self.assertIn("Choose cpu explicitly", problem)

    def test_a_supporting_build_is_available(self) -> None:
        ok, detail = self._status(
            torch_attrs={"is_available": lambda: True, "get_arch_list": lambda: ["sm_80", "sm_86"], "_cuda": "11.3"},
            devices=RTX3090)
        self.assertEqual((ok, detail), (True, ""))

    def test_unknown_capability_does_not_invent_a_block(self) -> None:
        ok, _ = self._status(torch_attrs={"is_available": lambda: True}, devices=None)
        self.assertTrue(ok)

    def test_no_cuda_at_all(self) -> None:
        ok, detail = self._status(torch_attrs={"is_available": lambda: False}, devices=RTX3090)
        self.assertEqual((ok, detail), (False, qx.NO_CUDA_DETAIL))


class TestAgentGpuMode(unittest.TestCase):
    def test_gpu_mode_blocks_and_never_falls_back(self) -> None:
        from batteryml_web.ui import agent_workflow as aw

        self.assertEqual(aw.gpu_mode_block(mode="GPU", gpu_options=[], chosen_gpu=None, cuda_status=(True, "")),
                         aw.NO_GPU_ENUMERATED)
        reason = aw.gpu_mode_block(mode="GPU", gpu_options=["cuda:0"], chosen_gpu="cuda:0",
                                   cuda_status=(False, "PyTorch 1.6.0 cannot run on RTX 3090 (sm_86)."))
        self.assertIn("sm_86", reason)
        self.assertIn("Choose cpu explicitly", reason)
        self.assertIsNone(aw.gpu_mode_block(mode="GPU", gpu_options=["cuda:0"], chosen_gpu="cuda:0",
                                            cuda_status=(True, "")))
        self.assertIsNone(aw.gpu_mode_block(mode="CPU", gpu_options=[], chosen_gpu=None, cuda_status=(False, "x")))
        plan = {"can_execute": True, "command": ["python", "run"], "reason": ""}
        blocked = aw.block_plan(plan, reason)
        self.assertFalse(aw.run_is_enabled(blocked))
        self.assertEqual(blocked["command"], plan["command"])                 # nothing rewritten to cpu
        self.assertIs(aw.block_plan(plan, None), plan)

    def test_the_page_applies_the_block_before_the_plan_is_stored(self) -> None:
        source = (REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py").read_text(encoding="utf-8")
        self.assertNotIn("Runs will use the CPU", source)
        block = source.index("plan = block_plan(plan, gpu_block)")
        store = source.index('_store_json_state("agent_execution_plan", plan)')
        self.assertLess(block, store)
        self.assertIn("gpu_block = gpu_mode_block(", source)


class TestOneDefaultProject(unittest.TestCase):
    def test_quick_falls_back_to_the_project_review_results_reads(self) -> None:
        tree = ast.parse((REPO_ROOT / "batteryml_web" / "app.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_data_action_root")
        code = "\n".join(ast.unparse(stmt) for stmt in fn.body[1:])      # without the docstring
        self.assertIn("_project_for_runs()['root']", code)
        self.assertNotIn("quick_experience", code)

    def test_an_empty_history_names_where_runs_are(self) -> None:
        from batteryml_web.ui.run_history import projects_with_runs

        tmp = Path(tempfile.mkdtemp(prefix="synthetic_projects_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "default").mkdir()
        (tmp / "empty_other").mkdir()
        run_dir = tmp / "quick_experience" / "quickstart" / "run_synthetic_1"
        run_dir.mkdir(parents=True)
        rm.write_manifest(run_dir, {"run_id": "run_synthetic_1", "status": "COMPLETED",
                                    "created_at": "2026-10-12T00:00:00+00:00"})
        self.assertEqual(projects_with_runs(tmp, exclude="default"), [("quick_experience", 1)])
        self.assertEqual(projects_with_runs(tmp / "missing"), [])


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestDefaultProjectOnThePage(_PageHarness, unittest.TestCase):
    def test_without_a_project_quick_runs_where_review_results_looks(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / "batteryml_web" / "app.py"), default_timeout=300)
        at.run()                                    # no project selected: nothing in session
        self.assertHealthy(at)
        run_location = [c.value for c in at.code if c.value.endswith(str(Path("runs") / "web_projects" / "default"))]
        self.assertEqual(run_location, [str(REPO_ROOT / "runs" / "web_projects" / "default")])
        self.assertEqual(self.calls, [])            # nothing ran


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
