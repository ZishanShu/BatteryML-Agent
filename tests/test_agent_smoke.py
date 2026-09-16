"""Backend tests for smoke-run policy, device inventory and failure diagnosis.

Milestone A. Nothing here trains, allocates GPU memory, calls ``nvidia-smi``
for real, or starts a subprocess: the one shell boundary
(:func:`batteryml_web.core.device_inventory.query_gpu_inventory`) takes an
injectable ``runner``, and everything else is pure.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_smoke -v
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import (  # noqa: E402
    REGISTRIES_AVAILABLE,
    REGISTRY_SKIP_MESSAGE,
)
from batteryml_web.agent.execution import build_execution_plan  # noqa: E402
from batteryml_web.agent.executor import execute_plan  # noqa: E402
from batteryml_web.agent.run_policy import (  # noqa: E402
    SMOKE_EPOCHS,
    SMOKE_NOTICE,
    classify_execution_failure,
    gpu_memory_advisory,
    recommends_cpu_smoke,
    smoke_device_recommendation,
    smoke_epochs_for,
)
from batteryml_web.core import device_inventory as di  # noqa: E402

#: The formal BatLiNet config, as a repo-relative path. The planner reports
#: it as an absolute path, because readiness derives config paths from the
#: repository root — so comparisons resolve both sides rather than matching
#: strings. The two spellings name the same file, and the planner's is the
#: correct one: it is unambiguous whatever the process's working directory.
BATLINET_CONFIG = "configs/external/batlinet/matr_1.yaml"
BATLINET_CONFIG_PATH = (REPO_ROOT / BATLINET_CONFIG).resolve()

_TWO_GPU_CSV = (
    "0, NVIDIA GeForce RTX 3090, 24576, 22500, 2076, 97\n"
    "1, NVIDIA GeForce RTX 3090, 24576, 14000, 10576, 12\n"
)

_TWO_GPU_INVENTORY = {
    "source": "nvidia-smi",
    "available": True,
    "device_count": 2,
    "devices": di.parse_nvidia_smi_gpu_csv(_TWO_GPU_CSV),
    "error": None,
}


def _runner(stdout: str = "", returncode: int = 0, stderr: str = ""):
    calls: list = []

    def run(argv, **kwargs):
        calls.append({"argv": list(argv), "kwargs": dict(kwargs)})
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    run.calls = calls  # type: ignore[attr-defined]
    return run


class TestNvidiaSmiParsing(unittest.TestCase):
    def test_rows_parse_positionally(self) -> None:
        rows = di.parse_nvidia_smi_gpu_csv(_TWO_GPU_CSV)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["device"], "cuda:0")
        self.assertEqual(rows[0]["name"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(rows[0]["memory_total_mib"], 24576)
        self.assertEqual(rows[0]["memory_used_mib"], 22500)
        self.assertEqual(rows[0]["memory_free_mib"], 2076)
        self.assertEqual(rows[1]["device"], "cuda:1")

    def test_blank_lines_ignored_and_empty_is_empty(self) -> None:
        self.assertEqual(di.parse_nvidia_smi_gpu_csv(""), [])
        self.assertEqual(di.parse_nvidia_smi_gpu_csv("\n   \n"), [])

    def test_not_available_memory_is_none_not_zero(self) -> None:
        rows = di.parse_nvidia_smi_gpu_csv("0, GPU, [N/A], [N/A], [N/A], [N/A]\n")
        self.assertIsNone(rows[0]["memory_free_mib"])
        self.assertIsNone(rows[0]["memory_total_mib"])

    def test_malformed_row_raises_and_names_the_line(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            di.parse_nvidia_smi_gpu_csv("0, GPU, 24576, 1\n")
        self.assertIn("expected 6", str(ctx.exception))
        self.assertIn("0, GPU, 24576", str(ctx.exception))

    def test_non_integer_index_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            di.parse_nvidia_smi_gpu_csv("GPU-abc, GPU, 1, 1, 1, 1\n")
        self.assertIn("non-integer GPU index", str(ctx.exception))


class TestGpuInventoryQuery(unittest.TestCase):
    def test_argv_list_and_no_shell(self) -> None:
        run = _runner(stdout=_TWO_GPU_CSV)
        inventory = di.query_gpu_inventory(runner=run)
        self.assertTrue(inventory["available"])
        self.assertEqual(inventory["device_count"], 2)
        call = run.calls[0]  # type: ignore[attr-defined]
        self.assertIsInstance(call["argv"], list)
        self.assertEqual(call["argv"][0], "nvidia-smi")
        self.assertIs(call["kwargs"]["shell"], False)

    def test_device_count_is_discovered_not_hardcoded(self) -> None:
        for count in (0, 1, 3, 8):
            csv = "".join(
                f"{i}, NVIDIA GeForce RTX 3090, 24576, 100, 24476, 0\n" for i in range(count)
            )
            inventory = di.query_gpu_inventory(runner=_runner(stdout=csv))
            self.assertEqual(inventory["device_count"], count)

    def test_missing_nvidia_smi_is_reported_not_raised(self) -> None:
        def run(argv, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        inventory = di.query_gpu_inventory(runner=run)
        self.assertFalse(inventory["available"])
        self.assertEqual(inventory["devices"], [])
        self.assertIn("nvidia-smi", inventory["error"])

    def test_timeout_is_reported_not_raised(self) -> None:
        def run(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)

        inventory = di.query_gpu_inventory(runner=run, timeout=1)
        self.assertFalse(inventory["available"])
        self.assertIn("did not return", inventory["error"])

    def test_nonzero_returncode_is_reported(self) -> None:
        inventory = di.query_gpu_inventory(
            runner=_runner(returncode=9, stderr="driver mismatch")
        )
        self.assertFalse(inventory["available"])
        self.assertIn("exited 9", inventory["error"])
        self.assertIn("driver mismatch", inventory["error"])

    def test_malformed_output_becomes_an_error_field(self) -> None:
        inventory = di.query_gpu_inventory(runner=_runner(stdout="garbage line\n"))
        self.assertFalse(inventory["available"])
        self.assertIn("expected 6", inventory["error"])


class TestDeviceChoices(unittest.TestCase):
    def test_choices_follow_the_enumerated_devices(self) -> None:
        self.assertEqual(
            di.device_choices(_TWO_GPU_INVENTORY), ["cpu", "cuda:0", "cuda:1"]
        )

    def test_manual_gpu_only_when_nothing_enumerated(self) -> None:
        empty = di.empty_inventory("nvidia-smi was not found on PATH")
        self.assertEqual(di.device_choices(empty), ["cpu"])
        self.assertEqual(di.device_choices_with_manual_gpu(empty), ["cpu", "cuda:0"])
        self.assertEqual(
            di.device_choices_with_manual_gpu(_TWO_GPU_INVENTORY),
            ["cpu", "cuda:0", "cuda:1"],
        )

    def test_labels_use_observed_values_only(self) -> None:
        self.assertEqual(di.device_label("cpu", _TWO_GPU_INVENTORY), "cpu")
        self.assertIn("2076 MiB free", di.device_label("cuda:0", _TWO_GPU_INVENTORY))
        self.assertIn(
            "availability unknown", di.device_label("cuda:7", _TWO_GPU_INVENTORY)
        )

    def test_free_memory_is_none_when_unknown(self) -> None:
        self.assertEqual(di.free_memory_mib("cuda:1", _TWO_GPU_INVENTORY), 10576)
        self.assertIsNone(di.free_memory_mib("cuda:5", _TWO_GPU_INVENTORY))
        self.assertIsNone(di.free_memory_mib("cpu", _TWO_GPU_INVENTORY))

    def test_rows_report_unknown_rather_than_zero(self) -> None:
        inventory = {"devices": di.parse_nvidia_smi_gpu_csv("0, GPU, [N/A], [N/A], [N/A], [N/A]\n")}
        rows = di.inventory_rows(inventory)
        self.assertEqual(rows[0]["Free (MiB)"], "unknown")


class TestDeviceValidation(unittest.TestCase):
    """One torch device per run; no multi-GPU string is ever accepted."""

    def test_single_devices_pass_through(self) -> None:
        for good in ("cpu", "cuda", "cuda:0", "cuda:7", "mps", " cuda:3 "):
            with self.subTest(device=good):
                self.assertEqual(di.validate_device(good), good.strip())

    def test_multi_device_strings_are_refused_with_the_reason(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            di.validate_device("cuda:0,1")
        message = str(ctx.exception)
        self.assertIn("more than one device", message)
        self.assertIn("GPU_MULTI_GPU_AUDIT", message)

    def test_other_nonsense_is_refused(self) -> None:
        for bad in ("", "   ", "gpu", "cuda:x", "cuda:-1", "cuda: 0"):
            with self.subTest(device=bad):
                with self.assertRaises(ValueError):
                    di.validate_device(bad)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_the_planner_refuses_a_multi_gpu_device(self) -> None:
        with self.assertRaises(ValueError):
            build_execution_plan(
                "MATR", "RUL prediction", "batlinet", device="cuda:0,1", epochs=1
            )

    def test_no_multi_gpu_machinery_is_exposed_by_the_web_layer(self) -> None:
        """The audit concludes there is none; no code may use one.

        Checked on the parsed syntax tree — names called and names imported —
        so the prose in run_policy.py that explains why there is no torchrun
        path does not count as one.
        """
        import ast

        forbidden = {
            "DataParallel",
            "DistributedDataParallel",
            "init_process_group",
            "torchrun",
            "spawn",
        }
        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "batteryml_web").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = (
                        func.id
                        if isinstance(func, ast.Name)
                        else func.attr
                        if isinstance(func, ast.Attribute)
                        else ""
                    )
                    if name in forbidden:
                        offenders.append(f"{path.name}:{node.lineno}: calls {name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if "distributed" in node.module or "parallel" in node.module:
                        offenders.append(f"{path.name}:{node.lineno}: imports {node.module}")
                    for alias in node.names:
                        if alias.name in forbidden:
                            offenders.append(
                                f"{path.name}:{node.lineno}: imports {alias.name}"
                            )
        self.assertEqual(offenders, [])

    def test_the_audit_document_exists_and_states_the_verdict(self) -> None:
        audit = (REPO_ROOT / "docs" / "GPU_MULTI_GPU_AUDIT.md").read_text(encoding="utf-8")
        self.assertIn("Multi-GPU execution is not implemented", audit)
        self.assertIn("no multi-GPU Run is offered", audit)
        self.assertIn("DataBundle.to(device)", audit)
        self.assertIn("What a real implementation would require", audit)


class TestSmokePolicy(unittest.TestCase):
    def test_smoke_epochs_is_one_or_none(self) -> None:
        self.assertEqual(smoke_epochs_for(True), 1)
        self.assertEqual(SMOKE_EPOCHS, 1)
        self.assertIsNone(smoke_epochs_for(False))

    def test_notice_wording_is_the_required_sentence(self) -> None:
        self.assertEqual(
            SMOKE_NOTICE,
            "1-epoch smoke test — workflow validation only; not benchmark performance.",
        )

    def test_batlinet_smoke_recommends_cpu(self) -> None:
        self.assertTrue(recommends_cpu_smoke("batlinet"))
        advice = smoke_device_recommendation(
            "batlinet", smoke=True, available_devices=["cpu", "cuda:0", "cuda:1"]
        )
        self.assertEqual(advice["device"], "cpu")
        self.assertTrue(advice["recommended"])
        self.assertIn("CPU", advice["reason"])

    def test_recommendation_never_points_at_an_unavailable_device(self) -> None:
        advice = smoke_device_recommendation(
            "batlinet", smoke=True, available_devices=["cuda:3"]
        )
        self.assertEqual(advice["device"], "cuda:3")
        self.assertFalse(advice["recommended"])

    def test_non_smoke_and_other_models_get_no_special_recommendation(self) -> None:
        for model, smoke in (("batlinet", False), ("cptransformer", True)):
            advice = smoke_device_recommendation(
                model, smoke=smoke, available_devices=["cpu", "cuda:0"]
            )
            self.assertEqual(advice["device"], "cpu")
            self.assertFalse(advice["recommended"])
            self.assertEqual(advice["reason"], "")

    def test_gpu_advisory_reports_only_observed_memory(self) -> None:
        self.assertIsNone(gpu_memory_advisory("cpu", _TWO_GPU_INVENTORY))
        text = gpu_memory_advisory("cuda:0", _TWO_GPU_INVENTORY)
        self.assertIn("2076 MiB free", text)
        unknown = gpu_memory_advisory("cuda:6", _TWO_GPU_INVENTORY)
        self.assertIn("not enumerated", unknown)
        self.assertNotIn("MiB free", unknown)


class TestFailureClassification(unittest.TestCase):
    def _failed(self, **overrides) -> dict:
        base = {
            "started": True,
            "completed": True,
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "",
            "error": "wrapper exited with returncode 1",
        }
        base.update(overrides)
        return base

    def test_success_and_none_yield_no_failure(self) -> None:
        self.assertEqual(classify_execution_failure(None)["kind"], "none")
        self.assertEqual(
            classify_execution_failure({"success": True})["kind"], "none"
        )

    def test_cuda_oom_is_named_with_the_run_s_own_line(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"bin/batteryml.py\", line 120, in run\n"
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        )
        diagnosis = classify_execution_failure(
            self._failed(stderr=stderr),
            device="cuda:0",
            inventory=_TWO_GPU_INVENTORY,
            smoke=True,
        )
        self.assertEqual(diagnosis["kind"], "cuda_oom")
        self.assertIn("cuda:0", diagnosis["headline"])
        self.assertIn("CUDA out of memory", diagnosis["evidence_line"])
        self.assertFalse(diagnosis["retried"])
        joined = " ".join(diagnosis["actions"]).lower()
        self.assertIn("cpu", joined)
        self.assertIn("never moves a failed", joined)

    def test_oom_lists_only_gpus_with_more_free_memory(self) -> None:
        diagnosis = classify_execution_failure(
            self._failed(stderr="CUDA out of memory"),
            device="cuda:0",
            inventory=_TWO_GPU_INVENTORY,
        )
        joined = " ".join(diagnosis["actions"])
        self.assertIn("cuda:1", joined)
        self.assertNotIn("cuda:0 (", joined)

    def test_oom_without_inventory_invents_no_gpu(self) -> None:
        diagnosis = classify_execution_failure(
            self._failed(stderr="CUDA out of memory"), device="cuda:0", inventory=None
        )
        joined = " ".join(diagnosis["actions"])
        self.assertNotIn("cuda:1", joined)
        self.assertNotIn("MiB free", joined)

    def test_cuda_unavailable_is_distinct_from_oom(self) -> None:
        diagnosis = classify_execution_failure(
            self._failed(stderr="RuntimeError: No CUDA GPUs are available"),
            device="cuda:0",
        )
        self.assertEqual(diagnosis["kind"], "cuda_unavailable")

    def test_not_started_and_plain_failure(self) -> None:
        self.assertEqual(
            classify_execution_failure(
                self._failed(started=False, completed=False, error="refused: bad plan")
            )["kind"],
            "not_started",
        )
        plain = classify_execution_failure(
            self._failed(stderr="AssertionError: data/processed/MATR")
        )
        self.assertEqual(plain["kind"], "process_failed")
        self.assertIn("return code 1", plain["headline"])
        self.assertEqual(plain["evidence_line"], "AssertionError: data/processed/MATR")

    def test_no_metric_is_ever_produced_by_the_diagnosis(self) -> None:
        diagnosis = classify_execution_failure(
            self._failed(stdout="RMSE 794.32 MAE 767.91 MAPE 1.35"),
        )
        blob = repr(diagnosis)
        for token in ("RMSE", "MAE", "MAPE"):
            self.assertNotIn(token, blob)


class TestExecutorNeverRetries(unittest.TestCase):
    """A CUDA OOM must not be retried, on another GPU or otherwise."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_failed_run_calls_the_runner_exactly_once(self) -> None:
        plan = build_execution_plan(
            "MATR",
            "RUL prediction",
            "batlinet",
            device="cuda:0",
            epochs=1,
            workspace="runs/agent_matr_batlinet_smoke",
        )
        self.assertTrue(plan["can_execute"])
        run = _runner(returncode=1, stderr="torch.cuda.OutOfMemoryError: CUDA out of memory")
        result = execute_plan(plan, runner=run)
        self.assertEqual(len(run.calls), 1)  # type: ignore[attr-defined]
        self.assertFalse(result["success"])
        self.assertIn("no retry performed", result["error"])
        self.assertNotIn("cuda:1", " ".join(result["command"]))


class TestBatLiNetSmokePlan(unittest.TestCase):
    """The smoke plan uses the formal config and changes only --epochs."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_smoke_plan_is_epochs_one_on_the_formal_config(self) -> None:
        plan = build_execution_plan(
            "MATR",
            "RUL prediction",
            "batlinet",
            device="cpu",
            seed=0,
            epochs=smoke_epochs_for(True),
            workspace="runs/agent_matr_batlinet_smoke",
        )
        self.assertTrue(plan["can_execute"])
        self.assertEqual(
            Path(plan["config_path"]).resolve(),
            BATLINET_CONFIG_PATH,
            "the plan must point at the formal BatLiNet config, however it is spelled",
        )
        self.assertTrue(BATLINET_CONFIG_PATH.is_file())
        command = plan["command"]
        self.assertIn("--epochs", command)
        self.assertEqual(command[command.index("--epochs") + 1], "1")
        self.assertIn("--train", command)
        self.assertIn("--eval", command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_non_smoke_plan_keeps_the_config_schedule(self) -> None:
        plan = build_execution_plan(
            "MATR", "RUL prediction", "batlinet", device="cpu", epochs=None
        )
        self.assertTrue(plan["can_execute"])
        self.assertNotIn("--epochs", plan["command"])

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_explicit_gpu_choice_is_honoured(self) -> None:
        plan = build_execution_plan(
            "MATR", "RUL prediction", "batlinet", device="cuda:3", epochs=1
        )
        self.assertEqual(plan["command"][plan["command"].index("--device") + 1], "cuda:3")

    def test_formal_config_is_not_mutated_for_smoke_runs(self) -> None:
        """No code path writes a smoke variant of the benchmark config."""
        import inspect

        from batteryml_web.agent import execution as execution_module
        from batteryml_web.agent import run_policy as policy_module

        for module in (execution_module, policy_module):
            source = inspect.getsource(module)
            for forbidden in ("write_text(", "yaml.dump", "safe_dump", "open("):
                self.assertNotIn(
                    forbidden,
                    source,
                    f"{module.__name__} must not write config files",
                )

    def test_batlinet_config_declares_the_scientific_settings_unchanged(self) -> None:
        """The smoke override touches epochs only; batch/support stay as shipped."""
        import yaml

        config = yaml.safe_load(BATLINET_CONFIG_PATH.read_text(encoding="utf-8"))
        model = config["model"]
        self.assertEqual(model["name"], "BatLiNetRULPredictor")
        self.assertEqual(model["batch_size"], 1)
        self.assertEqual(model["train_support_size"], 1)
        self.assertEqual(model["test_support_size"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
