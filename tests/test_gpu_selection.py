"""GPU selection as a structure, and the multi-GPU capability gate.

Milestone G. The point of these tests is what the platform must *not* do:
build a "cuda:0,1" device string, quietly reduce a multi-GPU choice to one
GPU, or offer a Run button for a combination the runtime cannot execute.

No GPU is required: every inventory is a parsed fixture.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_gpu_selection -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.run_policy import (  # noqa: E402
    MULTI_GPU_CAPABLE_MODELS,
    MULTI_GPU_STRATEGIES,
    MULTI_GPU_UNSUPPORTED_MESSAGE,
    execution_blockers,
    multi_gpu_capability,
)
from batteryml_web.core import device_inventory as di  # noqa: E402

EIGHT_GPU_CSV = "".join(
    f"{i}, NVIDIA GeForce RTX 3090, 24576, 22500, 2076, {40 + i}\n" for i in range(8)
)
EIGHT_GPUS = {
    "source": "nvidia-smi",
    "available": True,
    "device_count": 8,
    "devices": di.parse_nvidia_smi_gpu_csv(EIGHT_GPU_CSV),
    "error": None,
}


class TestInventoryCarriesUtilization(unittest.TestCase):
    def test_utilization_is_parsed(self) -> None:
        self.assertEqual(EIGHT_GPUS["devices"][0]["utilization_pct"], 40)
        self.assertEqual(EIGHT_GPUS["devices"][7]["utilization_pct"], 47)

    def test_unreported_utilization_is_unknown_not_zero(self) -> None:
        rows = di.parse_nvidia_smi_gpu_csv("0, GPU, 24576, 100, 24476, [N/A]\n")
        self.assertIsNone(rows[0]["utilization_pct"])
        self.assertEqual(di.inventory_rows({"devices": rows})[0]["Utilization (%)"], "unknown")

    def test_every_field_the_ui_shows_is_present(self) -> None:
        row = di.inventory_rows(EIGHT_GPUS)[0]
        for column in ("Device", "GPU", "Free (MiB)", "Used (MiB)", "Total (MiB)", "Utilization (%)"):
            self.assertIn(column, row)

    def test_the_gpu_count_is_still_discovered(self) -> None:
        self.assertEqual(EIGHT_GPUS["device_count"], 8)
        self.assertEqual(len(di.device_choices(EIGHT_GPUS)), 9)  # cpu + 8


class TestStructuredSelection(unittest.TestCase):
    def test_no_gpu_is_cpu(self) -> None:
        for empty in ([], None, ()):
            with self.subTest(value=empty):
                selection = di.build_device_selection(empty)
                self.assertEqual(selection, {"mode": di.MODE_CPU, "gpu_ids": []})

    def test_one_gpu_is_single(self) -> None:
        selection = di.build_device_selection([3])
        self.assertEqual(selection, {"mode": di.MODE_SINGLE_GPU, "gpu_ids": [3]})

    def test_several_gpus_are_multi(self) -> None:
        selection = di.build_device_selection([2, 0, 5])
        self.assertEqual(selection["mode"], di.MODE_MULTI_GPU)
        self.assertEqual(selection["gpu_ids"], [0, 2, 5])

    def test_duplicates_collapse_and_order_is_stable(self) -> None:
        self.assertEqual(
            di.build_device_selection([1, 0, 1, 0]), di.build_device_selection([0, 1])
        )

    def test_a_non_integer_index_is_refused(self) -> None:
        for bad in (["0"], [1.5], [True], [-1]):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    di.build_device_selection(bad)

    def test_selection_from_widget_device_names(self) -> None:
        self.assertEqual(
            di.selection_from_devices(["cuda:1", "cuda:0"]),
            {"mode": di.MODE_MULTI_GPU, "gpu_ids": [0, 1]},
        )
        self.assertEqual(di.selection_from_devices([]), {"mode": di.MODE_CPU, "gpu_ids": []})
        self.assertEqual(
            di.selection_from_devices(["cpu"]), {"mode": di.MODE_CPU, "gpu_ids": []}
        )

    def test_a_malformed_device_name_is_refused(self) -> None:
        for bad in (["gpu0"], ["cuda:x"], ["cuda:"]):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    di.selection_from_devices(bad)


class TestNoMultiDeviceStringIsEverBuilt(unittest.TestCase):
    """The central rule of this milestone."""

    def test_cpu_and_single_gpu_produce_a_device_string(self) -> None:
        self.assertEqual(
            di.device_string_for_selection(di.build_device_selection([])), "cpu"
        )
        self.assertEqual(
            di.device_string_for_selection(di.build_device_selection([4])), "cuda:4"
        )

    def test_multi_gpu_produces_no_device_string_at_all(self) -> None:
        selection = di.build_device_selection([0, 1])
        self.assertIsNone(di.device_string_for_selection(selection))

    def test_no_comma_separated_device_can_be_produced(self) -> None:
        for ids in ([0, 1], [0, 1, 2], list(range(8))):
            with self.subTest(ids=ids):
                value = di.device_string_for_selection(di.build_device_selection(ids))
                self.assertIsNone(value)

    def test_such_a_string_would_be_refused_by_the_validator_anyway(self) -> None:
        with self.assertRaises(ValueError):
            di.validate_device("cuda:0,1")

    def test_no_string_literal_in_the_web_layer_is_a_multi_device_name(self) -> None:
        """Docstrings may quote the forbidden form; a literal must not be one.

        Module, class and function docstrings are excluded by identity, so
        prose explaining why ``cuda:0,1`` is refused does not trip the check
        while a literal used as a device name would.
        """
        import ast
        import re

        pattern = re.compile(r"cuda:\s*\d+\s*,")
        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "batteryml_web").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    body = getattr(node, "body", [])
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and pattern.search(node.value)
                ):
                    offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
        self.assertEqual(offenders, [])

    def test_every_device_reaching_the_planner_is_validated(self) -> None:
        """The behavioural guarantee a source scan cannot give.

        However a device string is produced, ``build_execution_plan`` runs it
        through ``validate_device`` before building any argv, so a
        multi-device value cannot reach the CLI even if some future caller
        assembles one.
        """
        import ast
        import inspect

        from batteryml_web.agent import execution

        tree = ast.parse(inspect.getsource(execution.build_execution_plan))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("validate_device", called)
        with self.assertRaises(ValueError):
            di.validate_device("cuda:0,1")

    def test_describe_selection_is_for_humans_not_for_torch(self) -> None:
        self.assertEqual(di.describe_selection(di.build_device_selection([])), "CPU")
        self.assertEqual(di.describe_selection(di.build_device_selection([2])), "GPU 2")
        self.assertEqual(
            di.describe_selection(di.build_device_selection([0, 1])), "GPUs 0, 1"
        )


class TestMultiGpuCapabilityGate(unittest.TestCase):
    def test_no_model_is_registered_as_multi_gpu_capable(self) -> None:
        """The audit's conclusion, kept true by an empty registry."""
        self.assertEqual(MULTI_GPU_CAPABLE_MODELS, frozenset())
        self.assertEqual(MULTI_GPU_STRATEGIES, {})

    def test_capability_reports_false_with_a_reason(self) -> None:
        for model in ("batlinet", "cptransformer", "xgb", "anything"):
            with self.subTest(model=model):
                capability = multi_gpu_capability(model)
                self.assertFalse(capability["supports_multi_gpu"])
                self.assertIsNone(capability["multi_gpu_strategy"])
                self.assertIn(MULTI_GPU_UNSUPPORTED_MESSAGE, capability["reason"])
                self.assertIn("GPU_MULTI_GPU_AUDIT", capability["reason"])

    def test_cpu_and_single_gpu_are_not_blocked(self) -> None:
        for ids in ([], [0], [7]):
            with self.subTest(ids=ids):
                self.assertEqual(
                    execution_blockers("batlinet", di.build_device_selection(ids)), []
                )

    def test_multi_gpu_is_blocked_with_the_exact_message(self) -> None:
        blockers = execution_blockers("batlinet", di.build_device_selection([0, 1]))
        self.assertEqual(len(blockers), 1)
        self.assertTrue(blockers[0].startswith(MULTI_GPU_UNSUPPORTED_MESSAGE))

    def test_a_capable_model_would_pass_the_gate(self) -> None:
        """The gate reads the registry, so registering a model is all it takes."""
        import batteryml_web.agent.run_policy as policy

        original_models = policy.MULTI_GPU_CAPABLE_MODELS
        original_strategies = dict(policy.MULTI_GPU_STRATEGIES)
        policy.MULTI_GPU_CAPABLE_MODELS = frozenset({"future_model"})
        policy.MULTI_GPU_STRATEGIES = {"future_model": "ddp"}
        try:
            capability = policy.multi_gpu_capability("future_model")
            self.assertTrue(capability["supports_multi_gpu"])
            self.assertEqual(capability["multi_gpu_strategy"], "ddp")
            self.assertEqual(
                policy.execution_blockers(
                    "future_model", di.build_device_selection([0, 1])
                ),
                [],
            )
        finally:
            policy.MULTI_GPU_CAPABLE_MODELS = original_models
            policy.MULTI_GPU_STRATEGIES = original_strategies

    def test_a_blocked_selection_is_never_rewritten(self) -> None:
        """The gate reports; it does not reduce the choice to one GPU."""
        selection = di.build_device_selection([0, 1])
        execution_blockers("batlinet", selection)
        self.assertEqual(selection, {"mode": di.MODE_MULTI_GPU, "gpu_ids": [0, 1]})


class TestLiveProgressIsNotRegressed(unittest.TestCase):
    """H: the streaming execution path must survive this milestone."""

    def test_the_streaming_entry_point_is_still_the_only_execution_path(self) -> None:
        """The chain grew a link; it did not grow a second path.

        Execution used to be UI -> execute_plan_streaming. It is now
        UI -> RunManager -> execute_plan_streaming, so that the script is not
        blocked and Cancel can work. What must stay true is that the
        streaming executor remains the *only* thing that starts a run, and
        that the page still spawns nothing itself.
        """
        ui_source = (
            REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py"
        ).read_text(encoding="utf-8")
        manager_source = (
            REPO_ROOT / "batteryml_web" / "core" / "run_manager.py"
        ).read_text(encoding="utf-8")

        # The page starts runs only by handing them to the manager.
        self.assertIn("get_run_manager()", ui_source)
        self.assertNotIn("import subprocess", ui_source)
        self.assertNotIn("popen", ui_source.lower())

        # The manager starts them only through the streaming executor.
        self.assertIn("execute_plan_streaming(", manager_source)
        self.assertIn(
            "from ..agent.streaming import execute_plan_streaming", manager_source
        )

    def test_the_page_still_renders_real_progress_events(self) -> None:
        """The progress the page draws is the run's own reported state."""
        ui_source = (
            REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py"
        ).read_text(encoding="utf-8")
        # Drawn from the manager's snapshot of the run's latest event...
        self.assertIn("manager.snapshot(run_id)", ui_source)
        self.assertIn("progress_caption(", ui_source)
        # ...and the bar only where the run reported a fraction.
        self.assertIn('event.get("fraction")', ui_source)

    def test_progress_parsing_still_refuses_to_fabricate(self) -> None:
        from batteryml_web.agent.streaming import classify_stage, parse_progress_line

        self.assertIsNone(parse_progress_line("Loading data from data/processed/MATR"))
        self.assertIsNone(classify_stage("Traceback (most recent call last):"))
        parsed = parse_progress_line(
            "Training BatLiNet: 100%|##########| 1/1 [00:06<00:00,  6.80s/it]"
        )
        self.assertEqual((parsed["current"], parsed["total"]), (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
