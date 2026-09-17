"""The distributed launch contract, the rank rules, and a real torchrun.

These do not need GPUs. The orchestration — argv, environment, rank mapping,
sharding, rank-0-only side effects, batch policy — is what goes wrong in
practice, and all of it is testable on CPU with gloo. One test launches an
actual two-process ``torchrun`` job, because a launch contract that has never
launched anything is a guess.

What these deliberately do NOT claim: that any model trains correctly under
DDP. That needs the model's own source and real GPUs; see
docs/DDP_CONTRACT.md.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_distributed -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.run_policy import multi_gpu_capability  # noqa: E402
from batteryml_web.core.distributed_plan import (  # noqa: E402
    BATCH_POLICY_GLOBAL,
    DISTRIBUTED_RUNNER_MODULE,
    MODE_MULTI_GPU,
    build_distributed_launch,
    device_string_is_never_multi,
    normalise_gpu_ids,
    resolve_batch_size,
)
from batteryml_web.core.distributed_runner import (  # noqa: E402
    is_primary,
    rank_info,
    rank_log_name,
    select_device,
    unwrap,
)


class TestLaunchContract(unittest.TestCase):
    """B6: a structured selection becomes torchrun, never a device string."""

    def _launch(self, gpu_ids, **kwargs):
        return build_distributed_launch(
            selection={"mode": MODE_MULTI_GPU, "gpu_ids": gpu_ids},
            config_path="configs/x.yaml",
            workspace="runs/web_projects/alpha/agent_matr_cnn/run_1",
            seed=0,
            torchrun_executable="/usr/bin/torchrun",
            **kwargs,
        )

    def test_the_launch_is_torchrun_with_one_process_per_gpu(self) -> None:
        plan = self._launch([0, 1])
        self.assertTrue(plan["ok"], plan["error"])
        self.assertIn("--nproc_per_node=2", plan["argv"])
        self.assertIn("--standalone", plan["argv"])
        self.assertIn("--nnodes=1", plan["argv"])
        self.assertIn(DISTRIBUTED_RUNNER_MODULE, plan["argv"])
        self.assertEqual(plan["world_size"], 2)

    def test_no_multi_device_string_is_ever_produced(self) -> None:
        """The specific thing this replaces."""
        plan = self._launch([0, 1])
        blob = " ".join(plan["argv"]) + " " + json.dumps(plan["env"])
        self.assertNotIn("cuda:0,1", blob)
        self.assertNotIn("cuda:0,cuda:1", blob)
        for part in plan["argv"]:
            self.assertTrue(device_string_is_never_multi(part), part)

    def test_the_selected_physical_gpus_are_exported_not_embedded(self) -> None:
        """Picking 5 and 6 must work exactly like picking 0 and 1."""
        plan = self._launch([5, 6])
        self.assertEqual(plan["env"]["CUDA_VISIBLE_DEVICES"], "5,6")
        # The physical ids appear only in the environment; ranks use
        # cuda:{local_rank} after the renumbering.
        self.assertNotIn("5", " ".join(a for a in plan["argv"] if "seed" not in a))

    def test_the_argv_is_a_list_and_no_shell_is_involved(self) -> None:
        plan = self._launch([0, 1])
        self.assertIsInstance(plan["argv"], list)
        for part in plan["argv"]:
            self.assertIsInstance(part, str)
        source = (REPO_ROOT / "batteryml_web/core/distributed_plan.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("shell=True", source)
        self.assertNotIn("os.system", source)

    def test_a_single_gpu_or_cpu_selection_is_refused_here(self) -> None:
        for mode, ids in (("single_gpu", [0]), ("cpu", []), (MODE_MULTI_GPU, [0])):
            with self.subTest(mode=mode, ids=ids):
                plan = build_distributed_launch(
                    selection={"mode": mode, "gpu_ids": ids},
                    config_path="c", workspace="w",
                    torchrun_executable="/usr/bin/torchrun",
                )
                self.assertFalse(plan["ok"])
                self.assertEqual(plan["argv"], [], "a refused plan carries no command")

    def test_duplicate_gpu_ids_do_not_inflate_the_world_size(self) -> None:
        self.assertEqual(normalise_gpu_ids({"gpu_ids": [1, 1, 0]}), [0, 1])
        self.assertEqual(self._launch([1, 1, 0])["world_size"], 2)

    def test_a_missing_torchrun_is_reported_not_guessed(self) -> None:
        plan = build_distributed_launch(
            selection={"mode": MODE_MULTI_GPU, "gpu_ids": [0, 1]},
            config_path="c", workspace="w", torchrun_executable="",
            base_env={"PATH": "/nonexistent"},
        )
        if shutil.which("torchrun"):
            self.skipTest("torchrun is on PATH here, so this path cannot be forced")
        self.assertFalse(plan["ok"])
        self.assertIn("torchrun", plan["error"])


class TestBatchSizePolicy(unittest.TestCase):
    """B5: the effective batch must not change silently."""

    def test_the_configured_batch_is_global_and_is_divided(self) -> None:
        resolved = resolve_batch_size(32, 2)
        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["per_rank"], 16)
        self.assertEqual(resolved["policy"], BATCH_POLICY_GLOBAL)
        # The defining property: per-rank × ranks == the configured batch.
        self.assertEqual(resolved["per_rank"] * resolved["world_size"], 32)

    def test_the_batch_is_never_multiplied_by_the_world_size(self) -> None:
        for world in (2, 4, 8):
            with self.subTest(world=world):
                resolved = resolve_batch_size(64, world)
                self.assertEqual(resolved["per_rank"] * world, 64)
                self.assertLess(resolved["per_rank"], 64)

    def test_an_indivisible_batch_is_blocked_with_an_actionable_message(self) -> None:
        resolved = resolve_batch_size(33, 2)
        self.assertFalse(resolved["ok"])
        self.assertIn("not divisible", resolved["error"])
        self.assertIn("not being rounded", resolved["error"])

    def test_an_indivisible_batch_blocks_the_whole_launch(self) -> None:
        plan = build_distributed_launch(
            selection={"mode": MODE_MULTI_GPU, "gpu_ids": [0, 1, 2]},
            config_path="c", workspace="w", global_batch_size=32,
            torchrun_executable="/usr/bin/torchrun",
        )
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["argv"], [])

    def test_a_nonsense_batch_size_is_refused(self) -> None:
        for value in ("abc", None, 0, -4):
            with self.subTest(value=value):
                self.assertFalse(resolve_batch_size(value, 2)["ok"])


class TestRankRules(unittest.TestCase):
    """B8: exactly one rank performs the write-once side effects."""

    def test_rank_and_world_size_come_from_the_launcher_environment(self) -> None:
        info = rank_info({"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"})
        self.assertEqual((info["rank"], info["local_rank"], info["world_size"]), (3, 1, 8))

    def test_a_missing_environment_describes_a_single_process(self) -> None:
        info = rank_info({})
        self.assertEqual(info, {"rank": 0, "local_rank": 0, "world_size": 1})

    def test_exactly_one_rank_is_primary(self) -> None:
        primaries = [
            is_primary({"rank": rank, "local_rank": rank, "world_size": 4})
            for rank in range(4)
        ]
        self.assertEqual(primaries.count(True), 1)
        self.assertTrue(primaries[0])

    def test_each_rank_gets_its_own_log_name(self) -> None:
        names = {rank_log_name({"rank": r}) for r in range(4)}
        self.assertEqual(len(names), 4)

    def test_a_rank_uses_its_local_index_not_a_physical_id(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.skipTest("asserted for the CPU case in this environment")
        # Without CUDA every rank lands on cpu; the important half is that the
        # physical id never appears — it was consumed by CUDA_VISIBLE_DEVICES.
        self.assertEqual(select_device({"rank": 1, "local_rank": 1, "world_size": 2}), "cpu")

    def test_unwrap_returns_the_plain_module(self) -> None:
        import torch.nn as nn

        plain = nn.Linear(2, 1)
        self.assertIs(unwrap(plain), plain)

        class _Wrapped:
            def __init__(self, module):
                self.module = module

        self.assertIs(unwrap(_Wrapped(plain)), plain)

    def test_checkpoints_are_saved_unwrapped_so_they_load_single_gpu(self) -> None:
        """B10: no `module.` prefix may reach a stored checkpoint."""
        import torch.nn as nn

        class _DDPLike:
            def __init__(self, module):
                self.module = module

        model = nn.Linear(2, 1)
        wrapped = _DDPLike(model)
        keys = list(unwrap(wrapped).state_dict())
        self.assertTrue(keys)
        for key in keys:
            self.assertFalse(key.startswith("module."), key)


class TestRealTorchrunLaunch(unittest.TestCase):
    """A launch contract that has never launched anything is a guess.

    Runs a genuine two-process torchrun job over gloo and reads back what each
    rank reported. No GPU required, and no model is trained.
    """

    @unittest.skipUnless(shutil.which("torchrun"), "torchrun is not installed")
    def test_two_processes_start_with_distinct_ranks_and_one_primary(self) -> None:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [
                shutil.which("torchrun"),
                "--standalone", "--nnodes=1", "--nproc_per_node=2",
                "-m", DISTRIBUTED_RUNNER_MODULE,
                "--config", "unused.yaml",
                "--workspace", "unused",
                "--seed", "0",
                "--per-rank-batch-size", "16",
                "--global-batch-size", "32",
                "--describe-only",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            shell=False,
        )
        self.assertEqual(
            completed.returncode, 0,
            f"torchrun failed:\n{completed.stderr[-2000:]}",
        )
        # Two processes write to one pipe, so their JSON lines can interleave
        # on a single physical line. Scan the stream for objects rather than
        # assuming one per line.
        reports = []
        decoder = json.JSONDecoder()
        text = completed.stdout
        index = 0
        while index < len(text):
            start = text.find("{", index)
            if start == -1:
                break
            try:
                obj, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(obj, dict) and "rank" in obj:
                reports.append(obj)
            index = end
        self.assertEqual(len(reports), 2, completed.stdout)
        self.assertEqual({r["rank"] for r in reports}, {0, 1})
        self.assertEqual({r["world_size"] for r in reports}, {2})
        self.assertEqual([r["is_primary"] for r in reports].count(True), 1)
        # The per-rank batch really reached the workers.
        self.assertEqual({r["per_rank_batch_size"] for r in reports}, {16})
        self.assertEqual({r["global_batch_size"] for r in reports}, {32})


class TestCapabilityGating(unittest.TestCase):
    """B13: multi-GPU stays gated until a model is verified DDP-safe."""

    def test_no_model_is_marked_ddp_capable_without_verification(self) -> None:
        for model in ("cptransformer", "batlinet", "cnn", "xgb"):
            with self.subTest(model=model):
                capability = multi_gpu_capability(model)
                self.assertFalse(capability["supports_multi_gpu"])
                self.assertTrue(capability["reason"])

    def test_batlinet_has_its_own_recorded_reason(self) -> None:
        from batteryml_web.agent.run_policy import MULTI_GPU_BLOCKERS

        reason = MULTI_GPU_BLOCKERS["batlinet"]
        self.assertIn("support", reason.lower())

    def test_the_page_offers_no_multi_gpu_path_at_all(self) -> None:
        """The UI was simplified to CPU / one GPU.

        It no longer consults multi-GPU capability because it no longer
        offers the choice: a user cannot reach a distributed run, so no
        capability check or warning belongs on the page. The capability
        metadata still exists for the backend and for future work.
        """
        import ast

        source = (REPO_ROOT / "batteryml_web/ui/agent_workflow.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree)
        for name in ("CPTransformer", "cptransformer", "batlinet", "BatLiNet"):
            self.assertNotIn(name, code)
        # Neither a model name nor a capability lookup: the offer is gone.
        self.assertNotIn("multi_gpu_capability", code)
        self.assertNotIn("st.multiselect", code)


if __name__ == "__main__":
    unittest.main()
