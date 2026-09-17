"""The launch contract for true multi-GPU training.

A structured selection — ``{"mode": "multi_gpu", "gpu_ids": [0, 1]}`` — must
become a real distributed launch, not a device string. ``cuda:0,1`` is not a
torch device and never was; producing it is how a multi-GPU request turns
into a crash inside a run that has already started.

What this builds instead::

    env:  CUDA_VISIBLE_DEVICES=0,1
    argv: torchrun --standalone --nnodes=1 --nproc_per_node=2 <runner> ...

Two details that matter.

**Device mapping.** ``CUDA_VISIBLE_DEVICES`` renumbers devices for the child,
so the physical GPUs the user picked become local ranks 0..N-1 inside the
job. A rank therefore uses ``cuda:{local_rank}``, which is why the runner
never needs to know the physical id — and why picking GPUs 5 and 6 works
exactly like picking 0 and 1.

**No shell.** Everything is an argv list and an explicit environment dict,
executed with ``shell=False`` like every other execution path here. There is
no string to quote wrongly and no shell to interpret a dataset name.

Batch-size policy lives here too, because it is part of the contract rather
than an implementation detail: see :func:`resolve_batch_size`.

No Streamlit, no torch import at module level.
"""

from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional, Sequence

#: The runner torchrun executes. A module path, run with ``-m`` semantics via
#: torchrun's script argument, so no shell resolution is involved.
DISTRIBUTED_RUNNER_MODULE = "batteryml_web.core.distributed_runner"

MODE_CPU = "cpu"
MODE_SINGLE_GPU = "single_gpu"
MODE_MULTI_GPU = "multi_gpu"

#: What ``train_batch_size`` in a config means once several ranks exist.
#: GLOBAL is the conservative choice and the one this platform uses: the
#: number in the scientific config keeps meaning the number of samples per
#: optimizer step, so a 2-GPU run optimizes over the same batch as a 1-GPU
#: run. The alternative — per-rank — silently multiplies the effective batch
#: by the world size, which changes the optimization and therefore the result.
BATCH_POLICY_GLOBAL = "global"

BATCH_POLICY_NOTE = (
    "train_batch_size is treated as the GLOBAL batch size: it is divided by "
    "the number of ranks so that the effective batch per optimizer step is "
    "the same as a single-GPU run. The batch size is never silently "
    "multiplied by the world size."
)


class DistributedPlanError(ValueError):
    """Raised when a distributed launch cannot be built safely."""


def resolve_batch_size(global_batch_size: Any, world_size: int) -> Dict[str, Any]:
    """Per-rank batch size under the global-batch policy.

    Returns ``{"ok", "per_rank", "global", "world_size", "policy", "error"}``.

    Indivisible is an error, not a rounding exercise: a per-rank size that
    does not multiply back to the configured global batch would change the
    optimization silently, and the user can fix it by choosing a different
    number of GPUs or editing the config.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "per_rank": None,
        "global": global_batch_size,
        "world_size": int(world_size),
        "policy": BATCH_POLICY_GLOBAL,
        "error": "",
    }
    try:
        total = int(global_batch_size)
    except (TypeError, ValueError):
        result["error"] = (
            f"train_batch_size {global_batch_size!r} is not an integer, so a "
            "per-rank batch size cannot be derived."
        )
        return result
    if total <= 0:
        result["error"] = f"train_batch_size must be positive, not {total}."
        return result
    if world_size <= 0:
        result["error"] = f"world size must be positive, not {world_size}."
        return result
    if total % world_size != 0:
        result["error"] = (
            f"train_batch_size {total} is not divisible by the {world_size} "
            f"selected GPU(s). Choose a GPU count that divides {total}, or set "
            "a train_batch_size that does. The batch is not being rounded: "
            "that would change the effective batch size and the optimization."
        )
        return result
    result.update(ok=True, per_rank=total // world_size, error="")
    return result


def normalise_gpu_ids(selection: Optional[Dict[str, Any]]) -> List[int]:
    """The physical GPU indices in a selection, sorted and unique."""
    if not isinstance(selection, dict):
        return []
    ids: List[int] = []
    for value in selection.get("gpu_ids") or []:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids))


def build_distributed_launch(
    *,
    selection: Dict[str, Any],
    config_path: str,
    workspace: str,
    seed: Any = 0,
    epochs: Any = None,
    global_batch_size: Any = None,
    python_executable: str = "",
    torchrun_executable: str = "",
    base_env: Optional[Dict[str, str]] = None,
    rdzv_port: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the torchrun argv and environment for a multi-GPU run.

    Returns ``{"ok", "argv", "env", "world_size", "gpu_ids", "batch",
    "error"}``. On failure ``argv`` is empty — a refused plan carries no
    command, so there is nothing for a caller to execute by accident.
    """
    failure: Dict[str, Any] = {
        "ok": False, "argv": [], "env": {}, "world_size": 0,
        "gpu_ids": [], "batch": None, "error": "",
    }

    if str(selection.get("mode")) != MODE_MULTI_GPU:
        failure["error"] = (
            f"build_distributed_launch requires mode {MODE_MULTI_GPU!r}, got "
            f"{selection.get('mode')!r}. Single-GPU and CPU runs use the "
            "ordinary single-process plan."
        )
        return failure

    gpu_ids = normalise_gpu_ids(selection)
    if len(gpu_ids) < 2:
        failure["error"] = (
            "A distributed launch needs at least two distinct GPUs; the "
            f"selection resolved to {gpu_ids}."
        )
        return failure

    world_size = len(gpu_ids)
    batch = None
    if global_batch_size is not None:
        batch = resolve_batch_size(global_batch_size, world_size)
        if not batch["ok"]:
            failure["error"] = batch["error"]
            failure["batch"] = batch
            return failure

    launcher = torchrun_executable or shutil.which("torchrun") or ""
    if not launcher:
        failure["error"] = (
            "torchrun was not found on PATH, so a distributed run cannot be "
            "launched. Install a torch build that provides it, or run on a "
            "single GPU."
        )
        return failure

    argv: List[str] = [
        launcher,
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        "-m",
        DISTRIBUTED_RUNNER_MODULE,
        "--config", str(config_path),
        "--workspace", str(workspace),
        "--seed", str(seed),
    ]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    if batch is not None:
        argv += ["--per-rank-batch-size", str(batch["per_rank"])]
        argv += ["--global-batch-size", str(batch["global"])]

    env = dict(base_env or {})
    # The selected physical GPUs, in order, become local ranks 0..N-1 for the
    # child. This is the only place a physical index appears.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_ids)
    # Each rank talks to one GPU; letting a rank spawn many CPU threads on a
    # shared box is how a multi-GPU job starves its neighbours.
    env.setdefault("OMP_NUM_THREADS", "1")
    if rdzv_port is not None:
        env["MASTER_PORT"] = str(int(rdzv_port))

    return {
        "ok": True,
        "argv": argv,
        "env": env,
        "world_size": world_size,
        "gpu_ids": gpu_ids,
        "batch": batch,
        "error": "",
    }


def device_string_is_never_multi(device: Any) -> bool:
    """True when ``device`` is a single torch device, as it must be.

    Kept as a named predicate so the rule has one home: a comma in a device
    string means someone tried to express multi-GPU as a device, which this
    platform refuses at the planner boundary.
    """
    return "," not in str(device or "")
