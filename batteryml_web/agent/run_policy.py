"""Deterministic run-time policy for agent executions: smoke mode and failures.

Two backend-owned decisions the Streamlit layer must not invent for itself:

1. **What a smoke run is.** A smoke run is one epoch of the model's own
   *formal* config. Nothing else is changed: no batch size, no support size,
   no learning rate, no config file written or mutated. The single override
   is the planner's ``--epochs`` flag
   (:func:`batteryml_web.agent.execution.build_execution_plan`), so a smoke
   run exercises exactly the code path a benchmark run exercises.

2. **What a failed run means.** A CUDA out-of-memory failure on a shared
   server is not a model defect and not a reason to try another GPU behind
   the user's back. :func:`classify_execution_failure` names the failure and
   states what the user may do; the executor never retries, and the full
   stdout/stderr stay available for the debug view.

No Streamlit import, no subprocess, no network: every function here is pure
except for the optional GPU inventory the caller passes in.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: The only epoch count a smoke run uses. ``--epochs 1`` is validated by
#: ``build_execution_plan`` and asserted again by the executor.
SMOKE_EPOCHS = 1

#: The sentence the UI must show wherever smoke-run output is presented.
SMOKE_NOTICE = (
    "1-epoch smoke test — workflow validation only; not benchmark performance."
)

#: Models whose smoke run is recommended on CPU.
#:
#: BatLiNet: a 1-epoch CPU smoke run of ``configs/external/batlinet/matr_1.yaml``
#: was observed by the user to complete in about 6.8 s, while the same run on
#: ``cuda:0`` died with CUDA OOM because the shared development server's GPUs
#: each had roughly 2 GB free. CPU is therefore the recommended smoke device:
#: it is fast enough for workflow validation and cannot be starved by another
#: tenant's allocation. This is a *recommendation* — an explicit GPU choice is
#: always honoured (milestone A requirement 3).
SMOKE_CPU_RECOMMENDED_MODELS = frozenset({"batlinet"})

#: Per-model rationale shown next to the recommendation. Keys must exist in
#: :data:`SMOKE_CPU_RECOMMENDED_MODELS`.
_SMOKE_CPU_REASONS: Dict[str, str] = {
    "batlinet": (
        "BatLiNet smoke runs default to CPU: a 1-epoch CPU run of the formal "
        "config completes in seconds, while this server's GPUs are shared and "
        "have repeatedly had too little free memory for a CUDA run. Selecting a "
        "GPU below is still allowed."
    ),
}

CPU_DEVICE = "cpu"

#: Substrings that identify a CUDA allocation failure in a run's output.
#: Lower-cased comparison; each is a string PyTorch/CUDA actually emits.
CUDA_OOM_MARKERS: tuple[str, ...] = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cuda_error_out_of_memory",
)

#: Substrings that identify "this machine/driver has no usable CUDA device".
CUDA_UNAVAILABLE_MARKERS: tuple[str, ...] = (
    "no cuda gpus are available",
    "cuda driver initialization failed",
    "found no nvidia driver",
    "cuda error: invalid device ordinal",
    "torch not compiled with cuda enabled",
)


def smoke_epochs_for(enabled: bool) -> Optional[int]:
    """The planner's ``epochs`` argument for smoke mode: 1, or None."""
    return SMOKE_EPOCHS if enabled else None


def recommends_cpu_smoke(model_key: str) -> bool:
    """Whether this model's smoke run should default to CPU."""
    return str(model_key or "").strip().lower() in SMOKE_CPU_RECOMMENDED_MODELS


def smoke_device_recommendation(
    model_key: str,
    *,
    smoke: bool,
    available_devices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Advise a default device for one (model, smoke) pair.

    Args:
        model_key: canonical backend model key (``"batlinet"``, ...).
        smoke: whether the 1-epoch smoke override is enabled.
        available_devices: the device strings actually offered, so the advice
            can never point at a device the user cannot pick.

    Returns ``{"device", "recommended", "reason"}``. ``recommended`` is True
    only when this is a deliberate model-specific recommendation rather than
    the platform's ordinary CPU default. Nothing here selects a device by
    itself: the caller uses ``device`` as the *default* of an editable
    control.
    """
    devices = list(available_devices) if available_devices else [CPU_DEVICE]
    fallback = devices[0] if devices else CPU_DEVICE
    key = str(model_key or "").strip().lower()
    if smoke and recommends_cpu_smoke(key) and CPU_DEVICE in devices:
        return {
            "device": CPU_DEVICE,
            "recommended": True,
            "reason": _SMOKE_CPU_REASONS[key],
        }
    return {
        "device": CPU_DEVICE if CPU_DEVICE in devices else fallback,
        "recommended": False,
        "reason": "",
    }


def gpu_memory_advisory(
    device: str,
    inventory: Optional[Dict[str, Any]],
) -> Optional[str]:
    """A one-line note about the selected GPU's observed free memory.

    Returns None for CPU, and None when nothing was observed — an unknown
    free-memory figure is never presented as a number.
    """
    from ..core.device_inventory import find_device

    name = str(device or "").strip()
    if not name or name == CPU_DEVICE:
        return None
    entry = find_device(name, inventory)
    if entry is None:
        return (
            f"{name} was not enumerated on this machine, so its free memory is "
            "unknown. The run will still be sent to it exactly as selected."
        )
    free = entry.get("memory_free_mib")
    total = entry.get("memory_total_mib")
    if not isinstance(free, int):
        return f"{name} ({entry.get('name')}): free memory is not reported by nvidia-smi."
    suffix = f" of {total} MiB" if isinstance(total, int) else ""
    return (
        f"{name} ({entry.get('name')}) currently reports {free} MiB free{suffix}. "
        "This is a snapshot of a shared machine; another process may take that "
        "memory before the run starts."
    )


def _first_matching_line(text: str, markers: tuple[str, ...]) -> Optional[str]:
    """The first line of ``text`` containing any marker, verbatim."""
    for line in str(text or "").splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            return line.strip()
    return None


def _alternative_gpus(
    device: str,
    inventory: Optional[Dict[str, Any]],
    *,
    minimum_free_mib: int,
) -> List[str]:
    """Enumerated GPUs other than ``device`` with more free memory than it.

    Returned for *display*: the user chooses. Nothing in this module or the
    executor ever switches device automatically (milestone A requirement 4).
    """
    if not isinstance(inventory, dict):
        return []
    others: List[tuple[int, str]] = []
    for entry in inventory.get("devices", []):
        name = str(entry.get("device") or "")
        free = entry.get("memory_free_mib")
        if not name or name == device or not isinstance(free, int):
            continue
        if free > minimum_free_mib:
            others.append((free, f"{name} ({free} MiB free)"))
    others.sort(reverse=True)
    return [label for _, label in others]


def classify_execution_failure(
    execution_result: Optional[Dict[str, Any]],
    *,
    device: str = CPU_DEVICE,
    inventory: Optional[Dict[str, Any]] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Name a failed execution and state what the user may do about it.

    Args:
        execution_result: the dict returned by
            :func:`batteryml_web.agent.executor.execute_plan` (or its
            streaming sibling). ``None`` or a successful result yields
            ``kind="none"``.
        device: the device the failed run was sent to.
        inventory: optional GPU inventory from
            :mod:`batteryml_web.core.device_inventory`, used only to name
            *other* GPUs the user could choose.
        smoke: whether the failed run was already a 1-epoch smoke run.

    Returns ``{"kind", "headline", "actions", "evidence_line", "retried"}``.
    ``kind`` is one of ``none``, ``cuda_oom``, ``cuda_unavailable``,
    ``not_started``, ``timeout``, ``process_failed``. ``retried`` is always
    False: this platform never retries a failed run, on another GPU or
    otherwise. No metric, no traceback summary and no cause is invented —
    ``evidence_line`` is copied verbatim from the run's own output or is
    None.
    """
    if not isinstance(execution_result, dict):
        return _no_failure()
    if execution_result.get("success") is True:
        return _no_failure()

    stdout = str(execution_result.get("stdout") or "")
    stderr = str(execution_result.get("stderr") or "")
    combined = f"{stderr}\n{stdout}"
    error = str(execution_result.get("error") or "")
    returncode = execution_result.get("returncode")

    oom_line = _first_matching_line(combined, CUDA_OOM_MARKERS)
    if oom_line is not None:
        actions = [
            "Re-run on CPU. A 1-epoch CPU smoke run validates the same workflow "
            "and needs no GPU memory."
            if not smoke
            else "Re-run this smoke test on CPU; it needs no GPU memory.",
        ]
        alternatives = _alternative_gpus(
            device, inventory, minimum_free_mib=_observed_free(device, inventory)
        )
        if alternatives:
            actions.append(
                "Or select a GPU that currently reports more free memory: "
                + ", ".join(alternatives)
                + "."
            )
        else:
            actions.append(
                "Or select a GPU with more free memory once one is available; the "
                "GPU table above shows the current snapshot."
            )
        actions.append(
            "Nothing was retried automatically: the platform never moves a failed "
            "run to another GPU on its own."
        )
        return {
            "kind": "cuda_oom",
            "headline": (
                f"CUDA out of memory on {device}. This is a shared-server memory "
                "condition, not a model or workflow defect."
            ),
            "actions": actions,
            "evidence_line": oom_line,
            "retried": False,
        }

    unavailable_line = _first_matching_line(combined, CUDA_UNAVAILABLE_MARKERS)
    if unavailable_line is not None:
        return {
            "kind": "cuda_unavailable",
            "headline": f"CUDA is not usable for {device} on this machine.",
            "actions": [
                "Re-run on CPU.",
                "Check `nvidia-smi` and the installed torch build if a GPU is expected.",
            ],
            "evidence_line": unavailable_line,
            "retried": False,
        }

    if execution_result.get("started") is not True:
        return {
            "kind": "not_started",
            "headline": "The run never started.",
            "actions": [
                "Read the error below; it names the rule or the OS failure that "
                "prevented the subprocess from starting."
            ],
            "evidence_line": error or None,
            "retried": False,
        }

    if "timed out" in error.lower():
        return {
            "kind": "timeout",
            "headline": "The run exceeded its time limit and was stopped.",
            "actions": [
                "Re-run with the 1-epoch smoke option to validate the workflow first."
                if not smoke
                else "The 1-epoch smoke run itself timed out; check the machine's load.",
            ],
            "evidence_line": error or None,
            "retried": False,
        }

    return {
        "kind": "process_failed",
        "headline": (
            f"The BatteryML run exited with return code {returncode}. "
            "No metrics are reported for a failed run."
        ),
        "actions": [
            "Open the debug output below: the full stdout and stderr, including "
            "any traceback, are preserved exactly as the run produced them."
        ],
        "evidence_line": _last_nonempty_line(stderr),
        "retried": False,
    }


def _observed_free(device: str, inventory: Optional[Dict[str, Any]]) -> int:
    """Free MiB on ``device``, or 0 when unknown, for ranking alternatives."""
    from ..core.device_inventory import free_memory_mib

    value = free_memory_mib(device, inventory)
    return int(value) if isinstance(value, int) else 0


def _last_nonempty_line(text: str) -> Optional[str]:
    for line in reversed(str(text or "").splitlines()):
        if line.strip():
            return line.strip()
    return None


def _no_failure() -> Dict[str, Any]:
    return {
        "kind": "none",
        "headline": "",
        "actions": [],
        "evidence_line": None,
        "retried": False,
    }
