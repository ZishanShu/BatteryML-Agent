"""Deterministic compute-device inventory for the web layer.

One question, answered from the machine rather than from a constant: which
devices can a BatteryML run be sent to right now, and how much memory is
actually free on each one?

Design constraints (CLAUDE.md §3, §6):

* no Streamlit import — the backend, the verification harness and the tests
  must all be able to call this;
* no hardcoded GPU count. The server has eight RTX 3090s today; that number
  is discovered, never assumed, so a four-GPU or zero-GPU machine reports
  the truth;
* the risky part is one thin function (:func:`query_gpu_inventory`) that
  shells out to ``nvidia-smi`` with an argv list and ``shell=False``. All
  parsing is pure (:func:`parse_nvidia_smi_gpu_csv`) and unit-testable
  without a GPU;
* nothing here executes a model, allocates GPU memory, or imports torch.
  Importing torch to enumerate devices would allocate a CUDA context on a
  shared server; reading ``nvidia-smi`` does not.

Memory is reported in MiB exactly as ``nvidia-smi`` reports it. Free memory
is a snapshot: another process on a shared server can take it between this
call and the run, which is why the UI presents it as information and never
as a guarantee.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence

#: The query passed to ``nvidia-smi``. Field order is load-bearing:
#: :func:`parse_nvidia_smi_gpu_csv` reads the columns positionally.
NVIDIA_SMI_QUERY_FIELDS: tuple[str, ...] = (
    "index",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
)

NVIDIA_SMI_ARGV: tuple[str, ...] = (
    "nvidia-smi",
    f"--query-gpu={','.join(NVIDIA_SMI_QUERY_FIELDS)}",
    "--format=csv,noheader,nounits",
)

#: Offered by :func:`device_choices_with_manual_gpu` when enumeration failed,
#: so an explicit GPU choice stays possible (milestone A requirement 3). It is
#: a *choice*, never a claim that the device exists: the caller must display
#: ``inventory["error"]`` beside it.
MANUAL_CUDA_DEVICE = "cuda:0"

CPU_DEVICE = "cpu"

QUERY_TIMEOUT_S = 10.0


def parse_nvidia_smi_gpu_csv(text: str) -> List[Dict[str, Any]]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` output.

    Args:
        text: raw stdout of :data:`NVIDIA_SMI_ARGV`.

    Returns:
        One dict per GPU with ``index`` (int), ``name`` (str),
        ``memory_total_mib`` / ``memory_used_mib`` / ``memory_free_mib``
        (int or None), ``utilization_pct`` (int or None) and ``device``
        (``"cuda:<index>"``).

    Raises:
        ValueError: a non-empty row does not have the expected number of
            columns, or its index column is not an integer. The message
            quotes the offending line — a silently dropped row would
            under-report the machine's GPUs, which is worse than failing.

    A memory column that ``nvidia-smi`` reports as ``[N/A]`` (some vGPU and
    MIG configurations) becomes ``None`` rather than 0: unknown is not zero.
    """
    rows: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(NVIDIA_SMI_QUERY_FIELDS):
            raise ValueError(
                f"nvidia-smi line {line_number} has {len(parts)} column(s), expected "
                f"{len(NVIDIA_SMI_QUERY_FIELDS)} ({', '.join(NVIDIA_SMI_QUERY_FIELDS)}): {line!r}. "
                "Re-run `nvidia-smi "
                f"--query-gpu={','.join(NVIDIA_SMI_QUERY_FIELDS)} --format=csv,noheader,nounits` "
                "and report the output if the driver's format has changed."
            )
        index_text, name, total_text, used_text, free_text, util_text = parts
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(
                f"nvidia-smi line {line_number} has a non-integer GPU index {index_text!r}: "
                f"{line!r}."
            ) from exc
        rows.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": _optional_int(total_text),
                "memory_used_mib": _optional_int(used_text),
                "memory_free_mib": _optional_int(free_text),
                "utilization_pct": _optional_int(util_text),
                "device": f"cuda:{index}",
            }
        )
    return rows


def _optional_int(text: str) -> Optional[int]:
    """MiB column as int, or None when the driver reports it unavailable."""
    cleaned = str(text or "").strip()
    if not cleaned or cleaned.upper().strip("[]") in {"N/A", "NA", "NOT SUPPORTED"}:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def empty_inventory(error: Optional[str]) -> Dict[str, Any]:
    """The inventory shape used when no GPU could be enumerated."""
    return {
        "source": "nvidia-smi",
        "available": False,
        "device_count": 0,
        "devices": [],
        "error": error,
    }


def query_gpu_inventory(
    *,
    timeout: float = QUERY_TIMEOUT_S,
    runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Enumerate the machine's CUDA devices via ``nvidia-smi``.

    Args:
        timeout: seconds before the query is abandoned.
        runner: injectable ``subprocess.run`` replacement for tests.

    Returns a dict with ``source``, ``available`` (bool), ``device_count``
    (int, discovered — never a constant), ``devices`` (list, see
    :func:`parse_nvidia_smi_gpu_csv`) and ``error`` (str or None).

    A machine with no NVIDIA driver is not an error condition for the
    platform — it is a CPU machine — so the failure is reported in ``error``
    instead of raised. The message names the command that failed so the user
    can run it themselves.
    """
    run = runner if runner is not None else subprocess.run
    argv = list(NVIDIA_SMI_ARGV)
    try:
        proc = run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError:
        return empty_inventory(
            "nvidia-smi was not found on PATH, so no GPU could be enumerated. "
            "This is expected on a CPU-only machine."
        )
    except subprocess.TimeoutExpired:
        return empty_inventory(
            f"`{' '.join(argv)}` did not return within {timeout}s; GPU inventory is unknown."
        )
    except OSError as exc:
        return empty_inventory(
            f"`{' '.join(argv)}` could not start ({type(exc).__name__}: {exc}); "
            "GPU inventory is unknown."
        )

    returncode = getattr(proc, "returncode", None)
    stdout = str(getattr(proc, "stdout", "") or "")
    stderr = str(getattr(proc, "stderr", "") or "")
    if returncode != 0:
        return empty_inventory(
            f"`{' '.join(argv)}` exited {returncode}; GPU inventory is unknown. "
            f"stderr: {stderr.strip()[:400] or '(empty)'}"
        )

    try:
        devices = parse_nvidia_smi_gpu_csv(stdout)
    except ValueError as exc:
        # Reported, not swallowed: the caller shows this text and the raw
        # output stays quotable in the message.
        return empty_inventory(str(exc))

    return {
        "source": "nvidia-smi",
        "available": bool(devices),
        "device_count": len(devices),
        "devices": devices,
        "error": None if devices else "nvidia-smi reported no GPU.",
    }


def device_choices(inventory: Optional[Dict[str, Any]]) -> List[str]:
    """Selectable device strings: ``cpu`` first, then one entry per real GPU.

    The list is built from the enumerated devices, so an eight-GPU server
    offers eight entries and a two-GPU machine offers two. When enumeration
    failed, only ``cpu`` is listed as a *known* device and the caller is
    expected to surface ``inventory["error"]``; use
    :func:`device_choices_with_manual_gpu` where an explicit GPU choice must
    stay possible despite unknown inventory.
    """
    choices = [CPU_DEVICE]
    if not isinstance(inventory, dict):
        return choices
    for device in inventory.get("devices", []):
        name = str(device.get("device") or "").strip()
        if name and name not in choices:
            choices.append(name)
    return choices


def device_choices_with_manual_gpu(inventory: Optional[Dict[str, Any]]) -> List[str]:
    """:func:`device_choices`, plus ``cuda:0`` when nothing could be enumerated.

    Milestone A requirement 3: a user who knows the machine has a GPU must
    still be able to select one even when ``nvidia-smi`` is unreadable. The
    extra entry is a user choice, not a claim of availability; the caller
    must display the inventory error beside it.
    """
    choices = device_choices(inventory)
    if len(choices) == 1:
        choices.append(MANUAL_CUDA_DEVICE)
    return choices


def device_label(device: str, inventory: Optional[Dict[str, Any]]) -> str:
    """Human label for one device string, using only observed values."""
    name = str(device or "").strip()
    if name == CPU_DEVICE:
        return "cpu"
    entry = find_device(name, inventory)
    if entry is None:
        return f"{name} (availability unknown)"
    free = entry.get("memory_free_mib")
    total = entry.get("memory_total_mib")
    if free is None or total is None:
        return f"{name} — {entry.get('name')}"
    return f"{name} — {entry.get('name')} ({free} MiB free of {total} MiB)"


def find_device(device: str, inventory: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The enumerated entry for ``device``, or None when it was not seen."""
    if not isinstance(inventory, dict):
        return None
    wanted = str(device or "").strip()
    for entry in inventory.get("devices", []):
        if str(entry.get("device")) == wanted:
            return entry
    return None


def free_memory_mib(device: str, inventory: Optional[Dict[str, Any]]) -> Optional[int]:
    """Observed free MiB for ``device``; None when unknown (never 0 as a guess)."""
    entry = find_device(device, inventory)
    if entry is None:
        return None
    value = entry.get("memory_free_mib")
    return int(value) if isinstance(value, int) else None


def gpu_option_label(device: str, inventory: Optional[Dict[str, Any]]) -> str:
    """One GPU as an option line: index, name, free memory, utilization.

    Every component is read from the enumerated device. A figure the driver
    did not report is omitted rather than shown as zero, so the label is
    shorter on a machine that reports less — it never fills a gap with a
    plausible number.
    """
    name = str(device or "").strip()
    entry = find_device(name, inventory)
    if entry is None:
        return f"{name} (not enumerated)"
    # Led by the device string the run actually receives, so the option the
    # user picks and the ``--device`` value in the plan read the same.
    parts = [name, str(entry.get("name") or name)]
    free = entry.get("memory_free_mib")
    if isinstance(free, int):
        parts.append(f"{free / 1024:.1f} GB free")
    utilization = entry.get("utilization_pct")
    if isinstance(utilization, int):
        parts.append(f"{utilization}% util")
    return " · ".join(parts)


def inventory_rows(inventory: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Table rows for display; empty when nothing was enumerated."""
    if not isinstance(inventory, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for entry in inventory.get("devices", []):
        rows.append(
            {
                "Device": entry.get("device"),
                "GPU": entry.get("name"),
                "Free (MiB)": _display_mib(entry.get("memory_free_mib")),
                "Used (MiB)": _display_mib(entry.get("memory_used_mib")),
                "Total (MiB)": _display_mib(entry.get("memory_total_mib")),
                "Utilization (%)": _display_mib(entry.get("utilization_pct")),
            }
        )
    return rows


def _display_mib(value: Any) -> str:
    return str(value) if isinstance(value, int) else "unknown"


#: Device strings BatteryML can actually be given. ``bin/batteryml.py run
#: --device`` is forwarded unchanged to ``DataBundle.to(device)`` and
#: ``model.to(device)``, both of which take exactly one torch device.
#: Each GPU's compute capability, read the same way as the inventory (argv,
#: ``shell=False``, no torch, no CUDA context). ``compute_cap`` needs a recent
#: driver; an older one rejects the field, and the answer is then "unknown".
COMPUTE_CAP_ARGV: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=index,name,compute_cap",
    "--format=csv,noheader,nounits",
)

#: The first CUDA toolkit able to build kernels for each GPU generation (major
#: compute capability), from NVIDIA's CUDA release notes: Ampere 8.x → 11.0,
#: Hopper 9.0 → 11.8, Blackwell 10.x/12.x → 12.8. Consulted only when PyTorch
#: cannot list the architectures it was built for (PyTorch < 1.7 has no
#: ``get_arch_list``); when it can, its own list decides.
MIN_CUDA_FOR_ARCH_MAJOR: Dict[int, tuple[int, int]] = {8: (11, 0), 9: (11, 8), 10: (12, 8), 12: (12, 8)}


def parse_compute_capabilities(text: str) -> List[Dict[str, Any]]:
    """``[{"index", "name", "capability": (major, minor)}]`` from :data:`COMPUTE_CAP_ARGV`.

    Raises:
        ValueError: a row is malformed; the message quotes it.
    """
    rows: List[Dict[str, Any]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.rsplit(",", 1)]
        head = [part.strip() for part in parts[0].split(",", 1)]
        if len(parts) != 2 or len(head) != 2:
            raise ValueError(f"nvidia-smi compute_cap line is malformed: {line!r}.")
        match = re.fullmatch(r"(\d+)\.(\d+)", parts[1])
        if not match or not head[0].isdigit():
            raise ValueError(f"nvidia-smi compute_cap line is malformed: {line!r}.")
        rows.append({"index": int(head[0]), "name": head[1],
                     "capability": (int(match.group(1)), int(match.group(2)))})
    return rows


def query_compute_capabilities(
    *,
    timeout: float = QUERY_TIMEOUT_S,
    runner: Optional[Callable[..., Any]] = None,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """``(devices, None)``, or ``(None, why)`` when the machine cannot say."""
    run = runner if runner is not None else subprocess.run
    argv = list(COMPUTE_CAP_ARGV)
    try:
        proc = run(argv, capture_output=True, text=True, timeout=timeout, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"`{' '.join(argv)}` could not run ({type(exc).__name__})."
    if getattr(proc, "returncode", None) != 0:
        stderr = str(getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        return None, f"`{' '.join(argv)}` exited {getattr(proc, 'returncode', None)}: {stderr[:200] or '(empty)'}"
    try:
        return parse_compute_capabilities(str(getattr(proc, "stdout", "") or "")), None
    except ValueError as exc:
        return None, str(exc)


def gpu_build_incompatibilities(
    devices: Sequence[Dict[str, Any]],
    *,
    arch_list: Optional[Sequence[str]],
    torch_cuda: Optional[str],
) -> Optional[List[str]]:
    """The GPUs this PyTorch build cannot run on, or None when it cannot be decided.

    Pure. ``arch_list`` is PyTorch's ``get_arch_list()`` (e.g. ``["sm_37",
    "sm_75", "compute_75"]``) or None when that PyTorch cannot report it;
    ``torch_cuda`` is ``torch.version.cuda``. With an architecture list the
    rule is PyTorch's own (``torch.cuda._check_cubins``): a device works when
    a compiled ``sm_XY`` shares its major version. Without one, a device
    whose generation needs a newer CUDA than the build's is incompatible.
    Returns descriptions such as ``"NVIDIA GeForce RTX 3090 (sm_86)"``.
    """
    supported = [int(arch.split("_", 1)[1]) for arch in (arch_list or [])
                 if str(arch).startswith("sm_") and str(arch).split("_", 1)[1].isdigit()]
    cuda_version: Optional[tuple[int, int]] = None
    match = re.match(r"^(\d+)\.(\d+)", str(torch_cuda or ""))
    if match:
        cuda_version = (int(match.group(1)), int(match.group(2)))
    if not supported and cuda_version is None:
        return None
    bad: List[str] = []
    for device in devices:
        major, minor = device["capability"]
        if supported:
            ok = any(sm // 10 == major for sm in supported)
        else:
            needed = MIN_CUDA_FOR_ARCH_MAJOR.get(major)
            ok = needed is None or cuda_version >= needed
        if not ok:
            bad.append(f"{device['name']} (sm_{major}{minor})")
    return bad


_DEVICE_RE = re.compile(r"^(cpu|mps|cuda|cuda:\d+)$")


def validate_device(device: str) -> str:
    """Return ``device`` unchanged, or raise naming what is wrong with it.

    Accepts ``cpu``, ``mps``, ``cuda`` and ``cuda:<index>`` — the forms a
    single ``torch.device`` understands.

    A multi-device string such as ``"cuda:0,1"`` is rejected explicitly,
    with the reason: it is not a torch device, and accepting it would
    advertise a multi-GPU capability the platform does not have. See
    ``docs/GPU_MULTI_GPU_AUDIT.md`` — neither DataParallel nor DDP would
    help here anyway, because ``DataBundle.to(device)`` places the whole
    dataset on one device.
    """
    name = str(device or "").strip()
    if not name:
        raise ValueError(
            "device is empty. Use 'cpu', 'cuda', 'cuda:<index>' or 'mps'."
        )
    if "," in name:
        raise ValueError(
            f"device {name!r} names more than one device. BatteryML passes "
            "--device straight to torch's .to(device), which takes exactly one "
            "device, so a comma-separated value is not a valid GPU selection "
            "and this platform exposes no multi-GPU run. Select a single GPU "
            "('cuda:0', 'cuda:3', ...) or 'cpu'; see docs/GPU_MULTI_GPU_AUDIT.md."
        )
    if not _DEVICE_RE.match(name):
        raise ValueError(
            f"device {name!r} is not a device BatteryML accepts. Use 'cpu', "
            "'cuda', 'cuda:<index>' or 'mps'."
        )
    return name


# ---------------------------------------------------------------------------
# Structured device selection
# ---------------------------------------------------------------------------
#
# A selection is a structure, not a string, because "which devices did the
# user pick" and "what does BatteryML get on the command line" are different
# questions with different answers. Keeping them apart is what makes it
# impossible to answer the second one with "cuda:0,1" — a string torch does
# not accept and that would advertise a capability the platform does not have
# (see docs/GPU_MULTI_GPU_AUDIT.md).

MODE_CPU = "cpu"
MODE_SINGLE_GPU = "single_gpu"
MODE_MULTI_GPU = "multi_gpu"


def build_device_selection(gpu_ids: Optional[Sequence[int]]) -> Dict[str, Any]:
    """The structured selection for a set of chosen GPU indices.

    Args:
        gpu_ids: the GPU indices the user selected. Empty or None means CPU.

    Returns ``{"mode", "gpu_ids"}`` with ``mode`` one of ``cpu``,
    ``single_gpu`` or ``multi_gpu``. Indices are de-duplicated and sorted, so
    the same choice always produces the same selection.

    Raises:
        ValueError: an index is not a non-negative integer. A silently
            dropped index would run on a different GPU than the one asked
            for.
    """
    if not gpu_ids:
        return {"mode": MODE_CPU, "gpu_ids": []}
    cleaned: List[int] = []
    for value in gpu_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"GPU index {value!r} is not an integer; a selection must name "
                "device indices such as 0 or 3."
            )
        if value < 0:
            raise ValueError(f"GPU index {value!r} is negative.")
        if value not in cleaned:
            cleaned.append(value)
    cleaned.sort()
    mode = MODE_SINGLE_GPU if len(cleaned) == 1 else MODE_MULTI_GPU
    return {"mode": mode, "gpu_ids": cleaned}


def selection_from_devices(device_names: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Build a selection from ``cuda:<index>`` strings (what a widget returns)."""
    indices: List[int] = []
    for name in device_names or []:
        text = str(name).strip()
        if not text or text == CPU_DEVICE:
            continue
        if not text.startswith("cuda:"):
            raise ValueError(
                f"{text!r} is not a CUDA device name; expected 'cuda:<index>'."
            )
        suffix = text.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError(f"{text!r} does not carry an integer device index.")
        indices.append(int(suffix))
    return build_device_selection(indices)


def device_string_for_selection(selection: Optional[Dict[str, Any]]) -> Optional[str]:
    """The ``--device`` value for a selection, or None when there is none.

    ``cpu`` and a single GPU produce a device string. A multi-GPU selection
    produces **None**: there is no single torch device for it, and inventing
    one (``"cuda:0,1"``) would be a string that fails inside a started run.
    The caller must gate execution on the capability instead.
    """
    if not isinstance(selection, dict):
        return CPU_DEVICE
    mode = str(selection.get("mode") or MODE_CPU)
    if mode == MODE_CPU:
        return CPU_DEVICE
    if mode == MODE_SINGLE_GPU:
        ids = list(selection.get("gpu_ids") or [])
        if len(ids) != 1:
            raise ValueError(
                f"selection claims {MODE_SINGLE_GPU} but names {len(ids)} device(s)."
            )
        return f"cuda:{int(ids[0])}"
    return None


def describe_selection(selection: Optional[Dict[str, Any]]) -> str:
    """A short human description of a selection, for a caption."""
    if not isinstance(selection, dict):
        return "CPU"
    mode = str(selection.get("mode") or MODE_CPU)
    ids = [str(value) for value in selection.get("gpu_ids") or []]
    if mode == MODE_CPU:
        return "CPU"
    if mode == MODE_SINGLE_GPU:
        return f"GPU {ids[0]}"
    return f"GPUs {', '.join(ids)}"
