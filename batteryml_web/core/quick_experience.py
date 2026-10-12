"""Home → Quick experience: what may run, decided before anything runs.

The page (``batteryml_web/app.py``, ``_render_home_quickstart_panel``) keeps
its design; this module owns the decisions behind it so they can be tested
without Streamlit:

* :func:`dataset_choices` — the datasets the panel offers: every dataset the
  readiness layer knows (``readiness.list_dataset_readiness``) together with
  the tasks the config registry has protocol configs for. A dataset without
  one stays listed but is marked not runnable here (:func:`quick_limitation`)
  instead of being hidden or hardcoded away.
* :func:`download_action` / :func:`process_action` — whether each data action
  is available, already done, or unavailable (and why), from the readiness
  report alone. Nothing is downloaded or processed by deciding.
* :func:`config_choices` / :func:`reconcile_selection` — the protocol configs
  offered for the selected dataset, task and model. A task, model or config
  that the new selection does not offer is never kept selected: it is
  cleared, and the caller says so (switching RUL → SOH, or MATR → HUST, can no
  longer leave a config of the previous selection behind).
* :func:`device_problem` — a requested device that this machine cannot
  provide is reported, never silently replaced.
* :func:`build_quick_plan` — the execution gate. The page calls it to draw the
  Run button and again immediately before launching, so the command that runs
  is exactly the one that was validated: config readable, every class
  registered (``config_validator``), task established structurally by the
  label annotator (``config_task``) **and equal to the task on screen**,
  dataset offered by the page, data present, device available, and — for an
  uploaded config — no file/directory option outside ``data/processed``.
* :func:`run_metrics` — the metrics a finished run recorded in its own
  ``run.json`` (written from BatteryML's saved scores by the CPU-safe loader),
  so Home, Run History and Review Results show the same numbers.
* :func:`download_failure_host` — the host a failed download actually named.

No Streamlit import, no subprocess, no BatteryML run.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import yaml

from .config_task import SHORT_TASK, SUPPORTED_TASK_LABELS, canonical_task, resolve_config_task

CPU = "cpu"
CUDA = "cuda"
DEVICE_OPTIONS = (CPU, CUDA)

#: Readiness verdicts (``readiness.DatasetStatus`` values), repeated as plain
#: strings so this module stays importable without the readiness layer.
STATUS_READY = "READY"
STATUS_INCOMPLETE = "INCOMPLETE_PROCESSED"


# --- datasets ------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetChoice:
    """One dataset the platform knows, and what Quick experience can do with it."""

    name: str
    #: Readiness verdict: READY, RAW_ONLY, INCOMPLETE_PROCESSED or MISSING.
    status: str
    #: Canonical tasks the config registry has a protocol config for, in
    #: ``SUPPORTED_TASK_LABELS`` order. Empty: nothing here can run on it.
    tasks: Tuple[str, ...]

    @property
    def runnable(self) -> bool:
        """Whether a registered protocol config exists for it (data aside)."""
        return bool(self.tasks)


def dataset_choices(
    statuses: Mapping[str, str],
    config_pairs: Iterable[Tuple[Optional[str], Optional[str]]],
    *,
    default: str,
) -> List[DatasetChoice]:
    """Every known dataset, the default first, then runnable ones, then the rest.

    ``statuses`` maps each dataset the readiness layer reports to its verdict;
    ``config_pairs`` are the registry's ``(dataset, task)`` pairs. A dataset
    the registry names but ``statuses`` lacks is an error, not a guess: the
    caller must assess it.
    """
    tasks: Dict[str, set] = {}
    for dataset, task in config_pairs:
        label = canonical_task(task)
        if dataset and label:
            tasks.setdefault(str(dataset), set()).add(label)
    unassessed = sorted(set(tasks) - set(statuses))
    if unassessed:
        raise ValueError(f"dataset_choices: no readiness verdict for {', '.join(unassessed)}; "
                         "assess every registry dataset before offering it.")
    choices = [
        DatasetChoice(name, str(status), tuple(t for t in SUPPORTED_TASK_LABELS if t in tasks.get(name, ())))
        for name, status in statuses.items()
    ]
    return sorted(choices, key=lambda c: (c.name != default, not c.runnable, c.name.lower()))


def task_options(choice: DatasetChoice) -> List[str]:
    """The tasks offered for ``choice``: its configured ones, or all when it has none."""
    return list(choice.tasks) or list(SUPPORTED_TASK_LABELS)


def quick_limitation(choice: DatasetChoice) -> Optional[str]:
    """Why Quick experience cannot run ``choice``, or None when it can."""
    if choice.runnable:
        return None
    return (f"{choice.name} is known to the platform, but no protocol configuration is "
            "registered for it, so Quick experience cannot run a model on it. Its data can "
            "still be checked and prepared in Prepare Data; adding a configuration for it "
            "(the extension route) makes it runnable.")


# --- data actions ----------------------------------------------------------------

ACTION_AVAILABLE = "available"
ACTION_COMPLETED = "completed"
ACTION_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DataAction:
    """What one data action can do now: ``state`` plus one line for the page."""

    state: str
    note: str = ""


def download_action(*, dataset: str, downloadable: bool, raw_complete: bool,
                    raw_count: int = 0, raw_expected: int = 0) -> DataAction:
    """Download: done when the raw files are all present, else offered if BatteryML can."""
    if raw_complete:
        return DataAction(ACTION_COMPLETED, "Raw files present — no download needed.")
    if not downloadable:
        return DataAction(ACTION_UNAVAILABLE,
                          f"BatteryML has no automatic download for {dataset}. Place its raw "
                          f"files under data/raw/{dataset}/, or see Prepare Data.")
    if raw_count and raw_expected:
        return DataAction(ACTION_AVAILABLE, f"{raw_count} of {raw_expected} files present; "
                                            "downloading fetches the rest.")
    return DataAction(ACTION_AVAILABLE)


def process_action(*, status: str, raw_ready: bool, preprocessing_supported: bool) -> DataAction:
    """Preprocess: done when READY, offered only for raw data BatteryML can preprocess."""
    if status == STATUS_READY:
        return DataAction(ACTION_COMPLETED, "Processed data ready — no processing needed.")
    if not raw_ready:
        return DataAction(ACTION_UNAVAILABLE, "Available once the raw files are present.")
    if not preprocessing_supported:
        return DataAction(ACTION_UNAVAILABLE,
                          "This raw layout is detected, but BatteryML does not provide an "
                          "automatic preprocessing route for it.")
    return DataAction(ACTION_AVAILABLE)


# --- configs -------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigOption:
    """One selectable protocol config and what it trains for."""

    path: str
    task: Optional[str]
    model_key: Optional[str]


def config_choices(options: Sequence[ConfigOption], *, task: str, model_key: str) -> List[str]:
    """Paths of the configs for exactly this task and model, order kept, no duplicates."""
    return list(dict.fromkeys(
        option.path for option in options if option.task == task and option.model_key == model_key
    ))


def reconcile_selection(current: Optional[str], choices: Sequence[str],
                        default: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """``(selected, cleared)`` — one rule for a dataset-, task-, model- or config choice.

    ``current`` survives only if it is still one of ``choices``. Otherwise it
    is cleared (returned as ``cleared`` so the page can say so) and the
    default — or the first choice — is selected instead. With no choice at
    all, nothing is selected.
    """
    if current and current in choices:
        return current, None
    cleared = current or None
    if not choices:
        return None, cleared
    if default and default in choices:
        return default, cleared
    return choices[0], cleared


#: The name the config selector has always used for :func:`reconcile_selection`.
reconcile_config = reconcile_selection


# --- device --------------------------------------------------------------------

def cuda_available(probe: Optional[Callable[[], bool]] = None) -> Tuple[bool, str]:
    """``(available, detail)``. Never raises: a probe failure means "not available".

    Available means PyTorch sees a CUDA device **and** this PyTorch build can
    run kernels on it. A GPU the build was not compiled for (measured on the
    server: PyTorch 1.6 / CUDA 10.2 on RTX 3090, sm_86) is reported as not
    available with the reason, so the page blocks cuda instead of offering a
    run that would fail — and never swaps in the CPU.
    """
    try:
        if probe is not None:
            available = bool(probe())
            return (True, "") if available else (False, NO_CUDA_DETAIL)
        return _torch_cuda_status()
    except Exception as exc:  # noqa: BLE001 - a CPU machine must still render
        return False, f"CUDA availability could not be determined ({type(exc).__name__}: {exc})."


NO_CUDA_DETAIL = "This machine has no CUDA device that PyTorch can use."


@functools.lru_cache(maxsize=1)
def _torch_cuda_status() -> Tuple[bool, str]:
    """The machine's answer, computed once per process (driver and build do not change).

    Reads only compile-time facts from torch (its architecture list and CUDA
    version) and the GPUs' compute capability from ``nvidia-smi``; it does
    not initialise CUDA or allocate memory on a shared GPU.
    """
    import torch

    from .device_inventory import gpu_build_incompatibilities, query_compute_capabilities

    if not torch.cuda.is_available():
        return False, NO_CUDA_DETAIL
    if hasattr(torch.cuda, "get_arch_list"):
        arch_list: Optional[List[str]] = list(torch.cuda.get_arch_list())
    elif hasattr(torch._C, "_cuda_getArchFlags"):
        arch_list = str(torch._C._cuda_getArchFlags() or "").split()
    else:
        arch_list = None
    devices, _why = query_compute_capabilities()
    if not devices:
        return True, ""          # cannot be decided here; PyTorch says CUDA is usable
    bad = gpu_build_incompatibilities(devices, arch_list=arch_list, torch_cuda=torch.version.cuda)
    if not bad:
        return True, ""
    built = ", ".join(a for a in (arch_list or []) if a.startswith("sm_")) or f"CUDA {torch.version.cuda}"
    return False, (f"PyTorch {torch.__version__} (built for {built}) cannot run on "
                   f"{', '.join(sorted(set(bad)))}, so a GPU run would fail here. Install a "
                   "PyTorch build that supports this GPU.")


def device_problem(requested: str, *, cuda_ok: bool, cuda_detail: str = "") -> Optional[str]:
    """Why ``requested`` cannot be used here, or None. Never substitutes a device."""
    from .device_inventory import validate_device

    try:
        name = validate_device(requested)
    except ValueError as exc:
        return str(exc)
    if name.startswith(CUDA) and not cuda_ok:
        return (f"GPU (cuda) was requested, but it is not available on this machine. "
                f"{cuda_detail} Choose cpu explicitly to run on the CPU.").replace("  ", " ")
    return None


# --- the plan / execution gate ---------------------------------------------------

@dataclass
class QuickPlan:
    """Everything the Run button may execute, or why it may not."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    task: Optional[str] = None
    config_task: Optional[str] = None
    dataset: Optional[str] = None
    model_key: Optional[str] = None
    model_display: Optional[str] = None
    config_path: Optional[str] = None
    device: Optional[str] = None


def build_quick_plan(
    *,
    task: str,
    config_path: Optional[Path],
    device: str,
    repo_root: Path,
    allowed_datasets: Sequence[str],
    cuda_ok: bool,
    cuda_detail: str = "",
    uploaded: bool = False,
    require_data: bool = True,
) -> QuickPlan:
    """Decide whether this selection may run. Pure apart from reading the config file."""
    from .config_semantics import infer_config_semantics
    from .config_upload import anchor_data_paths, unsafe_options
    from .config_validator import validate_config

    plan = QuickPlan(valid=False, task=task, device=device,
                     config_path=str(config_path) if config_path else None)
    if config_path is None:
        plan.errors.append(f"No protocol configuration is selected for {task}. Choose one.")
        return plan
    path = Path(config_path)
    if not path.is_file():
        plan.errors.append(f"Configuration file is missing: {path}")
        return plan
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        plan.errors.append(f"Configuration {path.name} cannot be read: {exc}")
        return plan
    if not isinstance(loaded, dict):
        plan.errors.append(f"Configuration {path.name} is not a mapping of sections.")
        return plan

    resolution = resolve_config_task(loaded)
    plan.config_task = resolution.task
    if resolution.error:
        plan.errors.append(resolution.error)
    elif not resolution.structural:
        plan.errors.append(
            f"{path.name} does not establish its task through a registered label "
            "annotator, so it cannot be run from here.")
    elif resolution.task != task:
        plan.errors.append(
            f"The selected task is {task!r}, but {path.name} trains for "
            f"{resolution.task!r}. Choose a {task} configuration.")

    report = validate_config(
        {**anchor_data_paths(loaded, repo_root), "config_path": str(path), "task": SHORT_TASK.get(task, task)},
        require_data=require_data,
    )
    plan.errors.extend(report["errors"])
    plan.warnings.extend(report["warnings"])

    semantics = infer_config_semantics(loaded, source_path=path)
    plan.dataset = semantics.dataset
    plan.model_key = semantics.model_key
    plan.model_display = semantics.model_display
    if semantics.dataset not in allowed_datasets:
        plan.errors.append(f"{path.name} reads dataset {semantics.dataset or 'unknown'!r}; "
                           f"this page runs {', '.join(allowed_datasets)}.")
    if uploaded:
        plan.errors.extend(unsafe_options(loaded, repo_root=repo_root))

    problem = device_problem(device, cuda_ok=cuda_ok, cuda_detail=cuda_detail)
    if problem:
        plan.errors.append(problem)

    plan.errors = list(dict.fromkeys(plan.errors))
    plan.valid = not plan.errors
    return plan


def plan_command(plan: QuickPlan, *, workspace: str, seed: int = 0) -> List[str]:
    """The argv for a valid plan — built by the platform's one command builder."""
    if not plan.valid:
        raise ValueError("refusing to build a command for an invalid plan: " + "; ".join(plan.errors))
    from ..agent.execution import build_run_command

    return build_run_command(config_path=str(plan.config_path), workspace=workspace,
                             task=plan.task, device=str(plan.device), seed=seed)


# --- results -------------------------------------------------------------------

def run_metrics(run_dir: Path) -> Tuple[Dict[str, float], Optional[str]]:
    """``(metrics, problem)`` recorded in the run's own ``run.json``.

    ``_finish_page_run`` writes them from BatteryML's saved scores through
    ``build_result_context`` (CPU-safe artifact loader), exactly as Run History
    and Review Results read them. Nothing is unpickled here, and no other run
    is ever consulted.
    """
    from . import run_manifest as rm

    manifest = rm.read_manifest(run_dir)
    if manifest is None:
        return {}, f"{Path(run_dir) / rm.MANIFEST_NAME} is missing or unreadable."
    metrics = {str(k): float(v) for k, v in (manifest.get("metrics") or {}).items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    problem = manifest.get("metrics_error")
    return metrics, (str(problem) if problem else None)


# --- dataset manifest refresh ------------------------------------------------------

MANIFEST_SYNC_TOOL = Path("tools") / "sync_data_and_config_manifests.py"


def sync_manifests(
    repo_root: Path,
    *,
    python: str,
    run: Callable[..., Dict[str, object]],
) -> Tuple[bool, str]:
    """Run the repository's manifest sync tool; ``(ok, detail)``.

    ``run(command, cwd=...)`` returns ``{"returncode", "stdout", "stderr"}``
    (the page passes its ``_run_cmd``). ``detail`` names the command and its
    error whenever the refresh did not succeed, so the page never implies a
    refreshed dataset list it does not have.
    """
    command = [python, str(Path(repo_root) / MANIFEST_SYNC_TOOL)]
    try:
        result = run(command, cwd=Path(repo_root))
    except OSError as exc:
        return False, f"`{' '.join(command)}` could not start ({type(exc).__name__}: {exc})."
    code = result.get("returncode")
    if code == 0:
        return True, ""
    output = str(result.get("stderr") or result.get("stdout") or "").strip()
    tail = " ".join(output.splitlines()[-3:])
    return False, f"`{' '.join(command)}` exited {code}. {tail or 'It printed no error.'}"


# --- download failures -----------------------------------------------------------

_URL_IN_TEXT = re.compile(r"https?://[^\s'\"<>]+")
_HOST_KWARG = re.compile(r"host='([^']+)'")


def download_failure_host(output: str) -> Optional[str]:
    """The host a failed download named in its own error output, or None.

    Reads the last URL the output names (the safe downloader's error line
    names the exact URL it tried), falling back to requests' ``host='…'``.
    Never assumes a host the output does not name.
    """
    text = str(output or "")
    urls = _URL_IN_TEXT.findall(text)
    for url in reversed(urls):
        host = urlparse(url).hostname
        if host:
            return host
    hosts = _HOST_KWARG.findall(text)
    return hosts[-1] if hosts else None


__all__ = [
    "ACTION_AVAILABLE",
    "ACTION_COMPLETED",
    "ACTION_UNAVAILABLE",
    "CPU",
    "CUDA",
    "DEVICE_OPTIONS",
    "ConfigOption",
    "DataAction",
    "DatasetChoice",
    "QuickPlan",
    "build_quick_plan",
    "config_choices",
    "cuda_available",
    "dataset_choices",
    "device_problem",
    "download_action",
    "download_failure_host",
    "plan_command",
    "process_action",
    "quick_limitation",
    "reconcile_config",
    "reconcile_selection",
    "run_metrics",
    "sync_manifests",
    "task_options",
]
