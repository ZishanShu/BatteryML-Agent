"""Deterministic execution controller for validated agent execution plans.

Consumes ONLY the output of
:func:`batteryml_web.agent.execution.build_execution_plan` (or
:func:`~batteryml_web.agent.execution.plan_for_request`) and launches the
existing BatteryML CLI wrapper. The LLM itself never executes shell
commands: this module is the single place where a plan becomes a
subprocess, and it refuses every plan that is not fully validated.

Safety rules (enforced before any subprocess starts):

* ``can_execute`` and ``config_valid`` must both be exactly True;
* ``model_status`` and ``dataset_status`` must both be ``READY``;
* ``command`` must be a non-empty argv list — a shell string is rejected;
* the config argument inside the argv must equal ``plan["config_path"]``;
* ``config_path`` must exist on disk;
* the ``--workspace`` value must resolve inside the repository's ``runs/``
  area;
* external models (PBT, BatteryMFormer) are never executable here.

Execution uses ``subprocess.run`` with an argv list and ``shell=False``,
captures stdout/stderr, preserves non-zero return codes honestly, and never
retries. ``dry_run=True`` performs all validation and returns the exact
argv without starting a subprocess. Artifact discovery reports paths only
for files that really exist; metrics are never parsed or invented here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .execution import WRAPPER

#: Models that must never be executed through the BatteryML CLI wrapper,
#: even if a forged plan claims they are READY.
NON_EXECUTABLE_MODELS = frozenset({"PBT", "BatteryMFormer"})

_EMPTY_ARTIFACTS = (
    "metrics_files",
    "prediction_files",
    "log_files",
    "report_files",
    "figure_files",
    "checkpoint_files",
    "config_files",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_from_command(command: List[str]) -> Optional[str]:
    try:
        index = list(command).index("--workspace")
    except ValueError:
        return None
    following = list(command)[index + 1 :]
    return str(following[0]) if following else None


def _config_arg_from_command(command: List[str]) -> Optional[str]:
    parts = list(command)
    try:
        index = parts.index("run")
    except ValueError:
        return None
    following = parts[index + 1 :]
    return str(following[0]) if following else None


def validate_plan_for_execution(
    plan: Dict[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Check a plan against every pre-execution rule without running anything.

    Returns ``{"ok", "error", "argv", "workspace", "repo_root"}``. ``argv``
    is the exact list that would execute; ``workspace`` the ``--workspace``
    value. On failure ``ok`` is False and ``error`` names the violated rule.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    failure: Dict[str, Any] = {
        "ok": False,
        "error": "",
        "argv": [],
        "workspace": None,
        "repo_root": str(root),
    }

    def reject(reason: str) -> Dict[str, Any]:
        failure["error"] = reason
        return failure

    if not isinstance(plan, dict):
        return reject("plan must be a dict produced by build_execution_plan().")
    if plan.get("can_execute") is not True:
        return reject(
            f"plan is not executable (can_execute={plan.get('can_execute')!r}, "
            f"model_status={plan.get('model_status')!r}, "
            f"dataset_status={plan.get('dataset_status')!r}): "
            f"{plan.get('reason') or 'see plan reason'}"
        )
    if plan.get("config_valid") is not True:
        return reject("plan claims can_execute but config_valid is not True; refused.")
    if plan.get("model_status") != "READY":
        return reject(
            f"model_status is {plan.get('model_status')!r}, not READY; refused."
        )
    if plan.get("dataset_status") != "READY":
        return reject(
            f"dataset_status is {plan.get('dataset_status')!r}, not READY; refused."
        )
    if str(plan.get("model")) in NON_EXECUTABLE_MODELS:
        return reject(
            f"model {plan.get('model')!r} is external and not executable "
            "through the BatteryML CLI wrapper."
        )

    command = plan.get("command")
    if isinstance(command, str):
        return reject(
            "plan command is a shell string; only an argv list is accepted "
            "(shell execution is forbidden)."
        )
    if not isinstance(command, (list, tuple)) or not command:
        return reject("plan command must be a non-empty argv list.")
    argv = [str(part) for part in command]
    if any(not isinstance(part, str) for part in command):
        return reject("plan command argv must contain only strings.")

    config_path = plan.get("config_path")
    if not config_path or not isinstance(config_path, str):
        return reject("plan carries no config_path; refused.")
    if len(argv) < 4 or argv[1] != WRAPPER or argv[2] != "run":
        return reject(
            f"plan command does not follow the CLI contract "
            f"(<python> {WRAPPER} run <config> ...); refused."
        )
    if _config_arg_from_command(argv) != config_path:
        return reject(
            "command config argument differs from plan config_path; "
            "the command may have been rebuilt or forged. Refused."
        )
    if "--train" not in argv or "--eval" not in argv:
        return reject("plan command lacks mandatory --train/--eval flags; refused.")
    if not Path(config_path).is_file():
        return reject(f"config_path does not exist on disk: {config_path}")

    workspace = _workspace_from_command(argv)
    if not workspace:
        return reject("plan command carries no --workspace value; refused.")
    resolved = (
        Path(workspace)
        if Path(workspace).is_absolute()
        else (root / workspace)
    )
    try:
        resolved.resolve().relative_to(root.resolve() / "runs")
    except ValueError:
        return reject(
            f"workspace {workspace!r} is outside the repository runs/ area; refused."
        )

    return {
        "ok": True,
        "error": None,
        "argv": argv,
        "workspace": workspace,
        "repo_root": str(root),
    }


def _empty_artifacts() -> Dict[str, List[str]]:
    return {key: [] for key in _EMPTY_ARTIFACTS}


def discover_artifacts(
    workspace: str,
    *,
    repo_root: str | Path | None = None,
) -> Dict[str, List[str]]:
    """List real result files under ``workspace``; never invents paths."""
    from ..core.results_reader import find_result_files

    root = Path(repo_root) if repo_root is not None else _repo_root()
    resolved = (
        Path(workspace)
        if Path(workspace).is_absolute()
        else (root / workspace)
    )
    if not resolved.is_dir():
        return _empty_artifacts()
    found = find_result_files(resolved, resolved)
    artifacts = _empty_artifacts()
    for key in (
        "metrics_files",
        "prediction_files",
        "log_files",
        "report_files",
        "figure_files",
        "checkpoint_files",
    ):
        artifacts[key] = [
            str(p) for p in found.get(key, []) if Path(p).is_file()
        ]
    configs = sorted(resolved.rglob("*.yaml")) + sorted(resolved.rglob("*.yml"))
    artifacts["config_files"] = [str(p) for p in configs if p.is_file()]
    return artifacts


def execute_plan(
    plan: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    repo_root: str | Path | None = None,
    cwd: str | Path | None = None,
    runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Execute a validated plan, or validate-only when ``dry_run`` is True.

    Args:
        plan: dict from :func:`build_execution_plan`; re-validated here.
        timeout: seconds forwarded to ``subprocess.run`` (None = no limit).
        env: environment mapping for the child process (None = inherit).
        dry_run: validate and return the exact argv without subprocess.
        repo_root: repository root for workspace containment and ``cwd``.
        cwd: working directory for the child (defaults to the repo root).
        runner: injectable ``subprocess.run`` replacement for tests.

    Returns a dict with ``started``, ``completed``, ``success``,
    ``returncode``, ``command``, ``workspace``, ``stdout``, ``stderr``,
    ``error``, ``artifacts`` and ``dry_run``.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    checked = validate_plan_for_execution(plan, repo_root=root)
    base: Dict[str, Any] = {
        "started": False,
        "completed": False,
        "success": False,
        "returncode": None,
        "command": checked.get("argv", []),
        "workspace": checked.get("workspace"),
        "stdout": "",
        "stderr": "",
        "error": checked.get("error"),
        "artifacts": _empty_artifacts(),
        "dry_run": bool(dry_run),
    }
    if not checked["ok"]:
        return base
    if dry_run:
        base["completed"] = True
        base["success"] = True
        base["error"] = None
        return base

    argv: List[str] = checked["argv"]
    run = runner if runner is not None else subprocess.run
    try:
        proc = run(
            argv,
            cwd=str(cwd) if cwd is not None else str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        base["started"] = True
        base["stdout"] = str(getattr(exc, "stdout", "") or "")
        base["stderr"] = str(getattr(exc, "stderr", "") or "")
        base["error"] = f"execution timed out after {timeout}s; nothing retried."
        return base
    except OSError as exc:
        base["error"] = f"execution could not start ({type(exc).__name__}: {exc})."
        return base

    base["started"] = True
    base["completed"] = True
    base["returncode"] = getattr(proc, "returncode", None)
    base["stdout"] = str(getattr(proc, "stdout", "") or "")
    base["stderr"] = str(getattr(proc, "stderr", "") or "")
    base["success"] = base["returncode"] == 0
    if base["success"]:
        base["artifacts"] = discover_artifacts(
            str(checked["workspace"]), repo_root=root
        )
    else:
        base["error"] = (
            f"wrapper exited with returncode {base['returncode']}; "
            "no retry performed, no metrics fabricated."
        )
    return base
