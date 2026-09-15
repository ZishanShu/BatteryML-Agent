"""DEV-01 — the execution runtime contract: devices, smoke mode, failure honesty.

Establishes, without a GPU and without running the pipeline, that the pieces
the web layer relies on at run time still behave as specified:

* device enumeration shells out with an **argv list** and ``shell=False``,
  parses ``nvidia-smi`` positionally, and derives the GPU count from the
  machine rather than from a constant;
* a smoke run is exactly ``--epochs 1`` on the model's own formal config —
  no config is written or mutated anywhere in the planner or the policy;
* BatLiNet's smoke run recommends CPU, and the required smoke sentence is
  the one the UI shows;
* a CUDA out-of-memory failure is recognised, is never retried, and no
  metric is ever produced from a failed run.

The machine's actual GPU inventory is recorded as *evidence* when
``nvidia-smi`` is present. Its absence is never a failure: a CPU-only
machine is a supported machine.

This check never trains, never evaluates, never allocates GPU memory and
never writes into ``data/``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
from typing import Any

from .common import FAIL, PASS, SKIP, CheckResult, repo_root, run_check, write_json

CHECK_ID = "DEV-01"
CHECK_NAME = "runtime_contract"

BATLINET_CONFIG = Path("configs") / "external" / "batlinet" / "matr_1.yaml"

REQUIRED_SMOKE_NOTICE = (
    "1-epoch smoke test — workflow validation only; not benchmark performance."
)

#: A synthetic nvidia-smi sample. Never a claim about this machine.
_SAMPLE_CSV = (
    "0, NVIDIA GeForce RTX 3090, 24576, 22500, 2076\n"
    "1, NVIDIA GeForce RTX 3090, 24576, 14000, 10576\n"
)

_SAMPLE_OOM_STDERR = (
    "Traceback (most recent call last):\n"
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
)

def _device_inventory_problems(module: Any) -> list[str]:
    problems: list[str] = []
    source = inspect.getsource(module)
    if "shell=False" not in source:
        problems.append(
            "batteryml_web/core/device_inventory.py no longer passes shell=False to "
            "subprocess; restore it (a shell string would make the query injectable)."
        )
    if "shell=True" in source:
        problems.append(
            "batteryml_web/core/device_inventory.py uses shell=True; it must pass an "
            "argv list with shell=False."
        )
    if not isinstance(module.NVIDIA_SMI_ARGV, tuple) or module.NVIDIA_SMI_ARGV[0] != "nvidia-smi":
        problems.append(
            "device_inventory.NVIDIA_SMI_ARGV is not an argv tuple beginning with "
            "'nvidia-smi'."
        )

    rows = module.parse_nvidia_smi_gpu_csv(_SAMPLE_CSV)
    if [row["device"] for row in rows] != ["cuda:0", "cuda:1"]:
        problems.append(
            "parse_nvidia_smi_gpu_csv no longer maps the index column onto "
            f"'cuda:<index>' (got {[row.get('device') for row in rows]})."
        )
    if rows and rows[0].get("memory_free_mib") != 2076:
        problems.append(
            "parse_nvidia_smi_gpu_csv reads the memory columns out of order; the "
            f"free column must be the fifth field (got {rows[0].get('memory_free_mib')})."
        )
    try:
        module.parse_nvidia_smi_gpu_csv("0, GPU, 24576\n")
    except ValueError:
        pass
    else:
        problems.append(
            "parse_nvidia_smi_gpu_csv silently accepted a 3-column row; a dropped row "
            "under-reports the machine's GPUs and must raise instead."
        )

    # The count must follow the output, for any count.
    for count in (0, 1, 3, 8):
        sample = "".join(f"{i}, RTX 3090, 24576, 100, 24476\n" for i in range(count))
        if len(module.parse_nvidia_smi_gpu_csv(sample)) != count:
            problems.append(
                f"parse_nvidia_smi_gpu_csv did not report {count} device(s) for a "
                f"{count}-row sample; the GPU count must be discovered, never assumed."
            )
            break

    empty = module.empty_inventory("nvidia-smi was not found on PATH")
    if module.device_choices(empty) != ["cpu"]:
        problems.append("device_choices must offer only 'cpu' when nothing was enumerated.")
    if module.device_choices_with_manual_gpu(empty) != ["cpu", "cuda:0"]:
        problems.append(
            "device_choices_with_manual_gpu must keep one manual GPU entry so an "
            "explicit GPU choice stays possible when enumeration fails."
        )
    return problems


def _smoke_policy_problems(policy: Any, execution: Any) -> list[str]:
    problems: list[str] = []
    if policy.SMOKE_EPOCHS != 1:
        problems.append(f"SMOKE_EPOCHS is {policy.SMOKE_EPOCHS!r}, not 1.")
    if policy.SMOKE_NOTICE != REQUIRED_SMOKE_NOTICE:
        problems.append(
            "SMOKE_NOTICE no longer matches the sentence the UI must show: "
            f"{policy.SMOKE_NOTICE!r}."
        )
    if policy.smoke_epochs_for(True) != 1 or policy.smoke_epochs_for(False) is not None:
        problems.append("smoke_epochs_for must map True->1 and False->None.")
    if not policy.recommends_cpu_smoke("batlinet"):
        problems.append(
            "BatLiNet no longer recommends a CPU smoke run; the shared server's GPUs "
            "have too little free memory for it (milestone A requirement 1)."
        )
    advice = policy.smoke_device_recommendation(
        "batlinet", smoke=True, available_devices=["cpu", "cuda:0"]
    )
    if advice.get("device") != "cpu" or advice.get("recommended") is not True:
        problems.append(f"BatLiNet smoke recommendation is not CPU: {advice!r}.")

    # Neither the planner nor the policy may write or mutate a config.
    for module in (policy, execution):
        source = inspect.getsource(module)
        for forbidden in ("write_text(", "yaml.dump", "safe_dump"):
            if forbidden in source:
                problems.append(
                    f"{module.__name__} contains {forbidden!r}: a smoke run must never "
                    "write or mutate a benchmark config."
                )
    return problems


def _failure_problems(policy: Any, executor: Any) -> list[str]:
    problems: list[str] = []
    result = {
        "started": True,
        "completed": True,
        "success": False,
        "returncode": 1,
        "stdout": "RMSE 794.32",
        "stderr": _SAMPLE_OOM_STDERR,
        "error": "wrapper exited with returncode 1",
    }
    diagnosis = policy.classify_execution_failure(result, device="cuda:0")
    if diagnosis.get("kind") != "cuda_oom":
        problems.append(
            f"A CUDA OOM stderr was classified as {diagnosis.get('kind')!r}; the UI "
            "would show no actionable message."
        )
    if diagnosis.get("retried") is not False:
        problems.append("classify_execution_failure reported a retry; nothing may be retried.")
    if "CUDA out of memory" not in str(diagnosis.get("evidence_line") or ""):
        problems.append("The OOM evidence line is not copied from the run's own output.")
    blob = repr(diagnosis)
    for token in ("RMSE", "MAE", "MAPE"):
        if token in blob:
            problems.append(
                f"The failure diagnosis mentions {token}; a failed run must produce no metric."
            )

    problems.extend(_no_retry_loop_problems(executor))
    return problems


def _no_retry_loop_problems(executor: Any) -> list[str]:
    """The executor's subprocess call must not sit inside a loop.

    Checked on the parsed syntax tree rather than the file text, so the
    module's own prose ("never retries") cannot trip the check and a real
    ``for``/``while`` around the call cannot hide behind a comment.
    """
    problems: list[str] = []
    tree = ast.parse(inspect.getsource(executor))
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute_plan"
        ),
        None,
    )
    if function is None:
        return ["batteryml_web/agent/executor.py no longer defines execute_plan()."]

    for node in ast.walk(function):
        if not isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and _call_name(inner) in {"run", "subprocess.run"}:
                problems.append(
                    "batteryml_web/agent/executor.py calls the runner inside a loop in "
                    "execute_plan(); a failed run must never be retried."
                )
                break

    signature = inspect.signature(executor.execute_plan)
    for forbidden in ("retries", "max_retries", "attempts", "retry"):
        if forbidden in signature.parameters:
            problems.append(
                f"execute_plan() accepts a {forbidden!r} parameter; retrying is not a "
                "supported behaviour."
            )
    return problems


def _call_name(node: ast.Call) -> str:
    """Dotted name of a call target, e.g. ``subprocess.run``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        prefix = func.value.id if isinstance(func.value, ast.Name) else ""
        return f"{prefix}.{func.attr}" if prefix else func.attr
    return ""


def _observed_inventory(module: Any) -> dict[str, Any]:
    """Record what this machine reports. Absence is never a failure."""
    inventory = module.query_gpu_inventory()
    return {
        "available": inventory.get("available"),
        "device_count": inventory.get("device_count"),
        "devices": inventory.get("devices"),
        "error": inventory.get("error"),
    }


def check(output_dir: Path | None = None) -> CheckResult:
    """Establish the device / smoke / failure / streaming contracts statically."""
    root = repo_root()
    try:
        from batteryml_web.agent import execution as execution_module
        from batteryml_web.agent import executor as executor_module
        from batteryml_web.agent import run_policy as policy_module
        from batteryml_web.core import device_inventory as device_module
    except Exception as exc:  # noqa: BLE001 - reported as a SKIP, never hidden
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            SKIP,
            "The runtime modules could not be imported, so the contract is "
            f"unobservable here: {type(exc).__name__}: {exc}. Fix the environment "
            "first (see ENV-01).",
            {"exception_type": type(exc).__name__, "exception": str(exc)},
        )

    problems: list[str] = []
    problems.extend(_device_inventory_problems(device_module))
    problems.extend(_smoke_policy_problems(policy_module, execution_module))
    problems.extend(_failure_problems(policy_module, executor_module))

    config_path = root / BATLINET_CONFIG
    if not config_path.is_file():
        problems.append(
            f"The formal BatLiNet config {BATLINET_CONFIG.as_posix()} is missing; the "
            "smoke run has no config to use and must not invent one."
        )

    evidence: dict[str, Any] = {
        "batlinet_config": BATLINET_CONFIG.as_posix(),
        "batlinet_config_present": config_path.is_file(),
        "smoke_epochs": policy_module.SMOKE_EPOCHS,
        "smoke_notice": policy_module.SMOKE_NOTICE,
        "cpu_smoke_models": sorted(policy_module.SMOKE_CPU_RECOMMENDED_MODELS),
        "observed_gpu_inventory": _observed_inventory(device_module),
        "problems": problems,
    }

    artifacts: list[str] = []
    if output_dir is not None:
        written = write_json(Path(output_dir) / "runtime_contract.json", evidence)
        artifacts.append(written)

    if problems:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(problems)} runtime-contract problem(s). First: {problems[0]}",
            evidence,
            artifacts,
        )

    observed = evidence["observed_gpu_inventory"]
    gpu_note = (
        f"{observed['device_count']} GPU(s) observed"
        if observed.get("available")
        else "no GPU observed (not a failure)"
    )
    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        "Device enumeration is argv-based and count-agnostic; a smoke run is "
        "--epochs 1 on the formal config with no config written; BatLiNet smoke "
        "recommends CPU; CUDA OOM is recognised, never retried and yields no "
        f"metric. {gpu_note}.",
        evidence,
        artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
