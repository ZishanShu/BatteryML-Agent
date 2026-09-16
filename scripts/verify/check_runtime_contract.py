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
* a device string is a single torch device — a multi-GPU form such as
  ``"cuda:0,1"`` is refused at the planner boundary, a multi-GPU *selection*
  produces no device string at all, no model is registered as multi-GPU
  capable, and no DataParallel, DDP or torchrun path is exposed anywhere in
  the web layer (``docs/GPU_MULTI_GPU_AUDIT.md`` says why);
* a CUDA out-of-memory failure is recognised, is never retried, and no
  metric is ever produced from a failed run;
* the streaming executor stays Streamlit-independent, recognises the real
  tqdm lines the installed models emit, and fabricates no percentage.

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
    "0, NVIDIA GeForce RTX 3090, 24576, 22500, 2076, 97\n"
    "1, NVIDIA GeForce RTX 3090, 24576, 14000, 10576, 12\n"
)

_SAMPLE_OOM_STDERR = (
    "Traceback (most recent call last):\n"
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
)

#: Real progress lines the installed models emit; see
#: batteryml/models/nn_model.py (the upstream "Traning" typo) and
#: batteryml_web/models/batlinet.py.
_SAMPLE_PROGRESS_LINES = (
    ("Training BatLiNet:  10%|#         | 1/10 [00:06<00:54,  6.80s/it]", 1, 10),
    ("Traning:   1%|          | 1/100 [00:00<00:30,  3.20it/s]", 1, 100),
    ("Training CPTransformer: 100%|##########| 1/1 [00:07<00:00,  7.00s/it]", 1, 1),
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
        sample = "".join(f"{i}, RTX 3090, 24576, 100, 24476, 0\n" for i in range(count))
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


#: Names that would mean a multi-GPU execution path exists. The audit in
#: docs/GPU_MULTI_GPU_AUDIT.md concludes there is none and that none may be
#: advertised; this list keeps a fake one from appearing by accident.
_MULTI_GPU_TOKENS = (
    "DataParallel",
    "DistributedDataParallel",
    "init_process_group",
    "torchrun",
    "nproc_per_node",
)


def _device_validation_problems(device_module: Any, execution: Any) -> list[str]:
    """One torch device per run, enforced before the wrapper is invoked."""
    problems: list[str] = []
    for good in ("cpu", "cuda", "cuda:0", "cuda:7", "mps"):
        try:
            device_module.validate_device(good)
        except ValueError as exc:
            problems.append(f"validate_device rejected the valid device {good!r}: {exc}")
    for bad in ("cuda:0,1", "cuda:0,cuda:1", "", "gpu", "cuda:x"):
        try:
            device_module.validate_device(bad)
        except ValueError:
            continue
        problems.append(
            f"validate_device accepted {bad!r}; BatteryML passes --device straight to "
            ".to(device), which takes exactly one device."
        )

    source = inspect.getsource(execution)
    if "validate_device(" not in source:
        problems.append(
            "build_execution_plan no longer validates the device; a multi-device "
            "string would reach the CLI and fail inside a started run."
        )

    # A selection is structured, and a multi-GPU one yields no device string.
    for ids, expected in (([], "cpu"), ([0], "cuda:0"), ([3], "cuda:3")):
        selection = device_module.build_device_selection(ids)
        actual = device_module.device_string_for_selection(selection)
        if actual != expected:
            problems.append(
                f"a selection of {ids or 'no GPU'} produced device {actual!r}, "
                f"expected {expected!r}."
            )
    for ids in ([0, 1], [0, 1, 2], list(range(8))):
        selection = device_module.build_device_selection(ids)
        if selection.get("mode") != device_module.MODE_MULTI_GPU:
            problems.append(f"a selection of {ids} was not reported as multi-GPU.")
        if device_module.device_string_for_selection(selection) is not None:
            problems.append(
                f"a multi-GPU selection of {ids} produced a device string; there is "
                "no single torch device for it and inventing one would advertise a "
                "capability the platform does not have."
            )

    web_root = repo_root() / "batteryml_web"
    for path in sorted(web_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in _MULTI_GPU_TOKENS:
            # The audit document may name them; the code may not use them.
            if f"{token}(" in text:
                problems.append(
                    f"{path.relative_to(repo_root()).as_posix()} uses {token}; the web "
                    "layer must not expose a multi-GPU run (see "
                    "docs/GPU_MULTI_GPU_AUDIT.md)."
                )
    return problems


def _multi_gpu_gate_problems(policy: Any, device_module: Any) -> list[str]:
    """No model claims multi-GPU, and a multi-GPU choice blocks Run."""
    problems: list[str] = []
    if policy.MULTI_GPU_CAPABLE_MODELS:
        problems.append(
            "a model is registered as multi-GPU capable "
            f"({sorted(policy.MULTI_GPU_CAPABLE_MODELS)}), but no DDP or DataParallel "
            "path exists. Registering one is a claim that it works; see "
            "docs/GPU_MULTI_GPU_AUDIT.md."
        )
    for ids in ([], [0], [5]):
        blockers = policy.execution_blockers(
            "any_model", device_module.build_device_selection(ids)
        )
        if blockers:
            problems.append(
                f"a selection of {ids or 'no GPU'} was blocked: {blockers}. Single-GPU "
                "and CPU behaviour must be unchanged."
            )
    selection = device_module.build_device_selection([0, 1])
    blockers = policy.execution_blockers("any_model", selection)
    if not blockers:
        problems.append(
            "a multi-GPU selection was not blocked, so Run would be offered for a "
            "combination the runtime cannot execute."
        )
    elif not blockers[0].startswith(policy.MULTI_GPU_UNSUPPORTED_MESSAGE):
        problems.append(
            "the multi-GPU blocker no longer opens with the message the UI shows: "
            f"{blockers[0]!r}."
        )
    if selection != {"mode": device_module.MODE_MULTI_GPU, "gpu_ids": [0, 1]}:
        problems.append(
            "the capability gate rewrote the selection; it must report, never "
            "quietly reduce a multi-GPU choice to one GPU."
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


def _streaming_problems(streaming: Any) -> list[str]:
    """The streaming executor's parser and Streamlit-independence contract."""
    problems: list[str] = []
    source = inspect.getsource(streaming)
    if "shell=False" not in source:
        problems.append(
            "batteryml_web/agent/streaming.py no longer passes shell=False; a shell "
            "string must never be executed."
        )
    if "import streamlit" in source:
        problems.append(
            "batteryml_web/agent/streaming.py imports streamlit; the executor must stay "
            "Streamlit-independent."
        )
    for line, current, total in _SAMPLE_PROGRESS_LINES:
        parsed = streaming.parse_progress_line(line)
        if parsed is None:
            problems.append(
                f"parse_progress_line did not recognise a real tqdm line: {line!r}."
            )
            continue
        if parsed.get("current") != current or parsed.get("total") != total:
            problems.append(
                f"parse_progress_line read {parsed.get('current')}/{parsed.get('total')} "
                f"from {line!r}; expected {current}/{total}."
            )
    if streaming.parse_progress_line("Loading data from data/processed/MATR") is not None:
        problems.append(
            "parse_progress_line invented progress for a line carrying no counter; a "
            "fabricated percentage is worse than none."
        )
    stage = streaming.classify_stage("Traceback (most recent call last):")
    if stage is not None:
        problems.append(f"classify_stage invented the stage {stage!r} for a traceback line.")
    return problems


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
        from batteryml_web.agent import streaming as streaming_module
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
    problems.extend(_device_validation_problems(device_module, execution_module))
    problems.extend(_multi_gpu_gate_problems(policy_module, device_module))
    problems.extend(_smoke_policy_problems(policy_module, execution_module))
    problems.extend(_failure_problems(policy_module, executor_module))
    problems.extend(_streaming_problems(streaming_module))

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
        "recommends CPU; a device selection is structured and a multi-GPU one "
        "produces no device string and blocks Run; CUDA OOM is recognised, never "
        "retried and yields no metric; streaming progress is parsed, never "
        f"fabricated. {gpu_note}.",
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
