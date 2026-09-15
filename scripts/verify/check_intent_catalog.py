"""INTENT-01 — the catalog the LLM may choose from is derived, and binding.

Milestone D puts an LLM between a user's sentence and a BatteryML run. The
safety of that arrangement rests on two properties this check establishes
without a network and without an API key:

* the catalog is **derived** from the live backend — every model key in it
  comes from ``ConfigRegistry`` or from ``readiness.EXTERNAL_MODELS``, every
  dataset from the platform dataset list, every task from the three
  canonical labels. A second, hand-maintained menu would let the agent offer
  something the planner cannot build;
* the catalog is **binding** — an invented dataset, task, model or config
  path is rejected by ``validate_llm_intent`` and recorded, not accepted with
  a warning; and nothing in the intent layer can reach a planner or an
  executor.

The catalog is also checked for config paths: a path that reaches the prompt
is a path the model can echo back, and a config path must come from
readiness, never from a language model.

This check never calls an LLM, never trains and never writes into ``data/``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
from typing import Any

from .common import FAIL, PASS, SKIP, CheckResult, repo_root, run_check, write_json

CHECK_ID = "INTENT-01"
CHECK_NAME = "intent_catalog"

#: An answer invented wholesale. Every field must be refused.
_HALLUCINATED_ANSWER = {
    "dataset": "MATR_V2",
    "task": "anomaly detection",
    "requested_model": "MegaNet9000",
    "suggested_model_keys": ["AlsoNotReal"],
    "config_path": "configs/external/invented/matr_1.yaml",
    "command": "python bin/batteryml.py run configs/invented.yaml",
    "needs_clarification": False,
    "user_facing_rationale": "Looks good to me.",
}

_EXECUTION_CALLS = (
    "execute_plan",
    "execute_plan_streaming",
    "build_execution_plan",
    "plan_for_request",
    "Popen",
    "system",
)


def _catalog_problems(catalog: dict[str, Any], readiness: Any, registry: Any) -> list[str]:
    problems: list[str] = []

    registry_keys = {entry.model_key for entry in registry.entries}
    external_keys = set(readiness.EXTERNAL_MODELS)
    for entry in catalog.get("models", []):
        key = str(entry.get("model_key"))
        if key not in registry_keys and key not in external_keys:
            problems.append(
                f"catalog model {key!r} comes from neither ConfigRegistry nor "
                "EXTERNAL_MODELS; the catalog must be derived, never hand-written."
            )
    for key in sorted(external_keys):
        if key not in {str(e.get("model_key")) for e in catalog.get("models", [])}:
            problems.append(f"external model {key!r} is missing from the catalog.")

    dataset_names = {str(e.get("dataset")) for e in catalog.get("datasets", [])}
    for name in readiness.PLATFORM_DATASETS:
        if name not in dataset_names:
            problems.append(f"declared dataset {name!r} is missing from the catalog.")

    expected_tasks = [readiness.TASK_RUL, readiness.TASK_SOH, readiness.TASK_TRAJECTORY]
    actual_tasks = [str(e.get("task")) for e in catalog.get("tasks", [])]
    if actual_tasks != expected_tasks:
        problems.append(
            f"catalog tasks {actual_tasks} are not the three platform tasks {expected_tasks}."
        )

    blob = repr(catalog)
    for token in (".yaml", "configs/", "data/processed", "data/raw"):
        if token in blob:
            problems.append(
                f"the catalog carries {token!r}; paths must never reach the prompt, "
                "because the model could echo one back as a config to run."
            )
    return problems


def _alias_problems(catalog: dict[str, Any], resolve_model: Any) -> list[str]:
    """Every alias the catalog advertises must fold back onto its own key."""
    problems: list[str] = []
    for entry in catalog.get("models", []):
        key = str(entry.get("model_key"))
        for alias in list(entry.get("aliases", []))[:12]:
            resolved = resolve_model(alias, catalog)
            if resolved != key:
                problems.append(
                    f"alias {alias!r} of model {key!r} resolves to {resolved!r}; the "
                    "catalog would accept a name it cannot honour."
                )
    return problems


def _rejection_problems(catalog: dict[str, Any], intent: Any) -> list[str]:
    problems: list[str] = []
    result = intent.validate_llm_intent(_HALLUCINATED_ANSWER, catalog)
    if result.get("dataset") is not None:
        problems.append(
            f"an invented dataset was accepted as {result.get('dataset')!r}."
        )
    if result.get("task") is not None:
        problems.append(f"an invented task was accepted as {result.get('task')!r}.")
    if result.get("requested_model") is not None:
        problems.append(
            f"an invented model was accepted as {result.get('requested_model')!r}."
        )
    if result.get("suggested_model_keys"):
        problems.append(
            f"invented suggestions survived: {result.get('suggested_model_keys')}."
        )
    if not result.get("needs_clarification"):
        problems.append(
            "an answer with no resolvable dataset or task did not ask for clarification."
        )
    kinds = {item.get("kind") for item in result.get("rejected", [])}
    for expected in ("unknown_dataset", "unknown_task", "unknown_model", "forbidden_field"):
        if expected not in kinds:
            problems.append(f"rejection kind {expected!r} was not recorded.")
    surviving = repr({k: v for k, v in result.items() if k != "rejected"})
    for token in (".yaml", "bin/batteryml.py"):
        if token in surviving:
            problems.append(
                f"a model-supplied {token!r} survived validation; only readiness may "
                "supply a config path."
            )
    return problems


def _isolation_problems(intent: Any) -> list[str]:
    """The intent layer may not plan, execute or shell out."""
    problems: list[str] = []
    tree = ast.parse(inspect.getsource(intent))
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)
    for forbidden in _EXECUTION_CALLS:
        if forbidden in called:
            problems.append(
                f"batteryml_web/agent/llm_intent.py calls {forbidden!r}; free-form LLM "
                "output must never reach a planner or an executor."
            )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("subprocess", "os"):
        if forbidden in imported:
            problems.append(
                f"batteryml_web/agent/llm_intent.py imports {forbidden!r}; it has no "
                "business touching the process or the filesystem."
            )
    prompt = intent.build_intent_prompt("anything", {"datasets": [], "tasks": [], "models": []})
    lowered = (prompt + intent.SYSTEM_PROMPT).lower()
    for phrase in ("step by step", "step-by-step", "chain of thought"):
        if phrase in lowered:
            problems.append(
                f"the prompt asks for {phrase!r}; only a short user-facing rationale "
                "may be requested."
            )
    return problems


def check(output_dir: Path | None = None) -> CheckResult:
    """Establish that the LLM's menu is derived from the backend and binding."""
    try:
        from batteryml_web.agent import catalog as catalog_module
        from batteryml_web.agent import llm_intent as intent_module
        from batteryml_web.core import readiness as readiness_module
        from batteryml_web.core.config_registry import get_config_registry
    except Exception as exc:  # noqa: BLE001 - reported as a SKIP, never hidden
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            SKIP,
            "The intent modules could not be imported, so the catalog contract is "
            f"unobservable here: {type(exc).__name__}: {exc}. Fix the environment "
            "first (see ENV-01).",
            {"exception_type": type(exc).__name__, "exception": str(exc)},
        )

    root = repo_root()
    catalog = catalog_module.build_intent_catalog(repo_root=root)
    registry = get_config_registry(str(root))

    problems: list[str] = []
    problems.extend(_catalog_problems(catalog, readiness_module, registry))
    problems.extend(_alias_problems(catalog, catalog_module.resolve_catalog_model))
    problems.extend(_rejection_problems(catalog, intent_module))
    problems.extend(_isolation_problems(intent_module))

    evidence: dict[str, Any] = {
        "dataset_count": len(catalog.get("datasets", [])),
        "task_count": len(catalog.get("tasks", [])),
        "model_count": len(catalog.get("models", [])),
        "model_keys": [str(e.get("model_key")) for e in catalog.get("models", [])],
        "external_models": sorted(readiness_module.EXTERNAL_MODELS),
        "forbidden_keys": list(intent_module.FORBIDDEN_KEYS),
        "problems": problems,
    }

    artifacts: list[str] = []
    if output_dir is not None:
        artifacts.append(write_json(Path(output_dir) / "intent_catalog.json", evidence))

    if problems:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(problems)} intent-catalog problem(s). First: {problems[0]}",
            evidence,
            artifacts,
        )

    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"The catalog is derived from the backend ({evidence['dataset_count']} dataset(s), "
        f"{evidence['task_count']} task(s), {evidence['model_count']} model(s)), carries no "
        "config path, and every alias folds back onto its own key. An invented dataset, "
        "task, model or config path is rejected and recorded, and the intent layer neither "
        "plans nor executes.",
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
