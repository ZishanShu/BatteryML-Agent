"""RUN-01 — a run says where it goes, and says nothing it did not observe.

Three quiet ways a platform lies to a user, all checked here without a GPU,
without data and without a Streamlit runtime:

* **it runs somewhere else than it says.** The page shows project "alpha"
  while the run is written under the default project, because the workspace
  came from a fallback rather than from the loaded project. So: the resolved
  project is the only source of an agent workspace, and the provenance block
  reads the workspace back out of the plan's own argv — what will really
  execute — rather than from a parallel field that could drift;
* **it invents a number.** A progress percentage, a remaining time, a GPU
  that was never enumerated, a metric rounded into a different value. So:
  an ETA exists only once the run itself has produced enough observations to
  support one, and is otherwise the literal word "estimating…"; a metric's
  compact form is display only and the exact value stays intact;
* **it leaks a secret.** The LLM settings summary the page may print carries
  a boolean for "is a key set", never the key.

This check is static and pure: it imports the four backend modules, exercises
them on values constructed here, and reads the UI source with the AST. It
never executes BatteryML, never calls an LLM and never touches ``data/``.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

from .common import FAIL, PASS, CheckResult, repo_root, run_check, write_json

CHECK_ID = "RUN-01"
CHECK_NAME = "run_context"

#: UI modules that must contain no runtime value of their own.
_UI_MODULES = (
    "batteryml_web/ui/agent_workflow.py",
    "batteryml_web/ui/workflow_sidebar.py",
    "batteryml_web/ui/settings_panel.py",
    "batteryml_web/ui/formatting.py",
)

#: Substrings that would mean a runtime fact was written into the page.
#: Checked against code with docstrings removed — prose is documentation,
#: a literal in an expression is behaviour.
_FORBIDDEN_IN_UI = (
    "RTX 3090",
    "cuda:0,1",
    "cuda:0,cuda:1",
    "range(8)",
    "web_projects/default",
)


def _code_without_docstrings(path: Path) -> str:
    """The module's code with every docstring dropped, via the AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _gather() -> tuple[list[str], dict[str, Any]]:
    from batteryml_web.agent.progress_estimate import (
        ESTIMATING_LABEL,
        EpochTimer,
        progress_line,
    )
    from batteryml_web.agent.provenance import (
        UNKNOWN,
        build_run_context,
        context_rows,
    )
    from batteryml_web.core.project_context import agent_workspace, resolve_project
    from batteryml_web.ui.formatting import full_precision, metric_display

    root = repo_root()
    problems: list[str] = []
    evidence: dict[str, Any] = {}

    # --- 1. The project decides the workspace, and only the project. ------
    workspaces = {}
    for name in ("alpha", "beta"):
        project = resolve_project(project_name=name)
        workspaces[name] = agent_workspace(project, "MATR", "batlinet")
    evidence["workspaces"] = workspaces
    for name, path in workspaces.items():
        if not path.startswith(f"runs/web_projects/{name}/"):
            problems.append(
                f"project {name!r} produced workspace {path!r}, which is not under "
                f"runs/web_projects/{name}/. Fix batteryml_web/core/project_context.py."
            )
    if workspaces["alpha"] == workspaces["beta"]:
        problems.append(
            "two different projects produced the same workspace, so a run cannot "
            "be attributed to a project. Fix agent_workspace()."
        )

    escaped = agent_workspace(
        resolve_project(project_root="/etc/passwd"), "MATR", "batlinet"
    )
    evidence["escape_attempt"] = escaped
    if not escaped.startswith("runs/"):
        problems.append(
            f"a project root outside runs/ produced {escaped!r}. Every agent "
            "workspace must stay under runs/."
        )

    # --- 2. Provenance reads the workspace from what will really execute. -
    project = resolve_project(project_name="alpha")
    planned = agent_workspace(project, "MATR", "batlinet")
    plan = {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": "batlinet",
        "config_path": "configs/external/batlinet/matr_1.yaml",
        "command": ["python", "x.py", "run", "c.yaml", "--workspace", planned,
                    "--device", "cpu", "--seed", "0"],
        "epochs": None,
    }
    run_context = build_run_context(
        project=project, plan=plan, device_label="CPU", seed=0
    )
    evidence["run_context_workspace"] = run_context["workspace"]
    if run_context["workspace"] != planned:
        problems.append(
            f"provenance reported workspace {run_context['workspace']!r} for a plan "
            f"whose argv says {planned!r}. The two must not diverge."
        )
    labels = [label for label, _ in context_rows(run_context)]
    evidence["context_fields"] = labels
    for required in ("Project", "Run directory", "Dataset", "Task", "Model",
                     "Config", "Seed", "Epochs", "Device"):
        if required not in labels:
            problems.append(
                f"the run context omits {required!r}; a user cannot tell where "
                "the run went. Fix batteryml_web/agent/provenance.py FIELD_ORDER."
            )

    # A plan that carries nothing must say so, not render an empty cell.
    # "Epochs" is the one field with a truthful answer in the absence of a
    # plan: no override means the config's own default, which is a fact.
    empty_rows = context_rows(build_run_context(project=None, plan=None))
    empty_values = {label: value for label, value in empty_rows}
    blank = [
        label
        for label, value in empty_rows
        if label != "Epochs" and value != UNKNOWN
    ]
    evidence["empty_context"] = empty_values
    if blank:
        problems.append(
            f"an empty run context rendered something other than {UNKNOWN!r} for "
            f"{blank}. Missing is stated, never blank."
        )

    # --- 3. An ETA is earned, never invented. ----------------------------
    timer = EpochTimer()
    timer.observe(current=1, total=100, elapsed_s=10.0)
    evidence["eta_after_one_epoch"] = timer.eta_seconds()
    if timer.eta_seconds() is not None:
        problems.append(
            "an ETA was produced from a single observation. One epoch is not "
            "evidence of a rate; see MIN_OBSERVATIONS."
        )
    early = progress_line(
        "Training", current=1, total=100, elapsed_s=10.0,
        eta_s=None, estimating=True,
    )
    evidence["early_progress_line"] = early
    if ESTIMATING_LABEL not in early:
        problems.append(
            f"a progress line with no ETA evidence did not say {ESTIMATING_LABEL!r}: "
            f"{early!r}. Silence or a guessed number are both wrong."
        )
    timer.observe(current=2, total=100, elapsed_s=20.0)
    evidence["eta_after_two_epochs"] = timer.eta_seconds()
    if timer.eta_seconds() is not None:
        problems.append(
            "an ETA was produced from a single measured interval. One interval "
            "cannot distinguish a rate from a fluke; see MIN_OBSERVATIONS."
        )
    timer.observe(current=3, total=100, elapsed_s=30.0)
    eta = timer.eta_seconds()
    evidence["eta_after_three_epochs"] = eta
    if eta is None:
        problems.append(
            "no ETA was produced after three evenly spaced observations, so the "
            "estimator never becomes useful. Check EpochTimer.per_unit_seconds()."
        )

    # --- 4. A compact metric is a reading, not a new value. ---------------
    exact = 455.5853271484375
    display = metric_display({"RMSE": exact}, "RMSE")
    evidence["metric_display"] = display
    if display["exact"] != full_precision(exact) or str(exact) not in display["exact"]:
        problems.append(
            f"the exact metric reading {display['exact']!r} does not carry the stored "
            f"value {exact!r}. The compact form is display only."
        )
    if display["compact"] == display["exact"]:
        problems.append(
            "the compact and exact metric readings are identical, so nothing was "
            "actually made compact. Check ui/formatting.py METRIC_DECIMALS."
        )

    # --- 5. Nothing about this machine is written into the page. ----------
    found: dict[str, list[str]] = {}
    for relative in _UI_MODULES:
        path = root / relative
        if not path.exists():
            problems.append(f"{relative} does not exist; this check is out of date.")
            continue
        code = _code_without_docstrings(path)
        hits = [token for token in _FORBIDDEN_IN_UI if token in code]
        if hits:
            found[relative] = hits
            problems.append(
                f"{relative} contains hardcoded runtime value(s) {hits}. GPU names, "
                "device strings, GPU counts and project paths come from the runtime, "
                "never from the source."
            )
    evidence["hardcoded_hits"] = found

    # --- 6. The settings summary carries no secret. -----------------------
    settings_code = _code_without_docstrings(root / "batteryml_web/ui/settings_panel.py")
    summary_leaks = []
    for node in ast.walk(ast.parse(settings_code)):
        if isinstance(node, ast.FunctionDef) and node.name == "llm_settings_summary":
            for sub in ast.walk(node):
                # The key may only be reduced to a boolean.
                if (
                    isinstance(sub, ast.Constant)
                    and sub.value == "llm_api_key"
                    and not any(
                        isinstance(parent, ast.Call)
                        and isinstance(parent.func, ast.Name)
                        and parent.func.id == "bool"
                        for parent in ast.walk(node)
                    )
                ):
                    summary_leaks.append("llm_api_key is read without bool()")
    if 'type="password"' not in settings_code and "type='password'" not in settings_code:
        summary_leaks.append("the API key field is not rendered with type='password'")
    evidence["secret_findings"] = summary_leaks
    for leak in summary_leaks:
        problems.append(
            f"batteryml_web/ui/settings_panel.py: {leak}. The API key is never "
            "displayed, logged or returned."
        )

    return problems, evidence


def check(output_dir: Path | None = None) -> CheckResult:
    problems, evidence = _gather()
    artifacts: list[str] = []
    if output_dir is not None:
        artifacts.append(write_json(Path(output_dir) / "run_context.json", evidence))
    if problems:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(problems)} run-context problem(s). First: {problems[0]}",
            evidence,
            artifacts,
        )
    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        "The loaded project is the only source of an agent workspace and cannot "
        "escape runs/; provenance reads the run directory from the plan's own argv "
        "and names every field or states that it is unknown; an ETA appears only "
        "once the run has produced enough observations and says so until then; "
        "a compact metric is a reading beside the exact stored value; and "
        "no GPU name, device string, GPU count, project path or API key is written "
        "into the UI source.",
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
