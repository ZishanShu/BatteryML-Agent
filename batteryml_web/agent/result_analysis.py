"""The deterministic facts a result explanation is allowed to reason from.

The explanation used to be a narrator. Its prompt carried the dataset, the
task, the model, the config path, the metrics and a file list, and then
asked the model to "state which model/dataset/task ran ... restate the
metrics above exactly ... list the generated artifacts". It did that well,
and it answered none of the questions a battery scientist actually has: is
this plausible, is anything wrong with it, how does it compare, what should
I run next.

Answering those needs facts, not a better adjective. This module collects
them — from the run's own manifest, its own config and its own selected
artifact — so the assistant interprets evidence instead of guessing. Every
value here is read from the platform's existing deterministic layers:

* metrics come from the run context, unchanged and full precision;
* diagnostics come from
  :func:`~batteryml_web.agent.prediction_diagnostics.artifact_diagnostics`,
  which reads through the platform's one CPU-safe unpickler;
* the protocol comes from the config the run actually executed;
* the benchmark reference comes from
  :mod:`batteryml_web.core.benchmarks`, which is transcribed from the
  repository README and checked against it by a test.

**One artifact.** Diagnostics are read from the *selected* artifact and no
other. Enriching a prompt by reaching for a different file is the exact
substitution that made Review Results describe the wrong run.

**No science is recomputed here.** Where the backend already exposes a
number, it is passed through. This module's job is assembly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import benchmarks
from ..core.readiness import TASK_SOH
from .execution import metrics_for_task
from .prediction_diagnostics import artifact_diagnostics
from .run_policy import SMOKE_EPOCHS

#: How each metric should be read, per task. Units are part of the fact:
#: "RMSE 0.081" means nothing until it says whether that is cycles or a
#: capacity ratio, and the two differ by three orders of magnitude.
#:
#: SOH is a ratio (discharge capacity over first valid discharge capacity),
#: so its MAE and RMSE are ratios too — never silently rendered as percent.
#: RUL labels are cycle counts, so its RMSE and MAE are cycles while MAPE
#: stays a dimensionless ratio.
METRIC_UNITS: Dict[str, Dict[str, str]] = {
    TASK_SOH: {
        "MAE": "SOH ratio",
        "RMSE": "SOH ratio",
    },
    "RUL prediction": {
        "RMSE": "cycles",
        "MAE": "cycles",
        "MAPE": "ratio (multiply by 100 for percent)",
    },
}

#: The sentence a smoke run's analysis must carry.
SMOKE_STATEMENT = (
    "This run used --epochs 1: it is a workflow/deployment validation, not "
    "trained performance. Do not present its metric as a benchmark result "
    "and do not rank models from it."
)


def metric_unit(task: str | None, metric: str) -> str:
    """The unit for one metric under one task, or "" when not established."""
    return METRIC_UNITS.get(str(task or ""), {}).get(str(metric).upper(), "")


def read_protocol(config_path: str | Path | None) -> Dict[str, Any]:
    """The split, feature and label identity declared by the run's config.

    Read from the config the run executed, because that is what decides
    which benchmark subset the run is on. The split *class* is the thing
    that matters: ``MATRPrimaryTestTrainTestSplitter`` is MATR1 and
    ``MATRSecondaryTestTrainTestSplitter`` is MATR2, and the dataset name
    "MATR" alone distinguishes neither.
    """
    empty: Dict[str, Any] = {
        "splitter": None,
        "cell_data_path": None,
        "feature_extractor": None,
        "label_annotator": None,
        "model_class": None,
        "readable": False,
    }
    if not config_path:
        return empty
    path = Path(config_path)
    if not path.is_file():
        return empty
    try:
        import yaml

        parsed = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001 - an unreadable config is reported, not raised
        return empty
    if not isinstance(parsed, dict):
        return empty
    split = parsed.get("train_test_split") or {}
    return {
        "splitter": str(split.get("name") or "") or None,
        "cell_data_path": str(split.get("cell_data_path") or "") or None,
        "feature_extractor": str((parsed.get("feature") or {}).get("name") or "") or None,
        "label_annotator": str((parsed.get("label") or {}).get("name") or "") or None,
        "model_class": str((parsed.get("model") or {}).get("name") or "") or None,
        "readable": True,
    }


def _smoke_from_command(command: Any) -> bool:
    """Whether the executed argv asked for the smoke epoch count.

    Read from the command the run actually ran, which is the only place the
    epoch override appears: a smoke run is ``--epochs 1`` against the
    model's own formal config, with no config file written or mutated (see
    ``docs/SPEC.md`` EXEC-01).
    """
    argv = [str(part) for part in (command or [])]
    for index, part in enumerate(argv[:-1]):
        if part == "--epochs":
            return argv[index + 1].strip() == str(SMOKE_EPOCHS)
    return False


def _series_facts(series: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The subset of a described series worth putting in a prompt."""
    if not isinstance(series, dict) or not series.get("readable"):
        return None
    return {
        "count": series.get("count"),
        "finite": series.get("finite"),
        "non_finite": series.get("non_finite"),
        "distinct": series.get("distinct"),
        "constant": series.get("constant"),
        "min": series.get("min"),
        "max": series.get("max"),
    }


def build_analysis_context(
    result_context: Dict[str, Any],
    *,
    manifest: Optional[Dict[str, Any]] = None,
    artifact_path: str | Path | None = None,
    run_id: str = "",
    run_dir: str = "",
    project_name: str = "",
    identity_key: str = "",
    is_smoke: Optional[bool] = None,
    read_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Assemble every deterministic fact the analysis may use.

    Args:
        result_context: the run context from
            :func:`~batteryml_web.agent.results.build_result_context`.
        manifest: the selected run's manifest, for seed, device and command.
        artifact_path: the **selected** prediction artifact. Diagnostics are
            read from this file and no other.
        run_id / run_dir / project_name / identity_key: the selected-result
            identity, carried so the explanation is anchored to it.
        is_smoke: overrides the derived smoke flag when the caller knows.
        read_diagnostics: False skips the artifact read (for callers that
            already have it, and for tests).

    Returns a dict of sub-dicts. Nothing is invented: a fact that cannot be
    read is absent or None, never filled in with a plausible value.
    """
    manifest = dict(manifest or {})
    task = str(result_context.get("task") or manifest.get("task") or "")
    model_key = str(result_context.get("model") or manifest.get("model_key") or "")
    config_path = result_context.get("config_path") or manifest.get("config_path")

    smoke = (
        _smoke_from_command(manifest.get("command"))
        if is_smoke is None
        else bool(is_smoke)
    )
    # A smoke warning already on the context is evidence too. Either source
    # being true is enough: the cost of calling a trained run "smoke" is a
    # withheld comparison, and the cost of the reverse is presenting one
    # epoch as benchmark performance.
    if any("smoke" in str(w).lower() for w in (result_context.get("warnings") or [])):
        smoke = True

    metrics = dict(result_context.get("metrics") or {})
    metric_rows: List[Dict[str, Any]] = []
    for name in metrics_for_task(task):
        if name in metrics:
            metric_rows.append(
                {
                    "name": name,
                    "value": metrics[name],
                    "unit": metric_unit(task, name),
                }
            )
    for name, value in metrics.items():
        if name not in {row["name"] for row in metric_rows}:
            metric_rows.append(
                {"name": name, "value": value, "unit": metric_unit(task, name)}
            )

    protocol = read_protocol(config_path)

    diagnostics: Dict[str, Any] = {"read": False}
    if read_diagnostics and artifact_path:
        raw = artifact_diagnostics(artifact_path, run_id=run_id)
        diagnostics = {
            "read": True,
            "readable": bool(raw.get("readable")),
            "reason": raw.get("reason") or "",
            "predicted": _series_facts(raw.get("prediction")),
            "observed": _series_facts(raw.get("observed")),
            "sample_count": (raw.get("sample_index") or {}).get("length"),
            "distinct_cells": (raw.get("sample_index") or {}).get("distinct_cells"),
            "task_metadata": dict(raw.get("task_metadata") or {}),
            "saved_scores": dict(raw.get("scores") or {}),
        }

    primary = next(
        (row for row in metric_rows if row["name"] == benchmarks.BENCHMARK_METRIC),
        None,
    )
    primary_value = None
    if primary is not None and isinstance(primary["value"], (int, float)):
        primary_value = float(primary["value"])

    verdict = benchmarks.assess_comparability(
        task=task,
        splitter_name=protocol.get("splitter"),
        model_key=model_key,
        metric_name=benchmarks.BENCHMARK_METRIC if primary else None,
        metric_value=primary_value,
        unit=primary["unit"] if primary else None,
        is_smoke=smoke,
    )

    return {
        "identity": {
            "project": project_name,
            "run_id": run_id,
            "run_dir": run_dir,
            "identity_key": identity_key,
            "dataset": result_context.get("dataset"),
            "task": task,
            "model": model_key,
            "config_path": config_path,
            "seed": manifest.get("seed"),
            "device": manifest.get("device"),
            "run_status": result_context.get("run_status"),
            "success": result_context.get("success"),
            "is_smoke": smoke,
        },
        "metrics": metric_rows,
        "diagnostics": diagnostics,
        "protocol": protocol,
        "provenance": {
            "artifact": str(artifact_path) if artifact_path else None,
            "run_dir": run_dir,
            "identity_key": identity_key,
        },
        "benchmark": _benchmark_block(verdict),
        "warnings": [str(w) for w in (result_context.get("warnings") or [])],
    }


def _benchmark_block(verdict: benchmarks.Comparability) -> Dict[str, Any]:
    """The benchmark facts, flattened for the prompt and the UI.

    ``record`` may be present while ``comparable`` is False. That is not a
    contradiction: a published value can be worth naming as context when
    subtracting it would be wrong — a smoke run is exactly that case.
    """
    record = verdict.record
    block: Dict[str, Any] = {
        "comparable": verdict.comparable,
        "reasons": list(verdict.reasons),
        "explanation": verdict.explain(),
        "record": None,
        "difference": verdict.difference,
        "run_value": verdict.run_value,
    }
    if record is not None:
        block["record"] = {
            "source": record.source,
            "task": record.task,
            "dataset_subset": record.dataset_subset,
            "model_key": record.model_key,
            "metric": record.metric,
            "unit": record.unit,
            "lower_is_better": record.lower_is_better,
            "display": record.value.display(),
            "central": record.value.central,
            "dispersion": record.value.dispersion,
            "seeds": record.value.seeds,
            "censored": record.value.censored,
            "protocol": record.protocol,
        }
    return block


# --- the prompt -----------------------------------------------------------

#: The system prompt for result analysis.
#:
#: Two rules carry most of the weight. **Evidence is graded**: a number read
#: from the artifact is a fact, a calculation over it is a derived
#: diagnostic, and anything about *why* is a hypothesis that has to be
#: labelled and given a check. The old prompt had no such distinction, so a
#: guess and a measurement read identically.
#:
#: **Benchmarks are supplied, never recalled.** The model is told in as many
#: words that it may not produce a published value from its own knowledge.
#: A model that has read the BatteryML paper during training could otherwise
#: emit a number that looks authoritative and is attached to nothing.
ANALYSIS_SYSTEM_PROMPT = (
    "You are a battery-ML result analyst. Return JSON only.\n"
    "The structured context you are given is authoritative and complete: "
    "every number you use must come from it, verbatim and at full "
    "precision. Never recompute, round, rescale or invent a metric.\n"
    "Grade every claim. A FACT is present in the context. A DERIVED "
    "observation follows arithmetically from it. A HYPOTHESIS is a possible "
    "explanation — label it as one and name the check that would confirm "
    "it. Never state a cause as established.\n"
    "Benchmarks: use ONLY a reference supplied in the benchmark section. "
    "You must not recall, estimate or reconstruct any published value from "
    "your own knowledge, and if no reference is supplied you must say that "
    "no directly comparable published reference is available. Never compute "
    "a difference, percentage or ranking against a reference the context "
    "marks as not comparable.\n"
    "Do not call any model 'best' or rank models: one run is not a "
    "comparison. Do not claim convergence, superiority or benchmark "
    "performance from a smoke run.\n"
    "Be concise. Prefer one specific sentence to three general ones, and "
    "omit a section that has nothing substantive to say."
)

#: The JSON shape asked for. Sections are optional by design: forcing a
#: heading produces filler, and filler is what this step exists to remove.
ANALYSIS_SCHEMA = (
    '{"summary": str, "interpretation": str, "diagnostics": [str], '
    '"benchmark": str, "next_steps": [str]}'
)


def _render_metrics(analysis: Dict[str, Any]) -> List[str]:
    rows = analysis.get("metrics") or []
    if not rows:
        return ["metrics: unavailable (no canonical saved metrics for this run)."]
    lines = ["metrics (verbatim, full precision):"]
    for row in rows:
        unit = f" [{row['unit']}]" if row.get("unit") else ""
        lines.append(f"- {row['name']} = {row['value']!r}{unit}")
    return lines


def _render_diagnostics(analysis: Dict[str, Any]) -> List[str]:
    diagnostics = analysis.get("diagnostics") or {}
    if not diagnostics.get("read"):
        return ["prediction diagnostics: not read."]
    if not diagnostics.get("readable"):
        return [
            "prediction diagnostics: the selected artifact could not be read "
            f"({diagnostics.get('reason') or 'no reason recorded'})."
        ]
    lines = ["prediction diagnostics (from the selected artifact only):"]
    for label, key in (("predicted", "predicted"), ("observed", "observed")):
        facts = diagnostics.get(key)
        if not facts:
            lines.append(f"- {label}: unavailable")
            continue
        lines.append(
            f"- {label}: count={facts['count']}, finite={facts['finite']}, "
            f"non_finite={facts['non_finite']}, distinct={facts['distinct']}, "
            f"constant={facts['constant']}, min={facts['min']!r}, "
            f"max={facts['max']!r}"
        )
    if diagnostics.get("sample_count") is not None:
        lines.append(
            f"- samples={diagnostics.get('sample_count')}, "
            f"distinct_cells={diagnostics.get('distinct_cells')}"
        )
    metadata = diagnostics.get("task_metadata") or {}
    if metadata:
        lines.append(
            "- task metadata: "
            + ", ".join(f"{k}={v!r}" for k, v in sorted(metadata.items()))
        )
    return lines


def _render_benchmark(analysis: Dict[str, Any]) -> List[str]:
    block = analysis.get("benchmark") or {}
    record = block.get("record")
    if record is None:
        return [
            "benchmark: NO reference supplied. "
            f"{block.get('explanation') or ''} "
            "You must not supply one from your own knowledge.".strip()
        ]
    lines = [
        "benchmark reference (the only published value you may use):",
        f"- source: {record['source']}",
        f"- {record['model_key']} on {record['dataset_subset']}, "
        f"{record['metric']} = {record['display']} {record['unit']}"
        + (f" over {record['seeds']} seeds" if record["seeds"] > 1 else ""),
        f"- protocol: {record['protocol']}; lower is better",
    ]
    if record.get("censored"):
        lines.append(
            "- this cell is a censored threshold, not a measured value: it "
            "must not be treated as the number 1000 and no difference may "
            "be computed from it."
        )
    if block.get("comparable"):
        lines.append(
            f"- COMPARABLE. This run's value is {block.get('run_value')!r}; "
            f"the difference is {block.get('difference')!r} {record['unit']} "
            "(positive means this run has the larger error)."
        )
    else:
        lines.append(f"- NOT DIRECTLY COMPARABLE: {block.get('explanation')}")
        lines.append(
            "- State that the reference exists and why the comparison does "
            "not hold. Do not compute any difference or percentage."
        )
    return lines


def build_result_analysis_prompt(analysis: Dict[str, Any]) -> str:
    """The user prompt: the facts, then what to do with them.

    The ordering is deliberate — context first, instruction last — so the
    instruction is read against the facts rather than the facts against the
    instruction.
    """
    identity = analysis.get("identity") or {}
    protocol = analysis.get("protocol") or {}

    lines: List[str] = ["RUN (authoritative, do not contradict):"]
    for label, key in (
        ("project", "project"), ("run_id", "run_id"), ("dataset", "dataset"),
        ("task", "task"), ("model", "model"), ("config", "config_path"),
        ("seed", "seed"), ("device", "device"), ("status", "run_status"),
    ):
        lines.append(f"- {label}: {identity.get(key)}")
    if identity.get("is_smoke"):
        lines.append(f"- SMOKE RUN: {SMOKE_STATEMENT}")

    lines.append("")
    lines.extend(_render_metrics(analysis))

    lines.append("")
    lines.append("protocol:")
    for label, key in (
        ("train/test split", "splitter"),
        ("cell data", "cell_data_path"),
        ("feature extractor", "feature_extractor"),
        ("label annotator", "label_annotator"),
        ("model class", "model_class"),
    ):
        lines.append(f"- {label}: {protocol.get(key)}")

    lines.append("")
    lines.extend(_render_diagnostics(analysis))

    lines.append("")
    lines.extend(_render_benchmark(analysis))

    warnings = analysis.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("warnings recorded by the platform:")
        lines.extend(f"- {warning}" for warning in warnings)

    lines += [
        "",
        "Analyse this run. Do not narrate the configuration back: the reader "
        "can see it. Answer, in order:",
        "1. summary — what was run and what the headline number is.",
        "2. interpretation — what that magnitude means for THIS task and "
        "these units, and whether the sample and cell counts are plausible "
        "for this split.",
        "3. diagnostics — anything in the numbers above that warrants "
        "attention: near-constant predictions, non-finite values, a "
        "predicted range far from the observed range, a sample count that "
        "does not match the split, a smoke run being read as performance. "
        "Cite the number you are reacting to. If nothing warrants "
        "attention, say so in one line rather than inventing a concern.",
        "4. benchmark — only from the benchmark section above.",
        "5. next_steps — 2 to 4 concrete experiments for THIS run, each "
        "naming what it would resolve. No generic advice such as 'tune the "
        "hyperparameters'.",
        "",
        f"Return JSON with exactly these keys: {ANALYSIS_SCHEMA}. "
        "Use an empty string or empty list for a section with nothing "
        "substantive to say.",
    ]
    return "\n".join(lines)


def result_context_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The plain result context an analysis context was built around.

    :func:`~batteryml_web.agent.result_agent.explain_run_result` validates
    and echoes back a result context, and a caller that started from a
    manifest rather than from a finished agent run has one only implicitly.
    Deriving it here keeps the two objects in step: the metrics the
    validator checks are the metrics the prompt showed.
    """
    identity = analysis.get("identity") or {}
    return {
        "dataset": identity.get("dataset"),
        "task": identity.get("task"),
        "model": identity.get("model"),
        "config_path": identity.get("config_path"),
        "workspace": identity.get("run_dir"),
        "run_status": identity.get("run_status"),
        "success": identity.get("success"),
        "metrics": {
            row["name"]: row["value"] for row in (analysis.get("metrics") or [])
        },
        "warnings": list(analysis.get("warnings") or []),
    }


#: Section headings for the rendered analysis, in reading order.
ANALYSIS_SECTIONS: List[tuple] = [
    ("summary", "Result summary"),
    ("interpretation", "Interpretation"),
    ("diagnostics", "Diagnostics and possible concerns"),
    ("benchmark", "Benchmark comparison"),
    ("next_steps", "Recommended next steps"),
]


def render_analysis_markdown(parsed: Dict[str, Any]) -> str:
    """The parsed analysis as markdown, skipping empty sections.

    An empty section is omitted rather than printed with "N/A" under it: a
    heading with nothing beneath it is the filler this step set out to
    remove, and the prompt explicitly allows the model to leave one blank.
    """
    blocks: List[str] = []
    for key, heading in ANALYSIS_SECTIONS:
        value = parsed.get(key)
        if isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value if str(item).strip()]
            if not items:
                continue
            body = "\n".join(f"- {item}" for item in items)
        else:
            body = str(value or "").strip()
            if not body:
                continue
        blocks.append(f"**{heading}**\n\n{body}")
    if not blocks:
        return "The assistant returned no analysis content."
    return "\n\n".join(blocks)
