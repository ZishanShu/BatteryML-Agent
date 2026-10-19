"""Explain-metrics context and prompt, version 2 (EXPL-03).

Version 1 (:mod:`.result_analysis`) is kept unchanged: it is what produced
the archived invocations, and a reviewer must be able to regenerate exactly
what they were sent. Version 2 fixes what review of those invocations found:

* **UI task vs scientific target.** v1 sent ``task: RUL prediction`` and the
  annotator class name, so the model described the target from the class
  name. v2 sends the UI task label *and* the scientific target read from the
  label annotator actually configured — name, unit, parameters, definition,
  source file and SHA-256 — and whether rebuilt labels matched the stored
  targets (only when a native check did that).
* **Scale.** v1 sent the stored prediction/target ranges next to metrics in
  cycles without saying the stored arrays are label-transformed. v2 states
  the array scale with its status, and adds the native-scale ranges only when
  a native check replayed the metrics.
* **Missing is not zero.** v1 sent ``samples=0, distinct_cells=0`` when no
  sample index was saved. v2 reports ``null`` with ``NOT_RECORDED`` and keeps
  the legacy counters separately, labelled as such.
* **Benchmark.** v2 never sends ``COMPARABLE`` and never supplies a
  difference: comparability needs evidence about the published cohort,
  target, window, label handling and evaluator that no automatic step has.
  It lists each check with its status.
* **System prompt.** v2 uses ``system_prompt_v2.txt`` verbatim (pinned by
  SHA-256 in the tests).

Native evidence comes only from a report written by
``scripts/native_run_check.py`` for **this run and this exact prediction
file** (SHA-256 match). Without one, every native field says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import native_check as nc

CONTEXT_VERSION = "v2"

#: ``system_prompt_v2.txt`` from the reproducibility preparation package,
#: verbatim (1838 bytes, SHA-256 56c6276e…b3ac, including the final newline).
ANALYSIS_SYSTEM_PROMPT_V2 = (
    "You are a battery-ML result analyst. Return JSON only with exactly these keys: summary, interpretation, "
    "diagnostics, benchmark, next_steps. Do not use Markdown fences.\n"
    "Use the scientific_target and its evidence, not the UI task label, to describe what is predicted. If a "
    "definition, scale, identity or pairing is unrecorded, say that it is unrecorded.\n"
    "Preserve each stored metric value and unit exactly when reporting it. Do not replace or recompute the saved "
    "metrics. You may give a separately labelled DERIVED display conversion or rounded value, including MAPE ratio "
    "multiplied by 100, if you state the original value, formula, resulting unit and rounding. Do not invent "
    "measurements or fill missing values.\n"
    "Prefix factual statements with FACT, arithmetic displays with DERIVED, and possible explanations with "
    "HYPOTHESIS. A hypothesis must name a check that would test it. Do not state a mechanism or cause as "
    "established from aggregate scores.\n"
    "An array count is not a verified cell count. A missing metadata field is not zero. Do not infer cell identity, "
    "correct pairing or split coverage from equal array lengths.\n"
    "Keep metric scale and diagnostic-array scale distinct. A configured transformation does not establish its "
    "fitted parameters or which scale a saved array uses.\n"
    "Use only the supplied benchmark reference. Treat comparability as established only when the context records "
    "the matched cohort, target, observation window, label handling and evaluation definitions with supporting "
    "evidence. If these checks are incomplete, report comparability as NOT VERIFIED and do not claim reproduction, "
    "superiority or improved generalization.\n"
    "Separate a proposed action from an executed action and its recorded outcome. Give 2 to 4 specific next checks "
    "when evidence is missing. Do not claim that a suggested change has already been made.\n"
)

ANALYSIS_SCHEMA_V2 = '{"summary": str, "interpretation": str, "diagnostics": [str], "benchmark": str, "next_steps": [str]}'


def _config(config_path: Any) -> Optional[Dict[str, Any]]:
    if not config_path or not Path(str(config_path)).is_file():
        return None
    try:
        import yaml

        loaded = yaml.safe_load(Path(str(config_path)).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable config is reported as missing
        return None
    return loaded if isinstance(loaded, dict) else None


def _scientific_target(config: Optional[Dict[str, Any]], report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not config or not isinstance(config.get("label"), dict):
        return {"name": None, "unit": None, "status": nc.NOT_RECORDED,
                "reason": "the run config could not be read, so the label annotator is unknown"}
    try:
        import batteryml  # noqa: F401 - registers the annotators
        from batteryml.builders import LABEL_ANNOTATORS

        annotator = LABEL_ANNOTATORS.build(dict(config["label"]))
    except Exception as exc:  # noqa: BLE001 - reported, not guessed
        return {"name": None, "unit": None, "status": nc.NOT_RECORDED,
                "label_annotator": config["label"].get("name"), "reason": f"annotator not buildable: {exc}"}
    target = nc.scientific_target(annotator)
    target["source_scope"] = "implementation in the checkout that built this context"
    native = (report or {}).get("scientific_target") or {}
    target["rebuilt_labels_match_stored_targets"] = native.get("rebuilt_labels_match_stored_targets", nc.NOT_VERIFIED)
    return target


def _configured_order(config: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    section = (config or {}).get("label_transformation")
    if not isinstance(section, dict):
        return None
    children = section.get("transformations")
    if isinstance(children, list):
        return [str((child or {}).get("name")) for child in children]
    return [str(section.get("name"))]


def _replay_block(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    replay = (report or {}).get("metric_replay")
    if not replay:
        return {"status": nc.NOT_PERFORMED}
    return {
        "status": replay.get("status"),
        "pairs": replay.get("pairs"),
        "mape_zero_target_count": replay.get("mape_zero_target_count"),
        "mape_definition": replay.get("mape_definition"),
        "metrics": {
            name: {k: row.get(k) for k in ("stored_score", "evaluator_replay", "independent_float64",
                                            "evaluator_equals_stored", "independent_matches_stored")}
            for name, row in (replay.get("metrics") or {}).items()
        },
    }


def _range(facts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not facts or not facts.get("present"):
        return None
    return {k: facts.get(k) for k in ("count", "finite", "min", "max", "mean")}


def _benchmark_block(v1_block: Dict[str, Any], identity_status: str, replay_status: str) -> Dict[str, Any]:
    record = v1_block.get("record")
    if record is None:
        return {"reference": None, "comparability_status": "NO_REFERENCE",
                "explanation": v1_block.get("explanation")}
    this_run = "VERIFIED_FOR_THIS_RUN" if identity_status == nc.MATCH else nc.NOT_VERIFIED
    replayed = "VERIFIED_FOR_THIS_RUN" if replay_status == nc.VERIFIED else nc.NOT_VERIFIED
    return {
        "reference": {k: record.get(k) for k in ("source", "task", "dataset_subset", "model_key", "metric", "unit",
                                                   "lower_is_better", "display", "central", "dispersion", "seeds",
                                                   "censored", "protocol")},
        "comparability_status": nc.NOT_VERIFIED,
        "checks": [
            {"check": "actual held-out cell identities and exclusions of this run", "status": this_run},
            {"check": "same cohort as the published benchmark", "status": nc.NOT_VERIFIED},
            {"check": "scientific target and label handling against the benchmark protocol", "status": nc.NOT_VERIFIED},
            {"check": "observation window against the benchmark protocol", "status": nc.NOT_VERIFIED},
            {"check": "metric evaluator and zero-target policy of this run", "status": replayed},
            {"check": "metric evaluator against the benchmark protocol", "status": nc.NOT_VERIFIED},
        ],
        "note": "No difference to the reference is supplied while comparability is NOT_VERIFIED.",
    }


def build_analysis_context_v2(
    v1: Dict[str, Any],
    *,
    project_root: Optional[str],
    predictions_file: Optional[str],
    native_report: Optional[Dict[str, Any]] = None,
    lookup_native_report: bool = True,
) -> Dict[str, Any]:
    """The v2 context, built on the v1 facts of the same selected result.

    ``v1`` is :func:`~batteryml_web.agent.result_analysis.build_analysis_context`
    for the selected run; nothing in it is recomputed. ``native_report`` may be
    passed directly (tests); otherwise the newest report for this run whose
    prediction SHA-256 equals ``predictions_file``'s is used, if any.
    """
    identity = dict(v1.get("identity") or {})
    config = _config(identity.get("config_path"))
    predictions_sha = None
    if predictions_file and Path(str(predictions_file)).is_file():
        predictions_sha = nc.sha256_file(Path(str(predictions_file)))
    report = native_report
    if report is None and lookup_native_report and project_root and identity.get("run_id"):
        report = nc.matching_report(Path(project_root), str(identity["run_id"]), predictions_sha)
    if report is not None and ((report.get("files") or {}).get("predictions") or {}).get("sha256") != predictions_sha:
        report = None  # a report about another file is not evidence about this one

    diagnostics_v1 = v1.get("diagnostics") or {}
    identity_check = (report or {}).get("identity") or {}
    identity_status = identity_check.get("status", nc.NOT_VERIFIED)
    replay = _replay_block(report)
    scale = (report or {}).get("array_scale") or {}
    native_replay = (report or {}).get("metric_replay") or {}
    sample_count = diagnostics_v1.get("sample_count")
    saved_index_present = bool(sample_count) if diagnostics_v1.get("read") else None

    diagnostics = {
        "source": "selected prediction artifact (stored arrays)",
        "read": bool(diagnostics_v1.get("read")),
        "readable": diagnostics_v1.get("readable"),
        "predicted_stored": diagnostics_v1.get("predicted"),
        "observed_stored": diagnostics_v1.get("observed"),
        "array_scale": scale.get("stored_arrays"),
        "array_scale_status": scale.get("status", nc.NOT_RECORDED),
        "predicted_native_scale": _range(native_replay.get("prediction_native_scale")) if replay["status"] == nc.VERIFIED else None,
        "observed_native_scale": _range(native_replay.get("target_native_scale")) if replay["status"] == nc.VERIFIED else None,
        "saved_sample_index": (
            {"length": sample_count, "distinct_cells": diagnostics_v1.get("distinct_cells"),
             "status": "RECORDED_IN_ARTIFACT"} if saved_index_present else {"length": None, "status": nc.NOT_RECORDED}
        ),
        "verified_cell_count": identity_check.get("rebuilt_count") if identity_status == nc.MATCH else None,
        "cell_identity_status": "VERIFIED (split and labels rebuilt; labels matched in order)"
        if identity_status == nc.MATCH else nc.NOT_VERIFIED,
        "pairing_status": "VERIFIED (rebuilt labels match stored targets in order)"
        if identity_status == nc.MATCH else nc.NOT_VERIFIED,
        "task_metadata": diagnostics_v1.get("task_metadata") or None,
    }
    if diagnostics_v1.get("read") and not saved_index_present:
        diagnostics["legacy_metadata_counters"] = {
            "samples": sample_count, "distinct_cells": diagnostics_v1.get("distinct_cells"),
            "meaning": "v1 counters for an artifact without a saved sample index; not a count of evaluated cells",
        }

    stored_state = ((report or {}).get("label_transformation") or {}).get("stored_state")
    context: Dict[str, Any] = {
        "context_version": CONTEXT_VERSION,
        "identity": identity,
        "ui_task": identity.get("task"),
        "scientific_target": _scientific_target(config, report),
        "metrics": list(v1.get("metrics") or []),
        "metrics_source": "stored scores in the run manifest, verbatim",
        "protocol": dict(v1.get("protocol") or {}),
        "label_transformation": {
            "configured_order": _configured_order(config),
            "fitted_state": stored_state,
            "fitted_state_status": "READ_FROM_ARTIFACT" if stored_state else "NOT_SUPPLIED",
            "fitted_state_check": ((report or {}).get("fitted_state_check") or {}).get("status", nc.NOT_PERFORMED),
            "metric_inverse_transform_verified": replay["status"] == nc.VERIFIED,
        },
        "metric_replay": replay,
        "diagnostics": diagnostics,
        "benchmark": _benchmark_block(v1.get("benchmark") or {}, identity_status, replay["status"]),
        "evidence": None if report is None else {
            "native_check_report": report.get("_report_path"),
            "native_check_report_sha256": report.get("_report_sha256"),
            "checked_at": report.get("checked_at"),
            "checked_with_commit": (report.get("environment_current") or {}).get("git_head"),
            "predictions_sha256": predictions_sha,
        },
        "native_evidence_status": "NATIVE_CHECK_REPORT_MATCHED" if report is not None else "NO_NATIVE_CHECK_REPORT",
        "warnings": list(v1.get("warnings") or []),
    }
    return context


def build_result_analysis_prompt_v2(context: Dict[str, Any]) -> str:
    """The user message: the context as JSON, then the request."""
    return "\n".join(
        [
            "RESULT CONTEXT (authoritative JSON). A null value with status NOT_RECORDED, NOT_VERIFIED, "
            "NOT_PERFORMED or NOT_SUPPLIED is unknown, not zero.",
            json.dumps(context, ensure_ascii=False, indent=2, default=repr),
            "",
            "Analyse this run for a battery scientist. Do not narrate the configuration back. Answer in order: "
            "summary; interpretation (what the stored metrics mean for the scientific target and its unit); "
            "diagnostics (what in the context warrants attention, citing the field); benchmark (only from the "
            "benchmark section, with its comparability status); next_steps (2 to 4 checks for this run, each "
            "naming what it would resolve).",
            f"Return JSON with exactly these keys: {ANALYSIS_SCHEMA_V2}.",
        ]
    )
