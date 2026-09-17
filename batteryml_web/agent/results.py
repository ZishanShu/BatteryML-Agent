"""Structured result context for completed agent executions.

Deterministic layer: ``workspace -> actual result artifacts ->
structured result context -> agent-safe interpretation input``.

No training, no subprocess, no LLM calls here. Metrics come from exactly
one canonical place — the ``scores`` dict saved inside the predictions
pickle by :meth:`batteryml.pipeline.Pipeline.evaluate`, which itself calls
:meth:`batteryml.data.databundle.DataBundle.evaluate` (inverse label
transform, then ``_evaluate_score``: RMSE = mean-square-root, MAE = mean
absolute, MAPE = mean absolute relative error as a **ratio**, not a
percent). Stdout regex parsing is never a metric source.

A 1-epoch (or any short) smoke run is not a performance benchmark: every
successful context carries an explicit smoke-run warning so the agent can
restate it instead of claiming scientific significance.
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.results_reader import find_result_files

METRIC_KEYS = ("RMSE", "MAE", "MAPE")

SMOKE_RUN_WARNING = (
    "This was a short smoke run (e.g. 1 epoch), not a performance benchmark. "
    "Report the metrics as run output only; do not claim scientific "
    "superiority, convergence, or benchmark performance."
)


def _torch_cpu_load_from_bytes(data: bytes) -> Any:
    """CPU-only replacement for ``torch.storage._load_from_bytes``.

    The saved payload embeds tensor storage via
    ``torch.storage._load_from_bytes(bytes)``, whose implementation calls
    ``torch.load(BytesIO(b))`` with no ``map_location`` — unreachable from
    any outer ``map_location`` argument. Loading those inner bytes here
    with ``map_location="cpu"`` keeps CUDA-origin storage on CPU. The
    ``weights_only`` keyword is passed only when the installed torch
    supports it.
    """
    import io

    import torch

    kwargs: Dict[str, Any] = {}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
    except Exception:
        pass
    return torch.load(io.BytesIO(data), map_location="cpu", **kwargs)


class _CpuSafeUnpickler(pickle.Unpickler):
    """Local unpickler that remaps only torch storage bytes onto CPU.

    Overrides :meth:`find_class` for exactly one symbol —
    ``torch.storage._load_from_bytes`` — and delegates everything else to
    :class:`pickle.Unpickler` unchanged. No ``torch.serialization`` global
    is touched, so concurrent reads are safe and plain CPU payloads load
    through the identical code path.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return _torch_cpu_load_from_bytes
        return super().find_class(module, name)


def _load_prediction_payload(path: Path) -> Optional[Dict[str, Any]]:
    """Load a predictions pickle without requiring a GPU.

    The real smoke artifact stores a CUDA tensor via plain :func:`pickle`
    (``torch.storage._load_from_bytes`` + ``_rebuild_tensor_v2``), not via
    ``torch.save``. The inner storage load carries no ``map_location``,
    so no outer ``torch.load(..., map_location=...)`` form can reach it —
    every form still routes CUDA-tagged storage through the CUDA
    ``_deserialize`` backend, which raises when no GPU is visible.

    The local :class:`_CpuSafeUnpickler` intercepts only the storage
    bytes and reloads them with ``map_location="cpu"``; all other pickle
    behavior is unchanged and no torch global is mutated. The saved
    artifact is never modified.
    """
    try:
        with open(path, "rb") as handle:
            payload = _CpuSafeUnpickler(handle).load()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_prediction_payload(path: Path) -> Optional[Dict[str, Any]]:
    """Public entry point for :func:`_load_prediction_payload`.

    Exists so other modules (the result views) read the saved artifact
    through the same CPU-safe unpickler rather than writing a second one.
    The CUDA-remapping logic above is subtle and must have exactly one
    implementation.
    """
    return _load_prediction_payload(path)


def _canonical_metrics(payload: Dict[str, Any]) -> tuple[Dict[str, float], List[str]]:
    """Extract the canonical ``scores`` stored by ``Pipeline.evaluate``.

    Returns ``(metrics, warnings)``. Anything that is not a finite
    ``int``/``float`` under the exact keys RMSE/MAE/MAPE is reported as
    unavailable (a warning names it) rather than recomputed or guessed.
    """
    import math

    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        warnings.append(
            "No canonical 'scores' dict in the predictions artifact; "
            "metrics are unavailable."
        )
        return metrics, warnings
    for key in METRIC_KEYS:
        value = scores.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            warnings.append(f"Metric {key!r} is unavailable in the saved scores.")
            continue
        number = float(value)
        if not math.isfinite(number):
            warnings.append(f"Metric {key!r} is non-finite in the saved scores.")
            continue
        metrics[key] = number
    return metrics, warnings


AMBIGUOUS_SELECTION_WARNING = (
    "Multiple predictions artifacts exist in the workspace and no single "
    "execution artifact disambiguates them; refusing to choose one. Pass "
    "the execution_result whose artifacts identify the intended file."
)


def _select_prediction_file(
    candidates: List[str], execution_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Choose the predictions artifact deterministically, never by glob order.

    Rules (documented, refusal over guessing):

    * With ``execution_result``: only files listed under its ``artifacts``
      (any ``*_files`` list, e.g. ``prediction_files``) that actually exist
      **and** are among the workspace candidates may be used. Exactly one
      such file is selected; zero yields a warning and no selection (the
      workspace scan below then refuses on ambiguity when needed); more
      than one yields an explicit ambiguity warning and no selection.
    * Without ``execution_result``: exactly one workspace candidate is
      selected; zero or many yields no selection (many: ambiguity warning).

    Returns ``{"path", "warnings", "ambiguous"}``.
    """
    warnings: List[str] = []
    if execution_result is not None:
        artifacts = execution_result.get("artifacts") or {}
        listed: List[str] = []
        for key, values in artifacts.items():
            if not key.endswith("_files") or not isinstance(values, list):
                continue
            listed.extend(str(v) for v in values)
        # Match by basename: execution artifacts may record absolute
        # paths from another checkout while the workspace copy is the same
        # file. A listed artifact matches a candidate with the same final
        # path component; that is content-neutral routing, not provenance.
        by_base = {}
        for candidate in candidates:
            by_base.setdefault(Path(candidate).name, candidate)
        usable = []
        for entry in listed:
            hit = by_base.get(Path(entry).name)
            if hit is not None and Path(hit).is_file():
                usable.append(hit)
        # De-duplicate while preserving execution order.
        seen: set[str] = set()
        ordered = [v for v in usable if not (v in seen or seen.add(v))]
        if len(ordered) == 1:
            return {"path": ordered[0], "warnings": warnings, "ambiguous": False}
        if len(ordered) > 1:
            warnings.append(AMBIGUOUS_SELECTION_WARNING)
            return {"path": None, "warnings": warnings, "ambiguous": True}
        listed_existing = [v for v in listed if Path(v).is_file()]
        candidate_bases = {Path(c).name for c in candidates}
        if listed and not any(
            Path(v).name in candidate_bases for v in listed_existing
        ):
            warnings.append(
                "The execution_result artifacts name no predictions file "
                "present in this workspace; metrics are unavailable."
            )
            return {"path": None, "warnings": warnings, "ambiguous": False}
    if len(candidates) == 1:
        return {"path": candidates[0], "warnings": warnings, "ambiguous": False}
    if len(candidates) > 1:
        warnings.append(AMBIGUOUS_SELECTION_WARNING)
        return {"path": None, "warnings": warnings, "ambiguous": True}
    return {"path": None, "warnings": warnings, "ambiguous": False}


AMBIGUOUS_CONFIG_WARNING = (
    "Multiple workspace config copies exist and no single execution "
    "artifact disambiguates them; provenance from config content is "
    "ambiguous and will not be guessed. Pass the execution_result whose "
    "command or artifacts identify the executed config."
)


def _workspace_config_candidates(resolved: Path) -> List[str]:
    """All workspace config copies (``config_*.yaml``/``*.yml``), sorted."""
    return sorted(
        {
            str(c)
            for c in list(resolved.glob("config_*.yaml"))
            + list(resolved.glob("config_*.yml"))
            if c.is_file()
        }
    )


def _read_config_text(source: str) -> Optional[str]:
    try:
        return Path(source).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _command_config_fingerprint(executed: str) -> tuple[Optional[str], Optional[str]]:
    """Best-effort ``(dataset, model_name)`` for a repo source config path.

    Reads the repo config the execution command names (e.g.
    ``configs/external/cptransformer/matr_1.yaml``) for its declared
    ``cell_data_path`` and ``model.name``. Returns ``(None, None)`` when
    unreadable — the command then simply does not disambiguate.
    """
    from .executor import _repo_root as _agent_repo_root

    candidate = Path(executed)
    if not candidate.is_absolute():
        candidate = _agent_repo_root() / executed
    try:
        import yaml

        parsed = yaml.safe_load(
            candidate.read_text(encoding="utf-8", errors="ignore")
        )
    except Exception:
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    cell = str((parsed.get("train_test_split") or {}).get("cell_data_path", "") or "")
    name = str((parsed.get("model") or {}).get("name", "") or "")
    return (cell or None, name or None)


def _config_copy_signature(source: str) -> tuple[Optional[str], Optional[str]]:
    """``(cell_data_path, model.name)`` declared by one workspace copy."""
    try:
        import yaml

        parsed = yaml.safe_load(
            Path(source).read_text(encoding="utf-8", errors="ignore")
        )
    except Exception:
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    cell = str((parsed.get("train_test_split") or {}).get("cell_data_path", "") or "")
    name = str((parsed.get("model") or {}).get("name", "") or "")
    return (cell or None, name or None)


def _match_command_config(
    candidates: List[str], executed: str
) -> Optional[Dict[str, Any]]:
    """Use the execution command's config as content evidence.

    Returns a ``{"path", ...}`` selection when exactly one candidate's
    content signature matches the repo source config the command names,
    else None (no disambiguation). Never compares bare filenames.
    """
    fingerprint = _command_config_fingerprint(executed)
    if fingerprint == (None, None):
        return None
    matches = [
        c for c in candidates if _config_copy_signature(c) == fingerprint
    ]
    if len(matches) == 1:
        return {"path": matches[0], "warnings": [], "ambiguous": False}
    return None


def _select_config_source(
    candidates: List[str], execution_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Select the workspace config copy by execution evidence, never by time.

    Priority (refusal over guessing, mirroring prediction selection):

    1. ``execution_result`` ``command`` containing ``run <config>`` — the
       executed config path itself. When it names a file inside the
       workspace that exists, that copy is the evidence.
    2. ``execution_result.artifacts.config_files`` with exactly one
       existing candidate is used; several is ambiguous.
    3. Without ``execution_result``: exactly one workspace candidate may
       be used; zero or several yields no selection (several: ambiguity).

    Timestamp ordering and bare filenames are never consulted. Returns
    ``{"path", "warnings", "ambiguous"}``.
    """
    warnings: List[str] = []
    if execution_result is not None:
        command = execution_result.get("command")
        if isinstance(command, list):
            parts = [str(c) for c in command]
            if "run" in parts and parts.index("run") + 1 < len(parts):
                executed = parts[parts.index("run") + 1]
                # The command's config may be the repo-level source config
                # while the workspace holds the executed copy. A command
                # mentioning a registered source config disambiguates only
                # when exactly one candidate carries matching content.
                match = _match_command_config(candidates, executed)
                if match is not None:
                    return match
        artifacts = execution_result.get("artifacts") or {}
        listed: List[str] = []
        for values in [artifacts.get("config_files")]:
            if isinstance(values, list):
                listed.extend(str(v) for v in values)
        by_base = {}
        for candidate in candidates:
            by_base.setdefault(Path(candidate).name, candidate)
        usable = []
        for entry in listed:
            hit = by_base.get(Path(entry).name)
            if hit is not None and Path(hit).is_file():
                usable.append(hit)
        seen: set[str] = set()
        ordered = [v for v in usable if not (v in seen or seen.add(v))]
        if len(ordered) == 1:
            return {"path": ordered[0], "warnings": warnings, "ambiguous": False}
        if len(ordered) > 1:
            warnings.append(AMBIGUOUS_CONFIG_WARNING)
            return {"path": None, "warnings": warnings, "ambiguous": True}
    if len(candidates) == 1:
        return {"path": candidates[0], "warnings": warnings, "ambiguous": False}
    if len(candidates) > 1:
        warnings.append(AMBIGUOUS_CONFIG_WARNING)
        return {"path": None, "warnings": warnings, "ambiguous": True}
    return {"path": None, "warnings": warnings, "ambiguous": False}


def _verify_provenance(
    *,
    dataset: Optional[str],
    model: Optional[str],
    config_path: Optional[str],
    execution_result: Optional[Dict[str, Any]],
    resolved: Path,
    root: Path,
) -> Dict[str, Any]:
    """Cross-check caller provenance against deterministic workspace evidence.

    Evidence used (never filenames or timestamps alone):

    * the execution plan config — ``execution_result["command"]``'s
      ``run <config>`` argument or ``execution_result["config_path"]``,
      which must equal the caller's ``config_path``;
    * the workspace config copy (``config_*.yaml``) selected by
      :func:`_select_config_source`: execution ``command`` evidence first,
      then exactly one ``artifacts.config_files`` entry, then exactly one
      workspace candidate without ``execution_result``. Multiple
      candidates with no disambiguating evidence are ambiguous — never
      the "newest". Content checks (``cell_data_path`` mentions the
      dataset; ``model.name`` matches via registry alias tables) apply to
      the selected copy only.

    A direct contradiction fails verification; ambiguous or missing
    evidence leaves claims unverified (warning), never fabricated.
    Returns ``{"provenance_verified", "provenance_warnings"}``.
    """
    from ..core.config_registry import (
        MODEL_CLASS_KEYS,
        MODEL_DISPLAY_NAMES,
        MODEL_KEY_ALIASES,
        normalize_model_token,
    )

    warnings: List[str] = []
    failures: List[str] = []
    checks = 0

    early_selection = _select_config_source(
        _workspace_config_candidates(resolved), execution_result
    )
    warnings.extend(early_selection["warnings"])
    config_source = early_selection["path"]
    config_ambiguous = bool(early_selection.get("ambiguous"))
    config_text = _read_config_text(config_source) if config_source else None

    plan_config: Optional[str] = None
    if execution_result is not None:
        command = execution_result.get("command")
        if isinstance(command, list) and "run" in [str(c) for c in command]:
            parts = [str(c) for c in command]
            plan_config = parts[parts.index("run") + 1]
        elif isinstance(execution_result.get("config_path"), str):
            plan_config = execution_result["config_path"]
    if config_path is not None and plan_config is not None:
        if plan_config != config_path:
            checks += 1
            failures.append(
                f"Caller config_path {config_path!r} differs from the "
                f"execution plan config {plan_config!r}."
            )
        elif config_source is not None:
            # The command names the same source config AND the workspace
            # holds the executed copy selected by evidence: real check.
            checks += 1

    if config_ambiguous:
        pass
    elif config_text is None:
        if dataset is not None or model is not None or config_path is not None:
            warnings.append(
                "No workspace config copy (config_*.yaml) to verify "
                "dataset/model provenance against."
            )
    else:
        import yaml

        try:
            parsed = yaml.safe_load(config_text)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            warnings.append(
                f"Workspace config copy {config_source} could not be parsed; "
                "provenance unverified."
            )
        else:
            if dataset is not None:
                checks += 1
                cell_path = str(
                    (parsed.get("train_test_split") or {}).get("cell_data_path", "")
                )
                if dataset.lower() not in cell_path.lower():
                    failures.append(
                        f"Caller dataset {dataset!r} is not the dataset in "
                        f"workspace config {config_source} "
                        f"(cell_data_path={cell_path!r})."
                    )
            if model is not None:
                checks += 1
                declared = str((parsed.get("model") or {}).get("name", ""))
                normalized = normalize_model_token(model)
                canonical = MODEL_KEY_ALIASES.get(
                    normalized,
                    MODEL_CLASS_KEYS.get(normalized.replace(" ", ""), normalized),
                )
                if isinstance(canonical, str) and canonical in MODEL_DISPLAY_NAMES:
                    expected_class = {
                        v: k
                        for k, v in MODEL_CLASS_KEYS.items()
                    }.get(canonical)
                    aliases = {canonical, canonical.replace("_", " ")}
                    aliases.update(
                        a for a, t in MODEL_KEY_ALIASES.items() if t == canonical
                    )
                    aliases.update(
                        c for c, t in MODEL_CLASS_KEYS.items() if t == canonical
                    )
                    ok = declared.lower() in {a.lower() for a in aliases} or (
                        expected_class is not None
                        and declared.lower() == expected_class.lower()
                    )
                else:
                    ok = declared.lower() == normalized
                if not ok:
                    failures.append(
                        f"Caller model {model!r} does not match the model in "
                        f"workspace config {config_source} "
                        f"(model.name={declared!r})."
                    )
            _ = root  # workspace containment is the executor's rule, not re-checked here.

    if failures:
        return {
            "provenance_verified": False,
            "provenance_warnings": failures,
        }
    if checks == 0:
        warnings.append(
            "No deterministic evidence available to verify dataset/model "
            "provenance; claims are unverified."
        )
        return {"provenance_verified": False, "provenance_warnings": warnings}
    return {"provenance_verified": True, "provenance_warnings": warnings}


def _prediction_count(payload: Dict[str, Any]) -> Optional[int]:
    prediction = payload.get("prediction")
    if prediction is None:
        return None
    try:
        return int(len(prediction))
    except Exception:
        return None


def build_result_context(
    workspace: str | Path,
    *,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    run_status: str = "SUCCESS",
    execution_result: Optional[Dict[str, Any]] = None,
    smoke_run: bool = False,
) -> Dict[str, Any]:
    """Build the structured result context for one execution workspace.

    Args:
        workspace: run workspace directory (relative to the repo root or
            absolute).
        dataset / task / model / config_path: provenance supplied by the
            caller (execution plan); echoed, never inferred from artifacts.
        run_status: ``"SUCCESS"`` only when the executor reported success;
            anything else (``"FAILED"`` or unknown) yields metrics-free
            output with ``success`` False — a failed run is never reported
            as successful.
        execution_result: optional :func:`executor.execute_plan` result;
            its ``success``/``returncode`` gate the status honestly.
        smoke_run: when True (or when the workspace name suggests a smoke
            run), the smoke-run warning is included.

    Returns a dict with ``workspace``, ``run_status``, ``success``,
    ``dataset``, ``task``, ``model``, ``config_path``, ``metrics`` (only
    canonical saved values), ``num_predictions``, ``predictions_file``,
    ``figure_files``, ``checkpoint_files``, ``log_files``,
    ``config_files``, ``warnings``, ``provenance_verified`` and
    ``provenance_warnings``. Only paths that really exist appear.
    Provenance is cross-checked against the execution plan and the
    workspace config copy (never filenames alone); a contradiction fails
    verification and the result must not be presented as a clean run for
    the claimed dataset/model.
    """
    from .executor import _repo_root as _agent_repo_root

    root = _agent_repo_root()
    resolved = (
        Path(workspace)
        if Path(workspace).is_absolute()
        else (root / workspace)
    )
    warnings: List[str] = []
    artifacts = {
        "metrics_files": [],
        "prediction_files": [],
        "log_files": [],
        "report_files": [],
        "figure_files": [],
        "checkpoint_files": [],
        "config_files": [],
    }

    executor_success: Optional[bool] = None
    executor_returncode: Any = None
    if execution_result is not None:
        executor_success = execution_result.get("success")
        executor_returncode = execution_result.get("returncode")

    success = run_status == "SUCCESS" and executor_success is not False
    if execution_result is not None and executor_success is not True:
        success = False
    status = "SUCCESS" if success else "FAILED"

    if status != "SUCCESS":
        warnings.append(
            "The run did not succeed; no metrics are reported and no "
            "interpretation of metric values is permitted."
        )
        return {
            "workspace": str(workspace),
            "run_status": status,
            "success": False,
            "dataset": dataset,
            "task": task,
            "model": model,
            "config_path": config_path,
            "metrics": {},
            "num_predictions": None,
            "predictions_file": None,
            "figure_files": [],
            "checkpoint_files": [],
            "log_files": [],
            "config_files": [],
            "executor_returncode": executor_returncode,
            "warnings": warnings,
            "provenance_verified": False,
            "provenance_warnings": ["Run failed; provenance is not verified."],
        }

    if not resolved.is_dir():
        warnings.append(f"Workspace does not exist: {resolved}.")
        return {
            "workspace": str(workspace),
            "run_status": "FAILED",
            "success": False,
            "dataset": dataset,
            "task": task,
            "model": model,
            "config_path": config_path,
            "metrics": {},
            "num_predictions": None,
            "predictions_file": None,
            "figure_files": [],
            "checkpoint_files": [],
            "log_files": [],
            "config_files": [],
            "executor_returncode": executor_returncode,
            "warnings": warnings,
            "provenance_verified": False,
            "provenance_warnings": ["Workspace is missing; provenance unverifiable."],
        }

    found = find_result_files(resolved, resolved)
    candidates = sorted(
        {str(p) for p in found.get("prediction_files", []) if p.is_file()}
    )
    selection = _select_prediction_file(candidates, execution_result)
    predictions_file = selection["path"]
    warnings.extend(selection["warnings"])
    ambiguous = bool(selection.get("ambiguous"))
    payload: Optional[Dict[str, Any]] = None
    if ambiguous:
        return {
            "workspace": str(workspace),
            "run_status": "FAILED",
            "success": False,
            "dataset": dataset,
            "task": task,
            "model": model,
            "config_path": config_path,
            "metrics": {},
            "num_predictions": None,
            "predictions_file": None,
            "figure_files": [],
            "checkpoint_files": [],
            "log_files": [],
            "config_files": [],
            "executor_returncode": executor_returncode,
            "warnings": warnings,
            "provenance_verified": False,
            "provenance_warnings": [
                "Prediction-file selection is ambiguous; no artifact chosen."
            ],
        }
    if predictions_file is None:
        if not candidates:
            warnings.append(
                "No predictions artifact (*prediction*.pkl/csv) in the workspace; "
                "metrics are unavailable and must not be fabricated."
            )
    else:
        payload = _load_prediction_payload(Path(predictions_file))
        if payload is None:
            warnings.append(
                f"Predictions artifact {predictions_file} could not be read; "
                "metrics are unavailable."
            )

    metrics: Dict[str, float] = {}
    num_predictions: Optional[int] = None
    if payload is not None:
        metrics, metric_warnings = _canonical_metrics(payload)
        warnings.extend(metric_warnings)
        num_predictions = _prediction_count(payload)

    config_files = sorted(
        [p for p in resolved.rglob("*.yaml") if p.is_file()]
        + [p for p in resolved.rglob("*.yml") if p.is_file()]
    )

    provenance = _verify_provenance(
        dataset=dataset,
        model=model,
        config_path=config_path,
        execution_result=execution_result,
        resolved=resolved,
        root=root,
    )
    provenance_warnings: List[str] = list(provenance["provenance_warnings"])

    smoke = bool(smoke_run) or "smoke" in resolved.name.lower()
    if smoke:
        warnings.append(SMOKE_RUN_WARNING)

    claimed = dataset is not None or model is not None or config_path is not None
    verified = bool(provenance["provenance_verified"])
    contradictions = [
        w for w in provenance["provenance_warnings"]
        if "differs from the execution plan config" in w
        or "is not the dataset in workspace config" in w
        or "does not match the model in workspace config" in w
    ]
    if claimed and (contradictions or not verified):
        warnings.extend(provenance_warnings)
        if contradictions:
            warnings.append(
                "Provenance evidence contradicts the claimed "
                "dataset/model/config; this must not be presented as a run "
                "for the claimed provenance."
            )
            status = "FAILED"
            succeeded = False
        else:
            warnings.append(
                "Provenance could not be verified against workspace "
                "evidence; metrics are reported as read but this must not "
                "be presented as a clean verified run for the claimed "
                "dataset/model until the ambiguity is resolved."
            )
            status = "SUCCESS"
            succeeded = True
    else:
        warnings.extend(provenance_warnings)
        status = "SUCCESS"
        succeeded = True

    return {
        "workspace": str(workspace),
        "run_status": status,
        "success": succeeded,
        "dataset": dataset,
        "task": task,
        "model": model,
        "config_path": config_path,
        "metrics": metrics,
        "num_predictions": num_predictions,
        "predictions_file": predictions_file,
        "figure_files": [str(p) for p in found.get("figure_files", []) if p.is_file()],
        "checkpoint_files": [str(p) for p in found.get("checkpoint_files", []) if p.is_file()],
        "log_files": [str(p) for p in found.get("log_files", []) if p.is_file()],
        "config_files": [str(p) for p in config_files],
        "executor_returncode": executor_returncode,
        "warnings": warnings,
        "provenance_verified": verified,
        "provenance_warnings": provenance["provenance_warnings"],
    }


def build_result_interpretation_prompt(
    result_context: Dict[str, Any],
) -> str:
    """Assemble the user prompt that grounds result explanation in facts.

    The prompt carries the structured context (metrics verbatim, MAPE as
    the stored ratio plus its percent reading) and forbids inventing any
    value not present. A missing metric is stated as unavailable, never
    filled in.
    """
    lines: List[str] = ["Structured run result (authoritative, do not contradict):"]
    for key in ("dataset", "task", "model", "config_path", "workspace", "run_status"):
        lines.append(f"{key}: {result_context.get(key)}")
    metrics = result_context.get("metrics") or {}
    if metrics:
        rendered = []
        for key in METRIC_KEYS:
            if key in metrics:
                rendered.append(f"{key}={metrics[key]!r}")
        lines.append("metrics: " + "; ".join(rendered))
        if "MAPE" in metrics and isinstance(metrics["MAPE"], (int, float)):
            lines.append(
                f"note: MAPE is stored as a ratio ({metrics['MAPE']!r}, "
                f"i.e. approximately {float(metrics['MAPE']) * 100:.2f}% mean "
                "absolute percentage error). Preserve the raw numeric value."
            )
    else:
        lines.append("metrics: unavailable (no canonical saved metrics).")
    lines.append(f"predictions_file: {result_context.get('predictions_file')}")
    lines.append(f"num_predictions: {result_context.get('num_predictions')}")
    for key in ("figure_files", "checkpoint_files", "log_files", "config_files"):
        values = result_context.get(key) or []
        lines.append(f"{key}: {', '.join(str(v) for v in values) or '(none)'}")
    warnings = result_context.get("warnings") or []
    lines.append("warnings:")
    lines.extend(f"- {warning}" for warning in warnings) or lines.append("- (none)")
    lines.append(
        "Respond with run interpretation only: state which model/dataset/task "
        "ran, whether execution succeeded, restate the metrics above exactly "
        "(never recompute, round, or invent a value), list the generated "
        "artifacts, and restate the smoke-run warning when present. If metrics "
        "are unavailable, say so explicitly and stop."
    )
    return "\n".join(lines)
