"""Figure export for one exact prediction artifact: panels, plotted data, provenance.

Visualization only. Nothing here trains, evaluates, downloads, or looks for a
different run: it reads the one artifact it is given through the platform's
CPU-safe loader (:func:`batteryml_web.agent.results.load_prediction_payload`),
builds the same view the website builds, draws the same panels
(:func:`batteryml_web.viz.result_figures.build_panels`), and returns every file
as bytes. ``scripts/export_result_figures.py`` writes them to a new directory;
the page offers the same bytes as a zip. One code path, so the CSV is what was
drawn.

Files, where they apply:

=================================  ============================================
``soh_cycle_<cell>.svg/.png``      SOH estimates across observed cycles, one cell
``soh_cycle_<cell>_zoom.svg/.png`` the same, zoomed viewport
``soh_cycle_<cell>.csv``           that cell's plotted series, sorted by cycle
``parity.svg/.png``                observed vs predicted, full range
``parity_zoom.svg/.png``           the same, zoomed viewport
``residuals.svg/.png``             predicted − observed against observed
``residuals_zoom.svg/.png``        the same, zoomed viewport
``plotted_samples.csv``            every plotted pair, with artifact row and identity
``per_cell_errors.csv``            per-cell display diagnostic (cycle-level SOH)
``figure_sources.json``            provenance, units, counts, limits, saved metrics
``figure_notes.md``                captions and method notes
=================================  ============================================
"""

from __future__ import annotations

import csv
import io
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

SMOKE_BADGE = "smoke run · not benchmark performance"


class ExportError(RuntimeError):
    """The artifact cannot be exported; the message says why and what to do."""


@dataclass
class ExportResult:
    files: Dict[str, bytes] = field(default_factory=dict)
    sources: Dict[str, Any] = field(default_factory=dict)


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return cleaned or "cell"


def _csv_bytes(rows: List[Dict[str, Any]], columns: List[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        # repr() of a float is its shortest exact round-trip form: the CSV
        # holds the plotted number, not a rounded display of it.
        writer.writerow({
            key: (repr(value) if isinstance(value, float) else ("" if value is None else value))
            for key, value in row.items()
        })
    return buffer.getvalue().encode("utf-8")


def _small_value(value: Any) -> Any:
    """A JSON-able form of a transformation parameter, or None if not small."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    if hasattr(value, "detach") and hasattr(value, "numel"):
        try:
            if int(value.numel()) <= 16:
                return value.detach().cpu().reshape(-1).tolist()
        except Exception:  # noqa: BLE001 - recorded as unavailable below
            return None
    return None


def transformation_parameters(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The recorded label transformation(s) and their fitted parameters.

    Read from the transformation object saved in the artifact — fitted on the
    training labels by ``DataBundle`` — never re-estimated from test data.
    """
    transformation = getattr(payload.get("data"), "label_transformation", None)
    if transformation is None:
        return []
    members = getattr(transformation, "transformations", None)
    chain = list(members) if isinstance(members, (list, tuple)) and members else [transformation]
    described = []
    for member in chain:
        params: Dict[str, Any] = {}
        for name, value in sorted(vars(member).items()):
            if callable(value) and not hasattr(value, "numel"):
                continue
            small = _small_value(value)
            if small is not None:
                params[name.lstrip("_")] = small
        described.append({"class": type(member).__name__, "parameters": params})
    return described


def _git_state() -> Dict[str, str]:
    from .render import git_commit

    commit = git_commit(REPO_ROOT)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10, check=False,
        )
        dirty = "yes" if status.returncode == 0 and status.stdout.strip() else (
            "no" if status.returncode == 0 else "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        dirty = "unknown"
    return {"commit": commit, "uncommitted_tracked_changes": dirty}


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for module in ("matplotlib", "numpy", "torch"):
        try:
            out[module] = str(__import__(module).__version__)
        except Exception:  # noqa: BLE001 - version is provenance, not a requirement
            out[module] = "not importable"
    return out


def load_views_for_export(artifact: Path) -> Dict[str, Any]:
    """Payload, schema and the one view the export draws, or ExportError."""
    from ..agent.result_views import (
        SCALAR_SCHEMAS, SOH_SCHEMA, attach_artifact_context, build_rul_view,
        build_soh_view, detect_schema,
    )
    from ..agent.results import load_prediction_payload

    payload = load_prediction_payload(artifact)
    if payload is None:
        raise ExportError(
            f"{artifact} could not be read as a BatteryML prediction artifact "
            "by the platform's CPU-safe loader. Check the path."
        )
    schema = detect_schema(payload)
    if schema == SOH_SCHEMA:
        view = build_soh_view(payload)
    elif schema in SCALAR_SCHEMAS:
        view = build_rul_view(payload)
    else:
        raise ExportError(
            f"{artifact} holds a '{schema}' result. Only scalar-per-sample "
            "results (lifetime/RUL, SOH) have parity, residual and SOH-cycle "
            "panels; nothing is drawn for this shape."
        )
    if not view.get("available"):
        raise ExportError(f"{artifact}: {view.get('reason')}")
    attach_artifact_context(view, payload, artifact)
    return {"payload": payload, "schema": schema, "view": view}


def build_export(
    artifact: str | Path,
    *,
    cell_id: Optional[str] = None,
    dpi: float = 300.0,
    command: Optional[List[str]] = None,
    invocation: Optional[Dict[str, Any]] = None,
) -> ExportResult:
    """Every export file for ``artifact``, as bytes, plus the provenance dict.

    ``command`` is the literal argv of a CLI export, recorded as given.
    ``invocation`` is a structured record of an export that ran no command
    (the website's button); it is serialized as is and never turned into a
    command line.

    Raises:
        ExportError: unreadable or unsupported artifact, unknown ``cell_id``.
    """
    from ..agent.artifact_provenance import artifact_provenance, smoke_evidence
    from ..agent.result_views import TRANSFORM_INVERTED
    from .render import render_to_bytes
    from .result_figures import (
        DEFAULT_CELL_RULE, EXPLICIT_CELL_RULE, PER_CELL_NOTE, ZOOM_RULE,
        FigureDataError, build_panels, cell_choices, per_cell_errors,
        plotted_rows, residual_zoom, zoom_window,
    )

    path = Path(artifact)
    try:
        provenance = artifact_provenance(path)
    except FileNotFoundError as exc:
        raise ExportError(str(exc)) from exc
    loaded = load_views_for_export(path)
    payload, schema, view = loaded["payload"], loaded["schema"], loaded["view"]
    target = view.get("target") or {}
    soh_cycle = target.get("kind") == "soh" and bool(view.get("cycle_indices"))

    if cell_id is not None and soh_cycle and str(cell_id) not in cell_choices(view):
        raise ExportError(
            f"cell {cell_id!r} is not among the {len(cell_choices(view))} cells "
            f"plotted from {path}. List them with --list-cells."
        )
    if cell_id is not None and not soh_cycle:
        raise ExportError(
            "--cell-id applies only to a cycle-level SOH result; this one is "
            f"'{schema}' ({target.get('granularity')})."
        )

    badge = SMOKE_BADGE if provenance.get("smoke") is True else None
    saved = view.get("saved_metrics") or {}
    try:
        panels = build_panels(view, cell_id=cell_id, saved_metrics=saved,
                              badge=badge, include_zoom=True, dpi=dpi)
    except FigureDataError as exc:
        raise ExportError(str(exc)) from exc

    result = ExportResult()
    panel_records: Dict[str, Any] = {}
    cell_stub = None
    if "soh_cycle" in panels:
        cell_stub = f"soh_cycle_{_safe_name(panels['soh_cycle']['meta']['cell_id'])}"
    stems = {
        "soh_cycle": cell_stub, "soh_cycle_zoom": f"{cell_stub}_zoom" if cell_stub else None,
        "parity": "parity", "parity_zoom": "parity_zoom",
        "residuals": "residuals", "residuals_zoom": "residuals_zoom",
    }
    for panel_id, panel in panels.items():
        stem = stems[panel_id]
        files = []
        for fmt in ("svg", "png"):
            name = f"{stem}.{fmt}"
            result.files[name] = render_to_bytes(panel["figure"], fmt=fmt, dpi=dpi)
            files.append(name)
        panel_records[panel_id] = {"files": files, **panel["meta"]}

    # The exact arrays the parity/residual builders received, one row each.
    rows = plotted_rows(view)
    pz = panels["parity_zoom"]["meta"]["limits"]
    rz = panels["residuals_zoom"]["meta"]
    for row in rows:
        o, p, r = row["observed"], row["predicted"], row["residual_predicted_minus_observed"]
        row["in_parity_zoom_viewport"] = int(pz[0] <= o <= pz[1] and pz[0] <= p <= pz[1])
        row["in_residual_zoom_viewport"] = int(
            rz["xlim"][0] <= o <= rz["xlim"][1] and rz["ylim"][0] <= r <= rz["ylim"][1]
        )
    result.files["plotted_samples.csv"] = _csv_bytes(rows, [
        "artifact_row", "cell_id", "cycle_index", "observed", "predicted",
        "residual_predicted_minus_observed", "in_parity_zoom_viewport",
        "in_residual_zoom_viewport",
    ])

    selected = None
    if cell_stub:
        series = panels["soh_cycle"]["data"]
        result.files[f"{cell_stub}.csv"] = _csv_bytes(
            [
                {"artifact_row": series["artifact_rows"][i], "cell_id": series["cell_id"],
                 "cycle_index": series["cycles"][i], "observed": series["observed"][i],
                 "predicted": series["predicted"][i],
                 "residual_predicted_minus_observed": series["residuals"][i]}
                for i in range(len(series["cycles"]))
            ],
            ["artifact_row", "cell_id", "cycle_index", "observed", "predicted",
             "residual_predicted_minus_observed"],
        )
        selected = {
            "cell_id": series["cell_id"],
            "selection_rule": EXPLICIT_CELL_RULE if cell_id is not None else DEFAULT_CELL_RULE,
            "points": len(series["cycles"]),
            "duplicate_cell_cycle_records": series["duplicates"],
            "available_cells": len(cell_choices(view)),
        }

    per_cell = None
    if soh_cycle:
        per_cell_rows = per_cell_errors(view)
        result.files["per_cell_errors.csv"] = _csv_bytes(
            per_cell_rows, ["cell_id", "samples", "mean_abs_residual", "rms_residual"]
        )
        per_cell = {"file": "per_cell_errors.csv", "cells": len(per_cell_rows),
                    "note": PER_CELL_NOTE}

    sources = {
        "schema_version": 1,
        "source": {**{k: v for k, v in provenance.items() if k != "notes"},
                   "seed_recorded_in_artifact": payload.get("seed"),
                   "smoke_evidence": smoke_evidence(provenance)},
        "result_schema": schema,
        "target": target,
        "transformation": {
            "status": view.get("transform_status"),
            "inverse_applied_once": view.get("transform_status") == TRANSFORM_INVERTED,
            "chain_recorded_in_artifact": view.get("transform_chain") or [],
            "chain_in_config_copy": provenance.get("label_transformation_config") or [],
            "parameters": transformation_parameters(payload),
        },
        "units": {
            "canonical": target.get("unit"), "display": target.get("unit"),
            "residual": target.get("residual_unit"),
            "residual_note": target.get("residual_note") or "",
            "rescaled_for_display": False,
        },
        "counts": {
            "stored_rows": view.get("total_rows"),
            "plotted_rows": view.get("count"),
            "omitted_nonfinite_rows": view.get("omitted_nonfinite"),
            "cells_plotted": len(cell_choices(view)) or None,
        },
        "selected_cell": selected,
        "panels": panel_records,
        "zoom_policy": ZOOM_RULE,
        "parity_zoom_outside": zoom_window(view["observed"], view["predicted"])["outside"],
        "residual_zoom_outside": residual_zoom(view["observed"], view["residuals"])["outside"],
        "saved_metrics": {
            "values": saved,
            "source": ("payload['scores'] written by BatteryML Pipeline.evaluate "
                       "(DataBundle.evaluate inverse-transforms, then scores all "
                       "test rows). Shown verbatim; not recomputed here."),
        },
        "per_cell_diagnostic": per_cell,
        # How the artifact was produced: only what run.json beside it
        # records, never a command rebuilt from defaults.
        "execution_provenance": {
            "source": "run.json" if provenance.get("command") not in (None, "not recorded") else "not recorded",
            "command": (provenance.get("command")
                        if provenance.get("command") not in (None, "not recorded") else None),
        },
        "export": {"command": command or [],
                   "invocation": (dict(invocation) if invocation is not None
                                  else {"interface": "cli", "argv": list(command)} if command
                                  else {"interface": "not recorded"}),
                   "code": _git_state(),
                   "versions": _versions(), "dpi": dpi,
                   "figure_style": "batteryml_web.viz light theme, DejaVu Sans"},
        "notes": list(provenance.get("notes") or []),
    }
    result.sources = sources
    result.files["figure_sources.json"] = (
        json.dumps(sources, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    ).encode("utf-8")
    result.files["figure_notes.md"] = _notes(sources, panel_records).encode("utf-8")
    for panel in panels.values():
        panel["figure"].clear()
    return result


def _notes(sources: Dict[str, Any], panels: Dict[str, Any]) -> str:
    src, target, counts = sources["source"], sources["target"], sources["counts"]
    unit = target.get("unit")
    lines = [
        "# Figure notes",
        "",
        f"Source artifact: `{src['artifact_path']}`  ",
        f"SHA-256: `{src['artifact_sha256']}`  ",
        f"Run: {src['run_id']} · model {src['model_display'] if src['model_display'] != 'not recorded' else src['model_class']}"
        f" · split {src['split']} · seed {src['seed'] if src['seed'] != 'not recorded' else src['seed_recorded_in_artifact']}  ",
        f"Smoke / training evidence: {src['smoke_evidence']}",
        "",
        f"Target: **{target.get('quantity')}** in **{unit}** ({target.get('granularity')}). "
        f"{target.get('definition') or ''}",
        "",
        f"Plotted {counts['plotted_rows']} of {counts['stored_rows']} stored rows"
        + (f"; {counts['omitted_nonfinite_rows']} omitted as non-finite (one shared mask)."
           if counts["omitted_nonfinite_rows"] else "; none omitted."),
        "",
        f"Label transformation: {sources['transformation']['status']} "
        f"({', '.join(sources['transformation']['chain_recorded_in_artifact']) or 'none recorded'}); "
        "the inverse was applied exactly once, as BatteryML's evaluator does before scoring.",
        "",
        "## Panels",
        "",
    ]
    captions = {
        "soh_cycle": ("SOH estimates across observed cycles for one cell: observed (solid) and "
                      "estimated (dashed) SOH at each recorded cycle index, in recorded order. "
                      "Not a forecast; nothing smoothed or interpolated."),
        "soh_cycle_zoom": "The same cell, zoomed viewport.",
        "parity": ("Observed vs predicted, full range, shared equal-scale axes, y = x reference. "
                   "No fitted line."),
        "parity_zoom": "The same points, zoomed viewport.",
        "residuals": "Residual = predicted − observed against the observed value; zero reference line.",
        "residuals_zoom": "The same points, zoomed viewport.",
    }
    for panel_id, record in panels.items():
        extra = ""
        if record.get("view") == "zoomed":
            extra = f" {record['outside']} plotted point(s) lie outside this viewport."
        if record.get("cell_id"):
            extra += f" Cell {record['cell_id']} ({record['cell_rule']})."
        lines.append(f"- `{record['files'][0]}` / `.png`: {captions[panel_id]}{extra}")
    lines += [
        "",
        f"Zoom policy: {sources['zoom_policy']}.",
        "",
        "Saved metrics (verbatim): "
        + (", ".join(f"{k} = {v!r}" for k, v in sorted(sources['saved_metrics']['values'].items()))
           or "none recorded")
        + f". {sources['saved_metrics']['source']}",
    ]
    if target.get("residual_note"):
        lines += ["", target["residual_note"]]
    if sources.get("per_cell_diagnostic"):
        lines += ["", f"`per_cell_errors.csv`: {sources['per_cell_diagnostic']['note']}"]
    if sources.get("notes"):
        lines += ["", "## Provenance notes", ""] + [f"- {note}" for note in sources["notes"]]
    execution = sources.get("execution_provenance") or {}
    if execution.get("command"):
        recorded = execution["command"]
        recorded = " ".join(recorded) if isinstance(recorded, list) else str(recorded)
        lines += ["", "Command that produced the artifact (recorded in run.json):", "",
                  "```bash", recorded, "```"]
    else:
        lines += ["", "Command that produced the artifact: not recorded (no run.json "
                  "command beside it)."]
    export = sources["export"]
    if export.get("command"):
        lines += ["", "Export command (as run):", "", "```bash",
                  " ".join(export["command"]), "```", ""]
    else:
        lines += ["", "Export invocation (structured record; no export command was run):",
                  "", "```json",
                  json.dumps(export.get("invocation") or {}, indent=2, sort_keys=True,
                             ensure_ascii=False, default=str),
                  "```", ""]
    return "\n".join(lines)


def write_export(result: ExportResult, output: Path) -> List[Path]:
    """Write every file into ``output``, which must be new or empty.

    Raises:
        ExportError: ``output`` exists and is not empty. An export never
            overwrites earlier files, and never writes into a run directory.
    """
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ExportError(
            f"{output} already exists and is not empty. Choose a new export "
            "directory; existing files are never overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name in sorted(result.files):
        target = output / name
        target.write_bytes(result.files[name])
        written.append(target)
    return written


def zip_bytes(result: ExportResult) -> bytes:
    """The same files as one zip, for the page's download button."""
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(result.files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, result.files[name])
    return buffer.getvalue()


def export_command(artifact: str | Path, output: str = "runs/figure_exports/<NEW_DIRECTORY>",
                   cell_id: Optional[str] = None) -> str:
    """The exact shell command that regenerates an export of ``artifact``."""
    parts = [
        "python -m scripts.export_result_figures",
        f"--artifact {_quote(str(artifact))}",
        f"--output {output}",
    ]
    if cell_id is not None:
        parts.append(f"--cell-id {_quote(str(cell_id))}")
    return " \\\n  ".join(parts)


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def inspect_artifact(artifact: str | Path) -> Dict[str, Any]:
    """Read-only summary for choosing what to export. Writes nothing."""
    from ..agent.artifact_provenance import artifact_provenance, smoke_evidence
    from .result_figures import cell_choices

    path = Path(artifact)
    provenance = artifact_provenance(path)
    loaded = load_views_for_export(path)
    view = loaded["view"]
    cells = cell_choices(view)
    return {
        "artifact": str(path),
        "sha256": provenance["artifact_sha256"],
        "result_schema": loaded["schema"],
        "target": view.get("target"),
        "stored_rows": view.get("total_rows"),
        "plotted_rows": view.get("count"),
        "omitted_nonfinite_rows": view.get("omitted_nonfinite"),
        "cells": len(cells),
        "first_cells_sorted": cells[:10],
        "transform_status": view.get("transform_status"),
        "transform_chain": view.get("transform_chain"),
        "saved_metrics": view.get("saved_metrics"),
        "run_id": provenance["run_id"],
        "model": provenance["model_display"] if provenance["model_display"] != "not recorded" else provenance["model_class"],
        "split": provenance["split"],
        "seed": provenance["seed"],
        "seed_recorded_in_artifact": loaded["payload"].get("seed"),
        "smoke_evidence": smoke_evidence(provenance),
        "config_copy": provenance["config_copy"],
        "config_copy_epochs": provenance["config_copy_epochs"],
        "notes": provenance["notes"],
        "python": sys.version.split()[0],
    }
