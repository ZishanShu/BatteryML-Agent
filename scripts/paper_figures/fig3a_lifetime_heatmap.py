"""Fig. 3a — battery-lifetime error heatmap: models x dataset/protocol.

Rows are the models in the selection's ``model_order``; columns are dataset +
split protocol exactly as recorded (MATR primary, secondary and CLO stay
separate columns). A cell is the BatteryML evaluator's saved score for the
whole-cell lifetime target (``RULLabelAnnotator``: cycles to end of life
counted from the first cycle — cycle life, not remaining useful life):

* RMSE in cycles by default, ``--metric MAE`` for MAE. Both are read verbatim
  from ``payload['scores']``; MAE is never derived from RMSE.
* several independent seeds of ONE configuration -> mean ± sample SD (n-1),
  with n printed; one run -> the value and "1 run"; no run -> "NA" on a grey
  cell (never 0, never interpolated). No "Average" column.

``--source published`` draws the separate published-reference variant
(``fig3a_lifetime_heatmap_published_reference``) from the BatteryML README
table, RMSE only, with ">1000" kept as a censored cell. Local runs and
published values are never mixed in one heatmap.

Usage::

    python -m scripts.paper_figures.fig3a_lifetime_heatmap \\
        --data runs/paper_figures/fig3_data --output runs/paper_figures/fig3_v1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.paper_figures.common import (
    CENSORED_FILL, CREATED, INVALID_COMPARISON, MISSING_DATA, NA_FILL, SEQUENTIAL,
    TEXT_MUTED, TEXT_PRIMARY, PanelResult, PanelSkip, Style, base_parser, check_columns,
    check_observed_agree,
    collected, column_label, command_line, first_seen, fmt_value, group_runs,
    kept_or_skip, model_colors, model_label, model_marker, model_order, report,
    run_provenance, save_panel, smoke_annotation, style_from_args, summarize,
    to_float, unrecorded_smoke,
)

STEM = "fig3a_lifetime_heatmap"
PUBLISHED_STEM = "fig3a_lifetime_heatmap_published_reference"
METRICS = ("RMSE", "MAE")
COLOR_SCALES = ("global", "column", "none")
SOURCE_COLUMNS = [
    "source_type", "model", "model_label", "dataset", "protocol", "metric", "unit",
    "value_mean", "value_sd", "n_runs", "seeds", "entry_ids", "run_ids",
    "censored", "display_text", "color_fraction", "note", "determinism",
    "observation_budget",
]


def local_cells(runs: List[Dict[str, str]], order: List[str], meta: Dict[str, Any], *,
                metric: str, allow_smoke: bool,
                samples: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Pure: runs.csv rows -> the heatmap's cells (one dict per model x column)."""
    if metric not in METRICS:
        raise PanelSkip(INVALID_COMPARISON, f"--metric must be one of {METRICS}.")
    lifetime = [r for r in runs if r.get("task_kind") == "lifetime"]
    kept, excluded = kept_or_skip(lifetime, allow_smoke=allow_smoke,
                                  what="whole-cell lifetime runs (task_kind=lifetime)")
    check_columns(kept)
    if samples is not None:
        check_observed_agree(kept, samples)
    groups = group_runs(kept)
    columns = first_seen((r["dataset"], r["protocol"]) for r in kept)
    models = [m for m in order if any(r["model"] == m for r in kept)]
    unit = kept[0]["unit"]
    budgets = {m: sorted({str(r.get("observation_budget") or "not recorded")
                          for r in kept if r["model"] == m}) for m in models}
    cells = []
    for model in models:
        for dataset, protocol in columns:
            members = groups.get((model, dataset, protocol), [])
            values = [to_float(m.get(f"saved_{metric}")) for m in members]
            cell: Dict[str, Any] = {
                "source_type": "local run", "model": model,
                "model_label": model_label(meta, model), "dataset": dataset,
                "protocol": protocol, "metric": metric, "unit": unit,
                "value_mean": None, "value_sd": None, "n_runs": len(members),
                "seeds": ";".join(str(m["seed"]) for m in members),
                "entry_ids": ";".join(m["entry_id"] for m in members),
                "run_ids": ";".join(str(m["run_id"]) for m in members),
                "censored": False, "note": "",
                "determinism": ";".join(sorted({str(m.get("determinism") or "not recorded")
                                                for m in members})),
            }
            if not members:
                cell["note"] = "no selected run for this model/column"
            elif any(v is None for v in values):
                cell["note"] = (f"saved {metric} missing or non-finite for entries "
                                f"{[m['entry_id'] for m, v in zip(members, values) if v is None]}; "
                                "the cell is NA, not a partial mean")
            else:
                stats = summarize(values)
                cell["value_mean"], cell["value_sd"] = stats["mean"], stats["sd"]
                cell["note"] = ("mean ± sample SD over independent seeds"
                                if stats["n"] >= 2 else
                                "deterministic fit; one run, no error estimate"
                                if cell["determinism"] == "deterministic" else
                                "single run; no error estimate")
            cells.append(cell)
    for cell in cells:
        budget = budgets.get(cell["model"]) or []
        if len(budget) == 1 and budget[0] != "not recorded":
            cell["model_label"] = f"{cell['model_label']}\n({budget[0]})"
        cell["observation_budget"] = ";".join(budget)
    return {"cells": cells, "models": models, "columns": columns, "kept": kept,
            "excluded_smoke": excluded, "unit": unit, "order": list(order),
            "smoke_used": any(str(r.get("smoke")).lower() == "true" for r in kept)}


def published_cells(published: List[Dict[str, str]], order: List[str],
                    meta: Dict[str, Any], *, metric: str) -> Dict[str, Any]:
    """Pure: published_rmse.csv rows -> cells. RMSE only."""
    if metric != "RMSE":
        raise PanelSkip(INVALID_COMPARISON,
                        "the published BatteryML table reports RMSE only; MAE is never "
                        "derived from it.")
    if not published:
        raise PanelSkip(MISSING_DATA, "published_rmse.csv absent; set "
                        "published_reference.include=true in the selection and re-collect.")
    rows = [r for r in published if r["model"] in order]
    if not rows:
        raise PanelSkip(MISSING_DATA, "no published model is listed in model_order.")
    models = [m for m in order if any(r["model"] == m for r in rows)]
    columns = first_seen(("published", r["protocol"]) for r in rows)
    by_key = {(r["model"], r["protocol"]): r for r in rows}
    cells = []
    for model in models:
        for _, protocol in columns:
            row = by_key.get((model, protocol))
            censored = bool(row) and str(row["censored"]).lower() == "true"
            seeds = int(row["seeds"]) if row and row.get("seeds") not in (None, "", "NA") else 0
            cells.append({
                "source_type": "published reference", "model": model,
                # Same label as panels B-E when the selection names one.
                "model_label": model_label(meta, model) if model in (meta.get("model_labels") or {})
                else (row["model_label"] if row else model),
                "dataset": "published", "protocol": protocol, "metric": "RMSE",
                "unit": row["unit"] if row else "cycles",
                "value_mean": to_float(row["central"]) if row else None,
                "value_sd": to_float(row["dispersion"]) if row and seeds >= 2 else None,
                "n_runs": seeds, "seeds": "", "entry_ids": "", "run_ids": "",
                "censored": censored,
                "note": (row["source"] if row else "not in the published table")
                + (f"; censored >{row['censor_threshold']}" if censored else ""),
                "censor_threshold": row.get("censor_threshold") if row else None,
            })
    return {"cells": cells, "models": models, "columns": columns, "kept": [],
            "excluded_smoke": [], "unit": "cycles", "smoke_used": False, "order": list(order)}


def color_fractions(cells: List[Dict[str, Any]], scale: str) -> None:
    """Adds ``color_fraction`` in [0, 1] (None = no colour). Colour only."""
    def span(values: List[float]) -> tuple:
        return (min(values), max(values)) if values else (None, None)

    if scale not in COLOR_SCALES:
        raise PanelSkip(INVALID_COMPARISON, f"--color-scale must be one of {COLOR_SCALES}.")
    finite = [c["value_mean"] for c in cells if c["value_mean"] is not None]
    lo_all, hi_all = span(finite)
    for cell in cells:
        value = cell["value_mean"]
        if scale == "none" or value is None:
            cell["color_fraction"] = None
            continue
        if scale == "column":
            lo, hi = span([c["value_mean"] for c in cells if c["value_mean"] is not None
                           and (c["dataset"], c["protocol"]) == (cell["dataset"], cell["protocol"])])
        else:
            lo, hi = lo_all, hi_all
        # One value (or all equal): the middle step, no division by zero.
        cell["color_fraction"] = 0.5 if hi == lo else (value - lo) / (hi - lo)


def fmt_pair(mean: float, sd: float) -> tuple:
    """Mean and SD printed at the mean's precision (display only)."""
    text = fmt_value(mean)
    decimals = len(text.split(".")[1]) if "." in text else 0
    return text, f"{sd:.{decimals}f}"


def cell_text(cell: Dict[str, Any]) -> str:
    if cell["censored"]:
        return f">{fmt_value(to_float(cell.get('censor_threshold')) or 1000.0)}"
    if cell["value_mean"] is None:
        return "NA"
    if cell["value_sd"] is not None:
        mean, sd = fmt_pair(cell["value_mean"], cell["value_sd"])
        return f"{mean}\n±{sd}\nn={cell['n_runs']}"
    if cell["source_type"] == "local run":
        tag = "deterministic" if cell.get("determinism") == "deterministic" else "1 run"
        return f"{fmt_value(cell['value_mean'])}\n{tag}"
    return fmt_value(cell["value_mean"])


def draw(built: Dict[str, Any], style: Style, *, metric: str, scale: str):
    import matplotlib
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import Rectangle

    from scripts.paper_figures.common import new_figure

    models, columns, cells = built["models"], built["columns"], built["cells"]
    colors = model_colors(built["order"])  # slot = position in the global model_order
    with matplotlib.rc_context(style.rc()):
        fig = new_figure(style)
        ax = fig.add_subplot(111)
        index = {(c["model"], c["dataset"], c["protocol"]): c for c in cells}
        for i, model in enumerate(models):
            for j, (dataset, protocol) in enumerate(columns):
                cell = index[(model, dataset, protocol)]
                fraction = cell["color_fraction"]
                if cell["censored"]:
                    face, ink = CENSORED_FILL, TEXT_PRIMARY
                elif fraction is None:
                    face, ink = (NA_FILL, TEXT_MUTED) if cell["value_mean"] is None else ("#ffffff", TEXT_PRIMARY)
                else:
                    step = int(round(fraction * (len(SEQUENTIAL) - 1)))
                    face = SEQUENTIAL[step]
                    ink = "#ffffff" if step >= 7 else TEXT_PRIMARY
                cell["display_text"] = cell_text(cell)
                # 2 px surface gap between cells (inset rectangle on white).
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, facecolor=face,
                                       edgecolor="none"))
                ax.text(j, i, cell["display_text"], ha="center", va="center", color=ink,
                        fontsize=style.font_size - (1.5 if len(columns) >= 5 else 1),
                        linespacing=1.1)
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([p if d == "published" else column_label(d, p) for d, p in columns])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([c["model_label"] for c in cells[::len(columns)]])
        ax.tick_params(length=0)
        ax.tick_params(axis="y", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Identity marker in the model's fixed colour (same as panels B-E).
        for i, model in enumerate(models):
            ax.scatter([-0.58], [i], marker=model_marker(built["order"], model), s=14,
                       color=colors[model], clip_on=False, linewidths=0)
        if scale == "global" and any(c["color_fraction"] is not None for c in cells):
            values = [c["value_mean"] for c in cells if c["value_mean"] is not None]
            mappable = matplotlib.cm.ScalarMappable(
                norm=Normalize(min(values), max(values)), cmap=ListedColormap(SEQUENTIAL))
            bar = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.03)
            bar.set_label(f"{metric} ({built['unit']})")
            bar.outline.set_visible(False)
        elif scale == "column":
            ax.set_xlabel(f"{metric} ({built['unit']}); shading relative within each column",
                          fontsize=style.font_size - 1)
        else:
            ax.set_xlabel(f"{metric} ({built['unit']})", fontsize=style.font_size - 1)
        smoke_annotation(fig, style, built["smoke_used"])
        fig.tight_layout()
    return fig


def make_panel(data_dir: Path, out_dir: Path, *, style: Optional[Style] = None,
               metric: str = "RMSE", source: str = "local", color_scale: str = "global",
               allow_smoke: bool = False, argv: Optional[List[str]] = None,
               command: Optional[str] = None) -> PanelResult:
    style = style or Style(width_in=3.5, height_in=2.6)
    data = collected(data_dir)
    meta = data["metadata"]
    order = [m for m in model_order(meta)]
    if source == "published":
        built, stem = published_cells(data["published_rmse"], order, meta, metric=metric), PUBLISHED_STEM
    elif source == "local":
        built, stem = local_cells(data["runs"], order, meta, metric=metric,
                                  allow_smoke=allow_smoke, samples=data["samples"]), STEM
    else:
        raise PanelSkip(INVALID_COMPARISON, "--source must be 'local' or 'published'.")
    color_fractions(built["cells"], color_scale)
    fig = draw(built, style, metric=metric, scale=color_scale)
    n_cols = len(built["columns"])
    metadata = {
        "panel": stem,
        "source_type": "published reference" if source == "published" else "local run",
        "target": ("battery lifetime / cycle life (whole-cell RULLabelAnnotator: cycles to "
                   "end of life from the first cycle); not remaining useful life"),
        "metric": metric, "unit": built["unit"],
        "metric_source": ("BatteryML README benchmark table (published; not reproduced here)"
                          if source == "published" else
                          "payload['scores'] written by BatteryML Pipeline.evaluate, copied "
                          "verbatim (already in label units after the recorded inverse transform)"),
        "aggregation": ("mean ± sample SD (n-1) over independent seeds of one configuration; "
                        "a single run is shown as '1 run' without an error estimate"),
        "error_bar_definition": (
            "the '±' dispersion printed in the BatteryML README (ten seeds for neural "
            "models; its estimator is the README's, not recomputed here)"
            if source == "published" else "sample SD (n-1); only when n >= 2"),
        "missing_values": "NA cell (grey), never 0 and never interpolated",
        "non_finite": "a missing/non-finite saved score makes the cell NA; no partial mean",
        "color_scale": color_scale,
        "color_rule": {"global": "one sequential scale over all cells (colorbar)",
                       "column": "scale min-max within each column; printed values carry the number",
                       "none": "no shading"}[color_scale],
        "average_column": "not drawn",
        "single_column": n_cols == 1,
        "single_column_note": ("one dataset/protocol only; this is not a cross-dataset comparison"
                               if n_cols == 1 else ""),
        "models": built["models"],
        "columns": [{"dataset": d, "protocol": p} for d, p in built["columns"]],
        "runs": run_provenance(built["kept"]),
        "excluded_smoke_entries": [r["entry_id"] for r in built["excluded_smoke"]],
        "smoke_included": built["smoke_used"],
        "smoke_note": "smoke runs: workflow-validation only" if built["smoke_used"] else "",
        "smoke_status_not_recorded": unrecorded_smoke(built["kept"]),
        "collected_from": str(data_dir),
        "selection_sha256": meta.get("selection_sha256"),
        "command": command or command_line(argv, "scripts.paper_figures.fig3a_lifetime_heatmap"),
    }
    outputs = save_panel(fig, out_dir, stem, style, built["cells"], SOURCE_COLUMNS, metadata)
    return PanelResult(stem, CREATED, outputs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--metric", choices=METRICS, default="RMSE")
    parser.add_argument("--source", choices=("local", "published"), default="local")
    parser.add_argument("--color-scale", choices=COLOR_SCALES, default="global")
    args = parser.parse_args(argv)
    try:
        result = make_panel(args.data, args.output,
                            style=style_from_args(args, width=3.5, height=2.6),
                            metric=args.metric, source=args.source,
                            color_scale=args.color_scale, allow_smoke=args.allow_smoke,
                            argv=argv if argv is not None else sys.argv[1:])
    except PanelSkip as skip:
        return report(skip)
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
