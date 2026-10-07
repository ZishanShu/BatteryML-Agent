"""Fig. 3b — SOH estimation error: grouped bars per dataset/protocol.

x = dataset + protocol (as recorded), one bar per model in ``model_order``
(the same order and colours as panel A), y = SOH MAE.

* Default metric: the canonical cycle-pooled MAE — BatteryML's saved
  ``payload['scores']['MAE']``, in which every stored cell-cycle row weighs the
  same. ``--metric cell-macro`` instead draws the per-cell macro MAE (mean over
  cells of each cell's MAE, computed from samples.csv); it is a different
  statistic, written under its own file name and axis label.
* Default scale: ratio (the SOH label is a capacity ratio). ``--scale pp``
  multiplies by 100 and labels the axis "percentage points" — this is NOT MAPE.
* Error bars: sample SD (n-1) across independent seeds of one configuration,
  only when n >= 2. A single run has no error bar; cycles are not replicates.
* A model without a run in a group gets an "NA" label at the baseline, no bar.

Usage::

    python -m scripts.paper_figures.fig3b_soh_error \\
        --data runs/paper_figures/fig3_data --output runs/paper_figures/fig3_v1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.paper_figures.common import (
    CREATED, INVALID_COMPARISON, TEXT_MUTED, TEXT_PRIMARY, PanelResult, PanelSkip, Style,
    base_parser, check_columns, check_observed_agree, collected, column_label, command_line, first_seen,
    group_runs, kept_or_skip, model_colors, model_label, model_order, new_figure, report,
    run_provenance, save_panel, smoke_annotation, style_from_args, summarize, to_float,
    unrecorded_smoke,
)

STEM = "fig3b_soh_error"
MACRO_STEM = "fig3b_soh_error_cell_macro"
SCALES = ("ratio", "pp")
SOURCE_COLUMNS = [
    "model", "model_label", "dataset", "protocol", "statistic", "scale", "unit",
    "value_mean", "value_sd", "n_runs", "seeds", "entry_ids", "run_ids",
    "per_run_values", "n_rows", "n_cells", "note",
]


def cell_macro_mae(samples: List[Dict[str, str]], entry_id: str) -> Optional[float]:
    """Mean over cells of each cell's mean |predicted - observed|."""
    per_cell: Dict[str, List[float]] = {}
    for row in samples:
        if row["entry_id"] != entry_id:
            continue
        observed, predicted = to_float(row["observed"]), to_float(row["predicted"])
        if observed is None or predicted is None:
            raise PanelSkip(INVALID_COMPARISON,
                            f"samples.csv entry {entry_id} row {row['artifact_row']} is "
                            "non-finite; collect drops those, so the file was altered.")
        per_cell.setdefault(row["cell_id"], []).append(abs(predicted - observed))
    if not per_cell:
        return None
    return sum(sum(v) / len(v) for v in per_cell.values()) / len(per_cell)


def build(runs: List[Dict[str, str]], samples: List[Dict[str, str]], order: List[str],
          meta: Dict[str, Any], *, statistic: str, scale: str,
          allow_smoke: bool) -> Dict[str, Any]:
    """Pure: runs/samples rows -> one bar spec per (column, model)."""
    if scale not in SCALES:
        raise PanelSkip(INVALID_COMPARISON, f"--scale must be one of {SCALES}.")
    if statistic not in ("pooled", "cell-macro"):
        raise PanelSkip(INVALID_COMPARISON, "--metric must be 'pooled' or 'cell-macro'.")
    soh = [r for r in runs if r.get("task_kind") == "soh"]
    kept, excluded = kept_or_skip(soh, allow_smoke=allow_smoke, what="SOH runs (task_kind=soh)")
    check_columns(kept)
    check_observed_agree(kept, samples)
    groups = group_runs(kept)
    columns = first_seen((r["dataset"], r["protocol"]) for r in kept)
    models = [m for m in order if any(r["model"] == m for r in kept)]
    unit = kept[0]["unit"]
    factor = 100.0 if scale == "pp" else 1.0
    bars = []
    for dataset, protocol in columns:
        for model in models:
            members = groups.get((model, dataset, protocol), [])
            if statistic == "pooled":
                values = [to_float(m.get("saved_MAE")) for m in members]
            else:
                values = [cell_macro_mae(samples, m["entry_id"]) for m in members]
            bar: Dict[str, Any] = {
                "model": model, "model_label": model_label(meta, model),
                "dataset": dataset, "protocol": protocol,
                "statistic": ("SOH MAE, cycle-pooled (BatteryML saved score)"
                              if statistic == "pooled" else "SOH MAE, per-cell macro average"),
                "scale": scale,
                "unit": "percentage points (ratio x 100)" if scale == "pp" else unit,
                "value_mean": None, "value_sd": None, "n_runs": len(members),
                "seeds": ";".join(str(m["seed"]) for m in members),
                "entry_ids": ";".join(m["entry_id"] for m in members),
                "run_ids": ";".join(str(m["run_id"]) for m in members),
                "per_run_values": ";".join("NA" if v is None else repr(v * factor) for v in values),
                "n_rows": ";".join(str(m["plotted_rows"]) for m in members),
                "n_cells": ";".join(str(m["n_cells"]) for m in members),
                "note": "",
            }
            if not members:
                bar["note"] = "no selected run; drawn as NA"
            elif any(v is None for v in values):
                bar["note"] = "a run lacks the statistic; drawn as NA, not a partial mean"
            else:
                stats = summarize([v * factor for v in values])
                bar["value_mean"], bar["value_sd"] = stats["mean"], stats["sd"]
                bar["note"] = ("mean ± sample SD over independent seeds" if stats["n"] >= 2
                               else "single run; no error bar")
            bars.append(bar)
    return {"bars": bars, "models": models, "columns": columns, "kept": kept,
            "excluded_smoke": excluded, "unit": unit, "order": list(order),
            "smoke_used": any(str(r.get("smoke")).lower() == "true" for r in kept)}


def draw(built: Dict[str, Any], style: Style, *, statistic: str, scale: str):
    import matplotlib

    models, columns = built["models"], built["columns"]
    colors = model_colors(built["order"])  # slot = position in the global model_order
    width = 0.8 / max(len(models), 1)
    with matplotlib.rc_context(style.rc()):
        fig = new_figure(style)
        ax = fig.add_subplot(111)
        index = {(b["model"], b["dataset"], b["protocol"]): b for b in built["bars"]}
        labels = {b["model"]: b["model_label"] for b in built["bars"]}
        top = max([(b["value_mean"] or 0) + (b["value_sd"] or 0) for b in built["bars"]] + [0])
        for k, model in enumerate(models):
            xs, heights, errs, na_x = [], [], [], []
            for j, (dataset, protocol) in enumerate(columns):
                bar = index[(model, dataset, protocol)]
                x = j - 0.4 + width * (k + 0.5)
                if bar["value_mean"] is None:
                    na_x.append(x)
                    continue
                xs.append(x)
                heights.append(bar["value_mean"])
                errs.append(bar["value_sd"])
            # Bars inset by a hairline of surface so neighbours stay distinct.
            ax.bar(xs, heights, width=width * 0.9, color=colors[model],
                   label=labels[model],
                   linewidth=0)
            err_x = [x for x, e in zip(xs, errs) if e is not None]
            err_y = [h for h, e in zip(heights, errs) if e is not None]
            err_v = [e for e in errs if e is not None]
            if err_x:
                ax.errorbar(err_x, err_y, yerr=err_v, fmt="none", ecolor=TEXT_PRIMARY,
                            elinewidth=style.line_width, capsize=1.5, capthick=style.line_width)
            for x in na_x:
                ax.text(x, top * 0.02 if top > 0 else 0.0, "NA", ha="center", va="bottom",
                        rotation=90, color=TEXT_MUTED, fontsize=style.font_size - 1.5)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([column_label(d, p) for d, p in columns])
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(bottom=0)
        name = "SOH MAE" if statistic == "pooled" else "SOH MAE, per-cell macro"
        unit = "percentage points" if scale == "pp" else built["unit"]
        ax.set_ylabel(f"{name} ({unit})")
        ax.yaxis.grid(True, color="#e1e0d9", linewidth=style.line_width * 0.6)
        ax.set_axisbelow(True)
        if len(models) >= 2:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        smoke_annotation(fig, style, built["smoke_used"])
        fig.tight_layout()
    return fig


def make_panel(data_dir: Path, out_dir: Path, *, style: Optional[Style] = None,
               statistic: str = "pooled", scale: str = "ratio", allow_smoke: bool = False,
               argv: Optional[List[str]] = None,
               command: Optional[str] = None) -> PanelResult:
    style = style or Style(width_in=3.5, height_in=2.4)
    data = collected(data_dir)
    meta = data["metadata"]
    built = build(data["runs"], data["samples"], model_order(meta), meta,
                  statistic=statistic, scale=scale, allow_smoke=allow_smoke)
    fig = draw(built, style, statistic=statistic, scale=scale)
    stem = STEM if statistic == "pooled" else MACRO_STEM
    metadata = {
        "panel": stem, "source_type": "local run",
        "target": "state of health (capacity ratio), cycle-level estimates",
        "statistic": statistic,
        "statistic_definition": (
            "cycle-pooled MAE = mean over all stored cell-cycle rows of |predicted - observed|, "
            "as saved by BatteryML Pipeline.evaluate (payload['scores']['MAE'])"
            if statistic == "pooled" else
            "per-cell macro MAE = mean over cells of (mean over that cell's rows of "
            "|predicted - observed|), computed from samples.csv; NOT the canonical metric"),
        "scale": scale,
        "scale_rule": "ratio x 100 = percentage points (not MAPE)" if scale == "pp" else "ratio as stored",
        "unit": built["unit"],
        "aggregation": "mean ± sample SD (n-1) across independent seeds of one configuration",
        "error_bar_definition": "sample SD (n-1); drawn only when n >= 2; cycles are not replicates",
        "missing_values": "NA label, no bar, never 0",
        "models": built["models"],
        "columns": [{"dataset": d, "protocol": p} for d, p in built["columns"]],
        "runs": run_provenance(built["kept"]),
        "excluded_smoke_entries": [r["entry_id"] for r in built["excluded_smoke"]],
        "smoke_included": built["smoke_used"],
        "smoke_status_not_recorded": unrecorded_smoke(built["kept"]),
        "collected_from": str(data_dir),
        "selection_sha256": meta.get("selection_sha256"),
        "command": command or command_line(argv, "scripts.paper_figures.fig3b_soh_error"),
    }
    outputs = save_panel(fig, out_dir, stem, style, built["bars"], SOURCE_COLUMNS, metadata)
    return PanelResult(stem, CREATED, outputs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--metric", choices=("pooled", "cell-macro"), default="pooled")
    parser.add_argument("--scale", choices=SCALES, default="ratio")
    args = parser.parse_args(argv)
    try:
        result = make_panel(args.data, args.output,
                            style=style_from_args(args, width=3.5, height=2.4),
                            statistic=args.metric, scale=args.scale,
                            allow_smoke=args.allow_smoke,
                            argv=argv if argv is not None else sys.argv[1:])
    except PanelSkip as skip:
        return report(skip)
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
