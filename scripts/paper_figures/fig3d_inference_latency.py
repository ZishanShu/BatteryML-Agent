"""Fig. 3d — measured inference latency per model.

Reads ``latency.csv`` (copied by ``collect`` from the file the selection
names; template: ``fig3d_latency.template.csv``). Nothing here measures,
estimates or derives latency: training time, CLI wall time, model-load time
and timestamps in file names are not inference latency and are refused.

Every row must record the timing protocol: device, hardware, batch size, CPU
threads, warm-up runs, repeats, timing boundary, whether data transfer /
feature computation / post-processing are inside the boundary, and GPU
synchronisation. Rows measured under different protocols, hardware or units
(``ms_per_sample`` vs ``ms_per_batch``) are not drawn together
(INVALID_COMPARISON).

Error bars: the recorded SD, only for ``statistic = mean`` with ``repeats >= 2``.
``--yscale log`` requires every value to be positive.

When no measurements exist the panel reports MISSING_DATA and lists the
measurements to take (one row per model in model_order).

Usage::

    python -m scripts.paper_figures.fig3d_inference_latency \\
        --data runs/paper_figures/fig3_data --output runs/paper_figures/fig3_v1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.paper_figures.common import (
    CREATED, GRID, INVALID_COMPARISON, MISSING_DATA, TEXT_PRIMARY, PanelResult, PanelSkip,
    Style, base_parser, collected, command_line, model_colors, model_label, model_order,
    new_figure, report, save_panel, style_from_args, to_float,
)

STEM = "fig3d_inference_latency"
LATENCY_COLUMNS = [
    "model", "latency_value", "latency_unit", "statistic", "sd", "batch_size", "device",
    "hardware", "cpu_threads", "warmup_runs", "repeats", "timing_boundary",
    "includes_data_transfer", "includes_feature_computation", "includes_postprocessing",
    "gpu_sync", "measurement_kind", "source_file",
]
UNITS = ("ms_per_sample", "ms_per_batch")
STATISTICS = ("mean", "median")
#: Whole words that mark a timing as something other than model inference
#: ("preloaded features" is fine; "model load + predict" is not).
NOT_INFERENCE = re.compile(
    r"\b(train\w*|fit\w*|epochs?|cli|wall|load|loading|timestamps?|filenames?|"
    r"startup|imports?)\b")
#: Fields that must be identical for rows to share one axis.
PROTOCOL_FIELDS = ("latency_unit", "statistic", "batch_size", "device", "hardware",
                   "cpu_threads", "warmup_runs", "timing_boundary", "includes_data_transfer",
                   "includes_feature_computation", "includes_postprocessing", "gpu_sync")
BOOLEAN_FIELDS = ("includes_data_transfer", "includes_feature_computation",
                  "includes_postprocessing", "gpu_sync")
SOURCE_COLUMNS = ["model", "model_label", "latency_value", "latency_unit", "statistic",
                  "sd", "error_bar", "repeats", "batch_size", "device", "hardware",
                  "cpu_threads", "warmup_runs", "timing_boundary", "source_file"]


def _int(text: Any) -> Optional[int]:
    try:
        return int(str(text).strip())
    except ValueError:
        return None


def validate_latency_rows(rows: List[Dict[str, str]],
                          order: Optional[List[str]] = None) -> List[str]:
    """Pure. Every reason the rows cannot share one latency axis; [] if they can."""
    problems: List[str] = []
    if not rows:
        return problems
    missing_cols = [c for c in LATENCY_COLUMNS if c not in rows[0]]
    if missing_cols:
        return [f"latency.csv lacks columns {missing_cols}; use fig3d_latency.template.csv"]
    seen = set()
    for i, row in enumerate(rows, start=2):
        where = f"latency.csv line {i} ({row.get('model')})"
        if row["model"] in seen:
            problems.append(f"{where}: model listed twice")
        seen.add(row["model"])
        if order is not None and row["model"] not in order:
            problems.append(f"{where}: model not in model_order")
        if str(row["measurement_kind"]).strip().lower() != "inference":
            problems.append(f"{where}: measurement_kind must be 'inference'")
        boundary = str(row["timing_boundary"]).lower()
        if not boundary.strip():
            problems.append(f"{where}: timing_boundary not recorded")
        bad = sorted(set(NOT_INFERENCE.findall(boundary)))
        if bad:
            problems.append(f"{where}: timing_boundary mentions {bad}; training, CLI, "
                            "load or file-timestamp durations are not inference latency")
        if row["latency_unit"] not in UNITS:
            problems.append(f"{where}: latency_unit must be one of {UNITS}")
        if row["statistic"] not in STATISTICS:
            problems.append(f"{where}: statistic must be one of {STATISTICS}")
        value = to_float(row["latency_value"])
        if value is None or value <= 0:
            problems.append(f"{where}: latency_value must be a positive finite number")
        for name, minimum in (("batch_size", 1), ("repeats", 1), ("warmup_runs", 0),
                              ("cpu_threads", 1)):
            number = _int(row[name])
            if number is None or number < minimum:
                problems.append(f"{where}: {name} must be an integer >= {minimum}")
        for name in BOOLEAN_FIELDS:
            if str(row[name]).strip().lower() not in ("true", "false"):
                problems.append(f"{where}: {name} must be true or false")
        if str(row["device"]).lower().startswith("cuda") and str(row["gpu_sync"]).lower() != "true":
            problems.append(f"{where}: CUDA timing without gpu_sync=true is not a latency")
        if not str(row["hardware"]).strip():
            problems.append(f"{where}: hardware not recorded")
        if not str(row["source_file"]).strip():
            problems.append(f"{where}: source_file (raw timing log) not recorded")
        sd = to_float(row.get("sd"))
        if sd is not None and (row["statistic"] != "mean" or (_int(row["repeats"]) or 0) < 2):
            problems.append(f"{where}: an SD needs statistic=mean and repeats >= 2")
    for name in PROTOCOL_FIELDS:
        values = sorted({str(r.get(name)).strip() for r in rows})
        if len(values) > 1:
            problems.append(f"rows differ in {name} ({values}); different hardware or timing "
                            "definitions are not drawn on one axis")
    return problems


def missing_measurements(order: List[str], rows: List[Dict[str, str]]) -> List[str]:
    have = {r.get("model") for r in rows}
    return [m for m in order if m not in have]


def build(rows: List[Dict[str, str]], order: List[str], meta: Dict[str, Any], *,
          yscale: str) -> Dict[str, Any]:
    if not rows:
        raise PanelSkip(MISSING_DATA,
                        "no inference-latency measurements (latency.csv). Measure, per model "
                        f"in model_order {order}: inference only, fixed batch size, device, "
                        "threads, warm-up and repeats; fill fig3d_latency.template.csv and "
                        "name it in the selection as latency_csv.")
    problems = validate_latency_rows(rows, order)
    if problems:
        raise PanelSkip(INVALID_COMPARISON, "; ".join(problems))
    if yscale not in ("linear", "log"):
        raise PanelSkip(INVALID_COMPARISON, "--yscale must be 'linear' or 'log'.")
    models = [m for m in order if any(r["model"] == m for r in rows)]
    by_model = {r["model"]: r for r in rows}
    bars = []
    for model in models:
        row = by_model[model]
        sd = to_float(row.get("sd"))
        bars.append({
            "model": model, "model_label": model_label(meta, model),
            "latency_value": to_float(row["latency_value"]),
            "latency_unit": row["latency_unit"], "statistic": row["statistic"],
            "sd": sd, "error_bar": "sample SD over repeats" if sd is not None else "none",
            "repeats": _int(row["repeats"]), "batch_size": _int(row["batch_size"]),
            "device": row["device"], "hardware": row["hardware"],
            "cpu_threads": _int(row["cpu_threads"]), "warmup_runs": _int(row["warmup_runs"]),
            "timing_boundary": row["timing_boundary"], "source_file": row["source_file"],
        })
    protocol = {name: rows[0][name] for name in PROTOCOL_FIELDS}
    return {"bars": bars, "models": models, "protocol": protocol, "order": list(order),
            "not_measured": missing_measurements(order, rows)}


def draw(built: Dict[str, Any], style: Style, *, yscale: str):
    import matplotlib

    models = built["models"]
    colors = model_colors(built["order"])  # slot = position in the global model_order
    unit = {"ms_per_sample": "ms per sample", "ms_per_batch": "ms per batch"}[
        built["protocol"]["latency_unit"]]
    with matplotlib.rc_context(style.rc()):
        fig = new_figure(style)
        ax = fig.add_subplot(111)
        xs = list(range(len(models)))
        heights = [b["latency_value"] for b in built["bars"]]
        ax.bar(xs, heights, width=0.6, color=[colors[m] for m in models], linewidth=0)
        err = [(x, b["latency_value"], b["sd"]) for x, b in zip(xs, built["bars"]) if b["sd"] is not None]
        if err:
            ax.errorbar([e[0] for e in err], [e[1] for e in err], yerr=[e[2] for e in err],
                        fmt="none", ecolor=TEXT_PRIMARY, elinewidth=style.line_width,
                        capsize=1.5, capthick=style.line_width)
        ax.set_xticks(xs)
        ax.set_xticklabels([b["model_label"] for b in built["bars"]], rotation=30, ha="right")
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_ylim(bottom=0)
        ax.set_ylabel(f"Inference latency\n({built['protocol']['statistic']}, {unit})")
        ax.yaxis.grid(True, color=GRID, linewidth=style.line_width * 0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
    return fig


def make_panel(data_dir: Path, out_dir: Path, *, style: Optional[Style] = None,
               yscale: str = "linear", allow_smoke: bool = False,
               argv: Optional[List[str]] = None,
               command: Optional[str] = None) -> PanelResult:
    style = style or Style(width_in=3.0, height_in=2.4)
    data = collected(data_dir)
    meta = data["metadata"]
    built = build(data["latency"], model_order(meta), meta, yscale=yscale)
    fig = draw(built, style, yscale=yscale)
    metadata = {
        "panel": STEM, "source_type": "user-supplied inference timing (latency.csv)",
        "quantity": "model inference latency only",
        "excluded_quantities": "training time, CLI wall time, model/data load time, file-name timestamps",
        "timing_protocol": built["protocol"],
        "error_bar_definition": "recorded SD over repeats; only for statistic=mean with repeats >= 2",
        "yscale": yscale,
        "models": built["models"],
        "models_not_measured": built["not_measured"],
        "inputs": (meta.get("inputs") or {}).get("latency_csv"),
        "collected_from": str(data_dir),
        "command": command or command_line(argv, "scripts.paper_figures.fig3d_inference_latency"),
    }
    outputs = save_panel(fig, out_dir, STEM, style, built["bars"], SOURCE_COLUMNS, metadata)
    return PanelResult(STEM, CREATED, outputs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    args = parser.parse_args(argv)
    try:
        result = make_panel(args.data, args.output,
                            style=style_from_args(args, width=3.0, height=2.4),
                            yscale=args.yscale, allow_smoke=args.allow_smoke,
                            argv=argv if argv is not None else sys.argv[1:])
    except PanelSkip as skip:
        return report(skip)
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
