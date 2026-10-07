"""Fig. 3c — multi-criteria radar from measured, traceable metrics only.

Every axis is a dimension declared in ``radar_config.json`` with a
definition, unit, source, direction (``lower_is_better`` /
``higher_is_better``) and FIXED bounds ``[lo, hi]`` chosen before looking at
the models. The plotted score is::

    s = (value - lo) / (hi - lo)              # higher_is_better
    s = 1 - (value - lo) / (hi - lo)          # lower_is_better

so 1 is the better end of the declared range on every axis. Values come from
``radar_metrics.csv`` (long format: model, dimension, value, unit, source,
n_repeats).

The panel is refused rather than drawn when:

* the config or metrics file is absent                        -> MISSING_DATA
* fewer than 3 dimensions, or any model lacks any dimension    -> INSUFFICIENT_DATA
  (a missing value is never filled with 0)
* bounds are missing/degenerate (lo >= hi), a value lies outside them, a unit
  disagrees, a dimension is subjective (score/rating units, interpretability,
  usability ...), or a model is not in model_order            -> INVALID_COMPARISON

No composite or overall score is computed. Templates:
``fig3c_radar_config.template.json`` and ``fig3c_radar_metrics.template.csv``.

Usage::

    python -m scripts.paper_figures.fig3c_multicriteria_radar \\
        --data runs/paper_figures/fig3_data --output runs/paper_figures/fig3_v1
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.paper_figures.common import (
    CREATED, GRID, INSUFFICIENT_DATA, INVALID_COMPARISON, MISSING_DATA, TEXT_SECONDARY,
    PanelResult, PanelSkip, Style, base_parser, collected, command_line, model_colors,
    model_label, model_marker, model_order, new_figure, report, save_panel,
    style_from_args, to_float,
)

STEM = "fig3c_multicriteria_radar"
DIRECTIONS = ("lower_is_better", "higher_is_better")
DIMENSION_FIELDS = ("name", "definition", "unit", "source", "direction", "bounds")
SUBJECTIVE_UNITS = ("score", "rating", "likert", "subjective", "rank", "stars")
SUBJECTIVE_NAMES = ("interpretab", "explainab", "usability", "ease_of", "user_friendl",
                    "trust", "subjective")
MIN_DIMENSIONS = 3
SOURCE_COLUMNS = ["model", "model_label", "dimension", "value", "unit", "direction",
                  "bound_lo", "bound_hi", "score", "n_repeats", "source"]


def validate_radar_inputs(rows: List[Dict[str, str]], config: Optional[Dict[str, Any]],
                          order: Optional[List[str]] = None) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Pure. Returns (status, problems, scored rows). status is CREATED when drawable."""
    if not config or not rows:
        return MISSING_DATA, ["radar_config.json and radar_metrics.csv are both required; "
                              "see the fig3c_*.template files"], []
    dims = config.get("dimensions")
    if not isinstance(dims, list) or not dims:
        return INVALID_COMPARISON, ["radar_config.json has no 'dimensions' list"], []
    invalid: List[str] = []
    spec: Dict[str, Dict[str, Any]] = {}
    for dim in dims:
        name = str(dim.get("name") or "")
        missing = [f for f in DIMENSION_FIELDS if dim.get(f) in (None, "", [])]
        if missing:
            invalid.append(f"dimension {name or '?'}: missing {missing}")
            continue
        if name in spec:
            invalid.append(f"dimension {name} declared twice")
        if dim["direction"] not in DIRECTIONS:
            invalid.append(f"dimension {name}: direction must be one of {DIRECTIONS}")
        if str(dim["unit"]).strip().lower() in SUBJECTIVE_UNITS or any(
                token in name.lower() for token in SUBJECTIVE_NAMES):
            invalid.append(f"dimension {name}: subjective scores are not allowed; "
                           "use a measured quantity with a physical or count unit")
        bounds = dim["bounds"]
        lo = to_float(bounds[0]) if isinstance(bounds, list) and len(bounds) == 2 else None
        hi = to_float(bounds[1]) if isinstance(bounds, list) and len(bounds) == 2 else None
        if lo is None or hi is None or not lo < hi:
            invalid.append(f"dimension {name}: bounds must be two finite numbers lo < hi, "
                           f"fixed in advance (got {bounds})")
            continue
        spec[name] = {**dim, "lo": lo, "hi": hi}
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        model, name = row.get("model", ""), row.get("dimension", "")
        if order is not None and model not in order:
            invalid.append(f"model {model!r} is not in model_order")
        if name not in spec:
            if not any(str(d.get("name")) == name for d in dims):
                invalid.append(f"metric row {model}/{name}: dimension not declared in radar_config.json")
            continue
        if (model, name) in seen:
            invalid.append(f"{model}/{name} has more than one value")
            continue
        if str(row.get("unit")) != str(spec[name]["unit"]):
            invalid.append(f"{model}/{name}: unit {row.get('unit')!r} != declared {spec[name]['unit']!r}")
        if not str(row.get("source") or "").strip():
            invalid.append(f"{model}/{name}: no source recorded")
        value = to_float(row.get("value"))
        if value is not None and not spec[name]["lo"] <= value <= spec[name]["hi"]:
            invalid.append(f"{model}/{name}: value {value} outside fixed bounds "
                           f"[{spec[name]['lo']}, {spec[name]['hi']}]; widen the declared "
                           "bounds, never clip")
        seen[(model, name)] = {"row": row, "value": value}
    if invalid:
        return INVALID_COMPARISON, invalid, []
    if len(spec) < MIN_DIMENSIONS:
        return INSUFFICIENT_DATA, [f"{len(spec)} dimension(s); a radar needs at least "
                                   f"{MIN_DIMENSIONS} measured dimensions"], []
    models = sorted({m for m, _ in seen}, key=lambda m: order.index(m) if order and m in order else 0)
    gaps = [f"{m}/{d}" for m in models for d in spec
            if (m, d) not in seen or seen[(m, d)]["value"] is None]
    if gaps:
        return INSUFFICIENT_DATA, [f"missing or non-finite values (never filled with 0): {gaps}"], []
    scored = []
    for model in models:
        for name, dim in spec.items():
            value = seen[(model, name)]["value"]
            fraction = (value - dim["lo"]) / (dim["hi"] - dim["lo"])
            row = seen[(model, name)]["row"]
            scored.append({
                "model": model, "dimension": name, "value": value, "unit": dim["unit"],
                "direction": dim["direction"], "bound_lo": dim["lo"], "bound_hi": dim["hi"],
                "score": 1.0 - fraction if dim["direction"] == "lower_is_better" else fraction,
                "n_repeats": row.get("n_repeats") or "not recorded",
                "source": row.get("source"),
            })
    return CREATED, [], scored


def draw(scored: List[Dict[str, Any]], dims: List[str], models: List[str],
         labels: Dict[str, str], directions: Dict[str, str], style: Style):
    import matplotlib

    colors = model_colors(models)
    angles = [2 * math.pi * i / len(dims) for i in range(len(dims))]
    with matplotlib.rc_context(style.rc()):
        fig = new_figure(style)
        # Fixed inset so the outer axis labels and the legend below stay inside.
        ax = fig.add_axes([0.2, 0.22, 0.6, 0.62], projection="polar")
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        for model in models:
            values = [next(r["score"] for r in scored if r["model"] == model and r["dimension"] == d)
                      for d in dims]
            ax.plot(angles + angles[:1], values + values[:1], color=colors[model],
                    marker=model_marker(models, model), markersize=3.5,
                    linewidth=style.line_width * 1.5, label=labels[model])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"], color=TEXT_SECONDARY,
                           fontsize=style.font_size - 1.5)
        ax.set_xticks(angles)
        arrow = {"lower_is_better": "lower better", "higher_is_better": "higher better"}
        ax.set_xticklabels([f"{d}\n({arrow[directions[d]]})" for d in dims],
                           fontsize=style.font_size - 1)
        ax.grid(color=GRID, linewidth=style.line_width * 0.6)
        ax.spines["polar"].set_color(GRID)
        if len(models) >= 2:
            fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="lower center",
                       ncol=min(len(models), 3), bbox_to_anchor=(0.5, 0.0))
    return fig


def make_panel(data_dir: Path, out_dir: Path, *, style: Optional[Style] = None,
               allow_smoke: bool = False, argv: Optional[List[str]] = None,
               command: Optional[str] = None) -> PanelResult:
    style = style or Style(width_in=3.2, height_in=3.0)
    data = collected(data_dir)
    meta = data["metadata"]
    order = model_order(meta)
    status, problems, scored = validate_radar_inputs(data["radar_metrics"], data["radar_config"], order)
    if status != CREATED:
        raise PanelSkip(status, "; ".join(problems))
    dims = [str(d["name"]) for d in data["radar_config"]["dimensions"]]
    models = [m for m in order if any(r["model"] == m for r in scored)]
    labels = {m: model_label(meta, m) for m in models}
    directions = {str(d["name"]): d["direction"] for d in data["radar_config"]["dimensions"]}
    for row in scored:
        row["model_label"] = labels[row["model"]]
    fig = draw(scored, dims, models, labels, directions, style)
    metadata = {
        "panel": STEM, "source_type": "user-supplied measured metrics (radar_metrics.csv)",
        "dimensions": data["radar_config"]["dimensions"],
        "normalization": ("score = (value - lo)/(hi - lo) for higher_is_better; "
                          "1 - (value - lo)/(hi - lo) for lower_is_better; fixed bounds "
                          "from radar_config.json; no clipping; no composite score"),
        "missing_values": "panel refused (INSUFFICIENT_DATA); never filled with 0",
        "models": models,
        "inputs": (meta.get("inputs") or {}).get("radar"),
        "smoke_note": ("radar metrics are user-supplied; their smoke status is whatever their "
                       "'source' column records" if allow_smoke else ""),
        "collected_from": str(data_dir),
        "command": command or command_line(argv, "scripts.paper_figures.fig3c_multicriteria_radar"),
    }
    outputs = save_panel(fig, out_dir, STEM, style, scored, SOURCE_COLUMNS, metadata)
    return PanelResult(STEM, CREATED, outputs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    try:
        result = make_panel(args.data, args.output,
                            style=style_from_args(args, width=3.2, height=3.0),
                            allow_smoke=args.allow_smoke,
                            argv=argv if argv is not None else sys.argv[1:])
    except PanelSkip as skip:
        return report(skip)
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
