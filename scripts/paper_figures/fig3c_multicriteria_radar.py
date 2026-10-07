"""Fig. 3c — radar of measured, traceable metrics (no subjective scores).

Every axis is a dimension declared in ``radar_config.json`` with a
definition, unit, source and direction. Values come from
``radar_metrics.csv`` (long format: model, dimension, value, unit, source,
n_repeats). Two explicit normalization modes, chosen by
``radar_config.json -> normalization.mode``:

``fixed_bounds_linear`` (default when ``normalization`` is absent)
    FIXED bounds ``[lo, hi]`` per dimension, chosen before looking at the
    models::

        s = (value - lo) / (hi - lo)          # higher_is_better
        s = 1 - (value - lo) / (hi - lo)      # lower_is_better

    A value outside its bounds is refused, never clipped.

``reference_relative``
    Each dimension is scored against one named reference model measured
    under the same protocol (``normalization.reference_model``)::

        r = reference model's raw value,  x = this model's raw value
        s = r / (r + x)

    Requires r finite and > 0, x finite and >= 0, and every dimension
    ``lower_is_better`` (error and cost axes). The reference sits at 0.5 on
    every axis; further out = lower error/cost than the reference on that
    axis. Not a bounded "ability" score.

Refused rather than drawn when:

* the config or metrics file is absent                        -> MISSING_DATA
* fewer than 3 dimensions, or any model lacks any dimension    -> INSUFFICIENT_DATA
  (a missing value is never filled with 0)
* invalid bounds / values outside bounds (fixed mode), a missing or invalid
  reference value (reference mode), a unit mismatch, a subjective dimension,
  or a model outside model_order                               -> INVALID_COMPARISON

No composite score, polygon area or "overall best" is computed. Templates:
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
FIXED = "fixed_bounds_linear"
REFERENCE = "reference_relative"
MODES = (FIXED, REFERENCE)
DIRECTIONS = ("lower_is_better", "higher_is_better")
DIMENSION_FIELDS = ("name", "definition", "unit", "source", "direction")
SUBJECTIVE_UNITS = ("score", "rating", "likert", "subjective", "rank", "stars")
SUBJECTIVE_NAMES = ("interpretab", "explainab", "usability", "ease_of", "user_friendl",
                    "trust", "subjective", "robustness_score", "generalization_score")
MIN_DIMENSIONS = 3
SOURCE_COLUMNS = ["model", "model_label", "dimension", "value", "unit", "direction",
                  "normalization", "bound_lo", "bound_hi", "reference_model",
                  "reference_value", "score", "n_repeats", "source"]


def normalization_of(config: Dict[str, Any]) -> Dict[str, Any]:
    spec = config.get("normalization")
    if spec is None:
        return {"mode": FIXED}
    return dict(spec) if isinstance(spec, dict) else {"mode": spec}


def validate_radar_inputs(rows: List[Dict[str, str]], config: Optional[Dict[str, Any]],
                          order: Optional[List[str]] = None
                          ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Pure. Returns (status, problems, scored rows). status is CREATED when drawable."""
    if not config or not rows:
        return MISSING_DATA, ["radar_config.json and radar_metrics.csv are both required; "
                              "see the fig3c_*.template files"], []
    dims = config.get("dimensions")
    if not isinstance(dims, list) or not dims:
        return INVALID_COMPARISON, ["radar_config.json has no 'dimensions' list"], []
    norm = normalization_of(config)
    mode = norm.get("mode")
    if mode not in MODES:
        return INVALID_COMPARISON, [f"normalization.mode must be one of {MODES}"], []
    reference = str(norm.get("reference_model") or "")
    invalid: List[str] = []
    if mode == REFERENCE and not reference:
        invalid.append("reference_relative normalization needs normalization.reference_model")
    spec: Dict[str, Dict[str, Any]] = {}
    for dim in dims:
        name = str(dim.get("name") or "")
        required = DIMENSION_FIELDS + (("bounds",) if mode == FIXED else ())
        missing = [f for f in required if dim.get(f) in (None, "", [])]
        if missing:
            invalid.append(f"dimension {name or '?'}: missing {missing}")
            continue
        if name in spec:
            invalid.append(f"dimension {name} declared twice")
        if dim["direction"] not in DIRECTIONS:
            invalid.append(f"dimension {name}: direction must be one of {DIRECTIONS}")
        if mode == REFERENCE and dim["direction"] != "lower_is_better":
            invalid.append(f"dimension {name}: r/(r+x) is defined for lower_is_better "
                           "axes (errors and costs) only")
        if str(dim["unit"]).strip().lower() in SUBJECTIVE_UNITS or any(
                token in name.lower() for token in SUBJECTIVE_NAMES):
            invalid.append(f"dimension {name}: subjective scores are not allowed; "
                           "use a measured quantity with a physical or count unit")
        entry = dict(dim)
        if mode == FIXED:
            bounds = dim["bounds"]
            lo = to_float(bounds[0]) if isinstance(bounds, list) and len(bounds) == 2 else None
            hi = to_float(bounds[1]) if isinstance(bounds, list) and len(bounds) == 2 else None
            if lo is None or hi is None or not lo < hi:
                invalid.append(f"dimension {name}: bounds must be two finite numbers lo < hi, "
                               f"fixed in advance (got {bounds})")
                continue
            entry.update(lo=lo, hi=hi)
        spec[name] = entry
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
        if value is not None and mode == FIXED and not spec[name]["lo"] <= value <= spec[name]["hi"]:
            invalid.append(f"{model}/{name}: value {value} outside fixed bounds "
                           f"[{spec[name]['lo']}, {spec[name]['hi']}]; widen the declared "
                           "bounds, never clip")
        if value is not None and mode == REFERENCE and value < 0:
            invalid.append(f"{model}/{name}: value {value} < 0; r/(r+x) needs x >= 0")
        seen[(model, name)] = {"row": row, "value": value}
    if mode == REFERENCE and reference:
        for name in spec:
            r = seen.get((reference, name), {}).get("value")
            if r is None or not math.isfinite(r) or r <= 0:
                invalid.append(f"reference {reference}/{name}: value {r} must be finite and > 0")
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
            row = seen[(model, name)]["row"]
            out = {"model": model, "dimension": name, "value": value, "unit": dim["unit"],
                   "direction": dim["direction"], "normalization": mode,
                   "bound_lo": None, "bound_hi": None, "reference_model": None,
                   "reference_value": None,
                   "n_repeats": row.get("n_repeats") or "not recorded",
                   "source": row.get("source")}
            if mode == FIXED:
                fraction = (value - dim["lo"]) / (dim["hi"] - dim["lo"])
                out.update(bound_lo=dim["lo"], bound_hi=dim["hi"],
                           score=1.0 - fraction if dim["direction"] == "lower_is_better" else fraction)
            else:
                r = seen[(reference, name)]["value"]
                out.update(reference_model=reference, reference_value=r, score=r / (r + value))
            scored.append(out)
    return CREATED, [], scored


def draw(scored: List[Dict[str, Any]], dims: List[str], models: List[str], order: List[str],
         labels: Dict[str, str], directions: Dict[str, str], mode: str, style: Style):
    import matplotlib

    colors = model_colors(order)  # slot = position in the global model_order
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
                    marker=model_marker(order, model), markersize=3.5,
                    linewidth=style.line_width * 1.5, label=labels[model])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"], color=TEXT_SECONDARY,
                           fontsize=style.font_size - 1.5)
        ax.set_xticks(angles)
        if mode == REFERENCE:
            tick = [f"{d}\n(outward = lower)" for d in dims]
        else:
            arrow = {"lower_is_better": "lower better", "higher_is_better": "higher better"}
            tick = [f"{d}\n({arrow[directions[d]]})" for d in dims]
        ax.set_xticklabels(tick, fontsize=style.font_size - 1)
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
    config = data["radar_config"]
    norm = normalization_of(config)
    dims = [str(d["name"]) for d in config["dimensions"]]
    models = [m for m in order if any(r["model"] == m for r in scored)]
    labels = {m: model_label(meta, m) for m in models}
    directions = {str(d["name"]): d["direction"] for d in config["dimensions"]}
    for row in scored:
        row["model_label"] = labels[row["model"]]
    fig = draw(scored, dims, models, order, labels, directions, norm["mode"], style)
    formula = (
        "score = r / (r + x); r = reference model's raw value on the same axis and protocol, "
        "x = this model's raw value; reference = 0.5; outward = lower error/cost than the "
        "reference" if norm["mode"] == REFERENCE else
        "score = (value - lo)/(hi - lo) for higher_is_better; 1 - (value - lo)/(hi - lo) "
        "for lower_is_better; fixed bounds from radar_config.json; no clipping")
    metadata = {
        "panel": STEM,
        "display_name": config.get("display_name") or "",
        "source_type": "measured metrics (radar_metrics.csv)",
        "dimensions": config["dimensions"],
        "normalization": norm,
        "normalization_formula": formula,
        "composite_score": "none (no total, no polygon area, no overall ranking)",
        "missing_values": "panel refused (INSUFFICIENT_DATA); never filled with 0",
        "models": models,
        "raw_values": [{k: r[k] for k in ("model", "dimension", "value", "unit", "source")}
                       for r in scored],
        "inputs": (meta.get("inputs") or {}).get("radar"),
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
