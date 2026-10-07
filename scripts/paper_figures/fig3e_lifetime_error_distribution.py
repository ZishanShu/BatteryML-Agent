"""Fig. 3e — per-cell absolute lifetime error distribution, one column per model.

``absolute_error = |predicted - observed|`` for every test cell, in cycles,
from samples.csv (observed/predicted after the recorded label transformation
was inverted exactly once by ``collect``). Nothing is simulated from RMSE or
MAE.

Comparison rules (refused as INVALID_COMPARISON otherwise):

* one dataset/protocol (``--dataset`` / ``--protocol`` when several exist);
* one run per model: when a model has several seeds, ``--seed`` picks the
  same seed for every model — seeds are never pooled as if they were cells;
* every model must cover exactly the same test-cell IDs, one row per cell.

Display: violin + box when every model has >= 20 cells and >= 5 distinct
errors; otherwise box + every point (deterministic horizontal offsets), and
the reason is recorded. Outliers are drawn, never clipped. ``--yscale log``
adds no epsilon: zero errors cannot sit on a log axis, so they are counted,
labelled above the model, and left out of the marks; the strategy is saved.

Usage::

    python -m scripts.paper_figures.fig3e_lifetime_error_distribution \\
        --data runs/paper_figures/fig3_data --output runs/paper_figures/fig3_v1 \\
        [--dataset MATR --protocol MATR1] [--seed 0]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.paper_figures.common import (
    CREATED, GRID, INVALID_COMPARISON, MISSING_DATA, TEXT_PRIMARY, TEXT_SECONDARY,
    PanelResult, PanelSkip, Style, base_parser, check_columns, collected, command_line,
    check_observed_agree, first_seen, group_runs, kept_or_skip, model_colors, model_label, model_marker,
    model_order, new_figure, report, run_provenance, save_panel, smoke_annotation,
    style_from_args, to_float, unrecorded_smoke,
)

STEM = "fig3e_lifetime_error_distribution"
VIOLIN_MIN_N = 20
VIOLIN_MIN_DISTINCT = 5
SOURCE_COLUMNS = ["model", "model_label", "entry_id", "seed", "cell_id", "artifact_row",
                  "observed", "predicted", "abs_error", "unit", "on_axis"]


def _pick_column(kept: List[Dict[str, str]], dataset: Optional[str],
                 protocol: Optional[str]) -> tuple:
    columns = first_seen((r["dataset"], r["protocol"]) for r in kept)
    chosen = [c for c in columns if (dataset is None or c[0] == dataset)
              and (protocol is None or c[1] == protocol)]
    if not chosen:
        raise PanelSkip(MISSING_DATA, f"no lifetime run on dataset={dataset} protocol={protocol}; "
                        f"available: {columns}")
    if len(chosen) > 1:
        raise PanelSkip(INVALID_COMPARISON,
                        f"several dataset/protocol columns {chosen}; distributions from different "
                        "test sets are not drawn together. Pick one with --dataset/--protocol.")
    return chosen[0]


def build(runs: List[Dict[str, str]], samples: List[Dict[str, str]], order: List[str],
          meta: Dict[str, Any], *, dataset: Optional[str] = None,
          protocol: Optional[str] = None, seed: Optional[str] = None,
          allow_smoke: bool = False, yscale: str = "linear") -> Dict[str, Any]:
    """Pure: runs/samples rows -> per-model error lists plus the display decision."""
    if yscale not in ("linear", "log"):
        raise PanelSkip(INVALID_COMPARISON, "--yscale must be 'linear' or 'log'.")
    lifetime = [r for r in runs if r.get("task_kind") == "lifetime"]
    kept, excluded = kept_or_skip(lifetime, allow_smoke=allow_smoke,
                                  what="whole-cell lifetime runs (task_kind=lifetime)")
    column = _pick_column(kept, dataset, protocol)
    in_column = [r for r in kept if (r["dataset"], r["protocol"]) == column]
    check_columns(in_column)
    groups = group_runs(in_column)
    models = [m for m in order if (m, *column) in groups]
    chosen: Dict[str, Dict[str, str]] = {}
    for model in models:
        members = groups[(model, *column)]
        if seed is not None:
            match = [m for m in members if str(m["seed"]) == str(seed)]
            if not match:
                raise PanelSkip(INVALID_COMPARISON,
                                f"{model} has no run with seed {seed} on {column} "
                                f"(seeds: {[m['seed'] for m in members]}).")
            chosen[model] = match[0]
        elif len(members) > 1:
            raise PanelSkip(INVALID_COMPARISON,
                            f"{model} has {len(members)} seeds on {column} "
                            f"({[m['seed'] for m in members]}); choose one with --seed. "
                            "Seeds are not pooled as independent cells.")
        else:
            chosen[model] = members[0]

    by_entry: Dict[str, List[Dict[str, str]]] = {}
    for row in samples:
        by_entry.setdefault(row["entry_id"], []).append(row)
    # A model whose rows cannot be tied to one named test cell each is left
    # out with its reason; its rows are never averaged, deduplicated or cut.
    excluded_models: Dict[str, str] = {}
    for model, run in list(chosen.items()):
        rows = by_entry.get(run["entry_id"], [])
        cells = [str(r["cell_id"]) for r in rows]
        if run.get("identity_status") not in ("recorded", "recovered"):
            excluded_models[model] = (
                f"entry {run['entry_id']}: cell identity {run.get('identity_status') or 'unknown'} "
                f"({run.get('identity_source') or 'no identity source'}); rows cannot be "
                "matched to test cells")
        elif not rows:
            excluded_models[model] = f"entry {run['entry_id']}: no rows in samples.csv"
        elif len(cells) != len(set(cells)):
            excluded_models[model] = (
                f"entry {run['entry_id']}: {len(cells)} rows for {len(set(cells))} cells; "
                "repeated rows are distinct observations and are not averaged or deduplicated")
        if model in excluded_models:
            del chosen[model]
    if not chosen:
        raise PanelSkip(INVALID_COMPARISON, "no model has one verified row per test cell: "
                        + "; ".join(excluded_models.values()))
    models = [m for m in models if m in chosen]
    check_observed_agree(list(chosen.values()),
                         [r for r in samples if r["entry_id"] in {c["entry_id"] for c in chosen.values()}])

    unit = in_column[0]["unit"]
    points: List[Dict[str, Any]] = []
    errors: Dict[str, List[float]] = {}
    cell_sets: Dict[str, List[str]] = {}
    for model, run in chosen.items():
        rows = by_entry[run["entry_id"]]
        values = []
        for row in rows:
            observed, predicted = to_float(row["observed"]), to_float(row["predicted"])
            if observed is None or predicted is None:
                raise PanelSkip(INVALID_COMPARISON,
                                f"entry {run['entry_id']} row {row['artifact_row']} is non-finite; "
                                "collect drops those, so samples.csv was altered.")
            error = abs(predicted - observed)
            stored = to_float(row["abs_error"])
            if stored is None or abs(stored - error) > 1e-9 * max(1.0, abs(error)):
                raise PanelSkip(INVALID_COMPARISON,
                                f"entry {run['entry_id']} row {row['artifact_row']}: abs_error "
                                "does not equal |predicted - observed|; samples.csv was altered.")
            values.append(error)
            points.append({"model": model, "model_label": model_label(meta, model),
                           "entry_id": run["entry_id"], "seed": run["seed"],
                           "cell_id": row["cell_id"], "artifact_row": row["artifact_row"],
                           "observed": observed, "predicted": predicted, "abs_error": error,
                           "unit": unit, "on_axis": not (yscale == "log" and error == 0.0)})
        errors[model] = values
        cell_sets[model] = sorted(str(r["cell_id"]) for r in rows)
    reference = next(iter(cell_sets.values()))
    for model, cells in cell_sets.items():
        if cells != reference:
            only_here = sorted(set(cells) - set(reference))
            only_ref = sorted(set(reference) - set(cells))
            raise PanelSkip(INVALID_COMPARISON,
                            f"{model} covers different test cells than {models[0]} "
                            f"(only in {model}: {only_here[:5]}; missing: {only_ref[:5]}); "
                            "distributions over different cells are not compared, and no "
                            "intersection is taken.")

    small = {m: (len(v), len(set(v))) for m, v in errors.items()
             if len(v) < VIOLIN_MIN_N or len(set(v)) < VIOLIN_MIN_DISTINCT}
    if yscale == "log":
        mode = "box+points"
        reason = ("box + points: a KDE estimated on the linear scale would be distorted "
                  "on a log axis, so no violin is drawn")
    elif small:
        mode = "box+points"
        reason = (f"box + points: KDE not used because {small} (n, distinct) fall below "
                  f"n>={VIOLIN_MIN_N} and distinct>={VIOLIN_MIN_DISTINCT}")
    else:
        mode = "violin+box"
        reason = (f"violin + box: every model has n>={VIOLIN_MIN_N} and "
                  f">={VIOLIN_MIN_DISTINCT} distinct errors")
    zeros = {m: sum(1 for e in v if e == 0.0) for m, v in errors.items()}
    return {"points": points, "errors": errors, "models": models, "column": column,
            "labels": {m: model_label(meta, m) for m in models},
            "chosen": chosen, "mode": mode, "mode_reason": reason, "zeros": zeros,
            "unit": unit, "kept": list(chosen.values()), "excluded_smoke": excluded,
            "n_cells": len(reference), "excluded_models": excluded_models,
            "order": list(order),
            "smoke_used": any(str(r.get("smoke")).lower() == "true" for r in chosen.values())}


def _offsets(n: int, width: float = 0.28) -> List[float]:
    """Deterministic, evenly spread horizontal offsets (golden-ratio sequence)."""
    return [((i * 0.6180339887) % 1.0 - 0.5) * width for i in range(n)]


def draw(built: Dict[str, Any], style: Style, *, yscale: str):
    import matplotlib

    models = built["models"]
    colors = model_colors(built["order"])  # slot = position in the global model_order
    with matplotlib.rc_context(style.rc()):
        fig = new_figure(style)
        ax = fig.add_subplot(111)
        for i, model in enumerate(models):
            values = [e for e in built["errors"][model] if not (yscale == "log" and e == 0.0)]
            if not values:
                continue
            if built["mode"] == "violin+box":
                parts = ax.violinplot([values], positions=[i], widths=0.7,
                                      showextrema=False, showmedians=False)
                for body in parts["bodies"]:
                    body.set_facecolor(colors[model])
                    body.set_edgecolor(colors[model])
                    body.set_alpha(0.35)
                box_width = 0.16
            else:
                ax.scatter([i + o for o in _offsets(len(values))], values, s=9,
                           marker=model_marker(built["order"], model), color=colors[model],
                           edgecolors="#ffffff", linewidths=0.4, zorder=3)
                box_width = 0.45
            ax.boxplot([values], positions=[i], widths=box_width, whis=1.5,
                       showfliers=built["mode"] == "violin+box", patch_artist=True,
                       boxprops={"facecolor": "none", "edgecolor": TEXT_PRIMARY,
                                 "linewidth": style.line_width},
                       medianprops={"color": TEXT_PRIMARY, "linewidth": style.line_width * 1.5},
                       whiskerprops={"color": TEXT_PRIMARY, "linewidth": style.line_width},
                       capprops={"color": TEXT_PRIMARY, "linewidth": style.line_width},
                       flierprops={"marker": "o", "markersize": 2, "markerfacecolor": colors[model],
                                   "markeredgecolor": "none"})
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([built["labels"][m] for m in models], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_ylim(bottom=0)
        ax.set_ylabel(f"Absolute lifetime error ({built['unit']})")
        ax.yaxis.grid(True, color=GRID, linewidth=style.line_width * 0.6)
        ax.set_axisbelow(True)
        if yscale == "log":
            for i, model in enumerate(models):
                if built["zeros"][model]:
                    ax.annotate(f"{built['zeros'][model]} zero", xy=(i, 1.0),
                                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                                fontsize=style.font_size - 1.5, color=TEXT_SECONDARY)
        smoke_annotation(fig, style, built["smoke_used"])
        fig.tight_layout()
    return fig


def make_panel(data_dir: Path, out_dir: Path, *, style: Optional[Style] = None,
               dataset: Optional[str] = None, protocol: Optional[str] = None,
               seed: Optional[str] = None, yscale: str = "linear", allow_smoke: bool = False,
               argv: Optional[List[str]] = None,
               command: Optional[str] = None) -> PanelResult:
    style = style or Style(width_in=3.5, height_in=2.6)
    data = collected(data_dir)
    meta = data["metadata"]
    built = build(data["runs"], data["samples"], model_order(meta), meta, dataset=dataset,
                  protocol=protocol, seed=seed, allow_smoke=allow_smoke, yscale=yscale)
    fig = draw(built, style, yscale=yscale)
    metadata = {
        "panel": STEM, "source_type": "local run",
        "target": ("battery lifetime / cycle life (whole-cell RULLabelAnnotator); "
                   "not remaining useful life"),
        "quantity": "absolute_error = |predicted - observed| per test cell",
        "unit": built["unit"],
        "dataset": built["column"][0], "protocol": built["column"][1],
        "seed_selection": (f"--seed {seed} for every model" if seed is not None
                           else "each model has exactly one run in this column"),
        "seeds": {m: r["seed"] for m, r in built["chosen"].items()},
        "n_cells_per_model": built["n_cells"],
        "cell_coverage": "identical test-cell ID sets across models (checked)",
        "display": built["mode"], "display_reason": built["mode_reason"],
        "box_definition": "median, IQR box, whiskers at 1.5 IQR; outliers drawn, never clipped",
        "yscale": yscale,
        "log_zero_strategy": ("zero errors counted, labelled above the model, and not drawn "
                              "(no epsilon added); box/violin use the positive errors only"
                              if yscale == "log" else "not applicable (linear axis)"),
        "zero_error_counts": built["zeros"],
        "non_finite": "pairs with non-finite observed/predicted were dropped by collect and are "
                      "counted in runs.csv omitted_nonfinite",
        "models": built["models"],
        "excluded_models": built["excluded_models"],
        "runs": run_provenance(built["kept"]),
        "excluded_smoke_entries": [r["entry_id"] for r in built["excluded_smoke"]],
        "smoke_included": built["smoke_used"],
        "smoke_status_not_recorded": unrecorded_smoke(built["kept"]),
        "collected_from": str(data_dir),
        "selection_sha256": meta.get("selection_sha256"),
        "command": command or command_line(argv, "scripts.paper_figures.fig3e_lifetime_error_distribution"),
    }
    outputs = save_panel(fig, out_dir, STEM, style, built["points"], SOURCE_COLUMNS, metadata)
    note = "; ".join(f"excluded {m}: {why}" for m, why in built["excluded_models"].items())
    return PanelResult(STEM, CREATED, outputs, note)


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    args = parser.parse_args(argv)
    try:
        result = make_panel(args.data, args.output,
                            style=style_from_args(args, width=3.5, height=2.6),
                            dataset=args.dataset, protocol=args.protocol, seed=args.seed,
                            yscale=args.yscale, allow_smoke=args.allow_smoke,
                            argv=argv if argv is not None else sys.argv[1:])
    except PanelSkip as skip:
        return report(skip)
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
