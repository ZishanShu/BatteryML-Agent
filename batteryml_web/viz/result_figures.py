"""One figure path for the website and the export: view -> panel data -> figure.

The page and ``scripts/export_result_figures.py`` both call :func:`build_panels`
on the same view dict (``agent/result_views.py``). Each panel carries the exact
arrays handed to its figure builder, so the CSV an export writes is, by
construction, what was drawn — there is no second computation to drift.

Rules this module holds:

* **Nothing is recomputed as a metric.** The saved scores are displayed
  verbatim. The per-cell table (:func:`per_cell_errors`) is a *display
  diagnostic*: each cell's mean absolute residual and root-mean-square
  residual over its plotted samples. It weights every cell equally, while the
  saved scores pool every sample, so the two answer different questions and
  one is never substituted for the other.
* **One cell, chosen by a stated rule.** :data:`DEFAULT_CELL_RULE` — first
  cell ID in ascending text order — does not look at prediction quality. A
  caller may pass another *real* cell ID; an unknown one is refused.
* **Zoom moves the viewport only.** :func:`zoom_window` returns limits and how
  many plotted points fall outside them; the arrays are never clipped, and
  the full-range panel is always built beside the zoomed one.
* **Recorded order is kept.** Within a cell, records are sorted by the
  recorded cycle index with observed, predicted, residual and artifact row
  moving together. A repeated (cell, cycle) is reported, never averaged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: How the default SOH cell is chosen. Deliberately outcome-blind.
DEFAULT_CELL_RULE = (
    "first cell ID in ascending text order among the plotted samples "
    "(independent of prediction error)"
)
EXPLICIT_CELL_RULE = "cell ID given explicitly by the user"

ZOOM_QUANTILES: Tuple[float, float] = (0.01, 0.99)
ZOOM_RULE = (
    "viewport = 1st to 99th percentile of the plotted values, padded by 5 % "
    "of that span; only the visible range changes, no value is clipped or "
    "removed, and points outside the viewport are counted"
)
#: Above this many points a scatter uses small translucent markers. Nothing
#: is subsampled.
DENSE_POINTS = 400

PER_CELL_NOTE = (
    "Display diagnostic, not the run's metric: each cell's mean absolute "
    "residual and root-mean-square residual over its plotted samples. The "
    "saved scores pool all samples (each sample weighs the same); a mean over "
    "these rows would weigh each cell the same instead. Neither replaces the "
    "other."
)


class FigureDataError(ValueError):
    """A request the view cannot support, e.g. a cell ID it does not hold."""


# --- small numeric helpers (plain Python, so the data layer needs no numpy) --

def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise FigureDataError("no values to take a quantile of.")
    position = (len(sorted_values) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _padded(low: float, high: float, fraction: float = 0.05) -> Tuple[float, float]:
    span = high - low
    pad = span * fraction if span > 0 else max(abs(high), 1.0) * fraction
    return (low - pad, high + pad)


def full_limits(*series: Sequence[float]) -> Tuple[float, float]:
    """A padded range containing every value of every series."""
    values = [float(v) for s in series for v in s]
    if not values:
        raise FigureDataError("no values to plot.")
    return _padded(min(values), max(values))


def zoom_window(
    x: Sequence[float],
    y: Sequence[float],
    *,
    quantiles: Tuple[float, float] = ZOOM_QUANTILES,
) -> Dict[str, Any]:
    """A shared viewport on the central cluster of a parity cloud.

    Returns ``{"limits", "outside", "inside", "rule", "quantiles"}``.
    ``outside`` counts the points with **either** coordinate beyond the
    limits — they are still in the arrays and in the full-range panel.
    """
    pooled = sorted(float(v) for v in list(x) + list(y))
    low = _quantile(pooled, quantiles[0])
    high = _quantile(pooled, quantiles[1])
    lo, hi = _padded(low, high)
    outside = sum(
        1 for a, b in zip(x, y) if not (lo <= a <= hi and lo <= b <= hi)
    )
    return {
        "limits": [lo, hi],
        "outside": outside,
        "inside": len(list(x)) - outside,
        "rule": ZOOM_RULE,
        "quantiles": list(quantiles),
    }


def residual_limits(observed: Sequence[float], residuals: Sequence[float]) -> Dict[str, Any]:
    """The full-range residual viewport: every point visible."""
    biggest = max((abs(float(r)) for r in residuals), default=0.0)
    ylim = [-biggest * 1.15, biggest * 1.15] if biggest > 0 else [-1.0, 1.0]
    return {"xlim": list(full_limits(observed)), "ylim": ylim, "outside": 0}


def residual_zoom(
    observed: Sequence[float],
    residuals: Sequence[float],
    *,
    quantiles: Tuple[float, float] = ZOOM_QUANTILES,
) -> Dict[str, Any]:
    """Residual viewport on the central cluster: x from observed, y from residuals."""
    xs = sorted(float(v) for v in observed)
    ys = sorted(float(v) for v in residuals)
    xlim = _padded(_quantile(xs, quantiles[0]), _quantile(xs, quantiles[1]))
    ylo, yhi = _padded(_quantile(ys, quantiles[0]), _quantile(ys, quantiles[1]))
    # Keep the zero line in view: it is the reference the plot is read against.
    ylim = (min(ylo, 0.0), max(yhi, 0.0))
    outside = sum(
        1 for a, r in zip(observed, residuals)
        if not (xlim[0] <= a <= xlim[1] and ylim[0] <= r <= ylim[1])
    )
    return {
        "xlim": list(xlim), "ylim": list(ylim), "outside": outside,
        "inside": len(list(observed)) - outside, "rule": ZOOM_RULE,
        "quantiles": list(quantiles),
    }


# --- cells -------------------------------------------------------------------

def cell_choices(view: Dict[str, Any]) -> List[str]:
    """The real cell IDs among the plotted samples, ascending."""
    return sorted({str(cell) for cell in (view.get("cell_ids") or [])})


def default_cell(view: Dict[str, Any]) -> Optional[str]:
    choices = cell_choices(view)
    return choices[0] if choices else None


def soh_cell_series(view: Dict[str, Any], cell_id: str) -> Dict[str, Any]:
    """One cell's records, sorted by recorded cycle, every field together.

    Raises:
        FigureDataError: the view holds no such cell. Never substitutes
            another one.
    """
    cell_ids = [str(c) for c in (view.get("cell_ids") or [])]
    rows = [j for j, cell in enumerate(cell_ids) if cell == str(cell_id)]
    if not rows:
        raise FigureDataError(
            f"cell {cell_id!r} has no plotted sample in this result. Choose one "
            f"of its {len(set(cell_ids))} cell IDs."
        )
    cycles_all = view.get("cycle_indices") or []
    positions = view.get("row_positions") or list(range(len(cell_ids)))
    # Stable sort on (cycle, original order): complete records move together.
    rows.sort(key=lambda j: (int(cycles_all[j]), j))
    cycles = [int(cycles_all[j]) for j in rows]
    counts: Dict[int, int] = {}
    for cycle in cycles:
        counts[cycle] = counts.get(cycle, 0) + 1
    return {
        "cell_id": str(cell_id),
        "cycles": cycles,
        "observed": [view["observed"][j] for j in rows],
        "predicted": [view["predicted"][j] for j in rows],
        "residuals": [view["residuals"][j] for j in rows],
        "artifact_rows": [positions[j] for j in rows],
        "duplicates": [
            {"cycle_index": cycle, "count": count}
            for cycle, count in sorted(counts.items()) if count > 1
        ],
    }


def per_cell_errors(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-cell display diagnostics over the plotted samples (see module doc)."""
    groups: Dict[str, List[float]] = {}
    for cell, residual in zip(view.get("cell_ids") or [], view.get("residuals") or []):
        groups.setdefault(str(cell), []).append(float(residual))
    rows = []
    for cell in sorted(groups):
        values = groups[cell]
        count = len(values)
        rows.append({
            "cell_id": cell,
            "samples": count,
            "mean_abs_residual": sum(abs(v) for v in values) / count,
            "rms_residual": math.sqrt(sum(v * v for v in values) / count),
        })
    return rows


def plotted_rows(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every plotted sample, in artifact order, as written to the CSV."""
    count = int(view.get("count") or 0)
    positions = view.get("row_positions") or list(range(count))
    cells = view.get("cell_ids") or [None] * count
    cycles = view.get("cycle_indices") or [None] * count
    return [
        {
            "artifact_row": positions[i],
            "cell_id": cells[i],
            "cycle_index": cycles[i],
            "observed": view["observed"][i],
            "predicted": view["predicted"][i],
            "residual_predicted_minus_observed": view["residuals"][i],
        }
        for i in range(count)
    ]


# --- panels ------------------------------------------------------------------

def _labels(target: Dict[str, Any]) -> Dict[str, str]:
    quantity = str(target.get("quantity") or "Target value")
    unit = str(target.get("unit") or "unit not recorded")
    # Lower-case a leading word only when it is an ordinary word ("Battery"),
    # never an acronym ("SOH", "RUL/lifetime").
    first = quantity.split()[0] if quantity.split() else quantity
    lower = quantity if first[:2].isupper() else quantity[0].lower() + quantity[1:]
    soh = target.get("kind") == "soh"
    return {
        "x": f"Observed {lower} ({unit})",
        "y": f"{'Estimated' if soh else 'Predicted'} {lower} ({unit})",
        "residual_y": f"Predicted − observed ({target.get('residual_unit') or unit})",
        "parity_title": f"Observed vs {'estimated' if soh else 'predicted'} {lower}",
        "unit": unit,
    }


def _granularity(view: Dict[str, Any]) -> str:
    """Short enough for one subtitle line: counts and what one point is."""
    target = view.get("target") or {}
    cells = len(cell_choices(view))
    text = f"{int(view.get('count') or 0):,} points"
    if cells:
        text += f" · {cells} cells"
    return f"{text} · {target.get('granularity') or 'one point per stored row'}"


def _metric_line(saved_metrics: Optional[Dict[str, Any]], unit: str) -> Optional[str]:
    if not saved_metrics:
        return None
    parts = []
    for key in ("RMSE", "MAE"):
        value = saved_metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            parts.append(f"{key} {value:,.4g}")
    if not parts:
        return None
    return "Saved by BatteryML: " + ", ".join(parts) + f" ({unit})"


def _annotation(*lines: Optional[str]) -> Optional[str]:
    """In-axes text block; the badge goes here so it is never clipped."""
    kept = [line for line in lines if line]
    return "\n".join(kept) if kept else None


def build_panels(
    view: Dict[str, Any],
    *,
    cell_id: Optional[str] = None,
    saved_metrics: Optional[Dict[str, Any]] = None,
    badge: Optional[str] = None,
    include_zoom: bool = True,
    dpi: float = 160.0,
    theme_name: str = "light",
) -> Dict[str, Dict[str, Any]]:
    """Figures and their exact plotted data for one available view.

    Returns ``{panel_id: {"figure", "data", "meta"}}``. Panel ids:
    ``soh_cycle`` (SOH only), ``parity``, ``residuals`` and, with
    ``include_zoom``, ``soh_cycle_zoom`` (SOH only), ``parity_zoom`` and
    ``residuals_zoom``. ``data`` holds
    the arrays passed to the builder; ``meta`` holds limits, counts and the
    rules applied.

    Raises:
        FigureDataError: the view is unavailable, or ``cell_id`` is not a
            cell it holds.
    """
    from .figures import parity_figure, residual_figure, soh_cell_figure
    from .style import MARKER_AREA_PT2, get_theme

    if not view.get("available"):
        raise FigureDataError(str(view.get("reason") or "This view is unavailable."))
    theme = get_theme(theme_name)
    target = view.get("target") or {}
    labels = _labels(target)
    observed, predicted, residuals = view["observed"], view["predicted"], view["residuals"]
    dense = len(observed) > DENSE_POINTS
    marks = (
        {"marker_area": MARKER_AREA_PT2 * 0.24, "alpha": 0.35, "ring": False}
        if dense else {}
    )
    panels: Dict[str, Dict[str, Any]] = {}

    if target.get("kind") == "soh" and view.get("cycle_indices"):
        chosen = str(cell_id) if cell_id is not None else default_cell(view)
        rule = EXPLICIT_CELL_RULE if cell_id is not None else DEFAULT_CELL_RULE
        if chosen is None:
            raise FigureDataError("This SOH result names no cell.")
        series = soh_cell_series(view, chosen)
        subtitle = f"Cell {chosen} · {len(series['cycles'])} observed cycles"
        panels["soh_cycle"] = {
            "figure": soh_cell_figure(
                series["cycles"], series["observed"], series["predicted"],
                theme=theme, cell_id=chosen, subtitle=subtitle,
                unit_label=labels["unit"], annotation=_annotation(badge), dpi=dpi,
            ),
            "data": series,
            "meta": {"cell_id": chosen, "cell_rule": rule,
                     "points": len(series["cycles"]),
                     "duplicates": series["duplicates"]},
        }
        if include_zoom:
            values = sorted(series["observed"] + series["predicted"])
            ylim = _padded(_quantile(values, ZOOM_QUANTILES[0]),
                           _quantile(values, ZOOM_QUANTILES[1]))
            outside = sum(
                1 for o, q in zip(series["observed"], series["predicted"])
                if not (ylim[0] <= o <= ylim[1] and ylim[0] <= q <= ylim[1])
            )
            panels["soh_cycle_zoom"] = {
                "figure": soh_cell_figure(
                    series["cycles"], series["observed"], series["predicted"],
                    theme=theme, cell_id=chosen,
                    title="SOH estimates across observed cycles — zoomed view",
                    subtitle=(f"Cell {chosen} · {outside} of "
                              f"{len(series['cycles'])} cycles outside this viewport"),
                    unit_label=labels["unit"], annotation=_annotation(badge),
                    ylim=ylim, dpi=dpi,
                ),
                "data": series,
                "meta": {"cell_id": chosen, "cell_rule": rule, "view": "zoomed",
                         "ylim": list(ylim), "outside": outside,
                         "inside": len(series["cycles"]) - outside,
                         "rule": ZOOM_RULE, "quantiles": list(ZOOM_QUANTILES),
                         "points": len(series["cycles"])},
            }

    full = full_limits(observed, predicted)
    subtitle = _granularity(view)
    panels["parity"] = {
        "figure": parity_figure(
            observed, predicted, theme=theme, title=labels["parity_title"],
            subtitle=subtitle, unit_label=labels["unit"], band_fraction=0.0,
            annotation=_annotation(badge, _metric_line(saved_metrics, labels["unit"])),
            limits=full, x_label=labels["x"], y_label=labels["y"], dpi=dpi, **marks,
        ),
        "data": {"observed": observed, "predicted": predicted},
        "meta": {"view": "full range", "limits": list(full), "outside": 0,
                 "points": len(observed)},
    }
    full_res = residual_limits(observed, residuals)
    panels["residuals"] = {
        "figure": residual_figure(
            observed, residuals, theme=theme,
            title="Residuals (predicted − observed)", subtitle=subtitle,
            unit_label=labels["unit"], annotation=_annotation(badge),
            xlim=tuple(full_res["xlim"]),
            ylim=tuple(full_res["ylim"]), x_label=labels["x"],
            y_label=labels["residual_y"], dpi=dpi, **marks,
        ),
        "data": {"observed": observed, "residuals": residuals},
        "meta": {"view": "full range", **full_res, "points": len(observed)},
    }
    if include_zoom:
        zoom = zoom_window(observed, predicted)
        panels["parity_zoom"] = {
            "figure": parity_figure(
                observed, predicted, theme=theme,
                title=labels["parity_title"] + " — zoomed view",
                subtitle=f"{zoom['outside']:,} of {len(observed):,} points outside this viewport",
                unit_label=labels["unit"], band_fraction=0.0,
                annotation=_annotation(badge),
                limits=tuple(zoom["limits"]), x_label=labels["x"],
                y_label=labels["y"], dpi=dpi, **marks,
            ),
            "data": {"observed": observed, "predicted": predicted},
            "meta": {"view": "zoomed", **zoom, "points": len(observed)},
        }
        rzoom = residual_zoom(observed, residuals)
        panels["residuals_zoom"] = {
            "figure": residual_figure(
                observed, residuals, theme=theme,
                title="Residuals (predicted − observed) — zoomed view",
                subtitle=f"{rzoom['outside']:,} of {len(observed):,} points outside this viewport",
                unit_label=labels["unit"], annotation=_annotation(badge),
                xlim=tuple(rzoom["xlim"]),
                ylim=tuple(rzoom["ylim"]), x_label=labels["x"],
                y_label=labels["residual_y"], dpi=dpi, **marks,
            ),
            "data": {"observed": observed, "residuals": residuals},
            "meta": {"view": "zoomed", **rzoom, "points": len(observed)},
        }
    return panels
