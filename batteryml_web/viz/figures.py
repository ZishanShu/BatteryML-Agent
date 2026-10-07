"""Arrays -> ``matplotlib.figure.Figure``. Pure: no I/O, no global state.

Every builder here takes plain numpy arrays and a :class:`~batteryml_web.viz.style.Theme`
and returns a detached ``Figure``. None of them reads a file, writes a file, or
touches ``matplotlib.rcParams`` — two figures built in the same process cannot
influence each other, and each builder is checkable by calling it.

``matplotlib.pyplot`` is deliberately never imported: pyplot keeps a global
figure registry, which leaks figures between Streamlit reruns and is not safe
to touch from more than one thread.

Metric values are **arguments**, never computations. The one arithmetic these
builders do is on the values handed to them (sorting, normalising a ramp
position, a min/max for the axis range); the caller supplies residuals and
errors, and the caller got the aggregate metrics from BatteryML's evaluator
(``CLAUDE.md`` §3).

Two stated deviations from the reference mark specs, both because the target is
a 300 dpi print figure rather than a screen:

* **Square bar ends.** A 4 px rounded data-end is 0.3 mm at 300 dpi — invisible,
  and it moves the visual tip of the bar away from the value it encodes.
* **No hover layer.** These are static artifacts. Interaction belongs to the
  Streamlit page (``VIZ-06``).
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .style import (
    ANNOTATION_SIZE_PT,
    BAR_HEIGHT,
    HAIRLINE_PT,
    LABEL_SIZE_PT,
    LINE_WIDTH_PT,
    MARKER_AREA_PT2,
    MARKER_RING_WIDTH_PT,
    SUBTITLE_SIZE_PT,
    TITLE_SIZE_PT,
    Theme,
    add_titles,
    sequential_color,
    style_axes,
)


class VizFigureError(ValueError):
    """Inputs that cannot be drawn: mismatched lengths, empty, or non-finite."""


def _check_pair(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str = "y_true",
    second_name: str = "y_pred",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two same-length numeric arrays and return them as float64 1-D.

    Args:
        first: the array plotted on x.
        second: the array plotted on y.
        first_name: what to call ``first`` in an error message.
        second_name: what to call ``second`` in an error message.

    Raises:
        VizFigureError: on an empty input, a length mismatch, or a non-finite
            value.
    """
    first_arr = np.asarray(first, dtype=np.float64).reshape(-1)
    second_arr = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_arr.size == 0:
        raise VizFigureError(f"no data points: {first_name} is empty.")
    if first_arr.shape != second_arr.shape:
        raise VizFigureError(
            f"{first_name} has {first_arr.size} points but {second_name} has "
            f"{second_arr.size}. They must be the same length."
        )
    if not np.all(np.isfinite(first_arr)) or not np.all(np.isfinite(second_arr)):
        raise VizFigureError(
            f"{first_name} or {second_name} contains NaN or inf. The loader "
            "inverse-transforms and rejects non-finite values, so reaching a figure "
            "builder with one means the arrays did not come from "
            "load_prediction_artifact."
        )
    return first_arr, second_arr


def _new_figure(theme: Theme, size: tuple[float, float], dpi: float) -> Figure:
    """A blank figure painted with the theme's surface."""
    fig = Figure(figsize=size, dpi=dpi)
    fig.patch.set_facecolor(theme.surface)
    return fig


def format_metric_value(value: float) -> str:
    """Format a metric for display at four significant digits.

    The full-precision value is preserved in ``figure_manifest.json`` and in
    ``metrics.csv``; this is the readable form for a tile, not the record.

    Args:
        value: the metric as recorded by BatteryML's evaluator.

    Returns:
        A thousands-separated string, e.g. ``"1,235"`` or ``"0.1319"``.
    """
    return f"{float(value):,.4g}"


def parity_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    theme: Theme,
    title: str,
    subtitle: str | None = None,
    unit_label: str = "label",
    band_fraction: float = 0.10,
    annotation: str | None = None,
    cell_ids: list[str] | None = None,
    n_labelled: int = 3,
    dpi: float = 300.0,
    limits: tuple[float, float] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    marker_area: float | None = None,
    alpha: float = 1.0,
    ring: bool = True,
) -> Figure:
    """Predicted against measured, on a shared scale with the identity line.

    The reader's job is to see how far the cloud departs from ``y = x``, so the
    axes share one range and one aspect: a parity plot with unequal axes
    misleads about the size of the departure.

    Args:
        y_true: measured labels, in the space the metrics are in.
        y_pred: predictions, same space and length.
        theme: colours.
        title: headline naming what is plotted.
        subtitle: provenance line — dataset, model, seed.
        unit_label: the label's unit, e.g. ``"cycles"``. Used on both axes.
        band_fraction: half-width of the shaded tolerance band as a fraction of
            the measured value; ``0`` omits it. Skipped automatically when the
            data reaches zero or below, where a relative band is meaningless.
        annotation: a short line placed inside the axes, e.g. the recorded RMSE.
            Displayed verbatim — this function never computes a metric.
        cell_ids: one ID per point, used to direct-label the largest residuals.
        n_labelled: how many extreme points to direct-label. Labels are sparing
            by design; a value on every point goes unread.
        dpi: raster resolution.
        limits: a shared ``(low, high)`` viewport for both axes. Changes only
            what is visible — every point is still drawn, none is clipped
            from the data — so a zoomed view must say how many points fall
            outside it. Default: the data's own padded range.
        x_label / y_label: full axis labels; default ``Measured (<unit>)`` /
            ``Predicted (<unit>)``.
        marker_area / alpha / ring: marker density for a cloud of thousands
            of points. Defaults are the single-series spec.

    Returns:
        The figure. The caller saves it.

    Raises:
        VizFigureError: for empty, mismatched or non-finite inputs.
    """
    true_arr, pred_arr = _check_pair(y_true, y_pred)
    if cell_ids is not None and len(cell_ids) != true_arr.size:
        raise VizFigureError(
            f"cell_ids has {len(cell_ids)} entries but there are {true_arr.size} points."
        )

    fig = _new_figure(theme, (4.6, 4.9), dpi)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.16, right=0.965, top=0.855, bottom=0.115)
    style_axes(ax, theme)

    if limits is not None:
        lo, hi = float(limits[0]), float(limits[1])
    else:
        low = float(min(true_arr.min(), pred_arr.min()))
        high = float(max(true_arr.max(), pred_arr.max()))
        span = high - low
        pad = span * 0.05 if span > 0 else max(abs(high), 1.0) * 0.05
        lo, hi = low - pad, high + pad
    line = np.array([lo, hi], dtype=np.float64)

    if band_fraction > 0 and lo > 0:
        ax.fill_between(
            line,
            line * (1.0 - band_fraction),
            line * (1.0 + band_fraction),
            color=theme.series,
            alpha=0.10,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            [], [],
            color=theme.series, alpha=0.35, linewidth=LINE_WIDTH_PT,
            label=f"±{band_fraction:.0%} of measured",
        )

    ax.plot(
        line, line,
        color=theme.baseline,
        linewidth=LINE_WIDTH_PT,
        solid_capstyle="round",
        zorder=2,
        label="perfect prediction (y = x)",
    )
    ax.scatter(
        true_arr, pred_arr,
        s=MARKER_AREA_PT2 if marker_area is None else marker_area,
        color=theme.series,
        alpha=alpha,
        edgecolors=theme.surface if ring else "none",
        linewidths=MARKER_RING_WIDTH_PT if ring else 0.0,
        zorder=3,
    )

    if cell_ids and n_labelled > 0:
        residual = np.abs(pred_arr - true_arr)
        for index in np.argsort(residual)[::-1][: min(n_labelled, residual.size)]:
            ax.annotate(
                str(cell_ids[int(index)]),
                xy=(float(true_arr[index]), float(pred_arr[index])),
                xytext=(5.0, 4.0),
                textcoords="offset points",
                fontsize=ANNOTATION_SIZE_PT,
                color=theme.text_secondary,
                zorder=4,
            )

    if annotation:
        ax.annotate(
            annotation,
            xy=(0.03, 0.965),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=ANNOTATION_SIZE_PT,
            color=theme.text_secondary,
            zorder=5,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label or f"Measured ({unit_label})", color=theme.text_secondary, fontsize=LABEL_SIZE_PT)
    ax.set_ylabel(y_label or f"Predicted ({unit_label})", color=theme.text_secondary, fontsize=LABEL_SIZE_PT)
    add_titles(ax, theme, title=title, subtitle=subtitle)

    legend = ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=ANNOTATION_SIZE_PT,
        handlelength=1.6,
    )
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)
    return fig


def residual_figure(
    y_true: np.ndarray,
    residuals: np.ndarray,
    *,
    theme: Theme,
    title: str,
    subtitle: str | None = None,
    unit_label: str = "label",
    annotation: str | None = None,
    dpi: float = 300.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    marker_area: float | None = None,
    alpha: float = 1.0,
    ring: bool = True,
) -> Figure:
    """Signed error against the measured value, on a zero baseline.

    Answers the question a parity plot cannot: is the error a constant offset, a
    fan that widens with the label, or a systematic tilt?

    Args:
        y_true: measured labels.
        residuals: ``y_pred - y_true``, same length. Supplied by the caller
            rather than computed here so the sign convention is stated once.
        theme: colours.
        title: headline.
        subtitle: provenance line.
        unit_label: the label's unit.
        annotation: verbatim line placed in the axes.
        dpi: raster resolution.
        xlim / ylim: viewport only (see :func:`parity_figure`'s ``limits``).
            Default: x autoscaled to every point, y symmetric about zero at
            1.15 × the largest absolute residual, so extremes stay visible.
        x_label / y_label / marker_area / alpha / ring: as for
            :func:`parity_figure`.

    Returns:
        The figure.

    Raises:
        VizFigureError: for empty, mismatched or non-finite inputs.
    """
    true_arr, residual_arr = _check_pair(
        y_true, residuals, first_name="y_true", second_name="residuals"
    )

    fig = _new_figure(theme, (5.4, 3.4), dpi)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.815, bottom=0.155)
    style_axes(ax, theme)

    ax.axhline(
        0.0,
        color=theme.baseline,
        linewidth=LINE_WIDTH_PT,
        zorder=2,
    )
    ax.scatter(
        true_arr, residual_arr,
        s=MARKER_AREA_PT2 if marker_area is None else marker_area,
        color=theme.series,
        alpha=alpha,
        edgecolors=theme.surface if ring else "none",
        linewidths=MARKER_RING_WIDTH_PT if ring else 0.0,
        zorder=3,
    )

    if annotation:
        ax.annotate(
            annotation,
            xy=(0.02, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=ANNOTATION_SIZE_PT,
            color=theme.text_secondary,
        )

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        limit = float(np.max(np.abs(residual_arr)))
        if limit > 0:
            ax.set_ylim(-limit * 1.15, limit * 1.15)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_xlabel(x_label or f"Measured ({unit_label})", color=theme.text_secondary, fontsize=LABEL_SIZE_PT)
    ax.set_ylabel(
        y_label or f"Predicted − measured ({unit_label})",
        color=theme.text_secondary,
        fontsize=LABEL_SIZE_PT,
    )
    add_titles(ax, theme, title=title, subtitle=subtitle)
    return fig


def error_by_cell_figure(
    cell_ids: list[str],
    values: np.ndarray,
    *,
    theme: Theme,
    title: str,
    subtitle: str | None = None,
    value_label: str = "Absolute error",
    unit_label: str = "label",
    n_labelled: int = 5,
    dpi: float = 300.0,
) -> Figure:
    """Per-cell magnitude, sorted worst first.

    Horizontal because a battery test set runs to dozens of named cells and a
    vertical axis of rotated IDs is unreadable. Colour is sequential — the bars
    are one quantity at different magnitudes, not different series — so hue
    reinforces length rather than inventing an identity channel.

    Args:
        cell_ids: one ID per value, already aligned to ``values`` by the loader.
        values: the per-cell quantity, e.g. absolute error. Non-negative.
        theme: colours.
        title: headline.
        subtitle: provenance line.
        value_label: what the bars measure.
        unit_label: its unit.
        n_labelled: how many of the worst bars carry a value at the tip.
        dpi: raster resolution.

    Returns:
        The figure. Its height grows with the number of cells so the bars keep a
        constant pitch rather than being squeezed.

    Raises:
        VizFigureError: on a length mismatch, an empty input, or non-finite
            values.
    """
    value_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if value_arr.size == 0:
        raise VizFigureError("no data points: values is empty.")
    if len(cell_ids) != value_arr.size:
        raise VizFigureError(
            f"cell_ids has {len(cell_ids)} entries but values has {value_arr.size}."
        )
    if not np.all(np.isfinite(value_arr)):
        raise VizFigureError("values contains NaN or inf.")

    order = np.argsort(value_arr)  # ascending; barh draws bottom-up, so worst ends on top
    sorted_values = value_arr[order]
    sorted_ids = [str(cell_ids[int(index)]) for index in order]
    count = sorted_values.size

    height = max(2.6, 0.155 * count + 1.15)
    longest = max((len(name) for name in sorted_ids), default=4)
    left = min(0.34, 0.055 + 0.0125 * longest)
    fig = _new_figure(theme, (5.4, height), dpi)
    ax = fig.add_subplot(111)
    title_band = 0.62 / height
    fig.subplots_adjust(
        left=left,
        right=0.90,
        top=1.0 - title_band,
        bottom=min(0.16, 0.55 / height),
    )
    style_axes(ax, theme, grid_axis="x")

    peak = float(sorted_values.max())
    colors = [
        sequential_color(theme, (value / peak) if peak > 0 else 0.0)
        for value in sorted_values
    ]
    positions = np.arange(count, dtype=np.float64)
    ax.barh(
        positions,
        sorted_values,
        height=BAR_HEIGHT,
        color=colors,
        linewidth=0.0,
        zorder=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(sorted_ids)
    ax.set_ylim(-0.6, count - 0.4)
    ax.set_xlim(0.0, peak * 1.18 if peak > 0 else 1.0)

    for offset in range(min(n_labelled, count)):
        index = count - 1 - offset
        ax.annotate(
            format_metric_value(sorted_values[index]),
            xy=(float(sorted_values[index]), float(positions[index])),
            xytext=(4.0, 0.0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=ANNOTATION_SIZE_PT,
            color=theme.text_secondary,
            zorder=4,
        )

    ax.set_xlabel(f"{value_label} ({unit_label})", color=theme.text_secondary, fontsize=LABEL_SIZE_PT)
    add_titles(ax, theme, title=title, subtitle=subtitle)
    return fig


def metric_summary_figure(
    scores: dict[str, float],
    *,
    theme: Theme,
    title: str,
    subtitle: str | None = None,
    metric_units: dict[str, str] | None = None,
    note: str | None = None,
    dpi: float = 300.0,
) -> Figure:
    """A row of stat tiles — one per recorded metric.

    Deliberately **not** a bar chart. RMSE and MAE are in the label's unit while
    MAPE is a dimensionless ratio; putting them on one scale would be a
    two-measure chart with a single axis, which misstates their relative size.
    Three headline numbers are a tile row.

    The values are rendered exactly as recorded by BatteryML's evaluator. Note
    that BatteryML's MAPE is a **ratio**, not a percentage
    (``batteryml/data/databundle.py:106``) — pass a unit string that says so
    rather than multiplying it here.

    Args:
        scores: metric name -> value, verbatim from the artifact.
        theme: colours.
        title: headline.
        subtitle: provenance line.
        metric_units: metric name -> unit caption. Missing entries render blank.
        note: a footer line, e.g. where the numbers came from.
        dpi: raster resolution.

    Returns:
        The figure.

    Raises:
        VizFigureError: if ``scores`` is empty or holds a non-numeric value.
    """
    if not scores:
        raise VizFigureError("scores is empty; there is nothing to display.")
    units = metric_units or {}
    items: list[tuple[str, float]] = []
    for name, value in scores.items():
        try:
            items.append((str(name), float(value)))
        except (TypeError, ValueError) as exc:
            raise VizFigureError(
                f"metric {name!r} has non-numeric value {value!r}: {exc}."
            ) from exc

    width = max(4.0, 2.0 * len(items))
    fig = _new_figure(theme, (width, 1.95), dpi)
    fig.text(
        0.03, 0.93, title,
        ha="left", va="top",
        color=theme.text_primary, fontsize=TITLE_SIZE_PT, fontweight="semibold",
    )
    if subtitle:
        fig.text(
            0.03, 0.78, subtitle,
            ha="left", va="top",
            color=theme.text_secondary, fontsize=SUBTITLE_SIZE_PT,
        )
    if note:
        fig.text(
            0.03, 0.06, note,
            ha="left", va="bottom",
            color=theme.text_muted, fontsize=ANNOTATION_SIZE_PT,
        )

    slot = (1.0 - 0.06) / len(items)
    for position, (name, value) in enumerate(items):
        x = 0.03 + slot * position
        fig.text(
            x, 0.545, name,
            ha="left", va="bottom",
            color=theme.text_secondary, fontsize=SUBTITLE_SIZE_PT,
        )
        fig.text(
            x, 0.245, format_metric_value(value),
            ha="left", va="bottom",
            color=theme.text_primary, fontsize=20.0, fontweight="semibold",
        )
        unit = units.get(name)
        if unit:
            fig.text(
                x, 0.135, unit,
                ha="left", va="bottom",
                color=theme.text_muted, fontsize=ANNOTATION_SIZE_PT,
            )
        if position:
            fig.add_artist(
                _tile_divider(x - slot * 0.06, theme),
            )
    return fig


def _tile_divider(x: float, theme: Theme):
    """A hairline separating two stat tiles.

    Returns a Line2D with **no transform set**, so ``Figure.add_artist`` assigns
    the figure transform and the coordinates are read as figure fractions.
    Passing ``transform=None`` explicitly would mark the transform as set and
    leave the line in display coordinates, i.e. a half-pixel mark in the corner.
    """
    from matplotlib.lines import Line2D

    return Line2D(
        [x, x], [0.16, 0.66],
        color=theme.gridline,
        linewidth=HAIRLINE_PT,
    )


def soh_vs_cycle_figure(
    cell_ids: list[str],
    cycle_indices: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    theme: Theme,
    title: str,
    subtitle: str | None = None,
    unit_label: str = "SOH (ratio)",
    max_cells: int = 6,
    dpi: float = 300.0,
) -> Figure:
    """Observed and estimated SOH across the cycles of each test cell.

    **This is not a trajectory forecast.** Every point plotted is an estimate
    *at an observed cycle*, made from information available at that cycle. The
    line joining them is a reading aid across many independent estimates, not
    a prediction of what happens next. Calling it a forecast would claim a
    task this route does not perform.

    Each cell gets one hue from the sequential ramp: observed as a solid line,
    estimated as markers on top of it, so agreement reads as markers sitting
    on the line. Only ``max_cells`` cells are drawn — with more, the panel
    turns into a thicket and nothing is legible — and the subtitle says how
    many of how many are shown, so a reader is never left to assume it is all
    of them.

    Args:
        cell_ids: the cell each point belongs to, aligned to the arrays.
        cycle_indices: the observed cycle each point was estimated at.
        observed: measured SOH, on the canonical ratio scale.
        predicted: estimated SOH, same scale.
        theme: colours.
        title: headline.
        subtitle: provenance line; the cell count is appended to it.
        unit_label: axis unit. A ratio, not a percentage.
        max_cells: how many cells to draw.
        dpi: raster resolution.

    Returns:
        The figure. Never written to disk here.

    Raises:
        VizFigureError: if the four inputs disagree in length or are empty.
    """
    cycle_arr = np.asarray(cycle_indices, dtype=float).reshape(-1)
    observed_arr = np.asarray(observed, dtype=float).reshape(-1)
    predicted_arr = np.asarray(predicted, dtype=float).reshape(-1)
    lengths = {len(cell_ids), cycle_arr.size, observed_arr.size, predicted_arr.size}
    if len(lengths) != 1:
        raise VizFigureError(
            "soh_vs_cycle_figure needs cell_ids, cycle_indices, observed and "
            f"predicted to be the same length; got {len(cell_ids)}, "
            f"{cycle_arr.size}, {observed_arr.size}, {predicted_arr.size}. "
            "Each point must name the cell and cycle it was estimated at."
        )
    if observed_arr.size == 0:
        raise VizFigureError(
            "soh_vs_cycle_figure received no samples. A cycle-level artifact "
            "is required; a whole-cell run has no cycle to plot against."
        )

    grouped: dict[str, list[int]] = {}
    for position, cell_id in enumerate(cell_ids):
        grouped.setdefault(str(cell_id), []).append(position)
    # Most-sampled cells first, so the drawn subset is the informative one
    # rather than whichever happened to sort first.
    order = sorted(grouped, key=lambda key: (-len(grouped[key]), key))
    shown = order[: max(1, int(max_cells))]

    fig = _new_figure(theme, (6.4, 4.2), dpi)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.10, right=0.80, top=0.855, bottom=0.135)
    style_axes(ax, theme)

    for rank, cell_id in enumerate(shown):
        positions = sorted(grouped[cell_id], key=lambda i: cycle_arr[i])
        x = cycle_arr[positions]
        y_true = observed_arr[positions]
        y_pred = predicted_arr[positions]
        color = sequential_color(
            theme, rank / max(1, len(shown) - 1) if len(shown) > 1 else 0.5
        )
        ax.plot(
            x, y_true,
            color=color,
            linewidth=LINE_WIDTH_PT,
            zorder=2,
            label=str(cell_id),
        )
        ax.scatter(
            x, y_pred,
            s=MARKER_AREA_PT2 * 0.6,
            color=color,
            edgecolors=theme.surface,
            linewidths=MARKER_RING_WIDTH_PT,
            zorder=3,
        )

    ax.set_xlabel("Observed cycle", color=theme.text_secondary, fontsize=LABEL_SIZE_PT)
    ax.set_ylabel(unit_label, color=theme.text_secondary, fontsize=LABEL_SIZE_PT)

    detail = f"{len(shown)} of {len(order)} test cell(s); line = measured, markers = estimated"
    add_titles(
        ax, theme,
        title=title,
        subtitle=f"{subtitle} · {detail}" if subtitle else detail,
    )

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=ANNOTATION_SIZE_PT,
        handlelength=1.6,
        title="Cell",
    )
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)
    if legend.get_title() is not None:
        legend.get_title().set_color(theme.text_secondary)
        legend.get_title().set_fontsize(ANNOTATION_SIZE_PT)
    return fig


def soh_cell_figure(
    cycles: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    theme: Theme,
    cell_id: str,
    title: str = "SOH estimates across observed cycles",
    subtitle: str | None = None,
    unit_label: str = "ratio",
    annotation: str | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: float = 300.0,
) -> Figure:
    """Observed and estimated SOH for **one** cell, at its recorded cycles.

    One cell, so the legend means exactly one thing: observed is a solid
    neutral line, estimated a dashed accent line, both with a marker at every
    recorded point. Points are drawn in the order of the recorded cycle
    index, as stored — nothing is smoothed, interpolated, extrapolated or
    made monotonic, and the y range is the data's own (no forced [0, 1]).

    **Not a forecast.** Each estimate was made at an observed cycle from that
    cycle's data; the line joining them is a reading aid.

    Raises:
        VizFigureError: empty, mismatched, unsorted or non-finite input.
    """
    cycle_arr = np.asarray(cycles, dtype=np.float64).reshape(-1)
    true_arr, pred_arr = _check_pair(observed, predicted,
                                     first_name="observed", second_name="predicted")
    if cycle_arr.size != true_arr.size:
        raise VizFigureError(
            f"cycles has {cycle_arr.size} entries but there are {true_arr.size} points."
        )
    if not np.all(np.isfinite(cycle_arr)):
        raise VizFigureError("cycles contains NaN or inf.")
    if cycle_arr.size > 1 and np.any(np.diff(cycle_arr) < 0):
        raise VizFigureError(
            "cycles must be sorted ascending with observed/predicted reordered "
            "together; pass the series from soh_cell_series()."
        )

    fig = _new_figure(theme, (5.4, 3.6), dpi)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.82, bottom=0.15)
    style_axes(ax, theme)
    marker = 3.2 if cycle_arr.size <= 300 else 0.0
    ax.plot(
        cycle_arr, true_arr,
        color=theme.text_secondary, linewidth=LINE_WIDTH_PT, linestyle="-",
        marker="o" if marker else None, markersize=marker,
        zorder=2, label="Observed SOH",
    )
    ax.plot(
        cycle_arr, pred_arr,
        color=theme.series, linewidth=LINE_WIDTH_PT, linestyle="--",
        marker="o" if marker else None, markersize=marker,
        zorder=3, label="Estimated SOH",
    )
    if ylim is not None:
        # Viewport only: every point is still drawn; the caller counts and
        # states how many fall outside.
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if annotation:
        ax.annotate(
            annotation, xy=(0.02, 0.04), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=ANNOTATION_SIZE_PT,
            color=theme.text_secondary, zorder=5,
        )
    ax.set_xlabel("Observed cycle index", color=theme.text_secondary, fontsize=LABEL_SIZE_PT)
    ax.set_ylabel(f"SOH ({unit_label})", color=theme.text_secondary, fontsize=LABEL_SIZE_PT)
    add_titles(
        ax, theme, title=title,
        subtitle=subtitle if subtitle is not None else f"Cell {cell_id}",
    )
    legend = ax.legend(loc="best", frameon=False, fontsize=ANNOTATION_SIZE_PT,
                       handlelength=2.4)
    for text in legend.get_texts():
        text.set_color(theme.text_secondary)
    return fig

