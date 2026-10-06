"""Theme tokens and axis chrome. No data, no I/O, no matplotlib global state.

Colour values are taken unchanged from the validated reference palette (blue
categorical slot 1, its sequential ramp, and the chart-chrome neutrals). This
package draws **single-hue** figures only — one series plus neutrals — so there
is no categorical adjacency to validate. The moment a figure puts two or more
categorical series on one axis (the cross-model comparison planned as
``VIZ-05``), the palette validator must be run on the hue order before it ships.

Two deliberate deviations from the reference palette, both for reproducibility:

* **Typeface.** The reference specifies the system sans. Figures here keep
  matplotlib's bundled DejaVu Sans instead: a system font resolves differently
  on every machine, which would make byte-identical output impossible and break
  the determinism clause of ``CLAUDE.md`` §8.
* **No hover layer.** These are static raster/vector artifacts for the
  manuscript. The interactive layer belongs to the Streamlit page (``VIZ-06``),
  not to the files quoted by the paper.

Sequential ramps respect the ordinal floor: on the light surface the ramp
starts no lighter than step 250, and on the dark surface it goes no darker than
step 600, so the mark nearest the surface still clears 2:1 contrast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class Theme:
    """Every colour a figure in this package is allowed to use.

    Attributes:
        name: ``"light"`` or ``"dark"``. Recorded in the figure manifest, so a
            reader can tell which surface a figure was drawn against.
        surface: figure and axes background.
        text_primary: titles and stat-tile values.
        text_secondary: subtitles and axis labels.
        text_muted: tick labels and annotations.
        gridline: hairline grid, one step off the surface.
        baseline: axis spines and the parity / zero reference line.
        series: the single accent hue every mark is drawn in.
        sequential: low -> high ramp for magnitude-encoded marks.
        emphasis: reserved for the single worst-case mark. Paired with a text
            label at every use site — never carries meaning by colour alone.
    """

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    gridline: str
    baseline: str
    series: str
    sequential: tuple[str, ...]
    emphasis: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    series="#2a78d6",
    # blue steps 250 -> 700; 250 is the light-surface ordinal floor.
    sequential=("#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#0d366b"),
    emphasis="#d03b3b",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    series="#3987e5",
    # blue steps 100 -> 600; 600 is the dark-surface ordinal floor.
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95"),
    emphasis="#d03b3b",
)

THEMES: dict[str, Theme] = {LIGHT.name: LIGHT, DARK.name: DARK}

# Mark specs from the reference palette, in points rather than CSS pixels.
MARKER_SIZE_PT = 5.0          # marker diameter; >= 8px at 96 dpi
MARKER_AREA_PT2 = MARKER_SIZE_PT ** 2   # matplotlib's scatter ``s`` is an area
MARKER_RING_WIDTH_PT = 1.0    # the surface ring that keeps overlapping dots legible
LINE_WIDTH_PT = 1.5
HAIRLINE_PT = 0.8
BAR_GAP_FRACTION = 0.28       # leftover band width: the surface gap between bars
BAR_HEIGHT = 1.0 - BAR_GAP_FRACTION     # fraction of a slot a bar occupies

TITLE_SIZE_PT = 10.0
SUBTITLE_SIZE_PT = 8.0
LABEL_SIZE_PT = 8.5
TICK_SIZE_PT = 7.5
ANNOTATION_SIZE_PT = 7.5


class UnknownThemeError(ValueError):
    """Raised for a theme name that is not defined in :data:`THEMES`."""


def get_theme(name: str) -> Theme:
    """Return the theme called ``name``.

    Args:
        name: ``"light"`` or ``"dark"``, case-insensitive.

    Returns:
        The matching :class:`Theme`.

    Raises:
        UnknownThemeError: if no such theme exists. The message lists the
            names that do, because a silent fallback to the light theme would
            put a figure on the wrong surface without saying so.
    """
    key = str(name).strip().lower()
    if key not in THEMES:
        raise UnknownThemeError(
            f"unknown theme {name!r}. Known themes: {', '.join(sorted(THEMES))}. "
            "Pass --theme light or --theme dark."
        )
    return THEMES[key]


def sequential_color(theme: Theme, position: float) -> str:
    """Pick a ramp step for a value already normalised to ``[0, 1]``.

    Args:
        theme: the theme whose ``sequential`` ramp is sampled.
        position: 0.0 selects the step furthest from the surface end of the
            ramp, 1.0 the nearest. Values outside ``[0, 1]`` are clamped, so a
            caller that normalises against a degenerate range still gets a
            valid colour instead of an IndexError.

    Returns:
        A hex colour string from ``theme.sequential``.
    """
    ramp = theme.sequential
    if not ramp:  # defensive: a Theme constructed with an empty ramp
        raise ValueError(f"theme {theme.name!r} has an empty sequential ramp.")
    clamped = min(1.0, max(0.0, float(position)))
    index = int(round(clamped * (len(ramp) - 1)))
    return ramp[index]


def style_axes(
    ax: "Axes",
    theme: Theme,
    *,
    grid_axis: str = "both",
) -> None:
    """Apply the recessive chrome every figure in this package shares.

    Sets the axes surface, drops the top and right spines, makes the remaining
    spines and the grid hairline-weight, and recolours every tick label. Called
    by each figure builder; never mutates ``matplotlib.rcParams``, so two
    figures built in the same process cannot influence each other.

    Args:
        ax: the axes to style.
        theme: the theme supplying the colours.
        grid_axis: ``"x"``, ``"y"``, ``"both"`` or ``"none"``.
    """
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.baseline)
        ax.spines[side].set_linewidth(HAIRLINE_PT)
    ax.tick_params(
        colors=theme.text_muted,
        labelsize=TICK_SIZE_PT,
        length=3.0,
        width=HAIRLINE_PT,
    )
    if grid_axis != "none":
        ax.grid(
            True,
            axis=grid_axis,
            color=theme.gridline,
            linewidth=HAIRLINE_PT,
            linestyle="-",
        )
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def add_titles(
    ax: "Axes",
    theme: Theme,
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    """Put a title, and optionally a subtitle line, above ``ax``.

    Title and subtitle both wear text tokens, never the series colour.

    Args:
        ax: the axes to title.
        theme: the theme supplying the ink colours.
        title: the headline. Names what is plotted.
        subtitle: optional second line. Use it for provenance — dataset, model,
            seed, units — so the figure is readable detached from its caption.
    """
    ax.set_title(
        title,
        color=theme.text_primary,
        fontsize=TITLE_SIZE_PT,
        fontweight="semibold",
        loc="left",
        # 18 pt clears the subtitle line below it (6 pt offset + an 8 pt
        # line); 14 let the two touch in rendered output.
        pad=18.0 if subtitle else 8.0,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, 6.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=theme.text_secondary,
            fontsize=SUBTITLE_SIZE_PT,
        )
