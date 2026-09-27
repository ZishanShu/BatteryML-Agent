"""The figure naming contract. Pure data — no matplotlib, no numpy, no I/O.

``CLAUDE.md`` §8 makes artifact filenames part of the manuscript: the paper's
per-phase "Main outputs" lists quote them verbatim, so renaming one
desynchronises the code from the paper. These names are therefore declared once,
here, and every producer and checker reads them from this module rather than
formatting a path inline.

Adding a figure means adding a :class:`FigureSpec`. Renaming one means editing
the manuscript in the same change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Characters allowed in a filename field. Everything else collapses to '-', so a
# dataset or model name can never inject a path separator or a shell character
# into a filename the paper quotes.
_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]+")

RESULTS_SUBDIR = "results"
DATA_SUBDIR = "data"


@dataclass(frozen=True)
class FigureSpec:
    """One figure the platform is contracted to produce.

    Attributes:
        figure_id: stable identity used in the manifest and by the FIG-01 check.
        builder: the function in ``batteryml_web.viz.figures`` that draws it.
        template: filename template relative to the figures directory. Fields:
            ``dataset``, ``model``, ``seed``, ``fmt``.
        requires_cell_ids: True when the figure cannot be drawn unless the
            prediction rows can be matched to named cells.
        description: one line, used in docs and in the manifest.
    """

    figure_id: str
    builder: str
    template: str
    requires_cell_ids: bool
    description: str


RESULT_FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        figure_id="parity",
        builder="parity_figure",
        template=RESULTS_SUBDIR + "/parity_{dataset}_{model}_seed{seed}.{fmt}",
        requires_cell_ids=False,
        description="Predicted against measured, with the identity line and a relative tolerance band.",
    ),
    FigureSpec(
        figure_id="residuals",
        builder="residual_figure",
        template=RESULTS_SUBDIR + "/residuals_{dataset}_{model}_seed{seed}.{fmt}",
        requires_cell_ids=False,
        description="Signed error against the measured value, on a zero baseline.",
    ),
    FigureSpec(
        figure_id="error_by_cell",
        builder="error_by_cell_figure",
        template=RESULTS_SUBDIR + "/error_by_cell_{dataset}_{model}_seed{seed}.{fmt}",
        requires_cell_ids=True,
        description="Absolute error per test cell, sorted worst first.",
    ),
    FigureSpec(
        figure_id="soh_vs_cycle",
        builder="soh_vs_cycle_figure",
        template=RESULTS_SUBDIR + "/soh_vs_cycle_{dataset}_{model}_seed{seed}.{fmt}",
        requires_cell_ids=True,
        description=(
            "Measured and estimated SOH across each test cell's observed "
            "cycles. Every point is an at-cycle estimate, not a forecast."
        ),
    ),
    FigureSpec(
        figure_id="metrics_summary",
        builder="metric_summary_figure",
        template=RESULTS_SUBDIR + "/metrics_summary_{dataset}_{model}_seed{seed}.{fmt}",
        requires_cell_ids=False,
        description="A stat tile per recorded metric, displayed verbatim from the evaluator.",
    ),
)

# Data-side figures are VIZ-05 and are not implemented yet. Declared empty
# rather than omitted so the FIG-01 check has a single place to read the
# contract from once they land.
DATA_FIGURES: tuple[FigureSpec, ...] = ()

ALL_FIGURES: tuple[FigureSpec, ...] = RESULT_FIGURES + DATA_FIGURES


class UnknownFigureError(KeyError):
    """Raised for a figure id that is not part of the contract."""


def slugify(value: str) -> str:
    """Reduce a field to characters that are safe in a filename.

    Args:
        value: a dataset name, model key or seed.

    Returns:
        The value with every run of unsafe characters replaced by ``-`` and
        leading/trailing separators trimmed.

    Raises:
        ValueError: if nothing survives. An empty field would produce a filename
            like ``parity__xgb_seed0.png``, which the paper cannot quote.
    """
    cleaned = _UNSAFE.sub("-", str(value)).strip("-_.")
    if not cleaned:
        raise ValueError(
            f"{value!r} contains no characters that are safe in a filename. "
            "Pass an explicit --dataset / --model."
        )
    return cleaned


def get_spec(figure_id: str) -> FigureSpec:
    """Look up one spec by id.

    Raises:
        UnknownFigureError: listing the ids that do exist.
    """
    for spec in ALL_FIGURES:
        if spec.figure_id == figure_id:
            return spec
    raise UnknownFigureError(
        f"unknown figure id {figure_id!r}. Known: "
        f"{', '.join(item.figure_id for item in ALL_FIGURES)}."
    )


def figure_filename(spec: FigureSpec, *, dataset: str, model: str, seed: int | str, fmt: str) -> str:
    """Render one figure's contracted filename.

    Args:
        spec: the figure.
        dataset: dataset name, e.g. ``MATR``.
        model: model key, e.g. ``xgb``.
        seed: the seed the run used.
        fmt: file extension without the dot.

    Returns:
        A path relative to the figures directory, using ``/`` separators.
    """
    return spec.template.format(
        dataset=slugify(dataset),
        model=slugify(model),
        seed=slugify(seed),
        fmt=slugify(fmt),
    )
