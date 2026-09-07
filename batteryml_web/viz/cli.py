"""Offline figure generation. Route B: no API key, no Streamlit, no network.

Usage::

    python -m batteryml_web.viz.cli results \\
        --prediction runs/native_matr_xgb_test/predictions_seed_0_20260101120000.pkl \\
        --config configs/baselines/sklearn/xgb/matr_1.yaml \\
        --output-dir runs/web_projects/demo/figures

Writes the four result figures named by ``batteryml_web/viz/contract.py`` plus
``figure_manifest.json``. Figures that cannot be drawn are reported with a
reason and listed in the manifest's ``skipped`` array; they are never silently
omitted, and a wrong figure is never drawn in place of a missing one.

Writing into ``<workspace>/figures/`` is enough to reach a project directory:
``batteryml_web/app.py:302-311`` already copies that subdirectory into
``runs/web_projects/<project>/figures/`` when it collects run outputs.

Exit codes:

* ``0`` — every contracted figure was written, or the ones skipped were skipped
  for a stated reason and ``--strict`` was not given;
* ``1`` — an input could not be used, or ``--strict`` and something was skipped;
* ``2`` — a usage error (argparse).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# Unit caption for BatteryML's MAPE. Its evaluator returns
# ``mean(|target - prediction| / target)`` — a ratio, not a percentage
# (``batteryml/data/databundle.py:106``). The value is displayed verbatim; the
# caption says how to read it rather than the code multiplying by 100.
MAPE_UNIT = "ratio (x100 = %)"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m batteryml_web.viz.cli",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    results = sub.add_parser(
        "results",
        help="Draw the result figures for one evaluated run.",
        description="Draw parity, residual, per-cell error and metric-summary figures.",
    )
    results.add_argument(
        "--prediction",
        required=True,
        type=Path,
        help="predictions_seed_<seed>_<ts>.pkl written by BatteryML's evaluation.",
    )
    results.add_argument(
        "--config",
        type=Path,
        default=None,
        help="The run's YAML config. Without it, cell identity and the NaN-label mask "
             "cannot always be established and the per-cell figure is skipped.",
    )
    results.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Figures directory. The manifest is written here; figures go into its "
             "results/ subdirectory.",
    )
    results.add_argument("--dataset", default=None, help="Dataset name for the filename. Default: from the config.")
    results.add_argument("--model", default=None, help="Model key for the filename. Default: the config file's parent directory.")
    results.add_argument("--seed", type=int, default=None, help="Seed for the filename. Default: the seed recorded in the artifact.")
    results.add_argument("--theme", default="light", choices=("light", "dark"), help="Surface the figures are drawn against.")
    results.add_argument("--format", dest="fmt", default="png", choices=("png", "svg", "pdf"), help="Output format.")
    results.add_argument("--dpi", type=float, default=300.0, help="Raster resolution for png.")
    results.add_argument("--label-unit", default=None, help="Axis unit override. Default: derived from the config's label annotator.")
    results.add_argument("--band", type=float, default=0.10, help="Half-width of the parity tolerance band, as a fraction. 0 disables it.")
    results.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-4,
        help="Maximum relative difference between the recorded metrics and the ones "
             "recomputed from the inverse-transformed arrays.",
    )
    results.add_argument(
        "--verify-cell-alignment",
        action="store_true",
        help="Load the test cells from disk to prove the per-cell mapping even when a "
             "count match already implies it.",
    )
    results.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any contracted figure was skipped.",
    )
    return parser


def _resolve_identity(
    artifact: Any,
    config: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[str, str, int]:
    """Decide the dataset, model and seed that go into every filename.

    Precedence is explicit flag, then the config, then the artifact. A field
    that cannot be resolved is an error rather than a placeholder: the filenames
    are quoted by the manuscript, so ``parity_None_xgb_seed0.png`` is worse than
    a refusal.

    Returns:
        ``(dataset, model, seed)``.

    Raises:
        SystemExit: never. Raises ``ValueError`` for the caller to report.
    """
    from .loaders import dataset_from_config

    dataset = args.dataset or dataset_from_config(config)
    if not dataset:
        raise ValueError(
            "cannot determine the dataset name for the filenames. Pass --dataset, or "
            "pass --config so it can be read from train_test_split.cell_data_path."
        )

    model = args.model
    if not model and args.config is not None:
        model = Path(args.config).parent.name
    if not model and config:
        model = str((config.get("model") or {}).get("name", "")).strip().lower() or None
    if not model:
        raise ValueError(
            "cannot determine the model key for the filenames. Pass --model, or pass "
            "--config (the config file's parent directory names the model, e.g. "
            "configs/baselines/sklearn/xgb/matr_1.yaml -> xgb)."
        )

    seed = args.seed if args.seed is not None else artifact.seed
    if seed is None:
        raise ValueError(
            "the artifact records no seed and --seed was not given. The seed is part of "
            "every figure filename and of the reproducibility contract (CLAUDE.md §8)."
        )
    return str(dataset), str(model), int(seed)


def _annotation(artifact: Any, unit: str) -> str:
    """A one-line summary of the recorded metrics, displayed verbatim."""
    from .figures import format_metric_value

    parts: list[str] = []
    for metric in ("RMSE", "MAE"):
        if metric in artifact.scores:
            parts.append(f"{metric} {format_metric_value(artifact.scores[metric])} {unit}")
    if "MAPE" in artifact.scores:
        parts.append(f"MAPE {format_metric_value(artifact.scores['MAPE'])} ({MAPE_UNIT})")
    parts.append(f"n = {artifact.n_predictions}")
    return "  ".join(parts)


def _run_results(args: argparse.Namespace) -> int:
    from . import contract, figures, loaders, render, style

    theme = style.get_theme(args.theme)
    output_dir = Path(args.output_dir)

    artifact = loaders.load_prediction_artifact(
        args.prediction,
        config_path=args.config,
        score_tolerance=args.score_tolerance,
        verify_cell_alignment=args.verify_cell_alignment,
    )
    config = loaders.read_config(args.config) if args.config is not None else None
    dataset, model, seed = _resolve_identity(artifact, config, args)
    unit = args.label_unit or loaders.label_unit_from_config(config)
    subtitle = f"{dataset} · {model} · seed {seed} · n = {artifact.n_predictions}"
    annotation = _annotation(artifact, unit)

    written: list[render.RenderedFigure] = []
    skipped: list[render.SkippedFigure] = []

    for spec in contract.RESULT_FIGURES:
        if spec.requires_cell_ids and artifact.cells.cell_ids is None:
            skipped.append(
                render.SkippedFigure(
                    spec.figure_id,
                    f"cell identity unavailable: {artifact.cells.reason}",
                )
            )
            continue

        if spec.figure_id == "parity":
            fig = figures.parity_figure(
                artifact.y_true,
                artifact.y_pred,
                theme=theme,
                title="Predicted vs measured",
                subtitle=subtitle,
                unit_label=unit,
                band_fraction=args.band,
                annotation=annotation,
                cell_ids=artifact.cells.cell_ids,
                dpi=args.dpi,
            )
            details = {"n_points": artifact.n_predictions, "band_fraction": args.band}
        elif spec.figure_id == "residuals":
            fig = figures.residual_figure(
                artifact.y_true,
                artifact.residuals,
                theme=theme,
                title="Residuals",
                subtitle=subtitle,
                unit_label=unit,
                annotation="Positive = over-prediction",
                dpi=args.dpi,
            )
            details = {"n_points": artifact.n_predictions, "sign_convention": "y_pred - y_true"}
        elif spec.figure_id == "error_by_cell":
            fig = figures.error_by_cell_figure(
                artifact.cells.cell_ids or [],
                abs(artifact.residuals),
                theme=theme,
                title="Absolute error per test cell",
                subtitle=subtitle,
                unit_label=unit,
                dpi=args.dpi,
            )
            details = {
                "n_cells": artifact.n_predictions,
                "cell_id_source": artifact.cells.source,
                "alignment_verified": artifact.cells.alignment_verified,
                "n_dropped_nan_label": artifact.cells.n_dropped_nan_label,
            }
        elif spec.figure_id == "metrics_summary":
            fig = figures.metric_summary_figure(
                artifact.scores,
                theme=theme,
                title="Recorded metrics",
                subtitle=subtitle,
                metric_units={"RMSE": unit, "MAE": unit, "MAPE": MAPE_UNIT},
                note="Values as written by BatteryML's evaluator; never recomputed here.",
                dpi=args.dpi,
            )
            details = {"metrics": sorted(artifact.scores)}
        else:  # pragma: no cover - contract and dispatch drift
            raise RuntimeError(
                f"figure id {spec.figure_id!r} is declared in the contract but has no "
                "branch in batteryml_web/viz/cli.py. Add one, or remove the spec."
            )

        filename = contract.figure_filename(
            spec, dataset=dataset, model=model, seed=seed, fmt=args.fmt
        )
        written.append(
            render.save_figure(
                fig,
                output_dir / filename,
                figure_id=spec.figure_id,
                fmt=args.fmt,
                dpi=args.dpi,
                manifest_dir=output_dir,
                details={**details, "description": spec.description},
            )
        )

    manifest = render.build_manifest(
        figures=written,
        skipped=skipped,
        source={
            "prediction_artifact": str(args.prediction),
            "prediction_sha256": artifact.source_sha256,
            "config": str(args.config) if args.config else None,
            "config_sha256": loaders.sha256_of(args.config) if args.config else None,
            "dataset": dataset,
            "model": model,
            "seed": seed,
        },
        theme=theme.name,
        dpi=args.dpi,
        repo_root=REPO_ROOT,
        extra={
            "label_space": "raw (label transformations inverted)",
            "label_unit": unit,
            "label_transformations": artifact.label_transformations,
            "scores": artifact.scores,
            "score_consistency_check": artifact.score_check,
            "cell_identity": {
                "source": artifact.cells.source,
                "alignment_verified": artifact.cells.alignment_verified,
                "n_cells_total": artifact.cells.n_cells_total,
                "n_dropped_nan_label": artifact.cells.n_dropped_nan_label,
                "reason": artifact.cells.reason,
            },
        },
    )
    manifest_path = render.write_manifest(output_dir, manifest)

    for item in written:
        print(f"wrote  {item.filename}  ({item.size_bytes} bytes, sha256 {item.sha256[:12]}...)")
    for item in skipped:
        print(f"SKIP   {item.figure_id}: {item.reason}")
    print(f"wrote  {manifest_path.name}")
    print(
        f"label space: raw, unit {unit!r}; inverted "
        f"{' -> '.join(artifact.label_transformations) or 'nothing'}; "
        f"recorded metrics reproduced within {args.score_tolerance:g} relative."
    )

    if skipped and args.strict:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Turns every expected failure into a message and exit 1."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from . import loaders, render, style  # noqa: F401 - import check only
    except ImportError as exc:
        print(
            f"error: cannot import the visualisation modules: {exc}. They need numpy, "
            "matplotlib and torch. Install them with `pip install -r requirements.txt` "
            "and run from the repository root.",
            file=sys.stderr,
        )
        return 1

    from .figures import VizFigureError
    from .loaders import VizDataError
    from .render import VizRenderError
    from .style import UnknownThemeError

    try:
        if args.command == "results":
            return _run_results(args)
        parser.error(f"unknown command {args.command!r}")
        return 2
    except (VizDataError, VizFigureError, VizRenderError, UnknownThemeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
