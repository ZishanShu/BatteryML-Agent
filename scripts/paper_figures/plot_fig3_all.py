"""Draw every Fig. 3 panel that the collected data supports — as separate files.

Runs panels A–E one after another on one collected directory. A panel the
data cannot support is reported (MISSING_DATA / INSUFFICIENT_DATA /
INVALID_COMPARISON) with its reason and is NOT drawn; the others are still
drawn. No combined A–E figure is produced. The status table is printed and
saved as ``fig3_status.md`` / ``fig3_status.csv`` beside the panels.

Usage::

    python -m scripts.paper_figures.plot_fig3_all \\
        --data runs/paper_figures/fig3_data --output runs/paper_figures/fig3_v1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

from scripts.paper_figures import (
    fig3a_lifetime_heatmap as panel_a,
    fig3b_soh_error as panel_b,
    fig3c_multicriteria_radar as panel_c,
    fig3d_inference_latency as panel_d,
    fig3e_lifetime_error_distribution as panel_e,
)
from scripts.paper_figures.common import (
    PanelResult, PanelSkip, Style, base_parser, command_line, write_csv,
)

STATUS_FILES = ("fig3_status.md", "fig3_status.csv")


def _run(name: str, call: Callable[[], PanelResult]) -> PanelResult:
    try:
        return call()
    except PanelSkip as skip:
        return PanelResult(name, skip.status, [], skip.reason)


def plot_all(data_dir: Path, out_dir: Path, *, allow_smoke: bool = False,
             metric: str = "RMSE", color_scale: str = "global", published: bool = False,
             soh_scale: str = "ratio", e_dataset: Optional[str] = None,
             e_protocol: Optional[str] = None, e_seed: Optional[str] = None,
             style_overrides: Optional[dict] = None,
             argv: Optional[List[str]] = None) -> List[PanelResult]:
    out_dir = Path(out_dir)
    for name in STATUS_FILES:
        if (out_dir / name).exists():
            raise FileExistsError(f"{out_dir / name} exists; use a new output directory.")
    overrides = style_overrides or {}
    command = command_line(argv, "scripts.paper_figures.plot_fig3_all")

    def style(width: float, height: float) -> Style:
        return Style(width_in=overrides.get("width") or width,
                     height_in=overrides.get("height") or height,
                     font_size=overrides.get("font_size", 7.0),
                     line_width=overrides.get("line_width", 0.8),
                     dpi=overrides.get("dpi", 300.0))

    results = [
        _run(panel_a.STEM, lambda: panel_a.make_panel(
            data_dir, out_dir, style=style(3.5, 2.6), metric=metric,
            color_scale=color_scale, allow_smoke=allow_smoke, argv=argv, command=command)),
    ]
    if published:
        results.append(_run(panel_a.PUBLISHED_STEM, lambda: panel_a.make_panel(
            data_dir, out_dir, style=style(3.5, 2.6), metric=metric, source="published",
            color_scale=color_scale, argv=argv, command=command)))
    results += [
        _run(panel_b.STEM, lambda: panel_b.make_panel(
            data_dir, out_dir, style=style(3.5, 2.4), scale=soh_scale,
            allow_smoke=allow_smoke, argv=argv, command=command)),
        _run(panel_c.STEM, lambda: panel_c.make_panel(
            data_dir, out_dir, style=style(3.2, 3.0), allow_smoke=allow_smoke, argv=argv, command=command)),
        _run(panel_d.STEM, lambda: panel_d.make_panel(
            data_dir, out_dir, style=style(3.0, 2.4), allow_smoke=allow_smoke, argv=argv, command=command)),
        _run(panel_e.STEM, lambda: panel_e.make_panel(
            data_dir, out_dir, style=style(3.5, 2.6), dataset=e_dataset,
            protocol=e_protocol, seed=e_seed, allow_smoke=allow_smoke, argv=argv, command=command)),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"panel": r.panel, "status": r.status,
             "output_or_reason": "; ".join(r.outputs) if r.outputs else r.reason}
            for r in results]
    write_csv(out_dir / "fig3_status.csv", rows, ["panel", "status", "output_or_reason"])
    (out_dir / "fig3_status.md").write_text(status_table(results) + "\n", encoding="utf-8")
    return results


def status_table(results: List[PanelResult]) -> str:
    lines = ["| Panel | Status | Output / Reason |", "| --- | --- | --- |"]
    for r in results:
        detail = ", ".join(Path(o).name for o in r.outputs) if r.outputs else r.reason
        if r.outputs and r.reason:
            detail += f" — {r.reason}"
        lines.append(f"| {r.panel} | {r.status} | {detail.replace('|', '/')} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--metric", choices=("RMSE", "MAE"), default="RMSE",
                        help="Panel A metric.")
    parser.add_argument("--color-scale", choices=("global", "column", "none"), default="global")
    parser.add_argument("--published-reference", action="store_true",
                        help="Also draw panel A's separate published-reference variant.")
    parser.add_argument("--soh-scale", choices=("ratio", "pp"), default="ratio")
    parser.add_argument("--e-dataset", default=None)
    parser.add_argument("--e-protocol", default=None)
    parser.add_argument("--e-seed", default=None)
    args = parser.parse_args(argv)
    try:
        results = plot_all(
            args.data, args.output, allow_smoke=args.allow_smoke, metric=args.metric,
            color_scale=args.color_scale, published=args.published_reference,
            soh_scale=args.soh_scale, e_dataset=args.e_dataset, e_protocol=args.e_protocol,
            e_seed=args.e_seed,
            style_overrides={"width": args.width, "height": args.height,
                             "font_size": args.font_size, "line_width": args.line_width,
                             "dpi": args.dpi},
            argv=argv if argv is not None else sys.argv[1:])
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(status_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
