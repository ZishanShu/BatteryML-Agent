"""Export the result figures of ONE exact prediction artifact.

Visualization only: reads the artifact you name (through the platform's
CPU-safe loader), draws the same panels the website shows, and writes them
with the exact plotted data and a provenance record. It never trains,
evaluates, downloads data, or looks for another run.

Usage (from the repository root)::

    # read-only: what is in the artifact, which cells it has, smoke evidence
    python -m scripts.export_result_figures --artifact <ARTIFACT.pkl> --inspect

    # write the export into a NEW directory
    python -m scripts.export_result_figures \\
      --artifact <ARTIFACT.pkl> \\
      --output runs/figure_exports/<NEW_DIRECTORY> \\
      [--cell-id <REAL_CELL_ID>] [--dpi 300]

``runs/`` is git-ignored, so an export written there is never committed.

Exit codes: 0 success; 1 the artifact or a requested cell cannot be used, or
the output directory is not new/empty; 2 usage error (argparse).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_result_figures",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--artifact", required=True, type=Path,
                        help="Exact predictions_seed_<seed>_<ts>.pkl to draw. No search is done.")
    parser.add_argument("--output", type=Path, default=None,
                        help="New (or empty) directory to write into. Required unless --inspect/--list-cells.")
    parser.add_argument("--cell-id", default=None,
                        help="Cycle-level SOH only: the cell for the SOH-vs-cycle panel. "
                             "Default: first cell ID in ascending order (independent of error).")
    parser.add_argument("--dpi", type=float, default=300.0, help="PNG resolution (SVG is vector).")
    parser.add_argument("--inspect", action="store_true",
                        help="Print a read-only JSON summary and exit; writes nothing.")
    parser.add_argument("--list-cells", action="store_true",
                        help="Print every plotted cell ID, one per line, and exit; writes nothing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from batteryml_web.viz.result_export import (
        ExportError, build_export, inspect_artifact, load_views_for_export, write_export,
    )
    from batteryml_web.viz.result_figures import cell_choices

    args = _parser().parse_args(argv)
    try:
        if args.list_cells:
            for cell in cell_choices(load_views_for_export(args.artifact)["view"]):
                print(cell)
            return 0
        if args.inspect:
            print(json.dumps(inspect_artifact(args.artifact), indent=2, default=str))
            return 0
        if args.output is None:
            print("error: --output is required (a new directory) unless --inspect "
                  "or --list-cells is given.", file=sys.stderr)
            return 1
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            # Checked before anything is drawn; write_export checks again.
            print(f"error: {args.output} already exists and is not empty. Choose a "
                  "new export directory; existing files are never overwritten.",
                  file=sys.stderr)
            return 1
        command = ["python", "-m", "scripts.export_result_figures"] + [
            shlex.quote(part) for part in (argv if argv is not None else sys.argv[1:])
        ]
        result = build_export(args.artifact, cell_id=args.cell_id, dpi=args.dpi,
                              command=command)
        written = write_export(result, args.output)
    except (ExportError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = result.sources["counts"]
    print(f"Wrote {len(written)} files to {args.output}")
    for path in written:
        print(f"  {path.name}")
    print(f"plotted {counts['plotted_rows']} of {counts['stored_rows']} rows; "
          f"{counts['omitted_nonfinite_rows']} non-finite omitted")
    if result.sources.get("selected_cell"):
        cell = result.sources["selected_cell"]
        print(f"SOH cell: {cell['cell_id']} ({cell['selection_rule']})")
    print(f"smoke/training evidence: {result.sources['source']['smoke_evidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
