"""Which processed datasets can actually support SOH estimation?

Run this on the machine that holds the data. It opens the processed cells,
labels them through the same annotator and feature extractor a run would use,
and reports the counts it actually found. It never guesses from a README, a
results table or a directory listing.

    python -m scripts.audit_soh_readiness
    python -m scripts.audit_soh_readiness --dataset MATR --dataset HUST
    python -m scripts.audit_soh_readiness --min-cycle 10 --cycle-stride 5
    python -m scripts.audit_soh_readiness --json runs/verify/soh_readiness.json

Output is one row per dataset::

    Dataset | Cells | Eligible cells | SOH samples | Status | Reason

A dataset is ``SOH_READY`` only when enough cells each yield enough
cycle-level samples. ``SOH_NOT_PROCESSED`` means there is nothing on disk to
judge — it is not a verdict about the dataset.

This reads data and writes only the optional JSON report. It trains nothing,
allocates no GPU memory and never writes into ``data/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core.soh_readiness import (  # noqa: E402
    DEFAULT_MIN_ELIGIBLE_CELLS,
    DEFAULT_MIN_SAMPLES_PER_CELL,
    SOH_READY,
    assess_soh_dataset,
)

#: Looked at when no --dataset is given: whatever is actually on disk.
def discover_datasets(processed_root: Path) -> list[str]:
    if not processed_root.is_dir():
        return []
    return sorted(p.name for p in processed_root.iterdir() if p.is_dir())


def render_table(rows: list[dict]) -> str:
    columns = ["Dataset", "Cells", "Eligible cells", "SOH samples", "Status"]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    } if rows else {column: len(column) for column in columns}
    lines = [" | ".join(column.ljust(widths[column]) for column in columns)]
    lines.append("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        lines.append(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", default=None,
                        help="dataset name; repeatable. Default: every directory "
                             "under the processed root.")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--reference-capacity-mode", default=None,
                        help="Q_ref convention. Default: the label module's "
                             "default (first_valid_discharge_capacity).")
    parser.add_argument("--reference-n-cycles", type=int, default=5)
    parser.add_argument("--min-cycle", type=int, default=1)
    parser.add_argument("--max-cycle", type=int, default=None)
    parser.add_argument("--cycle-stride", type=int, default=1)
    parser.add_argument("--min-samples-per-cell", type=int,
                        default=DEFAULT_MIN_SAMPLES_PER_CELL)
    parser.add_argument("--min-eligible-cells", type=int,
                        default=DEFAULT_MIN_ELIGIBLE_CELLS)
    parser.add_argument("--max-cells", type=int, default=None,
                        help="inspect at most this many cells per dataset. The "
                             "report says how many were inspected.")
    parser.add_argument("--json", default=None, help="also write the full report here")
    args = parser.parse_args(argv)

    processed_root = Path(args.processed_root)
    datasets = args.dataset or discover_datasets(processed_root)
    if not datasets:
        print(
            f"No processed dataset found under {processed_root.as_posix()}. "
            "Run preprocessing first; SOH eligibility is decided from "
            "processed cells, never from a README.",
            file=sys.stderr,
        )
        return 2

    reports = []
    for dataset in datasets:
        print(f"inspecting {dataset} ...", file=sys.stderr, flush=True)
        reports.append(assess_soh_dataset(
            dataset,
            processed_root=processed_root,
            reference_capacity_mode=args.reference_capacity_mode,
            reference_n_cycles=args.reference_n_cycles,
            min_cycle=args.min_cycle,
            max_cycle=args.max_cycle,
            cycle_stride=args.cycle_stride,
            min_samples_per_cell=args.min_samples_per_cell,
            min_eligible_cells=args.min_eligible_cells,
            max_cells=args.max_cells,
        ))

    rows = [report.summary_row() for report in reports]
    print()
    print(render_table(rows))
    print()
    for report in reports:
        print(f"{report.dataset}: {report.reason}")
        if report.exclusions:
            detail = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(report.exclusions.items(), key=lambda kv: -kv[1])
            )
            print(f"    excluded: {detail}")

    ready = [report for report in reports if report.status == SOH_READY]
    print()
    if ready:
        best = max(ready, key=lambda report: (report.cells_eligible, report.total_samples))
        print(
            f"{len(ready)} dataset(s) are SOH_READY. The largest is "
            f"{best.dataset} with {best.cells_eligible} eligible cell(s) and "
            f"{best.total_samples} cycle-level sample(s)."
        )
        print(
            "Next: create ONE config for an eligible dataset — see "
            "docs/SOH_PROTOCOL.md for the template — then run the SOH smoke."
        )
    else:
        print(
            "No dataset is SOH_READY. Do not create an SOH config: there is "
            "nothing to run it against. The per-dataset reasons above say what "
            "is missing."
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([report.to_dict(include_cells=True) for report in reports],
                       indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"report: {out.as_posix()}")

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
