"""What do the real SOH labels actually look like?

Eligibility says a dataset *can* produce SOH labels. This says what those
labels are: their distribution, how many are impossible, and how many sit
above 1.0. Run it on the machine with the data, **before** training anything.

    python -m scripts.audit_soh_labels --dataset MATR --min-cycle 10 --cycle-stride 5
    python -m scripts.audit_soh_labels --dataset MATR --per-cell --json runs/verify/soh_labels_matr.json

Nothing is altered. In particular **SOH is never clipped to [0, 1]**: under a
first-cycle reference, a ratio above 1.0 early in life is a real measurement —
capacity activation in the first cycles is a known effect — and clipping it
would be fabricating data to make a distribution look tidy. The audit's job is
to surface a suspicious distribution, not to tidy one.

Read the output for structural corruption before writing a training config:

* a large ``<= 0`` or ``NaN`` count means the capacity traces are not what the
  label path assumes;
* a max far above ~1.2, or a median far from ~1.0, means the reference
  capacity is not being established where you think it is;
* a p1 near 0 means cells are being labelled long past any usable life.

If any of those appear, stop and report rather than training on them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Reported because a first-cycle reference legitimately produces some of
#: these. They are counted, never removed.
HIGH_THRESHOLDS = (1.0, 1.05, 1.10)


def percentile(sorted_values: List[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def audit_dataset(dataset: str,
                  processed_root: Path,
                  reference_capacity_mode: str,
                  reference_n_cycles: int,
                  min_cycle: int,
                  max_cycle: int | None,
                  cycle_stride: int,
                  max_cells: int | None,
                  per_cell: bool) -> dict:
    """Label statistics for one dataset, computed through the real annotator."""
    from batteryml.data.battery_data import BatteryData
    from batteryml.label.soh import CycleSOHLabelAnnotator, reference_capacity

    path = processed_root / dataset
    if not path.is_dir():
        raise SystemExit(
            f"{path.as_posix()} does not exist. SOH labels are computed from "
            "processed cells; run preprocessing for this dataset first."
        )
    files = sorted(p for p in path.glob("*.pkl") if p.is_file())
    if max_cells is not None:
        files = files[: int(max_cells)]
    if not files:
        raise SystemExit(f"{path.as_posix()} holds no .pkl cell.")

    values: List[float] = []
    references: List[float] = []
    per_cell_rows: List[dict] = []
    invalid = {"nan": 0, "inf": 0, "non_positive": 0}
    exclusions: dict[str, int] = {}
    cells_with_samples = 0

    for file in files:
        cell = BatteryData.load(str(file))
        annotator = CycleSOHLabelAnnotator(
            reference_capacity_mode=reference_capacity_mode,
            reference_n_cycles=reference_n_cycles,
            min_cycle=min_cycle,
            max_cycle=max_cycle,
            cycle_stride=cycle_stride,
        )
        labels, cycles = annotator.process_cell_cycles(cell)
        for reason, count in annotator.exclusions.items():
            exclusions[reason] = exclusions.get(reason, 0) + count

        q_ref, _ = reference_capacity(cell, reference_capacity_mode, reference_n_cycles)
        if q_ref == q_ref:
            references.append(float(q_ref))

        cell_values = [float(v) for v in labels.tolist()]
        for value in cell_values:
            if math.isnan(value):
                invalid["nan"] += 1
            elif math.isinf(value):
                invalid["inf"] += 1
            elif value <= 0.0:
                invalid["non_positive"] += 1
        values.extend(cell_values)
        if cell_values:
            cells_with_samples += 1
            if per_cell:
                ordered = sorted(cell_values)
                per_cell_rows.append({
                    "cell_id": str(cell.cell_id),
                    "samples": len(cell_values),
                    "first_cycle": int(cycles[0]),
                    "last_cycle": int(cycles[-1]),
                    "reference_capacity_in_Ah": float(q_ref) if q_ref == q_ref else None,
                    "soh_min": ordered[0],
                    "soh_median": percentile(ordered, 0.50),
                    "soh_max": ordered[-1],
                })

    finite = sorted(v for v in values if math.isfinite(v))
    ordered_refs = sorted(references)
    report = {
        "dataset": dataset,
        "processed_path": path.as_posix(),
        "cells_inspected": len(files),
        "cells_with_samples": cells_with_samples,
        "samples": len(values),
        "reference_capacity_mode": reference_capacity_mode,
        "sample_policy": {
            "min_cycle": min_cycle,
            "max_cycle": max_cycle,
            "cycle_stride": cycle_stride,
            "reference_n_cycles": reference_n_cycles,
        },
        "soh": {
            "min": finite[0] if finite else None,
            "p1": percentile(finite, 0.01),
            "p5": percentile(finite, 0.05),
            "median": percentile(finite, 0.50),
            "p95": percentile(finite, 0.95),
            "p99": percentile(finite, 0.99),
            "max": finite[-1] if finite else None,
        },
        "invalid": invalid,
        "high_values": {
            f"> {threshold}": sum(1 for v in finite if v > threshold)
            for threshold in HIGH_THRESHOLDS
        },
        "reference_capacity_in_Ah": {
            "min": ordered_refs[0] if ordered_refs else None,
            "median": percentile(ordered_refs, 0.50),
            "max": ordered_refs[-1] if ordered_refs else None,
        },
        "exclusions": exclusions,
    }
    if per_cell:
        report["cells"] = per_cell_rows
    return report


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(report: dict) -> str:
    soh = report["soh"]
    ref = report["reference_capacity_in_Ah"]
    lines = [
        f"dataset                 : {report['dataset']}",
        f"cells                   : {report['cells_inspected']} "
        f"({report['cells_with_samples']} produced samples)",
        f"samples                 : {report['samples']}",
        f"reference_capacity_mode : {report['reference_capacity_mode']}",
        f"sample policy           : min_cycle={report['sample_policy']['min_cycle']} "
        f"max_cycle={report['sample_policy']['max_cycle']} "
        f"cycle_stride={report['sample_policy']['cycle_stride']}",
        "",
        "SOH (ratio, never clipped):",
        f"    min    {_fmt(soh['min'])}",
        f"    p1     {_fmt(soh['p1'])}",
        f"    p5     {_fmt(soh['p5'])}",
        f"    median {_fmt(soh['median'])}",
        f"    p95    {_fmt(soh['p95'])}",
        f"    p99    {_fmt(soh['p99'])}",
        f"    max    {_fmt(soh['max'])}",
        "",
        "invalid:",
        f"    NaN    {report['invalid']['nan']}",
        f"    inf    {report['invalid']['inf']}",
        f"    <= 0   {report['invalid']['non_positive']}",
        "",
        "high values (expected under a first-cycle reference; not an error):",
    ]
    for label, count in report["high_values"].items():
        share = 100.0 * count / report["samples"] if report["samples"] else 0.0
        lines.append(f"    {label:<8} {count} ({share:.2f}%)")
    lines += [
        "",
        "reference capacity (Ah):",
        f"    min    {_fmt(ref['min'])}",
        f"    median {_fmt(ref['median'])}",
        f"    max    {_fmt(ref['max'])}",
    ]
    if report["exclusions"]:
        lines.append("")
        lines.append("excluded:")
        for reason, count in sorted(report["exclusions"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason} = {count}")
    return "\n".join(lines)


def structural_warnings(report: dict) -> List[str]:
    """Distributions worth stopping for. Reported, never acted on."""
    warnings: List[str] = []
    soh, invalid = report["soh"], report["invalid"]
    samples = report["samples"] or 1

    bad = invalid["nan"] + invalid["inf"] + invalid["non_positive"]
    if bad:
        warnings.append(
            f"{bad} label(s) are NaN, infinite or <= 0. An SOH ratio cannot be "
            "any of those; the capacity traces are not what the label path "
            "assumes."
        )
    if soh["median"] is not None and not (0.5 <= soh["median"] <= 1.2):
        warnings.append(
            f"median SOH is {soh['median']:.4f}. A cohort's median health "
            "should sit near 1.0 under a first-cycle reference; this suggests "
            "the reference capacity is not where you think it is."
        )
    if soh["max"] is not None and soh["max"] > 1.5:
        warnings.append(
            f"max SOH is {soh['max']:.4f}. Values slightly above 1.0 are "
            "expected from early capacity activation; 1.5 is not."
        )
    if soh["p1"] is not None and soh["p1"] < 0.2:
        warnings.append(
            f"p1 SOH is {soh['p1']:.4f}, so ~1% of samples are below 20% "
            "health. Check whether cells are being labelled past any usable "
            "life; consider max_cycle."
        )
    high = report["high_values"].get("> 1.1", 0)
    if high / samples > 0.05:
        warnings.append(
            f"{100.0 * high / samples:.1f}% of labels exceed 1.10, which is "
            "more capacity activation than a first-cycle reference usually "
            "explains."
        )
    return warnings


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", required=True,
                        help="dataset name; repeatable.")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--reference-capacity-mode", default=None)
    parser.add_argument("--reference-n-cycles", type=int, default=5)
    parser.add_argument("--min-cycle", type=int, default=1)
    parser.add_argument("--max-cycle", type=int, default=None)
    parser.add_argument("--cycle-stride", type=int, default=1)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--per-cell", action="store_true",
                        help="also report one row per cell.")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from batteryml.label.soh import DEFAULT_REFERENCE_CAPACITY_MODE

    mode = args.reference_capacity_mode or DEFAULT_REFERENCE_CAPACITY_MODE
    reports, all_warnings = [], []
    for dataset in args.dataset:
        print(f"auditing {dataset} ...", file=sys.stderr, flush=True)
        report = audit_dataset(
            dataset, Path(args.processed_root), mode, args.reference_n_cycles,
            args.min_cycle, args.max_cycle, args.cycle_stride, args.max_cells,
            args.per_cell,
        )
        reports.append(report)
        print()
        print(render(report))
        warnings = structural_warnings(report)
        all_warnings.extend(warnings)
        print()
        if warnings:
            print("SUSPICIOUS — inspect before training:")
            for warning in warnings:
                print(f"  ! {warning}")
        else:
            print("No structural problem found in the label distribution.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nreport: {out.as_posix()}")

    # Non-zero when something looks structurally wrong, so a scripted run
    # stops rather than proceeding to training on labels nobody looked at.
    return 1 if all_warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
