"""What did a real SOH run actually predict?

Reads a completed run's ``predictions_seed_*.pkl`` and reports what is in it,
so the first real SOH workflow can be checked without opening a notebook.

    python -m scripts.inspect_soh_artifact runs/soh_smoke_matr_ridge
    python -m scripts.inspect_soh_artifact runs/soh_smoke_matr_ridge/predictions_seed_0_*.pkl \
        --config configs/soh/ridge/matr.yaml --rows 10

Everything printed comes from the artifact. Metrics are the evaluator's own
values, not recomputed here; the observed and predicted arrays are
inverse-transformed through the run's own label transformation and
cross-checked against those recorded metrics, so a mismatch is reported rather
than plotted.

**This is workflow validation, not a quality judgement.** One Ridge run on one
dataset says whether the route works end to end. It says nothing about how
well SOH can be estimated, and nothing here should be quoted as a result.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def find_artifact(target: Path) -> Path:
    """The artifact at ``target``, or the newest one inside that directory."""
    if target.is_file():
        return target
    if target.is_dir():
        candidates = sorted(
            target.glob("predictions_seed_*.pkl"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit(
                f"{target.as_posix()} holds no predictions_seed_*.pkl. Run the "
                "SOH config with --train --eval first."
            )
        return candidates[-1]
    raise SystemExit(f"{target.as_posix()} does not exist.")


def _summary(values) -> str:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return "-"
    return (
        f"min {ordered[0]:.4f}   median {statistics.median(ordered):.4f}   "
        f"max {ordered[-1]:.4f}"
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="the run directory or the .pkl itself")
    parser.add_argument("--config", default=None,
                        help="the run's YAML config. Improves cell-identity "
                             "resolution for a whole-cell artifact; a "
                             "cycle-level one records its own identity.")
    parser.add_argument("--rows", type=int, default=10,
                        help="how many prediction rows to print.")
    args = parser.parse_args(argv)

    from batteryml_web.viz.loaders import VizDataError, load_prediction_artifact

    path = find_artifact(Path(args.target))
    try:
        artifact = load_prediction_artifact(
            path, config_path=Path(args.config) if args.config else None
        )
    except VizDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    metadata = artifact.task_metadata or {}
    print(f"artifact        : {path.as_posix()}")
    print(f"sha256          : {artifact.source_sha256}")
    print(f"seed            : {artifact.seed}")
    print(f"task            : {metadata.get('task') or '(not recorded)'}")
    print(f"label annotator : {metadata.get('label_annotator') or '(not recorded)'}")
    print(f"feature         : {metadata.get('feature_extractor') or '(not recorded)'}")
    print(f"reference mode  : {metadata.get('reference_capacity_mode') or '(not recorded)'}")
    print(
        "sample policy   : "
        f"min_cycle={metadata.get('min_cycle')} "
        f"max_cycle={metadata.get('max_cycle')} "
        f"cycle_stride={metadata.get('cycle_stride')}"
    )
    print(f"label transform : {artifact.label_transformations or '(none)'}")
    print()

    print(f"test predictions: {artifact.n_predictions}")
    if not artifact.is_cycle_level:
        print(
            "sample identity : NOT RECORDED — this artifact has no "
            "sample_index, so it is a whole-cell run (e.g. RUL), not a "
            "cycle-level SOH one."
        )
    else:
        records = artifact.sample_index or []
        cell_ids = [record["cell_id"] for record in records]
        cycles = [record["cycle_index"] for record in records]
        print(f"unique test cells: {len(set(cell_ids))}")
        print(f"cycle range      : {min(cycles)} .. {max(cycles)}")
    print()

    print(f"observed SOH    : {_summary(artifact.y_true)}")
    print(f"predicted SOH   : {_summary(artifact.y_pred)}")
    print()

    print("recorded metrics (the evaluator's own values):")
    for name in sorted(artifact.scores):
        print(f"    {name:<6} {artifact.scores[name]:.6f}")
    unexpected = sorted(set(artifact.scores) - {"MAE", "RMSE"})
    if unexpected:
        print(
            f"    note: {', '.join(unexpected)} also recorded. Canonical SOH "
            "metrics are MAE and RMSE; MAPE is not inherited from the RUL set."
        )
    if artifact.score_check:
        print()
        print("recomputed from the loaded arrays (proves the inverse transform):")
        for name, check in sorted(artifact.score_check.items()):
            print(
                f"    {name:<6} recorded {check['recorded']:.6f}   "
                f"recomputed {check['recomputed']:.6f}   "
                f"rel. diff {check['relative_difference']:.2e}"
            )

    if artifact.is_cycle_level and args.rows > 0:
        print()
        print(f"first {min(args.rows, artifact.n_predictions)} prediction(s):")
        header = f"    {'cell_id':<18} {'cycle':>7} {'observed':>10} {'predicted':>10} {'residual':>10}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        records = artifact.sample_index or []
        for i in range(min(args.rows, artifact.n_predictions)):
            observed = float(artifact.y_true[i])
            predicted = float(artifact.y_pred[i])
            print(
                f"    {records[i]['cell_id']:<18} {records[i]['cycle_index']:>7} "
                f"{observed:>10.4f} {predicted:>10.4f} {predicted - observed:>+10.4f}"
            )

    print()
    print(
        "Workflow validation only. One Ridge run says the route works end to "
        "end; it is not a measurement of how well SOH can be estimated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
