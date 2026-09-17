"""FIG-01 — the figure contract holds, and the builders actually render.

Four things are established, none of which needs a dataset, a trained model or
a GPU:

1. **The contract is well formed.** Every :class:`FigureSpec` in
   ``batteryml_web/viz/contract.py`` has a unique id, names a builder that
   exists and is callable in ``batteryml_web/viz/figures.py``, and renders to a
   filename that is safe and unique. ``CLAUDE.md`` §8 makes these filenames part
   of the manuscript, so a collision or a stray path separator is a defect in
   the paper, not only in the code.

2. **The dispatch covers the contract.** ``batteryml_web/viz/cli.py`` must have a
   branch for every declared figure id. Read from the source with ``ast``, so a
   spec added without a branch is caught before it raises at runtime.

3. **The builders render.** Each figure is drawn from a tiny synthetic array and
   encoded to PNG **in memory**. This is the only claim in this repository that
   can be established by execution rather than by reading, so it is worth
   making: an import error, a matplotlib API change or a typo in a builder
   surfaces here instead of on the server.

4. **The rendering is deterministic.** Each figure is rendered twice and the two
   byte strings are compared. ``CLAUDE.md`` §8 requires identical inputs to
   produce identical artifacts; matplotlib stamps timestamps into some formats,
   and this is what proves they were suppressed.

**On the synthetic arrays** (``CLAUDE.md`` §7): the smoke inputs below are
constant literals, they are rendered to ``io.BytesIO`` and never to disk, they
are never written into ``figures/``, ``metrics.csv`` or a report, and the
dataset registry cannot reach them. They exist to exercise the drawing code, and
the numbers in them are not results and are not presented as any.

When ``--output-dir`` holds a ``figure_manifest.json``, the check additionally
audits it: every listed figure exists at the recorded hash, and every figure
file in the directory is listed.

Never trains, never evaluates, never writes into ``data/``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .common import (
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    relative_to_repo,
    repo_root,
    run_check,
    try_import,
    write_json,
)

CHECK_ID = "FIG-01"
CHECK_NAME = "figure_contract"

CLI_SOURCE = Path("batteryml_web") / "viz" / "cli.py"
MANIFEST_FILENAME = "figure_manifest.json"
FIGURE_SUFFIXES = (".png", ".svg", ".pdf")

# A contracted filename: one subdirectory, then a safe basename and a known
# extension. Anything else could not be quoted verbatim by the manuscript.
FILENAME_PATTERN = re.compile(r"^(results|data)/[A-Za-z0-9._+-]+\.(png|svg|pdf)$")

# Synthetic smoke inputs. Not data, not results — see the module docstring.
_SYNTHETIC_SMOKE_TRUE = (150.0, 320.0, 480.0, 640.0, 900.0, 1180.0)
_SYNTHETIC_SMOKE_PRED = (168.0, 300.0, 505.0, 601.0, 962.0, 1103.0)
_SYNTHETIC_SMOKE_CELLS = ("cell-a", "cell-b", "cell-c", "cell-d", "cell-e", "cell-f")
_SYNTHETIC_SMOKE_SCORES = {"RMSE": 44.5, "MAE": 36.0, "MAPE": 0.071}

_SMOKE_IDENTITY = {"dataset": "SMOKE", "model": "smoke", "seed": 0, "fmt": "png"}


def _dispatched_figure_ids(cli_path: Path) -> tuple[set[str], str | None]:
    """Figure ids the CLI has a branch for, read statically from its source.

    Looks for comparisons of the form ``spec.figure_id == "<id>"``. Parsing
    rather than importing keeps this working when matplotlib is absent.

    Returns:
        ``(ids, error)`` — ``error`` is a message when the file could not be
        read or parsed, in which case ``ids`` is empty.
    """
    if not cli_path.is_file():
        return set(), f"{relative_to_repo(cli_path)} does not exist."
    try:
        tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return set(), f"cannot parse {relative_to_repo(cli_path)}: {type(exc).__name__}: {exc}"

    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not node.ops:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == "figure_id"):
            continue
        comparator = node.comparators[0]
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
            found.add(comparator.value)
    return found, None


def _smoke_render(figures: Any, spec_id: str, theme: Any) -> Any:
    """Build one figure from the synthetic smoke inputs.

    Raises:
        KeyError: for a figure id this function does not know how to build. That
            is itself a finding: a new contracted figure needs a smoke case here.
    """
    import numpy as np

    true_arr = np.asarray(_SYNTHETIC_SMOKE_TRUE, dtype=np.float64)
    pred_arr = np.asarray(_SYNTHETIC_SMOKE_PRED, dtype=np.float64)
    residuals = pred_arr - true_arr

    if spec_id == "parity":
        return figures.parity_figure(
            true_arr, pred_arr, theme=theme, title="smoke", subtitle="synthetic smoke input",
            unit_label="cycles", cell_ids=list(_SYNTHETIC_SMOKE_CELLS), dpi=72.0,
        )
    if spec_id == "residuals":
        return figures.residual_figure(
            true_arr, residuals, theme=theme, title="smoke", subtitle="synthetic smoke input",
            unit_label="cycles", dpi=72.0,
        )
    if spec_id == "error_by_cell":
        return figures.error_by_cell_figure(
            list(_SYNTHETIC_SMOKE_CELLS), np.abs(residuals), theme=theme, title="smoke",
            subtitle="synthetic smoke input", unit_label="cycles", dpi=72.0,
        )
    if spec_id == "soh_vs_cycle":
        # Two cells, three observed cycles each, on the SOH ratio scale. The
        # figure needs one (cell, cycle) per point, so the smoke input is
        # built that way rather than reusing the per-cell arrays above.
        cells = ["cell-a"] * 3 + ["cell-b"] * 3
        cycles = np.array([10, 20, 30, 10, 20, 30])
        observed = np.array([0.99, 0.94, 0.88, 0.97, 0.90, 0.81])
        predicted = observed + np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.005])
        return figures.soh_vs_cycle_figure(
            cells, cycles, observed, predicted, theme=theme, title="smoke",
            subtitle="synthetic smoke input", unit_label="SOH (ratio)", dpi=72.0,
        )
    if spec_id == "metrics_summary":
        return figures.metric_summary_figure(
            dict(_SYNTHETIC_SMOKE_SCORES), theme=theme, title="smoke",
            subtitle="synthetic smoke input",
            metric_units={"RMSE": "cycles", "MAE": "cycles", "MAPE": "ratio"}, dpi=72.0,
        )
    raise KeyError(spec_id)


def _audit_manifest(output_dir: Path) -> dict[str, Any] | None:
    """Cross-check a written manifest against the files on disk.

    Returns ``None`` when there is no manifest to audit — a check with nothing
    to observe reports SKIP for that part rather than inventing a finding.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "manifest": relative_to_repo(manifest_path),
            "problems": [f"cannot read the manifest: {type(exc).__name__}: {exc}"],
            "listed": 0,
            "unlisted": [],
        }

    problems: list[str] = []
    listed_names: set[str] = set()
    entries = manifest.get("figures")
    if not isinstance(entries, list):
        problems.append("'figures' is missing or is not a list")
        entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(f"figure entry is a {type(entry).__name__}, expected an object")
            continue
        name = entry.get("filename")
        if not name:
            problems.append("a figure entry has no 'filename'")
            continue
        listed_names.add(str(name))
        path = output_dir / str(name)
        if not path.is_file():
            problems.append(f"{name}: listed in the manifest but missing from disk")
            continue
        recorded = str(entry.get("sha256", ""))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if recorded and recorded != actual:
            problems.append(
                f"{name}: manifest records sha256 {recorded[:12]}... but the file hashes "
                f"to {actual[:12]}.... The file changed after it was recorded."
            )

    on_disk = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES
    }
    unlisted = sorted(on_disk - listed_names)
    for name in unlisted:
        problems.append(f"{name}: present in the figures directory but not listed in the manifest")

    return {
        "manifest": relative_to_repo(manifest_path),
        "listed": len(listed_names),
        "on_disk": len(on_disk),
        "unlisted": unlisted,
        "problems": problems,
    }


def check(output_dir: str | Path | None = None) -> CheckResult:
    """Run FIG-01.

    Args:
        output_dir: where ``figure_contract.json`` is written, and where a
            ``figure_manifest.json`` is looked for. Defaults to ``runs/verify/``.

    Returns:
        The check result.
    """
    root = repo_root()
    out_dir = Path(output_dir) if output_dir else root / "runs" / "verify"

    for module_name in ("numpy", "matplotlib"):
        module, error = try_import(module_name)
        if module is None:
            return CheckResult(
                CHECK_ID, CHECK_NAME, SKIP,
                f"{module_name} is not importable ({error}), so no figure can be drawn and "
                "the contract cannot be exercised. Install it with "
                "`pip install -r requirements.txt`; ENV-01 reports what is missing.",
                {"missing_module": module_name, "import_error": error},
            )

    contract, contract_error = try_import("batteryml_web.viz.contract")
    figures, figures_error = try_import("batteryml_web.viz.figures")
    render, render_error = try_import("batteryml_web.viz.render")
    style, style_error = try_import("batteryml_web.viz.style")
    import_errors = {
        "batteryml_web.viz.contract": contract_error,
        "batteryml_web.viz.figures": figures_error,
        "batteryml_web.viz.render": render_error,
        "batteryml_web.viz.style": style_error,
    }
    if any(module is None for module in (contract, figures, render, style)):
        broken = sorted(name for name, err in import_errors.items() if err)
        return CheckResult(
            CHECK_ID, CHECK_NAME, FAIL,
            f"the visualisation package does not import: {', '.join(broken)}. Nothing can "
            "draw a figure until this is fixed; see evidence.import_errors.",
            {"import_errors": import_errors},
        )

    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    specs = list(contract.ALL_FIGURES)
    if not specs:
        problems.append("the contract declares no figures at all")

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for spec in specs:
        row: dict[str, Any] = {
            "figure_id": spec.figure_id,
            "builder": spec.builder,
            "requires_cell_ids": spec.requires_cell_ids,
        }
        if spec.figure_id in seen_ids:
            problems.append(f"{spec.figure_id}: declared more than once")
        seen_ids.add(spec.figure_id)

        builder = getattr(figures, spec.builder, None)
        if builder is None or not callable(builder):
            problems.append(
                f"{spec.figure_id}: names builder {spec.builder!r}, which does not exist in "
                "batteryml_web/viz/figures.py"
            )
        try:
            name = contract.figure_filename(spec, **_SMOKE_IDENTITY)
            row["example_filename"] = name
            if not FILENAME_PATTERN.match(name):
                problems.append(
                    f"{spec.figure_id}: renders to {name!r}, which does not match the "
                    f"contracted pattern {FILENAME_PATTERN.pattern}"
                )
            if name in seen_names:
                problems.append(f"{spec.figure_id}: filename {name!r} collides with another figure")
            seen_names.add(name)
        except Exception as exc:
            problems.append(
                f"{spec.figure_id}: filename template {spec.template!r} does not render: "
                f"{type(exc).__name__}: {exc}"
            )
        rows.append(row)

    dispatched, dispatch_error = _dispatched_figure_ids(root / CLI_SOURCE)
    if dispatch_error:
        problems.append(dispatch_error)
    else:
        undispatched = sorted(spec.figure_id for spec in specs if spec.figure_id not in dispatched)
        for figure_id in undispatched:
            problems.append(
                f"{figure_id}: declared in the contract but has no branch in "
                f"{CLI_SOURCE.as_posix()}, so the CLI would raise on it"
            )
        stale = sorted(dispatched - seen_ids)
        for figure_id in stale:
            problems.append(
                f"{figure_id}: {CLI_SOURCE.as_posix()} dispatches on it but the contract "
                "does not declare it"
            )

    theme = style.get_theme("light")
    for row, spec in zip(rows, specs):
        try:
            first = render.render_to_bytes(
                _smoke_render(figures, spec.figure_id, theme), fmt="png", dpi=72.0
            )
            second = render.render_to_bytes(
                _smoke_render(figures, spec.figure_id, theme), fmt="png", dpi=72.0
            )
        except KeyError:
            problems.append(
                f"{spec.figure_id}: no smoke case in scripts/verify/check_figure_contract.py. "
                "A newly contracted figure needs one, or it ships unexercised."
            )
            row["rendered"] = False
            continue
        except Exception as exc:
            problems.append(
                f"{spec.figure_id}: rendering failed with {type(exc).__name__}: {exc}"
            )
            row["rendered"] = False
            continue

        row["rendered"] = True
        row["bytes"] = len(first)
        row["png_signature_ok"] = first.startswith(b"\x89PNG\r\n\x1a\n")
        row["deterministic"] = first == second
        if not row["png_signature_ok"]:
            problems.append(f"{spec.figure_id}: rendered bytes are not a PNG")
        if not row["deterministic"]:
            problems.append(
                f"{spec.figure_id}: two identical renders produced different bytes "
                f"(sha256 {hashlib.sha256(first).hexdigest()[:12]}... vs "
                f"{hashlib.sha256(second).hexdigest()[:12]}...). CLAUDE.md §8 requires "
                "identical inputs to produce identical artifacts; something "
                "non-deterministic — a timestamp or a salted id — is reaching the output."
            )

    audit = _audit_manifest(out_dir)
    if audit and audit["problems"]:
        problems.extend(audit["problems"])

    evidence: dict[str, Any] = {
        "figures": rows,
        "declared": len(specs),
        "dispatched_in_cli": sorted(dispatched),
        "manifest_audit": audit or "no figure_manifest.json in the output directory",
        "synthetic_smoke_inputs": {
            "rendered_to": "memory (io.BytesIO); never written to disk",
            "note": "constant literals used to exercise the drawing code; not results "
                    "(CLAUDE.md §7)",
        },
        "problems": problems,
    }
    artifacts = [write_json(out_dir / "figure_contract.json", evidence)]

    if problems:
        return CheckResult(
            CHECK_ID, CHECK_NAME, FAIL,
            f"{len(problems)} problem(s) in the figure contract; first: {problems[0]} "
            "See evidence.problems for the rest.",
            evidence, artifacts,
        )

    rendered = sum(1 for row in rows if row.get("rendered"))
    summary = (
        f"{len(specs)} contracted figure(s); {rendered} rendered to memory and byte-identical "
        "across two renders."
    )
    if audit:
        summary += f" Manifest audit: {audit['listed']} figure(s) listed, all present and hash-matched."
    return CheckResult(CHECK_ID, CHECK_NAME, PASS, summary, evidence, artifacts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figure_contract.json, and where a figure_manifest.json is "
             "looked for (default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
