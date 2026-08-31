"""SPLIT-01 — which processed datasets each split preset needs, and which are missing.

The dataset requirement is derived from two independent sources and the two are
cross-checked against each other:

1. **the splitter classes** — each splitter in ``batteryml/train_test_split/``
   hard-codes the cell IDs of its held-out (or held-in) split. The dataset a
   cell belongs to is the prefix of its ID (``RWTH_015`` -> RWTH,
   ``UL-PUR_N15-...`` -> UL_PUR). ``MATR_split.py`` and ``HUST_split.py`` use
   bare IDs (``b1c0``, ``1-3``) because their splitters resolve the prefix at
   load time, so for those the dataset comes from the module name.
   Read by parsing the source with ``ast``; nothing is imported, so this works
   without torch.

2. **the configs** — ``train_test_split.cell_data_path`` names the directories
   the splitter is actually pointed at. This is the operationally binding set:
   a splitter only ever sees cells under these paths.

A disagreement between the two is reported explicitly rather than silently
resolved: source 1 can only see datasets that contribute *test* cells, while
source 2 also covers datasets that contribute training cells only.

Never downloads. Reports SKIP when there is no processed data to compare against.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import yaml

from .common import (
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    known_datasets,
    relative_to_repo,
    repo_root,
    run_check,
    write_json,
)

CHECK_ID = "SPLIT-01"
CHECK_NAME = "split_requirements"

SPLIT_DIR = Path("batteryml") / "train_test_split"
PROCESSED_GLOB = "*.pkl"

# Names inside a splitter's __init__ that hold cell-ID literals.
ID_LIST_NAMES = ("test_ids", "train_ids", "valid_ids")


def _dataset_from_cell_id(cell_id: str, datasets: list[str]) -> str | None:
    """Map a cell ID to its dataset by prefix, tolerating '-' for '_'.

    ``UL-PUR_N15-OV3_...`` belongs to the ``UL_PUR`` dataset, so the comparison
    is done on a separator-folded form. Longest prefix wins, so ``UL_PUR`` is
    preferred over a hypothetical ``UL``.
    """
    folded = cell_id.replace("-", "_").upper()
    best: str | None = None
    for dataset in datasets:
        key = dataset.replace("-", "_").upper()
        if folded.startswith(key + "_") or folded == key:
            if best is None or len(key) > len(best.replace("-", "_")):
                best = dataset
    return best


def _string_literals(node: ast.AST) -> list[str]:
    """Every string constant inside a list/tuple literal."""
    values: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
    return values


def _splitters_from_source(datasets: list[str]) -> dict[str, dict[str, Any]]:
    """Parse every registered splitter class out of batteryml/train_test_split/.

    Returns splitter class name -> {module, datasets_from_ids, id_count,
    unprefixed_id_count, fallback_used}.
    """
    root = repo_root()
    split_dir = root / SPLIT_DIR
    result: dict[str, dict[str, Any]] = {}

    if not split_dir.is_dir():
        return result

    for source_path in sorted(split_dir.glob("*.py")):
        if source_path.name in {"__init__.py", "base.py"}:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        # HUST_split.py -> HUST, MIX100_split.py -> MIX100
        module_dataset = source_path.stem.split("_split")[0]

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            literals: list[str] = []
            for statement in ast.walk(node):
                if not isinstance(statement, ast.Assign):
                    continue
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id in ID_LIST_NAMES:
                        literals.extend(_string_literals(statement.value))

            if not literals:
                continue

            found: set[str] = set()
            unprefixed = 0
            for cell_id in literals:
                dataset = _dataset_from_cell_id(cell_id, datasets)
                if dataset is None:
                    unprefixed += 1
                else:
                    found.add(dataset)

            fallback_used = False
            if not found and module_dataset in datasets:
                # MATR/HUST splitters store bare IDs; the module names the dataset.
                found.add(module_dataset)
                fallback_used = True

            result[node.name] = {
                "module": relative_to_repo(source_path),
                "line": node.lineno,
                "datasets_from_ids": sorted(found),
                "id_count": len(literals),
                "unprefixed_id_count": unprefixed,
                "module_name_fallback_used": fallback_used,
            }

    return result


def _presets_from_configs(root: Path) -> dict[str, dict[str, Any]]:
    """Preset name -> {splitter, datasets_from_configs, configs, conflicts}.

    A preset is a config *stem* (``matr_1``, ``cruh``, ...); every model
    directory under configs/baselines/ carries the same eight.
    """
    presets: dict[str, dict[str, Any]] = {}
    baselines = root / "configs" / "baselines"
    if not baselines.is_dir():
        return presets

    for path in sorted(baselines.rglob("*.yaml")):
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            # CFG-01 owns unparsable configs; here it simply contributes nothing.
            continue
        if not isinstance(loaded, dict):
            continue
        split = loaded.get("train_test_split")
        if not isinstance(split, dict) or not split.get("name"):
            continue

        preset = path.stem
        entry = presets.setdefault(
            preset,
            {
                "preset": preset,
                "splitters": set(),
                "datasets_from_configs": set(),
                "configs": [],
            },
        )
        entry["splitters"].add(str(split["name"]))
        entry["configs"].append(relative_to_repo(path))

        value = split.get("cell_data_path")
        paths = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
        for declared in paths:
            entry["datasets_from_configs"].add(Path(str(declared)).name)

    return presets


def check(output_dir: Path | None = None) -> CheckResult:
    """Report per-preset dataset requirements and which processed datasets are missing."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"

    datasets, dataset_source = known_datasets()
    splitters = _splitters_from_source(datasets)
    presets = _presets_from_configs(root)

    if not presets:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            "No split presets found: no config under configs/baselines/ declares a "
            "train_test_split.name.",
            {"dataset_source": dataset_source, "splitters_parsed": sorted(splitters)},
        )

    processed_root = root / "data" / "processed"
    available: set[str] = set()
    if processed_root.is_dir():
        for candidate in processed_root.iterdir():
            if candidate.is_dir() and any(candidate.glob(PROCESSED_GLOB)):
                available.add(candidate.name)

    rows: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for preset in sorted(presets):
        entry = presets[preset]
        splitter_names = sorted(entry["splitters"])
        from_configs = sorted(entry["datasets_from_configs"])

        from_ids: set[str] = set()
        splitter_details = []
        for name in splitter_names:
            detail = splitters.get(name)
            if detail is None:
                splitter_details.append({"splitter": name, "parsed": False})
                continue
            splitter_details.append({"splitter": name, "parsed": True, **detail})
            from_ids.update(detail["datasets_from_ids"])

        # The config paths bind what the splitter can see, so they are the
        # requirement; the ID-derived set is the cross-check.
        required = sorted(set(from_configs) or from_ids)
        missing = sorted(dataset for dataset in required if dataset not in available)

        only_in_configs = sorted(set(from_configs) - from_ids)
        only_in_ids = sorted(from_ids - set(from_configs))
        if only_in_configs or only_in_ids:
            disagreements.append(
                {
                    "preset": preset,
                    "from_configs_only": only_in_configs,
                    "from_splitter_ids_only": only_in_ids,
                    "note": (
                        "datasets in from_configs_only contribute training cells but no "
                        "held-out cell IDs, so the splitter source alone cannot see them"
                    ),
                }
            )

        rows.append(
            {
                "preset": preset,
                "splitters": splitter_names,
                "splitter_details": splitter_details,
                "datasets_from_splitter_ids": sorted(from_ids),
                "datasets_from_configs": from_configs,
                "required": required,
                "available": sorted(dataset for dataset in required if dataset in available),
                "missing": missing,
                "ready": not missing,
                "config_count": len(entry["configs"]),
            }
        )

    payload = {
        "check_id": CHECK_ID,
        "dataset_source": dataset_source,
        "processed_root": relative_to_repo(processed_root),
        "processed_available": sorted(available),
        "presets": rows,
        "source_disagreements": disagreements,
        "splitters_parsed": splitters,
    }
    artifacts = [write_json(output_dir / "split_requirements.json", payload)]

    ready = [row["preset"] for row in rows if row["ready"]]
    not_ready = [row["preset"] for row in rows if not row["ready"]]
    evidence = {
        "dataset_source": dataset_source,
        "preset_count": len(rows),
        "processed_available": sorted(available),
        "ready_presets": ready,
        "not_ready_presets": {
            row["preset"]: row["missing"] for row in rows if not row["ready"]
        },
        "requirements": {row["preset"]: row["required"] for row in rows},
        "source_disagreements": disagreements,
    }

    if not processed_root.is_dir():
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            SKIP,
            f"{relative_to_repo(processed_root)} does not exist, so no preset can be "
            f"reported as ready. Requirements for all {len(rows)} preset(s) were still "
            "derived and written; see the artifact.",
            evidence,
            artifacts,
        )

    summary = (
        f"{len(ready)} of {len(rows)} preset(s) have all required processed datasets"
        f"{' (' + ', '.join(ready) + ')' if ready else ''}."
    )
    if not_ready:
        detail = "; ".join(
            f"{row['preset']} needs {', '.join(row['missing'])}"
            for row in rows
            if not row["ready"]
        )
        summary += f" Not ready: {detail}."
    if disagreements:
        summary += (
            f" {len(disagreements)} preset(s) where the splitter-ID and config-path "
            "sources disagree; see evidence.source_disagreements."
        )
    return CheckResult(CHECK_ID, CHECK_NAME, PASS, summary, evidence, artifacts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for split_requirements.json (default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
