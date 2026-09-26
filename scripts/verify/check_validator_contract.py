"""VAL-01 — the execution gate accepts every config the pipeline can build.

`CLAUDE.md` §3 makes validators the gate on execution. This check runs the real
``batteryml_web.core.config_validator.validate_config`` over every run config
and reports which ones it would refuse.

The point is agreement, not just acceptance. ``CFG-01`` establishes that a
config's declared names resolve; ``VAL-01`` establishes that the gate agrees.
A config that ``CFG-01`` accepts and ``VAL-01`` refuses means the validator has
drifted from the registries — exactly the failure that a hand-maintained
whitelist used to cause, when the gate accepted 4 of 130 configs.

Runs with ``require_data=False`` so it judges structure only: whether the
processed data is present is ``DATA-01``'s finding, not the validator's, and
this check must work with no data. It passes no ``output_dir``, so the
writability probe never runs and nothing is written.
"""

from __future__ import annotations

import argparse
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

CHECK_ID = "VAL-01"
CHECK_NAME = "validator_contract"

# Directories under configs/ that are not run configs.
NON_RUN_DIRS = ("cyclers",)


def _extension_key_of(path) -> str:
    """The extension module a config's ``model.name`` belongs to, or "".

    Parses the YAML directly rather than trusting the validator's error
    text: matching on a message string would break the moment the wording
    changed, and the wording is exactly what this check is improving.
    """
    try:
        import yaml

        from batteryml_web.core.config_registry import MODEL_CLASS_KEYS

        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        model = loaded.get("model")
        name = model.get("name") if isinstance(model, dict) else None
        return MODEL_CLASS_KEYS.get(str(name or "").strip().lower(), "")
    except Exception:  # noqa: BLE001 - classification only, never fail here
        return ""


def _unavailable_external_models() -> dict[str, str]:
    """Extensions skipped because their vendored third-party source is absent."""
    try:
        from batteryml_web.models import unavailable_extension_models

        return dict(unavailable_extension_models())
    except Exception:  # noqa: BLE001 - classification only
        return {}


def check(output_dir: Path | None = None) -> CheckResult:
    """Run the real validator over every config and report what it refuses."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"
    configs_root = root / "configs"

    if not configs_root.is_dir():
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{relative_to_repo(configs_root)} does not exist; there are no configs "
            "to put through the validator.",
            {"configs_root": relative_to_repo(configs_root)},
        )

    validator, reason = try_import("batteryml_web.core.config_validator")
    if validator is None:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            SKIP,
            f"batteryml_web.core.config_validator is not importable ({reason}), so the "
            "execution gate cannot be exercised. Fix the environment first (see "
            "ENV-01), then re-run this check.",
            {"import_error": reason},
        )

    paths = [
        path
        for path in sorted(configs_root.rglob("*.yaml"))
        if path.relative_to(configs_root).parts[0] not in NON_RUN_DIRS
    ]

    # The validator resolves names against the live registries, which needs
    # batteryml importable. Ask it once, before judging any config, so that a
    # missing dependency reads as SKIP rather than as 130 rejected configs.
    registries, registry_error = validator.registry_status()
    if registry_error is not None:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            SKIP,
            "The validator cannot read the BatteryML registries, so every config would "
            f"be refused for the same reason rather than on its own merits: "
            f"{registry_error} Fix the environment first (see ENV-01).",
            {
                "registry_error": registry_error,
                "run_config_count": len(paths),
            },
        )

    accepted: list[str] = []
    refused: list[dict[str, Any]] = []
    gated_on_external: list[dict[str, Any]] = []
    with_warnings: list[dict[str, Any]] = []

    unavailable = _unavailable_external_models()

    for path in paths:
        report = validator.load_and_validate_config_file(path, require_data=False)
        relative = relative_to_repo(path)
        if report.get("valid"):
            accepted.append(relative)
        else:
            # Two different things look identical here, and only one is a
            # contract violation. This check exists to catch the gate
            # *drifting* from the registries — refusing something the
            # pipeline could actually build. A model whose vendored
            # third-party source was never cloned cannot be built by anyone,
            # so refusing it is the gate working correctly, not drift.
            # Separated rather than ignored: it is still counted and still
            # named in the summary.
            source = unavailable.get(_extension_key_of(path))
            entry = {"config": relative, "errors": list(report.get("errors", []))}
            if source is not None:
                entry["requires"] = source
                gated_on_external.append(entry)
            else:
                refused.append(entry)
        warnings = list(report.get("warnings", []))
        if warnings:
            with_warnings.append({"config": relative, "warnings": warnings})

    payload = {
        "check_id": CHECK_ID,
        "validator": "batteryml_web.core.config_validator.validate_config",
        "require_data": False,
        "registered_counts": {
            attribute: len(names) for attribute, names in sorted(registries.items())
        },
        "run_config_count": len(paths),
        "accepted_count": len(accepted),
        "refused": refused,
        "gated_on_external_source": gated_on_external,
        "warnings": with_warnings,
        "accepted": accepted,
    }
    artifacts = [write_json(output_dir / "validator_contract.json", payload)]

    evidence = {
        "run_config_count": len(paths),
        "accepted_count": len(accepted),
        "refused_count": len(refused),
        "refused": refused,
        "gated_on_external_source": gated_on_external,
        "warning_count": len(with_warnings),
        "registered_counts": payload["registered_counts"],
    }

    if refused:
        first = refused[0]
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"The execution gate refuses {len(refused)} of {len(paths)} run config(s). "
            f"First: {first['config']} — {first['errors'][0]} "
            "Either the config is genuinely wrong (CFG-01 will say so too) or the "
            "validator has drifted from the registries; see evidence.refused.",
            evidence,
            artifacts,
        )

    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"The execution gate accepts all {len(paths) - len(gated_on_external)} "
        f"buildable run config(s) of {len(paths)} "
        f"[{len(registries.get('MODELS', ()))} models registered]. "
        f"{len(with_warnings)} carry warnings, which do not block a run. "
        + (
            f"{len(gated_on_external)} config(s) are correctly refused because a "
            f"third-party source is not vendored here "
            f"({', '.join(sorted({entry['requires'] for entry in gated_on_external}))}); "
            f"see evidence.gated_on_external_source."
            if gated_on_external
            else "No config is gated on an unvendored third-party source."
        ),
        evidence,
        artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for validator_contract.json (default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
