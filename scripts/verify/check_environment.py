"""ENV-01 — Python version, declared packages, batteryml importability, GPU visibility.

Emits ``environment_check.json`` and ``package_versions.txt`` into the output
directory (default ``runs/verify/``). The filenames are the ones the
reproducibility contract in CLAUDE.md §8 quotes verbatim.

Never installs, upgrades or downloads anything.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .common import (
    FAIL,
    PASS,
    CheckResult,
    repo_root,
    run_check,
    try_import,
    write_json,
    write_text,
)

CHECK_ID = "ENV-01"
CHECK_NAME = "environment"

# Lowest interpreter this repository is expected to run on. The codebase uses
# PEP 604 unions at runtime in dataclass fields (batteryml_web/core/
# config_registry.py) and the match-free 3.9 syntax elsewhere; 3.9 is the
# floor, 3.10+ is what the code is written against.
MINIMUM_PYTHON = (3, 9)

# `name[extras]<specifier>` with an optional trailing comment.
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"(?P<specifier>.*)$"
)


def parse_requirements(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Parse requirements.txt into ``[{name, specifier, raw}]`` plus unparsed lines.

    Deliberately does not implement PEP 440. It extracts the distribution name
    and keeps the specifier as text so the report can show declared-vs-installed
    without this harness taking on a `packaging` dependency.
    """
    requirements: list[dict[str, str]] = []
    unparsed: list[str] = []

    if not path.exists():
        return requirements, unparsed

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            # -r other.txt / -e . / --index-url ... : not a distribution.
            unparsed.append(raw_line.strip())
            continue
        match = _REQUIREMENT_RE.match(line)
        if not match:
            unparsed.append(raw_line.strip())
            continue
        requirements.append(
            {
                "name": match.group("name"),
                "specifier": match.group("specifier").strip(),
                "raw": line,
            }
        )
    return requirements, unparsed


def _installed_version(distribution: str) -> str | None:
    """Installed version of a distribution, or None. Does not import the package."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a broken dist-info must not abort the check
        return None


def _pinned_mismatch(specifier: str, installed: str | None) -> bool:
    """True only for an unambiguous ``==`` pin that the installed version violates.

    Range specifiers (``>=``, ``<``) are reported but never judged here, because
    judging them correctly needs PEP 440 comparison and this harness is
    stdlib-only.
    """
    if installed is None or not specifier.startswith("=="):
        return False
    pinned = specifier[2:].split(",", 1)[0].strip()
    return bool(pinned) and pinned != installed


def _gpu_report() -> dict[str, Any]:
    """Whether a GPU is visible, via torch and via nvidia-smi. Absence is not an error."""
    report: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "nvidia_smi_on_path": shutil.which("nvidia-smi") is not None,
        "nvidia_smi_query": None,
        "notes": [],
    }

    torch, reason = try_import("torch")
    if torch is None:
        report["notes"].append(f"torch not importable ({reason})")
    else:
        report["torch_installed"] = True
        report["torch_version"] = getattr(torch, "__version__", None)
        try:
            report["cuda_available"] = bool(torch.cuda.is_available())
            if report["cuda_available"]:
                report["device_count"] = int(torch.cuda.device_count())
                report["device_names"] = [
                    torch.cuda.get_device_name(i) for i in range(report["device_count"])
                ]
        except Exception as exc:  # noqa: BLE001
            report["notes"].append(f"torch.cuda probe failed: {type(exc).__name__}: {exc}")

    if report["nvidia_smi_on_path"]:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            report["nvidia_smi_query"] = (
                proc.stdout.strip() if proc.returncode == 0 else f"exit {proc.returncode}: {proc.stderr.strip()[:400]}"
            )
        except subprocess.TimeoutExpired:
            report["nvidia_smi_query"] = "nvidia-smi timed out after 15s"
        except OSError as exc:
            report["nvidia_smi_query"] = f"{type(exc).__name__}: {exc}"

    return report


def check(output_dir: Path | None = None) -> CheckResult:
    """Report the interpreter, declared packages, batteryml importability and GPU."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"

    python_info = {
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "minimum_required": ".".join(str(p) for p in MINIMUM_PYTHON),
        "meets_minimum": sys.version_info[:2] >= MINIMUM_PYTHON,
    }

    requirements_path = root / "requirements.txt"
    requirements, unparsed = parse_requirements(requirements_path)

    packages: list[dict[str, Any]] = []
    missing: list[str] = []
    pin_mismatches: list[str] = []
    for requirement in requirements:
        installed = _installed_version(requirement["name"])
        mismatch = _pinned_mismatch(requirement["specifier"], installed)
        packages.append(
            {
                "name": requirement["name"],
                "declared": requirement["specifier"] or "(unpinned)",
                "installed": installed,
                "present": installed is not None,
                "pin_mismatch": mismatch,
            }
        )
        if installed is None:
            missing.append(requirement["name"])
        elif mismatch:
            pin_mismatches.append(
                f"{requirement['name']} declared {requirement['specifier']}, installed {installed}"
            )

    batteryml, batteryml_reason = try_import("batteryml")
    batteryml_info = {
        "importable": batteryml is not None,
        "path": getattr(batteryml, "__file__", None),
        "error": batteryml_reason,
    }

    gpu = _gpu_report()

    payload = {
        "check_id": CHECK_ID,
        "python": python_info,
        "requirements_file": "requirements.txt" if requirements_path.exists() else None,
        "packages": packages,
        "unparsed_requirement_lines": unparsed,
        "batteryml": batteryml_info,
        "gpu": gpu,
    }

    artifacts = [
        write_json(output_dir / "environment_check.json", payload),
        write_text(
            output_dir / "package_versions.txt",
            _package_versions_text(python_info, packages, batteryml_info, gpu),
        ),
    ]

    problems: list[str] = []
    if not python_info["meets_minimum"]:
        problems.append(
            f"Python {python_info['version']} is below the minimum {python_info['minimum_required']}"
        )
    if not requirements_path.exists():
        problems.append("requirements.txt is missing")
    if missing:
        problems.append(f"{len(missing)} declared package(s) not installed: {', '.join(sorted(missing))}")
    if not batteryml_info["importable"]:
        problems.append(f"batteryml is not importable ({batteryml_reason})")

    evidence = {
        "python": python_info,
        "package_count": len(packages),
        "missing_packages": sorted(missing),
        "pin_mismatches": pin_mismatches,
        "batteryml": batteryml_info,
        "gpu_visible": bool(gpu["cuda_available"]),
        "gpu": gpu,
    }

    if problems:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            "; ".join(problems)
            + ". Install the missing distributions with `pip install -r requirements.txt`"
            " and re-run this check.",
            evidence,
            artifacts,
        )

    gpu_note = (
        f"GPU visible ({gpu['device_count']} device(s))"
        if gpu["cuda_available"]
        else "no GPU visible (cpu-only is sufficient for the Route B demo)"
    )
    pin_note = f"; {len(pin_mismatches)} pin mismatch(es)" if pin_mismatches else ""
    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"Python {python_info['version']}, {len(packages)} declared package(s) installed, "
        f"batteryml importable, {gpu_note}{pin_note}.",
        evidence,
        artifacts,
    )


def _package_versions_text(
    python_info: dict[str, Any],
    packages: list[dict[str, Any]],
    batteryml_info: dict[str, Any],
    gpu: dict[str, Any],
) -> str:
    lines = [
        "# BatteryML package versions",
        f"# generated by scripts/verify/check_environment.py ({CHECK_ID})",
        "",
        f"python                {python_info['version']}  ({python_info['implementation']})",
        f"platform              {python_info['platform']}",
        f"executable            {python_info['executable']}",
        f"batteryml importable  {batteryml_info['importable']}",
        f"torch                 {gpu['torch_version'] or 'not installed'}",
        f"cuda available        {gpu['cuda_available']}",
        "",
        "# declared in requirements.txt",
    ]
    width = max((len(p["name"]) for p in packages), default=1)
    for package in packages:
        installed = package["installed"] or "NOT INSTALLED"
        declared = package["declared"]
        lines.append(f"{package['name']:<{width}}  {installed:<16}  declared: {declared}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for environment_check.json and package_versions.txt "
             "(default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
