"""May this dataset be used for an SOH run? Decided on its data, cached.

Having an SOH config for a dataset is not enough: SOH needs a valid reference
capacity per cell and enough cycles carrying both a discharge capacity (the
label) and a voltage trace (the features). :mod:`.soh_readiness` answers that
from the processed cells — but it unpickles every cell, which takes far too
long to repeat on every page rerun of a large dataset.

So the verdict is computed on request ("Check SOH data") and stored under
``data/metadata/soh_readiness/`` together with:

* the **label policy** it was computed for, read from the selected config's
  ``label:`` section (reference-capacity mode, ``min_cycle``, ``max_cycle``,
  ``cycle_stride``) — a different policy is a different question;
* a **fingerprint** of the processed folder (file count, total bytes, newest
  modification time) — re-preprocessing the data makes the stored verdict
  stale, and a stale verdict is never shown as current.

Nothing here fabricates a verdict: no stored file, a stale one or an
unreadable one all mean "not checked yet".
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .soh_readiness import SOH_READY, assess_soh_dataset

CACHE_SUBDIR = Path("data") / "metadata" / "soh_readiness"
PROCESSED_SUBDIR = Path("data") / "processed"


@dataclass(frozen=True)
class SOHPolicy:
    """The label sampling a verdict is valid for (from the config's ``label:``)."""

    reference_capacity_mode: Optional[str]
    min_cycle: int
    max_cycle: Optional[int]
    cycle_stride: int

    def key(self) -> str:
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def policy_from_config(config_path: str | Path) -> SOHPolicy:
    """Read the SOH label policy from a config file.

    Raises:
        ValueError: the file is not a mapping, has no ``label`` mapping, or a
            sampling value is not an integer — the caller shows the message.
    """
    path = Path(config_path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("label"), dict):
        raise ValueError(f"{path.as_posix()} has no `label:` section, so no SOH sample policy can be read from it.")
    label = loaded["label"]

    def _int(name: str, default: Optional[int]) -> Optional[int]:
        value = label.get(name, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path.as_posix()}: label.{name} = {value!r} is not an integer.")
        return value

    min_cycle = _int("min_cycle", 1)
    stride = _int("cycle_stride", 1)
    mode = label.get("reference_capacity_mode")
    return SOHPolicy(
        reference_capacity_mode=str(mode) if mode is not None else None,
        min_cycle=int(min_cycle if min_cycle is not None else 1),
        max_cycle=_int("max_cycle", None),
        cycle_stride=int(stride if stride is not None else 1),
    )


def processed_fingerprint(processed_dir: str | Path) -> Dict[str, Any]:
    """Cheap identity of a processed folder: ``*.pkl`` count, bytes, newest mtime."""
    files = 0
    total = 0
    newest = 0.0
    folder = Path(processed_dir)
    if folder.is_dir():
        for entry in folder.glob("*.pkl"):
            if entry.is_file():
                stat = entry.stat()
                files += 1
                total += stat.st_size
                newest = max(newest, stat.st_mtime)
    return {"files": files, "bytes": total, "newest_mtime": round(newest, 3)}


def cache_file(repo_root: str | Path, dataset: str, policy: SOHPolicy) -> Path:
    return Path(repo_root) / CACHE_SUBDIR / f"{dataset}__{policy.key()}.json"


def cached_verdict(repo_root: str | Path, dataset: str, policy: SOHPolicy) -> Optional[Dict[str, Any]]:
    """The stored verdict, or None when there is none or it no longer applies."""
    path = cache_file(repo_root, dataset, policy)
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stored, dict):
        return None
    current = processed_fingerprint(Path(repo_root) / PROCESSED_SUBDIR / dataset)
    if stored.get("fingerprint") != current or stored.get("policy") != asdict(policy):
        return None
    return stored


def check_and_store(repo_root: str | Path, dataset: str, policy: SOHPolicy) -> Dict[str, Any]:
    """Run the SOH readiness audit for ``dataset`` under ``policy`` and store it."""
    root = Path(repo_root)
    kwargs: Dict[str, Any] = {
        "processed_root": root / PROCESSED_SUBDIR,
        "min_cycle": policy.min_cycle,
        "max_cycle": policy.max_cycle,
        "cycle_stride": policy.cycle_stride,
    }
    if policy.reference_capacity_mode:
        kwargs["reference_capacity_mode"] = policy.reference_capacity_mode
    report = assess_soh_dataset(dataset, **kwargs)
    payload = report.to_dict()
    payload["policy"] = asdict(policy)
    payload["fingerprint"] = processed_fingerprint(root / PROCESSED_SUBDIR / dataset)
    payload["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    target = cache_file(root, dataset, policy)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return payload


def gate(verdict: Optional[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """``(allowed, message)``: True ready, False checked and not ready, None unchecked."""
    if verdict is None:
        return None, (
            "SOH runs need a data check first: SOH requires a valid reference "
            "capacity per cell and enough cycles with both discharge capacity "
            "and a voltage trace. Press 'Check SOH data'."
        )
    status = str(verdict.get("status") or "")
    counts = (
        f"{verdict.get('cells_eligible', 0)} of {verdict.get('cells_inspected', 0)} cell(s) "
        f"eligible, {verdict.get('total_samples', 0)} SOH sample(s)"
    )
    reason = str(verdict.get("reason") or "")
    if status == SOH_READY:
        return True, f"SOH data check passed ({status}): {counts}. {reason}".strip()
    return False, f"SOH data check did not pass ({status}): {counts}. {reason}".strip()
