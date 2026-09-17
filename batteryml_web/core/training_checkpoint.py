"""Checkpoints that can actually resume training, and the rules that guard them.

The existing ``.ckpt`` in this repository is ``torch.save(state_dict)`` — the
model's weights and nothing else (audit: ``docs/TRAINING_STATE_AUDIT.md``).
Restoring one and continuing gives a fresh optimizer, an epoch counter back
at zero and a different data order. That is a warm start. Calling it a resume
would misreport what the numbers mean, so this module draws the line
explicitly and refuses to blur it.

A **resumable** checkpoint carries everything needed for training to continue
as though it had never stopped:

* model and optimizer state — Adam's moment estimates are part of the
  trajectory, not an implementation detail;
* the scheduler and AMP scaler when the run has them (neither exists in the
  current loop, so both are ``None`` today rather than invented);
* the epoch already completed, so the continuation runs 41..100 and not
  another 100;
* the RNG state of every generator that affects training — Python, NumPy,
  torch CPU and each CUDA device — because the data order and any stochastic
  layer come from them;
* the identity of what was being trained, so a checkpoint cannot be resumed
  into a different model, dataset, split or seed.

Nothing here is secret-bearing: the fields are named explicitly and a
credential has no route in.

No Streamlit. torch is imported lazily so that a machine without it can still
import this module to classify a file.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Bumped when the stored structure changes incompatibly. A checkpoint from a
#: future version is refused rather than read optimistically.
SCHEMA_VERSION = 1

#: Marks a payload as ours. A bare ``state_dict`` has no such key, which is
#: exactly how a weights-only file is told apart from a resumable one.
SCHEMA_KEY = "batteryml_resume_schema"

KIND_RESUMABLE = "resumable"
KIND_WEIGHTS_ONLY = "weights_only"
KIND_UNREADABLE = "unreadable"
KIND_FUTURE_VERSION = "future_version"

WEIGHTS_ONLY_MESSAGE = (
    "This checkpoint contains model weights but not optimizer/epoch/RNG "
    "state and cannot reproduce an interrupted training continuation. It can "
    "still be used for inference, evaluation, or an explicitly requested "
    "warm start."
)

UNREADABLE_MESSAGE = (
    "This checkpoint could not be read back. It may be incomplete, or have "
    "been written by an incompatible environment."
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def config_fingerprint(config: Any) -> str:
    """A stable short digest of the training configuration.

    Serialized with sorted keys so that the same configuration always yields
    the same fingerprint regardless of dict ordering. Used to refuse a resume
    into a configuration the checkpoint was not produced under.
    """
    return _sha256(json.dumps(config, sort_keys=True, default=str))


def split_fingerprint(train_size: Any, test_size: Any, extra: Any = None) -> str:
    """A digest of the split's shape, so a different split cannot be resumed into.

    Sizes rather than contents: hashing the tensors themselves would read the
    whole dataset on every check. This catches the realistic mistake — a
    different split or dataset — without pretending to be a content hash.
    """
    return _sha256(json.dumps([train_size, test_size, extra], default=str))


def capture_rng_state() -> Dict[str, Any]:
    """Every generator that affects training, captured now.

    CUDA states are captured per visible device. On a CPU-only machine the
    list is empty, which is the truth rather than a placeholder.
    """
    import numpy as np
    import torch

    cuda_states: List[Any] = []
    if torch.cuda.is_available():
        cuda_states = [
            torch.cuda.get_rng_state(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(state: Optional[Dict[str, Any]]) -> List[str]:
    """Put the generators back. Returns warnings for anything not restored.

    A missing or unusable generator state is reported rather than ignored: a
    resume that silently reseeds is not the continuation it claims to be, and
    the caller records the warning on the run.
    """
    import numpy as np
    import torch

    warnings: List[str] = []
    if not isinstance(state, dict):
        return ["No RNG state was stored; the continuation is not deterministic."]

    try:
        random.setstate(state["python"])
    except (KeyError, TypeError, ValueError) as exc:
        warnings.append(f"Python RNG state not restored ({exc}).")
    try:
        np.random.set_state(state["numpy"])
    except (KeyError, TypeError, ValueError) as exc:
        warnings.append(f"NumPy RNG state not restored ({exc}).")
    try:
        torch.set_rng_state(_as_byte_tensor(state["torch_cpu"]))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        warnings.append(f"torch CPU RNG state not restored ({exc}).")

    stored = state.get("torch_cuda") or []
    if stored and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if len(stored) != available:
            warnings.append(
                f"The checkpoint carries {len(stored)} CUDA RNG state(s) but "
                f"{available} device(s) are visible; CUDA RNG was not restored."
            )
        else:
            for index, value in enumerate(stored):
                try:
                    torch.cuda.set_rng_state(_as_byte_tensor(value), index)
                except (TypeError, ValueError, RuntimeError) as exc:
                    warnings.append(f"CUDA RNG state for device {index} not restored ({exc}).")
    elif stored and not torch.cuda.is_available():
        warnings.append(
            "The checkpoint carries CUDA RNG state but no CUDA device is "
            "visible; continuing on CPU is not a bitwise continuation."
        )
    return warnings


def _as_byte_tensor(value: Any) -> Any:
    """torch RNG states must be a ByteTensor; round-trips can widen them."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.cpu().to(torch.uint8)
    return torch.tensor(value, dtype=torch.uint8)


def build_checkpoint(
    *,
    model_state: Any,
    optimizer_state: Any,
    epoch: int,
    training_identity: Dict[str, Any],
    scheduler_state: Any = None,
    scaler_state: Any = None,
    global_step: Optional[int] = None,
    best_state: Any = None,
    rng_state: Optional[Dict[str, Any]] = None,
    target_total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble a resumable checkpoint payload.

    ``epoch`` is the number of epochs **completed**, so a resume starts at
    ``epoch + 1``. Fields are named explicitly rather than swept up from an
    ambient dict, so nothing unintended — a credential above all — can arrive
    here.
    """
    return {
        SCHEMA_KEY: SCHEMA_VERSION,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "epoch": int(epoch),
        "global_step": None if global_step is None else int(global_step),
        "best_state": best_state,
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
        "target_total_epochs": (
            None if target_total_epochs is None else int(target_total_epochs)
        ),
        "training_identity": dict(training_identity),
    }


def save_checkpoint(payload: Dict[str, Any], path: str | Path) -> Path:
    """Write a checkpoint, atomically.

    A checkpoint written half-way through is worse than none: it looks
    resumable and is not. Written to a temporary file beside the target and
    moved into place, so a reader sees the whole file or no file.
    """
    import os
    import tempfile

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=str(target.parent), prefix=".ckpt-", suffix=".tmp", delete=False
    )
    try:
        with handle as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return target


def load_raw(path: str | Path) -> Any:
    """Read a checkpoint file, or return None.

    ``weights_only=False`` is required because a resumable payload holds
    optimizer and RNG structures, not just tensors. These files are written by
    this platform into its own run directories; they are not user uploads.
    """
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - a bad file is a verdict, not a crash
        return None


def classify_checkpoint(path: str | Path) -> Dict[str, Any]:
    """What kind of checkpoint this is, and what may be done with it.

    Returns ``{"kind", "message", "epoch", "schema_version", "identity",
    "path"}``. This is the single place that decides whether something may be
    resumed, so no caller has to re-derive it — and a weights-only file is
    never silently promoted.
    """
    target = Path(path)
    result: Dict[str, Any] = {
        "path": str(target),
        "kind": KIND_UNREADABLE,
        "message": UNREADABLE_MESSAGE,
        "epoch": None,
        "schema_version": None,
        "identity": {},
        "target_total_epochs": None,
    }
    payload = load_raw(target)
    if payload is None:
        return result
    if not isinstance(payload, dict) or SCHEMA_KEY not in payload:
        # A bare state_dict, or anything else. Usable for inference and for an
        # explicit warm start; not a resume.
        result.update(kind=KIND_WEIGHTS_ONLY, message=WEIGHTS_ONLY_MESSAGE)
        return result

    version = payload.get(SCHEMA_KEY)
    result["schema_version"] = version
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        result.update(
            kind=KIND_FUTURE_VERSION,
            message=(
                f"This checkpoint declares resume schema version {version}, "
                f"which this build ({SCHEMA_VERSION}) does not understand. "
                "It is not being read optimistically."
            ),
        )
        return result

    result.update(
        kind=KIND_RESUMABLE,
        message="",
        epoch=payload.get("epoch"),
        identity=dict(payload.get("training_identity") or {}),
        target_total_epochs=payload.get("target_total_epochs"),
    )
    return result


def is_resumable(path: str | Path) -> bool:
    return classify_checkpoint(path)["kind"] == KIND_RESUMABLE


# --- compatibility --------------------------------------------------------

#: Identity fields that must match exactly for a resume to be allowed.
#: Changing any of them makes the continuation a different experiment.
BLOCKING_IDENTITY_FIELDS: tuple[str, ...] = (
    "model_key",
    "dataset",
    "task",
    "config_fingerprint",
    "split_fingerprint",
    "seed",
)

#: What a resume may legitimately change. Training further is a decision the
#: user is allowed to make; training something else is not.
ALLOWED_OVERRIDES: tuple[str, ...] = ("target_total_epochs",)


def validate_resume(
    checkpoint: Dict[str, Any],
    requested_identity: Dict[str, Any],
    *,
    target_total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Whether this checkpoint may be resumed into this configuration.

    Returns ``{"ok", "errors", "warnings", "start_epoch", "remaining"}``. Each
    error names the field, the checkpoint's value and the requested one, so
    the message says what to change rather than only that something is wrong.
    """
    errors: List[str] = []
    warnings: List[str] = []

    kind = checkpoint.get("kind")
    if kind == KIND_WEIGHTS_ONLY:
        return {
            "ok": False,
            "errors": [WEIGHTS_ONLY_MESSAGE],
            "warnings": [],
            "start_epoch": None,
            "remaining": None,
        }
    if kind != KIND_RESUMABLE:
        return {
            "ok": False,
            "errors": [str(checkpoint.get("message") or "This checkpoint cannot be resumed.")],
            "warnings": [],
            "start_epoch": None,
            "remaining": None,
        }

    stored = checkpoint.get("identity") or {}
    for field in BLOCKING_IDENTITY_FIELDS:
        if field not in requested_identity:
            continue
        if field not in stored:
            # The checkpoint does not record this field, so it cannot be
            # compared. Reported rather than silently passed *or* silently
            # failed: a checkpoint written by a producer that stores less
            # than this one checks is usable, but the reader should know
            # which guarantee it did not get.
            warnings.append(
                f"The checkpoint records no {field}, so it could not be "
                f"checked against {requested_identity.get(field)!r}."
            )
            continue
        have, want = stored.get(field), requested_identity.get(field)
        if str(have) != str(want):
            errors.append(
                f"{field}: the checkpoint was produced with {have!r} but this "
                f"run requests {want!r}. Resuming across a different {field} "
                "would not continue the same experiment."
            )

    completed = checkpoint.get("epoch")
    if not isinstance(completed, int) or completed < 0:
        errors.append("The checkpoint records no usable completed-epoch count.")
        completed = None

    target = target_total_epochs
    if target is None:
        target = checkpoint.get("target_total_epochs")
    remaining = None
    if completed is not None and isinstance(target, int):
        remaining = target - completed
        if remaining <= 0:
            errors.append(
                f"The checkpoint has already completed {completed} epoch(s) and "
                f"the target total is {target}. Raise the target above "
                f"{completed} to continue training."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "start_epoch": None if completed is None else completed + 1,
        "remaining": remaining,
        "completed_epochs": completed,
        "target_total_epochs": target,
    }
