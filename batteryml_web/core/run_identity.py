"""Identity for one explicit execution.

A project holds many runs, and "the MATR + BatLiNet workspace" is not an
identity: run it twice and the second silently overwrites the first, so a
result can no longer be attributed to the execution that produced it. This
module mints a real identifier instead.

Shape (readable first, unique second)::

    run_20260916_185742_batlinet_matr_seed0_a1b2c3
    <prefix><timestamp>_<model>_<dataset>_seed<seed>_<suffix>

The timestamp orders runs at a glance, the model/dataset/seed segment says
what it was without opening anything, and the random suffix makes two runs
of the *identical* configuration in the same second still distinct — which
is exactly the collision a timestamp alone does not cover.

No Streamlit, no filesystem, no clock beyond the one passed in: minting is a
pure function of (now, plan fields, random source) so tests can pin all
three.
"""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

RUN_ID_PREFIX = "run_"

#: Characters a run id may contain. Everything else is folded to "_", so an
#: id is always safe as a single path segment on any filesystem.
_SAFE = re.compile(r"[^A-Za-z0-9]+")

#: Length of the random tail. Six hex characters is 24 bits: ample against
#: same-second collisions, short enough to read aloud.
SUFFIX_CHARS = 6

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

#: A run id that matches this is one of ours. Used by history discovery to
#: tell a run directory from any other directory under a workspace.
RUN_ID_PATTERN = re.compile(
    r"^run_\d{8}_\d{6}_[A-Za-z0-9_]*_seed[A-Za-z0-9]+_[0-9a-f]{%d}$" % SUFFIX_CHARS
)


def safe_token(value: Any, *, fallback: str = "unknown") -> str:
    """One path-safe lowercase token, never empty.

    ``None`` is the only missing value. Seed 0 is the platform's documented
    default (CLAUDE.md §11), and ``str(value or "")`` would turn it into the
    fallback — every default-seed run would have claimed an unknown seed.
    """
    if value is None:
        return fallback
    token = _SAFE.sub("_", str(value)).strip("_").lower()
    return token or fallback


def new_run_id(
    *,
    dataset: Any,
    model: Any,
    seed: Any,
    now: Optional[datetime] = None,
    entropy: Optional[Callable[[int], str]] = None,
) -> str:
    """Mint a run id.

    Args:
        dataset: dataset name, for the readable segment.
        model: canonical model key, for the readable segment.
        seed: the run's seed, recorded because two runs differing only by
            seed are different experiments.
        now: injectable clock. Defaults to UTC now, so ids from different
            machines sort together.
        entropy: injectable random-hex source, for tests.

    Called at Run, never at Analyze: an id that existed before the user
    pressed Run would name an execution that never happened.
    """
    moment = now or datetime.now(timezone.utc)
    stamp = moment.strftime(TIMESTAMP_FORMAT)
    tail = (entropy or secrets.token_hex)(SUFFIX_CHARS // 2)
    return (
        f"{RUN_ID_PREFIX}{stamp}_{safe_token(model, fallback='model')}"
        f"_{safe_token(dataset, fallback='dataset')}"
        f"_seed{safe_token(seed, fallback='na')}_{tail}"
    )


def is_run_id(value: Any) -> bool:
    """Whether ``value`` looks like an id this module minted."""
    return bool(RUN_ID_PATTERN.match(str(value or "")))


def describe_run_id(run_id: str) -> Dict[str, Optional[str]]:
    """Best-effort read-back of the readable segments.

    Convenience for display only. The manifest is the authority on what a
    run actually was; nothing decides behaviour from a parsed id.
    """
    empty: Dict[str, Optional[str]] = {"timestamp": None, "model": None,
                                       "dataset": None, "seed": None}
    if not is_run_id(run_id):
        return empty
    body = str(run_id)[len(RUN_ID_PREFIX):]
    parts = body.split("_")
    # date, time, <model...>, <dataset>, seed<N>, suffix
    try:
        seed_index = max(
            i for i, part in enumerate(parts) if part.startswith("seed")
        )
    except ValueError:
        return empty
    return {
        "timestamp": f"{parts[0]}_{parts[1]}",
        "model": "_".join(parts[2:seed_index - 1]) or None,
        "dataset": parts[seed_index - 1] if seed_index >= 3 else None,
        "seed": parts[seed_index][len("seed"):] or None,
    }
