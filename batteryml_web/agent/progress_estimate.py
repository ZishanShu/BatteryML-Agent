"""Time remaining, estimated only where there is evidence for it.

A progress bar that says "8s left" when nothing has been measured is worse
than no estimate: it is a number the user will plan around, invented by the
UI. This module will produce an ETA only from observations the run itself
produced, and returns ``None`` — meaning "say 'estimating…' or say nothing"
— in every other case.

The rules, in order:

* an ETA needs a **counter** (``current`` of ``total``) that the run printed,
  and at least :data:`MIN_OBSERVATIONS` completed units to time. One epoch
  tells you almost nothing about the next hundred;
* the estimate is the mean of the most recent
  :data:`RECENT_WINDOW` per-unit durations, so a run that speeds up or slows
  down is tracked rather than averaged flat over its whole history;
* a stage with no denominator — "Loading data" has no idea how much data is
  left — never gets an ETA from timing, and never gets a countdown at all
  unless a *previous comparable run* measured that stage;
* prior-run evidence is used only when it matches on the axes that change the
  duration (dataset, config, device class, stage) and is always labelled
  estimated.

Nothing here touches Streamlit, the filesystem or a clock of its own: the
caller supplies the elapsed seconds it observed, which is what makes the
whole estimator testable without waiting for real time to pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: Completed units required before any ETA is offered.
MIN_OBSERVATIONS = 2

#: How many recent per-unit durations the estimate averages over.
RECENT_WINDOW = 10

#: What the UI shows while a counter exists but the evidence does not yet.
ESTIMATING_LABEL = "estimating…"


class EpochTimer:
    """Per-unit durations, derived from (counter, elapsed) observations.

    Feed it what the run reported — ``observe(current, total, elapsed_s)`` —
    and it records how long each newly completed unit took. It never invents
    an observation: a repeated or backwards counter is ignored, and a run
    that reports no counter produces no observations at all.
    """

    def __init__(self) -> None:
        self._last_current: Optional[int] = None
        self._last_elapsed: float = 0.0
        self._durations: List[float] = []
        self._stage: Optional[str] = None
        self.total: Optional[int] = None
        self.current: Optional[int] = None

    def reset(self) -> None:
        """Forget every observation, keeping no counter and no rate."""
        self._last_current = None
        self._last_elapsed = 0.0
        self._durations = []
        self.total = None
        self.current = None

    def observe(self, current: Any, total: Any, elapsed_s: float,
                stage: Any = None) -> None:
        """Record one progress report from the run.

        ``stage`` is optional and, when given, scopes the timing to it:
        arriving in a new stage discards the previous stage's per-unit
        durations. Reading 41 cells and extracting samples from 41 cells are
        not the same unit of work, so an ETA for the second computed from the
        rate of the first would be a number with nothing behind it. Omitting
        ``stage`` keeps the original whole-run behaviour.
        """
        if stage is not None:
            stage_text = str(stage)
            if self._stage is not None and stage_text != self._stage:
                self.reset()
            self._stage = stage_text
        if not isinstance(current, int) or isinstance(current, bool):
            return
        if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
            return
        elapsed = float(elapsed_s)
        self.total = total
        self.current = current
        if self._last_current is None:
            self._last_current = current
            self._last_elapsed = elapsed
            return
        advanced = current - self._last_current
        if advanced <= 0:
            # A repaint of the same counter, or a counter that went backwards
            # (a new stage reusing the bar). Neither times anything.
            self._last_current = current
            self._last_elapsed = elapsed
            return
        span = elapsed - self._last_elapsed
        if span > 0:
            per_unit = span / advanced
            self._durations.extend([per_unit] * advanced)
            del self._durations[:-RECENT_WINDOW]
        self._last_current = current
        self._last_elapsed = elapsed

    @property
    def observations(self) -> int:
        return len(self._durations)

    def per_unit_seconds(self) -> Optional[float]:
        """Mean recent per-unit duration, or None without enough evidence."""
        if len(self._durations) < MIN_OBSERVATIONS:
            return None
        return sum(self._durations) / len(self._durations)

    def eta_seconds(self) -> Optional[float]:
        """Seconds remaining, or None when the run has not shown enough."""
        per_unit = self.per_unit_seconds()
        if per_unit is None or self.total is None or self.current is None:
            return None
        remaining = self.total - self.current
        if remaining <= 0:
            return 0.0
        return per_unit * remaining


def format_duration(seconds: Optional[float]) -> str:
    """``MM:SS`` (or ``H:MM:SS`` past an hour); '' for None."""
    if seconds is None:
        return ""
    total = max(0, int(round(float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_line(
    stage: str,
    *,
    current: Optional[int] = None,
    total: Optional[int] = None,
    elapsed_s: float = 0.0,
    eta_s: Optional[float] = None,
    estimating: bool = False,
) -> str:
    """The one-line status: stage, real counter, elapsed, and ETA if earned.

    ``estimating`` is the honest middle state — a counter exists but not
    enough of it has completed to time. It renders "estimating…" rather than
    a number.
    """
    parts = [str(stage or "Running")]
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        parts.append(f"{current} / {total}")
    parts.append(f"elapsed {format_duration(elapsed_s)}")
    if eta_s is not None:
        parts.append(f"ETA ~{format_duration(eta_s)}")
    elif estimating:
        parts.append(ESTIMATING_LABEL)
    return " · ".join(parts)


def stage_evidence_key(
    *,
    dataset: Any,
    config_path: Any,
    device: Any,
    stage: Any,
) -> str:
    """The axes a previous run must match for its timing to mean anything.

    Device is reduced to a *class* (``cpu`` / ``cuda``) rather than an index:
    two runs on cuda:0 and cuda:3 of the same machine are comparable, and the
    index would fragment the evidence for no reason.
    """
    device_text = str(device or "cpu").strip().lower()
    device_class = "cuda" if device_text.startswith("cuda") else device_text or "cpu"
    return "|".join(
        [
            str(dataset or ""),
            str(config_path or ""),
            device_class,
            str(stage or ""),
        ]
    )


def historical_eta_seconds(
    history: Optional[Sequence[Dict[str, Any]]],
    key: str,
    elapsed_s: float,
) -> Optional[float]:
    """Remaining seconds for a stage with no counter, from matching past runs.

    ``history`` entries are ``{"key": str, "duration_s": float}`` recorded by
    previous runs. Only entries whose key matches exactly are used — same
    dataset, same config, same device class, same stage.

    Returns None when there is no matching evidence, and also when the
    current run has already outlived the historical median: at that point the
    past run no longer predicts this one, and a negative or zero countdown
    would be a worse answer than none.
    """
    if not history or not key:
        return None
    durations = sorted(
        float(entry["duration_s"])
        for entry in history
        if str(entry.get("key")) == key
        and isinstance(entry.get("duration_s"), (int, float))
        and not isinstance(entry.get("duration_s"), bool)
        and float(entry["duration_s"]) > 0
    )
    if not durations:
        return None
    middle = len(durations) // 2
    median = (
        durations[middle]
        if len(durations) % 2
        else (durations[middle - 1] + durations[middle]) / 2
    )
    remaining = median - float(elapsed_s)
    return remaining if remaining > 0 else None
