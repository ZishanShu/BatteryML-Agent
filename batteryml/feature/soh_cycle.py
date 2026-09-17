# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""The canonical SOH feature: the voltage curve observed at cycle t.

One feature path, not five. It is deliberately the simplest thing that is
scientifically honest about what SOH estimation is for.

**Why the discharge capacity is not a feature.** The label is
``SOH(t) = Q_discharge(t) / Q_ref``. Feeding ``Q_discharge(t)`` in as a
feature would make the target an arithmetic function of the input, and the
model would "learn" a division. Worse, it would answer a question nobody
asks: if you have already run a full discharge to measure the capacity, you
know the SOH and need no estimator. The point of SOH estimation is to read
health off the *shape* of a cycle's measured curve. So this extractor reads
voltage, current and time, and never capacity.

**Causality.** The row for cycle ``t`` reads cycle ``t`` and, when a history
window is configured, the cycles before it. It never reads ``t+1``, the
cell's last cycle, or its lifetime. ``tests/test_soh_features.py`` proves
this by mutating every later cycle and checking the row is unchanged.

**Shape.** Fixed and documented, per requested cycle::

    n_history_cycles * (n_voltage_points + len(SUMMARY_CHANNELS))
      + (1 if include_cycle_number else 0)

The default — one cycle of history, 32 resampled voltage points, 5 summary
channels — is a 37-value row.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from batteryml.builders import FEATURE_EXTRACTORS
from batteryml.data import BatteryData

from .base import BaseCycleFeatureExtractor

#: Per-cycle scalars appended after the resampled voltage curve, in order.
#: All are read from the cycle's own trace, none involve capacity.
SUMMARY_CHANNELS = (
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "current_abs_mean",
    "duration_in_s",
)


def _finite(values) -> List[float]:
    """The finite floats in a trace, in order. Never substitutes a value."""
    if values is None:
        return []
    try:
        iterator = list(values)
    except TypeError:
        iterator = [values]
    out: List[float] = []
    for value in iterator:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            out.append(number)
    return out


def resample(trace: Sequence[float], n_points: int) -> List[float]:
    """``trace`` linearly resampled onto ``n_points`` evenly spaced samples.

    Index-based rather than time-based, so a cycle with no time channel is
    still usable and a cycle whose sampling rate varies is not silently
    re-timed. A single-point trace is repeated, which is the only reading
    that point supports; an empty trace has no resampling and is rejected by
    the caller instead.
    """
    if n_points < 1:
        raise ValueError(f"n_voltage_points={n_points} must be >= 1.")
    count = len(trace)
    if count == 0:
        raise ValueError("cannot resample an empty trace")
    if count == 1:
        return [float(trace[0])] * n_points
    if n_points == 1:
        return [float(trace[0])]
    out: List[float] = []
    for i in range(n_points):
        position = i * (count - 1) / (n_points - 1)
        low = int(position)
        high = min(low + 1, count - 1)
        weight = position - low
        out.append(float(trace[low]) * (1.0 - weight) + float(trace[high]) * weight)
    return out


@FEATURE_EXTRACTORS.register()
class SOHVoltageCurveFeatureExtractor(BaseCycleFeatureExtractor):
    """Voltage-curve features for SOH estimation at an observed cycle.

    Args:
        n_voltage_points: how many points the cycle's voltage trace is
            resampled to. Fixed shape, so any tabular regressor can consume it.
        n_history_cycles: how many cycles the row covers, ending at ``t``.
            ``1`` uses cycle ``t`` alone. Earlier cycles are information
            available at ``t``; later ones are not and are never read.
        include_cycle_number: append the observed cycle number. It is
            genuinely available at ``t``, but it lets a model lean on "how old
            is this cell" rather than on the measurement, which generalises
            poorly to a cell that ages at a different rate. Off by default for
            that reason; turn it on deliberately.

    A cycle with no usable voltage trace yields a row of NaN. ``Task.build``
    drops those rows and reports the count — the sample is not silently
    replaced by a zero or a neighbouring cycle's curve.
    """

    def __init__(self,
                 n_voltage_points: int = 32,
                 n_history_cycles: int = 1,
                 include_cycle_number: bool = False):
        if int(n_voltage_points) < 1:
            raise ValueError(f"n_voltage_points={n_voltage_points} must be >= 1.")
        if int(n_history_cycles) < 1:
            raise ValueError(f"n_history_cycles={n_history_cycles} must be >= 1.")
        self.n_voltage_points = int(n_voltage_points)
        self.n_history_cycles = int(n_history_cycles)
        self.include_cycle_number = bool(include_cycle_number)

    @property
    def feature_dim(self) -> int:
        """The row width, documented rather than discovered at runtime."""
        per_cycle = self.n_voltage_points + len(SUMMARY_CHANNELS)
        return per_cycle * self.n_history_cycles + int(self.include_cycle_number)

    def describe(self) -> dict:
        """The feature definition, for provenance and result metadata."""
        return {
            "name": type(self).__name__,
            "n_voltage_points": self.n_voltage_points,
            "n_history_cycles": self.n_history_cycles,
            "include_cycle_number": self.include_cycle_number,
            "summary_channels": list(SUMMARY_CHANNELS),
            "feature_dim": self.feature_dim,
            "reads": "voltage, current and time at cycles <= t; never capacity, "
                     "never a cycle after t",
        }

    def _cycle_block(self, cycle) -> List[float]:
        """One cycle's contribution, or NaNs when it cannot be read."""
        nan_block = [float("nan")] * (self.n_voltage_points + len(SUMMARY_CHANNELS))
        if cycle is None:
            return nan_block
        voltage = _finite(getattr(cycle, "voltage_in_V", None))
        if not voltage:
            return nan_block
        current = _finite(getattr(cycle, "current_in_A", None))
        time = _finite(getattr(cycle, "time_in_s", None))

        block = resample(voltage, self.n_voltage_points)
        block.append(sum(voltage) / len(voltage))
        block.append(min(voltage))
        block.append(max(voltage))
        block.append(
            sum(abs(value) for value in current) / len(current) if current else float("nan")
        )
        block.append(time[-1] - time[0] if len(time) >= 2 else float("nan"))
        return block

    def process_cell_cycles(
        self, cell_data: BatteryData, cycle_indices: Sequence[int]
    ) -> torch.Tensor:
        cycles = getattr(cell_data, "cycle_data", None) or []
        by_number = {}
        for position, cycle in enumerate(cycles):
            number = getattr(cycle, "cycle_number", None)
            by_number[int(number) if number is not None else position + 1] = cycle

        rows: List[List[float]] = []
        for target in cycle_indices:
            target = int(target)
            row: List[float] = []
            # Oldest first, so the row reads left to right in time. Only
            # cycles at or before the target are ever looked up.
            for offset in range(self.n_history_cycles - 1, -1, -1):
                row.extend(self._cycle_block(by_number.get(target - offset)))
            if self.include_cycle_number:
                row.append(float(target))
            rows.append(row)

        if not rows:
            return torch.zeros(0, self.feature_dim, dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)
