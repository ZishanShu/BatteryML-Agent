# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""State-of-health labels.

Two annotators live here.

:class:`SOHLabelAnnotator` is the original one-label-per-cell annotator, kept
for the existing ``configs/soh/dummy/*`` examples. Its one behavioural change
is that the reference capacity is now **explicit**: see
:data:`REFERENCE_CAPACITY_MODES`.

:class:`CycleSOHLabelAnnotator` is the canonical cycle-level annotator for the
SOH estimation task: it labels every observed cycle the sample policy accepts,
so one cell yields many samples.

**The canonical task is SOH at an observed cycle**, estimated from information
available at or before that cycle. It is not next-cycle capacity forecasting,
not remaining useful life, not lifetime, and not degradation-trajectory
forecasting. The placeholder docstring in
``batteryml/models/soh_predictors/__init__.py`` that called SOH "the discharge
capacity of the next cycle" describes none of this and is marked historical.

**The canonical label is a ratio**, not a percentage::

    SOH(cell, t) = Q_discharge(cell, t) / Q_ref(cell)

A UI may render ``0.973`` as ``97.3%``; the stored value stays ``0.973``.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import List, Sequence, Tuple

import torch

from batteryml.builders import LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData

from .base import BaseCycleLabelAnnotator, BaseLabelAnnotator

#: Reference capacity Q_ref, chosen once per run and applied to every cell.
#:
#: The previous implementation chose per cell — nominal capacity when the cell
#: happened to carry one, otherwise that cell's first-cycle discharge capacity
#: — so two cells in one dataset could be labelled under two different
#: definitions with nothing recording which. That behaviour is gone; there is
#: no automatic fallback and no per-cell switching.
REFERENCE_CAPACITY_MODES = (
    # This run's default: Q_ref is the cell's first cycle whose discharge
    # capacity is valid. Every dataset can satisfy it, which is why it is the
    # default; it measures health relative to the cell's own start.
    "first_valid_discharge_capacity",
    # Q_ref is the manufacturer's nominal capacity. Absolute rather than
    # self-relative, but a cell that does not carry one is excluded, never
    # silently switched to another convention.
    "nominal_capacity",
    # Q_ref is the mean over the first N valid cycles, N = reference_n_cycles.
    # Less sensitive to a single noisy first cycle.
    "first_n_valid_mean",
)

DEFAULT_REFERENCE_CAPACITY_MODE = "first_valid_discharge_capacity"

#: Why a cell or a sample was excluded. Counted and reported, never silent.
EXCLUSION_NO_CYCLES = "no_cycle_data"
EXCLUSION_NO_REFERENCE = "no_valid_reference_capacity"
EXCLUSION_INVALID_CAPACITY = "invalid_discharge_capacity"
EXCLUSION_NO_SAMPLED_CYCLE = "no_cycle_in_sample_policy"


class SOHReferenceError(ValueError):
    """A reference capacity could not be established as configured."""


def _validate_mode(mode: str) -> str:
    if mode not in REFERENCE_CAPACITY_MODES:
        raise SOHReferenceError(
            f"reference_capacity_mode={mode!r} is not supported. Choose one "
            f"of {', '.join(REFERENCE_CAPACITY_MODES)}. The mode is fixed for "
            "a whole run and recorded in the run's provenance; it is never "
            "chosen per cell."
        )
    return mode


def cycle_discharge_capacity(cycle) -> float:
    """The discharge capacity of one cycle in Ah, or NaN if unusable.

    BatteryML stores ``discharge_capacity_in_Ah`` as the per-sample trace
    within a cycle, so the cycle's capacity is its maximum — the same
    convention the original annotator used. A trace that is missing, empty, or
    whose maximum is non-finite or non-positive yields NaN: a cell cannot have
    discharged a negative amount, and a zero would make the ratio undefined.
    Nothing is interpolated and nothing is substituted.
    """
    trace = getattr(cycle, "discharge_capacity_in_Ah", None)
    if trace is None:
        return float("nan")
    try:
        values = [float(v) for v in trace]
    except TypeError:
        try:
            values = [float(trace)]
        except (TypeError, ValueError):
            return float("nan")
    values = [v for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if not values:
        return float("nan")
    capacity = max(values)
    if capacity <= 0.0:
        return float("nan")
    return capacity


def reference_capacity(
    cell_data: BatteryData,
    mode: str = DEFAULT_REFERENCE_CAPACITY_MODE,
    reference_n_cycles: int = 5,
) -> Tuple[float, str | None]:
    """``(Q_ref, exclusion_reason)`` for one cell under one configured mode.

    Returns NaN with a reason rather than raising, so a whole dataset can be
    labelled and the excluded cells counted and reported. There is no
    fallback between modes: a cell that cannot satisfy the configured mode is
    excluded under it.
    """
    _validate_mode(mode)
    cycles = getattr(cell_data, "cycle_data", None) or []
    if not cycles:
        return float("nan"), EXCLUSION_NO_CYCLES

    if mode == "nominal_capacity":
        nominal = getattr(cell_data, "nominal_capacity_in_Ah", None)
        if nominal is None or not float(nominal) > 0.0:
            return float("nan"), EXCLUSION_NO_REFERENCE
        return float(nominal), None

    valid: List[float] = []
    for cycle in cycles:
        capacity = cycle_discharge_capacity(cycle)
        if capacity == capacity:  # not NaN
            valid.append(capacity)
            if mode == "first_valid_discharge_capacity":
                break
            if len(valid) >= max(1, int(reference_n_cycles)):
                break
    if not valid:
        return float("nan"), EXCLUSION_NO_REFERENCE
    if mode == "first_valid_discharge_capacity":
        return valid[0], None
    return sum(valid) / len(valid), None


def sampled_cycle_numbers(
    cell_data: BatteryData,
    min_cycle: int = 1,
    max_cycle: int | None = None,
    cycle_stride: int = 1,
) -> List[int]:
    """The cycle numbers this cell offers under the sample policy.

    Policy values are configuration, never constants baked into the science:
    ``min_cycle`` skips a formation period, ``max_cycle`` caps the observation
    window (``None`` means "to the end of the cell"), ``cycle_stride``
    subsamples. Cycle numbers are BatteryML's own
    ``CycleData.cycle_number``.
    """
    if int(cycle_stride) < 1:
        raise ValueError(
            f"cycle_stride={cycle_stride} must be >= 1; a stride of zero or "
            "less would sample the same cycle forever."
        )
    cycles = getattr(cell_data, "cycle_data", None) or []
    numbers: List[int] = []
    for position, cycle in enumerate(cycles):
        number = getattr(cycle, "cycle_number", None)
        number = int(number) if number is not None else position + 1
        if number < int(min_cycle):
            continue
        if max_cycle is not None and number > int(max_cycle):
            continue
        numbers.append(number)
    return numbers[:: int(cycle_stride)]


@LABEL_ANNOTATORS.register()
class SOHLabelAnnotator(BaseLabelAnnotator):
    """One SOH label per cell, at a single configured cycle.

    Retained for ``configs/soh/dummy/*``, which are examples of the label
    plumbing rather than the SOH estimation task. New SOH work uses
    :class:`CycleSOHLabelAnnotator`.

    Two changes from the original:

    * ``reference_capacity_mode`` is explicit and applies to every cell. The
      old "nominal if present, else first cycle" fallback mixed two label
      definitions inside one dataset and is gone.
    * ``mode='absolute'`` is gone with it. It returned capacity in Ah, not a
      ratio, under a parameter whose other value returned a ratio. Canonical
      SOH is a ratio; a raw capacity target is a different label.
    """

    def __init__(self,
                 cycle_index: int = 100,
                 soh_filepath: str = None,
                 reference_capacity_mode: str = DEFAULT_REFERENCE_CAPACITY_MODE,
                 reference_n_cycles: int = 5):
        self.cycle_index = cycle_index
        self.reference_capacity_mode = _validate_mode(reference_capacity_mode)
        self.reference_n_cycles = reference_n_cycles
        self.soh_dict = None

        if soh_filepath and os.path.exists(soh_filepath):
            root, extension = os.path.splitext(soh_filepath)
            if extension == '.json':
                self.cycle_index = str(self.cycle_index)
                with open(soh_filepath, 'rb') as f:
                    self.soh_dict = json.load(f)
            elif extension == '.pkl':
                with open(soh_filepath, 'rb') as f:
                    self.soh_dict = pickle.load(f)
            else:
                raise SOHReferenceError(
                    f"soh_filepath={soh_filepath!r} must be .json or .pkl; "
                    f"got {extension!r}."
                )
        elif soh_filepath:
            raise SOHReferenceError(
                f"soh_filepath={soh_filepath!r} does not exist. Provide the "
                "file or remove the field; a missing file must not silently "
                "fall back to computed labels."
            )

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        if self.soh_dict is not None:
            cell_id = cell_data.cell_id
            entry = self.soh_dict.get(cell_id, {}) if cell_id else {}
            value = entry.get(self.cycle_index, {}) if isinstance(entry, dict) else {}
            label = value.get('relative', float('nan')) if isinstance(value, dict) else float('nan')
            return torch.tensor(float(label))

        cycles = getattr(cell_data, 'cycle_data', None) or []
        index = int(self.cycle_index)
        if len(cycles) < index or index < 1:
            return torch.tensor(float('nan'))
        capacity = cycle_discharge_capacity(cycles[index - 1])
        q_ref, _ = reference_capacity(
            cell_data, self.reference_capacity_mode, self.reference_n_cycles)
        if capacity != capacity or q_ref != q_ref:
            return torch.tensor(float('nan'))
        return torch.tensor(capacity / q_ref)


@LABEL_ANNOTATORS.register()
class CycleSOHLabelAnnotator(BaseCycleLabelAnnotator):
    """The canonical SOH label: one ratio per observed cycle.

    For every cycle the sample policy accepts::

        SOH(cell, t) = Q_discharge(cell, t) / Q_ref(cell)

    with ``Q_ref`` fixed for the whole run by ``reference_capacity_mode``.

    A cell with no valid reference capacity contributes nothing, and a cycle
    with no valid discharge capacity is dropped rather than interpolated. Both
    are counted in :attr:`exclusions` so a run can report what it left out
    instead of quietly shrinking. Labels are never clipped: a ratio above 1.0
    early in life is a real measurement, and clipping it to make training
    easier would be fabricating data.
    """

    def __init__(self,
                 reference_capacity_mode: str = DEFAULT_REFERENCE_CAPACITY_MODE,
                 reference_n_cycles: int = 5,
                 min_cycle: int = 1,
                 max_cycle: int | None = None,
                 cycle_stride: int = 1):
        self.reference_capacity_mode = _validate_mode(reference_capacity_mode)
        self.reference_n_cycles = reference_n_cycles
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.cycle_stride = cycle_stride
        self.exclusions: dict[str, int] = {}

    def _exclude(self, reason: str, count: int = 1) -> None:
        self.exclusions[reason] = self.exclusions.get(reason, 0) + count

    def sample_policy(self) -> dict:
        """The policy as configured, for provenance and result metadata."""
        return {
            "reference_capacity_mode": self.reference_capacity_mode,
            "reference_n_cycles": self.reference_n_cycles,
            "min_cycle": self.min_cycle,
            "max_cycle": self.max_cycle,
            "cycle_stride": self.cycle_stride,
        }

    def process_cell_cycles(
        self, cell_data: BatteryData
    ) -> Tuple[torch.Tensor, Sequence[int]]:
        q_ref, reason = reference_capacity(
            cell_data, self.reference_capacity_mode, self.reference_n_cycles)
        if reason is not None:
            self._exclude(reason)
            return torch.zeros(0), []

        wanted = sampled_cycle_numbers(
            cell_data, self.min_cycle, self.max_cycle, self.cycle_stride)
        if not wanted:
            self._exclude(EXCLUSION_NO_SAMPLED_CYCLE)
            return torch.zeros(0), []

        by_number = {}
        for position, cycle in enumerate(getattr(cell_data, "cycle_data", None) or []):
            number = getattr(cycle, "cycle_number", None)
            by_number[int(number) if number is not None else position + 1] = cycle

        labels: List[float] = []
        cycles: List[int] = []
        invalid = 0
        for number in wanted:
            cycle = by_number.get(number)
            capacity = cycle_discharge_capacity(cycle) if cycle is not None else float("nan")
            if capacity != capacity:
                invalid += 1
                continue
            labels.append(capacity / q_ref)
            cycles.append(number)
        if invalid:
            self._exclude(EXCLUSION_INVALID_CAPACITY, invalid)
        if not labels:
            self._exclude(EXCLUSION_NO_SAMPLED_CYCLE)
        return torch.tensor(labels, dtype=torch.float32), cycles
