# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""Synthetic battery cells for the SOH tests.

Per ``CLAUDE.md`` §7 these are test fixtures and nothing else: the
``synthetic_`` prefix is on the module and on every cell id, they are built in
memory rather than written under ``data/``, and no dataset registry can find
them. A fixture may never stand in for a missing dataset or reach a metric a
reader could mistake for a result.

Capacities here are chosen so the expected SOH ratios are exact in binary
floating point (halves and quarters), which lets the label tests assert
equality rather than a tolerance.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from batteryml.data.battery_data import BatteryData, CycleData


def make_cycle(cycle_number: int, discharge_capacity: float | None,
               n_points: int = 4) -> CycleData:
    """One cycle whose discharge-capacity trace peaks at ``discharge_capacity``.

    BatteryML stores the within-cycle trace, and a cycle's capacity is its
    maximum, so the trace ramps up to the requested value. ``None`` produces a
    cycle with no usable capacity, for the exclusion tests.
    """
    if discharge_capacity is None:
        trace = None
    else:
        trace = [
            discharge_capacity * (i + 1) / n_points for i in range(n_points)
        ]
    voltage = [3.0 + 0.1 * i for i in range(n_points)] if trace else None
    current = [-1.0] * n_points if trace else None
    time = [float(60 * i) for i in range(n_points)] if trace else None
    return CycleData(
        cycle_number=cycle_number,
        voltage_in_V=voltage,
        current_in_A=current,
        discharge_capacity_in_Ah=trace,
        charge_capacity_in_Ah=trace,
        time_in_s=time,
    )


def make_cell(cell_id: str,
              capacities: Sequence[float | None],
              nominal_capacity_in_Ah: float | None = None,
              first_cycle_number: int = 1) -> BatteryData:
    """A cell whose cycle ``i`` has discharge capacity ``capacities[i]``."""
    cycles = [
        make_cycle(first_cycle_number + i, capacity)
        for i, capacity in enumerate(capacities)
    ]
    return BatteryData(
        cell_id=cell_id,
        cycle_data=cycles,
        form_factor="cylindrical_18650",
        nominal_capacity_in_Ah=nominal_capacity_in_Ah,
    )


#: A cell that degrades in exact halves: 1.0, 0.75, 0.5, 0.25.
#: With Q_ref = first valid capacity = 1.0 the SOH labels are the same values.
SYNTHETIC_CLEAN_CAPACITIES: List[float] = [1.0, 0.75, 0.5, 0.25]


def clean_cell(cell_id: str = "synthetic_clean") -> BatteryData:
    return make_cell(cell_id, SYNTHETIC_CLEAN_CAPACITIES)


def nominal_cell(cell_id: str = "synthetic_nominal",
                 nominal: float = 2.0) -> BatteryData:
    """Carries a nominal capacity twice its first measured capacity.

    This is the cell that exposes convention mixing: under
    ``first_valid_discharge_capacity`` its first label is 1.0, under
    ``nominal_capacity`` it is 0.5. A run that switched conventions per cell
    would produce both and record neither.
    """
    return make_cell(cell_id, SYNTHETIC_CLEAN_CAPACITIES, nominal_capacity_in_Ah=nominal)


def cell_with_invalid_cycles(cell_id: str = "synthetic_invalid") -> BatteryData:
    """Valid, missing, zero, negative, then valid again."""
    return make_cell(cell_id, [1.0, None, 0.0, -0.5, 0.5])


def cell_without_reference(cell_id: str = "synthetic_no_reference") -> BatteryData:
    """No cycle has a usable capacity, so no reference can be established."""
    return make_cell(cell_id, [None, None])


def cohort(n_cells: int = 6, n_cycles: int = 10,
           prefix: str = "synthetic_cell") -> List[BatteryData]:
    """A small cohort for split and sample-count tests.

    Cell ``k`` fades linearly from 1.0 at a rate that differs per cell, so a
    model cannot do well on a held-out cell by memorising one curve.

    The rate is expressed as a fraction of the cell's life rather than a
    per-cycle constant, so **every cycle keeps a positive capacity** whatever
    ``n_cycles`` is. A constant rate silently drove late cycles negative,
    which the label annotator correctly excluded — and a count test then
    failed for a reason that had nothing to do with the code under test.
    """
    cells = []
    for k in range(n_cells):
        total_fade = 0.1 + 0.05 * k        # 10%..(10+5k)% lost over the cell's life
        per_cycle = total_fade / max(1, n_cycles - 1)
        capacities = [1.0 - per_cycle * i for i in range(n_cycles)]
        assert min(capacities) > 0.0, "cohort fixture must not produce a dead cell"
        cells.append(make_cell(f"{prefix}_{k}", capacities))
    return cells


def cell_ids(cells: Iterable[BatteryData]) -> List[str]:
    return [cell.cell_id for cell in cells]
