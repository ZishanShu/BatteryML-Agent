"""Is a processed dataset actually usable for SOH estimation?

A dataset being listed in a README, or having an SOH row in a results table,
says nothing about whether *these* processed files can produce SOH labels
under *this* run's conventions. That is an empirical question about the data
on disk, and it is the only thing this module answers.

Being ready for RUL and being ready for SOH are different questions. RUL needs
a cell to run to end of life; SOH needs a valid reference capacity and enough
cycles carrying both a usable discharge capacity (for the label) and a usable
voltage trace (for the features). A dataset can easily satisfy one and not the
other, and the two verdicts are reported separately rather than merged.

Nothing here fabricates a count. A dataset with no processed files reports
zero and says so; it never reports a plausible-looking number.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List

# Verdicts. Deliberately not the generic READY, so an SOH claim can never be
# read off a RUL readiness check.
SOH_READY = "SOH_READY"
SOH_INSUFFICIENT = "SOH_INSUFFICIENT"
SOH_NOT_ELIGIBLE = "SOH_NOT_ELIGIBLE"
SOH_NOT_PROCESSED = "SOH_NOT_PROCESSED"

#: A cell must offer at least this many valid cycle-level samples to count.
#: Configuration, not science: the default is a floor below which a cell
#: contributes almost nothing, not a claim about battery physics.
DEFAULT_MIN_SAMPLES_PER_CELL = 20

#: A dataset must offer at least this many eligible cells to be called ready.
DEFAULT_MIN_ELIGIBLE_CELLS = 5


@dataclass
class SOHCellReport:
    """What one processed cell can contribute."""

    cell_id: str
    eligible: bool
    n_cycles: int
    n_label_cycles: int
    n_samples: int
    reason: str | None = None


@dataclass
class SOHDatasetReport:
    """What one processed dataset can contribute, and why."""

    dataset: str
    processed_path: str
    status: str
    cells_inspected: int = 0
    cells_with_reference: int = 0
    cells_eligible: int = 0
    total_samples: int = 0
    reference_capacity_mode: str = ""
    sample_policy: dict = field(default_factory=dict)
    exclusions: dict = field(default_factory=dict)
    reason: str = ""
    cells: List[SOHCellReport] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.status == SOH_READY

    def to_dict(self, include_cells: bool = False) -> dict:
        data = asdict(self)
        if not include_cells:
            data.pop("cells", None)
        return data

    def summary_row(self) -> dict:
        """One row of the compact server table."""
        return {
            "Dataset": self.dataset,
            "Cells": self.cells_inspected,
            "Eligible cells": self.cells_eligible,
            "SOH samples": self.total_samples,
            "Status": self.status,
            "Reason": self.reason,
        }


def _load_cell(path: Path):
    """Load one processed cell, or None with the reason left to the caller.

    Uses ``BatteryData.load`` so this reads exactly what training reads; a
    file that training cannot open must not be counted as eligible here.
    """
    from batteryml.data.battery_data import BatteryData

    try:
        return BatteryData.load(str(path))
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, KeyError):
        return None


def assess_soh_cell(cell,
                    reference_capacity_mode: str,
                    reference_n_cycles: int,
                    min_cycle: int,
                    max_cycle: int | None,
                    cycle_stride: int,
                    min_samples_per_cell: int,
                    require_voltage: bool = True) -> SOHCellReport:
    """How many cycle-level SOH samples one cell offers.

    Counts through the same annotator and extractor a run would use, so the
    number reported here is the number a run would get — not an estimate from
    the cycle count.
    """
    from batteryml.feature.soh_cycle import SOHVoltageCurveFeatureExtractor
    from batteryml.label.soh import CycleSOHLabelAnnotator

    cell_id = str(getattr(cell, "cell_id", "") or "<unknown>")
    n_cycles = len(getattr(cell, "cycle_data", None) or [])

    annotator = CycleSOHLabelAnnotator(
        reference_capacity_mode=reference_capacity_mode,
        reference_n_cycles=reference_n_cycles,
        min_cycle=min_cycle,
        max_cycle=max_cycle,
        cycle_stride=cycle_stride,
    )
    labels, cycles = annotator.process_cell_cycles(cell)
    n_label_cycles = len(cycles)
    if n_label_cycles == 0:
        reason = next(iter(annotator.exclusions), "no_label_cycle")
        return SOHCellReport(cell_id, False, n_cycles, 0, 0, reason)

    n_samples = n_label_cycles
    if require_voltage:
        import torch

        features = SOHVoltageCurveFeatureExtractor().process_cell_cycles(cell, cycles)
        usable = torch.isfinite(features.view(len(features), -1)).all(dim=1)
        n_samples = int(usable.sum())
        if n_samples == 0:
            return SOHCellReport(
                cell_id, False, n_cycles, n_label_cycles, 0, "no_usable_voltage_trace"
            )

    if n_samples < min_samples_per_cell:
        return SOHCellReport(
            cell_id, False, n_cycles, n_label_cycles, n_samples,
            "too_few_valid_samples",
        )
    return SOHCellReport(cell_id, True, n_cycles, n_label_cycles, n_samples)


def assess_soh_dataset(dataset: str,
                       processed_root: str | Path = "data/processed",
                       reference_capacity_mode: str | None = None,
                       reference_n_cycles: int = 5,
                       min_cycle: int = 1,
                       max_cycle: int | None = None,
                       cycle_stride: int = 1,
                       min_samples_per_cell: int = DEFAULT_MIN_SAMPLES_PER_CELL,
                       min_eligible_cells: int = DEFAULT_MIN_ELIGIBLE_CELLS,
                       max_cells: int | None = None,
                       require_voltage: bool = True) -> SOHDatasetReport:
    """Whether ``dataset``'s processed cells can support SOH estimation.

    Args:
        dataset: the dataset name, e.g. ``"MATR"``.
        processed_root: where ``<dataset>/`` lives.
        reference_capacity_mode: the run's Q_ref convention. Defaults to the
            label module's default, so the audit reports what a run would get.
        max_cells: stop after this many cells. For a quick look on a large
            dataset; the report says how many were inspected, so a partial
            audit cannot be mistaken for a full one.

    Returns:
        A report carrying real counts, or a ``SOH_NOT_PROCESSED`` verdict with
        zero counts when there is nothing on disk.
    """
    from batteryml.label.soh import DEFAULT_REFERENCE_CAPACITY_MODE

    mode = reference_capacity_mode or DEFAULT_REFERENCE_CAPACITY_MODE
    policy = {
        "min_cycle": min_cycle,
        "max_cycle": max_cycle,
        "cycle_stride": cycle_stride,
        "min_samples_per_cell": min_samples_per_cell,
        "min_eligible_cells": min_eligible_cells,
    }
    path = Path(processed_root) / dataset
    report = SOHDatasetReport(
        dataset=dataset,
        processed_path=path.as_posix(),
        status=SOH_NOT_PROCESSED,
        reference_capacity_mode=mode,
        sample_policy=policy,
    )

    if not path.is_dir():
        report.reason = (
            f"{path.as_posix()} does not exist. Run preprocessing for "
            f"{dataset} first; SOH eligibility is decided from processed "
            "cells, never from a README."
        )
        return report

    files = sorted(p for p in path.glob("*.pkl") if p.is_file())
    if not files:
        report.reason = (
            f"{path.as_posix()} exists but holds no .pkl cell. Preprocessing "
            "has not produced output for this dataset."
        )
        return report

    if max_cells is not None:
        files = files[: int(max_cells)]

    unreadable = 0
    for file in files:
        cell = _load_cell(file)
        if cell is None:
            unreadable += 1
            continue
        report.cells_inspected += 1
        cell_report = assess_soh_cell(
            cell, mode, reference_n_cycles, min_cycle, max_cycle, cycle_stride,
            min_samples_per_cell, require_voltage,
        )
        report.cells.append(cell_report)
        if cell_report.n_label_cycles > 0:
            report.cells_with_reference += 1
        if cell_report.eligible:
            report.cells_eligible += 1
            report.total_samples += cell_report.n_samples
        elif cell_report.reason:
            report.exclusions[cell_report.reason] = \
                report.exclusions.get(cell_report.reason, 0) + 1
    if unreadable:
        report.exclusions["unreadable_file"] = unreadable

    if report.cells_inspected == 0:
        report.status = SOH_NOT_PROCESSED
        report.reason = (
            f"none of the {len(files)} file(s) under {path.as_posix()} could "
            "be loaded as a BatteryData cell."
        )
    elif report.cells_eligible >= min_eligible_cells:
        report.status = SOH_READY
        report.reason = (
            f"{report.cells_eligible} of {report.cells_inspected} cell(s) "
            f"yield at least {min_samples_per_cell} SOH sample(s) each, "
            f"{report.total_samples} in total."
        )
    elif report.cells_eligible > 0:
        report.status = SOH_INSUFFICIENT
        report.reason = (
            f"only {report.cells_eligible} eligible cell(s), fewer than the "
            f"{min_eligible_cells} required. A cell-level split needs enough "
            "cells on both sides to mean anything."
        )
    else:
        report.status = SOH_NOT_ELIGIBLE
        top = sorted(report.exclusions.items(), key=lambda kv: -kv[1])
        detail = ", ".join(f"{reason}={count}" for reason, count in top[:3])
        report.reason = (
            f"no cell yields a usable SOH sample under "
            f"reference_capacity_mode={mode}. Exclusions: {detail or 'none recorded'}."
        )
    return report


def assess_soh_datasets(datasets: Iterable[str],
                        processed_root: str | Path = "data/processed",
                        **kwargs) -> List[SOHDatasetReport]:
    """:func:`assess_soh_dataset` over several datasets, in the order given."""
    return [assess_soh_dataset(name, processed_root, **kwargs) for name in datasets]
