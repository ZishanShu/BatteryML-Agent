# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import abc
import torch
from typing import List, Sequence, Tuple

from batteryml.data.battery_data import BatteryData


class BaseLabelAnnotator(abc.ABC):
    """One label per cell. The original contract, unchanged."""

    def __call__(self, cells: List[BatteryData]):
        return torch.stack([
            self.process_cell(cell) for cell in cells]).float().view(-1)

    @abc.abstractmethod
    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        """Generate label for a single cell.

        Args:
            cell_data (BatteryData): data for single cell.

        Returns:
            torch.Tensor: the processed label.
        """


class BaseCycleLabelAnnotator(BaseLabelAnnotator):
    """Many labels per cell — one per observed cycle.

    A cycle-level task such as SOH estimation produces a label for each cycle
    the sample policy accepts, so one cell yields ``n`` rows rather than one.
    This subclass exists to make that visible in the type: :class:`Task`
    switches to the cycle-level path when its label annotator is one of these,
    and a one-sample-per-cell annotator keeps the old path untouched.

    **The annotator owns which cycles are sampled.** It is the component that
    knows which cycles carry a usable label, and the feature extractor is then
    asked for those exact cycles. Having one owner is what makes the feature
    rows and the label rows line up by construction, instead of two components
    independently deciding and being trusted to agree.

    ``process_cell`` is still implemented, returning the first sampled label,
    so a cycle-level annotator remains usable wherever the old contract is
    expected. That path is a convenience, not the intended use.
    """

    @abc.abstractmethod
    def process_cell_cycles(
        self, cell_data: BatteryData
    ) -> Tuple[torch.Tensor, Sequence[int]]:
        """Labels for one cell, and the cycles they were taken at.

        Args:
            cell_data: data for a single cell.

        Returns:
            ``(labels, cycle_indices)`` where ``labels`` is a 1-D tensor and
            ``cycle_indices`` has the same length, in the same order, using
            BatteryML's 1-based ``CycleData.cycle_number`` convention. Both
            are empty when the cell contributes no valid sample.
        """

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        labels, _ = self.process_cell_cycles(cell_data)
        if len(labels) == 0:
            return torch.tensor(float('nan'))
        return labels.view(-1)[0]
