# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import abc
import torch

from tqdm import tqdm
from typing import List, Sequence

from batteryml.data import BatteryData


class BaseFeatureExtractor(abc.ABC):
    """One feature row per cell. The original contract, unchanged."""

    def __call__(self, cells: List[BatteryData]):
        pbar = tqdm(cells, desc='Extracting features')
        # features = torch.stack([self.process_cell(cell) for cell in pbar])
        features = []
        for i, cell in enumerate(pbar):
            features.append(self.process_cell(cell))
        features = torch.stack(features)
        return features.float()

    @abc.abstractmethod
    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        """Generate feature for a single cell.

        Args:
            cell_data (BatteryData): data for single cell.

        Returns:
            torch.Tensor: the processed feature.
        """


class BaseCycleFeatureExtractor(BaseFeatureExtractor):
    """Many feature rows per cell — one per requested observed cycle.

    The cycle list is *given*, not chosen here: the cycle-level label
    annotator decides which cycles are sampled and this extractor is asked for
    exactly those, in that order. That is what keeps feature rows and label
    rows aligned without the two components having to agree independently.

    Implementations must honour the causality rule for an at-cycle estimate:
    the row for cycle ``t`` may read cycles up to and including ``t`` and
    nothing after it. Reading ``t+1``, the cell's final cycle, or its lifetime
    turns the task into forecasting and is no longer a current-cycle estimate.
    """

    @abc.abstractmethod
    def process_cell_cycles(
        self, cell_data: BatteryData, cycle_indices: Sequence[int]
    ) -> torch.Tensor:
        """Features for one cell at the given cycles.

        Args:
            cell_data: data for a single cell.
            cycle_indices: the observed cycles to produce a row for, in
                BatteryML's 1-based ``CycleData.cycle_number`` convention.

        Returns:
            A tensor whose leading dimension is ``len(cycle_indices)``, in the
            same order.
        """

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        raise TypeError(
            f"{type(self).__name__} is a cycle-level feature extractor and "
            "needs the cycles to extract at. It is driven by a cycle-level "
            "label annotator through Task.build(); pairing it with a "
            "whole-cell annotator is a configuration error."
        )
