# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch

from tqdm import tqdm

from batteryml.builders import (
    FEATURE_EXTRACTORS,
    LABEL_ANNOTATORS,
    TRAIN_TEST_SPLITTERS,
    DATA_TRANSFORMATIONS
)
from batteryml.data import BatteryData, DataBundle
from batteryml.data.sample_index import SampleIndex, assert_disjoint_cells
from batteryml.data.transformation.base import BaseDataTransformation
from batteryml.feature.base import BaseCycleFeatureExtractor
from batteryml.label.base import BaseCycleLabelAnnotator


class Task:
    def __init__(self,
                 train_test_splitter: dict,
                 feature_extractor: dict,
                 label_annotator: dict,
                 feature_transformation: BaseDataTransformation = None,
                 label_transformation: BaseDataTransformation = None):
        if isinstance(train_test_splitter, dict):
            train_test_splitter = \
                TRAIN_TEST_SPLITTERS.build(train_test_splitter, 'raise')
        if isinstance(feature_extractor, dict):
            feature_extractor = FEATURE_EXTRACTORS.build(feature_extractor)
        if isinstance(label_annotator, dict):
            label_annotator = LABEL_ANNOTATORS.build(label_annotator)
        if isinstance(feature_transformation, dict):
            feature_transformation = DATA_TRANSFORMATIONS.build(
                feature_transformation)
        if isinstance(label_transformation, dict):
            label_transformation = DATA_TRANSFORMATIONS.build(
                label_transformation)

        self.train_test_splitter = train_test_splitter
        self.feature_extractor = feature_extractor
        self.label_annotator = label_annotator
        self.feature_transformation = feature_transformation
        self.label_transformation = label_transformation

    @property
    def is_cycle_level(self) -> bool:
        """Whether this task produces many samples per cell.

        Decided by the label annotator, which is the component that owns which
        cycles are sampled. A whole-cell annotator keeps the original
        one-row-per-cell path untouched.
        """
        return isinstance(self.label_annotator, BaseCycleLabelAnnotator)

    def build(self) -> DataBundle:
        # Loading data. The split is over *cells*, and it happens here, before
        # anything is expanded into cycle-level samples. For a cycle-level
        # task that ordering is the scientific invariant: splitting expanded
        # rows instead would put one cell's early cycles in train and its late
        # cycles in test, and the score would measure memorisation of that
        # cell. assert_disjoint_cells below checks the result rather than
        # trusting this ordering to stay correct.
        train_list, test_list = self.train_test_splitter.split()
        pbar = tqdm(train_list, desc='Reading train data')
        train_cells = [BatteryData.load(path) for path in pbar]
        pbar = tqdm(test_list, desc='Reading test data')
        test_cells = [BatteryData.load(path) for path in pbar]

        self.train_cells = train_cells
        self.test_cells = test_cells

        assert_disjoint_cells({
            'train': [cell.cell_id for cell in train_cells],
            'test': [cell.cell_id for cell in test_cells],
        })

        if self.is_cycle_level:
            train_features, train_labels, train_index = \
                self._build_cycle_level(train_cells, 'train')
            test_features, test_labels, test_index = \
                self._build_cycle_level(test_cells, 'test')
        else:
            train_features = self.feature_extractor(train_cells)
            test_features = self.feature_extractor(test_cells)
            train_labels = self.label_annotator(train_cells)
            test_labels = self.label_annotator(test_cells)
            train_index = SampleIndex.for_whole_cells(
                [cell.cell_id for cell in train_cells])
            test_index = SampleIndex.for_whole_cells(
                [cell.cell_id for cell in test_cells])

        # Omit NaN label samples. The sample index is filtered by the same
        # mask, so a row's identity still matches its feature and label.
        train_mask = ~torch.isnan(train_labels)
        test_mask = ~torch.isnan(test_labels)
        train_features = train_features[train_mask]
        test_features = test_features[test_mask]
        train_index = train_index.select(train_mask.tolist())
        test_index = test_index.select(test_mask.tolist())
        train_labels = train_labels[train_mask]
        test_labels = test_labels[test_mask]

        dataset = DataBundle(
            train_features, train_labels, test_features, test_labels,
            feature_transformation=self.feature_transformation,
            label_transformation=self.label_transformation,
            train_sample_index=train_index,
            test_sample_index=test_index,
        )

        return dataset

    def _build_cycle_level(self, cells, split_name: str):
        """Expand already-split cells into cycle-level samples.

        The label annotator decides which cycles each cell offers, and the
        feature extractor is then asked for exactly those cycles in that
        order. One owner of the cycle list is what makes the feature rows and
        label rows line up by construction.
        """
        if not isinstance(self.feature_extractor, BaseCycleFeatureExtractor):
            raise TypeError(
                f"{type(self.label_annotator).__name__} is a cycle-level "
                "label annotator, so the feature extractor must be a "
                "BaseCycleFeatureExtractor; got "
                f"{type(self.feature_extractor).__name__}. Pair the SOH "
                "annotator with an SOH feature extractor in the config."
            )

        feature_blocks, label_blocks, index_parts = [], [], []
        pbar = tqdm(cells, desc=f'Extracting {split_name} cycle samples')
        for cell in pbar:
            labels, cycles = self.label_annotator.process_cell_cycles(cell)
            if len(cycles) == 0:
                continue
            features = self.feature_extractor.process_cell_cycles(cell, cycles)
            if len(features) != len(labels):
                raise ValueError(
                    f"cell {cell.cell_id}: feature extractor returned "
                    f"{len(features)} row(s) for {len(cycles)} requested "
                    f"cycle(s) with {len(labels)} label(s). A cycle-level "
                    "extractor must return one row per requested cycle, in "
                    "order."
                )
            feature_blocks.append(features)
            label_blocks.append(labels.view(-1))
            index_parts.append(SampleIndex.for_cell(cell.cell_id, cycles))

        if not feature_blocks:
            raise ValueError(
                f"No cycle-level SOH sample could be built for the "
                f"{split_name} split: every cell was excluded. Check the "
                "reference-capacity mode and the sample policy "
                "(min_cycle/max_cycle/cycle_stride) against the processed "
                "data; scripts/audit_soh_readiness.py reports why per dataset."
            )

        features = torch.cat(feature_blocks, dim=0).float()
        labels = torch.cat(label_blocks, dim=0).float().view(-1)
        index = SampleIndex.concat(index_parts)
        index.check_matches(len(features), f'{split_name} split')
        return features, labels, index

    def get_raw_data(self):
        return self.train_cells, self.test_cells
