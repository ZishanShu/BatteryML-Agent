"""The SOH route end to end: features, training, evaluation, artifact.

These tests run the real BatteryML pipeline — a real splitter, the real
``Task.build``, a real registered model, real ``pipeline.evaluate`` — over
synthetic cells written to a temporary directory. Nothing is mocked, so what
is established here is the route itself rather than a description of it.

The feature tests are the causality ones: a row for cycle ``t`` must be
unchanged when every later cycle is replaced. That is proved by mutation, not
asserted.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_soh_pipeline -v
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from batteryml.feature.soh_cycle import (  # noqa: E402
    SUMMARY_CHANNELS,
    SOHVoltageCurveFeatureExtractor,
    resample,
)
from batteryml.label.soh import CycleSOHLabelAnnotator  # noqa: E402
from batteryml.task import Task  # noqa: E402
from tests.fixtures.synthetic_soh_cells import (  # noqa: E402
    cohort,
    make_cell,
    make_cycle,
)


def _write_cells(cells, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        cell.dump(str(directory / f"{cell.cell_id}.pkl"))
    return directory


class TestSOHFeatureExtractor(unittest.TestCase):
    """One canonical feature path: the voltage curve observed at cycle t."""

    def setUp(self) -> None:
        self.extractor = SOHVoltageCurveFeatureExtractor()
        self.cell = cohort(n_cells=1, n_cycles=12)[0]
        self.cycles = list(range(3, 10))

    def test_shape_is_fixed_and_documented(self) -> None:
        features = self.extractor.process_cell_cycles(self.cell, self.cycles)
        self.assertEqual(features.shape[0], len(self.cycles))
        self.assertEqual(features.shape[1], self.extractor.feature_dim)
        self.assertEqual(
            self.extractor.feature_dim, 32 + len(SUMMARY_CHANNELS)
        )
        self.assertEqual(self.extractor.describe()["feature_dim"], features.shape[1])

    def test_one_row_per_requested_cycle_in_order(self) -> None:
        features = self.extractor.process_cell_cycles(self.cell, [5, 3, 7])
        alone = [
            self.extractor.process_cell_cycles(self.cell, [cycle])[0]
            for cycle in (5, 3, 7)
        ]
        for i, row in enumerate(alone):
            self.assertTrue(torch.equal(features[i], row))

    def test_extraction_is_deterministic(self) -> None:
        first = self.extractor.process_cell_cycles(self.cell, self.cycles)
        second = self.extractor.process_cell_cycles(self.cell, self.cycles)
        self.assertTrue(torch.equal(first, second))

    def test_capacity_is_not_a_feature(self) -> None:
        """Feeding Q_discharge(t) in would make the label a division.

        Two cells with identical voltage traces but different capacities must
        produce identical features — otherwise capacity has leaked in and the
        model is being handed its own target.
        """
        low = make_cell("synthetic_low", [0.5, 0.5, 0.5])
        high = make_cell("synthetic_high", [1.0, 1.0, 1.0])
        # make_cell derives the voltage trace identically regardless of
        # capacity, so any difference here comes from capacity.
        features_low = self.extractor.process_cell_cycles(low, [2])
        features_high = self.extractor.process_cell_cycles(high, [2])
        self.assertTrue(torch.equal(features_low, features_high))

        source = Path(SOHVoltageCurveFeatureExtractor.__module__.replace(".", "/") + ".py")
        text = (REPO_ROOT / source).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("#")
        )
        self.assertNotIn("discharge_capacity_in_Ah", code.split('"""')[-1])

    def test_a_future_cycle_cannot_change_an_earlier_sample(self) -> None:
        """The leakage mutation test: rewrite the future, check the past.

        For a sample at cycle t, replacing every cycle after t with wildly
        different data must leave the row bit-identical. If it changes, the
        feature is reading information that would not exist yet at t.
        """
        cell = cohort(n_cells=1, n_cycles=20)[0]
        target = 8
        before = self.extractor.process_cell_cycles(cell, [target])

        mutated = cohort(n_cells=1, n_cycles=20)[0]
        for position, cycle in enumerate(mutated.cycle_data):
            if int(cycle.cycle_number) <= target:
                continue
            mutated.cycle_data[position] = make_cycle(
                int(cycle.cycle_number), 0.001, n_points=9
            )
            mutated.cycle_data[position].voltage_in_V = [99.0] * 9
        after = self.extractor.process_cell_cycles(mutated, [target])

        self.assertTrue(
            torch.equal(before, after),
            "a cycle after t changed the feature for t: the extractor is "
            "reading the future",
        )

    def test_the_mutation_test_is_not_vacuous(self) -> None:
        """Mutating cycle t itself must change the row, or the check proves nothing."""
        cell = cohort(n_cells=1, n_cycles=20)[0]
        target = 8
        before = self.extractor.process_cell_cycles(cell, [target])
        for position, cycle in enumerate(cell.cycle_data):
            if int(cycle.cycle_number) == target:
                cell.cycle_data[position].voltage_in_V = [99.0] * 4
        after = self.extractor.process_cell_cycles(cell, [target])
        self.assertFalse(torch.equal(before, after))

    def test_a_history_window_reads_backwards_only(self) -> None:
        extractor = SOHVoltageCurveFeatureExtractor(n_history_cycles=3)
        cell = cohort(n_cells=1, n_cycles=20)[0]
        target = 10
        before = extractor.process_cell_cycles(cell, [target])
        for position, cycle in enumerate(cell.cycle_data):
            if int(cycle.cycle_number) > target:
                cell.cycle_data[position].voltage_in_V = [42.0] * 4
        after = extractor.process_cell_cycles(cell, [target])
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(before.shape[1], 3 * (32 + len(SUMMARY_CHANNELS)))

    def test_an_unreadable_cycle_yields_nan_not_a_substitute(self) -> None:
        cell = make_cell("synthetic_novoltage", [1.0, 0.9, 0.8])
        cell.cycle_data[1].voltage_in_V = None
        features = self.extractor.process_cell_cycles(cell, [1, 2, 3])
        self.assertTrue(torch.isfinite(features[0]).all())
        self.assertTrue(torch.isnan(features[1]).all())
        self.assertTrue(torch.isfinite(features[2]).all())

    def test_resample_is_linear_and_fixed_length(self) -> None:
        self.assertEqual(resample([0.0, 1.0], 3), [0.0, 0.5, 1.0])
        self.assertEqual(resample([5.0], 4), [5.0, 5.0, 5.0, 5.0])
        self.assertEqual(len(resample([0.0, 1.0, 2.0, 3.0], 7)), 7)

    def test_it_refuses_the_whole_cell_contract(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self.extractor.process_cell(self.cell)
        self.assertIn("cycle-level", str(ctx.exception))


class TestSOHTaskBuild(unittest.TestCase):
    """Task.build over real files, with a real cell-level splitter."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cells = cohort(n_cells=8, n_cycles=30)
        _write_cells(self.cells, self.root / "cells")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _task(self, **label_kwargs) -> Task:
        return Task(
            train_test_splitter={
                "name": "RandomTrainTestSplitter",
                "cell_data_path": str(self.root / "cells"),
                "seed": 0,
                "train_test_split_ratio": 0.6,
            },
            feature_extractor={"name": "SOHVoltageCurveFeatureExtractor"},
            label_annotator={"name": "CycleSOHLabelAnnotator", **label_kwargs},
        )

    def test_it_builds_cycle_level_samples(self) -> None:
        task = self._task()
        self.assertTrue(task.is_cycle_level)
        bundle = task.build()
        n_train = len(bundle.train_data.label)
        n_test = len(bundle.test_data.label)
        # Many samples per cell, not one.
        self.assertGreater(n_train, len(task.train_cells))
        self.assertGreater(n_test, len(task.test_cells))

    def test_identity_is_carried_and_aligned(self) -> None:
        bundle = self._task().build()
        for split in (bundle.train_data, bundle.test_data):
            self.assertIsNotNone(split.sample_index)
            self.assertEqual(len(split.sample_index), len(split.label))
            self.assertEqual(len(split.sample_index), len(split.feature))

    def test_no_cell_appears_in_both_splits(self) -> None:
        bundle = self._task().build()
        train_ids = set(bundle.train_data.sample_index.unique_cell_ids)
        test_ids = set(bundle.test_data.sample_index.unique_cell_ids)
        self.assertTrue(train_ids)
        self.assertTrue(test_ids)
        self.assertEqual(train_ids & test_ids, set())

    def test_the_sample_policy_controls_the_count(self) -> None:
        dense = self._task().build()
        sparse = self._task(min_cycle=10, cycle_stride=5).build()
        self.assertLess(len(sparse.train_data.label), len(dense.train_data.label))
        cycles = sparse.train_data.sample_index.cycle_indices
        self.assertTrue(all(cycle >= 10 for cycle in cycles))

    def test_the_build_is_reproducible(self) -> None:
        first = self._task().build()
        second = self._task().build()
        self.assertEqual(
            first.train_data.sample_index.to_records(),
            second.train_data.sample_index.to_records(),
        )
        self.assertTrue(torch.equal(first.train_data.label, second.train_data.label))

    def test_pairing_a_whole_cell_extractor_with_the_soh_annotator_is_refused(self) -> None:
        task = Task(
            train_test_splitter={
                "name": "RandomTrainTestSplitter",
                "cell_data_path": str(self.root / "cells"),
            },
            feature_extractor={
                "name": "VoltageCapacityMatrixFeatureExtractor",
                "diff_base": 2, "max_cycle_index": 19, "cycles_to_keep": 19,
            },
            label_annotator={"name": "CycleSOHLabelAnnotator"},
        )
        with self.assertRaises(TypeError) as ctx:
            task.build()
        self.assertIn("BaseCycleFeatureExtractor", str(ctx.exception))


class TestSOHTrainEvaluateArtifact(unittest.TestCase):
    """A real train/eval run, and what it leaves on disk."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_cells(cohort(n_cells=10, n_cycles=40), self.root / "cells")
        self.workspace = self.root / "workspace"
        self.config_path = self.root / "soh_test.yaml"
        self.config_path.write_text(
            "model:\n"
            "    name: 'RidgeRULPredictor'\n"
            "train_test_split:\n"
            "    name: 'RandomTrainTestSplitter'\n"
            f"    cell_data_path: '{(self.root / 'cells').as_posix()}'\n"
            "    seed: 0\n"
            "    train_test_split_ratio: 0.6\n"
            "feature:\n"
            "    name: 'SOHVoltageCurveFeatureExtractor'\n"
            "    n_voltage_points: 16\n"
            "label:\n"
            "    name: 'CycleSOHLabelAnnotator'\n"
            "    reference_capacity_mode: 'first_valid_discharge_capacity'\n"
            "    min_cycle: 2\n"
            "    cycle_stride: 2\n"
            "feature_transformation:\n"
            "    name: 'ZScoreDataTransformation'\n"
            "label_transformation:\n"
            "    name: 'ZScoreDataTransformation'\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self):
        from batteryml.pipeline import Pipeline

        pipeline = Pipeline(str(self.config_path), workspace=str(self.workspace))
        model, dataset = pipeline.train(device="cpu", skip_if_executed=False)
        pipeline.evaluate(model=model, dataset=dataset, metric=["MAE", "RMSE"],
                          skip_if_executed=False)
        return pipeline, model, dataset

    def test_the_route_trains_and_evaluates(self) -> None:
        _, model, dataset = self._run()
        prediction = model.predict(dataset)
        self.assertEqual(len(prediction), len(dataset.test_data.label))

    def test_canonical_metrics_are_mae_and_rmse(self) -> None:
        _, model, dataset = self._run()
        prediction = model.predict(dataset)
        for metric in ("MAE", "RMSE"):
            value = dataset.evaluate(prediction, metric)
            self.assertGreaterEqual(value, 0.0)
            # SOH is a ratio in roughly [0, 1], so an error of this scale is
            # a ratio error too, not a cycle count.
            self.assertLess(value, 1.0, f"{metric} looks like it is not on a ratio scale")

    def test_the_artifact_carries_cell_and_cycle(self) -> None:
        self._run()
        files = sorted(self.workspace.glob("predictions_seed_*.pkl"))
        self.assertTrue(files, "evaluate wrote no prediction artifact")
        with open(files[-1], "rb") as handle:
            payload = pickle.load(handle)

        self.assertIn("sample_index", payload)
        records = payload["sample_index"]
        self.assertEqual(len(records), len(payload["prediction"]))
        for record in records[:5]:
            self.assertIn("cell_id", record)
            self.assertIn("cycle_index", record)
            self.assertTrue(str(record["cell_id"]).startswith("synthetic_cell_"))
            self.assertGreaterEqual(record["cycle_index"], 2)

    def test_the_artifact_records_the_reference_convention(self) -> None:
        self._run()
        files = sorted(self.workspace.glob("predictions_seed_*.pkl"))
        with open(files[-1], "rb") as handle:
            payload = pickle.load(handle)
        metadata = payload["task_metadata"]
        self.assertEqual(metadata["label_annotator"], "CycleSOHLabelAnnotator")
        self.assertEqual(
            metadata["reference_capacity_mode"], "first_valid_discharge_capacity"
        )
        self.assertEqual(metadata["feature_extractor"], "SOHVoltageCurveFeatureExtractor")

    def test_observed_and_predicted_can_be_paired_per_cell_and_cycle(self) -> None:
        """What the SOH-vs-cycle view needs: rows that know where they belong."""
        _, model, dataset = self._run()
        prediction = model.predict(dataset)
        index = dataset.test_data.sample_index
        observed = dataset.label_transformation.inverse_transform(
            dataset.test_data.label) if dataset.label_transformation else dataset.test_data.label

        by_cell: dict[str, list[tuple[int, float]]] = {}
        for i in range(len(index)):
            record = index[i]
            by_cell.setdefault(record["cell_id"], []).append(
                (record["cycle_index"], float(observed[i]))
            )
        self.assertTrue(by_cell)
        for cell_id, points in by_cell.items():
            cycles = [cycle for cycle, _ in points]
            self.assertEqual(len(cycles), len(set(cycles)),
                             f"{cell_id} has a duplicated cycle")
        self.assertEqual(len(prediction), len(index))

    def test_no_rul_label_is_involved(self) -> None:
        _, _, dataset = self._run()
        labels = dataset.test_data.label
        if dataset.label_transformation is not None:
            labels = dataset.label_transformation.inverse_transform(labels)
        self.assertLess(float(labels.max()), 2.0,
                        "labels look like cycle counts, not SOH ratios")
        self.assertGreater(float(labels.min()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
