"""Read-only native verification of a saved run (NATIVE-01).

A synthetic run is built in a temporary directory with BatteryML's own
``DataBundle`` and its LogScale→ZScore label transformation, scored with
BatteryML's own evaluator, and pickled the way ``Pipeline.evaluate`` saves
it. Every value is synthetic and lives only in the temporary directory.

    python -m unittest tests.test_native_check -v
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import batteryml  # noqa: E402,F401 - registers transformations and annotators
from batteryml.builders import DATA_TRANSFORMATIONS  # noqa: E402
from batteryml.data import DataBundle  # noqa: E402
from batteryml.data.sample_index import SampleIndex  # noqa: E402
from batteryml.label.rul import RULLabelAnnotator  # noqa: E402
from batteryml_web.core import native_check as nc  # noqa: E402
from scripts import native_run_check  # noqa: E402

LABEL_TRANSFORMATION = {
    "name": "SequentialDataTransformation",
    "transformations": [{"name": "LogScaleDataTransformation"}, {"name": "ZScoreDataTransformation"}],
}
SYNTHETIC_TRAIN = [480.0, 520.0, 700.0, 860.0, 1010.0, 1190.0, 1500.0, 1730.0]
SYNTHETIC_TEST = [450.0, 610.0, 905.0, 1320.0, 2100.0]
SYNTHETIC_IDS = ["synthetic_c0", "synthetic_c1", "synthetic_c2", "synthetic_c3", "synthetic_c4"]


def _synthetic_run(root: Path) -> dict:
    run_dir = root / "synthetic_project" / "quickstart" / "run_synthetic_xgb"
    run_dir.mkdir(parents=True)
    transform = DATA_TRANSFORMATIONS.build(dict(LABEL_TRANSFORMATION))
    bundle = DataBundle(
        torch.ones(len(SYNTHETIC_TRAIN), 3), torch.tensor(SYNTHETIC_TRAIN),
        torch.ones(len(SYNTHETIC_TEST), 3), torch.tensor(SYNTHETIC_TEST),
        label_transformation=transform,
        test_sample_index=SampleIndex.for_whole_cells(SYNTHETIC_IDS),
    )
    prediction = bundle.test_data.label + torch.tensor([0.05, -0.1, 0.2, -0.05, 0.1])
    scores = {m: bundle.evaluate(prediction, m) for m in ("RMSE", "MAE", "MAPE")}
    artifact = run_dir / "predictions_seed_0_20260101000000.pkl"
    artifact.write_bytes(pickle.dumps({"prediction": prediction, "scores": scores, "data": bundle, "seed": 0}))
    config = run_dir / "config_20260101000000.yaml"
    config.write_text("label:\n    name: 'RULLabelAnnotator'\n", encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_dir.name, "dataset": "MATR", "task": "RUL prediction", "model_key": "xgb",
        "seed": 0, "status": "COMPLETED", "metrics": scores, "config_path": str(config),
    }), encoding="utf-8")
    record = root / "record.json"
    record.write_text(json.dumps({"record_id": "synthetic_record", "files": {
        "predictions": {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}}), encoding="utf-8")
    return {"run_dir": run_dir, "record": record, "scores": scores, "artifact": artifact}


class TestPureChecks(unittest.TestCase):
    def test_labels_compared_in_order(self) -> None:
        ok = nc.compare_labels(["a", "b"], [500.0, 900.0], [500.0001, 899.99])
        self.assertEqual(ok["status"], nc.MATCH)
        swapped = nc.compare_labels(["a", "b"], [500.0, 900.0], [900.0, 500.0])
        self.assertEqual(swapped["status"], nc.MISMATCH)
        self.assertEqual(nc.compare_labels(["a"], [1.0], [1.0, 2.0])["status"], nc.MISMATCH)

    def test_saved_ids_must_equal_rebuilt_ids(self) -> None:
        result = nc.compare_labels(["a", "b"], [1.0, 2.0], [1.0, 2.0], saved_ids=["b", "a"])
        self.assertFalse(result["saved_cell_ids_equal_rebuilt"])
        self.assertEqual(result["status"], nc.MISMATCH)

    def test_rul_target_is_read_from_the_annotator_not_its_name(self) -> None:
        target = nc.scientific_target(RULLabelAnnotator())
        self.assertEqual((target["name"], target["unit"]), ("total cycle life", "cycles"))
        self.assertEqual(target["parameters"], {"eol_soh": 0.8, "pad_eol": True, "min_rul_limit": 100.0})
        self.assertIn("No observation point is subtracted", target["definition"])
        self.assertEqual(len(target["source_sha256"]), 64)

    def test_an_unknown_annotator_is_not_guessed(self) -> None:
        self.assertEqual(nc.scientific_target(object())["status"], nc.NOT_RECORDED)

    def test_fitted_state_is_read_with_its_order(self) -> None:
        transform = DATA_TRANSFORMATIONS.build(dict(LABEL_TRANSFORMATION))
        transform.fit(torch.tensor(SYNTHETIC_TRAIN))
        state = nc.transform_state(transform)
        self.assertEqual(state["order"], ["LogScaleDataTransformation", "ZScoreDataTransformation"])
        self.assertEqual(len(state["children"][1]["mean"]), 1)
        self.assertEqual(nc.compare_fitted_state(state, nc.transform_state(transform))["status"], nc.MATCH)


class TestNativeRunCheck(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run = _synthetic_run(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _check(self, *extra: str) -> dict:
        out = self.root / "out" / f"check{len(list((self.root).glob('out/*')))}"
        native_run_check.main(["--run-dir", str(self.run["run_dir"]), "--output", str(out), *extra])
        return json.loads((out / nc.REPORT_FILENAME).read_text(encoding="utf-8"))

    def test_a_matching_artifact_is_replayed_and_its_scale_established(self) -> None:
        report = self._check("--record", str(self.run["record"]))
        self.assertEqual(report["files"]["predictions"]["fingerprint_status"], nc.MATCH)
        replay = report["metric_replay"]
        self.assertEqual(replay["status"], nc.VERIFIED)
        for metric, row in replay["metrics"].items():
            self.assertEqual(row["stored_score"], self.run["scores"][metric])
            self.assertTrue(row["evaluator_equals_stored"], metric)
            self.assertTrue(row["independent_matches_stored"], metric)
            self.assertTrue(row["manifest_equals_stored"], metric)
        self.assertEqual(replay["mape_zero_target_count"], 0)
        self.assertEqual(report["array_scale"]["status"], nc.VERIFIED)
        self.assertIn("LogScaleDataTransformation then ZScoreDataTransformation", report["array_scale"]["stored_arrays"])
        # The native-scale targets are the synthetic cycle counts again.
        for got, want in zip(replay["target_native_values"], SYNTHETIC_TEST):
            self.assertAlmostEqual(got, want, delta=want * 1e-4)
        self.assertEqual(report["payload"]["saved_cell_ids"], SYNTHETIC_IDS)
        self.assertEqual(report["identity"]["status"], nc.NOT_PERFORMED)

    def test_a_mismatched_artifact_is_not_loaded(self) -> None:
        record = json.loads(self.run["record"].read_text())
        record["files"]["predictions"]["sha256"] = "0" * 64
        self.run["record"].write_text(json.dumps(record))
        report = self._check("--record", str(self.run["record"]))
        self.assertEqual(report["files"]["predictions"]["fingerprint_status"], nc.MISMATCH)
        self.assertEqual(report["payload"]["status"], nc.NOT_PERFORMED)
        self.assertNotIn("metric_replay", report)

    def test_an_unrecorded_artifact_needs_explicit_acceptance(self) -> None:
        self.assertEqual(self._check()["payload"]["status"], nc.NOT_PERFORMED)
        self.assertEqual(self._check("--accept-unrecorded")["metric_replay"]["status"], nc.VERIFIED)

    def test_the_run_directory_is_never_written(self) -> None:
        before = sorted((p.name, p.stat().st_mtime_ns) for p in self.run["run_dir"].iterdir())
        self._check("--record", str(self.run["record"]))
        after = sorted((p.name, p.stat().st_mtime_ns) for p in self.run["run_dir"].iterdir())
        self.assertEqual(before, after)
        with self.assertRaises(SystemExit):
            native_run_check.main(["--run-dir", str(self.run["run_dir"]),
                                   "--output", str(self.run["run_dir"] / "inside")])

    def test_the_default_report_location_is_the_project_native_checks_folder(self) -> None:
        folder = nc.default_output_dir(self.run["run_dir"], "STAMP")
        self.assertEqual(folder, self.root / "synthetic_project" / "native_checks" / "run_synthetic_xgb" / "STAMP")


if __name__ == "__main__":
    unittest.main()


class _FakeSplitter:
    """Returns fixed synthetic cell files; stands in for MATR's fixed id lists."""

    def __init__(self, train, test):
        self.train, self.test = train, test

    def split(self):
        return self.train, self.test


class TestRebuiltIdentities(unittest.TestCase):
    """--rebuild-identities on synthetic cells: labels, order, exclusions, fitted state."""

    def test_rebuilt_labels_match_the_stored_targets_in_order(self) -> None:
        from unittest import mock

        from batteryml.builders import TRAIN_TEST_SPLITTERS
        from tests.fixtures.synthetic_soh_cells import make_cell

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            processed.mkdir()

            def cell(name: str, life: int):
                capacities = [1.1 - 0.4 * i / life for i in range(life + 20)]
                path = processed / f"synthetic_{name}.pkl"
                make_cell(f"synthetic_{name}", capacities, nominal_capacity_in_Ah=1.1).dump(str(path))
                return path

            train = [cell(f"tr{i}", life) for i, life in enumerate((180, 260, 340, 420, 500))]
            test = [cell("te0", 300), cell("te1_short", 40), cell("te2", 210), cell("te3", 460)]
            annotator = RULLabelAnnotator()
            from batteryml.data import BatteryData

            def labels(paths):
                values = annotator([BatteryData.load(str(p)) for p in paths])
                keep = ~torch.isnan(values)
                return values[keep], [p.stem for p, k in zip(paths, keep.tolist()) if k]

            train_labels, _ = labels(train)
            test_labels, test_ids = labels(test)
            self.assertNotIn("synthetic_te1_short", test_ids)          # label <= 100 is excluded

            run_dir = root / "proj" / "quickstart" / "run_synthetic_rebuilt"
            run_dir.mkdir(parents=True)
            transform = DATA_TRANSFORMATIONS.build(dict(LABEL_TRANSFORMATION))
            bundle = DataBundle(torch.ones(len(train_labels), 2), train_labels,
                                torch.ones(len(test_labels), 2), test_labels, label_transformation=transform)
            prediction = bundle.test_data.label * 0.9
            scores = {m: bundle.evaluate(prediction, m) for m in ("RMSE", "MAE", "MAPE")}
            artifact = run_dir / "predictions_seed_0_20260101000000.pkl"
            artifact.write_bytes(pickle.dumps({"prediction": prediction, "scores": scores, "data": bundle, "seed": 0}))
            config = root / "synthetic_config.yaml"
            config.write_text(
                "train_test_split:\n    name: 'SyntheticSplitter'\n    cell_data_path: '" + str(processed) + "'\n"
                "label:\n    name: 'RULLabelAnnotator'\n"
                "label_transformation:\n    name: 'SequentialDataTransformation'\n    transformations:\n"
                "        - name: 'LogScaleDataTransformation'\n        - name: 'ZScoreDataTransformation'\n",
                encoding="utf-8")
            (run_dir / "run.json").write_text(json.dumps({"run_id": run_dir.name, "metrics": scores,
                                                         "config_path": str(config)}))
            out = root / "report"
            with mock.patch.object(TRAIN_TEST_SPLITTERS, "build",
                                   lambda *a, **k: _FakeSplitter([str(p) for p in train], [str(p) for p in test])):
                native_run_check.main(["--run-dir", str(run_dir), "--output", str(out),
                                       "--accept-unrecorded", "--rebuild-identities"])
            report = json.loads((out / nc.REPORT_FILENAME).read_text())
            identity = report["identity"]
            self.assertEqual(identity["status"], nc.MATCH)
            self.assertEqual(identity["rebuilt_cell_ids"], test_ids)
            self.assertEqual(identity["test_excluded_ids"], ["synthetic_te1_short"])
            self.assertEqual(report["fitted_state_check"]["status"], nc.MATCH)
            self.assertEqual(report["scientific_target"]["rebuilt_labels_match_stored_targets"], nc.MATCH)
            self.assertEqual(report["scientific_target"]["name"], "total cycle life")
