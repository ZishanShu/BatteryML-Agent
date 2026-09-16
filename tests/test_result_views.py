"""Scientific result views, the export bundle and artifact-browser safety.

Fixtures here are synthetic and clearly named as such (CLAUDE.md §7): they
exist only to exercise the reading and shaping code, are built in temporary
directories, and can never be picked up as a dataset or reach a metrics file.
No number produced here is a metric.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_result_views -v
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.result_views import (  # noqa: E402
    RUL_SCHEMA,
    TRAJECTORY_SCHEMA,
    UNAVAILABLE_MESSAGE,
    UNKNOWN_SCHEMA,
    build_rul_view,
    build_trajectory_view,
    detect_schema,
    prediction_table,
)
from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.artifact_browser import (  # noqa: E402
    UnsafePathError,
    list_artifacts,
    read_artifact,
    resolve_within,
)
from batteryml_web.core.reproducibility import (  # noqa: E402
    BUNDLE_ENVIRONMENT,
    BUNDLE_MANIFEST,
    BUNDLE_METRICS,
    BUNDLE_README,
    dataset_fingerprint,
    environment_snapshot,
    predictions_csv_rows,
    write_bundle,
)


class _SyntheticDataset:
    """Stand-in for batteryml's Dataset: only the attribute views read."""

    def __init__(self, label):
        self.label = label


class _SyntheticBundle:
    """Stand-in for DataBundle. `synthetic_` by policy: never a real dataset."""

    def __init__(self, label, label_transformation=None, cell_ids=None):
        self.test_data = _SyntheticDataset(label)
        self.label_transformation = label_transformation
        if cell_ids is not None:
            self.test_cell_ids = cell_ids


class _SyntheticDoublingTransform:
    """A label transformation whose inverse is unmistakable (halve -> double).

    Chosen so a view that forgot to invert is caught by the numbers rather
    than by a comment claiming it inverted.
    """

    @staticmethod
    def inverse_transform(values):
        return [v * 2 for v in values]


def synthetic_rul_payload(
    observed=(100.0, 200.0, 300.0),
    predicted=(110.0, 190.0, 330.0),
    *,
    transformation=None,
    cell_ids=None,
    with_labels=True,
):
    bundle = _SyntheticBundle(
        list(observed) if with_labels else None,
        label_transformation=transformation,
        cell_ids=cell_ids,
    )
    return {
        "prediction": list(predicted),
        "scores": {"RMSE": 1.0, "MAE": 2.0, "MAPE": 0.25},
        "data": bundle,
        "seed": 0,
    }


class TestSchemaDetection(unittest.TestCase):
    """B3: an RUL scalar is never re-read as a trajectory."""

    def test_a_scalar_per_sample_is_a_rul_result(self) -> None:
        self.assertEqual(detect_schema(synthetic_rul_payload()), RUL_SCHEMA)

    def test_a_sequence_per_sample_is_a_trajectory(self) -> None:
        payload = {"prediction": [[1.0, 0.9, 0.8], [1.0, 0.95, 0.9]]}
        self.assertEqual(detect_schema(payload), TRAJECTORY_SCHEMA)

    def test_nothing_recognisable_is_unknown_rather_than_guessed(self) -> None:
        for payload in (None, {}, {"prediction": None}, {"prediction": "text"}):
            self.assertEqual(detect_schema(payload), UNKNOWN_SCHEMA)

    def test_a_rul_result_is_refused_by_the_trajectory_view(self) -> None:
        view = build_trajectory_view(synthetic_rul_payload())
        self.assertFalse(view["available"])
        self.assertIn("not a degradation trajectory", view["reason"])


class TestObservedVersusPredicted(unittest.TestCase):
    """B2: the figure uses the arrays the run actually stored."""

    def test_the_view_carries_the_stored_arrays(self) -> None:
        view = build_rul_view(
            synthetic_rul_payload((100.0, 200.0), (110.0, 190.0))
        )
        self.assertTrue(view["available"])
        self.assertEqual(view["observed"], [100.0, 200.0])
        self.assertEqual(view["predicted"], [110.0, 190.0])
        self.assertEqual(view["count"], 2)

    def test_residuals_are_predicted_minus_observed_from_stored_values(self) -> None:
        view = build_rul_view(
            synthetic_rul_payload((100.0, 200.0, 300.0), (110.0, 190.0, 330.0))
        )
        self.assertEqual(view["residuals"], [10.0, -10.0, 30.0])

    def test_the_label_transformation_is_inverted_as_evaluation_does(self) -> None:
        """Otherwise the axes would be in a different space from the metrics.

        BatteryML's DataBundle.evaluate inverse-transforms both the labels
        and the prediction before scoring. A view that skipped this would
        plot log-space values beside a metric in cycles and say nothing.
        """
        view = build_rul_view(
            synthetic_rul_payload(
                (50.0, 100.0), (55.0, 95.0),
                transformation=_SyntheticDoublingTransform(),
            )
        )
        self.assertEqual(view["observed"], [100.0, 200.0])
        self.assertEqual(view["predicted"], [110.0, 190.0])
        self.assertTrue(view["inverse_transformed"])
        self.assertIn("same space as the reported metrics", view["note"])

    def test_a_run_without_ground_truth_gets_no_chart_and_no_fake_one(self) -> None:
        view = build_rul_view(synthetic_rul_payload(with_labels=False))
        self.assertFalse(view["available"])
        self.assertEqual(view["observed"], [])
        self.assertEqual(view["predicted"], [])
        self.assertEqual(view["residuals"], [])
        self.assertIn("no observed labels", view["reason"])

    def test_mismatched_lengths_are_refused_rather_than_zipped_short(self) -> None:
        view = build_rul_view(synthetic_rul_payload((1.0, 2.0, 3.0), (1.0, 2.0)))
        self.assertFalse(view["available"])
        self.assertIn("cannot be paired", view["reason"])

    def test_an_unreadable_payload_reports_unavailable(self) -> None:
        view = build_rul_view(None)
        self.assertFalse(view["available"])
        self.assertEqual(view["reason"], UNAVAILABLE_MESSAGE)

    def test_no_metric_is_computed_anywhere_in_the_views(self) -> None:
        """Residuals are a display of stored values, not a score."""
        import ast

        source = (
            REPO_ROOT / "batteryml_web" / "agent" / "result_views.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree)
        for forbidden in ("RMSE", "MAE", "MAPE", "sqrt", "** 0.5", "mean("):
            self.assertNotIn(forbidden, code)


class TestPredictionTable(unittest.TestCase):
    """B2: ids are used only where the artifact really has them."""

    def test_real_cell_ids_are_used_when_present(self) -> None:
        view = build_rul_view(
            synthetic_rul_payload((1.0, 2.0), (1.5, 2.5), cell_ids=["b1c0", "b1c1"])
        )
        rows = prediction_table(view)
        self.assertEqual([row["Cell"] for row in rows], ["b1c0", "b1c1"])

    def test_no_identifier_is_invented_when_the_artifact_has_none(self) -> None:
        view = build_rul_view(synthetic_rul_payload((1.0, 2.0), (1.5, 2.5)))
        rows = prediction_table(view)
        self.assertNotIn("Cell", rows[0])
        self.assertEqual(rows[0]["#"], 0)

    def test_the_table_values_are_the_view_values(self) -> None:
        view = build_rul_view(synthetic_rul_payload((10.0,), (12.0,)))
        row = prediction_table(view)[0]
        self.assertEqual((row["Observed"], row["Predicted"], row["Residual"]),
                         (10.0, 12.0, 2.0))

    def test_truncation_never_misreports_the_real_count(self) -> None:
        view = build_rul_view(
            synthetic_rul_payload(tuple(range(50)), tuple(range(50)))
        )
        self.assertEqual(len(prediction_table(view, limit=10)), 10)
        self.assertEqual(view["count"], 50)


class TestTrajectoryView(unittest.TestCase):
    def test_a_real_trajectory_schema_is_accepted(self) -> None:
        payload = {
            "prediction": [[1.0, 0.9, 0.8], [1.0, 0.95, 0.9]],
            "data": _SyntheticBundle([[1.0, 0.92, 0.82], [1.0, 0.96, 0.91]]),
        }
        view = build_trajectory_view(payload)
        self.assertTrue(view["available"])
        self.assertEqual(view["count"], 2)
        self.assertEqual(view["length"], 3)
        self.assertEqual(len(view["observed"]), 2)

    def test_a_trajectory_without_stored_truth_still_shows_predictions(self) -> None:
        payload = {"prediction": [[1.0, 0.9], [1.0, 0.8]]}
        view = build_trajectory_view(payload)
        self.assertTrue(view["available"])
        self.assertEqual(view["observed"], [])


class _RunDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "project" / "agent_matr_cnn" / "run_1"
        self.run_dir.mkdir(parents=True)
        manifest = rm.build_manifest(
            run_id="run_1", project="alpha", workspace=str(self.run_dir),
            dataset="MATR", task="RUL prediction", model_key="cnn",
            model_display="CNN", seed=0, device="cpu", git_commit="abc123",
        )
        manifest["status"] = rm.STATUS_COMPLETED
        manifest["metrics"] = {"RMSE": 716.48}
        rm.write_manifest(self.run_dir, manifest)
        self.manifest = manifest
        (self.run_dir / "stdout.log").write_text("training output\n", encoding="utf-8")
        (self.run_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        (self.run_dir / "predictions_seed_0.pkl").write_bytes(pickle.dumps({"x": 1}))
        (self.run_dir / "model.ckpt").write_bytes(b"0" * 1024)


class TestArtifactBrowserSafety(_RunDirCase):
    """B7: nothing outside the run directory can be reached."""

    def test_only_files_that_exist_are_listed(self) -> None:
        names = {entry["name"] for entry in list_artifacts(self.run_dir)}
        self.assertIn("stdout.log", names)
        self.assertIn("config.yaml", names)
        self.assertNotIn("report.html", names)

    def test_artifacts_are_categorised_from_what_is_there(self) -> None:
        by_name = {e["name"]: e["category"] for e in list_artifacts(self.run_dir)}
        self.assertEqual(by_name["stdout.log"], "logs")
        self.assertEqual(by_name["config.yaml"], "config")
        self.assertEqual(by_name["predictions_seed_0.pkl"], "predictions")
        self.assertEqual(by_name["model.ckpt"], "checkpoints")

    def test_parent_traversal_is_refused(self) -> None:
        secret = self.root / "secret.txt"
        secret.write_text("private", encoding="utf-8")
        for attempt in ("../../../secret.txt", "../secret.txt", "..", "../"):
            with self.subTest(attempt=attempt):
                with self.assertRaises(UnsafePathError):
                    resolve_within(self.run_dir, attempt)

    def test_an_absolute_path_outside_the_run_is_refused(self) -> None:
        with self.assertRaises(UnsafePathError):
            read_artifact(self.run_dir, "/etc/passwd")

    def test_a_sibling_sharing_a_name_prefix_is_refused(self) -> None:
        """startswith() would have allowed this; relative_to does not."""
        sibling = self.run_dir.parent / "run_1_evil"
        sibling.mkdir()
        (sibling / "loot.txt").write_text("nope", encoding="utf-8")
        with self.assertRaises(UnsafePathError):
            resolve_within(self.run_dir, str(sibling / "loot.txt"))

    def test_a_symlink_pointing_out_of_the_run_is_refused(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("private", encoding="utf-8")
        link = self.run_dir / "innocent.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlinks are not supported on this filesystem")
        with self.assertRaises(UnsafePathError):
            resolve_within(self.run_dir, "innocent.txt")
        # ...and it is not even offered in the listing.
        self.assertNotIn("innocent.txt", {e["name"] for e in list_artifacts(self.run_dir)})

    def test_a_legitimate_artifact_reads_normally(self) -> None:
        self.assertEqual(
            read_artifact(self.run_dir, "stdout.log"), b"training output\n"
        )


class TestReproducibilityBundle(_RunDirCase):
    """B6: what a reviewer gets, and what they must never get."""

    def _bundle(self, **kwargs) -> tuple[Path, dict]:
        target = self.root / "bundle.zip"
        summary = write_bundle(self.run_dir, target, manifest=self.manifest, **kwargs)
        return target, summary

    def test_the_bundle_carries_the_expected_metadata(self) -> None:
        target, _ = self._bundle()
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            self.assertIn(BUNDLE_README, names)
            self.assertIn(BUNDLE_MANIFEST, names)
            self.assertIn(BUNDLE_METRICS, names)
            self.assertIn(BUNDLE_ENVIRONMENT, names)
            self.assertIn("stdout.log", names)
            self.assertIn("config.yaml", names)
            metrics = json.loads(archive.read(BUNDLE_METRICS))
            manifest = json.loads(archive.read(BUNDLE_MANIFEST))
        # Metrics are the manifest's stored values, not recomputed.
        self.assertEqual(metrics, {"RMSE": 716.48})
        self.assertEqual(manifest["run_id"], "run_1")

    def test_the_bundle_contains_no_api_key_or_environment_secret(self) -> None:
        target, _ = self._bundle()
        with zipfile.ZipFile(target) as archive:
            blob = b"".join(archive.read(name) for name in archive.namelist())
        lowered = blob.lower()
        for secret in (b"api_key", b"llm_api_key", b"authorization", b"bearer",
                       b"http_proxy", b"password", b"secret_key"):
            self.assertNotIn(secret, lowered)

    def test_the_environment_snapshot_lists_no_environment_variables(self) -> None:
        snapshot = environment_snapshot()
        # The shape is an allow-list, not a dump: these keys and no others.
        self.assertEqual(
            set(snapshot),
            {"python_version", "python_implementation", "platform", "machine",
             "packages", "note"},
        )
        self.assertIsInstance(snapshot["packages"], dict)
        # Whatever is on this machine's environment must not be in here.
        import os

        for name, value in os.environ.items():
            if len(str(value)) < 8:
                continue  # too short to be a meaningful match
            self.assertNotIn(
                str(value), json.dumps(snapshot),
                f"the value of ${name} leaked into the environment snapshot",
            )
        import ast

        source = (
            REPO_ROOT / "batteryml_web" / "core" / "reproducibility.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]
        self.assertNotIn("os.environ", ast.unparse(tree))

    def test_checkpoints_are_excluded_but_their_location_is_stated(self) -> None:
        target, summary = self._bundle()
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            readme = archive.read(BUNDLE_README).decode("utf-8")
        self.assertNotIn("model.ckpt", names)
        self.assertTrue(any("model.ckpt" in item["path"] for item in summary["omitted"]))
        self.assertIn("model.ckpt", readme)

    def test_the_raw_dataset_is_never_bundled(self) -> None:
        raw = self.root / "data" / "raw" / "MATR"
        raw.mkdir(parents=True)
        (raw / "cell.csv").write_text("huge", encoding="utf-8")
        target, _ = self._bundle()
        with zipfile.ZipFile(target) as archive:
            self.assertFalse([n for n in archive.namelist() if "data/raw" in n])

    def test_a_bundle_for_one_run_cannot_contain_another_runs_files(self) -> None:
        other = self.run_dir.parent / "run_2"
        other.mkdir()
        (other / "other_run_output.log").write_text("not mine", encoding="utf-8")
        target, _ = self._bundle()
        with zipfile.ZipFile(target) as archive:
            blob = b"".join(archive.read(n) for n in archive.namelist())
            self.assertNotIn("other_run_output.log", archive.namelist())
        self.assertNotIn(b"not mine", blob)

    def test_an_oversized_file_is_referenced_rather_than_copied(self) -> None:
        (self.run_dir / "big.csv").write_bytes(b"0" * 4096)
        target, summary = self._bundle(max_bytes=1024)
        with zipfile.ZipFile(target) as archive:
            self.assertNotIn("big.csv", archive.namelist())
        self.assertTrue(any("big.csv" in item["path"] for item in summary["omitted"]))

    def test_predictions_csv_is_written_only_from_paired_values(self) -> None:
        view = build_rul_view(synthetic_rul_payload((1.0, 2.0), (1.5, 2.5)))
        csv = predictions_csv_rows(view)
        self.assertIn("observed,predicted,residual", csv)
        self.assertEqual(len(csv.strip().splitlines()), 3)
        # A run with no ground truth exports no CSV at all.
        self.assertIsNone(
            predictions_csv_rows(build_rul_view(synthetic_rul_payload(with_labels=False)))
        )


class TestDatasetFingerprint(unittest.TestCase):
    """B5: a fingerprint that is cheap and honest, or none."""

    def test_a_missing_directory_reports_itself_rather_than_faking_one(self) -> None:
        result = dataset_fingerprint("/does/not/exist")
        self.assertFalse(result["available"])
        self.assertIsNone(result["file_count"])
        self.assertIn("does not exist", result["reason"])

    def test_a_real_directory_is_summarised_without_hashing_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "a.pkl").write_bytes(b"12345")
            (directory / "b.pkl").write_bytes(b"678")
            result = dataset_fingerprint(directory)
        self.assertTrue(result["available"])
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["total_bytes"], 8)
        self.assertIn("not hashed", result["method"])


if __name__ == "__main__":
    unittest.main()
