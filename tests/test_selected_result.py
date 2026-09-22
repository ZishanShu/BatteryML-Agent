"""The result on screen must be the run the user selected. All of it.

Reproduced before it was repaired, with two runs on disk carrying
deliberately different numbers:

    load_metrics(find_result_files(project, …)["metrics_files"])
        -> RMSE 0.111      (run A, the OLDER run)
    load_predictions(...)
        -> [1.0, 2.0, 3.0] (run A)

…whichever run was selected. ``find_result_files`` ``rglob``\\s the whole
project and sorts by path; a run id begins with a timestamp, so path order is
chronological and the ``[0]`` those readers take is the oldest run in the
project. Review Results built its Metrics, Summary and Logs tabs and **both**
LLM prompts from that, while its Predictions and Files tabs read the selected
run's manifest — one page showing two different runs under one heading.

That single mechanism produced three of the reported symptoms: selecting a
run changed nothing; predictions looked identical between runs because they
were the same file; and the explanation described another experiment.

These tests are behavioural — they build real run directories with real
manifests and real artifacts and assert on the values that come back, not on
the source text that produces them.

Run from the repository root::

    conda run -n batteryml python -m unittest tests.test_selected_result -v
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.prediction_diagnostics import (  # noqa: E402
    artifact_diagnostics,
    compare_boundaries,
    describe_series,
)
from batteryml_web.core import selected_result as sr  # noqa: E402


class _RunFixtures(unittest.TestCase):
    """Two runs of the *same* experiment, distinguishable only by identity."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="selected-result-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.project_root = self.tmp / "runs" / "web_projects" / "alpha"
        self.project = {"name": "alpha", "root": str(self.project_root)}

    def make_run(
        self,
        run_id: str,
        *,
        metrics: dict | None = None,
        predictions=(1.0, 2.0, 3.0),
        write_artifact: bool = True,
        record_artifact: bool = True,
        project_root: Path | None = None,
    ) -> dict:
        """One run directory with a manifest and its own artifact."""
        root = project_root or self.project_root
        run_dir = root / "agent_matr_ridge" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact = run_dir / "predictions_seed_0_x.pkl"
        if write_artifact:
            with artifact.open("wb") as handle:
                pickle.dump(
                    {
                        "prediction": list(predictions),
                        "scores": dict(metrics or {}),
                    },
                    handle,
                )
        manifest = {
            "run_id": run_id,
            "project": "alpha",
            "status": "COMPLETED",
            "run_dir": str(run_dir),
            "workspace": str(run_dir),
            # Identical experiment across runs on purpose: only the run id
            # and the artifact differ, which is exactly the case a plan
            # signature cannot tell apart.
            "dataset": "MATR",
            "task": "SOH estimation",
            "model_key": "ridge",
            "model_display": "Ridge Regression",
            "seed": 0,
            "metrics": dict(metrics or {}),
            "artifacts": {
                "prediction_files": [str(artifact)] if record_artifact else []
            },
        }
        (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


class TestSelectionSwitchesEverything(_RunFixtures):
    """Case 1 and 2: selecting B gives B, and B never collides with A."""

    def test_selecting_b_after_a_resolves_bs_own_artifact(self) -> None:
        a = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa",
                          metrics={"RMSE": 0.111}, predictions=(1.0, 2.0, 3.0))
        b = self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb",
                          metrics={"RMSE": 0.999}, predictions=(7.0, 8.0, 9.0))

        first = sr.resolve_selected_result(self.project, a, repo_root=self.tmp)
        second = sr.resolve_selected_result(self.project, b, repo_root=self.tmp)

        self.assertTrue(first.available)
        self.assertTrue(second.available)
        self.assertNotEqual(first.predictions_file, second.predictions_file)
        # Each identity names its own run, and its own file.
        self.assertIn(first.run_id, first.predictions_file)
        self.assertIn(second.run_id, second.predictions_file)

        # And the values that come back are each run's own.
        self.assertEqual(
            artifact_diagnostics(first.predictions_file)["prediction"]["sample"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            artifact_diagnostics(second.predictions_file)["prediction"]["sample"],
            [7.0, 8.0, 9.0],
        )

    def test_identical_experiments_have_different_identities(self) -> None:
        """Case 2: same model, config and seed; different runs.

        A signature built from dataset/model/config/seed is identical for
        both of these, which is why it is not what the identity is made of.
        """
        a = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa",
                          metrics={"RMSE": 0.111})
        b = self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb",
                          metrics={"RMSE": 0.999})
        for field in ("dataset", "task", "model_key", "seed"):
            self.assertEqual(a[field], b[field])

        key_a = sr.resolve_selected_result(
            self.project, a, repo_root=self.tmp
        ).identity_key()
        key_b = sr.resolve_selected_result(
            self.project, b, repo_root=self.tmp
        ).identity_key()
        self.assertNotEqual(key_a, key_b)

    def test_rewriting_an_artifact_changes_the_identity(self) -> None:
        """A changed artifact is never served from the old cache entry."""
        import os
        import time

        manifest = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa",
                                 predictions=(1.0, 2.0, 3.0))
        before = sr.resolve_selected_result(
            self.project, manifest, repo_root=self.tmp
        ).identity_key()
        target = Path(manifest["artifacts"]["prediction_files"][0])
        time.sleep(0.01)
        with target.open("wb") as handle:
            pickle.dump({"prediction": [4.0, 5.0, 6.0, 7.0]}, handle)
        os.utime(target, None)
        after = sr.resolve_selected_result(
            self.project, manifest, repo_root=self.tmp
        ).identity_key()
        self.assertNotEqual(before, after)


class TestNoSilentFallback(_RunFixtures):
    """Case 5: an unavailable result says so; it never borrows another's."""

    def test_a_run_with_no_recorded_or_discoverable_artifact_is_unavailable(self) -> None:
        self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa",
                      predictions=(1.0, 2.0, 3.0))
        b = self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb",
                          write_artifact=False, record_artifact=False)
        result = sr.resolve_selected_result(self.project, b, repo_root=self.tmp)
        self.assertFalse(result.available)
        self.assertIsNone(result.predictions_file)
        self.assertEqual(result.reason, sr.NO_ARTIFACT)
        # The other run exists and is readable; it is still not offered.
        self.assertNotIn("aaaaaa", result.reason)

    def test_a_recorded_artifact_that_is_missing_names_the_path(self) -> None:
        b = self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb",
                          write_artifact=False, record_artifact=True)
        result = sr.resolve_selected_result(self.project, b, repo_root=self.tmp)
        self.assertFalse(result.available)
        self.assertIn(sr.ARTIFACT_MISSING, result.reason)
        self.assertIn("bbbbbb", result.reason)

    def test_an_artifact_outside_the_run_directory_is_refused(self) -> None:
        """A manifest cannot point the page at another run's file."""
        a = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa",
                          predictions=(1.0, 2.0, 3.0))
        b = self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb")
        b["artifacts"]["prediction_files"] = a["artifacts"]["prediction_files"]
        result = sr.resolve_selected_result(self.project, b, repo_root=self.tmp)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, sr.OUTSIDE_RUN_DIR)

    def test_several_artifacts_require_an_explicit_choice(self) -> None:
        manifest = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa")
        run_dir = Path(manifest["run_dir"])
        extra = run_dir / "predictions_seed_1_y.pkl"
        with extra.open("wb") as handle:
            pickle.dump({"prediction": [9.0]}, handle)
        manifest["artifacts"]["prediction_files"].append(str(extra))

        ambiguous = sr.resolve_selected_result(
            self.project, manifest, repo_root=self.tmp
        )
        self.assertFalse(ambiguous.available)
        self.assertEqual(ambiguous.reason, sr.AMBIGUOUS_ARTIFACT)
        self.assertEqual(len(ambiguous.candidates), 2)

        chosen = sr.resolve_selected_result(
            self.project, manifest, chosen_artifact=str(extra), repo_root=self.tmp
        )
        self.assertTrue(chosen.available)
        self.assertEqual(chosen.predictions_file, str(extra))

    def test_a_choice_from_another_run_is_not_honoured(self) -> None:
        a = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa")
        b = self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb")
        result = sr.resolve_selected_result(
            self.project,
            b,
            chosen_artifact=a["artifacts"]["prediction_files"][0],
            repo_root=self.tmp,
        )
        # Falls back to B's own single artifact, never to A's.
        self.assertTrue(result.available)
        self.assertIn("bbbbbb", result.predictions_file)

    def test_a_legacy_run_is_discovered_inside_its_own_directory_only(self) -> None:
        self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa",
                      predictions=(1.0, 2.0, 3.0))
        b = self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb",
                          predictions=(7.0, 8.0, 9.0), record_artifact=False)
        result = sr.resolve_selected_result(self.project, b, repo_root=self.tmp)
        self.assertTrue(result.available)
        self.assertIn("bbbbbb", result.predictions_file)
        self.assertEqual(
            artifact_diagnostics(result.predictions_file)["prediction"]["sample"],
            [7.0, 8.0, 9.0],
        )

    def test_no_selection_is_not_a_result(self) -> None:
        result = sr.resolve_selected_result(self.project, None, repo_root=self.tmp)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, sr.NO_RUN_SELECTED)


class TestProjectIsolation(_RunFixtures):
    """Case 6: a selection never reaches across projects."""

    def test_a_result_identity_carries_its_project(self) -> None:
        other_root = self.tmp / "runs" / "web_projects" / "beta"
        mine = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa")
        theirs = self.make_run(
            "run_20260101_000000_ridge_matr_seed0_aaaaaa",
            project_root=other_root,
        )
        key_mine = sr.resolve_selected_result(
            self.project, mine, repo_root=self.tmp
        ).identity_key()
        key_theirs = sr.resolve_selected_result(
            {"name": "beta", "root": str(other_root)}, theirs, repo_root=self.tmp
        ).identity_key()
        # Same run id, different projects: never the same identity.
        self.assertNotEqual(key_mine, key_theirs)


class TestExplanationFollowsTheSelection(_RunFixtures):
    """Cases 3 and 4: the explanation belongs to one result or to none."""

    def setUp(self) -> None:
        super().setUp()
        self.a = sr.resolve_selected_result(
            self.project,
            self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa"),
            repo_root=self.tmp,
        )
        self.b = sr.resolve_selected_result(
            self.project,
            self.make_run("run_20260202_000000_ridge_matr_seed0_bbbbbb"),
            repo_root=self.tmp,
        )

    def test_an_explanation_is_shown_only_under_its_own_result(self) -> None:
        session = {
            "review_llm_output": "A's explanation",
            "review_llm_output_identity": self.a.identity_key(),
        }
        self.assertEqual(
            sr.cached_for(session, "review_llm_output",
                          "review_llm_output_identity", self.a),
            "A's explanation",
        )
        # Case 3/4: under B it is withheld, not relabelled.
        self.assertIsNone(
            sr.cached_for(session, "review_llm_output",
                          "review_llm_output_identity", self.b)
        )

    def test_a_late_arrival_for_a_cannot_appear_as_bs(self) -> None:
        """Case 4, in the order it actually happens.

        The request for A is in flight while the user selects B. It lands and
        writes its answer with A's identity; B's panel asks for its own and
        gets nothing.
        """
        session: dict = {}
        # ...user selects B...
        # ...A's slow reply lands, stamped with A's identity...
        session["review_llm_output"] = "A's explanation"
        session["review_llm_output_identity"] = self.a.identity_key()
        self.assertIsNone(
            sr.cached_for(session, "review_llm_output",
                          "review_llm_output_identity", self.b)
        )

    def test_an_unstamped_value_belongs_to_nothing(self) -> None:
        session = {"review_llm_output": "from who knows where"}
        self.assertIsNone(
            sr.cached_for(session, "review_llm_output",
                          "review_llm_output_identity", self.a)
        )


class TestPredictionDiagnostics(_RunFixtures):
    """Cases 7, 8 and 9: measure the values; never manufacture them."""

    def test_varying_predictions_are_reported_as_varying(self) -> None:
        manifest = self.make_run(
            "run_20260101_000000_ridge_matr_seed0_aaaaaa",
            predictions=(0.9312, 0.8124, 0.7731, 0.6002),
        )
        result = sr.resolve_selected_result(
            self.project, manifest, repo_root=self.tmp
        )
        diagnostics = artifact_diagnostics(
            result.predictions_file, run_id=result.run_id
        )
        prediction = diagnostics["prediction"]
        self.assertTrue(diagnostics["readable"])
        self.assertEqual(prediction["count"], 4)
        self.assertEqual(prediction["distinct"], 4)
        self.assertFalse(prediction["constant"])
        # Stored order preserved, full precision, before any formatting.
        self.assertEqual(prediction["sample"], [0.9312, 0.8124, 0.7731, 0.6002])
        self.assertEqual(prediction["min"], 0.6002)

    def test_constant_predictions_are_reported_not_repaired(self) -> None:
        """Case 8. The honest finding, preserved exactly as stored."""
        manifest = self.make_run(
            "run_20260101_000000_ridge_matr_seed0_aaaaaa",
            predictions=(0.5, 0.5, 0.5, 0.5),
        )
        result = sr.resolve_selected_result(
            self.project, manifest, repo_root=self.tmp
        )
        prediction = artifact_diagnostics(result.predictions_file)["prediction"]
        self.assertEqual(prediction["distinct"], 1)
        self.assertTrue(prediction["constant"])
        # No jitter, no spread, no substitution.
        self.assertEqual(prediction["sample"], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(prediction["min"], prediction["max"])

    def test_a_flattened_display_is_told_apart_from_a_constant_model(self) -> None:
        """The two findings that look identical on screen."""
        genuinely_constant = compare_boundaries([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.assertTrue(genuinely_constant["genuinely_constant"])
        self.assertFalse(genuinely_constant["flattened_by_display"])

        flattened = compare_boundaries([0.1, 0.2, 0.3], [0.1, 0.1, 0.1])
        self.assertTrue(flattened["flattened_by_display"])
        self.assertFalse(flattened["genuinely_constant"])
        self.assertFalse(flattened["aligned"])

    def test_values_that_differ_only_past_the_display_precision_are_distinct(self) -> None:
        """Case 9: rounding must not be mistaken for a constant model."""
        prediction = describe_series(
            [0.1428684138, 0.1428684139, 0.1428684140]
        )
        self.assertEqual(prediction["distinct"], 3)
        self.assertFalse(prediction["constant"])
        # The canonical values are untouched by the description of them.
        self.assertEqual(prediction["sample"][0], 0.1428684138)

    def test_a_scalar_broadcast_over_rows_is_visible(self) -> None:
        """One of the candidate causes, made detectable rather than assumed."""
        self.assertEqual(describe_series(0.5)["count"], 1)
        self.assertEqual(describe_series([[0.5], [0.5], [0.5]])["count"], 3)

    def test_an_unreadable_artifact_is_reported_not_replaced(self) -> None:
        manifest = self.make_run("run_20260101_000000_ridge_matr_seed0_aaaaaa")
        target = Path(manifest["artifacts"]["prediction_files"][0])
        target.write_bytes(b"not a pickle")
        diagnostics = artifact_diagnostics(str(target))
        self.assertFalse(diagnostics["readable"])
        self.assertIn("could not be read", diagnostics["reason"])
        self.assertIsNone(diagnostics["prediction"])

    def test_diagnostics_never_reorder_the_stored_series(self) -> None:
        """Sorting predictions away from their labels is a misalignment bug."""
        descending = describe_series([3.0, 1.0, 2.0])
        self.assertEqual(descending["sample"], [3.0, 1.0, 2.0])


class TestNoProjectWideSearchRemains(_RunFixtures):
    """The mechanism itself: a whole-project read must not come back.

    Behavioural rather than a source scan — it builds the two runs that
    reproduced the bug and asserts the resolver answers with the *selected*
    one, which is the property that was violated.
    """

    def test_resolution_ignores_every_other_run_in_the_project(self) -> None:
        # Five older runs, all readable, all with different values.
        for index in range(5):
            self.make_run(
                f"run_2026010{index}_000000_ridge_matr_seed0_old{index:03d}",
                predictions=(float(index), float(index), float(index)),
            )
        newest = self.make_run(
            "run_20261231_000000_ridge_matr_seed0_target",
            predictions=(42.0, 43.0, 44.0),
        )
        result = sr.resolve_selected_result(
            self.project, newest, repo_root=self.tmp
        )
        self.assertEqual(
            artifact_diagnostics(result.predictions_file)["prediction"]["sample"],
            [42.0, 43.0, 44.0],
        )
        # And selecting one of the older ones gives that one, not the newest.
        older = sr.resolve_selected_result(
            self.project,
            json.loads(
                (self.project_root / "agent_matr_ridge"
                 / "run_20260103_000000_ridge_matr_seed0_old003"
                 / "run.json").read_text(encoding="utf-8")
            ),
            repo_root=self.tmp,
        )
        self.assertEqual(
            artifact_diagnostics(older.predictions_file)["prediction"]["sample"],
            [3.0, 3.0, 3.0],
        )


if __name__ == "__main__":
    unittest.main()
