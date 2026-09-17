"""The first real MATR SOH config, checked as far as this container allows.

`configs/soh/ridge/matr.yaml` is the first SOH configuration pointed at a real
dataset. This container has no `data/`, so nothing here trains: what is
established is everything about the config that does not need the cells —
which task it declares, which annotator and extractor it selects, the sample
policy and reference convention it fixes, that its splitter partitions cells,
and that it is the only thing the catalog offers for SOH.

The parts that genuinely need the data — the label distribution, the sample
counts, the metrics — are the server's to establish, and
`docs/SOH_PROTOCOL.md` carries the commands.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_soh_matr_config -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "configs" / "soh" / "ridge" / "matr.yaml"
SPLIT_SOURCE = REPO_ROOT / "batteryml" / "train_test_split" / "MATR_split.py"

TASK_SOH = "SOH estimation"
TASK_RUL = "RUL prediction"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _splitter_ids(class_name: str) -> tuple[list[str], list[str]]:
    """The train/test cell IDs a MATR splitter hard-codes.

    Read from the source rather than by instantiating the splitter, which
    needs the processed cells on disk. The point is the *partition*, and the
    partition is written literally in the class.
    """
    tree = ast.parse(SPLIT_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        found: dict[str, list[str]] = {}
        for statement in ast.walk(node):
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in ("train_ids", "test_ids")
                and isinstance(statement.value, ast.List)
            ):
                found[statement.targets[0].id] = [
                    element.value for element in statement.value.elts
                ]
        train, test = found.get("train_ids", []), found.get("test_ids", [])
        # Upstream removes the b2c1 outlier from the primary test split.
        if class_name == "MATRPrimaryTestTrainTestSplitter":
            test = [cell for cell in test if cell != "b2c1"]
        return train, test
    raise AssertionError(f"{class_name} not found in {SPLIT_SOURCE}")


class TestTheConfigExistsAndDeclaresSOH(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(CONFIG_PATH.is_file(), f"{CONFIG_PATH} is missing")
        self.data = _config()

    def test_it_declares_the_task(self) -> None:
        self.assertEqual(self.data["task"], TASK_SOH)

    def test_it_uses_the_cycle_level_soh_annotator(self) -> None:
        self.assertEqual(self.data["label"]["name"], "CycleSOHLabelAnnotator")

    def test_it_uses_the_soh_feature_extractor(self) -> None:
        self.assertEqual(self.data["feature"]["name"], "SOHVoltageCurveFeatureExtractor")

    def test_the_reference_convention_is_explicit(self) -> None:
        self.assertEqual(
            self.data["label"]["reference_capacity_mode"],
            "first_valid_discharge_capacity",
        )

    def test_the_approved_sample_policy(self) -> None:
        self.assertEqual(self.data["label"]["min_cycle"], 10)
        self.assertEqual(self.data["label"]["cycle_stride"], 5)
        self.assertIsNone(self.data["label"]["max_cycle"])

    def test_the_sample_policy_is_not_duplicated_onto_the_feature(self) -> None:
        """The annotator owns the cycle list; the extractor is told about it.

        Putting min_cycle/cycle_stride under `feature:` would be keys the
        extractor does not accept, and would invite the two components to
        disagree about which cycles are sampled.
        """
        for key in ("min_cycle", "max_cycle", "cycle_stride"):
            self.assertNotIn(key, self.data["feature"])

    def test_the_model_is_ridge_and_not_a_dummy(self) -> None:
        self.assertEqual(self.data["model"]["name"], "RidgeRULPredictor")
        self.assertNotIn("Dummy", self.data["model"]["name"])

    def test_the_label_transformation_does_not_stack_a_log(self) -> None:
        """A log of a ratio near 1.0 compresses the range SOH lives in."""
        transformation = self.data["label_transformation"]
        self.assertEqual(transformation["name"], "ZScoreDataTransformation")
        self.assertNotIn("LogScale", yaml.safe_dump(transformation))

    def test_it_points_at_the_real_processed_dataset(self) -> None:
        self.assertEqual(
            self.data["train_test_split"]["cell_data_path"], "data/processed/MATR"
        )

    def test_it_carries_no_rul_label(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("RULLabelAnnotator", text)
        self.assertNotIn("eol_soh", text)


class TestTheSplitIsCellLevel(unittest.TestCase):
    """The hard invariant, for the splitter this config actually names."""

    def setUp(self) -> None:
        self.splitter = _config()["train_test_split"]["name"]

    def test_the_config_names_a_matr_cell_level_splitter(self) -> None:
        self.assertIn(self.splitter, (
            "MATRPrimaryTestTrainTestSplitter",
            "MATRSecondaryTestTrainTestSplitter",
            "MATRCLOTestTrainTestSplitter",
        ))

    def test_train_and_test_cells_are_disjoint(self) -> None:
        train, test = _splitter_ids(self.splitter)
        self.assertTrue(train and test)
        self.assertEqual(
            set(train) & set(test), set(),
            "the split shares a cell between train and test",
        )

    def test_the_partition_has_no_duplicates(self) -> None:
        train, test = _splitter_ids(self.splitter)
        self.assertEqual(len(train), len(set(train)))
        self.assertEqual(len(test), len(set(test)))

    def test_the_configured_split_sizes_are_what_the_report_claims(self) -> None:
        train, test = _splitter_ids("MATRPrimaryTestTrainTestSplitter")
        self.assertEqual(len(train), 41)
        self.assertEqual(len(test), 42)

    def test_every_matr_splitter_partitions_cells(self) -> None:
        """Whichever of the three is chosen, it must not share a cell."""
        for name in (
            "MATRPrimaryTestTrainTestSplitter",
            "MATRSecondaryTestTrainTestSplitter",
            "MATRCLOTestTrainTestSplitter",
        ):
            with self.subTest(splitter=name):
                train, test = _splitter_ids(name)
                self.assertEqual(set(train) & set(test), set())

    def test_the_splitter_partitions_cells_not_rows(self) -> None:
        """Source-level: it returns file lists, so expansion happens after."""
        source = SPLIT_SOURCE.read_text(encoding="utf-8")
        self.assertIn("return self.train_cells, self.test_cells", source)
        self.assertNotIn("cycle", source.lower().split("def split")[1][:400])


class TestTheCatalogOffersItAndOnlyIt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from batteryml_web.core.config_registry import get_config_registry

        cls.entries = list(get_config_registry(str(REPO_ROOT)).entries)
        cls.soh = [entry for entry in cls.entries if entry.task == TASK_SOH]

    @staticmethod
    def _relative(entry) -> str:
        path = Path(entry.config_path)
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    def test_the_real_config_is_registered_as_soh(self) -> None:
        paths = [self._relative(entry) for entry in self.soh]
        self.assertIn("configs/soh/ridge/matr.yaml", paths)

    def test_it_is_matr_and_ridge_regression(self) -> None:
        entry = next(
            e for e in self.soh
            if self._relative(e) == "configs/soh/ridge/matr.yaml"
        )
        self.assertEqual(entry.dataset, "MATR")
        self.assertEqual(entry.task, TASK_SOH)
        self.assertEqual(entry.model_key, "ridge")
        self.assertEqual(entry.model_display, "Ridge Regression")

    def test_no_rul_model_is_offered_for_soh(self) -> None:
        """The catalog must not dump every RUL model into the SOH task."""
        for entry in self.soh:
            data = yaml.safe_load(Path(entry.config_path).read_text(encoding="utf-8"))
            self.assertIn("SOH", str(data.get("label", {}).get("name", "")))

    def test_the_dummy_configs_stay_out(self) -> None:
        offered = [
            self._relative(entry) for entry in self.entries
            if "configs/soh/dummy/" in self._relative(entry)
        ]
        self.assertEqual(offered, [])

    def test_the_rul_catalog_is_untouched(self) -> None:
        rul = [entry for entry in self.entries if entry.task == TASK_RUL]
        self.assertGreater(len(rul), 100)
        for entry in rul:
            text = Path(entry.config_path).read_text(encoding="utf-8")
            if "eol_soh" in text:
                self.assertEqual(entry.task, TASK_RUL)


class TestTheAgentOffersOnlySOHModelsForSOH(unittest.TestCase):
    """The catalog must not dump every RUL model into the SOH task."""

    @staticmethod
    def _configured_models(task: str) -> list[str]:
        """Models the backend has a config for, for MATR and this task.

        Status is not asserted: in this container the data is absent and the
        optional CPTransformer extension cannot be imported, so every model is
        blocked for *both* tasks. What is asserted is which models the backend
        has a validated config for, which is what decides the menu.
        """
        from batteryml_web.agent.prompts import build_agent_workflow_context

        context = build_agent_workflow_context("MATR", task)
        entries = list(context.get("runnable_models", [])) + list(
            context.get("blocked_models", [])
        )
        return sorted(
            str(entry.get("model"))
            for entry in entries
            if entry.get("status") != "UNREGISTERED"
        )

    def test_soh_offers_exactly_the_audited_baselines(self) -> None:
        """The five estimators that passed the compatibility audit, and no more.

        Each one is a scalar regressor reached through ``SklearnModel``, whose
        ``fit`` is ``model.fit(feature.view(n, -1), label)`` — no lifetime,
        end-of-life or cycle semantics anywhere — and each has a config under
        ``configs/soh/`` declaring the identical protocol. Compatible
        baselines, not a ranking: no claim is made here about which estimates
        SOH better. See docs/SOH_PROTOCOL.md.
        """
        self.assertEqual(
            self._configured_models(TASK_SOH),
            ["elastic_net", "lasso", "rf", "ridge", "xgb"],
        )

    def test_the_soh_menu_is_exactly_what_configs_soh_declares(self) -> None:
        """Backend-authoritative: the menu is the configs, not a UI list.

        Adding a directory under ``configs/soh/`` is the only way a model can
        reach the SOH menu, and it is sufficient — nothing enumerates model
        names in the frontend.
        """
        from batteryml_web.core.config_registry import get_config_registry

        registry = get_config_registry(str(REPO_ROOT))
        from_configs = sorted(
            {
                entry.model_key
                for entry in registry.get_entries(dataset="MATR", task=TASK_SOH)
            }
        )
        self.assertEqual(self._configured_models(TASK_SOH), from_configs)

    def test_no_rul_only_model_appears_under_soh(self) -> None:
        """The models that carry RUL-specific assumptions stay out.

        ``xgb`` and ``rf`` are deliberately absent from this list now: they are
        task-agnostic regressors that passed the audit and have their own SOH
        configs, so they appear under both tasks — which is correct. The models
        below are the ones that must not: sequence and inter-cell architectures
        built around early-cycle lifetime prediction, and the dummy baseline.
        """
        soh = set(self._configured_models(TASK_SOH))
        for rul_only in (
            "cnn", "lstm", "transformer", "batlinet", "cptransformer",
            "dummy", "svm", "gpr", "plsr", "pcr", "knn", "huber",
            "kernel_ridge", "extra_trees", "linear_regression",
            "discharge_model", "full_model", "variance_model",
        ):
            with self.subTest(model=rul_only):
                self.assertNotIn(rul_only, soh)

    def test_mlp_is_not_offered_for_soh(self) -> None:
        """Audited and rejected, for a reason that is about the features.

        ``MLPRULPredictor`` is an ``NNModel`` whose ``forward`` unpacks
        ``B, _, H, W = feature.size()``, so it requires a 3- or 4-dimensional
        feature. An SOH row is the flat 37-wide voltage-curve vector, and
        ``fit`` raises ``ValueError: not enough values to unpack (expected 4,
        got 2)`` on it. Reshaping the SOH feature to satisfy the model would be
        changing the science to fit the UI, so MLP has no SOH config.
        """
        self.assertNotIn("mlp", set(self._configured_models(TASK_SOH)))
        self.assertFalse(
            (REPO_ROOT / "configs" / "soh" / "mlp").exists(),
            "an SOH config for MLP exists; the audit rejected it",
        )

    def test_the_rul_menu_is_unchanged(self) -> None:
        rul = self._configured_models(TASK_RUL)
        self.assertGreater(len(rul), 20)
        self.assertIn("ridge", rul)
        self.assertIn("xgb", rul)

    def test_the_agent_resolves_the_real_soh_requests(self) -> None:
        from batteryml_web.agent.intent import resolve_workflow_intent

        for request in (
            "Estimate SOH on MATR.",
            "Use Ridge Regression on MATR to estimate SOH.",
            "What is the battery health on MATR?",
            "帮我用 MATR 估计 SOH。",
        ):
            with self.subTest(request=request):
                resolved = resolve_workflow_intent(request)
                self.assertEqual(resolved.get("dataset"), "MATR")
                self.assertEqual(resolved.get("task"), TASK_SOH)

    def test_lifetime_on_matr_is_still_rul(self) -> None:
        from batteryml_web.agent.intent import resolve_workflow_intent

        resolved = resolve_workflow_intent("Predict battery lifetime on MATR.")
        self.assertEqual(resolved.get("dataset"), "MATR")
        self.assertEqual(resolved.get("task"), TASK_RUL)


class TestTheConfigValidates(unittest.TestCase):
    """As far as it can without the cells on disk."""

    def test_nothing_about_this_config_is_invalid(self) -> None:
        """No error attributable to the config itself.

        The validator also reports that the optional CPTransformer extension
        cannot be imported, because ``third_party/`` is absent in this
        container — the same environmental cause as the pre-existing EXT-01
        and VAL-01 verify failures. That error is about the environment, not
        about this config, so it is excluded by name rather than by lowering
        the assertion to "some errors are acceptable".
        """
        from batteryml_web.core.config_validator import load_and_validate_config_file

        report = load_and_validate_config_file(CONFIG_PATH, require_data=False)
        errors = [
            str(error) for error in report.get("errors", [])
            if "extension models could not be registered" not in str(error)
        ]
        self.assertEqual(
            errors, [],
            f"the config itself is invalid: {errors}",
        )

    def test_its_registered_components_all_exist(self) -> None:
        from batteryml.builders import (
            FEATURE_EXTRACTORS,
            LABEL_ANNOTATORS,
            MODELS,
            TRAIN_TEST_SPLITTERS,
        )
        import batteryml.feature  # noqa: F401  (registers the extractors)
        import batteryml.label  # noqa: F401
        import batteryml.models  # noqa: F401
        import batteryml.train_test_split  # noqa: F401

        data = _config()
        self.assertIsNotNone(MODELS.build(dict(data["model"])))
        self.assertIsNotNone(
            LABEL_ANNOTATORS.build(dict(data["label"]))
        )
        self.assertIsNotNone(
            FEATURE_EXTRACTORS.build(dict(data["feature"]))
        )
        self.assertIn(
            data["train_test_split"]["name"].lower(),
            {key.lower() for key in TRAIN_TEST_SPLITTERS.class_mapping},
        )

    def test_the_built_annotator_carries_the_configured_policy(self) -> None:
        from batteryml.builders import LABEL_ANNOTATORS
        import batteryml.label  # noqa: F401

        annotator = LABEL_ANNOTATORS.build(dict(_config()["label"]))
        policy = annotator.sample_policy()
        self.assertEqual(policy["reference_capacity_mode"], "first_valid_discharge_capacity")
        self.assertEqual(policy["min_cycle"], 10)
        self.assertEqual(policy["cycle_stride"], 5)

    def test_the_declared_task_reaches_the_result_metadata(self) -> None:
        """What a result artifact will say about which question it answered."""
        from batteryml.pipeline import _declared_task, _task_metadata

        self.assertEqual(_declared_task(CONFIG_PATH), TASK_SOH)
        data = dict(_config())
        metadata = _task_metadata(data)
        self.assertEqual(metadata["task"], TASK_SOH)
        self.assertEqual(metadata["label_annotator"], "CycleSOHLabelAnnotator")
        self.assertEqual(
            metadata["reference_capacity_mode"], "first_valid_discharge_capacity"
        )
        self.assertEqual(metadata["min_cycle"], 10)
        self.assertEqual(metadata["cycle_stride"], 5)
        self.assertEqual(metadata["feature_extractor"], "SOHVoltageCurveFeatureExtractor")


class TestTheSOHRunAsksForTheRightMetrics(unittest.TestCase):
    def test_mae_and_rmse_only(self) -> None:
        from batteryml_web.agent.execution import metrics_for_task

        self.assertEqual(set(metrics_for_task(TASK_SOH)), {"MAE", "RMSE"})

    def test_mape_is_not_inherited(self) -> None:
        from batteryml_web.agent.execution import metrics_for_task

        self.assertNotIn("MAPE", metrics_for_task(TASK_SOH))
        self.assertIn("MAPE", metrics_for_task(TASK_RUL))


class TestTheAuditCommandsExist(unittest.TestCase):
    """The server's two inspection commands, importable and wired."""

    def test_the_label_audit_refuses_a_missing_dataset(self) -> None:
        from scripts.audit_soh_labels import main

        with self.assertRaises(SystemExit) as ctx:
            main(["--dataset", "NOPE", "--processed-root", "/nonexistent"])
        self.assertIn("does not exist", str(ctx.exception))

    def test_the_label_audit_never_clips(self) -> None:
        import scripts.audit_soh_labels as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        code = source.split('"""', 2)[2]
        for forbidden in ("clip(", "np.clip", "min(1.0", "max(0.0"):
            self.assertNotIn(forbidden, code)
        self.assertIn("never clipped", source)

    def test_the_artifact_inspector_refuses_a_missing_run(self) -> None:
        from scripts.inspect_soh_artifact import find_artifact

        with self.assertRaises(SystemExit) as ctx:
            find_artifact(Path("/nonexistent/run"))
        self.assertIn("does not exist", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
