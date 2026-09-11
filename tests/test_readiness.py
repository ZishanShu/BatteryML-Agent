"""Focused tests for the backend readiness layer.

Every test is offline and cheap: temporary directories for the dataset and
external-model paths, no GPU, no training, no dataset downloads. The
registry-backed tests additionally exercise the live BatteryML registries the
way ``run_batteryml_cli.py`` populates them; they skip (rather than fail) in
an environment where ``batteryml`` cannot be imported, because a missing
dependency is an environment fact, not a defect in the readiness layer.

Run from the repository root:

    python -m unittest tests.test_readiness -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core.readiness import (  # noqa: E402
    DatasetStatus,
    EXTERNAL_MODELS,
    ModelStatus,
    PLATFORM_DATASETS,
    assess_dataset,
    assess_external_model,
    assess_model,
    assess_registry_model,
    get_workflow_context,
    list_dataset_readiness,
    list_model_readiness,
)


def _batteryml_registries_available() -> bool:
    """Whether the live registries can be read in this environment."""
    try:
        from batteryml_web.core.config_validator import registry_status

        registries, error = registry_status()
        return error is None and "MODELS" in registries
    except Exception:
        return False


REGISTRIES_AVAILABLE = _batteryml_registries_available()


class _TempRoots:
    """Create paired raw/processed roots that are removed on exit."""

    def __enter__(self) -> tuple[Path, Path]:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.raw = base / "raw"
        self.processed = base / "processed"
        self.raw.mkdir()
        self.processed.mkdir()
        return self.raw, self.processed

    def __exit__(self, *exc_info) -> None:
        self._tmp.cleanup()


class TestDatasetReadiness(unittest.TestCase):
    """The four dataset states, each on temporary paths only."""

    def test_ready_dataset(self) -> None:
        with _TempRoots() as (raw, processed):
            (raw / "MATR").mkdir()
            (raw / "MATR" / "MATR_batch_20170512.mat").write_bytes(b"x")
            (processed / "MATR").mkdir()
            (processed / "MATR" / "MATR_b1c0.pkl").write_bytes(b"x")
            (processed / "MATR" / "MATR_b1c1.pkl").write_bytes(b"x")

            report = assess_dataset("MATR", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.READY)
        self.assertTrue(report.raw_available)
        self.assertTrue(report.processed_available)
        self.assertEqual(report.processed_sample_count, 2)
        self.assertTrue(report.recommended_action)
        self.assertIn("2 usable", report.reason)

    def test_raw_only_dataset(self) -> None:
        with _TempRoots() as (raw, processed):
            (raw / "CALCE").mkdir()
            (raw / "CALCE" / "CS2_33.zip").write_bytes(b"x")

            report = assess_dataset("CALCE", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.RAW_ONLY)
        self.assertTrue(report.raw_available)
        self.assertFalse(report.processed_available)
        self.assertEqual(report.processed_sample_count, 0)
        self.assertIn("preprocessing", report.recommended_action.lower())

    def test_incomplete_processed_dataset(self) -> None:
        with _TempRoots() as (raw, processed):
            (raw / "SNL").mkdir()
            (raw / "SNL" / "cycle_data.csv").write_bytes(b"x")
            (processed / "SNL").mkdir()

            report = assess_dataset("SNL", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.INCOMPLETE_PROCESSED)
        self.assertEqual(report.processed_sample_count, 0)
        self.assertIn("empty", report.reason)

    def test_missing_dataset(self) -> None:
        with _TempRoots() as (raw, processed):
            report = assess_dataset("MICH", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.MISSING)
        self.assertFalse(report.raw_available)
        self.assertFalse(report.processed_available)

    def test_only_pkl_files_count_as_processed(self) -> None:
        with _TempRoots() as (raw, processed):
            (processed / "HNEI").mkdir()
            (processed / "HNEI" / "notes.txt").write_bytes(b"x")

            report = assess_dataset("HNEI", raw_root=raw, processed_root=processed)

        self.assertEqual(report.processed_sample_count, 0)
        self.assertIsNot(report.status, DatasetStatus.READY)

    def test_declared_datasets_reported_even_when_absent(self) -> None:
        with _TempRoots() as (raw, processed):
            reports = list_dataset_readiness(raw_root=raw, processed_root=processed)

        by_name = {r.dataset: r for r in reports}
        for declared in PLATFORM_DATASETS:
            self.assertIn(declared, by_name)
        self.assertIs(by_name["MICH"].status, DatasetStatus.MISSING)

    def test_empty_dataset_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assess_dataset("   ")


class TestRegistryModelReadiness(unittest.TestCase):
    """CPTransformer and BatLiNet must stay READY and registered (regression)."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_cptransformer_ready(self) -> None:
        report = assess_registry_model("cptransformer", dataset="MATR", task="RUL prediction")

        self.assertTrue(report.registered)
        self.assertTrue(report.ready)
        self.assertIs(report.status, ModelStatus.READY)
        self.assertEqual(report.missing, ())
        self.assertTrue(any("configs/external/cptransformer" in p for p in report.config_paths))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_batlinet_ready(self) -> None:
        report = assess_registry_model("batlinet", dataset="MATR", task="RUL prediction")

        self.assertTrue(report.registered)
        self.assertTrue(report.ready)
        self.assertIs(report.status, ModelStatus.READY)
        self.assertEqual(report.missing, ())

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_required_classes_registered(self) -> None:
        from batteryml_web.core.config_validator import registry_status

        registries, error = registry_status()
        self.assertIsNone(error)
        self.assertIn("CPTransformerRULPredictor", registries["MODELS"])
        self.assertIn("BatLiNetRULPredictor", registries["MODELS"])
        self.assertIn("BatteryLifeCycleCurveFeatureExtractor", registries["FEATURE_EXTRACTORS"])
        self.assertIn("BatLiNetFeatureExtractor", registries["FEATURE_EXTRACTORS"])

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_matr_external_configs_pass_validator(self) -> None:
        from batteryml_web.core.config_validator import load_and_validate_config_file

        for config in (
            REPO_ROOT / "configs/external/cptransformer/matr_1.yaml",
            REPO_ROOT / "configs/external/batlinet/matr_1.yaml",
        ):
            report = load_and_validate_config_file(config, require_data=True)
            self.assertTrue(
                report["valid"],
                f"{config} rejected by the execution gate: {report['errors']}",
            )

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_model_without_configs_is_unregistered(self) -> None:
        report = assess_registry_model("definitely_not_a_model_key")

        self.assertFalse(report.registered)
        self.assertFalse(report.ready)
        self.assertIs(report.status, ModelStatus.UNREGISTERED)
        self.assertTrue(report.missing)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_assess_model_resolves_aliases(self) -> None:
        for name, expected in (("XGBoost", "xgb"), ("CPTransformerRULPredictor", "cptransformer")):
            report = assess_model(name, dataset="MATR")
            self.assertEqual(report.model, expected)


class TestExternalModelReadiness(unittest.TestCase):
    """PBT and BatteryMFormer readiness on real paths and on synthetic trees."""

    def test_pbt_real_repo_missing_assets(self) -> None:
        report = assess_external_model("PBT", check_environment=False)

        self.assertTrue(report.registered)
        self.assertFalse(report.ready)
        self.assertIs(report.status, ModelStatus.NEEDS_ASSETS)
        self.assertTrue(any("checkpoint" in item for item in report.missing))
        self.assertTrue(any("PCA" in item for item in report.missing))
        self.assertEqual(report.task, "RUL prediction")

    def test_batterymformer_real_repo_missing_data(self) -> None:
        report = assess_external_model("BatteryMFormer", check_environment=False)

        self.assertTrue(report.registered)
        self.assertFalse(report.ready)
        self.assertIs(report.status, ModelStatus.NEEDS_DATA)
        self.assertTrue(any("processed_SOH" in item for item in report.missing))
        self.assertTrue(any("BatteryLife v11" in item for item in report.missing))
        self.assertFalse(any("Qwen3_total.pkl" in item for item in report.missing))
        self.assertEqual(report.task, "Degradation trajectory forecasting")

    def test_pbt_synthetic_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "third_party" / "PBT"
            (source / "models").mkdir(parents=True)
            for relative in ("models/PBT.py", "run_main.py", "finetune_model.py", "evaluate_model.py"):
                (source / relative).write_text("# source stub", encoding="utf-8")
            checkpoint = source / "pretrained_assets" / "checkpoints" / "PBTs" / "run_seed2021"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"x")
            (checkpoint / "args.json").write_text("{}", encoding="utf-8")
            (checkpoint / "label_scaler").write_bytes(b"x")
            (source / "pretrained_assets" / "condition_embeddings").mkdir(parents=True)
            for embedding in (
                "training_DKP_embed_all_Llama.pkl",
                "validation_DKP_embed_all_Llama.pkl",
                "testing_DKP_embed_all_Llama.pkl",
            ):
                (source / "pretrained_assets" / "condition_embeddings" / embedding).write_bytes(b"x")
            (source / "MIX_large_pca_0.99_Llama.pkl").write_bytes(b"x")
            (source / "dataset").mkdir()
            (source / "dataset" / "HUST_dataset").mkdir()
            (source / "dataset" / "HUST_dataset" / "data.csv").write_bytes(b"x")
            llm = root / "llm"
            llm.mkdir()
            (llm / "config.json").write_text("{}", encoding="utf-8")

            report = assess_external_model(
                "PBT",
                repo_root=root,
                config={"assets": {"LLM_path": str(llm)}},
                check_environment=False,
            )

        self.assertTrue(report.ready, report.missing)
        self.assertIs(report.status, ModelStatus.READY)
        self.assertEqual(report.missing, ())

    def test_pbt_synthetic_data_only_missing_is_needs_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "third_party" / "PBT"
            (source / "models").mkdir(parents=True)
            for relative in ("models/PBT.py", "run_main.py", "finetune_model.py", "evaluate_model.py"):
                (source / relative).write_text("# source stub", encoding="utf-8")
            checkpoint = source / "pretrained_assets" / "checkpoints" / "PBTs" / "run_seed2021"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"x")
            (checkpoint / "args.json").write_text("{}", encoding="utf-8")
            (checkpoint / "label_scaler").write_bytes(b"x")
            (source / "pretrained_assets" / "condition_embeddings").mkdir(parents=True)
            for embedding in (
                "training_DKP_embed_all_Llama.pkl",
                "validation_DKP_embed_all_Llama.pkl",
                "testing_DKP_embed_all_Llama.pkl",
            ):
                (source / "pretrained_assets" / "condition_embeddings" / embedding).write_bytes(b"x")
            (source / "MIX_large_pca_0.99_Llama.pkl").write_bytes(b"x")
            llm = root / "llm"
            llm.mkdir()
            (llm / "config.json").write_text("{}", encoding="utf-8")

            report = assess_external_model(
                "PBT",
                repo_root=root,
                config={"assets": {"LLM_path": str(llm)}},
                check_environment=False,
            )

        self.assertFalse(report.ready)
        self.assertIs(report.status, ModelStatus.NEEDS_DATA)
        self.assertEqual(len(report.missing), 1)
        self.assertIn("dataset", report.missing[0])

    def test_batterymformer_synthetic_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "third_party" / "BatteryMFormer"
            (source / "models").mkdir(parents=True)
            (source / "models" / "BatteryMFormer.py").write_text("# source stub", encoding="utf-8")
            (source / "run_main.py").write_text("# source stub", encoding="utf-8")
            prompts = source / "data_provider" / "prompt_embeddings"
            prompts.mkdir(parents=True)
            (prompts / "Qwen3_total.pkl").write_bytes(b"x")
            splits = source / "data_provider" / "split_json" / "pure_ood"
            splits.mkdir(parents=True)
            (splits / "Liion_split_seed2024.json").write_text("{}", encoding="utf-8")
            v11 = root / "data" / "BatteryLife" / "v11" / "CALCE"
            v11.mkdir(parents=True)
            (v11 / "CALCE_cell.csv").write_bytes(b"x")
            soh = root / "data" / "processed_SOH"
            soh.mkdir(parents=True)
            (soh / "cell.pkl").write_bytes(b"x")

            report = assess_external_model(
                "BatteryMFormer",
                repo_root=root,
                config={"root_path": str(v11), "processed_SOH_path": str(soh)},
                check_environment=False,
            )

        self.assertTrue(report.ready, report.missing)
        self.assertIs(report.status, ModelStatus.READY)

    def test_unknown_external_model_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assess_external_model("NotAModel", check_environment=False)


class TestWorkflowContext(unittest.TestCase):
    """The agent-facing context for the primary worked case and the trajectory task."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_matr_rul_context(self) -> None:
        context = get_workflow_context("MATR", "RUL prediction")

        self.assertEqual(context["dataset"]["status"], "READY")
        self.assertEqual(context["task"], "RUL prediction")
        available_keys = {item["model"] for item in context["available_models"]}
        self.assertIn("cptransformer", available_keys)
        self.assertIn("batlinet", available_keys)
        self.assertIn("xgb", available_keys)
        blocked = {item["model"]: item for item in context["blocked_models"]}
        self.assertIn("PBT", blocked)
        self.assertEqual(blocked["PBT"]["status"], "NEEDS_ASSETS")
        self.assertNotIn("BatteryMFormer", blocked)
        self.assertNotIn("BatteryMFormer", available_keys)
        self.assertTrue(context["recommended_next_steps"])

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_trajectory_task_reports_batterymformer_blocked(self) -> None:
        context = get_workflow_context("MATR", "Degradation trajectory forecasting")

        self.assertEqual(context["task"], "Degradation trajectory forecasting")
        self.assertEqual(context["available_models"], [])
        blocked = {item["model"]: item for item in context["blocked_models"]}
        self.assertIn("BatteryMFormer", blocked)
        self.assertEqual(blocked["BatteryMFormer"]["status"], "NEEDS_DATA")
        steps = " ".join(context["recommended_next_steps"])
        self.assertIn("trajectory", steps.lower())

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_raw_only_dataset_requests_preprocessing_first(self) -> None:
        context = get_workflow_context("CALCE", "RUL prediction")

        self.assertEqual(context["dataset"]["status"], "RAW_ONLY")
        steps = " ".join(context["recommended_next_steps"])
        self.assertIn("preprocess", steps.lower())

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_default_dataset_is_matr_and_task_prefixes_are_folded(self) -> None:
        context = get_workflow_context()
        self.assertEqual(context["dataset"]["dataset"], "MATR")
        self.assertEqual(get_workflow_context("MATR", "RUL")["task"], "RUL prediction")
        self.assertEqual(get_workflow_context("MATR", "soh")["task"], "SOH estimation")

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_missing_dataset_reports_missing_state(self) -> None:
        context = get_workflow_context("__READINESS_TEST_MISSING_DATASET__", "RUL prediction")

        self.assertEqual(context["dataset"]["status"], "MISSING")
        self.assertEqual(context["available_models"], [])
        self.assertTrue(context["recommended_next_steps"])

    def test_unsupported_task_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_workflow_context("MATR", "dream interpretation")


class TestFullModelList(unittest.TestCase):
    """list_model_readiness must never present a blocked model as runnable."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_every_model_has_consistent_status_and_action(self) -> None:
        reports = list_model_readiness()

        self.assertTrue(reports)
        for report in reports:
            self.assertEqual(report.ready, report.status is ModelStatus.READY)
            self.assertTrue(report.recommended_action)
            if report.status is ModelStatus.READY:
                self.assertEqual(report.missing, ())
            else:
                self.assertTrue(report.missing, f"{report.model} blocked without a reason")
        external = {report.model for report in reports}
        self.assertLessEqual(set(EXTERNAL_MODELS), external)


if __name__ == "__main__":
    unittest.main(verbosity=2)
