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
import zipfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core.readiness import (  # noqa: E402
    BATTERYLIFE_RAW_PARTS,
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


def _registry_status_reason() -> str | None:
    """Why the live registries cannot be read here, or None when they can.

    The reason matters: "batteryml is not importable" and "batteryml imports
    but the extension models have no source checked out" are different
    problems with different fixes, and a skip message that names the wrong
    one sends the reader after the wrong thing.
    """
    try:
        from batteryml_web.core.config_validator import registry_status

        registries, error = registry_status()
        if error is not None:
            return str(error)
        if "MODELS" not in registries:
            return "the MODELS registry is absent from registry_status()."
        return None
    except Exception as exc:  # noqa: BLE001 - reported as the skip reason
        return f"{type(exc).__name__}: {exc}"


def _demo_data_reason() -> str | None:
    """Why the MATR demo cannot be exercised here, or None when it can.

    The tests behind this gate assert that a model is READY and that its
    plan is executable. Both require the processed demo dataset: without
    it every readiness report is ``NEEDS_DATA`` and every plan is blocked,
    so the assertions fail for a reason that has nothing to do with what
    they are testing.

    This used to be covered by accident. ``register_extension_models()``
    aborted on any checkout without ``third_party/BatteryLife``, so the
    registry gate below was closed and these tests skipped — in an
    environment that also happened to have no data. Registration no longer
    aborts for one un-vendored optional model, which is the point of that
    fix, so the data requirement has to be stated in its own right.
    """
    processed = REPO_ROOT / "data" / "processed" / "MATR"
    if not processed.is_dir():
        return (
            f"{processed.relative_to(REPO_ROOT).as_posix()} does not exist. "
            "These tests assert a runnable MATR demo; preprocess it first "
            "(see README.md §9)."
        )
    if not any(processed.glob("*.pkl")):
        return (
            f"{processed.relative_to(REPO_ROOT).as_posix()} holds no *.pkl "
            "cell file, so no model can be READY here."
        )
    return None


#: Why the MATR demo environment is unusable here, or None when it is usable.
#:
#: Two independent requirements, reported separately because they have
#: different fixes: the registries must be readable, and the processed demo
#: data must be on disk.
DEMO_ENVIRONMENT_SKIP_REASON = _registry_status_reason() or _demo_data_reason()
DEMO_ENVIRONMENT_AVAILABLE = DEMO_ENVIRONMENT_SKIP_REASON is None

#: What ``skipUnless`` prints, naming the actual blocker.
DEMO_ENVIRONMENT_SKIP_MESSAGE = (
    "the MATR demo cannot be exercised here: "
    f"{(DEMO_ENVIRONMENT_SKIP_REASON or '')[:300]}"
)


class _TempRoots:
    """Paired raw/processed roots on an isolated repository, removed on exit.

    Isolating ``raw_root`` and ``processed_root`` is not enough, and that
    gap made two of these tests read the developer's own disk.

    ``assess_dataset`` recognises three raw shapes. Two of them — the
    extracted ``data/raw/BatteryLife/v11/<NAME>/`` directory and the
    ``<NAME>.zip`` archive beside it — live at a **fixed repository path**
    and are deliberately *not* resolved against a caller-supplied
    ``raw_root``: see the comment in ``_raw_layouts``, where gating them on
    ``raw_root is None`` is recorded as a bug that made the detection vanish
    for any caller naming its own raw directory. That behaviour is correct
    and is left alone.

    The consequence for a test is that ``raw_root`` cannot reach those two
    shapes, and ``assess_dataset`` takes no repository-root argument — it
    calls ``readiness._repo_root()``. So a machine that genuinely holds
    ``data/raw/BatteryLife/v11/MICH`` answered ``RAW_ONLY`` for MICH, which
    is the *right* answer for that machine and the wrong one for a test that
    believed it had supplied an empty world. It passed here and failed on
    the server, which is the signature of a fixture, not of a defect.

    The repository root is therefore redirected at the same temporary base
    for the duration, so every shape — native, extracted and archived —
    resolves inside the fixture and nothing on the host is consulted.
    """

    def __enter__(self) -> tuple[Path, Path]:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root = base / "repo"
        self.raw = base / "raw"
        self.processed = base / "processed"
        self.root.mkdir()
        self.raw.mkdir()
        self.processed.mkdir()

        from batteryml_web.core import readiness as readiness_mod

        self._module = readiness_mod
        self._saved_repo_root = readiness_mod._repo_root
        readiness_mod._repo_root = lambda: self.root
        return self.raw, self.processed

    def batterylife_raw(self, dataset: str) -> Path:
        """The extracted BatteryLife v11 directory for ``dataset``, created.

        Lets a test exercise the fixed-repository-path shape on purpose,
        rather than inheriting it by accident from the host.
        """
        target = self.root.joinpath(*BATTERYLIFE_RAW_PARTS) / dataset
        target.mkdir(parents=True, exist_ok=True)
        return target

    def __exit__(self, *exc_info) -> None:
        self._module._repo_root = self._saved_repo_root
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

    def test_extracted_batterylife_raw_is_raw_only(self) -> None:
        """The other half of the contract, stated instead of inherited.

        MICH has no native BatteryML preprocessor; its raw data arrives as
        the BatteryLife v11 distribution at a fixed repository path. Before
        this, no test placed that data on purpose — the only machine where
        the shape was exercised was one that happened to have it on disk,
        where it silently contradicted ``test_missing_dataset``. Both
        answers are now asserted, from data this test puts there itself.
        """
        fixture = _TempRoots()
        with fixture as (raw, processed):
            (fixture.batterylife_raw("MICH") / "cell_0.pkl").write_bytes(b"x")

            report = assess_dataset("MICH", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.RAW_ONLY)
        self.assertTrue(report.raw_available)
        self.assertFalse(report.processed_available)

    def test_a_downloaded_batterylife_archive_is_raw_only(self) -> None:
        """The third recognised shape: downloaded but not yet extracted.

        ``_raw_layouts`` recognises the ``<NAME>.zip`` archive beside the
        extracted directory, and that branch had no test at all — a mutation
        deleting it passed the whole suite. It is the plainest reading of
        "downloaded recognised raw only", so it is asserted here.
        """
        fixture = _TempRoots()
        with fixture as (raw, processed):
            v11 = fixture.root.joinpath(*BATTERYLIFE_RAW_PARTS)
            v11.mkdir(parents=True, exist_ok=True)
            # A real archive, because the production signature is
            # `zipfile.is_zipfile`, not the `.zip` suffix. A first draft of
            # this test wrote plausible-looking bytes and reported MISSING —
            # the code being stricter than the test expected.
            with zipfile.ZipFile(v11 / "MICH.zip", "w") as archive:
                archive.writestr("MICH/cell_0.pkl", "x")

            report = assess_dataset("MICH", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.RAW_ONLY)
        self.assertTrue(report.raw_available)
        self.assertFalse(report.processed_available)

    def test_a_half_downloaded_archive_is_not_counted_as_present(self) -> None:
        """The strictness above, asserted rather than assumed.

        A file carrying the ``.zip`` name but not the data is exactly the
        state an interrupted download leaves behind. Reporting it as raw
        data would send the user to preprocess something unreadable.
        """
        fixture = _TempRoots()
        with fixture as (raw, processed):
            v11 = fixture.root.joinpath(*BATTERYLIFE_RAW_PARTS)
            v11.mkdir(parents=True, exist_ok=True)
            (v11 / "MICH.zip").write_bytes(b"PK\x03\x04truncated")

            report = assess_dataset("MICH", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.MISSING)
        self.assertFalse(report.raw_available)

    def test_a_supplied_raw_root_cannot_reach_the_batterylife_path(self) -> None:
        """Why the fixture redirects the repository root at all.

        ``raw_root`` names the *native* raw directory only. Putting the
        BatteryLife layout under it must change nothing, because that shape
        is resolved against the repository root by design. This pins the
        boundary the two failing tests fell through.
        """
        with _TempRoots() as (raw, processed):
            decoy = raw.joinpath(*BATTERYLIFE_RAW_PARTS) / "MICH"
            decoy.mkdir(parents=True)
            (decoy / "cell_0.pkl").write_bytes(b"x")

            report = assess_dataset("MICH", raw_root=raw, processed_root=processed)

        self.assertIs(report.status, DatasetStatus.MISSING)

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

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_cptransformer_ready(self) -> None:
        report = assess_registry_model("cptransformer", dataset="MATR", task="RUL prediction")

        self.assertTrue(report.registered)
        self.assertTrue(report.ready)
        self.assertIs(report.status, ModelStatus.READY)
        self.assertEqual(report.missing, ())
        self.assertTrue(any("configs/external/cptransformer" in p for p in report.config_paths))

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_batlinet_ready(self) -> None:
        report = assess_registry_model("batlinet", dataset="MATR", task="RUL prediction")

        self.assertTrue(report.registered)
        self.assertTrue(report.ready)
        self.assertIs(report.status, ModelStatus.READY)
        self.assertEqual(report.missing, ())

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_required_classes_registered(self) -> None:
        from batteryml_web.core.config_validator import registry_status

        registries, error = registry_status()
        self.assertIsNone(error)
        self.assertIn("CPTransformerRULPredictor", registries["MODELS"])
        self.assertIn("BatLiNetRULPredictor", registries["MODELS"])
        self.assertIn("BatteryLifeCycleCurveFeatureExtractor", registries["FEATURE_EXTRACTORS"])
        self.assertIn("BatLiNetFeatureExtractor", registries["FEATURE_EXTRACTORS"])

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
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

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_model_without_configs_is_unregistered(self) -> None:
        report = assess_registry_model("definitely_not_a_model_key")

        self.assertFalse(report.registered)
        self.assertFalse(report.ready)
        self.assertIs(report.status, ModelStatus.UNREGISTERED)
        self.assertTrue(report.missing)

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
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
        # The name says it: this case is "the checkout is here, the data is
        # not". Where third_party/BatteryMFormer is absent (it is gitignored,
        # so most checkouts do not have it) the readiness layer correctly
        # reports the source files and the prompt-embedding pickle as missing
        # too, and the assertion below does not apply. Skipping states that;
        # weakening the assertion would hide a real regression on a machine
        # that does have the checkout.
        if not (REPO_ROOT / "third_party" / "BatteryMFormer").is_dir():
            self.skipTest(
                "third_party/BatteryMFormer is not checked out here, so the "
                "'repo present, data missing' case cannot be observed."
            )
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

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
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

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_trajectory_task_reports_batterymformer_blocked(self) -> None:
        context = get_workflow_context("MATR", "Degradation trajectory forecasting")

        self.assertEqual(context["task"], "Degradation trajectory forecasting")
        self.assertEqual(context["available_models"], [])
        blocked = {item["model"]: item for item in context["blocked_models"]}
        self.assertIn("BatteryMFormer", blocked)
        self.assertEqual(blocked["BatteryMFormer"]["status"], "NEEDS_DATA")
        steps = " ".join(context["recommended_next_steps"])
        self.assertIn("trajectory", steps.lower())

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_raw_only_dataset_requests_preprocessing_first(self) -> None:
        context = get_workflow_context("CALCE", "RUL prediction")

        self.assertEqual(context["dataset"]["status"], "RAW_ONLY")
        steps = " ".join(context["recommended_next_steps"])
        self.assertIn("preprocess", steps.lower())

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_default_dataset_is_matr_and_task_prefixes_are_folded(self) -> None:
        context = get_workflow_context()
        self.assertEqual(context["dataset"]["dataset"], "MATR")
        self.assertEqual(get_workflow_context("MATR", "RUL")["task"], "RUL prediction")
        self.assertEqual(get_workflow_context("MATR", "soh")["task"], "SOH estimation")

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
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

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
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
