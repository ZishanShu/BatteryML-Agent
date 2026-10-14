"""The SOH data gate: computed on request, cached, and never shown stale.

Uses the synthetic processed cells under ``tests/fixtures`` (never picked up
by the dataset registry) in a temporary repository root.

    python -m unittest tests.test_soh_gate -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import soh_gate  # noqa: E402
from batteryml_web.core.config_registry import get_config_registry  # noqa: E402
from batteryml_web.core.soh_readiness import SOH_NOT_ELIGIBLE, SOH_READY  # noqa: E402
from tests.fixtures.synthetic_soh_cells import cohort  # noqa: E402

SOH_MODELS = ("elastic_net", "lasso", "rf", "ridge", "xgb")


class _Root(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.policy = soh_gate.policy_from_config(REPO_ROOT / "configs" / "soh" / "ridge" / "hust.yaml")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cells(self, dataset: str, cells) -> Path:
        directory = self.root / "data" / "processed" / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            cell.dump(str(directory / f"{cell.cell_id}.pkl"))
        return directory


class TestPolicy(_Root):
    def test_the_policy_is_read_from_the_config_label(self) -> None:
        self.assertEqual(self.policy.min_cycle, 10)
        self.assertEqual(self.policy.cycle_stride, 5)
        self.assertIsNone(self.policy.max_cycle)
        self.assertEqual(self.policy.reference_capacity_mode, "first_valid_discharge_capacity")

    def test_a_config_without_a_label_is_refused_with_its_path(self) -> None:
        bad = self.root / "bad.yaml"
        bad.write_text("model: {name: x}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bad.yaml has no `label:` section"):
            soh_gate.policy_from_config(bad)


class TestGate(_Root):
    def test_unchecked_then_checked_then_cached(self) -> None:
        self._cells("SYNTH", cohort(n_cells=8, n_cycles=200))
        self.assertIsNone(soh_gate.cached_verdict(self.root, "SYNTH", self.policy))
        allowed, message = soh_gate.gate(None)
        self.assertIsNone(allowed)
        self.assertIn("Check SOH data", message)

        stored = soh_gate.check_and_store(self.root, "SYNTH", self.policy)
        self.assertEqual(stored["status"], SOH_READY)
        cached = soh_gate.cached_verdict(self.root, "SYNTH", self.policy)
        self.assertEqual(cached["status"], SOH_READY)
        allowed, message = soh_gate.gate(cached)
        self.assertTrue(allowed)
        self.assertIn("8 of 8 cell(s) eligible", message)

    def test_changed_processed_data_makes_the_verdict_stale(self) -> None:
        folder = self._cells("SYNTH", cohort(n_cells=8, n_cycles=200))
        soh_gate.check_and_store(self.root, "SYNTH", self.policy)
        newest = max(p.stat().st_mtime for p in folder.glob("*.pkl"))
        extra = next(folder.glob("*.pkl"))
        os.utime(extra, (newest + 10, newest + 10))
        self.assertIsNone(soh_gate.cached_verdict(self.root, "SYNTH", self.policy))

    def test_a_different_policy_is_a_different_question(self) -> None:
        self._cells("SYNTH", cohort(n_cells=8, n_cycles=200))
        soh_gate.check_and_store(self.root, "SYNTH", self.policy)
        other = soh_gate.SOHPolicy(self.policy.reference_capacity_mode, 1, None, 1)
        self.assertIsNone(soh_gate.cached_verdict(self.root, "SYNTH", other))

    def test_data_that_cannot_carry_soh_blocks_the_run(self) -> None:
        cells = cohort(n_cells=6, n_cycles=200)
        for cell in cells:
            for cycle in cell.cycle_data:
                cycle.discharge_capacity_in_Ah = None
        self._cells("SYNTH", cells)
        stored = soh_gate.check_and_store(self.root, "SYNTH", self.policy)
        self.assertEqual(stored["status"], SOH_NOT_ELIGIBLE)
        allowed, message = soh_gate.gate(stored)
        self.assertIs(allowed, False)
        self.assertIn("did not pass", message)

    def test_missing_processed_data_is_not_ready(self) -> None:
        stored = soh_gate.check_and_store(self.root, "NOTHING", self.policy)
        self.assertIs(soh_gate.gate(stored)[0], False)

    def test_a_corrupt_cache_file_counts_as_unchecked(self) -> None:
        self._cells("SYNTH", cohort(n_cells=6, n_cycles=200))
        soh_gate.check_and_store(self.root, "SYNTH", self.policy)
        soh_gate.cache_file(self.root, "SYNTH", self.policy).write_text("{not json", encoding="utf-8")
        self.assertIsNone(soh_gate.cached_verdict(self.root, "SYNTH", self.policy))


class TestHUSTConfigs(unittest.TestCase):
    def test_hust_offers_the_same_soh_models_as_matr(self) -> None:
        registry = get_config_registry(str(REPO_ROOT))
        for dataset in ("MATR", "HUST"):
            with self.subTest(dataset=dataset):
                models = sorted(e.model_key for e in registry.get_entries(dataset, "SOH estimation"))
                self.assertEqual(models, sorted(SOH_MODELS))

    def test_hust_configs_differ_from_matr_only_in_split_and_path(self) -> None:
        for model in SOH_MODELS:
            with self.subTest(model=model):
                matr = (REPO_ROOT / "configs" / "soh" / model / "matr.yaml").read_text(encoding="utf-8")
                hust = (REPO_ROOT / "configs" / "soh" / model / "hust.yaml").read_text(encoding="utf-8")
                body = lambda text: [
                    line for line in text[text.index("task: 'SOH estimation'"):].splitlines()
                    if not line.strip().startswith("#")
                ]
                expected = [
                    line.replace("MATRPrimaryTestTrainTestSplitter", "HUSTTrainTestSplitter")
                        .replace("data/processed/MATR", "data/processed/HUST")
                    for line in body(matr)
                ]
                self.assertEqual(body(hust), expected)


if __name__ == "__main__":
    unittest.main()
