"""Tests for the user-facing meaning extracted from BatteryML configs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core.config_registry import ConfigEntry  # noqa: E402
from batteryml_web.core.config_semantics import (  # noqa: E402
    infer_config_semantics,
    load_config_semantics,
    model_display_name,
)


class ConfigSemanticsTests(unittest.TestCase):
    def test_yaml_class_names_are_translated_to_user_labels(self) -> None:
        semantics = infer_config_semantics(
            {
                "train_test_split": {"cell_data_path": "data/processed/MATR"},
                "label": {"name": "RULLabelAnnotator", "eol_soh": 0.9},
                "model": {"name": "XGBoostRULPredictor"},
            }
        )
        self.assertEqual(semantics.dataset, "MATR")
        self.assertEqual(semantics.task, "RUL prediction")
        self.assertEqual(semantics.model_key, "xgb")
        self.assertEqual(semantics.model_display, "XGBoost — Extreme Gradient Boosting")

    def test_entry_display_is_authoritative_when_yaml_is_sparse(self) -> None:
        entry = ConfigEntry(
            dataset="MATR",
            task="RUL prediction",
            model_key="xgb",
            model_display="XGBoost — Extreme Gradient Boosting",
            split_preset="matr_1",
            split_label="Official BatteryML split 1",
            config_path=Path("matr_1.yaml"),
        )
        semantics = infer_config_semantics({}, entry=entry)
        self.assertEqual(semantics.to_dict()["dataset"], "MATR")
        self.assertEqual(semantics.task_label, "RUL prediction")
        self.assertEqual(semantics.model_label, entry.model_display)
        self.assertEqual(model_display_name(entry), entry.model_display)

    def test_loader_handles_uploaded_yaml_and_unreadable_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matr_1.yaml"
            path.write_text(
                "train_test_split:\n  cell_data_path: data\\processed\\MATR\n"
                "task: RUL\nmodel:\n  name: xgboost\n",
                encoding="utf-8",
            )
            loaded = load_config_semantics(path)
            self.assertEqual(loaded.dataset, "MATR")
            self.assertEqual(loaded.task, "RUL prediction")
            self.assertEqual(loaded.model_key, "xgb")

            missing = load_config_semantics(Path(tmp) / "missing.yaml")
            self.assertIsNone(missing.dataset)
            self.assertEqual(missing.model_label, "Unknown model")

    def test_path_fallback_does_not_call_a_model_a_dataset(self) -> None:
        semantics = load_config_semantics(Path("configs/baselines/nn_models/cnn/cnn.yaml"))
        self.assertIsNone(semantics.dataset)


if __name__ == "__main__":
    unittest.main()
