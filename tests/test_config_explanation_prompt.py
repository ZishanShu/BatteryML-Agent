"""Run Model "Explain this config": the prompt carries the facts, not just a path.

It used to send only the config path; from ``configs/external/batlinet/matr_1.yaml``
the assistant called the dataset "BatLiNet" and the model "MATR" and invented
``data:`` / ``train:`` / ``eval:`` sections. These tests pin what the prompt now
carries. Pure: no Streamlit, no provider.

    python -m unittest tests.test_config_explanation_prompt -v
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent import config_explanation as ce  # noqa: E402

BATLINET_MATR = REPO_ROOT / "configs" / "external" / "batlinet" / "matr_1.yaml"
APP = REPO_ROOT / "batteryml_web" / "app.py"


def _prompt(**overrides) -> str:
    args = dict(
        dataset="MATR",
        task="RUL prediction",
        model_key="batlinet",
        model_display="BatLiNet",
        config_path="configs/external/batlinet/matr_1.yaml",
        config_text=BATLINET_MATR.read_text(encoding="utf-8"),
    )
    args.update(overrides)
    return ce.build_config_explanation_prompt(**args)


class TestConfigExplanationPrompt(unittest.TestCase):
    def test_dataset_and_model_are_stated_as_facts(self) -> None:
        prompt = _prompt()
        self.assertIn("- Dataset: MATR", prompt)
        self.assertIn("- Model: BatLiNet (`batlinet`)", prompt)
        self.assertIn("- Prediction task: RUL prediction", prompt)
        self.assertIn("Never present a model name as a dataset", prompt)

    def test_the_file_content_is_sent(self) -> None:
        prompt = _prompt()
        self.assertIn("name: 'BatLiNetRULPredictor'", prompt)
        self.assertIn("cell_data_path: 'data/processed/MATR'", prompt)
        self.assertIn("MATRPrimaryTestTrainTestSplitter", prompt)

    def test_the_real_layout_is_described_and_inventing_is_forbidden(self) -> None:
        prompt = _prompt()
        for section in ("`model`", "`train_test_split`", "`feature`", "`label`"):
            self.assertIn(section, prompt)
        self.assertIn("Explain only the sections and keys that appear in the file", prompt)
        self.assertIn("Do not state metric values", prompt)

    def test_an_unreadable_file_is_said_not_guessed(self) -> None:
        prompt = _prompt(config_text=None)
        self.assertIn("could not be read", prompt)
        self.assertNotIn("```yaml", prompt)

    def test_a_large_file_is_truncated_and_says_so(self) -> None:
        prompt = _prompt(config_text="x" * (ce.MAX_CONFIG_CHARS + 50))
        self.assertIn(f"Truncated to the first {ce.MAX_CONFIG_CHARS} characters", prompt)
        self.assertNotIn("x" * (ce.MAX_CONFIG_CHARS + 1), prompt)

    def test_the_page_uses_it_and_no_longer_sends_only_the_path(self) -> None:
        app = APP.read_text(encoding="utf-8")
        self.assertIn("build_config_explanation_prompt(", app)
        self.assertNotIn('f"Explain this BatteryML config: {config_path}', app)
        # The suggestion prompt follows the selected task, not a fixed RUL label.
        block = app[app.index("build_config_suggestion_prompt(\n                                        dataset=ds"):]
        self.assertTrue(re.search(r"task=task_choice,", block[:400]), block[:400])


if __name__ == "__main__":
    unittest.main()
