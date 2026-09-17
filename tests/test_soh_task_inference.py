"""A config's task comes from its structure, never from a substring.

The audit found the platform advertising 42 RUL configs as SOH estimation.
Every RUL baseline sets ``label: {name: RULLabelAnnotator, eol_soh: 0.9}``,
and the classifier asked whether the string ``soh`` appeared anywhere in the
YAML. ``eol_soh`` is RUL's *end-of-life threshold* — the SOH value at which a
cell counts as dead — so it says the run is about lifetime, not health.

These tests pin the structural hierarchy that replaced it, and they run over
the repository's real configs rather than over invented ones, because the
defect was in the real ones.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_soh_task_inference -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core.config_registry import get_config_registry  # noqa: E402

TASK_SOH = "SOH estimation"
TASK_RUL = "RUL prediction"


def _annotator_of(config_path: Path) -> str:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    label = data.get("label", {}) if isinstance(data, dict) else {}
    return str(label.get("name", "") or "") if isinstance(label, dict) else ""


class TestRealConfigsAreClassifiedByStructure(unittest.TestCase):
    """Run over configs/ as it actually is."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.entries = list(get_config_registry(str(REPO_ROOT)).entries)

    def test_the_task_always_matches_the_label_annotator(self) -> None:
        """The property the old classifier violated 42 times."""
        mismatched = []
        for entry in self.entries:
            annotator = _annotator_of(Path(entry.config_path))
            expected = TASK_SOH if "SOH" in annotator else TASK_RUL if "RUL" in annotator else None
            if expected and entry.task != expected:
                mismatched.append((entry.config_path.as_posix(), annotator, entry.task))
        self.assertEqual(
            mismatched, [],
            "a config's task disagrees with the label annotator it trains: "
            f"{mismatched}",
        )

    def test_no_rul_config_is_advertised_as_soh(self) -> None:
        offenders = [
            entry.config_path.as_posix()
            for entry in self.entries
            if entry.task == TASK_SOH and "RUL" in _annotator_of(Path(entry.config_path))
        ]
        self.assertEqual(
            offenders, [],
            "battery-lifetime configs are being presented as state-of-health "
            f"estimation: {offenders}",
        )

    def test_every_eol_soh_config_stays_rul(self) -> None:
        """The exact decoy, over every config that carries it."""
        checked = 0
        for entry in self.entries:
            text = Path(entry.config_path).read_text(encoding="utf-8")
            if "eol_soh" not in text:
                continue
            checked += 1
            self.assertEqual(
                entry.task, TASK_RUL,
                f"{entry.config_path.as_posix()} carries eol_soh (an RUL "
                "end-of-life threshold) and was classified as "
                f"{entry.task!r}",
            )
        self.assertGreater(
            checked, 30,
            "expected the repository's RUL baselines to carry eol_soh; the "
            "scan found almost none, so it is not testing what it claims",
        )

    def test_any_soh_entry_trains_an_soh_annotator(self) -> None:
        """Whatever the catalog offers as SOH must actually be SOH."""
        for entry in (e for e in self.entries if e.task == TASK_SOH):
            annotator = _annotator_of(Path(entry.config_path))
            self.assertIn(
                "SOH", annotator,
                f"{entry.config_path.as_posix()} is classified SOH but trains "
                f"{annotator!r}",
            )

    def test_the_legacy_dummy_configs_are_not_offered_as_soh_support(self) -> None:
        """They exist on disk; they are not the platform's SOH capability.

        configs/soh/dummy/* train DummyRegressor on one label per cell. Listing
        them would claim SOH support the platform does not have yet. The real
        config is written after scripts/audit_soh_readiness.py names an
        eligible dataset, so an empty SOH catalog here is the honest state,
        not a bug.
        """
        offered = [
            entry.config_path.as_posix() for entry in self.entries
            if "configs/soh/dummy/" in entry.config_path.as_posix()
        ]
        self.assertEqual(offered, [], f"legacy examples are in the catalog: {offered}")

        for name in ("matr_1.yaml", "matr_1_with_soh_filepath.yaml"):
            path = REPO_ROOT / "configs" / "soh" / "dummy" / name
            self.assertTrue(path.is_file(), f"{name} was deleted rather than withheld")
            text = path.read_text(encoding="utf-8")
            self.assertIn("legacy_example: true", text)
            self.assertIn("LEGACY EXAMPLE", text)

    def test_the_catalog_is_overwhelmingly_rul(self) -> None:
        """A sanity floor: the repository is an RUL benchmark with SOH added.

        Before the fix this ratio was 128 RUL / 44 SOH, which should have been
        visibly implausible.
        """
        rul = sum(1 for e in self.entries if e.task == TASK_RUL)
        soh = sum(1 for e in self.entries if e.task == TASK_SOH)
        self.assertGreater(rul, 100)
        self.assertLess(soh, 10, f"{soh} configs claim SOH; the audit expected a handful")


class TestInferenceHierarchy(unittest.TestCase):
    """The rules, on synthetic configs, independent of what configs/ holds."""

    def setUp(self) -> None:
        self.registry = get_config_registry(str(REPO_ROOT))

    def _task(self, data: dict, path: str = "configs/baselines/sklearn/xgb/x.yaml") -> str | None:
        hay = f"{path} {Path(path).name} {yaml.safe_dump(data)}".lower()
        return self.registry._infer_task(hay, data)

    def test_eol_soh_alone_never_implies_soh(self) -> None:
        task = self._task({"label": {"name": "RULLabelAnnotator", "eol_soh": 0.9}})
        self.assertEqual(task, TASK_RUL)

    def test_eol_soh_cannot_even_rescue_a_config_with_no_annotator(self) -> None:
        self.assertIsNone(self._task({"model": {"name": "X"}, "eol_soh": 0.9}))

    def test_the_soh_annotator_implies_soh(self) -> None:
        for annotator in ("SOHLabelAnnotator", "CycleSOHLabelAnnotator"):
            with self.subTest(annotator=annotator):
                self.assertEqual(self._task({"label": {"name": annotator}}), TASK_SOH)

    def test_an_explicit_task_field_wins(self) -> None:
        task = self._task({"task": "SOH estimation", "label": {"name": "SOHLabelAnnotator"}})
        self.assertEqual(task, TASK_SOH)

    def test_the_directory_is_only_a_fallback(self) -> None:
        self.assertEqual(
            self._task({"model": {"name": "X"}}, "configs/soh/dummy/matr_1.yaml"),
            TASK_SOH,
        )
        # ...and it does not override a declared annotator.
        self.assertEqual(
            self._task(
                {"label": {"name": "RULLabelAnnotator", "eol_soh": 0.9}},
                "configs/soh/dummy/matr_1.yaml",
            ),
            TASK_RUL,
        )

    def test_a_model_name_containing_rul_does_not_decide_the_task(self) -> None:
        """SOH runs use task-agnostic models whose class names say RUL."""
        task = self._task({
            "model": {"name": "XGBoostRULPredictor"},
            "label": {"name": "CycleSOHLabelAnnotator"},
        })
        self.assertEqual(task, TASK_SOH)

    def test_a_config_stating_nothing_is_unclassified(self) -> None:
        self.assertIsNone(self._task({"model": {"name": "XGBoostRULPredictor"}}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
