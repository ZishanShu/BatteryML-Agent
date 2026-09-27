"""The SOH baseline pack: one protocol, several estimators.

`configs/soh/ridge/matr.yaml` proved the SOH route end to end on the server.
This module is about the estimators added beside it, and it asserts the one
thing that makes them comparable at all: **every SOH config declares the
identical protocol and differs only in its ``model:`` block.**

A per-model split, reference convention, sample policy or feature extractor
would mean two SOH numbers that cannot be put next to each other, while
looking exactly as if they could. That is the failure this file exists to
prevent, so the comparison is made field by field against the Ridge config
rather than by re-asserting the expected values one config at a time.

What this file does *not* establish: anything about accuracy. No model here is
better than another until a controlled multi-seed comparison has been run on
real data, and none has been. See docs/SOH_PROTOCOL.md.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_soh_baseline_pack -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOH_ROOT = REPO_ROOT / "configs" / "soh"
REFERENCE = SOH_ROOT / "ridge" / "matr.yaml"

TASK_SOH = "SOH estimation"

#: The estimators the audit enabled, and the class each must name. Written out
#: rather than discovered, so that adding a config without auditing it fails
#: here instead of silently joining the menu.
EXPECTED_PACK = {
    "ridge": "RidgeRULPredictor",
    "lasso": "LassoRULPredictor",
    "elastic_net": "ElasticNetRULPredictor",
    "rf": "RandomForestRULPredictor",
    "xgb": "XGBoostRULPredictor",
}

#: Every top-level section that is protocol rather than estimator. If two SOH
#: configs disagree on any of these, their metrics are not comparable.
PROTOCOL_SECTIONS = (
    "task",
    "train_test_split",
    "feature",
    "label",
    "feature_transformation",
    "label_transformation",
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _pack_paths() -> dict[str, Path]:
    """The MATR SOH config for each model in the pack."""
    return {model: SOH_ROOT / model / "matr.yaml" for model in EXPECTED_PACK}


class TestThePackExists(unittest.TestCase):
    def test_every_audited_model_has_a_matr_config(self) -> None:
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                self.assertTrue(path.is_file(), f"{path} is missing")

    def test_each_config_names_its_own_estimator(self) -> None:
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                self.assertEqual(_load(path)["model"]["name"], EXPECTED_PACK[model])

    def test_no_unaudited_soh_config_has_appeared(self) -> None:
        """A directory under configs/soh/ is all it takes to reach the menu.

        The legacy ``dummy`` example is excluded by its own ``legacy_example``
        flag in the registry and is not part of the pack; anything else that
        turns up here has joined the SOH menu without being audited.
        """
        found = {
            path.parent.name
            for path in SOH_ROOT.rglob("*.yaml")
            if not bool(_load(path).get("legacy_example"))
        }
        self.assertEqual(
            found, set(EXPECTED_PACK),
            "an SOH config exists for a model the audit did not enable",
        )


class TestOneProtocolForEveryEstimator(unittest.TestCase):
    """C1: same dataset, split, reference convention, policy and feature."""

    def setUp(self) -> None:
        self.reference = _load(REFERENCE)

    def test_every_protocol_section_matches_the_ridge_config(self) -> None:
        for model, path in _pack_paths().items():
            if path == REFERENCE:
                continue
            data = _load(path)
            for section in PROTOCOL_SECTIONS:
                with self.subTest(model=model, section=section):
                    self.assertEqual(
                        data.get(section), self.reference.get(section),
                        f"{model}'s {section!r} differs from the Ridge config; "
                        "the two results would not be comparable",
                    )

    def test_the_only_difference_is_the_model_block(self) -> None:
        """Non-vacuity, both ways.

        Every top-level key is either identical to Ridge's or is ``model:``,
        and ``model:`` genuinely differs — otherwise the check above could
        pass on five copies of the same file.
        """
        for model, path in _pack_paths().items():
            if path == REFERENCE:
                continue
            data = _load(path)
            with self.subTest(model=model):
                self.assertEqual(
                    set(data), set(self.reference),
                    f"{model} declares a different set of top-level sections",
                )
                differing = {
                    key for key in data if data[key] != self.reference[key]
                }
                self.assertEqual(
                    differing, {"model"},
                    f"{model} differs from the Ridge config in {differing}, "
                    "not in the estimator alone",
                )

    def test_every_config_declares_the_soh_task(self) -> None:
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                self.assertEqual(_load(path)["task"], TASK_SOH)

    def test_every_config_uses_the_cycle_level_soh_label(self) -> None:
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                label = _load(path)["label"]
                self.assertEqual(label["name"], "CycleSOHLabelAnnotator")
                self.assertEqual(
                    label["reference_capacity_mode"],
                    "first_valid_discharge_capacity",
                )
                self.assertEqual(label["min_cycle"], 10)
                self.assertEqual(label["cycle_stride"], 5)
                self.assertIsNone(label["max_cycle"])

    def test_every_config_uses_the_soh_feature_extractor(self) -> None:
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                self.assertEqual(
                    _load(path)["feature"]["name"],
                    "SOHVoltageCurveFeatureExtractor",
                )

    def test_no_config_carries_an_rul_label_or_its_threshold(self) -> None:
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("RULLabelAnnotator", text)
                self.assertNotIn("eol_soh", text)

    def test_no_config_stacks_a_log_on_the_label(self) -> None:
        """A log of a ratio near 1.0 compresses the range SOH lives in."""
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                transformation = _load(path)["label_transformation"]
                self.assertEqual(transformation["name"], "ZScoreDataTransformation")
                self.assertNotIn("LogScale", yaml.safe_dump(transformation))


class TestNoEstimatorWasTuned(unittest.TestCase):
    """C6: model coverage, not benchmark optimization.

    A hyperparameter that was chosen by looking at a score would make the pack
    a (badly run) comparison instead of a compatibility statement. The four
    added configs declare no hyperparameters at all, matching every
    corresponding config in ``configs/baselines/``.
    """

    def test_the_added_configs_declare_no_hyperparameters(self) -> None:
        for model, path in _pack_paths().items():
            if path == REFERENCE:
                continue  # Ridge's alpha: 1.0 predates this pack.
            with self.subTest(model=model):
                block = dict(_load(path)["model"])
                block.pop("name")
                self.assertEqual(
                    block, {},
                    f"{model} declares {sorted(block)}; the pack uses library "
                    "defaults so that no score influenced a config",
                )

    def test_no_random_state_is_pinned_over_the_run_seed(self) -> None:
        """``--seed`` must keep meaning what it says.

        Pinning ``random_state`` in a config would fix the estimator's
        randomness regardless of the seed the run was launched with, making
        the recorded seed a claim that is not true. The seed reaches sklearn
        through ``batteryml.pipeline.set_seed``, which is called immediately
        before ``fit``.
        """
        for model, path in _pack_paths().items():
            with self.subTest(model=model):
                self.assertNotIn("random_state", _load(path)["model"])


class TestTheCatalogIsBackendAuthoritative(unittest.TestCase):
    def test_the_registry_lists_exactly_the_pack_for_matr_soh(self) -> None:
        from batteryml_web.core.config_registry import get_config_registry

        registry = get_config_registry(str(REPO_ROOT))
        keys = sorted(
            {
                entry.model_key
                for entry in registry.get_entries(dataset="MATR", task=TASK_SOH)
            }
        )
        self.assertEqual(keys, sorted(EXPECTED_PACK))

    def test_each_pack_entry_is_displayed_by_its_generic_name(self) -> None:
        """Never "XGBoost RUL" for an SOH run; the RUL is naming only."""
        from batteryml_web.core.config_registry import get_config_registry

        registry = get_config_registry(str(REPO_ROOT))
        for entry in registry.get_entries(dataset="MATR", task=TASK_SOH):
            with self.subTest(model=entry.model_key):
                self.assertNotIn("RUL", entry.model_display)

    def test_the_rul_catalog_did_not_shrink(self) -> None:
        """Adding SOH configs may not remove anything from the RUL menu."""
        from batteryml_web.core.config_registry import get_config_registry

        registry = get_config_registry(str(REPO_ROOT))
        rul = {
            entry.model_key
            for entry in registry.get_entries(task="RUL prediction")
        }
        self.assertGreater(len(rul), 20)
        for expected in ("ridge", "xgb", "rf", "lasso", "elastic_net", "cnn", "lstm"):
            with self.subTest(model=expected):
                self.assertIn(expected, rul)


class TestEveryPackModelIsATaskAgnosticScalarRegressor(unittest.TestCase):
    """The audit's actual finding, re-checked against the source.

    ``SklearnModel.fit`` is ``model.fit(feature.view(n, -1), label)`` and
    ``predict`` returns a flat tensor — no lifetime, end-of-life or cycle
    semantics anywhere — which is *why* these estimators can take a
    cycle-level SOH bundle at all. If a pack model ever stopped inheriting
    from it, that reasoning would no longer hold.
    """

    @staticmethod
    def _build(class_name: str):
        import importlib

        import batteryml.models  # noqa: F401  (registers the core models)
        from batteryml.builders import MODELS

        if class_name not in MODELS.class_mapping:
            # Extension models register only through the CLI wrapper; import
            # the one module rather than the whole extension set, which needs
            # third_party/ that this container does not have.
            importlib.import_module("batteryml_web.models.lasso")
        return MODELS.build({"name": class_name})

    def test_every_pack_model_is_a_sklearn_model(self) -> None:
        from batteryml.models.sklearn_model import SklearnModel

        for model, class_name in EXPECTED_PACK.items():
            with self.subTest(model=model):
                self.assertIsInstance(self._build(class_name), SklearnModel)

    def test_mlp_is_not_a_sklearn_model_which_is_why_it_was_rejected(self) -> None:
        """Non-vacuity: the check above must be able to fail.

        ``MLPRULPredictor`` is an ``NNModel``, not a ``SklearnModel``. Its
        constructor alone makes the incompatibility concrete — it *requires*
        ``input_height`` and ``input_width``, i.e. a two-dimensional
        per-sample feature geometry — and its ``forward`` then unpacks
        ``B, _, H, W = feature.size()``. A flat 37-wide SOH row has no such
        geometry. The model is registered and importable; it is simply not
        compatible with this feature path, and reshaping the SOH feature to
        suit it would be changing the science to fit the model.
        """
        import batteryml.models  # noqa: F401  (registers the core models)
        from batteryml.builders import MODELS
        from batteryml.models.sklearn_model import SklearnModel

        mlp = MODELS.class_mapping["MLPRULPredictor"]
        self.assertFalse(issubclass(mlp, SklearnModel))

        import inspect

        required = [
            name
            for name, parameter in inspect.signature(mlp.__init__).parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
            and name != "self"
        ]
        self.assertIn("input_height", required)
        self.assertIn("input_width", required)


if __name__ == "__main__":
    unittest.main(verbosity=2)
