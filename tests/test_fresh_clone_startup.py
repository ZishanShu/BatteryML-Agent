"""A fresh clone must start. Both halves of that, held by assertions.

Two independent defects made `git clone && pip install && run` impossible,
and neither was visible from inside the machine the platform was developed
on, because that machine happened to satisfy both by accident.

1. ``batteryml_web/models/cptransformer.py`` wraps a network whose source
   lives in ``third_party/BatteryLife``. ``third_party/`` is gitignored, so
   a fresh clone does not carry it; the module raised ``FileNotFoundError``
   at import, ``register_extension_models()`` turned that into
   ``ExtensionRegistrationError``, and ``run_batteryml_cli.py`` turned that
   into ``SystemExit(2)``. One un-vendored *optional* model stopped the
   entire command line — XGBoost, Random Forest and every other model with
   it.

2. ``tools/sync_data_and_config_manifests.py`` hard-coded one developer's
   absolute server path as its root, so on any other clone it wrote the
   dataset manifest outside the repository. The web UI reads only
   ``data/metadata/dataset_manifest.json`` to learn what datasets exist, so
   Prepare Data showed "No datasets are available yet" however much data had
   been preprocessed.

These tests are about *startability*, not about CPTransformer. The thing
being protected is that one absent optional extra can never again take the
platform down with it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TOOL = REPO_ROOT / "tools" / "sync_data_and_config_manifests.py"


class TestOptionalExtensionsDoNotBlockStartup(unittest.TestCase):
    """An un-vendored optional wrapper is skipped, never fatal."""

    def setUp(self) -> None:
        from batteryml_web import models

        self.models = models

    def test_the_external_requirement_is_declared_not_guessed(self) -> None:
        """Declared, so the check needs neither torch nor an exception."""
        self.assertIn("cptransformer", self.models.EXTERNAL_SOURCES)
        for name, relative in self.models.EXTERNAL_SOURCES.items():
            with self.subTest(model=name):
                self.assertIn(
                    name,
                    self.models.EXTENSION_MODULES,
                    "an external source is declared for a module that is not "
                    "a registered extension",
                )
                self.assertTrue(
                    relative.startswith("third_party/"),
                    "only a vendored third-party source may be declared "
                    f"optional; got {relative!r}",
                )

    def test_an_absent_source_is_reported_rather_than_raised(self) -> None:
        absent = self.models.unavailable_extension_models(REPO_ROOT / "no-such-root")
        self.assertEqual(
            set(absent), set(self.models.EXTERNAL_SOURCES),
            "with nothing vendored, every declared external must be reported",
        )

    def test_a_present_source_is_not_reported(self) -> None:
        """The probe must key off the file, not off a hard-coded verdict."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in self.models.EXTERNAL_SOURCES.values():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# stand-in for a vendored source\n")
            self.assertEqual(
                self.models.unavailable_extension_models(root),
                {},
                "a vendored source must not still be reported as missing",
            )

    def test_registration_survives_the_absent_source(self) -> None:
        """The defect itself: this raised SystemExit(2) before the fix."""
        imported = self.models.register_extension_models()
        unavailable = self.models.unavailable_extension_models()

        for name in unavailable:
            with self.subTest(skipped=name):
                self.assertNotIn(f"{self.models.__name__}.{name}", imported)

        expected = [
            name for name in self.models.EXTENSION_MODULES if name not in unavailable
        ]
        self.assertEqual(
            [name.rsplit(".", 1)[-1] for name in imported],
            expected,
            "every extension whose source is present must still register",
        )

    def test_the_cli_wrapper_starts(self) -> None:
        """End to end, through the entry point a user actually types."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "batteryml_web/core/run_batteryml_cli.py",
                "run",
                "--help",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"the CLI refused to start.\nstdout: {result.stdout[:400]}\n"
            f"stderr: {result.stderr[:400]}",
        )
        self.assertIn("--train", result.stdout)
        self.assertNotIn("Could not import extension model module", result.stderr)

    def test_a_broken_extension_still_fails_loudly(self) -> None:
        """The skip is narrow on purpose.

        Only a *declared* external whose file is *provably* absent may be
        skipped. A module that is present and raises is a defect, and must
        still stop the run rather than disappear from the registry.
        """
        import batteryml_web.models as models

        saved = models.EXTENSION_MODULES
        try:
            models.EXTENSION_MODULES = saved + ("definitely_not_a_module",)
            with self.assertRaises(models.ExtensionRegistrationError):
                models.register_extension_models()
        finally:
            models.EXTENSION_MODULES = saved

    def test_the_module_still_explains_itself_on_a_direct_import(self) -> None:
        """A direct import bypasses the registry, so the guard stays."""
        source = (
            REPO_ROOT / "batteryml_web" / "models" / "cptransformer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("raise FileNotFoundError", source)
        self.assertIn("third_party/BatteryLife", source)


class TestAnUnavailableModelIsExplainedNotMislabelled(unittest.TestCase):
    """"Not registered" would send the reader to fix the wrong thing."""

    def test_the_validator_names_the_missing_source(self) -> None:
        from batteryml_web.core import config_validator
        from batteryml_web.models import unavailable_extension_models

        unavailable = unavailable_extension_models()
        if not unavailable:
            self.skipTest("every external source is vendored in this checkout")

        config = REPO_ROOT / "configs" / "external" / "cptransformer" / "matr_1.yaml"
        if not config.is_file():
            self.skipTest(f"{config} is not present in this checkout")

        report = config_validator.load_and_validate_config_file(
            config, require_data=False
        )
        joined = " ".join(report.get("errors", []))
        self.assertIn("third_party/BatteryLife", joined)
        self.assertIn("not vendored", joined)
        self.assertNotIn(
            "register the missing implementation",
            joined,
            "the reader is being told to write code that already exists",
        )


class TestManifestToolUsesThisCheckout(unittest.TestCase):
    """The dataset manifest must land inside the repository it describes."""

    def setUp(self) -> None:
        self.source = MANIFEST_TOOL.read_text(encoding="utf-8")

    def test_no_developer_machine_is_hard_coded(self) -> None:
        tree = ast.parse(self.source)
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for literal in literals:
            with self.subTest(literal=literal[:60]):
                self.assertFalse(
                    literal.startswith("/") and "BatteryML" in literal,
                    f"an absolute path to one machine is hard-coded: {literal!r}. "
                    "The manifest would be written outside any other clone.",
                )

    def test_the_root_resolves_to_this_repository(self) -> None:
        import runpy

        namespace = runpy.run_path(str(MANIFEST_TOOL), run_name="__not_main__")
        self.assertEqual(
            Path(namespace["ROOT"]).resolve(),
            REPO_ROOT.resolve(),
            "the tool does not describe the checkout it lives in",
        )

    def test_an_explicit_root_is_honoured(self) -> None:
        """The multi-checkout case the absolute path was presumably for."""
        import os
        import runpy
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            saved = os.environ.get("BATTERYML_ROOT")
            os.environ["BATTERYML_ROOT"] = tmp
            try:
                namespace = runpy.run_path(
                    str(MANIFEST_TOOL), run_name="__not_main__"
                )
                self.assertEqual(
                    Path(namespace["ROOT"]).resolve(), Path(tmp).resolve()
                )
            finally:
                if saved is None:
                    os.environ.pop("BATTERYML_ROOT", None)
                else:
                    os.environ["BATTERYML_ROOT"] = saved

    def _run_in(self, tmp: str) -> dict:
        import os
        import runpy

        saved = os.environ.get("BATTERYML_ROOT")
        os.environ["BATTERYML_ROOT"] = tmp
        try:
            return runpy.run_path(str(MANIFEST_TOOL), run_name="__not_main__")
        finally:
            if saved is None:
                os.environ.pop("BATTERYML_ROOT", None)
            else:
                os.environ["BATTERYML_ROOT"] = saved

    def test_loading_the_tool_creates_no_batterylife_folder(self) -> None:
        """The empty data/raw/BatteryLife/v12 became a phantom dataset row."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._run_in(tmp)
            self.assertFalse((Path(tmp) / "data" / "raw" / "BatteryLife").exists())
            self.assertTrue((Path(tmp) / "data" / "metadata").is_dir())

    def test_the_dataset_manifest_builds_without_a_batterylife_folder(self) -> None:
        import contextlib
        import io
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            namespace = self._run_in(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["build_dataset_manifest"]()
            manifest = json.loads(
                (Path(tmp) / "data" / "metadata" / "dataset_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(any(e.get("source") == "BatteryLife/v12" for e in manifest))
            self.assertFalse((Path(tmp) / "data" / "raw" / "BatteryLife").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
