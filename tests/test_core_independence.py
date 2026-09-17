"""Core BatteryML must not import the web application layer.

``CLAUDE.md`` §3 puts training in ``batteryml/`` and the application in
``batteryml_web/``, and the arrow only ever points one way. That rule was
briefly bent: the resumable-checkpoint schema lived in the web layer, so
``batteryml/pipeline.py`` had to import ``batteryml_web`` to resume its own
training. The schema now lives in ``batteryml/training_checkpoint.py`` and
the web layer re-exports it.

A rule nothing enforces is a rule that comes back, so these tests check the
property directly:

* no module under ``batteryml/`` imports ``batteryml_web``, by AST rather
  than by text — a docstring that *mentions* the web layer is documentation,
  not a dependency;
* the compatibility re-export hands back *the same objects*, so there is one
  schema, one serializer and one validator rather than two that drift.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_core_independence -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORE_ROOT = REPO_ROOT / "batteryml"
FORBIDDEN_ROOT_PACKAGE = "batteryml_web"


def _core_modules() -> list[Path]:
    """Every Python source file in the core package."""
    return sorted(
        path for path in CORE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imported_root_packages(source: str, path: Path) -> set[str]:
    """The top-level package names imported by ``source``.

    ``import a.b`` and ``from a.b import c`` both yield ``a``. A relative
    import (``from . import x``) has no root package outside this one and is
    skipped. Parsing rather than grepping is the point: a reference inside a
    string or a comment is not an import.
    """
    tree = ast.parse(source, filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: stays inside batteryml/
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


class TestCoreDoesNotImportWeb(unittest.TestCase):
    def test_scan_covers_the_files_that_matter(self):
        """Guard against the scan silently finding nothing to check."""
        modules = _core_modules()
        self.assertGreater(len(modules), 10, "core package scan found almost nothing")
        names = {path.relative_to(REPO_ROOT).as_posix() for path in modules}
        self.assertIn("batteryml/pipeline.py", names)
        self.assertIn("batteryml/training_checkpoint.py", names)
        self.assertIn("batteryml/models/nn_model.py", names)

    def test_no_core_module_imports_batteryml_web(self):
        offenders: list[str] = []
        for path in _core_modules():
            source = path.read_text(encoding="utf-8")
            if FORBIDDEN_ROOT_PACKAGE in _imported_root_packages(source, path):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(
            offenders, [],
            "core BatteryML must not depend on the web application layer; "
            f"these modules import {FORBIDDEN_ROOT_PACKAGE}: {offenders}. "
            "Move the shared contract into batteryml/ and re-export it from "
            "batteryml_web/ instead (see docs/UPSTREAM_PATCHES.md, UP-03).",
        )

    def test_scan_would_catch_a_real_import(self):
        """Non-vacuity: the detector fires on the thing it is meant to catch."""
        fake = Path("fake.py")
        for source in (
            "from batteryml_web.core import training_checkpoint",
            "import batteryml_web.core.training_checkpoint as tc",
            "import batteryml_web",
        ):
            with self.subTest(source=source):
                self.assertIn(
                    FORBIDDEN_ROOT_PACKAGE, _imported_root_packages(source, fake))

    def test_scan_ignores_prose(self):
        """A docstring naming the web layer is documentation, not a dependency."""
        source = '"""Re-exported by batteryml_web.core.training_checkpoint."""\n'
        self.assertNotIn(
            FORBIDDEN_ROOT_PACKAGE, _imported_root_packages(source, Path("fake.py")))


class TestOneSchemaOnly(unittest.TestCase):
    """The re-export is an alias, not a second copy."""

    def test_reexport_is_the_same_object(self):
        from batteryml import training_checkpoint as core
        from batteryml_web.core import training_checkpoint as shim

        exported = getattr(shim, "__all__", None)
        self.assertTrue(exported, "the compatibility shim must declare __all__")

        mismatched = [
            name for name in exported
            if getattr(shim, name) is not getattr(core, name)
        ]
        self.assertEqual(
            mismatched, [],
            "the web re-export must hand back the core objects themselves, "
            f"not copies; these differ: {mismatched}",
        )

    def test_reexport_covers_the_whole_public_surface(self):
        """Everything the core module *defines* is still reachable by the old path.

        Defined, not merely present: ``vars()`` would also list what the
        module imported for its own use (``Path``, ``Optional``), which was
        never part of the old import path's contract. The AST distinguishes
        the two exactly.
        """
        from batteryml import training_checkpoint as core
        from batteryml_web.core import training_checkpoint as shim

        source = Path(core.__file__).read_text(encoding="utf-8")
        defined: set[str] = set()
        for node in ast.parse(source, filename=core.__file__).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)

        public = {name for name in defined if not name.startswith("_")}
        self.assertGreater(len(public), 10, "public-surface scan found almost nothing")
        missing = sorted(public - set(shim.__all__))
        self.assertEqual(
            missing, [],
            f"the old import path no longer exposes: {missing}",
        )

    def test_pipeline_uses_the_core_schema(self):
        from batteryml import pipeline, training_checkpoint

        self.assertIs(pipeline._resume_schema_module, training_checkpoint)


if __name__ == "__main__":
    unittest.main()
