"""Every module the committed code imports is actually committed.

This exists because of a defect that no test, lint or type check would have
caught. `.gitignore` carried a bare ``data`` rule. An unanchored pattern with
no slash matches *any* path component of that name at any depth, so it matched
the package source directory ``batteryml/data/`` — and silently kept
``batteryml/data/sample_index.py`` out of the commit whose ``task.py`` imports
it. The working tree was fine, every test passed, and the branch was broken
for anyone who cloned it:

    ModuleNotFoundError: No module named 'batteryml.data.sample_index'

The general shape is worth guarding: a file present locally but absent from
git is invisible to everything except a fresh checkout. So these tests ask git
what it has, not the filesystem.

They are skipped where git is unavailable or the tree is not a repository,
because then there is nothing to establish.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_source_completeness -v
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The packages whose every module must be committed. Scripts and tests are
#: included because the server runs them straight from a checkout.
SOURCE_ROOTS = ("batteryml", "batteryml_web", "scripts", "tests")

#: Modules the SOH route imports and that a clean checkout therefore needs.
#: Listed explicitly, so that deleting one fails here by name rather than as
#: an ImportError somewhere far away.
SOH_MODULES = (
    "batteryml.data.sample_index",
    "batteryml.feature.soh_cycle",
    "batteryml.label.soh",
    "batteryml.label.base",
    "batteryml.feature.base",
    "batteryml.task",
    "batteryml.pipeline",
    "batteryml_web.core.soh_readiness",
    "batteryml_web.core.config_registry",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _git_available() -> bool:
    try:
        return _git("rev-parse", "--is-inside-work-tree").returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


GIT_AVAILABLE = _git_available()


@unittest.skipUnless(GIT_AVAILABLE, "not a git work tree")
class TestEverySourceFileIsTracked(unittest.TestCase):
    """A .py file that git does not have does not exist for a fresh clone."""

    @classmethod
    def setUpClass(cls) -> None:
        listed = _git("ls-files", "--", *SOURCE_ROOTS)
        cls.tracked = {
            line.strip() for line in listed.stdout.splitlines() if line.strip()
        }

    def _on_disk(self) -> set[str]:
        found: set[str] = set()
        for root in SOURCE_ROOTS:
            for path in (REPO_ROOT / root).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                found.add(path.relative_to(REPO_ROOT).as_posix())
        return found

    def test_the_scan_finds_the_source_tree(self) -> None:
        self.assertGreater(len(self.tracked), 100, "git listed almost nothing")
        self.assertIn("batteryml/task.py", self.tracked)

    def test_no_python_source_file_is_untracked(self) -> None:
        missing = sorted(self._on_disk() - self.tracked)
        self.assertEqual(
            missing, [],
            "these source files exist locally but are not in git, so a fresh "
            f"checkout would not have them: {missing}. If .gitignore is the "
            "cause, anchor the offending rule (see the bare `data` rule this "
            "test was written for) rather than force-adding the file.",
        )

    def test_sample_index_is_tracked(self) -> None:
        """The specific file whose absence broke the branch."""
        self.assertIn(
            "batteryml/data/sample_index.py", self.tracked,
            "batteryml/data/sample_index.py is not committed, and "
            "batteryml/task.py imports it: a clean checkout raises "
            "ModuleNotFoundError before any SOH run starts.",
        )

    def test_no_source_file_is_ignored(self) -> None:
        """Catch the cause, not just this one symptom.

        ``git status --ignored`` over the source trees must list no ``.py``
        file. An ignored source file is the state that produced this defect.
        """
        status = _git(
            "status", "--short", "--ignored", "--untracked-files=all",
            "--", *SOURCE_ROOTS,
        )
        offenders = []
        for line in status.stdout.splitlines():
            if not line.startswith("!!"):
                continue
            path = line[2:].strip()
            if path.endswith(".py") and "__pycache__" not in path:
                offenders.append(path)
        self.assertEqual(
            offenders, [],
            f".gitignore is ignoring source files: {offenders}",
        )


@unittest.skipUnless(GIT_AVAILABLE, "not a git work tree")
class TestGitignoreDoesNotSwallowPackageSource(unittest.TestCase):
    """The rule that caused it, and the protections that must survive."""

    def test_the_data_rule_is_anchored(self) -> None:
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        rules = [line.strip() for line in text.splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        self.assertNotIn(
            "data", rules,
            "a bare `data` rule matches any directory named data at any "
            "depth, including the package source batteryml/data/. Anchor it "
            "as /data/.",
        )
        self.assertIn("/data/", rules)

    def test_package_source_is_not_ignored(self) -> None:
        for path in (
            "batteryml/data/sample_index.py",
            "batteryml/data/databundle.py",
            "batteryml/data/battery_data.py",
        ):
            with self.subTest(path=path):
                result = _git("check-ignore", "-q", "--no-index", path)
                self.assertNotEqual(
                    result.returncode, 0, f"{path} is ignored by .gitignore",
                )

    def test_real_datasets_are_still_ignored(self) -> None:
        """Fixing the rule must not start tracking the datasets."""
        for path in (
            "data/raw/MATR/MATR_b1c0.pkl",
            "data/processed/MATR/MATR_b1c0.pkl",
            "data/metadata/dataset_manifest.json",
            "runs/soh_smoke_matr_ridge/predictions_seed_0_x.pkl",
            "cache/battery_cache_x.pkl",
        ):
            with self.subTest(path=path):
                result = _git("check-ignore", "-q", "--no-index", path)
                self.assertEqual(
                    result.returncode, 0,
                    f"{path} is no longer ignored; dataset and artifact "
                    "protections must not be weakened by this fix.",
                )


class TestTheSOHModulesImportFromSource(unittest.TestCase):
    """Import what the committed SOH path imports, from this tree."""

    def test_sample_index_imports_and_carries_its_contract(self) -> None:
        from batteryml.data.sample_index import (  # noqa: F401
            WHOLE_CELL_CYCLE,
            SampleIndex,
            SampleIndexError,
            assert_disjoint_cells,
        )

        index = SampleIndex.for_cell("cell-a", [1, 2, 3])
        self.assertEqual(len(index), 3)
        self.assertEqual(index[1], {"cell_id": "cell-a", "cycle_index": 2})
        self.assertEqual(index.unique_cell_ids, ["cell-a"])
        with self.assertRaises(SampleIndexError):
            assert_disjoint_cells({"train": ["a"], "test": ["a"]})

    def test_every_listed_soh_module_imports(self) -> None:
        import importlib

        for name in SOH_MODULES:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_task_imports_sample_index_by_the_committed_path(self) -> None:
        """The import that was broken, read from the source that breaks."""
        source = (REPO_ROOT / "batteryml" / "task.py").read_text(encoding="utf-8")
        self.assertIn(
            "from batteryml.data.sample_index import", source,
            "task.py no longer imports sample_index; this test is out of date "
            "with the module it guards",
        )
        tree = ast.parse(source)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertIn("batteryml.data.sample_index", imported)


if __name__ == "__main__":
    unittest.main(verbosity=2)
