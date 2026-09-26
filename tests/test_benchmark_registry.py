"""The published benchmark values, and what they may be compared with.

The registry is a transcription, so the first test here re-parses the table
out of ``README_BATTERYML.md`` and compares it cell by cell. That is the check that
matters: a hand-copied table is only as good as its last proofread, and this
one is proofread on every test run.

What the registry's own docstring records about verification is asserted
here too — the metric, the unit, the seed protocol and the censoring — so
the claims in the prose cannot drift away from the data.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import benchmarks as bm  # noqa: E402


def _readme_table() -> dict:
    """The benchmark table as README_BATTERYML.md currently prints it.

    Labels keep their internal spaces (``Dummy regressor`` is two words);
    values lose all whitespace, because the README writes ``168±9`` and
    ``168 ± 9`` interchangeably and the spacing is not the fact.
    """
    rows = {}
    for line in (REPO_ROOT / "README_BATTERYML.md").read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        raw = line.strip().strip("|").split("|")
        if len(raw) != 9:
            continue
        label = re.sub(r"\s+", " ", raw[0].replace("**", "")).strip()
        if not label or not set(label) - set("-"):
            continue
        rows[label] = tuple(
            re.sub(r"\*\*|\s", "", cell) for cell in raw[1:]
        )
    rows.pop("Models", None)
    return rows


class TestTheRegistryMatchesTheReadme(unittest.TestCase):
    """The transcription is checked against its source, not trusted."""

    def test_every_cell_matches_the_readme(self) -> None:
        readme = _readme_table()
        self.assertTrue(readme, "no benchmark table found in README_BATTERYML.md")
        for model_key, row_label in bm.ROW_BY_MODEL_KEY.items():
            with self.subTest(model=model_key):
                self.assertIn(
                    row_label, readme,
                    f"README has no row labelled {row_label!r}",
                )
                expected = readme[row_label]
                actual = tuple(
                    bm.lookup(model_key, subset).value for subset in bm.SUBSETS
                )
                for subset, printed, value in zip(bm.SUBSETS, expected, actual):
                    self.assertEqual(
                        _printed(value), printed,
                        f"{row_label} / {subset} drifted from the README",
                    )

    def test_the_registry_covers_every_published_row(self) -> None:
        """A row in the README with no model key is a silent omission."""
        readme = _readme_table()
        mapped = set(bm.ROW_BY_MODEL_KEY.values())
        self.assertEqual(
            set(readme) - mapped, set(),
            "README rows with no model key in ROW_BY_MODEL_KEY",
        )

    def test_the_table_has_the_expected_shape(self) -> None:
        self.assertEqual(len(bm.SUBSETS), 8)
        self.assertEqual(len(bm.ROW_BY_MODEL_KEY), 14)
        self.assertEqual(len(bm.all_records()), 8 * 14)


def _printed(value: bm.BenchmarkValue) -> str:
    """The value back in the README's own notation, for comparison."""
    if value.censored:
        return f">{value.censor_threshold:.0f}"
    if value.dispersion is None:
        return f"{value.central:.0f}"
    return f"{value.central:.0f}±{value.dispersion:.0f}"


class TestLookup(unittest.TestCase):
    """Requirements 5-7."""

    def test_a_known_model_and_subset_resolves(self) -> None:
        record = bm.lookup("ridge", "MATR1")
        self.assertIsNotNone(record)
        self.assertEqual(record.value.central, 116.0)
        self.assertEqual(record.metric, "RMSE")
        self.assertEqual(record.unit, "cycles")
        self.assertEqual(record.task, "RUL prediction")
        self.assertTrue(record.lower_is_better)
        self.assertEqual(record.protocol, "MATRPrimaryTestTrainTestSplitter")

    def test_an_unknown_model_has_no_reference(self) -> None:
        """Requirement 6, and the mapping rule that matters most.

        CPTransformer is not the paper's "Transformer" and BatLiNet is not
        its "MLP". The platform has a ``transformer`` key *and* a
        ``cptransformer`` key, so a fuzzy match would have found one.
        """
        for model_key in ("cptransformer", "batlinet", "PBT", "BatteryMFormer"):
            with self.subTest(model=model_key):
                self.assertIsNone(bm.lookup(model_key, "MATR1"))
        # Non-vacuity: the key it must not be confused with does resolve.
        self.assertIsNotNone(bm.lookup("transformer", "MATR1"))

    def test_an_unknown_subset_has_no_reference(self) -> None:
        for subset in ("MATR", "matr1", "CALCE", "", None):
            with self.subTest(subset=subset):
                self.assertIsNone(bm.lookup("ridge", subset))


class TestValueRepresentation(unittest.TestCase):
    """Requirements 8-9."""

    def test_a_censored_cell_is_not_the_number_1000(self) -> None:
        """Requirement 8.

        Reading ">1000" as 1000 would make a run scoring 900 look like an
        improvement on a cell the paper declined to score at all.
        """
        record = bm.lookup("ridge", "HUST")
        self.assertTrue(record.value.censored)
        self.assertIsNone(record.value.central)
        self.assertEqual(record.value.censor_threshold, 1000.0)
        self.assertEqual(record.value.display(), ">1000")

    def test_dispersion_stays_distinct_from_the_central_value(self) -> None:
        """Requirement 9."""
        record = bm.lookup("cnn", "MATR1")
        self.assertEqual(record.value.central, 102.0)
        self.assertEqual(record.value.dispersion, 94.0)
        self.assertEqual(record.value.seeds, bm.SEEDS_MULTI)
        self.assertEqual(record.value.display(), "102 ± 94")

    def test_single_seed_rows_carry_no_dispersion(self) -> None:
        """``result.ipynb`` reads sklearn models from ``log.0`` alone."""
        record = bm.lookup("ridge", "MATR1")
        self.assertIsNone(record.value.dispersion)
        self.assertEqual(record.value.seeds, bm.SEEDS_SKLEARN)
        self.assertEqual(record.value.display(), "116")

    def test_a_zero_dispersion_is_kept_not_dropped(self) -> None:
        """``197±0`` is ten seeds that agreed, not one seed.

        A truthy test on the dispersion would read 0.0 as "absent" and
        quietly demote the row to a single run.
        """
        record = bm.lookup("rf", "MIX")
        self.assertEqual(record.value.dispersion, 0.0)
        self.assertEqual(record.value.seeds, bm.SEEDS_MULTI)

    def test_the_seed_count_follows_the_notation_not_a_model_list(self) -> None:
        """Random forest is where the two candidate sources disagree.

        ``result.ipynb`` groups ``rf`` with the single-seed sklearn
        baselines and formats that group as ``int(score)`` with no
        dispersion — yet the README's Random forest row reads ``168±9``. The
        saved notebook therefore does not reproduce that row, so it cannot
        be the authority on seed counts. The README's own sentence can:
        a dispersion means the ten-seed mean, its absence means one run.
        """
        self.assertEqual(bm.lookup("rf", "MATR1").value.seeds, bm.SEEDS_MULTI)
        self.assertEqual(bm.lookup("xgb", "MATR1").value.seeds, bm.SEEDS_SKLEARN)
        for record in bm.all_records():
            with self.subTest(model=record.model_key, subset=record.dataset_subset):
                if record.value.censored:
                    self.assertIsNone(record.value.seeds)
                elif record.value.dispersion is None:
                    self.assertEqual(record.value.seeds, bm.SEEDS_SKLEARN)
                else:
                    self.assertEqual(record.value.seeds, bm.SEEDS_MULTI)

    def test_parse_cell_rejects_nonsense(self) -> None:
        with self.assertRaises(ValueError):
            bm.parse_cell("about 400")


class TestSubsetMapping(unittest.TestCase):
    """Requirement 15's foundation: MATR1 vs MATR2 is read, never guessed."""

    def test_the_two_matr_subsets_come_from_their_split_classes(self) -> None:
        self.assertEqual(
            bm.subset_for_splitter("MATRPrimaryTestTrainTestSplitter"), "MATR1"
        )
        self.assertEqual(
            bm.subset_for_splitter("MATRSecondaryTestTrainTestSplitter"), "MATR2"
        )

    def test_the_generic_matr_splitter_identifies_no_subset(self) -> None:
        """The whole point: "MATR" alone does not choose MATR1.

        Defaulting to MATR1 because it is listed first would attach a run to
        a protocol it did not use.
        """
        self.assertIsNone(bm.subset_for_splitter("MATRTrainTestSplitter"))
        self.assertIsNone(bm.subset_for_splitter("RandomTrainTestSplitter"))
        self.assertIsNone(bm.subset_for_splitter(""))
        self.assertIsNone(bm.subset_for_splitter(None))

    def test_the_mapping_matches_the_shipped_configs(self) -> None:
        """Every mapped splitter is one a config in this repository uses.

        Guards the direction the README cannot: a splitter renamed upstream
        would leave this mapping pointing at a class that no longer exists.
        """
        import yaml

        used = set()
        for path in (REPO_ROOT / "configs").rglob("*.yaml"):
            try:
                parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a bad config is not this test's subject
                continue
            if isinstance(parsed, dict):
                name = (parsed.get("train_test_split") or {}).get("name")
                if name:
                    used.add(str(name))
        missing = set(bm.SUBSET_BY_SPLITTER) - used
        self.assertEqual(
            missing, set(),
            f"mapped splitters that no shipped config uses: {sorted(missing)}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
