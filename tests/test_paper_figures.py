"""Standalone Fig. 3 panel scripts (``scripts/paper_figures``).

Every artifact here is synthetic (``CLAUDE.md`` §7): built in a temporary
directory with ``synthetic_`` cell ids, never under ``runs/`` or ``data/``.
Real BatteryML label transformations (LogScale -> ZScore, ZScore) are used
so the inverse path under test is the real one. Nothing trains, evaluates,
downloads or reads a dataset.

Run from the repository root::

    python -m unittest tests.test_paper_figures -v
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from batteryml.data.transformation.log_scale import LogScaleDataTransformation  # noqa: E402
from batteryml.data.transformation.sequential import SequentialDataTransformation  # noqa: E402
from batteryml.data.transformation.z_score import ZScoreDataTransformation  # noqa: E402
from scripts.paper_figures import collect_fig3_data as collect_mod  # noqa: E402
from scripts.paper_figures import common  # noqa: E402
from scripts.paper_figures import fig3a_lifetime_heatmap as panel_a  # noqa: E402
from scripts.paper_figures import fig3b_soh_error as panel_b  # noqa: E402
from scripts.paper_figures import fig3c_multicriteria_radar as panel_c  # noqa: E402
from scripts.paper_figures import fig3d_inference_latency as panel_d  # noqa: E402
from scripts.paper_figures import fig3e_lifetime_error_distribution as panel_e  # noqa: E402
from scripts.paper_figures import plot_fig3_all  # noqa: E402
from tests.fixtures import synthetic_result_artifacts as fx  # noqa: E402

PKG = REPO_ROOT / "scripts" / "paper_figures"
RUL_META = {"task": "RUL prediction", "label_annotator": "RULLabelAnnotator"}
SOH_META = {"task": "SOH estimation", "label_annotator": "CycleSOHLabelAnnotator",
            "reference_capacity_mode": "first_valid_discharge_capacity"}
LIFE_CONFIG = ("model:\n  name: {model_class}\nlabel:\n  name: {annotator}\n"
               "train_test_split:\n  name: {split}\n"
               "label_transformation:\n  name: SequentialDataTransformation\n")


def _chain(*members) -> SequentialDataTransformation:
    chain = SequentialDataTransformation.__new__(SequentialDataTransformation)
    chain.transformations = list(members)
    return chain


def _bundle(stored: torch.Tensor, transformation: Any) -> Any:
    bundle = fx._Bundle([], transformation)
    bundle.test_data.label = stored
    return bundle


def lifetime_payload(cells: Sequence[str], observed: Sequence[float],
                     predicted: Sequence[float], scores: Dict[str, float],
                     seed: int = 0) -> Dict[str, Any]:
    """Physical lifetimes stored LogScale -> ZScore, as the MATR configs do."""
    t = _chain(LogScaleDataTransformation(), ZScoreDataTransformation())
    t.fit(torch.tensor([[300.0], [500.0], [900.0], [1200.0], [700.0]]))
    payload = fx._payload(
        prediction=t.transform(torch.tensor(list(predicted)).reshape(-1, 1)),
        observed=None,
        sample_index=fx._records([(c, fx.WHOLE_CELL_CYCLE) for c in cells]),
        task_metadata=dict(RUL_META), scores=scores, seed=seed,
    )
    payload["data"] = _bundle(t.transform(torch.tensor(list(observed)).reshape(-1, 1)), t)
    return payload


def soh_payload(pairs, observed, predicted, scores, seed: int = 0) -> Dict[str, Any]:
    z = ZScoreDataTransformation()
    z.fit(torch.tensor([[0.95], [0.9], [0.85], [0.8], [1.0]]))
    payload = fx._payload(
        prediction=z.transform(torch.tensor(list(predicted)).reshape(-1, 1)),
        observed=None, sample_index=fx._records(pairs), task_metadata=dict(SOH_META),
        scores=scores, seed=seed,
    )
    payload["data"] = _bundle(z.transform(torch.tensor(list(observed)).reshape(-1, 1)), z)
    return payload


def write_run(root: Path, name: str, payload: Dict[str, Any], *, model_key: str,
              seed: int, dataset: str = "MATR", smoke: Optional[bool] = False,
              split: str = "MATRPrimaryTestTrainTestSplitter",
              annotator: str = "RULLabelAnnotator", config_tag: str = "") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    artifact = directory / f"predictions_seed_{seed}_20260101000000.pkl"
    artifact.write_bytes(pickle.dumps(payload))
    if smoke is not None:
        (directory / "run.json").write_text(json.dumps({
            "run_id": f"synthetic_{name}", "model_key": model_key, "dataset": dataset,
            "task": "RUL", "config_path": f"configs/synthetic/{model_key}{config_tag}.yaml",
            "seed": seed, "smoke": smoke, "status": "completed",
            **({"epochs": 1} if smoke else {}),
        }))
    (directory / "config_20260101000000.yaml").write_text(LIFE_CONFIG.format(
        model_class=f"Synthetic{model_key}{config_tag}", annotator=annotator, split=split))
    return artifact


CELLS = [f"synthetic_b1c{i}" for i in range(6)]
OBS = [512.0, 845.0, 1001.0, 433.0, 690.0, 777.0]


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="synthetic_fig3_"))
        self.runs = self.tmp / "runs"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def life(self, name: str, model: str, seed: int, offset: float, **kw) -> Path:
        predicted = [o + offset * (1 + 0.1 * i) for i, o in enumerate(OBS)]
        errors = [abs(p - o) for p, o in zip(predicted, OBS)]
        scores = {"RMSE": math.sqrt(sum(e * e for e in errors) / len(errors)),
                  "MAE": sum(errors) / len(errors), "MAPE": 0.05}
        cells = kw.pop("cells", CELLS)
        return write_run(self.runs, name, lifetime_payload(cells, OBS, predicted, scores, seed),
                         model_key=model, seed=seed, **kw)

    def select(self, entries: List[Dict[str, Any]], order: List[str], **extra) -> Path:
        path = self.tmp / f"selection_{len(list(self.tmp.glob('selection_*')))}.json"
        path.write_text(json.dumps({"model_order": order, "runs": entries, **extra}))
        return path

    def collect(self, entries, order, name="data", **extra) -> Path:
        out = self.tmp / name
        collect_mod.collect(self.select(entries, order, **extra), out, argv=["collect"])
        return out


class TestCollect(_Base):
    def test_units_inverted_once_and_aligned_by_cell(self) -> None:
        # Storage order deliberately unsorted; each cell keeps its own pair.
        order = [3, 0, 5, 1, 4, 2]
        cells, obs = [CELLS[i] for i in order], [OBS[i] for i in order]
        pred = [o + 10.0 * (k + 1) for k, o in enumerate(obs)]
        art = write_run(self.runs, "xgb", lifetime_payload(cells, obs, pred, {"RMSE": 1.0, "MAE": 1.0}),
                        model_key="xgb", seed=0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(art)}], ["xgb"])
        run = _read(data / "runs.csv")[0]
        self.assertEqual((run["task_kind"], run["unit"], run["transform_status"]),
                         ("lifetime", "cycles", "inverted"))
        self.assertEqual(run["transform_chain"], "LogScaleDataTransformation|ZScoreDataTransformation")
        samples = {r["cell_id"]: r for r in _read(data / "samples.csv")}
        for cell, o, p in zip(cells, obs, pred):
            # Physical cycles, not exp(cycles) (a double inverse) nor z-scores.
            self.assertAlmostEqual(float(samples[cell]["observed"]), o, places=2)
            self.assertAlmostEqual(float(samples[cell]["predicted"]), p, places=2)
            self.assertAlmostEqual(float(samples[cell]["abs_error"]), abs(p - o), places=2)
        self.assertEqual(run["protocol"], "MATR1")
        self.assertEqual(run["protocol_source"], "recorded")

    def test_declared_protocol_that_contradicts_the_record_is_refused(self) -> None:
        art = self.life("xgb", "xgb", 0, 20.0)
        with self.assertRaisesRegex(collect_mod.CollectError, "protocol"):
            self.collect([{"id": "a", "model": "xgb", "artifact": str(art),
                           "protocol": "MATR2"}], ["xgb"])

    def test_model_must_be_in_model_order_and_output_must_be_new(self) -> None:
        art = self.life("xgb", "xgb", 0, 20.0)
        with self.assertRaisesRegex(collect_mod.CollectError, "model_order"):
            self.collect([{"id": "a", "model": "xgb", "artifact": str(art)}], ["rf"])
        self.collect([{"id": "a", "model": "xgb", "artifact": str(art)}], ["xgb"], name="d1")
        with self.assertRaisesRegex(collect_mod.CollectError, "not empty"):
            self.collect([{"id": "a", "model": "xgb", "artifact": str(art)}], ["xgb"], name="d1")

    def test_nonfinite_predictions_are_dropped_and_counted(self) -> None:
        pred = [o + 5.0 for o in OBS]
        pred[2] = float("nan")
        art = write_run(self.runs, "xgb", lifetime_payload(CELLS, OBS, pred, {"RMSE": 1.0, "MAE": 1.0}),
                        model_key="xgb", seed=0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(art)}], ["xgb"])
        run = _read(data / "runs.csv")[0]
        self.assertEqual((run["stored_rows"], run["plotted_rows"], run["omitted_nonfinite"]),
                         ("6", "5", "1"))
        self.assertNotIn(CELLS[2], [r["cell_id"] for r in _read(data / "samples.csv")])

    def test_list_runs_lists_everything_by_path_and_chooses_nothing(self) -> None:
        self.life("b_run", "xgb", 0, 20.0)
        self.life("a_run", "rf", 1, 5.0, smoke=True)
        rows = collect_mod.list_runs(self.runs)
        self.assertEqual([Path(r["artifact"]).parent.name for r in rows], ["a_run", "b_run"])
        self.assertEqual(rows[0]["smoke"], True)
        self.assertIn("smoke=true", rows[0]["smoke_evidence"])

    def test_different_selection_does_not_reuse_stale_data(self) -> None:
        a = self.life("xgb", "xgb", 0, 20.0)
        b = self.life("rf", "rf", 0, 60.0)
        d1 = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)}], ["xgb", "rf"], name="d1")
        d2 = self.collect([{"id": "b", "model": "rf", "artifact": str(b)}], ["xgb", "rf"], name="d2")
        panel_a.make_panel(d1, self.tmp / "o1")
        panel_a.make_panel(d2, self.tmp / "o2")
        rows1 = _read(self.tmp / "o1" / f"{panel_a.STEM}_source_data.csv")
        rows2 = _read(self.tmp / "o2" / f"{panel_a.STEM}_source_data.csv")
        self.assertEqual([r["model"] for r in rows1], ["xgb"])
        self.assertEqual([r["model"] for r in rows2], ["rf"])
        self.assertEqual(rows2[0]["run_ids"], "synthetic_rf")


class TestHeatmap(_Base):
    def test_missing_cell_is_na_not_zero_and_single_run_has_no_sd(self) -> None:
        a = self.life("xgb_m1", "xgb", 0, 20.0)
        b = self.life("xgb_m2", "xgb", 0, 30.0, split="MATRSecondaryTestTrainTestSplitter")
        c = self.life("rf_m1", "rf", 0, 50.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)},
                             {"id": "b", "model": "xgb", "artifact": str(b)},
                             {"id": "c", "model": "rf", "artifact": str(c)}], ["xgb", "rf"])
        panel_a.make_panel(data, self.tmp / "out")
        rows = {(r["model"], r["protocol"]): r
                for r in _read(self.tmp / "out" / f"{panel_a.STEM}_source_data.csv")}
        self.assertEqual(rows[("rf", "MATR2")]["value_mean"], "NA")
        self.assertEqual(rows[("rf", "MATR2")]["display_text"], "NA")
        self.assertEqual(rows[("xgb", "MATR1")]["value_sd"], "NA")
        self.assertIn("1 run", rows[("xgb", "MATR1")]["display_text"])
        saved = float(_read(data / "runs.csv")[0]["saved_RMSE"])
        self.assertEqual(float(rows[("xgb", "MATR1")]["value_mean"]), saved)
        self.assertNotIn("Average", (self.tmp / "out" / f"{panel_a.STEM}.svg").read_text())

    def test_two_seeds_give_mean_and_sample_sd(self) -> None:
        a = self.life("s0", "xgb", 0, 20.0)
        b = self.life("s1", "xgb", 1, 40.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)},
                             {"id": "b", "model": "xgb", "artifact": str(b)}], ["xgb"])
        panel_a.make_panel(data, self.tmp / "out", metric="MAE")
        row = _read(self.tmp / "out" / f"{panel_a.STEM}_source_data.csv")[0]
        values = [float(r["saved_MAE"]) for r in _read(data / "runs.csv")]
        self.assertAlmostEqual(float(row["value_mean"]), sum(values) / 2)
        self.assertAlmostEqual(float(row["value_sd"]), abs(values[0] - values[1]) / math.sqrt(2))
        self.assertEqual(row["n_runs"], "2")

    def test_mixed_configs_or_repeated_seeds_are_invalid(self) -> None:
        a = self.life("c0", "xgb", 0, 20.0)
        b = self.life("c1", "xgb", 1, 40.0, config_tag="_tuned")
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)},
                             {"id": "b", "model": "xgb", "artifact": str(b)}], ["xgb"])
        with self.assertRaises(common.PanelSkip) as ctx:
            panel_a.make_panel(data, self.tmp / "out")
        self.assertEqual(ctx.exception.status, common.INVALID_COMPARISON)
        c = self.life("r0", "rf", 0, 20.0)
        d = self.life("r0b", "rf", 0, 25.0)
        data2 = self.collect([{"id": "c", "model": "rf", "artifact": str(c)},
                              {"id": "d", "model": "rf", "artifact": str(d)}], ["rf"], name="d2")
        with self.assertRaisesRegex(common.PanelSkip, "seeds"):
            panel_a.make_panel(data2, self.tmp / "out2")

    def test_incompatible_label_definitions_in_one_column_are_invalid(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        b = self.life("r", "rf", 0, 20.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)},
                             {"id": "b", "model": "rf", "artifact": str(b)}], ["xgb", "rf"])
        runs = _read(data / "runs.csv")
        runs[1]["sample_weighting"] = "cycle-pooled"
        with self.assertRaisesRegex(common.PanelSkip, "sample_weighting"):
            panel_a.local_cells(runs, ["xgb", "rf"], {}, metric="RMSE", allow_smoke=False)

    def test_smoke_excluded_by_default_and_labelled_when_allowed(self) -> None:
        a = self.life("smoke", "cnn", 0, 200.0, smoke=True)
        data = self.collect([{"id": "a", "model": "cnn", "artifact": str(a)}], ["cnn"])
        with self.assertRaises(common.PanelSkip) as ctx:
            panel_a.make_panel(data, self.tmp / "out")
        self.assertEqual(ctx.exception.status, common.MISSING_DATA)
        self.assertIn("--allow-smoke", ctx.exception.reason)
        panel_a.make_panel(data, self.tmp / "out", allow_smoke=True)
        meta = json.loads((self.tmp / "out" / f"{panel_a.STEM}_metadata.json").read_text())
        self.assertTrue(meta["smoke_included"])
        self.assertIn("workflow-validation", (self.tmp / "out" / f"{panel_a.STEM}.svg").read_text())

    def test_published_reference_is_separate_and_never_mae(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)}], ["ridge", "xgb"],
                            published_reference={"include": True, "models": ["ridge", "xgb"],
                                                 "protocols": ["MATR1", "HUST"]})
        with self.assertRaisesRegex(common.PanelSkip, "RMSE only"):
            panel_a.make_panel(data, self.tmp / "out", source="published", metric="MAE")
        panel_a.make_panel(data, self.tmp / "out", source="published")
        rows = {(r["model"], r["protocol"]): r for r in
                _read(self.tmp / "out" / f"{panel_a.PUBLISHED_STEM}_source_data.csv")}
        self.assertEqual(rows[("ridge", "HUST")]["censored"], "True")
        self.assertEqual(rows[("ridge", "HUST")]["value_mean"], "NA")
        self.assertEqual(float(rows[("ridge", "MATR1")]["value_mean"]), 116.0)
        self.assertEqual({r["source_type"] for r in rows.values()}, {"published reference"})

    def test_outputs_are_editable_svg_png_csv_metadata_and_never_overwritten(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)}], ["xgb"])
        result = panel_a.make_panel(data, self.tmp / "out")
        self.assertEqual(len(result.outputs), 4)
        svg = (self.tmp / "out" / f"{panel_a.STEM}.svg").read_text()
        self.assertIn("<text", svg)
        self.assertNotIn("<image", svg)
        self.assertNotIn("@font-face", svg)
        self.assertTrue((self.tmp / "out" / f"{panel_a.STEM}.png").read_bytes().startswith(b"\x89PNG"))
        meta = json.loads((self.tmp / "out" / f"{panel_a.STEM}_metadata.json").read_text())
        for key in ("runs", "metric", "unit", "aggregation", "missing_values",
                    "command", "code_version", "error_bar_definition"):
            self.assertIn(key, meta)
        self.assertEqual(meta["runs"][0]["artifact_sha256"], _read(data / "runs.csv")[0]["artifact_sha256"])
        self.assertTrue(meta["single_column"])
        with self.assertRaises(FileExistsError):
            panel_a.make_panel(data, self.tmp / "out")

    def test_more_models_than_palette_slots_is_refused(self) -> None:
        with self.assertRaises(common.PanelSkip):
            common.model_colors([f"m{i}" for i in range(9)])


class TestSohBars(_Base):
    PAIRS = [("synthetic_b2c1", 20), ("synthetic_b1c0", 15), ("synthetic_b2c1", 10),
             ("synthetic_b1c0", 10)]
    OBS = [0.93, 0.97, 0.95, 0.99]

    def soh(self, name, model, seed, delta) -> Path:
        pred = [o + delta * (1 + i) for i, o in enumerate(self.OBS)]
        mae = sum(abs(p - o) for p, o in zip(pred, self.OBS)) / len(pred)
        return write_run(self.runs, name, soh_payload(self.PAIRS, self.OBS, pred,
                                                       {"RMSE": mae * 1.2, "MAE": mae}, seed),
                         model_key=model, seed=seed, split="MATRPrimaryTestTrainTestSplitter",
                         annotator="CycleSOHLabelAnnotator")

    def test_pooled_mae_is_the_saved_score_and_pp_is_x100(self) -> None:
        a = self.soh("r0", "ridge", 0, 0.01)
        data = self.collect([{"id": "a", "model": "ridge", "artifact": str(a)}], ["ridge", "xgb"])
        run = _read(data / "runs.csv")[0]
        self.assertTrue(run["sample_weighting"].startswith("cycle-pooled"))
        panel_b.make_panel(data, self.tmp / "out", scale="pp")
        bar = _read(self.tmp / "out" / f"{panel_b.STEM}_source_data.csv")[0]
        self.assertAlmostEqual(float(bar["value_mean"]), float(run["saved_MAE"]) * 100)
        self.assertEqual(bar["value_sd"], "NA")
        self.assertIn("percentage points", bar["unit"])
        svg = (self.tmp / "out" / f"{panel_b.STEM}.svg").read_text()
        self.assertNotIn("MAPE", svg)

    def test_cell_macro_is_a_separate_statistic(self) -> None:
        a = self.soh("r0", "ridge", 0, 0.01)
        data = self.collect([{"id": "a", "model": "ridge", "artifact": str(a)}], ["ridge"])
        panel_b.make_panel(data, self.tmp / "out", statistic="cell-macro")
        bar = _read(self.tmp / "out" / f"{panel_b.MACRO_STEM}_source_data.csv")[0]
        samples = _read(data / "samples.csv")
        per_cell: Dict[str, List[float]] = {}
        for r in samples:
            per_cell.setdefault(r["cell_id"], []).append(float(r["abs_error"]))
        expected = sum(sum(v) / len(v) for v in per_cell.values()) / len(per_cell)
        self.assertAlmostEqual(float(bar["value_mean"]), expected)
        self.assertIn("macro", bar["statistic"])

    def test_two_seeds_give_error_bar(self) -> None:
        a = self.soh("r0", "ridge", 0, 0.01)
        b = self.soh("r1", "ridge", 1, 0.02)
        data = self.collect([{"id": "a", "model": "ridge", "artifact": str(a)},
                             {"id": "b", "model": "ridge", "artifact": str(b)}], ["ridge"])
        panel_b.make_panel(data, self.tmp / "out")
        bar = _read(self.tmp / "out" / f"{panel_b.STEM}_source_data.csv")[0]
        self.assertNotEqual(bar["value_sd"], "NA")
        self.assertEqual(bar["n_runs"], "2")

    def test_no_soh_runs_is_missing_data(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)}], ["xgb"])
        with self.assertRaises(common.PanelSkip) as ctx:
            panel_b.make_panel(data, self.tmp / "out")
        self.assertEqual(ctx.exception.status, common.MISSING_DATA)


RADAR_CONFIG = {"dimensions": [
    {"name": "lifetime_RMSE_MATR1", "definition": "d", "unit": "cycles", "source": "runs.csv",
     "direction": "lower_is_better", "bounds": [0, 400]},
    {"name": "lifetime_RMSE_MATR2", "definition": "d", "unit": "cycles", "source": "runs.csv",
     "direction": "lower_is_better", "bounds": [0, 800]},
    {"name": "cells_within_10pct", "definition": "d", "unit": "fraction", "source": "samples.csv",
     "direction": "higher_is_better", "bounds": [0, 1]},
]}


def _radar_rows(values: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    units = {d["name"]: d["unit"] for d in RADAR_CONFIG["dimensions"]}
    return [{"model": m, "dimension": d, "value": str(v), "unit": units[d],
             "source": "synthetic", "n_repeats": "1"}
            for m, dims in values.items() for d, v in dims.items()]


class TestRadar(unittest.TestCase):
    FULL = {"xgb": {"lifetime_RMSE_MATR1": 100.0, "lifetime_RMSE_MATR2": 200.0,
                    "cells_within_10pct": 0.5},
            "rf": {"lifetime_RMSE_MATR1": 300.0, "lifetime_RMSE_MATR2": 400.0,
                   "cells_within_10pct": 0.25}}

    def test_scores_follow_fixed_bounds_and_direction(self) -> None:
        status, problems, scored = panel_c.validate_radar_inputs(
            _radar_rows(self.FULL), RADAR_CONFIG, ["xgb", "rf"])
        self.assertEqual((status, problems), (common.CREATED, []))
        score = {(r["model"], r["dimension"]): r["score"] for r in scored}
        self.assertAlmostEqual(score[("xgb", "lifetime_RMSE_MATR1")], 0.75)
        self.assertAlmostEqual(score[("rf", "cells_within_10pct")], 0.25)
        self.assertNotIn("composite", json.dumps(scored))

    def test_no_basis_no_scores(self) -> None:
        self.assertEqual(panel_c.validate_radar_inputs([], None)[0], common.MISSING_DATA)
        two = {"dimensions": RADAR_CONFIG["dimensions"][:2]}
        rows = [r for r in _radar_rows(self.FULL) if r["dimension"] != "cells_within_10pct"]
        self.assertEqual(panel_c.validate_radar_inputs(rows, two)[0], common.INSUFFICIENT_DATA)
        partial = _radar_rows(self.FULL)[:-1]
        status, problems, scored = panel_c.validate_radar_inputs(partial, RADAR_CONFIG)
        self.assertEqual((status, scored), (common.INSUFFICIENT_DATA, []))
        self.assertIn("never filled with 0", problems[0])
        subjective = json.loads(json.dumps(RADAR_CONFIG))
        subjective["dimensions"][2].update(name="interpretability", unit="score")
        self.assertEqual(panel_c.validate_radar_inputs(_radar_rows(self.FULL), subjective)[0],
                         common.INVALID_COMPARISON)
        template = json.loads((PKG / "fig3c_radar_config.template.json").read_text())
        self.assertEqual(panel_c.validate_radar_inputs(_radar_rows(self.FULL), template)[0],
                         common.INVALID_COMPARISON)
        degenerate = json.loads(json.dumps(RADAR_CONFIG))
        degenerate["dimensions"][0]["bounds"] = [5, 5]
        self.assertEqual(panel_c.validate_radar_inputs(_radar_rows(self.FULL), degenerate)[0],
                         common.INVALID_COMPARISON)
        outside = json.loads(json.dumps(self.FULL))
        outside["rf"]["lifetime_RMSE_MATR1"] = 900.0
        self.assertEqual(panel_c.validate_radar_inputs(_radar_rows(outside), RADAR_CONFIG)[0],
                         common.INVALID_COMPARISON)


def _latency_row(model: str, **over) -> Dict[str, str]:
    row = {"model": model, "latency_value": "0.42", "latency_unit": "ms_per_sample",
           "statistic": "mean", "sd": "0.03", "batch_size": "1", "device": "cpu",
           "hardware": "synthetic CPU", "cpu_threads": "4", "warmup_runs": "10",
           "repeats": "100", "timing_boundary": "model.predict on preloaded features",
           "includes_data_transfer": "false", "includes_feature_computation": "false",
           "includes_postprocessing": "false", "gpu_sync": "false",
           "measurement_kind": "inference", "source_file": "synthetic_timing.log"}
    row.update(over)
    return row


class TestLatency(unittest.TestCase):
    def test_valid_rows_pass(self) -> None:
        self.assertEqual(panel_d.validate_latency_rows([_latency_row("xgb"), _latency_row("rf")],
                                                       ["xgb", "rf"]), [])

    def test_training_or_cli_time_is_not_latency(self) -> None:
        for over in ({"timing_boundary": "full training loop"},
                     {"timing_boundary": "CLI wall time"},
                     {"measurement_kind": "training"},
                     {"timing_boundary": "model load + predict"}):
            self.assertTrue(panel_d.validate_latency_rows([_latency_row("xgb", **over)]), over)

    def test_mixed_conditions_and_units_are_invalid(self) -> None:
        for over in ({"hardware": "other CPU"}, {"latency_unit": "ms_per_batch"},
                     {"batch_size": "32"}, {"includes_data_transfer": "true"}):
            problems = panel_d.validate_latency_rows([_latency_row("xgb"), _latency_row("rf", **over)])
            self.assertTrue(problems, over)

    def test_sd_needs_repeats_and_cuda_needs_sync(self) -> None:
        self.assertTrue(panel_d.validate_latency_rows([_latency_row("xgb", repeats="1")]))
        self.assertTrue(panel_d.validate_latency_rows([_latency_row("xgb", device="cuda:0")]))

    def test_no_measurements_lists_what_to_measure(self) -> None:
        with self.assertRaises(common.PanelSkip) as ctx:
            panel_d.build([], ["xgb", "rf"], {}, yscale="linear")
        self.assertEqual(ctx.exception.status, common.MISSING_DATA)
        self.assertIn("xgb", ctx.exception.reason)

    def test_template_has_exactly_the_required_columns(self) -> None:
        header = (PKG / "fig3d_latency.template.csv").read_text().strip().split(",")
        self.assertEqual(header, panel_d.LATENCY_COLUMNS)


class TestDistribution(_Base):
    def test_errors_are_per_cell_differences_not_from_rmse(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        b = self.life("r", "rf", 0, -35.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)},
                             {"id": "b", "model": "rf", "artifact": str(b)}], ["xgb", "rf"])
        panel_e.make_panel(data, self.tmp / "out")
        rows = _read(self.tmp / "out" / f"{panel_e.STEM}_source_data.csv")
        self.assertEqual(len(rows), 2 * len(CELLS))
        for r in rows:
            self.assertAlmostEqual(float(r["abs_error"]),
                                   abs(float(r["predicted"]) - float(r["observed"])))
        rf_errors = sorted(float(r["abs_error"]) for r in rows if r["model"] == "rf")
        expected = sorted(35.0 * (1 + 0.1 * i) for i in range(6))
        for got, want in zip(rf_errors, expected):
            self.assertAlmostEqual(got, want, places=2)
        meta = json.loads((self.tmp / "out" / f"{panel_e.STEM}_metadata.json").read_text())
        self.assertEqual(meta["display"], "box+points")
        self.assertIn("KDE not used", meta["display_reason"])

    def test_cell_coverage_mismatch_is_invalid(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        b = self.life("r", "rf", 0, 20.0, cells=[c.replace("b1", "b9") for c in CELLS])
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)},
                             {"id": "b", "model": "rf", "artifact": str(b)}], ["xgb", "rf"])
        with self.assertRaisesRegex(common.PanelSkip, "different test cells"):
            panel_e.make_panel(data, self.tmp / "out")

    def test_multi_seed_needs_explicit_seed(self) -> None:
        a = self.life("s0", "xgb", 0, 20.0)
        b = self.life("s1", "xgb", 1, 30.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)},
                             {"id": "b", "model": "xgb", "artifact": str(b)}], ["xgb"])
        with self.assertRaisesRegex(common.PanelSkip, "--seed"):
            panel_e.make_panel(data, self.tmp / "out")
        panel_e.make_panel(data, self.tmp / "out", seed="1")
        rows = _read(self.tmp / "out" / f"{panel_e.STEM}_source_data.csv")
        self.assertEqual({r["seed"] for r in rows}, {"1"})

    def test_constant_and_zero_errors_on_log_axis(self) -> None:
        art = write_run(self.runs, "x", lifetime_payload(CELLS, OBS, list(OBS), {"RMSE": 0.0, "MAE": 0.0}),
                        model_key="xgb", seed=0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(art)}], ["xgb"])
        rows = _read(data / "samples.csv")
        built = panel_e.build(_read(data / "runs.csv"), rows, ["xgb"], {}, yscale="log")
        # LogScale/ZScore round trips may leave float dust; count exact zeros only.
        zeros = sum(1 for r in rows if float(r["abs_error"]) == 0.0)
        self.assertEqual(built["zeros"]["xgb"], zeros)
        self.assertEqual(built["mode"], "box+points")
        self.assertEqual(sum(1 for p in built["points"] if not p["on_axis"]), zeros)
        result = panel_e.make_panel(data, self.tmp / "out", yscale="log")
        self.assertEqual(result.status, common.CREATED)
        meta = json.loads((self.tmp / "out" / f"{panel_e.STEM}_metadata.json").read_text())
        self.assertIn("no epsilon", meta["log_zero_strategy"])

    def test_altered_abs_error_is_detected(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)}], ["xgb"])
        samples = _read(data / "samples.csv")
        samples[0]["abs_error"] = "1.0"
        with self.assertRaisesRegex(common.PanelSkip, "altered"):
            panel_e.build(_read(data / "runs.csv"), samples, ["xgb"], {})


class TestPlotAll(_Base):
    def test_status_table_draws_valid_panels_only(self) -> None:
        a = self.life("x", "xgb", 0, 20.0)
        data = self.collect([{"id": "a", "model": "xgb", "artifact": str(a)}], ["xgb"])
        results = plot_fig3_all.plot_all(data, self.tmp / "out", argv=["--data", str(data)])
        status = {r.panel: r.status for r in results}
        self.assertEqual(status[panel_a.STEM], common.CREATED)
        self.assertEqual(status[panel_e.STEM], common.CREATED)
        self.assertEqual(status[panel_b.STEM], common.MISSING_DATA)
        self.assertEqual(status[panel_c.STEM], common.MISSING_DATA)
        self.assertEqual(status[panel_d.STEM], common.MISSING_DATA)
        self.assertFalse((self.tmp / "out" / f"{panel_b.STEM}.svg").exists())
        table = (self.tmp / "out" / "fig3_status.md").read_text()
        self.assertIn("| Panel | Status | Output / Reason |", table)
        meta = json.loads((self.tmp / "out" / f"{panel_a.STEM}_metadata.json").read_text())
        self.assertIn("plot_fig3_all", meta["command"])
        self.assertEqual(list((self.tmp / "out").glob("*fig3_all*.svg")), [])
        with self.assertRaises(FileExistsError):
            plot_fig3_all.plot_all(data, self.tmp / "out")

    def test_example_selection_is_a_template_without_real_runs(self) -> None:
        selection = json.loads((PKG / "fig3_selection.example.json").read_text())
        for entry in selection["runs"]:
            self.assertIn("<", entry["artifact"])
        with self.assertRaises(collect_mod.CollectError):
            collect_mod.collect(PKG / "fig3_selection.example.json", self.tmp / "d")


if __name__ == "__main__":
    unittest.main()
