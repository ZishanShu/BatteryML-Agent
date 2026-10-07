"""Fig. 3 experiment tooling: identity audit, measurement maths, runner logic.

Synthetic fixtures only (``synthetic_`` cell ids, temporary directories).
No training is started: the runner's launcher is replaced by a function that
writes a synthetic artifact, so plan/resume/acceptance logic is exercised
without BatteryML ever fitting a model.

Run from the repository root::

    python -m unittest tests.test_fig3_experiments -v
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_figures import artifact_audit as aa  # noqa: E402
from scripts.paper_figures import collect_fig3_data as collect_mod  # noqa: E402
from scripts.paper_figures import common  # noqa: E402
from scripts.paper_figures import fig3_experiments as fx3  # noqa: E402
from scripts.paper_figures import fig3_measure as fm  # noqa: E402
from scripts.paper_figures import fig3a_lifetime_heatmap as panel_a  # noqa: E402
from scripts.paper_figures import fig3c_multicriteria_radar as panel_c  # noqa: E402
from scripts.paper_figures import fig3d_inference_latency as panel_d  # noqa: E402
from scripts.paper_figures import fig3e_lifetime_error_distribution as panel_e  # noqa: E402
from tests.test_paper_figures import (  # noqa: E402
    CELLS, OBS, _Base, _read, lifetime_payload, soh_payload, write_run,
)


def _strip_index(payload: Dict[str, Any]) -> Dict[str, Any]:
    """An artifact as an old cached dataset would produce it: no identity."""
    payload.pop("sample_index", None)
    payload.pop("task_metadata", None)
    return payload


class TestIdentity(_Base):
    def _unnamed(self, name: str, model: str, offset: float) -> Path:
        pred = [o + offset for o in OBS]
        return write_run(self.runs, name, _strip_index(lifetime_payload(
            CELLS, OBS, pred, {"RMSE": abs(offset), "MAE": abs(offset)})), model_key=model, seed=0)

    def test_missing_identity_is_reported_not_turned_into_one_repeated_cell(self) -> None:
        art = self._unnamed("bat", "batlinet", 30.0)
        data = self.collect([{"id": "b", "model": "batlinet", "artifact": str(art)}], ["batlinet"])
        run = _read(data / "runs.csv")[0]
        self.assertEqual(run["identity_status"], "not_recorded")
        self.assertEqual({r["cell_id"] for r in _read(data / "samples.csv")}, {""})
        report = aa.audit_payload(pickle.loads(art.read_bytes()))
        self.assertIn("not recorded", report["identity"])
        self.assertEqual(report["total_rows"], len(CELLS))

    def test_e_excludes_unnamed_model_with_reason_and_draws_the_others(self) -> None:
        bat = self._unnamed("bat", "batlinet", 30.0)
        xgb = self.life("xgb", "xgb", 0, 20.0)
        data = self.collect([{"id": "b", "model": "batlinet", "artifact": str(bat)},
                             {"id": "x", "model": "xgb", "artifact": str(xgb)}], ["xgb", "batlinet"])
        result = panel_e.make_panel(data, self.tmp / "out")
        self.assertIn("excluded batlinet", result.reason)
        meta = json.loads((self.tmp / "out" / f"{panel_e.STEM}_metadata.json").read_text())
        self.assertEqual(meta["models"], ["xgb"])
        self.assertIn("batlinet", meta["excluded_models"])

    def test_verified_sidecar_restores_identity_and_wrong_artifact_is_refused(self) -> None:
        bat = self._unnamed("bat", "batlinet", 30.0)
        payload = pickle.loads(bat.read_bytes())
        observed = aa._flat(aa._inverse(payload, payload["data"].test_data.label))
        verdict = aa.match_identity(observed, {"labels": list(OBS), "cell_ids": list(CELLS)})
        self.assertEqual(verdict["status"], "verified")
        from batteryml_web.agent.artifact_provenance import sha256_file

        sidecar = self.tmp / "identity.json"
        sidecar.write_text(json.dumps({"kind": aa.SIDECAR_KIND, "status": "verified",
                                       "artifact_sha256": sha256_file(bat), "cell_ids": CELLS,
                                       "method": "synthetic", "max_abs_label_diff": 0.0}))
        data = self.collect([{"id": "b", "model": "batlinet", "artifact": str(bat),
                              "identity_sidecar": str(sidecar)}], ["batlinet"])
        run = _read(data / "runs.csv")[0]
        self.assertEqual((run["identity_status"], run["rows_per_cell_max"]), ("recovered", "1"))
        self.assertEqual(sorted(r["cell_id"] for r in _read(data / "samples.csv")), sorted(CELLS))
        other = self.life("xgb", "xgb", 0, 20.0)
        with self.assertRaisesRegex(collect_mod.CollectError, "different artifact"):
            self.collect([{"id": "x", "model": "xgb", "artifact": str(other),
                           "identity_sidecar": str(sidecar)}], ["xgb"], name="d2")

    def test_label_mismatch_refuses_recovery(self) -> None:
        verdict = aa.match_identity([1.0, 2.0], {"labels": [1.0, 3.0], "cell_ids": ["a", "b"]})
        self.assertEqual(verdict["status"], "mismatch")
        verdict = aa.match_identity([1.0], {"labels": [1.0, 3.0], "cell_ids": ["a", "b"]})
        self.assertEqual(verdict["status"], "mismatch")

    def test_real_repeated_cells_are_reported_and_excluded_never_averaged(self) -> None:
        cells = [CELLS[0], CELLS[0], CELLS[1], CELLS[1]]
        obs = [OBS[0], OBS[0], OBS[1], OBS[1]]
        pred = [o + d for o, d in zip(obs, (10.0, 30.0, 5.0, 15.0))]
        payload = lifetime_payload(cells, obs, pred, {"RMSE": 1.0, "MAE": 1.0})
        report = aa.audit_payload(payload)
        self.assertEqual(report["rows_per_cell_distribution"], {"2": 2})
        self.assertTrue(report["observed_constant_within_repeated_cell"])
        self.assertEqual(len(report["repeated_example"]), 2)
        art = write_run(self.runs, "rep", payload, model_key="batlinet", seed=0)
        xgb = write_run(self.runs, "x2", lifetime_payload(
            cells, obs, [o + 1 for o in obs], {"RMSE": 1.0, "MAE": 1.0}), model_key="xgb", seed=0)
        data = self.collect([{"id": "r", "model": "batlinet", "artifact": str(art)},
                             {"id": "x", "model": "xgb", "artifact": str(xgb)}], ["xgb", "batlinet"])
        with self.assertRaisesRegex(common.PanelSkip, "not averaged"):
            panel_e.make_panel(data, self.tmp / "out")

    def test_evaluator_recompute_matches_saved_scores(self) -> None:
        import torch

        from batteryml.data.databundle import DataBundle
        from batteryml.data.transformation.log_scale import LogScaleDataTransformation
        from batteryml.data.transformation.z_score import ZScoreDataTransformation
        from tests.test_paper_figures import _chain

        # A real DataBundle, so the audit calls BatteryML's own evaluator.
        bundle = DataBundle(torch.zeros(5, 3), torch.tensor([300.0, 500.0, 900.0, 1200.0, 700.0]),
                            torch.zeros(len(OBS), 3), torch.tensor(OBS),
                            label_transformation=_chain(LogScaleDataTransformation(),
                                                        ZScoreDataTransformation()))
        pred = torch.tensor([o + 10.0 for o in OBS])
        payload = {"prediction": bundle.label_transformation.transform(pred), "data": bundle,
                   "scores": {"RMSE": 10.0, "MAE": 10.0}, "seed": 0}
        report = aa.audit_payload(payload)
        self.assertLess(report["saved_vs_evaluator"]["MAE"], 1e-2)
        self.assertAlmostEqual(report["evaluator_recomputed"]["MAE"], 10.0, places=2)
        self.assertAlmostEqual(report["row_pooled_recomputed"]["RMSE"], 10.0, places=2)
        self.assertEqual(report["evaluator_broadcast_risk"], "none")

    def test_observed_labels_must_agree_within_a_column(self) -> None:
        rows = [{"entry_id": "a", "identity_status": "recorded", "dataset": "MATR", "protocol": "MATR1"},
                {"entry_id": "b", "identity_status": "recorded", "dataset": "MATR", "protocol": "MATR1"}]
        samples = [{"entry_id": "a", "cell_id": "c1", "cycle_index": "-1", "observed": "500"},
                   {"entry_id": "b", "cell_id": "c1", "cycle_index": "-1", "observed": "620"}]
        with self.assertRaisesRegex(common.PanelSkip, "different ground truth"):
            common.check_observed_agree(rows, samples)


class TestHeatmapLabels(_Base):
    def test_deterministic_and_observation_budget_are_shown(self) -> None:
        art = self.life("x", "ridge", 0, 20.0)
        data = self.collect([{"id": "a", "model": "ridge", "artifact": str(art),
                              "determinism": "deterministic",
                              "observation_budget": "cycles <= 98"}], ["ridge"])
        panel_a.make_panel(data, self.tmp / "out")
        row = _read(self.tmp / "out" / f"{panel_a.STEM}_source_data.csv")[0]
        self.assertIn("deterministic", row["display_text"])
        self.assertIn("cycles <= 98", row["model_label"])


REF_CONFIG = {
    "normalization": {"mode": "reference_relative", "reference_model": "ridge"},
    "dimensions": [
        {"name": "SOH_MAE", "unit": "ratio", "direction": "lower_is_better",
         "definition": "d", "source": "s"},
        {"name": "inference_latency", "unit": "ms/batch", "direction": "lower_is_better",
         "definition": "d", "source": "s"},
        {"name": "inference_files_size", "unit": "MiB", "direction": "lower_is_better",
         "definition": "d", "source": "s"},
    ],
}


def _ref_rows(values: Dict[str, List[float]]) -> List[Dict[str, str]]:
    dims = [d["name"] for d in REF_CONFIG["dimensions"]]
    units = {d["name"]: d["unit"] for d in REF_CONFIG["dimensions"]}
    return [{"model": m, "dimension": d, "value": str(v), "unit": units[d], "source": "measured",
             "n_repeats": "1"} for m, vs in values.items() for d, v in zip(dims, vs)]


class TestReferenceRadar(unittest.TestCase):
    def test_reference_is_half_and_formula_is_r_over_r_plus_x(self) -> None:
        status, problems, scored = panel_c.validate_radar_inputs(
            _ref_rows({"ridge": [0.02, 0.5, 0.01], "rf": [0.01, 5.0, 40.0]}), REF_CONFIG,
            ["ridge", "rf"])
        self.assertEqual((status, problems), (common.CREATED, []))
        score = {(r["model"], r["dimension"]): r for r in scored}
        for dim in ("SOH_MAE", "inference_latency", "inference_files_size"):
            self.assertAlmostEqual(score[("ridge", dim)]["score"], 0.5)
        self.assertAlmostEqual(score[("rf", "SOH_MAE")]["score"], 0.02 / 0.03)
        self.assertAlmostEqual(score[("rf", "inference_latency")]["score"], 0.5 / 5.5)
        self.assertEqual(score[("rf", "SOH_MAE")]["reference_value"], 0.02)

    def test_invalid_reference_inputs_are_refused(self) -> None:
        bad_ref = _ref_rows({"ridge": [0.0, 0.5, 0.01], "rf": [0.01, 5.0, 40.0]})
        self.assertEqual(panel_c.validate_radar_inputs(bad_ref, REF_CONFIG)[0], common.INVALID_COMPARISON)
        negative = _ref_rows({"ridge": [0.02, 0.5, 0.01], "rf": [-0.01, 5.0, 40.0]})
        self.assertEqual(panel_c.validate_radar_inputs(negative, REF_CONFIG)[0], common.INVALID_COMPARISON)
        no_ref = _ref_rows({"rf": [0.01, 5.0, 40.0]})
        self.assertEqual(panel_c.validate_radar_inputs(no_ref, REF_CONFIG)[0], common.INVALID_COMPARISON)
        higher = json.loads(json.dumps(REF_CONFIG))
        higher["dimensions"][0]["direction"] = "higher_is_better"
        rows = _ref_rows({"ridge": [0.02, 0.5, 0.01]})
        self.assertEqual(panel_c.validate_radar_inputs(rows, higher)[0], common.INVALID_COMPARISON)
        two = json.loads(json.dumps(REF_CONFIG))
        two["dimensions"] = two["dimensions"][:2]
        rows2 = [r for r in rows if r["dimension"] != "inference_files_size"]
        self.assertEqual(panel_c.validate_radar_inputs(rows2, two)[0], common.INSUFFICIENT_DATA)
        unknown = json.loads(json.dumps(REF_CONFIG))
        unknown["normalization"]["mode"] = "min_max_over_models"
        self.assertEqual(panel_c.validate_radar_inputs(rows, unknown)[0], common.INVALID_COMPARISON)

    def test_measured_radar_files_pass_the_panel_validator(self) -> None:
        entries = [{"model": m, "artifact": f"synthetic_{m}.pkl", "saved_mae": mae,
                    "mean_ms_per_batch": ms, "repeats": 100, "inference_mib": mib}
                   for m, mae, ms, mib in (("ridge", 0.02, 0.1, 0.01), ("xgb", 0.01, 2.0, 0.6))]
        radar = fm.reference_radar(entries, reference_model="ridge", batch_size=32)
        rows = [{k: str(v) for k, v in r.items()} for r in radar["rows"]]
        status, problems, _ = panel_c.validate_radar_inputs(rows, radar["config"], ["ridge", "xgb"])
        self.assertEqual((status, problems), (common.CREATED, []))
        self.assertIn("trade-offs", radar["config"]["display_name"])


class TestMeasurementMaths(unittest.TestCase):
    def test_batch_positions_are_rank_based_and_fixed(self) -> None:
        self.assertEqual(fm.batch_positions(100, 4), [0, 33, 66, 99])
        self.assertEqual(fm.batch_positions(32, 32), list(range(32)))
        with self.assertRaises(fm.MeasureError):
            fm.batch_positions(10, 32)

    def test_timing_stats_convert_ns_to_ms_with_sample_sd(self) -> None:
        stats = fm.timing_stats([1_000_000, 3_000_000])
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["sd"], 2 ** 0.5)
        self.assertAlmostEqual(stats["median"], 2.0)
        with self.assertRaises(fm.MeasureError):
            fm.timing_stats([5])

    def test_measurement_rows_satisfy_the_latency_validator(self) -> None:
        row = {"model": "ridge", "latency_value": "0.21", "latency_unit": "ms_per_batch",
               "statistic": "mean", "sd": "0.01", "batch_size": "32", "device": "cpu",
               "hardware": "synthetic CPU", "cpu_threads": "1", "warmup_runs": "20",
               "repeats": "100", "timing_boundary": fm.TIMING_BOUNDARY,
               "includes_data_transfer": "false", "includes_feature_computation": "false",
               "includes_postprocessing": "false", "gpu_sync": "false",
               "measurement_kind": "inference", "source_file": "latency_raw.csv"}
        self.assertEqual(list(row), fm.LATENCY_COLUMNS)
        self.assertEqual(fm.LATENCY_COLUMNS, panel_d.LATENCY_COLUMNS)
        self.assertEqual(panel_d.validate_latency_rows([row]), [])

    def test_measure_refuses_without_single_thread_environment(self) -> None:
        saved = {k: os.environ.pop(k, None) for k in fm.THREAD_ENV}
        try:
            with self.assertRaisesRegex(fm.MeasureError, "thread environment"):
                fm.measure({"protocol": {}, "entries": []}, Path(tempfile.mkdtemp()) / "m")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_latency_is_never_derived_from_training_time(self) -> None:
        row = {c: "x" for c in panel_d.LATENCY_COLUMNS}
        row.update(model="ridge", latency_value="12", latency_unit="ms_per_batch",
                   statistic="mean", sd="", batch_size="32", cpu_threads="1", warmup_runs="0",
                   repeats="1", includes_data_transfer="false",
                   includes_feature_computation="false", includes_postprocessing="false",
                   gpu_sync="false", measurement_kind="inference",
                   timing_boundary="training wall time from run.json")
        self.assertTrue(panel_d.validate_latency_rows([row]))


def _candidate(**over) -> Dict[str, Any]:
    base = {"artifact": "runs/x/predictions_seed_0_1.pkl", "run_json": "run.json",
            "status": "COMPLETED", "return_code": 0, "smoke": False, "epochs_override": None,
            "seed": "0", "config_copy_sha256": "abc", "predictions_for_seed": 1,
            "model_checkpoint": "runs/x/1.ckpt", "finished_at": "2026-09-18T00:00:00", "model": "ridge"}
    base.update(over)
    return base


class TestPlanning(unittest.TestCase):
    def test_reuse_rules(self) -> None:
        task = {"seed": 0, "config_sha256": "abc", "needs_checkpoint": True}
        self.assertEqual(fx3.eligible_reuse(_candidate(), task), [])
        self.assertTrue(fx3.eligible_reuse(_candidate(smoke=True), task))
        self.assertTrue(fx3.eligible_reuse(_candidate(epochs_override="1"), task))
        torch_task = dict(task, training_budget={"epochs": 100})
        self.assertEqual(fx3.eligible_reuse(_candidate(epochs_override="100"), torch_task), [])
        self.assertTrue(fx3.eligible_reuse(_candidate(epochs_override="1"), torch_task))
        self.assertTrue(fx3.eligible_reuse(_candidate(config_copy_sha256="zzz"), task))
        self.assertTrue(fx3.eligible_reuse(_candidate(run_json=None), task))
        self.assertTrue(fx3.eligible_reuse(_candidate(return_code=1), task))
        self.assertTrue(fx3.eligible_reuse(_candidate(model_checkpoint=None), task))
        choice = fx3.choose_reuse([_candidate(artifact="b", finished_at="2026-09-19"),
                                   _candidate(artifact="a", finished_at="2026-09-18"),
                                   _candidate(artifact="c", smoke=True, finished_at="2026-01-01")], task)
        self.assertEqual(choice["chosen"], "a")
        self.assertIn("not chosen by any score", choice["rule"])

    def test_plan_from_repository_configs(self) -> None:
        plan = fx3.build_plan(repo_root=REPO_ROOT, candidates=[], cpu_threads=4,
                              data_counts={"data/processed/MATR": 180, "data/processed/HUST": 0},
                              present_paths=[])
        ids = {t["task_id"] for t in plan["tasks"]}
        self.assertIn("soh-MATR-MATR1-ridge-seed0", ids)
        self.assertNotIn("soh-MATR-MATR1-ridge-seed1", ids)       # deterministic: one fit
        self.assertNotIn("soh-MATR-MATR1-xgb-seed1", ids)         # seed does not reach XGB
        self.assertIn("soh-MATR-MATR1-rf-seed2", ids)
        self.assertIn("lifetime-MATR-MATR1-cnn-seed2", ids)
        by_id = {t["task_id"]: t for t in plan["tasks"]}
        self.assertEqual(by_id["lifetime-HUST-HUST-ridge-seed0"]["action"], "blocked")
        self.assertEqual(by_id["lifetime-MATR-MATR1-cptransformer-seed0"]["action"], "blocked")
        self.assertEqual(by_id["lifetime-MATR-MATR1-batlinet-seed0"]["action"], "new run")
        self.assertEqual(by_id["lifetime-MATR-MATR1-batlinet-seed0"]["observation_budget"], "cycles 1-20")
        self.assertTrue(by_id["soh-MATR-MATR1-rf-seed0"]["needs_checkpoint"])
        self.assertFalse(by_id["soh-MATR-MATR1-rf-seed1"]["needs_checkpoint"])
        self.assertTrue(any("HUST/batlinet" in n["what"] for n in plan["not_planned"]))
        for task in plan["tasks"]:
            self.assertNotIn("--epochs", json.dumps(task))

    def test_gpu_choice_skips_busy_devices(self) -> None:
        gpus = ("0, GPU-a, 30000, 40000, 95\n1, GPU-b, 3, 40000, 0\n2, GPU-c, 3, 40000, 0\n")
        apps = "GPU-b, 1234\n"
        self.assertEqual(fx3.pick_free_gpu(gpus, apps)["index"], 2)
        self.assertIsNone(fx3.pick_free_gpu("0, GPU-a, 30000, 40000, 95\n", ""))

    def test_failure_classes_and_epoch_evidence(self) -> None:
        self.assertEqual(fx3.classify_failure("AssertionError: data/processed/MATR"), "shared_data")
        self.assertEqual(fx3.classify_failure("CUDA out of memory"), "task")
        tmp = Path(tempfile.mkdtemp(prefix="synthetic_epochs_"))
        try:
            (tmp / "20260101_seed_0_epoch_1000.ckpt").write_bytes(b"x")
            self.assertEqual(fx3.epochs_completed(tmp, "", 1000), 1000)
            (tmp / "20260101_seed_0_epoch_1000.ckpt").unlink()
            self.assertEqual(fx3.epochs_completed(tmp, "[100/100] RMSE 12.0", 100), 100)
            self.assertIsNone(fx3.epochs_completed(tmp, "[90/100] RMSE 12.0", 100))
        finally:
            shutil.rmtree(tmp)

    def test_seed_effect_detects_identical_replicates(self) -> None:
        self.assertTrue(fx3.seed_effect({0: [1.0, 2.0], 1: [1.0, 2.0]})["identical_predictions"])
        self.assertFalse(fx3.seed_effect({0: [1.0, 2.0], 1: [1.0, 2.5]})["identical_predictions"])


SOH_PAIRS = [(f"synthetic_s{c}", cyc) for c in range(2) for cyc in (10, 15, 20)]
SOH_OBS = [0.99, 0.98, 0.97, 0.995, 0.985, 0.975]


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="synthetic_campaign_"))
        self.campaign = self.tmp / "campaign"
        self.campaign.mkdir()
        plan = fx3.build_plan(repo_root=REPO_ROOT, candidates=[], cpu_threads=2,
                              data_counts={"data/processed/MATR": 1, "data/processed/HUST": 0},
                              present_paths=[])
        plan["tasks"] = [t for t in plan["tasks"] if t["group"] == "soh"]
        self.plan = plan
        fx3.write_json_atomic(self.campaign / "experiment_plan.json", plan)
        fx3.save_state(self.campaign, fx3.initial_state(plan))
        self.args = SimpleNamespace(retry_failed=False, only_groups=[], max_tasks=0, gpu="none",
                                    cpu_threads=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_launcher(self, fail: bool = False):
        def launch(command: List[str], env: Dict[str, str], log_path: Path) -> int:
            run_dir = log_path.parent
            config = command[command.index("run") + 1]
            seed = int(command[command.index("--seed") + 1])
            self.assertEqual(env["OMP_NUM_THREADS"], "2")
            self.assertNotIn("--epochs", command)
            log_path.write_text("synthetic launcher\n")
            if fail:
                log_path.write_text("AssertionError: data/processed/MATR\n")
                return 1
            shutil.copyfile(REPO_ROOT / config, run_dir / "config_20260101000000.yaml")
            (run_dir / "20260101000000.ckpt").write_bytes(b"synthetic")
            pred = [o + 0.01 for o in SOH_OBS]
            payload = soh_payload(SOH_PAIRS, SOH_OBS, pred, {"RMSE": 0.01, "MAE": 0.01}, seed)
            (run_dir / f"predictions_seed_{seed}_20260101000000.pkl").write_bytes(pickle.dumps(payload))
            return 0
        return launch

    def test_run_accepts_and_resume_skips_accepted_tasks(self) -> None:
        self.args.max_tasks = 2
        fx3.stage_run(self.campaign, self.args, self.plan, launcher=self._fake_launcher())
        state = fx3.load_state(self.campaign)
        accepted = [t for t, e in state["tasks"].items() if e["status"] == fx3.ACCEPTED]
        self.assertEqual(len(accepted), 2)
        first = state["tasks"][accepted[0]]["attempts"][0]
        manifest = json.loads((Path(first["run_dir"]) / "run.json").read_text())
        self.assertEqual((manifest["status"], manifest["return_code"], manifest["smoke"]),
                         ("COMPLETED", 0, False))
        self.assertTrue((Path(first["run_dir"]) / "fig3_task_record.json").is_file())
        todo = fx3.runnable(self.plan, state, retry_failed=False)
        self.assertEqual(len(todo), len(self.plan["tasks"]) - 2)
        self.assertNotIn(accepted[0], [t["task_id"] for t in todo])

    def test_shared_data_failure_stops_the_group_and_is_kept(self) -> None:
        fx3.stage_run(self.campaign, self.args, self.plan, launcher=self._fake_launcher(fail=True))
        state = fx3.load_state(self.campaign)
        failed = [e for e in state["tasks"].values() if e["status"] == fx3.FAILED]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["failure_class"], "shared_data")
        self.assertTrue((Path(failed[0]["attempts"][0]["run_dir"]) / "output.log").is_file())
        self.assertEqual(sum(1 for e in state["tasks"].values() if e["status"] == fx3.PLANNED),
                         len(self.plan["tasks"]) - 1)

    def test_interrupted_run_is_marked_and_rerun_with_new_id(self) -> None:
        def interrupted(command, env, log_path):
            raise KeyboardInterrupt
        state = fx3.load_state(self.campaign)
        task = self.plan["tasks"][0]
        with self.assertRaises(KeyboardInterrupt):
            fx3.execute_task(self.campaign, task, state, device="cpu", extra_env={},
                             launcher=interrupted)
        state = fx3.load_state(self.campaign)
        entry = state["tasks"][task["task_id"]]
        self.assertEqual(entry["status"], fx3.PLANNED)
        manifest = json.loads((Path(entry["attempts"][0]["run_dir"]) / "run.json").read_text())
        self.assertEqual(manifest["status"], "INTERRUPTED")
        # A runner killed hard leaves RUNNING behind; the next one recovers it.
        entry["status"] = fx3.RUNNING
        fx3.save_state(self.campaign, state)
        self.assertEqual(fx3.recover_interrupted(self.campaign, state), [task["task_id"]])

    def test_lock_refuses_a_live_runner(self) -> None:
        lock = self.campaign / ".runner.lock"
        lock.write_text(str(os.getpid()))
        with self.assertRaises(fx3.CampaignError):
            with fx3.Lock(lock):
                pass
        lock.write_text("99999999")
        with fx3.Lock(lock):
            self.assertEqual(lock.read_text(), str(os.getpid()))
        self.assertFalse(lock.exists())

    def test_selection_carries_provenance_fields_and_drops(self) -> None:
        state = fx3.load_state(self.campaign)
        tid = self.plan["tasks"][0]["task_id"]
        state["tasks"][tid].update(status=fx3.ACCEPTED, artifact="runs/a.pkl")
        selection = fx3.build_selection(self.plan, state, sidecars={tid: "id.json"}, drop={},
                                        measurement=self.tmp / "m")
        entry = selection["runs"][0]
        self.assertEqual(entry["identity_sidecar"], "id.json")
        self.assertIn(entry["determinism"], ("deterministic", "seeded"))
        self.assertTrue(selection["latency_csv"].endswith("latency.csv"))
        dropped = fx3.build_selection(self.plan, state, sidecars={}, drop={tid: "identical"},
                                      measurement=None)
        self.assertEqual(dropped["runs"], [])
        self.assertNotIn("latency_csv", dropped)

    def test_numbered_outputs_never_collide(self) -> None:
        first = fx3.next_numbered(self.tmp / "sel", "fig3_selection", ".json")
        first.write_text("{}")
        self.assertNotEqual(first, fx3.next_numbered(self.tmp / "sel", "fig3_selection", ".json"))


if __name__ == "__main__":
    unittest.main()


class TestMeasureEndToEnd(unittest.TestCase):
    """fig3_measure on synthetic SOH artifacts in a 1-thread child process."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="synthetic_measure_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _artifact(self, name: str, alpha: float) -> Path:
        import torch
        from sklearn.linear_model import Ridge

        from batteryml.data.databundle import DataBundle
        from batteryml.data.sample_index import SampleIndex
        from batteryml.data.transformation.z_score import ZScoreDataTransformation

        generator = torch.Generator().manual_seed(0)
        train_x = torch.rand(40, 8, generator=generator)
        test_x = torch.rand(36, 8, generator=generator)
        train_y = 0.8 + 0.2 * train_x[:, 0]
        test_y = 0.8 + 0.2 * test_x[:, 0]
        index = SampleIndex.concat([SampleIndex.for_cell(f"synthetic_c{c}", list(range(10, 100, 10)))
                                    for c in range(4)])
        bundle = DataBundle(train_x, train_y, test_x, test_y,
                            feature_transformation=ZScoreDataTransformation(),
                            label_transformation=ZScoreDataTransformation(),
                            test_sample_index=index)
        model = Ridge(alpha=alpha).fit(bundle.train_data.feature.numpy(),
                                       bundle.train_data.label.numpy())
        run_dir = self.tmp / name
        run_dir.mkdir()
        (run_dir / "config_20260101000000.yaml").write_text(
            "model:\n  name: RidgeRULPredictor\n  alpha: 1.0\n"
            "label:\n  name: CycleSOHLabelAnnotator\n"
            "train_test_split:\n  name: MATRPrimaryTestTrainTestSplitter\n")
        (run_dir / "20260101000000.ckpt").write_bytes(pickle.dumps(model))
        (run_dir / "latest.ckpt").write_bytes(pickle.dumps(model))
        prediction = torch.from_numpy(model.predict(bundle.test_data.feature.numpy())).float()
        payload = {"prediction": prediction, "data": bundle, "seed": 0,
                   "scores": {"RMSE": 0.02, "MAE": 0.01 + alpha / 1000},
                   "sample_index": index.to_records(),
                   "task_metadata": {"task": "SOH estimation",
                                     "label_annotator": "CycleSOHLabelAnnotator"}}
        artifact = run_dir / "predictions_seed_0_20260101000000.pkl"
        artifact.write_bytes(pickle.dumps(payload))
        return artifact

    def test_measure_writes_raw_timings_sizes_and_radar(self) -> None:
        import subprocess

        entries = [{"entry_id": f"soh-{m}", "model": m, "artifact": str(self._artifact(m, a))}
                   for m, a in (("ridge", 1.0), ("lasso", 5.0))]
        spec = {"protocol": {"device": "cpu", "threads": 1, "batch_size": 8, "warmup": 2,
                             "repeats": 5}, "reference_model": "ridge", "entries": entries}
        spec_path = self.tmp / "spec.json"
        spec_path.write_text(json.dumps(spec))
        out = self.tmp / "measure_001"
        env = {**os.environ, **fx3.thread_env(1), "CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run([sys.executable, "-m", "scripts.paper_figures.fig3_measure",
                               "--spec", str(spec_path), "--output", str(out)],
                              cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        raw = _read(out / "latency_raw.csv")
        self.assertEqual(len(raw), 10)
        self.assertAlmostEqual(float(raw[0]["elapsed_ms"]), int(raw[0]["elapsed_ns"]) / 1e6)
        latency = _read(out / "latency.csv")
        self.assertEqual(panel_d.validate_latency_rows(latency, ["ridge", "lasso"]), [])
        self.assertEqual({r["latency_unit"] for r in latency}, {"ms_per_batch"})
        files = _read(out / "inference_files.csv")
        roles = {(r["model"], r["role"]) for r in files}
        self.assertIn(("ridge", "model checkpoint"), roles)
        self.assertIn(("ridge", "fitted label transformation"), roles)
        self.assertFalse(any(r["path"].endswith("latest.ckpt") for r in files))
        meta = json.loads((out / "measurement_metadata.json").read_text())
        self.assertEqual(len(meta["batch_keys"]), 8)
        self.assertLess(meta["checks"]["soh-ridge"]["max_abs_diff_vs_saved"], 1e-5)
        rows = _read(out / "radar_metrics.csv")
        status, problems, scored = panel_c.validate_radar_inputs(
            rows, json.loads((out / "radar_config.json").read_text()), ["ridge", "lasso"])
        self.assertEqual((status, problems), (common.CREATED, []))
        self.assertEqual({round(r["score"], 9) for r in scored if r["model"] == "ridge"}, {0.5})
