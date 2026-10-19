"""Explain-metrics context and prompt v2 (EXPL-03).

    python -m unittest tests.test_result_analysis_v2 -v
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent import explanation_checklist as ec  # noqa: E402
from batteryml_web.agent import explanation_record as er  # noqa: E402
from batteryml_web.agent import result_analysis_v2 as v2  # noqa: E402
from batteryml_web.agent.result_agent import explain_run_result  # noqa: E402
from batteryml_web.agent.result_analysis import build_analysis_context, result_context_from_analysis  # noqa: E402
from batteryml_web.core import native_check as nc  # noqa: E402

MATR_1 = REPO_ROOT / "configs" / "baselines" / "sklearn" / "xgb" / "matr_1.yaml"
PROMPT_V2_SHA256 = "56c6276e25ddd2feac2711b1aba9ca7ea0beec75ee22d7be8d8f2e3d2ff4b3ac"
#: Synthetic stored scores; not a result of any run.
SYNTHETIC_METRICS = {"RMSE": 111.0, "MAE": 222.0, "MAPE": 0.333}


def _v1() -> dict:
    return build_analysis_context(
        {"dataset": "MATR", "task": "RUL prediction", "model": "xgb", "config_path": str(MATR_1),
         "run_status": "COMPLETED", "success": True, "metrics": dict(SYNTHETIC_METRICS), "warnings": []},
        manifest={"seed": 0, "device": "cpu", "command": ["run"]}, run_id="run_synthetic", run_dir="",
        project_name="synthetic_project", read_diagnostics=False,
    )


class TestPrompt(unittest.TestCase):
    def test_the_system_prompt_is_the_reviewed_text_verbatim(self) -> None:
        self.assertEqual(hashlib.sha256(v2.ANALYSIS_SYSTEM_PROMPT_V2.encode("utf-8")).hexdigest(), PROMPT_V2_SHA256)


class TestWithoutNativeEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = v2.build_analysis_context_v2(_v1(), project_root=None, predictions_file=None,
                                                   lookup_native_report=False)
        cls.prompt = v2.build_result_analysis_prompt_v2(cls.context)

    def test_ui_task_and_scientific_target_are_separate(self) -> None:
        self.assertEqual(self.context["ui_task"], "RUL prediction")
        target = self.context["scientific_target"]
        self.assertEqual((target["name"], target["unit"]), ("total cycle life", "cycles"))
        self.assertEqual(target["label_annotator"], "RULLabelAnnotator")
        self.assertEqual(target["rebuilt_labels_match_stored_targets"], nc.NOT_VERIFIED)

    def test_metrics_are_sent_verbatim_with_units(self) -> None:
        for name, value in SYNTHETIC_METRICS.items():
            self.assertIn(f'"value": {value!r}', self.prompt)
        self.assertEqual(self.context["metric_replay"], {"status": nc.NOT_PERFORMED})

    def test_unknowns_are_null_with_a_status_never_zero(self) -> None:
        diagnostics = self.context["diagnostics"]
        self.assertIsNone(diagnostics["array_scale"])
        self.assertEqual(diagnostics["array_scale_status"], nc.NOT_RECORDED)
        self.assertIsNone(diagnostics["verified_cell_count"])
        self.assertEqual(diagnostics["cell_identity_status"], nc.NOT_VERIFIED)
        self.assertEqual(diagnostics["pairing_status"], nc.NOT_VERIFIED)
        self.assertEqual(self.context["label_transformation"]["configured_order"],
                         ["LogScaleDataTransformation", "ZScoreDataTransformation"])
        self.assertIsNone(self.context["label_transformation"]["fitted_state"])
        self.assertEqual(self.context["label_transformation"]["fitted_state_status"], "NOT_SUPPLIED")

    def test_the_benchmark_is_never_comparable_and_carries_no_difference(self) -> None:
        bench = self.context["benchmark"]
        self.assertEqual(bench["comparability_status"], nc.NOT_VERIFIED)
        self.assertNotIn("difference", json.dumps(bench).replace("No difference", ""))
        self.assertNotIn("COMPARABLE.", self.prompt)
        self.assertEqual(bench["reference"]["dataset_subset"], "MATR1")

    def test_the_v2_prompt_is_selected_for_a_v2_context(self) -> None:
        calls = []

        def stub(prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            return json.dumps({"summary": "FACT: synthetic.", "interpretation": "", "diagnostics": [],
                               "benchmark": "", "next_steps": []})

        explain_run_result(result_context_from_analysis(self.context), analysis_context=self.context,
                           llm_call=stub, base_url="x", api_key="y", model="z")
        self.assertEqual(calls[0]["system_prompt"], v2.ANALYSIS_SYSTEM_PROMPT_V2)
        self.assertIn('"context_version": "v2"', calls[0]["prompt"])

    def test_the_checklist_reads_v2_metrics_and_benchmark(self) -> None:
        record = {"request": {"messages": [{"role": "system", "content": v2.ANALYSIS_SYSTEM_PROMPT_V2},
                                           {"role": "user", "content": self.prompt}],
                              "analysis_context": self.context},
                  "response": {"parsed": {"summary": "FACT: RMSE = 111.0 cycles.",
                                          "benchmark": "The run is better than the published reference."}}}
        sheet = ec.build_checklist(record)
        self.assertIn("RMSE", sheet["sent_metrics"])
        self.assertTrue(sheet["benchmark_supplied"])
        self.assertFalse(sheet["benchmark_comparable"])
        bench_row = next(r for r in sheet["rows"] if r["section"] == "benchmark")
        self.assertEqual(bench_row["automatic_result"], ec.FLAG)


class TestWithNativeEvidence(unittest.TestCase):
    def test_a_report_for_this_exact_file_fills_the_native_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "synthetic_predictions.pkl"
            artifact.write_bytes(b"synthetic bytes")
            sha = nc.sha256_file(artifact)
            report = {
                "files": {"predictions": {"sha256": sha}},
                "metric_replay": {"status": nc.VERIFIED, "pairs": 3, "mape_zero_target_count": 0,
                                  "metrics": {"RMSE": {"stored_score": 111.0, "evaluator_replay": 111.0}},
                                  "target_native_scale": {"present": True, "count": 3, "min": 400.0, "max": 900.0},
                                  "prediction_native_scale": {"present": True, "count": 3, "min": 380.0, "max": 950.0}},
                "array_scale": {"status": nc.VERIFIED, "stored_arrays": "label-transformed (A then B)"},
                "identity": {"status": nc.MATCH, "rebuilt_count": 3},
                "label_transformation": {"stored_state": {"class": "SequentialDataTransformation"}},
                "fitted_state_check": {"status": nc.MATCH},
                "scientific_target": {"rebuilt_labels_match_stored_targets": nc.MATCH},
            }
            context = v2.build_analysis_context_v2(_v1(), project_root=None, predictions_file=str(artifact),
                                                   native_report=report)
            diagnostics = context["diagnostics"]
            self.assertEqual(diagnostics["array_scale_status"], nc.VERIFIED)
            self.assertEqual(diagnostics["verified_cell_count"], 3)
            self.assertEqual(diagnostics["observed_native_scale"]["max"], 900.0)
            self.assertTrue(context["label_transformation"]["metric_inverse_transform_verified"])
            self.assertEqual(context["scientific_target"]["rebuilt_labels_match_stored_targets"], nc.MATCH)
            self.assertEqual(context["benchmark"]["comparability_status"], nc.NOT_VERIFIED)   # still
            checks = {c["check"]: c["status"] for c in context["benchmark"]["checks"]}
            self.assertEqual(checks["same cohort as the published benchmark"], nc.NOT_VERIFIED)

            other = dict(report, files={"predictions": {"sha256": "0" * 64}})
            stale = v2.build_analysis_context_v2(_v1(), project_root=None, predictions_file=str(artifact),
                                                 native_report=other)
            self.assertEqual(stale["native_evidence_status"], "NO_NATIVE_CHECK_REPORT")
            self.assertEqual(stale["diagnostics"]["array_scale_status"], nc.NOT_RECORDED)

    def test_the_record_states_its_context_version(self) -> None:
        context = v2.build_analysis_context_v2(_v1(), project_root=None, predictions_file=None,
                                               lookup_native_report=False)
        record = er.build_record(
            exchange={}, analysis_context=context, manifest={}, project_name="p", project_root="", run_id="r",
            run_dir="", identity_key="k", predictions_file=None, provider="x", requested_model="m",
            repo_root=REPO_ROOT, started_at="t0", finished_at="t1", parsed=None, error="not sent")
        self.assertEqual(record["context_version"], "v2")


if __name__ == "__main__":
    unittest.main()
