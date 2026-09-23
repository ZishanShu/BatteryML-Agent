"""The result analysis: its facts, its gate, and what its prompt asks for.

The explanation this replaces was a narrator. Its prompt carried the
dataset, task, model, metrics and a file list, and then asked the model to
"state which model/dataset/task ran ... restate the metrics above exactly
... list the generated artifacts". Every test here is about the difference
between that and an analysis: are the diagnostics present, is the benchmark
gated, and does the instruction ask for interpretation and next steps.

No paid calls: the LLM boundary is a stub that records what it was sent.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent import result_analysis as ra  # noqa: E402
from batteryml_web.agent.result_agent import explain_run_result  # noqa: E402
from batteryml_web.core import benchmarks as bm  # noqa: E402

RUL = "RUL prediction"
SOH = "SOH estimation"

MATR1_SPLITTER = "MATRPrimaryTestTrainTestSplitter"


def _context(**overrides: Any) -> Dict[str, Any]:
    base = {
        "dataset": "MATR",
        "task": RUL,
        "model": "ridge",
        "config_path": "configs/baselines/sklearn/ridge/matr_1.yaml",
        "workspace": "runs/web_projects/alpha/run_x",
        "run_status": "COMPLETED",
        "success": True,
        "metrics": {"RMSE": 123.456789, "MAE": 90.1, "MAPE": 0.1234},
        "warnings": [],
    }
    base.update(overrides)
    return base


def _analysis(
    *,
    task: str = RUL,
    model: str = "ridge",
    splitter: str = MATR1_SPLITTER,
    metrics: Dict[str, Any] | None = None,
    is_smoke: bool = False,
    diagnostics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """An analysis context built without touching the filesystem."""
    context = _context(task=task, model=model)
    if metrics is not None:
        context["metrics"] = metrics
    analysis = ra.build_analysis_context(
        context,
        manifest={"seed": 0, "device": "cpu", "command": []},
        run_id="run_20260101_000000_ridge_matr_seed0_aaaaaa",
        run_dir="runs/web_projects/alpha/run_x",
        project_name="alpha",
        identity_key="alpha|run_x|pred.pkl|1-2",
        is_smoke=is_smoke,
        read_diagnostics=False,
    )
    # The protocol is read from a config file that does not exist in this
    # fixture, so it is injected here: what is under test is the gate, not
    # yaml parsing (which has its own test below against a real config).
    analysis["protocol"]["splitter"] = splitter
    analysis["benchmark"] = ra._benchmark_block(
        bm.assess_comparability(
            task=task,
            splitter_name=splitter,
            model_key=model,
            metric_name="RMSE" if "RMSE" in (analysis_metrics := {
                row["name"] for row in analysis["metrics"]
            }) else None,
            metric_value=next(
                (
                    float(row["value"])
                    for row in analysis["metrics"]
                    if row["name"] == "RMSE"
                ),
                None,
            ),
            unit=ra.metric_unit(task, "RMSE") if "RMSE" in analysis_metrics else None,
            is_smoke=is_smoke,
        )
    )
    if diagnostics is not None:
        analysis["diagnostics"] = diagnostics
    return analysis


# --------------------------------------------------------------------------
# A. the deterministic context (requirements 1-4)
# --------------------------------------------------------------------------


class TestTheContextCarriesTheFacts(unittest.TestCase):
    def test_the_selected_runs_metrics_are_included_with_units(self) -> None:
        """Requirement 1, plus the unit.

        "RMSE 0.081" is not a fact until it says whether that is cycles or
        a capacity ratio; the two differ by three orders of magnitude, and
        the confusion between them is what this whole step guards against.
        """
        analysis = _analysis()
        rows = {row["name"]: row for row in analysis["metrics"]}
        self.assertEqual(rows["RMSE"]["value"], 123.456789)
        self.assertEqual(rows["RMSE"]["unit"], "cycles")
        self.assertEqual(rows["MAPE"]["unit"], "ratio (multiply by 100 for percent)")

    def test_soh_metrics_are_ratios_not_cycles(self) -> None:
        analysis = _analysis(task=SOH, model="ridge", metrics={"MAE": 0.05, "RMSE": 0.081})
        rows = {row["name"]: row for row in analysis["metrics"]}
        self.assertEqual(rows["RMSE"]["unit"], "SOH ratio")
        self.assertEqual(rows["MAE"]["unit"], "SOH ratio")

    def test_diagnostics_are_included(self) -> None:
        """Requirement 2, through the real artifact reader."""
        analysis = _analysis(
            diagnostics={
                "read": True,
                "readable": True,
                "predicted": {
                    "count": 42, "finite": 42, "non_finite": 0, "distinct": 1,
                    "constant": True, "min": 500.0, "max": 500.0,
                },
                "observed": {
                    "count": 42, "finite": 42, "non_finite": 0, "distinct": 40,
                    "constant": False, "min": 300.0, "max": 1100.0,
                },
                "sample_count": 42,
                "distinct_cells": 42,
                "task_metadata": {"label_annotator": "RULLabelAnnotator"},
                "saved_scores": {},
            }
        )
        prompt = ra.build_result_analysis_prompt(analysis)
        self.assertIn("distinct=1", prompt)
        self.assertIn("constant=True", prompt)
        self.assertIn("non_finite=0", prompt)
        self.assertIn("RULLabelAnnotator", prompt)

    def test_smoke_status_is_included(self) -> None:
        """Requirement 3."""
        analysis = _analysis(is_smoke=True)
        self.assertTrue(analysis["identity"]["is_smoke"])
        self.assertIn("SMOKE RUN", ra.build_result_analysis_prompt(analysis))

    def test_smoke_is_derived_from_the_executed_command(self) -> None:
        """``--epochs 1`` is where a smoke run is actually recorded."""
        self.assertTrue(ra._smoke_from_command(["x", "--epochs", "1"]))
        self.assertFalse(ra._smoke_from_command(["x", "--epochs", "50"]))
        self.assertFalse(ra._smoke_from_command(["x"]))
        # A trailing --epochs with no value must not crash or be read as 1.
        self.assertFalse(ra._smoke_from_command(["x", "--epochs"]))

    def test_the_selected_result_identity_is_preserved(self) -> None:
        """Requirement 4. The explanation is anchored to one result."""
        analysis = _analysis()
        identity = analysis["identity"]
        self.assertEqual(identity["run_id"], "run_20260101_000000_ridge_matr_seed0_aaaaaa")
        self.assertEqual(identity["project"], "alpha")
        self.assertEqual(analysis["provenance"]["identity_key"], "alpha|run_x|pred.pkl|1-2")
        prompt = ra.build_result_analysis_prompt(analysis)
        self.assertIn("run_20260101_000000_ridge_matr_seed0_aaaaaa", prompt)

    def test_the_protocol_is_read_from_the_real_config(self) -> None:
        """The split class is what decides MATR1 vs MATR2, so it is read."""
        protocol = ra.read_protocol(
            REPO_ROOT / "configs/baselines/sklearn/ridge/matr_1.yaml"
        )
        self.assertTrue(protocol["readable"])
        self.assertEqual(protocol["splitter"], MATR1_SPLITTER)
        self.assertEqual(protocol["label_annotator"], "RULLabelAnnotator")
        self.assertEqual(
            protocol["feature_extractor"], "VoltageCapacityMatrixFeatureExtractor"
        )

    def test_matr_2_config_reads_as_the_secondary_split(self) -> None:
        protocol = ra.read_protocol(
            REPO_ROOT / "configs/baselines/sklearn/ridge/matr_2.yaml"
        )
        self.assertEqual(protocol["splitter"], "MATRSecondaryTestTrainTestSplitter")
        self.assertEqual(bm.subset_for_splitter(protocol["splitter"]), "MATR2")

    def test_a_missing_config_is_reported_not_guessed(self) -> None:
        protocol = ra.read_protocol("configs/does/not/exist.yaml")
        self.assertFalse(protocol["readable"])
        self.assertIsNone(protocol["splitter"])


# --------------------------------------------------------------------------
# C. comparability (requirements 10-15)
# --------------------------------------------------------------------------


class TestComparability(unittest.TestCase):
    def test_a_full_matching_run_is_comparable(self) -> None:
        """Requirement 10."""
        verdict = bm.assess_comparability(
            task=RUL, splitter_name=MATR1_SPLITTER, model_key="ridge",
            metric_name="RMSE", metric_value=130.0, unit="cycles",
            is_smoke=False,
        )
        self.assertTrue(verdict.comparable)
        self.assertEqual(verdict.reasons, ())
        self.assertAlmostEqual(verdict.difference, 130.0 - 116.0)
        self.assertIn("Directly comparable", verdict.explain())

    def test_soh_is_never_compared_with_a_rul_reference(self) -> None:
        """Requirement 11 — the rule that matters most.

        SOH RMSE is a capacity ratio of order 0.1; these references are
        cycle counts of order 100. Printing "116 cycles" beside "0.081"
        would invite exactly the comparison that means nothing.
        """
        verdict = bm.assess_comparability(
            task=SOH, splitter_name=MATR1_SPLITTER, model_key="ridge",
            metric_name="RMSE", metric_value=0.081, unit="SOH ratio",
            is_smoke=False,
        )
        self.assertFalse(verdict.comparable)
        self.assertIn(bm.REASON_TASK, verdict.reasons)
        self.assertIsNone(verdict.difference)

    def test_a_different_metric_is_not_comparable(self) -> None:
        """Requirement 12."""
        verdict = bm.assess_comparability(
            task=RUL, splitter_name=MATR1_SPLITTER, model_key="ridge",
            metric_name="MAE", metric_value=90.0, unit="cycles",
            is_smoke=False,
        )
        self.assertFalse(verdict.comparable)
        self.assertIn(bm.REASON_METRIC, verdict.reasons)

    def test_a_different_split_is_not_comparable(self) -> None:
        """Requirement 13."""
        verdict = bm.assess_comparability(
            task=RUL, splitter_name="RandomTrainTestSplitter", model_key="ridge",
            metric_name="RMSE", metric_value=130.0, unit="cycles",
            is_smoke=False,
        )
        self.assertFalse(verdict.comparable)
        self.assertIn(bm.REASON_SUBSET, verdict.reasons)

    def test_a_smoke_run_is_not_directly_comparable(self) -> None:
        """Requirement 14. One epoch is not trained performance."""
        verdict = bm.assess_comparability(
            task=RUL, splitter_name=MATR1_SPLITTER, model_key="cnn",
            metric_name="RMSE", metric_value=800.0, unit="cycles",
            is_smoke=True,
        )
        self.assertFalse(verdict.comparable)
        self.assertIn(bm.REASON_SMOKE, verdict.reasons)
        self.assertIsNone(verdict.difference)
        # The reference still travels: worth naming, not worth subtracting.
        self.assertIsNotNone(verdict.record)

    def test_an_ambiguous_matr_split_is_not_comparable(self) -> None:
        """Requirement 15. "MATR" alone does not choose MATR1."""
        verdict = bm.assess_comparability(
            task=RUL, splitter_name="MATRTrainTestSplitter", model_key="ridge",
            metric_name="RMSE", metric_value=130.0, unit="cycles",
            is_smoke=False,
        )
        self.assertFalse(verdict.comparable)
        self.assertIn(bm.REASON_SUBSET, verdict.reasons)
        self.assertIsNone(verdict.record)

    def test_a_model_with_no_published_row_is_not_comparable(self) -> None:
        verdict = bm.assess_comparability(
            task=RUL, splitter_name=MATR1_SPLITTER, model_key="cptransformer",
            metric_name="RMSE", metric_value=130.0, unit="cycles",
            is_smoke=False,
        )
        self.assertFalse(verdict.comparable)
        self.assertIn(bm.REASON_MODEL, verdict.reasons)
        self.assertIsNone(verdict.record)
        self.assertIn("no directly corresponding published", verdict.explain())

    def test_a_censored_reference_yields_no_difference(self) -> None:
        verdict = bm.assess_comparability(
            task=RUL, splitter_name="HUSTTrainTestSplitter", model_key="ridge",
            metric_name="RMSE", metric_value=900.0, unit="cycles",
            is_smoke=False,
        )
        self.assertFalse(verdict.comparable)
        self.assertIn(bm.REASON_CENSORED, verdict.reasons)
        self.assertIsNone(verdict.difference)

    def test_every_failure_is_reported_not_just_the_first(self) -> None:
        """A run can be wrong in several ways at once, and saying so is
        cheaper than three rounds of correcting one thing each.

        ``model_not_published`` is deliberately *absent* here: on a task the
        registry does not cover, "this model has no published baseline"
        would imply the model would have had one on RUL. The task is the
        reason, and it is the only honest one.
        """
        verdict = bm.assess_comparability(
            task=SOH, splitter_name="RandomTrainTestSplitter",
            model_key="cptransformer", metric_name="MAE",
            metric_value=0.05, unit="SOH ratio", is_smoke=True,
        )
        for code in (
            bm.REASON_TASK, bm.REASON_SUBSET, bm.REASON_METRIC, bm.REASON_SMOKE,
        ):
            self.assertIn(code, verdict.reasons)
        self.assertNotIn(bm.REASON_MODEL, verdict.reasons)

    def test_an_soh_run_is_given_no_rul_record_at_all(self) -> None:
        """§11, at the level that actually protects the user.

        A ridge/MATR1 SOH run resolves the *same* model and the *same*
        split as the published ridge/MATR1 RUL row, so a lookup that
        ignored the task would find it and the page would print "116
        cycles" beside an SOH RMSE of 0.081. Marking the pair "not
        comparable" does not undo that: the two numbers are three orders of
        magnitude apart and showing them together is itself the error.
        """
        verdict = bm.assess_comparability(
            task=SOH, splitter_name=MATR1_SPLITTER, model_key="ridge",
            metric_name="RMSE", metric_value=0.081, unit="SOH ratio",
            is_smoke=False,
        )
        self.assertIsNone(verdict.record)
        # Non-vacuity: the identical lookup on the RUL task does find it.
        self.assertIsNotNone(
            bm.assess_comparability(
                task=RUL, splitter_name=MATR1_SPLITTER, model_key="ridge",
                metric_name="RMSE", metric_value=130.0, unit="cycles",
                is_smoke=False,
            ).record
        )


# --------------------------------------------------------------------------
# D. prompt behaviour (requirements 16-21)
# --------------------------------------------------------------------------


class _StubLLM:
    """Records the prompt, answers with a fixed analysis."""

    def __init__(self, reply: Dict[str, Any] | None = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.reply = reply or {
            "summary": "s", "interpretation": "i",
            "diagnostics": ["d"], "benchmark": "b", "next_steps": ["n"],
        }

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return json.dumps(self.reply)


class TestThePromptBehaviour(unittest.TestCase):
    def test_a_benchmark_value_reaches_the_prompt_only_from_the_registry(self) -> None:
        """Requirement 16."""
        analysis = _analysis(metrics={"RMSE": 130.0})
        prompt = ra.build_result_analysis_prompt(analysis)
        self.assertIn("116", prompt)
        self.assertIn("MATR1", prompt)
        self.assertIn(bm.BENCHMARK_SOURCE, prompt)

    def test_with_no_reference_the_prompt_forbids_inventing_one(self) -> None:
        """Requirement 17."""
        analysis = _analysis(task=SOH, model="ridge", metrics={"RMSE": 0.081})
        prompt = ra.build_result_analysis_prompt(analysis)
        self.assertIn("NO reference supplied", prompt)
        self.assertIn("must not supply one from your own knowledge", prompt)
        # And no cycle count from the RUL table leaked in beside the ratio.
        self.assertNotIn("116", prompt)

    def test_the_system_prompt_forbids_recalled_benchmarks(self) -> None:
        self.assertIn("must not recall", ra.ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("ONLY a reference supplied", ra.ANALYSIS_SYSTEM_PROMPT)

    def test_a_non_comparable_reference_forbids_arithmetic(self) -> None:
        analysis = _analysis(is_smoke=True, metrics={"RMSE": 800.0})
        prompt = ra.build_result_analysis_prompt(analysis)
        self.assertIn("NOT DIRECTLY COMPARABLE", prompt)
        self.assertIn("Do not compute any difference or percentage", prompt)

    def test_the_prompt_separates_fact_from_hypothesis(self) -> None:
        """Requirement 19."""
        for marker in ("FACT", "DERIVED", "HYPOTHESIS"):
            self.assertIn(marker, ra.ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("Never state a cause as established", ra.ANALYSIS_SYSTEM_PROMPT)

    def test_recommendations_are_requested_and_grounded(self) -> None:
        """Requirement 20."""
        prompt = ra.build_result_analysis_prompt(_analysis())
        self.assertIn("next_steps", prompt)
        self.assertIn("2 to 4 concrete experiments", prompt)
        self.assertIn("tune the", prompt)  # named as the thing to avoid

    def test_no_best_model_language_is_requested(self) -> None:
        """Requirement 21. One run is not a comparison."""
        self.assertIn("Do not call any model 'best'", ra.ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("one run is not a", ra.ANALYSIS_SYSTEM_PROMPT)

    def test_the_prompt_asks_for_interpretation_not_narration(self) -> None:
        """The defect this step exists to fix.

        The old instruction was "state which model/dataset/task ran ...
        restate the metrics above exactly ... list the generated artifacts".
        """
        prompt = ra.build_result_analysis_prompt(_analysis())
        self.assertIn("Do not narrate the configuration back", prompt)
        self.assertIn("interpretation", prompt)
        self.assertIn("diagnostics", prompt)
        self.assertNotIn("list the generated artifacts", prompt)

    def test_the_analysis_context_selects_the_analysis_prompt(self) -> None:
        """Requirement 18's mechanism: one call, the richer prompt."""
        stub = _StubLLM()
        analysis = _analysis()
        explain_run_result(
            _context(),
            analysis_context=analysis,
            base_url="u", api_key="k", model="m", llm_call=stub,
        )
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0]["system_prompt"], ra.ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("Do not narrate the configuration back", stub.calls[0]["prompt"])

    def test_without_an_analysis_context_the_old_prompt_still_runs(self) -> None:
        """A caller that supplies no evidence is not broken by this step."""
        stub = _StubLLM({"explanation": "x"})
        explain_run_result(
            _context(), base_url="u", api_key="k", model="m", llm_call=stub
        )
        self.assertEqual(len(stub.calls), 1)
        self.assertNotEqual(stub.calls[0]["system_prompt"], ra.ANALYSIS_SYSTEM_PROMPT)

    def test_the_run_metrics_survive_the_reply(self) -> None:
        """Requirement 18. Prose is never the source of a number."""
        stub = _StubLLM({"summary": "s", "metrics": {"RMSE": 0.0}})
        parsed = explain_run_result(
            _context(),
            analysis_context=_analysis(),
            base_url="u", api_key="k", model="m", llm_call=stub,
        )
        self.assertEqual(parsed["metrics"]["RMSE"], 123.456789)

    def test_the_analysis_context_is_echoed_back_unchanged(self) -> None:
        """So the panel and the prompt cannot show different references."""
        stub = _StubLLM()
        analysis = _analysis()
        parsed = explain_run_result(
            _context(),
            analysis_context=analysis,
            base_url="u", api_key="k", model="m", llm_call=stub,
        )
        self.assertIs(parsed["analysis_context"], analysis)


class TestRendering(unittest.TestCase):
    def test_empty_sections_are_omitted_not_printed_empty(self) -> None:
        """A heading with nothing under it is the filler this removes."""
        rendered = ra.render_analysis_markdown(
            {"summary": "ran ridge", "interpretation": "", "diagnostics": [],
             "benchmark": "", "next_steps": ["try matr_2"]}
        )
        self.assertIn("Result summary", rendered)
        self.assertIn("Recommended next steps", rendered)
        self.assertNotIn("Interpretation", rendered)
        self.assertNotIn("Benchmark comparison", rendered)

    def test_an_empty_reply_says_so(self) -> None:
        self.assertIn("no analysis content", ra.render_analysis_markdown({}))

    def test_the_derived_result_context_matches_the_analysis(self) -> None:
        analysis = _analysis()
        derived = ra.result_context_from_analysis(analysis)
        self.assertEqual(derived["task"], RUL)
        self.assertEqual(derived["model"], "ridge")
        self.assertEqual(derived["metrics"]["RMSE"], 123.456789)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
