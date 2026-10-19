"""Traceable LLM explanation records and their review sheet (EXPL-01, EXPL-02).

No provider is contacted: ``requests.post`` is replaced with a fake that
returns a recorded-looking chat-completions body. The API key is a made-up
string whose only purpose is to prove it never reaches a file.

    python -m unittest tests.test_explanation_record -v
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent import explanation_checklist as ec  # noqa: E402
from batteryml_web.agent import explanation_record as er  # noqa: E402
from batteryml_web.agent import llm_client  # noqa: E402
from batteryml_web.agent.result_agent import explain_run_result  # noqa: E402
from batteryml_web.agent.result_analysis import (  # noqa: E402
    build_analysis_context,
    result_context_from_analysis,
)

REAL_RUN_LLM_PROMPT = llm_client.run_llm_prompt
SYNTHETIC_KEY = "sk-synthetic-NEVERWRITE-0123456789abcdef0123"
CREDS = {"base_url": "https://llm.example.invalid/v1", "api_key": SYNTHETIC_KEY, "model": "requested-model-x"}


class _Response:
    def __init__(self, status: int, body: Any, text: str | None = None) -> None:
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self) -> Any:
        return self._body


def _chat_body(content: str, *, model: str = "served-model-x-2026-01") -> Dict[str, Any]:
    return {
        "id": "chatcmpl-synthetic-1",
        "model": model,
        "created": 1760000000,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020},
    }


def _analysis() -> Dict[str, Any]:
    """A RUL analysis context with metrics only (no artifact read)."""
    result_context = {
        "dataset": "MATR", "task": "RUL prediction", "model": "xgb",
        "config_path": "configs/baselines/sklearn/xgb/matr_1.yaml",
        "run_status": "COMPLETED", "success": True,
        "metrics": {"RMSE": 103.84571838378906, "MAE": 78.21234130859375, "MAPE": 0.0812345678},
        "warnings": [],
    }
    return build_analysis_context(
        result_context, manifest={"seed": 0, "device": "cpu", "command": []},
        run_id="run_synthetic_xgb", run_dir="", project_name="synthetic_project",
        read_diagnostics=False,
    )


#: A synthetic answer exercising every automatic check. Not a model output.
SYNTHETIC_ANSWER = {
    "summary": "XGBoost on MATR reached an RMSE of 103.84571838378906 cycles and a MAPE of about 8.1%.",
    "interpretation": (
        "The MAE of 78.2 cycles is the typical error. "
        "RUL here is the number of cycles until the capacity falls to 80% of nominal. "
        "The error is due to lithium plating in fast-charged LFP cells. "
        "This model will generalize to other chemistries."
    ),
    "diagnostics": ["MAPE equals 0.08% of the label.", "A spread of 412 cycles was observed."],
    "benchmark": "",
    "next_steps": ["Run seeds 1 and 2 to measure the seed spread."],
}


class TestClientExchange(unittest.TestCase):
    def test_the_exchange_holds_what_was_sent_and_received_but_no_header(self) -> None:
        sent: Dict[str, Any] = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent.update({"url": url, "headers": headers, "json": json})
            return _Response(200, _chat_body("hello"))

        exchange: Dict[str, Any] = {}
        with mock.patch.object(llm_client.requests, "post", fake_post):
            reply = REAL_RUN_LLM_PROMPT("user text", system_prompt="sys text", exchange=exchange, **CREDS)
        self.assertEqual(reply, "hello")
        self.assertEqual(exchange["request_body"], sent["json"])                 # exactly the body sent
        self.assertEqual(exchange["request_body"]["temperature"], 0.2)
        self.assertEqual(exchange["response_meta"]["model"], "served-model-x-2026-01")
        self.assertEqual(exchange["response_meta"]["finish_reason"], "stop")
        self.assertEqual(exchange["raw_content"], "hello")
        self.assertEqual(exchange["stage"], "received")
        self.assertNotIn(SYNTHETIC_KEY, json.dumps(exchange))
        self.assertNotIn("headers", exchange)

    def test_a_provider_error_echoing_the_key_is_scrubbed(self) -> None:
        def fake_post(url, headers=None, json=None, timeout=None):
            return _Response(401, {}, text=f"Incorrect API key provided: {SYNTHETIC_KEY}")

        exchange: Dict[str, Any] = {}
        with mock.patch.object(llm_client.requests, "post", fake_post), self.assertRaises(RuntimeError):
            REAL_RUN_LLM_PROMPT("x", exchange=exchange, **CREDS)
        self.assertEqual(exchange["stage"], "http_error")
        self.assertEqual(exchange["http_status"], 401)
        self.assertNotIn(SYNTHETIC_KEY, json.dumps(exchange))

    def test_without_an_exchange_the_request_is_unchanged(self) -> None:
        bodies: List[Any] = []

        def fake_post(url, headers=None, json=None, timeout=None):
            bodies.append(json)
            return _Response(200, _chat_body("ok"))

        with mock.patch.object(llm_client.requests, "post", fake_post):
            REAL_RUN_LLM_PROMPT("p", system_prompt="s", **CREDS)
        self.assertEqual(set(bodies[0]), {"model", "messages", "stream", "temperature"})


class TestRecord(unittest.TestCase):
    def _explain(self, content: str) -> tuple[Dict[str, Any], Dict[str, Any], Any]:
        analysis = _analysis()
        exchange: Dict[str, Any] = {}
        with mock.patch.object(llm_client.requests, "post", lambda *a, **k: _Response(200, _chat_body(content))):
            try:
                parsed = explain_run_result(
                    result_context_from_analysis(analysis), analysis_context=analysis,
                    llm_call=lambda prompt, **kw: REAL_RUN_LLM_PROMPT(prompt, exchange=exchange, **kw), **CREDS,
                )
                error = None
            except ValueError as exc:
                parsed, error = None, str(exc)
        record = er.build_record(
            exchange=exchange, analysis_context=analysis, manifest={"task": "RUL prediction", "seed": 0},
            project_name="synthetic_project", project_root="", run_id="run_synthetic_xgb", run_dir="",
            identity_key="k", predictions_file=None, provider="Custom OpenAI-compatible",
            requested_model=CREDS["model"], repo_root=REPO_ROOT, started_at="t0", finished_at="t1",
            parsed=parsed, rendered="md" if parsed else None, error=error,
        )
        return record, analysis, parsed

    def test_a_record_carries_the_exact_messages_models_and_settings(self) -> None:
        record, analysis, _ = self._explain(json.dumps(SYNTHETIC_ANSWER))
        self.assertEqual(record["outcome"], er.OUTCOME_ANSWER)
        messages = record["request"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("- RMSE = 103.84571838378906 [cycles]", messages[1]["content"])
        self.assertEqual(record["call"]["requested_model"], "requested-model-x")
        self.assertEqual(record["call"]["response_model"], "served-model-x-2026-01")
        self.assertEqual(record["call"]["generation_settings_sent"], {"stream": False, "temperature": 0.2})
        self.assertIn("max_tokens", record["call"]["settings_not_sent_provider_defaults_apply"])
        self.assertEqual(record["response"]["raw_content"], json.dumps(SYNTHETIC_ANSWER))
        self.assertNotIn("result_context", record["response"]["parsed"])
        self.assertEqual(record["human_review"]["status"], "NOT STARTED")
        self.assertTrue(record["code"].get("commit") or record["code"].get("unavailable_reason"))

    def test_an_unparseable_reply_is_kept_as_a_failure(self) -> None:
        record, _, parsed = self._explain("not json at all")
        self.assertIsNone(parsed)
        self.assertEqual(record["outcome"], er.OUTCOME_FAILED)
        self.assertEqual(record["response"]["raw_content"], "not json at all")
        self.assertIn("not valid JSON", record["error"]["message"])

    def test_writing_refuses_a_record_holding_the_key_and_never_overwrites(self) -> None:
        record, _, _ = self._explain(json.dumps(SYNTHETIC_ANSWER))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "r"
            path = er.write_record(record, target, secrets=[SYNTHETIC_KEY])
            self.assertNotIn(SYNTHETIC_KEY, path.read_text(encoding="utf-8"))
            self.assertEqual(er.load_record(path)["record_id"], record["record_id"])
            with self.assertRaises(FileExistsError):
                er.write_record(record, target, secrets=[SYNTHETIC_KEY])
            leaked = json.loads(json.dumps(record))
            leaked["request"]["analysis_context"]["identity"]["project"] = SYNTHETIC_KEY
            with self.assertRaises(er.RecordLeakError):
                er.write_record(leaked, Path(tmp) / "leak", secrets=[SYNTHETIC_KEY])
            self.assertFalse((Path(tmp) / "leak" / er.RECORD_FILENAME).exists())


class TestChecklist(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        record, _, _ = TestRecord()._explain(json.dumps(SYNTHETIC_ANSWER))
        cls.record = record
        cls.sheet = ec.build_checklist(record)
        cls.by_text = {row["model_text"]: row for row in cls.sheet["rows"]}

    def row(self, start: str) -> Dict[str, Any]:
        return next(row for text, row in self.by_text.items() if text.startswith(start))

    def findings(self, start: str) -> str:
        return " ".join(self.row(start)["automatic_findings"])

    def test_human_columns_are_empty_and_nothing_is_marked_approved(self) -> None:
        self.assertTrue(self.sheet["rows"])
        for row in self.sheet["rows"]:
            self.assertEqual(row["human_judgement"], "")
            self.assertEqual(row["author_notes"], "")
            self.assertIn(row["automatic_result"], {ec.FLAG, ec.NOTE, ec.NONE_FOUND})
        self.assertEqual(self.sheet["human_review_status"], "NOT STARTED")
        text = ec.checklist_markdown(self.sheet, self.record).lower()
        self.assertNotIn("approved", text)
        self.assertNotIn("passed", text)

    def test_exact_metric_and_mape_conversion(self) -> None:
        summary = self.findings("XGBoost on MATR")
        self.assertIn("RMSE = 103.84571838378906: matches the sent RMSE exactly", summary)
        self.assertIn("MAPE written as 8.1%", summary)
        self.assertIn("consistent", summary)

    def test_rounding_wrong_percent_and_unknown_numbers_are_flagged(self) -> None:
        self.assertIn("rounded", self.findings("The MAE of 78.2"))
        self.assertIn("does not match", self.findings("MAPE equals 0.08%"))
        self.assertIn("not found in the sent context", self.findings("A spread of 412"))

    def test_task_definition_beyond_the_context_is_flagged(self) -> None:
        row = self.row("RUL here is")
        self.assertEqual(row["automatic_result"], ec.FLAG)
        self.assertIn("does not define the label", self.findings("RUL here is"))
        self.assertIn("not sent to the model", self.findings("RUL here is"))

    def test_mechanism_and_generalisation_are_flagged(self) -> None:
        mechanism = self.findings("The error is due to")
        self.assertIn("lithium plating", mechanism)
        self.assertIn("Labelled as a hypothesis: no", mechanism)
        self.assertIn("Generalisation", self.findings("This model will generalize"))

    def test_csv_and_markdown_round_trip(self) -> None:
        rows = list(csv.DictReader(io.StringIO(ec.checklist_csv(self.sheet))))
        self.assertEqual(len(rows), len(self.sheet["rows"]))
        self.assertEqual(list(rows[0]), list(ec.COLUMNS))
        with tempfile.TemporaryDirectory() as tmp:
            paths = ec.write_checklist(self.record, tmp)
            self.assertTrue(Path(paths["markdown"]).is_file())
            with self.assertRaises(FileExistsError):
                ec.write_checklist(self.record, tmp)

    def test_a_failed_record_gives_an_empty_sheet_that_says_so(self) -> None:
        record, _, _ = TestRecord()._explain("not json")
        sheet = ec.build_checklist(record)
        self.assertEqual(sheet["rows"], [])
        self.assertIn("No answer was parsed", ec.checklist_markdown(sheet, record))


try:
    from tests.test_llm_results_flow import SETTINGS, _ResultsHarness
    from tests.test_quick_experience_fixes import AppTest
except ImportError:  # pragma: no cover
    AppTest = None
    _ResultsHarness = object


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestExplainMetricsWritesARecord(_ResultsHarness, unittest.TestCase):
    """The real button, real explain_run_result and client; a fake HTTP endpoint."""

    def setUp(self) -> None:
        super().setUp()
        from batteryml_web.agent import result_agent

        self.stack.enter_context(mock.patch.object(llm_client, "run_llm_prompt", REAL_RUN_LLM_PROMPT))
        self.stack.enter_context(mock.patch.object(result_agent, "explain_run_result", explain_run_result))
        self.posts: List[Any] = []

        def fake_post(url, headers=None, json=None, timeout=None):
            self.posts.append(json)
            return _Response(200, _chat_body(_json.dumps(SYNTHETIC_ANSWER)))

        _json = json
        self.stack.enter_context(mock.patch.object(llm_client.requests, "post", fake_post))

    def test_one_click_writes_one_record_and_one_sheet(self) -> None:
        at = self.page(**self.connected())
        self.click(at, "review_explain_metrics")
        self.assertEqual(len(self.posts), 1)
        records = er.list_records(self.review_project)
        self.assertEqual(len(records), 1)
        record = er.load_record(records[0])
        self.assertEqual(record["outcome"], er.OUTCOME_ANSWER)
        self.assertEqual(record["call"]["response_model"], "served-model-x-2026-01")
        self.assertEqual(record["request"]["messages"], self.posts[0]["messages"])   # what was sent
        folder = records[0].parent
        self.assertTrue((folder / "review_checklist.md").is_file())
        for path in folder.iterdir():
            self.assertNotIn(SYNTHETIC_KEY, path.read_text(encoding="utf-8-sig"))
        self.assertTrue(any("Explanation record (answer, context v2," in c.value for c in at.caption))
        at.run()                                   # a rerun sends and writes nothing more
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(len(er.list_records(self.review_project)), 1)


if __name__ == "__main__":
    unittest.main()
