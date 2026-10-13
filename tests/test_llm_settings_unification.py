"""One LLM configuration, shared by every surface, spent only on request.

The complaint this answers: the assistant was configured near the start of
the workflow, and pages later "Explain Result" asked for the provider, the
Base URL, the API key and the model all over again. The values were never
actually lost — five surfaces rendered the same form over the same session
keys — but four of those surfaces had no business asking, and one of them
asked on the page where the user least expected it.

What is asserted here, in three groups:

* **The store.** ``ui/llm_state`` owns provider, endpoint, key, model and
  connection state, and is the only thing that may hand the key out. A
  change made once is visible everywhere because there is only one place to
  read; switching project does not touch it; nothing serialisable carries
  the key.
* **The wiring.** Agent explanation and Result explanation call the shared
  client with the shared credentials, and neither constructs its own.
* **The spending.** No LLM call happens because a page rendered. An explicit
  Explain makes exactly one, and a rerun afterwards makes none.

No test here makes a network call. The client is replaced by a stub that
records its arguments, and the one test that exercises response parsing
feeds a canned OpenAI-compatible body through a fake ``requests.post``.
"""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

from batteryml_web.agent import llm_client
from batteryml_web.ui import agent_workflow as ui
from batteryml_web.ui import feedback as ui_feedback
from batteryml_web.ui import llm_state as store
from batteryml_web.ui import settings_panel as ui_settings
from batteryml_web.ui import workflow_sidebar as ui_sidebar

from tests.test_agent_web_ui import FakeStreamlit, _MATR_CONTEXT, _render


CONFIGURED = {
    "llm_provider_preset": "Custom OpenAI-compatible",
    "llm_base_url": "https://maas-openapi.example.invalid/api/v1",
    "llm_api_key": "sk-session-only",
    "llm_model": "glm-5.3-flashx",
}


@contextmanager
def _session(values: Dict[str, Any] | None = None):
    """Point every module that holds its own ``st`` at one fake session.

    Missing one does not raise — it silently reads the real Streamlit's
    empty session state, which makes a wiring bug look like a test bug.
    """
    fake = FakeStreamlit(dict(values or {}))
    modules = (store, ui_settings, ui_sidebar, ui, ui_feedback)
    saved = [(module, module.st) for module in modules]
    try:
        for module in modules:
            module.st = fake
        yield fake
    finally:
        for module, original in saved:
            module.st = original


# --------------------------------------------------------------------------
# 1-6: one store, read the same way everywhere
# --------------------------------------------------------------------------


class TestOneStore(unittest.TestCase):
    def test_configuring_once_is_what_every_surface_reads(self) -> None:
        """Requirements 1 and 2, asserted through the surfaces themselves.

        Not "the keys are equal" — that is true of any two readers of the
        same dict. What matters is that the credentials the Agent page would
        send and the credentials the Review page would send both come out of
        one write, so this drives the write through the settings form's own
        entry point and reads back through the accessor both pages use.
        """
        with _session():
            store.set_llm_settings(
                CONFIGURED["llm_base_url"],
                CONFIGURED["llm_api_key"],
                CONFIGURED["llm_model"],
                provider=CONFIGURED["llm_provider_preset"],
            )
            credentials = store.llm_credentials()
            state = store.get_llm_state()

        self.assertEqual(credentials["base_url"], CONFIGURED["llm_base_url"])
        self.assertEqual(credentials["model"], CONFIGURED["llm_model"])
        self.assertEqual(credentials["api_key"], CONFIGURED["llm_api_key"])
        self.assertEqual(state.provider, CONFIGURED["llm_provider_preset"])
        self.assertEqual(state.model, CONFIGURED["llm_model"])

    def test_the_provider_is_part_of_the_configuration(self) -> None:
        """It used to live beside the form rather than in the store.

        A surface asking "what is configured" could get the endpoint, the
        model and the key but not the provider, so the provider could only
        be read by the one module that rendered the dropdown.
        """
        with _session():
            store.set_llm_settings("u", "k", "m", provider="GLM / Zhipu")
            self.assertEqual(store.get_llm_provider(), "GLM / Zhipu")
            self.assertEqual(store.llm_settings_summary()["provider"], "GLM / Zhipu")

    def test_changing_the_model_once_changes_it_for_every_surface(self) -> None:
        """Requirement 5."""
        with _session():
            store.set_llm_settings(**{
                "base_url": CONFIGURED["llm_base_url"],
                "api_key": CONFIGURED["llm_api_key"],
                "model": "first-model",
            })
            self.assertEqual(store.llm_credentials()["model"], "first-model")
            store.set_llm_settings(
                CONFIGURED["llm_base_url"], CONFIGURED["llm_api_key"], "second-model"
            )
            self.assertEqual(store.llm_credentials()["model"], "second-model")
            self.assertEqual(store.get_llm_state().model, "second-model")
            self.assertEqual(store.llm_settings_summary()["model"], "second-model")

    def test_switching_project_leaves_the_configuration_alone(self) -> None:
        """Requirement 6, and the serialisation half of requirement 4.

        Driven through the real project-creation path rather than by poking
        session keys: what matters is that creating or loading a project
        does not disturb the configuration *and* does not write any of it to
        the project's own state file. The way to know both is to create one
        and read what landed on disk.
        """
        import tempfile

        from batteryml_web import app as web_app

        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeStreamlit(dict(CONFIGURED))
            saved = [(module, module.st) for module in (store, web_app)]
            try:
                for module, _ in saved:
                    module.st = fake
                before = dict(store.llm_credentials())
                web_app._ensure_project("alpha", tmp)
                web_app._ensure_project("beta", tmp)
                after = dict(store.llm_credentials())
            finally:
                for module, original in saved:
                    module.st = original

            self.assertEqual(before, after)
            self.assertEqual(after["model"], CONFIGURED["llm_model"])
            for state_file in Path(tmp).rglob("project_state.json"):
                written = state_file.read_text(encoding="utf-8")
                self.assertNotIn(CONFIGURED["llm_api_key"], written)
                self.assertNotIn(CONFIGURED["llm_base_url"], written)

    def test_an_unchanged_save_keeps_a_verified_connection(self) -> None:
        """Requirement in section 5: Connect's verdict is not thrown away.

        Save used to drop the status to "not tested yet" unconditionally, so
        pressing Save after a successful Connect discarded the one piece of
        evidence the app had actually gathered.
        """
        with _session():
            store.set_llm_settings("u", "k", "m")
            store.set_llm_connection_result(True, "verified")
            self.assertEqual(store.get_llm_status(), store.LLM_STATUS_CONNECTED)
            store.set_llm_settings("u", "k", "m")
            self.assertEqual(store.get_llm_status(), store.LLM_STATUS_CONNECTED)

    def test_changing_the_endpoint_makes_the_connection_unverified(self) -> None:
        """The other half: evidence about the old endpoint is not evidence."""
        with _session():
            store.set_llm_settings("u", "k", "m")
            store.set_llm_connection_result(True, "verified")
            store.set_llm_settings("https://elsewhere.invalid/v1", "k", "m")
            self.assertEqual(store.get_llm_status(), store.LLM_STATUS_CONFIGURED)


# --------------------------------------------------------------------------
# 3-4: the key is shared in session and reaches nothing durable
# --------------------------------------------------------------------------


class TestTheKeyStaysInTheSession(unittest.TestCase):
    SECRET = "sk-must-never-be-written"

    def test_the_key_is_shared_in_session(self) -> None:
        """Requirement 3: one entry, and every caller gets the same key."""
        with _session():
            store.set_llm_settings("u", self.SECRET, "m")
            self.assertEqual(store.llm_credentials()["api_key"], self.SECRET)
            self.assertEqual(store.get_llm_state().api_key, self.SECRET)

    def test_nothing_meant_for_display_or_storage_carries_the_key(self) -> None:
        """Requirement 4, at the boundary the key could actually cross.

        The summary and the fingerprint are what captions, toasts, provenance
        records and the sidebar badge are built from. Both report *whether* a
        key is set and never its value, so nothing derived from them can
        carry it into a manifest, an export or a log.
        """
        with _session():
            store.set_llm_settings("u", self.SECRET, "m", provider="OpenAI")
            summary = store.llm_settings_summary()
            fingerprint = store.llm_settings_fingerprint()

        self.assertTrue(summary["api_key_set"])
        self.assertNotIn(self.SECRET, json.dumps(summary))
        self.assertNotIn(self.SECRET, fingerprint)
        self.assertIn("key", fingerprint)

    def test_only_one_function_hands_the_key_out(self) -> None:
        """The rule that keeps the boundary small enough to check.

        ``llm_credentials`` exists to reach the HTTP client. If a second
        function started returning the key, "where can it go" would stop
        being answerable by reading one function.
        """
        source = (REPO_ROOT / "batteryml_web" / "ui" / "llm_state.py").read_text(
            encoding="utf-8"
        )
        returning_the_key = source.count('"api_key": str(st.session_state')
        self.assertEqual(returning_the_key, 1, source.count("api_key"))

    def test_no_other_module_reads_the_credential_keys(self) -> None:
        """The store is authoritative only if nothing goes around it.

        Four call sites used to read ``st.session_state["llm_base_url"]`` and
        friends by hand to build a client. Each was a place a second copy
        could start, and one of them would raise ``KeyError`` on a page the
        user had not visited yet.
        """
        offenders: List[str] = []
        for path in sorted((REPO_ROOT / "batteryml_web").rglob("*.py")):
            if path.name == "llm_state.py":
                continue
            text = path.read_text(encoding="utf-8")
            for key in store.CREDENTIAL_KEYS:
                if f'session_state["{key}"]' in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {key}")
        # settings_panel renders the form, so it reads the stored values to
        # seed its widgets; that is the store's own UI and uses .get().
        self.assertEqual(offenders, [])


# --------------------------------------------------------------------------
# 7-8: both explanations use the shared client
# --------------------------------------------------------------------------


class _StubClient:
    """Records what it was asked, answers with a fixed body."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self.answer


class TestBothExplanationsUseTheSharedClient(unittest.TestCase):
    """Requirements 7 and 8."""

    RESULT_CONTEXT = {
        "workspace": "runs/web_projects/alpha/agent_matr_batlinet",
        "success": True,
        "metrics": {"RMSE": 1.0, "MAE": 2.0, "MAPE": 0.25},
        "warnings": [],
        "predictions_file": "runs/web_projects/alpha/agent_matr_batlinet/p.pkl",
    }

    def test_agent_result_explanation_sends_the_stored_credentials(self) -> None:
        stub = _StubClient({"summary": "explained"})
        real = ui.explain_run_result
        ui.explain_run_result = stub
        try:
            with _session(dict(CONFIGURED)):
                ui._explain_result(dict(self.RESULT_CONTEXT))
        finally:
            ui.explain_run_result = real

        self.assertEqual(len(stub.calls), 1)
        sent = stub.calls[0]
        self.assertEqual(sent["base_url"], CONFIGURED["llm_base_url"])
        self.assertEqual(sent["api_key"], CONFIGURED["llm_api_key"])
        self.assertEqual(sent["model"], CONFIGURED["llm_model"])

    def test_agent_workflow_explanation_sends_the_stored_credentials(self) -> None:
        stub = _StubClient({"guidance": "narrated"})
        real = ui.explain_workflow_state
        ui.explain_workflow_state = stub
        try:
            with _session(dict(CONFIGURED)):
                ui._explain_workflow("predict MATR lifetime")
        finally:
            ui.explain_workflow_state = real

        self.assertEqual(len(stub.calls), 1)
        sent = stub.calls[0]
        self.assertEqual(sent["base_url"], CONFIGURED["llm_base_url"])
        self.assertEqual(sent["api_key"], CONFIGURED["llm_api_key"])
        self.assertEqual(sent["model"], CONFIGURED["llm_model"])

    def test_the_review_page_builds_no_client_of_its_own(self) -> None:
        """Result explanation reaches the LLM through the one client.

        A source check, because the Review page's explanation lives inside a
        long render function that a unit test cannot enter without a full
        project on disk. What it rules out is the thing that actually went
        wrong elsewhere: credentials assembled at the call site out of raw
        session keys.
        """
        app_code = (REPO_ROOT / "batteryml_web" / "app.py").read_text(encoding="utf-8")
        for key in store.CREDENTIAL_KEYS:
            self.assertNotIn(f'session_state["{key}"]', app_code)
        self.assertIn("llm_credentials()", app_code)
        self.assertIn("run_llm_prompt", app_code)


# --------------------------------------------------------------------------
# 9-13: nothing is spent by rendering
# --------------------------------------------------------------------------


class TestExplanationsAreOnDemand(unittest.TestCase):
    """Requirements 9-13."""

    def setUp(self) -> None:
        self.explain = _StubClient({"summary": "explained"})
        self.guide = _StubClient({"guidance": "narrated"})
        self._real_explain = ui.explain_run_result
        self._real_guide = ui.explain_workflow_state
        ui.explain_run_result = self.explain
        ui.explain_workflow_state = self.guide

    def tearDown(self) -> None:
        ui.explain_run_result = self._real_explain
        ui.explain_workflow_state = self._real_guide

    def _configured_session(self) -> dict:
        session = dict(CONFIGURED)
        session.update(
            {
                "agent_workflow_context": _MATR_CONTEXT,
                "agent_selected_model": "batlinet",
                "project_root": "runs/web_projects/alpha",
                "agent_result_context": {
                    "workspace": "runs/web_projects/alpha/agent_matr_batlinet",
                    "success": True,
                    "metrics": {"RMSE": 1.0, "MAE": 2.0, "MAPE": 0.25},
                    "warnings": [],
                    "predictions_file": "runs/web_projects/alpha/x.pkl",
                },
            }
        )
        return session

    def test_rendering_the_page_calls_nothing(self) -> None:
        """Requirement 9. A page that draws must not spend an API call."""
        session = self._configured_session()
        _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        self.assertEqual(self.explain.calls, [])
        self.assertEqual(self.guide.calls, [])

    def test_analyze_no_longer_explains_on_its_own(self) -> None:
        """Analyze resolved the request and then narrated it, unasked.

        Pressing Analyze is a deliberate act, but it is a request for the
        deterministic resolution — dataset, task, model, readiness — and it
        billed the user for an explanation as well, every time.
        """
        with _session(dict(CONFIGURED)):
            ui._analyze_request("predict MATR lifetime")
        self.assertEqual(self.guide.calls, [])

    def test_explicit_explain_makes_exactly_one_call(self) -> None:
        """Requirement 12."""
        session = self._configured_session()
        _render(
            session,
            clicked={ui.EXPLAIN_BUTTON_LABEL},
            radio_values={ui.COMPUTE_LABEL: "CPU"},
        )
        self.assertEqual(len(self.explain.calls), 1)

    def test_a_rerun_after_explaining_does_not_explain_again(self) -> None:
        """Requirement 13. Streamlit reruns the script constantly.

        The stored text is what the panel shows on every later run, so the
        second render must add no call — otherwise every widget interaction
        on the page would be billed.
        """
        session = self._configured_session()
        _render(
            session,
            clicked={ui.EXPLAIN_BUTTON_LABEL},
            radio_values={ui.COMPUTE_LABEL: "CPU"},
        )
        self.assertEqual(len(self.explain.calls), 1)
        _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        self.assertEqual(len(self.explain.calls), 1)

    def test_the_agent_explanation_panel_is_collapsed(self) -> None:
        """Requirement 10.

        It opened itself, which pushed the metrics it was commenting on off
        the first screen — and it was the LLM's prose that the reader met
        first, above BatteryML's own numbers.
        """
        session = self._configured_session()
        fake = _render(
            session,
            clicked={ui.EXPLAIN_BUTTON_LABEL},
            radio_values={ui.COMPUTE_LABEL: "CPU"},
        )
        fake = _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        states = dict(fake.rec["expander_states"])
        self.assertIn("Agent explanation", states)
        self.assertFalse(
            states["Agent explanation"],
            "the explanation panel must not open by itself",
        )

    def test_the_result_explanation_panel_is_shown_open(self) -> None:
        """Review Results draws the answer directly (user request superseding Requirement 11).

        The user pressed Summarize / Explain to get this text; a collapsed
        expander hid the answer behind a second click. It is drawn once, in a
        bordered block, and still sends nothing when drawn.
        """
        app_code = (REPO_ROOT / "batteryml_web" / "app.py").read_text(encoding="utf-8")
        block = app_code.split("def render_llm_output_panel")[1].split("\ndef ")[0]
        self.assertNotIn("st.expander(", block)
        self.assertIn("st.container(border=True", block)
        self.assertEqual(app_code.count('render_llm_output_panel("review_llm_output"'), 1)


# --------------------------------------------------------------------------
# 14-15: the deterministic platform does not depend on any of this
# --------------------------------------------------------------------------


class TestTheDeterministicPathIsUnaffected(unittest.TestCase):
    def test_with_no_llm_the_agent_still_resolves_and_renders(self) -> None:
        """Requirement 14."""
        session = {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_selected_model": "batlinet",
            "project_root": "runs/web_projects/alpha",
        }
        real = ui.is_llm_configured
        ui.is_llm_configured = lambda: False
        try:
            fake = _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        finally:
            ui.is_llm_configured = real
        rendered = " ".join(
            fake.rec["markdown"]
            + fake.rec["writes"]
            + fake.rec["headings"]
            + fake.rec["captions"]
            + fake.rec["expander_writes"]
        )
        # The backend's own answer: the models it says can run for this
        # dataset and task. None of it comes from an assistant.
        self.assertIn("BatLiNet", rendered)
        self.assertIn("READY", rendered)
        self.assertTrue(fake.rec["buttons"])

    def test_an_llm_failure_leaves_the_result_on_the_page(self) -> None:
        """Requirement 15. The metrics come from BatteryML's saved scores."""

        def _boom(*args: Any, **kwargs: Any):
            raise RuntimeError("LLM API request failed with status 503")

        session = {
            "agent_workflow_context": _MATR_CONTEXT,
            "agent_selected_model": "batlinet",
            "project_root": "runs/web_projects/alpha",
            "agent_result_context": {
                "workspace": "runs/web_projects/alpha/agent_matr_batlinet",
                "success": True,
                "metrics": {"RMSE": 1.0, "MAE": 2.0, "MAPE": 0.25},
                "warnings": [],
                "predictions_file": "runs/web_projects/alpha/x.pkl",
            },
            **CONFIGURED,
        }
        real = ui.explain_run_result
        ui.explain_run_result = _boom
        try:
            _render(
                session,
                clicked={ui.EXPLAIN_BUTTON_LABEL},
                radio_values={ui.COMPUTE_LABEL: "CPU"},
            )
            fake = _render(session, radio_values={ui.COMPUTE_LABEL: "CPU"})
        finally:
            ui.explain_run_result = real

        self.assertIsInstance(session.get("agent_explanation_failure"), dict)
        rendered = " ".join(fake.rec["markdown"] + fake.rec["writes"])
        self.assertIn("RMSE", rendered)

    def test_a_workflow_explanation_failure_is_classified_not_pasted(self) -> None:
        """The failure vocabulary, rather than the raw exception text.

        ``f"LLM guidance failed: {exc}"`` put the provider's response body on
        the page and stored it where an explanation belongs, so a 401 read as
        an explanation of the workflow.
        """

        def _boom(*args: Any, **kwargs: Any):
            raise RuntimeError("LLM API request failed with status 401: bad key")

        real = ui.explain_workflow_state
        ui.explain_workflow_state = _boom
        try:
            with _session(dict(CONFIGURED)) as fake:
                ui._explain_workflow("predict MATR lifetime")
                finding = fake.session_state.get(ui.WORKFLOW_EXPLANATION_FAILURE_KEY)
                self.assertIsInstance(finding, dict)
                self.assertNotIn(ui.WORKFLOW_EXPLANATION_KEY, fake.session_state)
                self.assertNotIn("401", str(fake.session_state.get(ui.WORKFLOW_EXPLANATION_KEY)))
        finally:
            ui.explain_workflow_state = real
        self.assertEqual(finding["category"], "llm_authentication_failed")


# --------------------------------------------------------------------------
# 16 and section 6: the OpenAI-compatible request and response
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


class TestTheOpenAICompatibleCall(unittest.TestCase):
    def _post(self, payload: Dict[str, Any]):
        captured: Dict[str, Any] = {}

        def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return _FakeResponse(payload)

        return captured, fake_post

    def test_only_the_content_is_shown_never_the_reasoning(self) -> None:
        """Requirement 16.

        The verified endpoint returns ``reasoning_content`` beside
        ``content``. The model's working is not its answer, and putting it on
        the page would present scratch reasoning as the assistant's reply.
        """
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "visible answer",
                        "reasoning_content": "hidden reasoning",
                    }
                }
            ]
        }
        captured, fake_post = self._post(payload)
        real = llm_client.requests.post
        llm_client.requests.post = fake_post
        try:
            text = llm_client.run_llm_prompt(
                "hello",
                base_url="https://maas-openapi.example.invalid/api/v1",
                api_key="sk-x",
                model="glm-5.3-flashx",
            )
        finally:
            llm_client.requests.post = real
        self.assertEqual(text, "visible answer")
        self.assertNotIn("hidden reasoning", text)

    def test_reasoning_alone_is_an_empty_answer_not_a_fallback(self) -> None:
        """An empty ``content`` is reported, not quietly replaced.

        Falling back to ``reasoning_content`` would be the one way the
        model's working could still reach the page.
        """
        payload = {
            "choices": [
                {"message": {"content": "", "reasoning_content": "hidden reasoning"}}
            ]
        }
        _captured, fake_post = self._post(payload)
        real = llm_client.requests.post
        llm_client.requests.post = fake_post
        try:
            with self.assertRaises(RuntimeError) as caught:
                llm_client.run_llm_prompt(
                    "hello", base_url="https://h.invalid/v1", api_key="k", model="m"
                )
        finally:
            llm_client.requests.post = real
        self.assertNotIn("hidden reasoning", str(caught.exception))

    def test_the_endpoint_is_the_base_url_plus_chat_completions(self) -> None:
        """Section 6: never ``/api/v1/v1/chat/completions``.

        The Base URL is the API root and already carries the provider's own
        version segment, so nothing is inserted. The three inputs checked are
        the ones a user actually produces: the documented root, the same with
        a trailing slash, and the full URL pasted out of the docs.
        """
        expected = "https://maas-openapi.wanjiedata.com/api/v1/chat/completions"
        for base in (
            "https://maas-openapi.wanjiedata.com/api/v1",
            "https://maas-openapi.wanjiedata.com/api/v1/",
            "https://maas-openapi.wanjiedata.com/api/v1/chat/completions",
        ):
            with self.subTest(base=base):
                self.assertEqual(
                    llm_client._join_base_url(base, llm_client.CHAT_COMPLETIONS_PATH),
                    expected,
                )
                self.assertNotIn("/v1/v1/", expected)

    def test_the_request_body_is_plain_chat_completions(self) -> None:
        """Section 6: ``{model, messages, stream: false}``, nothing exotic.

        ``stream`` is sent explicitly because this client parses one whole
        JSON body; a provider defaulting to streaming would hand back an
        event stream that ``response.json()`` cannot read.
        """
        payload = {"choices": [{"message": {"content": "ok"}}]}
        captured, fake_post = self._post(payload)
        real = llm_client.requests.post
        llm_client.requests.post = fake_post
        try:
            llm_client.run_llm_prompt(
                "hello",
                base_url="https://maas-openapi.example.invalid/api/v1",
                api_key="sk-x",
                model="glm-5.3-flashx",
            )
        finally:
            llm_client.requests.post = real

        body = captured["json"]
        self.assertEqual(body["model"], "glm-5.3-flashx")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["messages"][-1]["role"], "user")
        self.assertEqual(
            set(body) - {"model", "messages", "stream", "temperature"}, set()
        )
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-x")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
