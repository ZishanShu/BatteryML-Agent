"""The improved UX stays backend-derived, and the settings stay singular.

Three things these protect:

* the request aid is a *template* filled from this installation, not a
  gallery of curated sentences and not a hardcoded demo string;
* LLM settings live in exactly one place, so entering them once is enough;
* the Files/Predictions views describe artifacts that were actually
  discovered, and a .pkl this platform does not recognise fails gracefully
  rather than being reinterpreted.

The dynamic-contract class at the end is written to fail if anyone later
writes a runtime value — a model count, a GPU, a project path, a prediction
count, an artifact name, an example request — into the UI source.

No Streamlit runtime, no GPU, no network.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_ux_contract -v
"""

from __future__ import annotations

import ast
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.artifact_preview import (  # noqa: E402
    KIND_BATTERYML_PREDICTION,
    KIND_UNREADABLE,
    KIND_UNSUPPORTED_PICKLE,
    SUPPORTED_PREVIEW_FORMATS,
    describe_structure,
    preview_prediction_file,
)
from batteryml_web.agent.catalog import (  # noqa: E402
    REQUEST_TEMPLATE_STRUCTURE,
    build_request_template,
)
from batteryml_web.ui import run_history as history  # noqa: E402
from batteryml_web.ui import settings_panel as settings  # noqa: E402

UI_DIR = REPO_ROOT / "batteryml_web" / "ui"


def _code_without_prose(path: Path) -> str:
    """A module's code with docstrings dropped (comments go with unparse).

    Static scans here assert what the code *does*. These modules explain in
    prose why an old hardcoded example was removed, and reading that prose as
    behaviour would fail the test for documenting itself.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


class _SyntheticDataset:
    def __init__(self, label):
        self.label = label


class _SyntheticBundle:
    """Stand-in for DataBundle; `synthetic_` by policy, never a real dataset."""

    def __init__(self, label, transformation=None):
        self.test_data = _SyntheticDataset(label)
        self.label_transformation = transformation


def _synthetic_prediction_pkl(directory: Path, name: str = "predictions_seed_0.pkl") -> Path:
    path = directory / name
    path.write_bytes(
        pickle.dumps(
            {
                "prediction": [110.0, 190.0, 330.0],
                "scores": {"RMSE": 1.0, "MAE": 2.0, "MAPE": 0.25},
                "data": _SyntheticBundle([100.0, 200.0, 300.0]),
                "seed": 0,
            }
        )
    )
    return path


class TestRequestTemplate(unittest.TestCase):
    """A: one template helper, filled from the backend."""

    def test_the_visible_structure_keeps_its_slots(self) -> None:
        for slot in ("[Model]", "[dataset]", "[task]"):
            self.assertIn(slot, REQUEST_TEMPLATE_STRUCTURE)

    def test_the_filled_template_follows_the_installation(self) -> None:
        """Two different catalogs must produce two different sentences."""
        first = build_request_template(
            {
                "datasets": [{"dataset": "HUST", "status": "READY"}],
                "models": [
                    {"model_key": "xgb", "model_display": "XGBoost — Boosted trees",
                     "datasets": ["HUST"], "external": False}
                ],
            }
        )
        second = build_request_template(
            {
                "datasets": [{"dataset": "CALCE", "status": "READY"}],
                "models": [
                    {"model_key": "cnn", "model_display": "CNN — Convolutional",
                     "datasets": ["CALCE"], "external": False}
                ],
            }
        )
        self.assertEqual(first, "Use XGBoost on HUST to predict battery lifetime.")
        self.assertEqual(second, "Use CNN on CALCE to predict battery lifetime.")
        self.assertNotEqual(first, second)

    def test_an_empty_catalog_keeps_the_placeholders_rather_than_inventing(self) -> None:
        filled = build_request_template({})
        self.assertIn("[Model]", filled)
        self.assertIn("[dataset]", filled)
        # ...and names nothing this installation does not have.
        for invented in ("MATR", "CPTransformer", "BatLiNet"):
            self.assertNotIn(invented, filled)

    def test_a_runnable_model_is_preferred_when_readiness_says_so(self) -> None:
        catalog = {
            "datasets": [{"dataset": "MATR", "status": "READY"}],
            "models": [
                {"model_key": "blocked", "model_display": "Blocked — x",
                 "datasets": ["MATR"], "external": False},
                {"model_key": "runnable", "model_display": "Runnable — y",
                 "datasets": ["MATR"], "external": False},
            ],
        }
        filled = build_request_template(catalog, runnable_keys=["runnable"])
        self.assertIn("Runnable", filled)
        self.assertNotIn("Blocked", filled)

    def test_no_example_gallery_survives_anywhere(self) -> None:
        for name in ("agent_workflow.py", "run_history.py"):
            code = _code_without_prose(UI_DIR / name)
            for gone in (
                "build_example_requests",
                "EXAMPLE_REQUEST_TEMPLATES",
                "agent_example_requests",
                "build_request_suggestion",
                "REQUEST_TABS",
            ):
                self.assertNotIn(gone, code, f"{gone} still referenced in {name}")

    def test_the_backend_no_longer_ships_a_hardcoded_example_set(self) -> None:
        code = _code_without_prose(
            REPO_ROOT / "batteryml_web" / "agent" / "catalog.py"
        )
        self.assertNotIn("EXAMPLE_REQUEST_TEMPLATES", code)
        self.assertNotIn("FALLBACK_REQUEST_SUGGESTION", code)

    def test_no_keyboard_hook_is_used_for_the_template(self) -> None:
        code = _code_without_prose(UI_DIR / "agent_workflow.py")
        for forbidden in ("keydown", "addEventListener", "preventDefault",
                          "window.parent", "unsafe_allow_html"):
            self.assertNotIn(forbidden, code)


class TestOneSettingsSourceOfTruth(unittest.TestCase):
    """B: entering settings once is enough."""

    SESSION_KEYS = ("llm_base_url", "llm_api_key", "llm_model", "llm_status")

    def test_only_the_settings_panel_writes_the_credential_keys(self) -> None:
        """The duplicate Home form is what made the app ask twice.

        It bound `st.text_input(key="llm_base_url")` straight to session
        state, bypassing set_llm_settings, so the status was never updated
        and the values did not register as configured elsewhere.
        """
        app_code = _code_without_prose(REPO_ROOT / "batteryml_web" / "app.py")
        for key in ("llm_base_url", "llm_api_key", "llm_model"):
            self.assertNotIn(
                f"key='{key}'", app_code,
                f"app.py must not bind a widget directly to {key}",
            )
            self.assertNotIn(f'key="{key}"', app_code)

    def test_every_surface_goes_through_the_one_form(self) -> None:
        app_code = _code_without_prose(REPO_ROOT / "batteryml_web" / "app.py")
        sidebar_code = _code_without_prose(UI_DIR / "workflow_sidebar.py")
        self.assertIn("render_inline_llm_settings", app_code)
        self.assertIn("render_llm_settings_form", app_code)
        self.assertIn("render_sidebar_settings", sidebar_code)

    def test_status_is_read_from_one_place_not_re_derived(self) -> None:
        app_code = _code_without_prose(REPO_ROOT / "batteryml_web" / "app.py")
        self.assertIn("describe_llm_status()", app_code)
        # The old ad-hoc rule: two fields non-empty means "Connected".
        self.assertNotIn(
            "'Connected' if st.session_state.get('llm_base_url')", app_code
        )

    def test_configured_is_not_reported_as_connected(self) -> None:
        from batteryml_web.ui.llm_state import (
            LLM_STATUS_CONFIGURED,
            LLM_STATUS_CONNECTED,
            LLM_STATUS_LABELS,
        )

        self.assertNotEqual(
            LLM_STATUS_LABELS[LLM_STATUS_CONFIGURED],
            LLM_STATUS_LABELS[LLM_STATUS_CONNECTED],
        )
        self.assertIn("not tested", LLM_STATUS_LABELS[LLM_STATUS_CONFIGURED].lower())

    def test_save_stores_without_a_network_call(self) -> None:
        code = _code_without_prose(UI_DIR / "settings_panel.py")
        save_block = code.split("if columns[0].button(SAVE_LABEL")[1].split("if columns[1]")[0]
        self.assertIn("set_llm_settings", save_block)
        self.assertNotIn("test_llm_connection", save_block)
        self.assertNotIn("connect_llm", save_block)

    def test_connect_stores_then_tests_through_the_existing_client(self) -> None:
        code = _code_without_prose(UI_DIR / "settings_panel.py")
        connect = code.split("def connect_llm")[1].split("\ndef ")[0]
        self.assertIn("set_llm_settings", connect)
        self.assertIn("test_llm_connection", connect)
        for forbidden in ("requests.post", "urlopen", "httpx"):
            self.assertNotIn(forbidden, connect)


class TestSettingsPropagateBetweenSurfaces(unittest.TestCase):
    """B: saving on one surface is visible on every other one.

    Each surface needs distinct widget keys (Streamlit rejects duplicates
    when two forms render in the same run), and a keyed widget's session
    value wins over its ``value=`` argument. That combination quietly turns
    per-surface keys into per-surface *copies*, which is the "enter it again"
    bug in a new form — a browser check caught it after the obvious
    duplicate form had already been deleted. These pin the fix.
    """

    def setUp(self) -> None:
        from tests.test_agent_web_ui import FakeStreamlit, _fake_streamlit

        self.FakeStreamlit = FakeStreamlit
        self._fake_streamlit = _fake_streamlit

    def _render(self, fake, prefix):
        with self._fake_streamlit(fake):
            return settings.render_llm_settings_form(key_prefix=prefix)

    def test_a_value_saved_on_one_surface_seeds_the_other(self) -> None:
        session = {}
        fake = self.FakeStreamlit(session)
        # Both surfaces render once, as they do on every script run.
        self._render(fake, "sidebar_llm")
        self._render(fake, "home")
        # The user types into the sidebar form and saves.
        session["sidebar_llm_base_url"] = "https://example.invalid/v1"
        session["sidebar_llm_api_key"] = "k"
        session["sidebar_llm_model"] = "m"
        saved = self.FakeStreamlit(session, {settings.SAVE_LABEL})
        with self._fake_streamlit(saved):
            settings.render_llm_settings_form(key_prefix="sidebar_llm")
        self.assertEqual(session["llm_base_url"], "https://example.invalid/v1")

        # Next run: the *other* form must show it without being retyped.
        again = self.FakeStreamlit(session)
        self._render(again, "home")
        self.assertEqual(session["home_base_url"], "https://example.invalid/v1")
        self.assertEqual(session["home_model"], "m")

    def test_half_typed_text_is_not_discarded_by_the_sync(self) -> None:
        """The store has not changed, so nothing may overwrite the box."""
        session = {}
        fake = self.FakeStreamlit(session)
        self._render(fake, "home")
        session["home_base_url"] = "https://half-typed"
        self._render(self.FakeStreamlit(session), "home")
        self.assertEqual(session["home_base_url"], "https://half-typed")

    def test_a_never_synced_form_fills_only_what_is_missing(self) -> None:
        session = {"home_base_url": "typed-before-first-render"}
        self._render(self.FakeStreamlit(session), "home")
        self.assertEqual(session["home_base_url"], "typed-before-first-render")
        # ...and the fields it had nothing for are seeded from the store.
        self.assertEqual(session["home_model"], "")

    def test_explain_reuses_saved_settings_without_asking_again(self) -> None:
        """D: settings entered earlier are already there for Explain."""
        from batteryml_web.ui.llm_state import is_llm_configured

        session = {}
        fake = self.FakeStreamlit(session)
        session["dialog_llm_base_url"] = "https://example.invalid/v1"
        session["dialog_llm_api_key"] = "k"
        session["dialog_llm_model"] = "m"
        with self._fake_streamlit(self.FakeStreamlit(session, {settings.SAVE_LABEL})):
            settings.render_llm_settings_form(key_prefix="dialog_llm")
        with self._fake_streamlit(self.FakeStreamlit(session)):
            self.assertTrue(is_llm_configured())


class TestProviderPresets(unittest.TestCase):
    """C: a preset fills the URL; it never locks it."""

    def test_presets_cover_the_named_providers(self) -> None:
        names = " ".join(settings.PROVIDER_PRESETS).lower()
        for provider in ("openai", "gemini", "claude", "kimi", "glm", "custom"):
            self.assertIn(provider, names)

    def test_each_preset_supplies_a_base_url_except_custom(self) -> None:
        for name, url in settings.PROVIDER_PRESETS.items():
            if name == settings.CUSTOM_PROVIDER:
                self.assertEqual(url, "")
            else:
                self.assertTrue(url.startswith("https://"), f"{name}: {url!r}")

    def test_no_model_names_are_baked_into_the_presets(self) -> None:
        """Model names change faster than this file; the user types them."""
        blob = repr(settings.PROVIDER_PRESETS).lower()
        for model_name in ("gpt-4", "gpt-5", "claude-3", "gemini-1.5", "glm-4"):
            self.assertNotIn(model_name, blob)

    def test_the_base_url_field_stays_editable(self) -> None:
        code = _code_without_prose(UI_DIR / "settings_panel.py")
        self.assertIn("st.text_input(FIELD_LABELS['llm_base_url']", code)
        self.assertNotIn("disabled=True", code)

    def test_the_supported_api_note_is_precise_about_compatibility(self) -> None:
        """Still a compatibility claim, now in one line.

        The caveat about provider-specific features moved into the constant's
        own comment when the helper copy was shortened: a maintainer needs
        the precision, a user needs to know which endpoints work.
        """
        note = settings.SUPPORTED_API_NOTE.lower()
        self.assertIn("openai-compatible", note)
        self.assertIn("custom", note)
        source = Path(settings.__file__).read_text(encoding="utf-8")
        self.assertIn("are not\n#: implemented", source)

    def test_the_api_key_is_masked(self) -> None:
        code = _code_without_prose(UI_DIR / "settings_panel.py")
        self.assertIn("type='password'", code)


class TestConnectionErrorClassification(unittest.TestCase):
    """C: network, auth and unknown are told apart."""

    def test_the_three_cases_are_distinguished(self) -> None:
        from batteryml_web.agent.failure_taxonomy import (
            CATEGORY_LLM_AUTH,
            CATEGORY_LLM_TIMEOUT,
            CATEGORY_UNKNOWN,
            classify_llm_failure,
        )

        self.assertEqual(
            classify_llm_failure("LLM connection failed: Network is unreachable")[
                "category"
            ],
            CATEGORY_LLM_TIMEOUT,
        )
        self.assertEqual(
            classify_llm_failure("HTTP 401 Unauthorized")["category"], CATEGORY_LLM_AUTH
        )
        self.assertEqual(
            classify_llm_failure("something nobody predicted")["category"],
            CATEGORY_UNKNOWN,
        )

    def test_an_unfinished_form_is_not_reported_as_a_connection_failure(self) -> None:
        """Nothing was sent, so nothing can have failed to connect."""
        code = _code_without_prose(UI_DIR / "settings_panel.py")
        self.assertIn("message == INCOMPLETE_MESSAGE", code)


class TestArtifactPreview(unittest.TestCase):
    """D: .pkl is no longer a dead end, and unknown files fail gracefully."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def test_a_batteryml_prediction_payload_is_recognised_and_summarised(self) -> None:
        preview = preview_prediction_file(_synthetic_prediction_pkl(self.directory))
        self.assertEqual(preview["kind"], KIND_BATTERYML_PREDICTION)
        self.assertEqual(preview["summary"]["Predictions"], "3")
        self.assertEqual(preview["summary"]["Ground truth stored"], "Yes")
        self.assertEqual(preview["summary"]["Seed recorded"], "0")
        self.assertEqual(preview["summary"]["Saved metrics present"], "Yes")

    def test_the_preview_never_restates_a_metric_value(self) -> None:
        """Presence only: the numbers stay with the canonical result."""
        preview = preview_prediction_file(_synthetic_prediction_pkl(self.directory))
        blob = repr(preview["summary"])
        for value in ("1.0", "2.0", "0.25", "RMSE", "MAE", "MAPE"):
            self.assertNotIn(value, blob.replace("'Predictions': '3'", ""))

    def test_the_preview_carries_the_stored_arrays_for_the_figures(self) -> None:
        preview = preview_prediction_file(_synthetic_prediction_pkl(self.directory))
        self.assertEqual(preview["rul"]["observed"], [100.0, 200.0, 300.0])
        self.assertEqual(preview["rul"]["predicted"], [110.0, 190.0, 330.0])

    def test_an_unrecognised_pickle_is_described_not_interpreted(self) -> None:
        path = self.directory / "weights.pkl"
        path.write_bytes(pickle.dumps({"weights": [1, 2, 3], "note": "x"}))
        preview = preview_prediction_file(path)
        self.assertEqual(preview["kind"], KIND_UNSUPPORTED_PICKLE)
        self.assertIn("not a supported prediction-view format", preview["message"])
        # Structure, and nothing more.
        names = {field["name"] for field in preview["structure"]["fields"]}
        self.assertEqual(names, {"weights", "note"})
        self.assertIsNone(preview["rul"])
        self.assertEqual(preview["summary"], {})

    def test_a_dict_with_a_prediction_key_but_no_usable_shape_is_unsupported(self) -> None:
        """A familiar key name is not a format."""
        path = self.directory / "odd.pkl"
        path.write_bytes(pickle.dumps({"prediction": "not an array"}))
        self.assertEqual(
            preview_prediction_file(path)["kind"], KIND_UNSUPPORTED_PICKLE
        )

    def test_an_unreadable_pickle_says_so_rather_than_raising(self) -> None:
        path = self.directory / "broken.pkl"
        path.write_bytes(b"\x80\x04 not really a pickle")
        preview = preview_prediction_file(path)
        self.assertEqual(preview["kind"], KIND_UNREADABLE)
        self.assertIn("could not be read back", preview["message"])

    def test_the_structure_description_reports_only_what_it_reads(self) -> None:
        described = describe_structure({"a": [1, 2, 3], "b": {"x": 1}})
        by_name = {f["name"]: f for f in described["fields"]}
        self.assertEqual(by_name["a"]["shape"], "3 item(s)")
        self.assertEqual(by_name["b"]["shape"], "1 key(s)")
        self.assertEqual(described["type"], "dict")

    def test_only_one_unpickler_exists_on_this_path(self) -> None:
        code = _code_without_prose(
            REPO_ROOT / "batteryml_web" / "agent" / "artifact_preview.py"
        )
        self.assertIn("load_prediction_payload", code)
        self.assertNotIn("pickle.load", code)
        self.assertNotIn("Unpickler", code)

    def test_the_supported_format_list_matches_what_the_code_handles(self) -> None:
        blob = " ".join(SUPPORTED_PREVIEW_FORMATS).lower()
        self.assertIn(".pkl", blob)
        self.assertIn(".csv", blob)
        self.assertIn(".json", blob)


class TestFileBrowserGrouping(unittest.TestCase):
    """D: grouped by what files are, from what was discovered."""

    def test_groups_come_from_discovered_entries_only(self) -> None:
        grouped = history.group_artifacts(
            [
                {"category": "logs", "relative_path": "stdout.log"},
                {"category": "predictions", "relative_path": "p.pkl"},
            ]
        )
        self.assertEqual(set(grouped), {"logs", "predictions"})
        # No empty group is invented to make the view look complete.
        self.assertNotIn("figures", grouped)
        self.assertNotIn("checkpoints", grouped)

    def test_known_groups_lead_and_unknown_ones_still_appear(self) -> None:
        grouped = history.group_artifacts(
            [{"category": "mystery"}, {"category": "predictions"}]
        )
        self.assertEqual(list(grouped)[0], "predictions")
        self.assertIn("mystery", grouped)

    def test_sizes_are_formatted_and_missing_ones_are_dashes(self) -> None:
        self.assertEqual(history._human_size(512), "512 B")
        self.assertEqual(history._human_size(1536), "1.5 KB")
        self.assertEqual(history._human_size(None), "—")


class TestNothingRuntimeIsHardcodedInTheNewUx(unittest.TestCase):
    """E: designed to fail if a runtime value is written into the UI."""

    SOURCES = (
        "agent_workflow.py",
        "run_history.py",
        "settings_panel.py",
        "workflow_sidebar.py",
        "formatting.py",
    )

    def _code(self) -> str:
        return "\n".join(_code_without_prose(UI_DIR / name) for name in self.SOURCES)

    def test_no_model_or_dataset_name_is_written_into_the_ui(self) -> None:
        code = self._code()
        for name in ("CPTransformer", "BatLiNet", "XGBoost", "PBT",
                     "BatteryMFormer", "MATR", "CALCE", "HUST"):
            self.assertNotIn(name, code)

    def test_no_fixed_counts(self) -> None:
        code = self._code()
        for forbidden in ("23 models", "= 23", "blocked_count = 1",
                          "range(8)", "num_predictions = "):
            self.assertNotIn(forbidden, code)

    def test_no_fixed_project_or_workspace_path(self) -> None:
        code = self._code()
        for forbidden in ("web_projects/default", "runs/native_matr",
                          "/root/share", "web_projects/1"):
            self.assertNotIn(forbidden, code)

    def test_no_gpu_hardware_is_named(self) -> None:
        code = self._code()
        for forbidden in ("RTX", "Tesla", "A100", "cuda:1", "cuda:7", "24576"):
            self.assertNotIn(forbidden, code)

    def test_no_metric_values_are_written_into_the_ui(self) -> None:
        code = self._code()
        for forbidden in ("794.32", "455.58", "716.48", "RMSE ="):
            self.assertNotIn(forbidden, code)

    def test_no_artifact_filename_is_assumed_to_exist(self) -> None:
        """Artifact names come from discovery, never from a fixed list."""
        code = self._code()
        for forbidden in ("predictions_seed_0.pkl", "metrics.csv\"", "report.html"):
            self.assertNotIn(forbidden, code)

    def test_no_example_request_sentence_survives(self) -> None:
        code = self._code()
        for forbidden in ("predict battery lifetime.", "帮我用", "Recommend a model"):
            self.assertNotIn(forbidden, code)


if __name__ == "__main__":
    unittest.main()
