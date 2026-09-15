"""Home entry point, sidebar shape, and project-creation feedback.

Milestone C. The Streamlit-free half of these tests reads
``batteryml_web/app.py`` as source and checks it with the ``ast`` module, so
the Home page's boundary (it navigates; it does not resolve intent, read
readiness, plan or execute) is established without importing torch,
pandas or Streamlit. The sidebar half imports the sidebar module and skips
where Streamlit is absent.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_web_navigation -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_PATH = REPO_ROOT / "batteryml_web" / "app.py"
SIDEBAR_PATH = REPO_ROOT / "batteryml_web" / "ui" / "workflow_sidebar.py"


def _streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return False
    return True


STREAMLIT_AVAILABLE = _streamlit_available()

PLACEHOLDER = "Use CPTransformer on MATR to predict battery lifetime."


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        func = inner.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            prefix = func.value.id if isinstance(func.value, ast.Name) else ""
            names.add(f"{prefix}.{func.attr}" if prefix else func.attr)
    return names


class TestHomeAgentEntry(unittest.TestCase):
    """Home offers the agent at Quick Start's level and only navigates."""

    def setUp(self) -> None:
        self.source = APP_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_entry_exists_with_the_required_copy(self) -> None:
        entry = _function(self.tree, "render_home_agent_entry")
        self.assertIsNotNone(entry, "Home must offer a natural-language agent entry")
        body = ast.get_source_segment(self.source, entry)
        self.assertIn("Describe what you want to do", body)
        self.assertIn(PLACEHOLDER, body)
        self.assertIn("Analyze with Agent", body)

    def test_quick_start_and_agent_share_the_same_heading_level(self) -> None:
        home = _function(self.tree, "render_home_page")
        entry = _function(self.tree, "render_home_agent_entry")
        home_body = ast.get_source_segment(self.source, home)
        entry_body = ast.get_source_segment(self.source, entry)
        self.assertIn('st.subheader("⚡ Quick Start", anchor=False)', home_body)
        self.assertIn('st.subheader("🤖 Agent Workflow", anchor=False)', entry_body)
        self.assertIn("render_home_agent_entry()", home_body)

    def test_home_records_the_request_and_navigates(self) -> None:
        entry = _function(self.tree, "render_home_agent_entry")
        body = ast.get_source_segment(self.source, entry)
        self.assertIn('st.session_state["agent_original_request"] = trimmed', body)
        self.assertIn('st.session_state["agent_pending_request"] = trimmed', body)
        self.assertIn('set_current_step("agent_workflow")', body)

    def test_home_duplicates_no_backend_logic_and_executes_nothing(self) -> None:
        entry = _function(self.tree, "render_home_agent_entry")
        called = _called_names(entry)
        for forbidden in (
            "resolve_workflow_intent",
            "build_agent_workflow_context",
            "get_workflow_context",
            "assess_dataset",
            "assess_model",
            "build_execution_plan",
            "execute_plan",
            "execute_plan_streaming",
            "plan_for_request",
            "subprocess.run",
            "subprocess.Popen",
        ):
            self.assertNotIn(
                forbidden,
                called,
                f"render_home_agent_entry must not call {forbidden}",
            )

    def test_no_raw_markdown_page_headings_remain(self) -> None:
        """Markdown headings carry Streamlit's anchor link; native ones do not."""
        for bad in ('st.markdown("# ', 'st.markdown("## '):
            self.assertNotIn(bad, self.source)

    def test_sidebar_is_not_redefined_in_the_app(self) -> None:
        definitions = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "render_sidebar_workflow"
        ]
        self.assertEqual(
            definitions,
            [],
            "app.py must use the sidebar from batteryml_web/ui/workflow_sidebar.py, "
            "not shadow it with a second definition",
        )


class TestProjectCreationFeedback(unittest.TestCase):
    def setUp(self) -> None:
        self.source = APP_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_success_state_reports_what_was_created(self) -> None:
        notice = _function(self.tree, "_render_project_created_notice")
        self.assertIsNotNone(notice)
        body = ast.get_source_segment(self.source, notice)
        self.assertIn("✅ Project created successfully", body)
        self.assertIn("**Project:**", body)
        self.assertIn("**Workspace:**", body)

    def test_notice_is_stored_from_the_real_created_path(self) -> None:
        home = _function(self.tree, "render_home_page")
        body = ast.get_source_segment(self.source, home)
        self.assertIn("created = _ensure_project(project_name, base_dir)", body)
        self.assertIn('"workspace": created.as_posix()', body)
        self.assertIn("_render_project_created_notice()", body)

    def test_project_filesystem_semantics_are_unchanged(self) -> None:
        ensure = _function(self.tree, "_ensure_project")
        body = ast.get_source_segment(self.source, ensure)
        self.assertIn("project_root.mkdir(parents=True, exist_ok=True)", body)
        self.assertIn('(project_root / "project_state.json").write_text', body)


@unittest.skipUnless(STREAMLIT_AVAILABLE, "streamlit is not importable in this environment")
class TestSidebarShape(unittest.TestCase):
    """The agent is an alternative entry point, not a numbered protocol step."""

    def setUp(self) -> None:
        from batteryml_web.ui import workflow_sidebar as sidebar

        self.sidebar = sidebar

    def test_agent_is_not_a_numbered_workflow_step(self) -> None:
        numbered = [step["key"] for step in self.sidebar.WORKFLOW_STEPS]
        self.assertNotIn("agent_workflow", numbered)
        self.assertEqual(numbered, ["prepare_data", "run_model", "review_results"])
        self.assertEqual(
            [step["step_no"] for step in self.sidebar.WORKFLOW_STEPS], [1, 2, 3]
        )

    def test_agent_has_its_own_group(self) -> None:
        self.assertEqual([step["key"] for step in self.sidebar.AGENT_STEPS], ["agent_workflow"])
        agent = self.sidebar.AGENT_STEPS[0]
        self.assertEqual(agent["group"], "agent")
        self.assertNotIn("step_no", agent)

    def test_every_page_key_is_still_reachable(self) -> None:
        self.assertEqual(
            set(self.sidebar.STEP_KEYS),
            {"home", "agent_workflow", "prepare_data", "run_model", "review_results"},
        )

    def test_sidebar_source_names_it_an_alternative(self) -> None:
        source = SIDEBAR_PATH.read_text(encoding="utf-8")
        self.assertIn("Intelligent entry point", source)
        self.assertIn("not a step in it", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
