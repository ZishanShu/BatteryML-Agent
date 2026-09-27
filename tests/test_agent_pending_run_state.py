"""The reserved run id is a pair of session keys, and they move together.

The Agent Workflow page crashed on the real server:

    KeyError: st.session_state has no key "agent_pending_run_id"

Two session keys hold one value. ``agent_pending_run_subject`` is the cache
key — what makes this plan *this* plan — and ``agent_pending_run_id`` is the
id reserved for it. Pressing Run popped the id alone. On the next rerun of
the same plan the subject still matched, so the "has the plan changed?"
branch was skipped, minting never happened, and the function then read a key
that was no longer there. The page died with Streamlit's native traceback.

These tests are about the contract, not the rendering: the reservation logic
takes a plain mapping, so a session is a dict here and the scenarios are
exact. The crash case is reproduced deliberately — a half-cleared pair — and
required to heal rather than raise.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_pending_run_state -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py"

PLAN = {
    "dataset": "MATR",
    "model": "ridge",
    "seed": 0,
    "device": "cpu",
    "smoke": False,
}


def _strip_docstrings(source: str) -> str:
    """The module's executable source, with every docstring removed.

    Prose is documentation: the docstring quoting the original
    ``KeyError: ... "agent_pending_run_id"`` must not read as a second use of
    the key.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return False
    return True


STREAMLIT_AVAILABLE = _streamlit_available()


@unittest.skipUnless(STREAMLIT_AVAILABLE, "streamlit is not importable in this environment")
class PendingRunTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from batteryml_web.ui import agent_workflow

        self.ui = agent_workflow
        self.ID = agent_workflow.PENDING_RUN_ID_KEY
        self.SUBJECT = agent_workflow.PENDING_RUN_SUBJECT_KEY

    def reserve(self, session: dict, **overrides) -> str:
        plan = {**PLAN, **overrides}
        return self.ui.reserved_run_id(session, **plan)


class TestFreshSession(PendingRunTestCase):
    """Nothing in session state is not an error."""

    def test_a_missing_key_does_not_raise(self) -> None:
        session: dict = {}
        self.assertNotIn(self.ID, session)
        run_id = self.reserve(session)
        self.assertTrue(run_id)

    def test_the_first_reservation_creates_exactly_one_id(self) -> None:
        session: dict = {}
        run_id = self.reserve(session)
        self.assertEqual(session[self.ID], run_id)
        self.assertIn(self.SUBJECT, session)

    def test_the_id_is_never_the_string_none(self) -> None:
        for prior in ({}, {self.ID: None}, {self.ID: ""}, {self.ID: "   "}):
            with self.subTest(prior=prior):
                session = dict(prior)
                run_id = self.reserve(session)
                self.assertNotIn(run_id, ("None", "", "none"))
                self.assertNotEqual(run_id.strip(), "")

    def test_the_id_looks_like_a_run_id(self) -> None:
        run_id = self.reserve({}).lower()
        self.assertIn("matr", run_id)
        self.assertIn("ridge", run_id)
        self.assertIn("seed0", run_id)


class TestStabilityAcrossReruns(PendingRunTestCase):
    """A rerun must not renumber the run."""

    def test_the_same_plan_keeps_the_same_id(self) -> None:
        session: dict = {}
        first = self.reserve(session)
        for _ in range(5):
            self.assertEqual(self.reserve(session), first)

    def test_no_new_id_is_minted_on_a_rerun(self) -> None:
        session: dict = {}
        self.reserve(session)
        before = session[self.ID]
        self.reserve(session)
        self.assertIs(session[self.ID], before)


class TestInvalidation(PendingRunTestCase):
    """A different experiment is a different run."""

    def test_each_execution_defining_input_invalidates(self) -> None:
        for field, changed in (
            ("dataset", "HUST"),
            ("model", "xgb"),
            ("seed", 7),
            ("device", "cuda:0"),
            ("smoke", True),
        ):
            with self.subTest(field=field):
                session: dict = {}
                first = self.reserve(session)
                second = self.reserve(session, **{field: changed})
                self.assertNotEqual(
                    second, first,
                    f"changing {field} must reserve a different run id",
                )

    def test_changing_back_does_not_restore_the_old_id(self) -> None:
        """A reservation is not a cache of every plan ever seen."""
        session: dict = {}
        first = self.reserve(session)
        self.reserve(session, seed=7)
        back = self.reserve(session)
        self.assertNotEqual(back, first)

    def test_the_subject_is_pure_and_order_independent(self) -> None:
        a = self.ui.pending_run_subject(**PLAN)
        b = self.ui.pending_run_subject(
            smoke=PLAN["smoke"], device=PLAN["device"], seed=PLAN["seed"],
            model=PLAN["model"], dataset=PLAN["dataset"],
        )
        self.assertEqual(a, b)
        self.assertNotEqual(a, self.ui.pending_run_subject(**{**PLAN, "seed": 1}))

    def test_seed_is_compared_as_text(self) -> None:
        """0 and "0" are the same seed and must not renumber the run."""
        session: dict = {}
        first = self.reserve(session, seed=0)
        self.assertEqual(self.reserve(session, seed="0"), first)


class TestTheCrashCase(PendingRunTestCase):
    """The exact server failure, reproduced and required to heal."""

    def test_a_half_cleared_pair_remints_instead_of_raising(self) -> None:
        session: dict = {}
        first = self.reserve(session)

        # What pressing Run used to do: pop the id, keep the subject.
        session.pop(self.ID)
        self.assertIn(self.SUBJECT, session)

        try:
            second = self.reserve(session)
        except KeyError as exc:  # pragma: no cover - this is the regression
            self.fail(f"the pending run id still raises KeyError: {exc!r}")
        self.assertTrue(second)
        self.assertNotEqual(second, first, "the spent reservation was reused")

    def test_clearing_releases_both_keys(self) -> None:
        session: dict = {}
        self.reserve(session)
        self.ui.clear_pending_run(session)
        self.assertNotIn(self.ID, session)
        self.assertNotIn(self.SUBJECT, session)

    def test_clearing_an_empty_session_is_safe(self) -> None:
        session: dict = {}
        self.ui.clear_pending_run(session)
        self.assertEqual(session, {})

    def test_a_launched_run_id_is_not_reused(self) -> None:
        """Launch consumes the reservation; the next plan gets a new id."""
        session: dict = {}
        launched = self.reserve(session)
        self.ui.clear_pending_run(session)          # what launching does now
        next_reservation = self.reserve(session)    # same plan, next rerun
        self.assertNotEqual(next_reservation, launched)

    def test_a_historical_id_left_in_session_is_not_adopted(self) -> None:
        session = {self.ID: "run_from_last_week", self.SUBJECT: "a different plan"}
        run_id = self.reserve(session)
        self.assertNotEqual(run_id, "run_from_last_week")


class TestTheSourceKeepsTheContract(unittest.TestCase):
    """Guards that outlive this particular fix."""

    def setUp(self) -> None:
        self.source = MODULE_PATH.read_text(encoding="utf-8")
        self.code = _strip_docstrings(self.source)

    def test_the_pending_id_is_never_read_with_brackets(self) -> None:
        """A bracket read is what turned a missing key into a dead page."""
        self.assertNotIn("st.session_state[PENDING_RUN_ID_KEY]", self.source)
        self.assertNotIn('st.session_state["agent_pending_run_id"]', self.source)

    def test_every_release_goes_through_the_one_helper(self) -> None:
        """No site may pop one key of the pair on its own."""
        for lone_pop in (
            "st.session_state.pop(PENDING_RUN_ID_KEY",
            "st.session_state.pop(PENDING_RUN_SUBJECT_KEY",
            'st.session_state.pop("agent_pending_run_id"',
            'st.session_state.pop("agent_pending_run_subject"',
        ):
            with self.subTest(pop=lone_pop):
                self.assertNotIn(
                    lone_pop, self.source,
                    "release the reservation with clear_pending_run(), which "
                    "pops the id and its subject together",
                )
        self.assertIn("clear_pending_run(st.session_state)", self.source)

    def test_the_subject_key_has_a_named_constant(self) -> None:
        """A literal in three places is how the pair drifted apart."""
        tree = ast.parse(self.source)
        assigned = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("PENDING_RUN_SUBJECT_KEY", assigned)
        self.assertIn("PENDING_RUN_ID_KEY", assigned)
        # The literal appears only where the constants are defined. Counted
        # in the executable source: the docstring that quotes the original
        # KeyError is documentation, not a second use of the key.
        self.assertEqual(self.code.count("'agent_pending_run_subject'"), 1)
        self.assertEqual(self.code.count("'agent_pending_run_id'"), 1)

    def test_no_unguarded_bracket_read_of_session_state_remains(self) -> None:
        """Every ``st.session_state[...]`` read is guarded or an assignment.

        Scanned by AST: a subscript that is the target of an assignment is a
        write and is fine; a subscript being *read* is only fine when the
        module has checked for the key. The known-guarded reads are listed,
        so a new one shows up here rather than on a user's screen.
        """
        allowed_reads = {
            "llm_base_url", "llm_api_key", "llm_model",   # behind is_llm_configured()
            "agent_explanation",                          # behind a .get() truth check
        }
        tree = ast.parse(self.source)
        offenders: list[str] = []
        written: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        written.add(id(target))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript) or id(node) in written:
                continue
            value = node.value
            if not (
                isinstance(value, ast.Attribute)
                and value.attr == "session_state"
            ):
                continue
            key = node.slice
            name = key.value if isinstance(key, ast.Constant) else (
                key.id if isinstance(key, ast.Name) else "<dynamic>"
            )
            if name not in allowed_reads:
                offenders.append(f"{name} (line {node.lineno})")
        self.assertEqual(
            offenders, [],
            f"unguarded session-state read(s): {offenders}. Use .get() with a "
            "default, or add to allowed_reads with the guard named.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
