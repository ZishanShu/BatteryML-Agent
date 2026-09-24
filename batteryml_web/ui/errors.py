"""A page-level error boundary, and our own copy control.

When a page raises, Streamlit renders its built-in exception panel: a red
traceback with *Copy*, *Ask Google* and *Ask ChatGPT* buttons. That is a
developer's view shown to a scientist, and the first thing it does is invite
them to paste an internal traceback into a search engine.

This module renders our own view instead, for errors we can catch:

    Something went wrong
    <one actionable sentence>
    [ Show technical details ]        <- the traceback, redacted, copyable

**Streamlit's own exception panel is never touched.** No monkey-patching, no
DOM surgery, no JavaScript reaching into the host page to find a private
button. A genuinely uncaught framework failure still shows Streamlit's panel,
and that is the correct outcome: the alternative is fragile interception that
breaks on the next release and hides errors when it does.

Copying is deliberately *not* a JavaScript button of ours — see
:func:`render_copy_control` for the browser measurement that ruled one out.
The details are rendered with ``st.code``, whose copy icon is a documented
feature on our own element.

Secrets never reach the details. Everything rendered or copied goes through
:func:`~batteryml_web.agent.failure_taxonomy.redact`, the same filter the run
failure path already uses, so an API key in an environment dump or a request
URL is replaced before it is displayed, let alone copied.
"""

from __future__ import annotations

import html
import traceback
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import streamlit as st

from ..agent.failure_taxonomy import redact
from .feedback import toast
from .surfaces import NAV_ROLE

#: Shown as the headline for any caught error. Deliberately plain: the user
#: needs to know the app failed, not which module did.
ERROR_HEADLINE = "Something went wrong"

#: The transient confirmation. Transient on purpose — a permanent green box
#: for "you clicked a button" is noise a minute later. It says the details
#: are ready rather than that they are on the clipboard, because this control
#: cannot put them there; see :func:`render_copy_control`.
COPIED_MESSAGE = "Error details ready to copy"

#: What the control says before it is pressed.
COPY_LABEL = "Copy details"

#: Where the working copy affordance actually is.
COPY_HINT = "Use the copy icon at the top-right of the block above."


def format_details(
    error: BaseException | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    include_traceback: bool = True,
) -> str:
    """The technical block, redacted, as plain text.

    Args:
        error: the caught exception, if there is one.
        context: extra named facts worth carrying — the page, the project,
            the selected dataset. Values are stringified and redacted.
        include_traceback: whether to append the formatted traceback.

    Returns:
        Text safe to display and to copy. Contains no secret that
        :func:`redact` recognises.
    """
    lines: list[str] = []
    if context:
        for key, value in context.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    if error is not None:
        lines.append(f"{type(error).__name__}: {error}")
        if include_traceback:
            lines.append("")
            lines.append(
                "".join(
                    traceback.format_exception(
                        type(error), error, error.__traceback__
                    )
                ).rstrip()
            )
    return redact("\n".join(lines).strip())


def render_copy_control(
    text: str,
    *,
    key: str,
    label: str = COPY_LABEL,
    message: str = COPIED_MESSAGE,
) -> bool:
    """Show ``text`` in a copyable block, and say where the copy control is.

    **Why this is not a JavaScript copy button.** A first version wrote the
    clipboard from a zero-height component. Measured in a browser, it did not
    work: the clipboard was empty afterwards while the button had already
    toasted "copied". A Streamlit component runs in a sandboxed iframe that
    the Clipboard API refuses, and ``document.execCommand`` in an unfocused
    iframe fares no better. On the server it would fail harder still —
    ``navigator.clipboard`` does not exist at all outside a secure context,
    and the app is served over plain http. Reaching into the parent frame
    would be the "fragile JS interception" this milestone rules out.

    A control that claims a copy it did not perform is worse than no control,
    so the text is rendered in ``st.code``, whose copy icon is a documented
    Streamlit feature **on our own element** — not the private exception
    panel — and the caption points at it. What is displayed is what gets
    copied, and it is redacted before either.

    Pressing the button reveals the block and toasts once, transiently. It
    returns whether it was pressed on this run, so a rerun does not
    re-announce it.
    """
    if not text:
        return False
    revealed_key = f"{key}__revealed"
    pressed = st.button(label, type=NAV_ROLE, key=key, use_container_width=False)
    if pressed:
        st.session_state[revealed_key] = True
    if not st.session_state.get(revealed_key):
        return False
    st.code(text, language="text")
    st.caption(COPY_HINT)
    if pressed:
        toast(message, icon="📋")
    return bool(pressed)


def render_error(
    message: str,
    *,
    key: str,
    error: BaseException | None = None,
    details: str | None = None,
    context: Mapping[str, Any] | None = None,
    headline: str = ERROR_HEADLINE,
) -> None:
    """Our error view: headline, one actionable line, details on request.

    Args:
        message: what the user can do about it, in a sentence. Not the
            exception text — that belongs in the details.
        key: unique per call site; the details expander and the copy button
            derive their widget keys from it.
        error: the caught exception, if any.
        details: pre-built technical text. Built from ``error`` and
            ``context`` when omitted.
        context: named facts to include in the details.
        headline: overridable, but the default is what users should see.
    """
    st.error(f"{headline}")
    if message:
        st.write(redact(message))

    body = details if details is not None else format_details(error, context=context)
    if not body:
        return
    with st.expander("Show technical details", expanded=False):
        st.caption(
            "Redacted: API keys and other credentials are removed before this "
            "is shown or copied."
        )
        st.code(body, language="text")
        st.caption(COPY_HINT)


@contextmanager
def error_boundary(
    *,
    key: str,
    message: str,
    context: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Render our error view instead of Streamlit's panel, for this block.

    Only application errors are caught. ``KeyboardInterrupt`` and
    ``SystemExit`` pass through untouched, and so does Streamlit's own
    control flow — ``st.rerun()`` and ``st.stop()`` raise exceptions that
    *are* the framework working correctly, and swallowing them would break
    every rerun on the page.

    The exception is not re-raised: the point is that the user sees an
    actionable page rather than a traceback. It is printed to the server log,
    where an operator will look for it.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - this is the boundary
        if _is_streamlit_control_flow(exc):
            raise
        # The operator's copy, with the traceback intact.
        traceback.print_exc()
        render_error(message, key=key, error=exc, context=context)


def _is_streamlit_control_flow(exc: BaseException) -> bool:
    """Whether ``exc`` is Streamlit steering the script, not a failure.

    ``st.rerun()`` and ``st.stop()`` raise; catching them would strand the
    page. Matched by class name rather than by importing private Streamlit
    exception types, which have moved between releases.
    """
    names = {type(base).__name__ for base in (exc,)} | {
        cls.__name__ for cls in type(exc).__mro__
    }
    return bool(names & {"RerunException", "StopException", "RerunData"})


def escape(text: Any) -> str:
    """HTML-escape for the few places that build markup by hand."""
    return html.escape(str(text or ""))
