# Tab-to-accept: what the installed Streamlit can and cannot do

Milestone section 4 asks for an audit before implementing Tab-to-accept, and
requires reporting the limitation rather than faking the interaction. This is
that audit.

**Conclusion: Tab-key acceptance is not implementable robustly here today.**
The request box ships with the suggestion as grey placeholder text and one
quiet `Use suggestion` button. Nothing in the page claims that Tab accepts
the suggestion.

Every fact below is `[VERIFIED]` — observed by running the interpreter in the
authoring container — or `[STATIC]`, read from source. Nothing here was
observed in a browser.

---

## What was checked

`[VERIFIED]` Installed version: **Streamlit 1.64.0** (`streamlit.__version__`).

`[VERIFIED]` `st.text_area` accepts exactly these parameters
(`inspect.signature`):

```
label, value, height, max_chars, key, help, on_change, args, kwargs,
placeholder, disabled, label_visibility, width, bind, persist_state
```

There is no key-event parameter. `on_change` fires when the *value* commits,
not on a keystroke, and it cannot see which key was pressed or suppress the
browser's default Tab behaviour.

`[VERIFIED]` `st.components.v1` exposes `declare_component`, `html` and
`iframe`. `st.components.v1.html` takes `html, width, height, scrolling,
tab_index` — no return channel to Python.

`[VERIFIED]` A second, newer component API exists: `st._bidi_component`,
backed by `streamlit/components/v2/`. Its signature carries
`isolate_styles: ComponentIsolateStyles = True`, documented as "whether to
isolate the component's styles in a **shadow root**".

`[VERIFIED]` There is no public `st.component`; only the underscore-prefixed
`st._bidi_component` exists in this build.

`[VERIFIED]` `requirements.txt:41` reads `streamlit  # PIN: fill from
environment_check.json` — Streamlit is **unpinned**, and the version on the
server has never been observed from this environment.

---

## Why option A (native) fails

There is no native keyboard mechanism to use. `st.text_area` has no key
handler, and a placeholder is decorative: the browser shows it and discards
it. Pressing Tab in a Streamlit text area moves focus to the next control,
which is standard and correct behaviour we must not break.

## Why option B (custom component) fails *today*

Two routes exist, and each is blocked by something specific.

**v1 `declare_component`** renders the component in an `<iframe>`. To own the
Tab key the component must own the text field, which puts the platform's
primary request input inside an iframe. That input then no longer inherits
Streamlit's theme, fonts or focus ring; its height cannot follow its content
without a `setFrameHeight` round trip; and its value reaches Python only
asynchronously. The alternative — leaving the input native and having an
iframe reach into `window.parent` to write it — requires locating the widget
in the parent DOM and forging a React input event so Streamlit's state
tracker notices. Section 4 forbids fragile DOM selectors, and this is the
fragile case: it depends on React internals that no Streamlit version
promises.

**v2 `st._bidi_component`** is the better design — a shadow root rather than
an iframe, so the field stays in the main document and the browser's focus
order works normally. It is also **private and undocumented**: an
underscore-prefixed API with no public alias in this build. Combined with the
unpinned `streamlit` requirement above, building the primary Route B input on
it means the request box disappears on any server whose Streamlit differs
from this container's. `CLAUDE.md` §1 makes Route B the reviewer's
reproduction path and says it must never break; §6 says keep the risky part
small and add no unpinned dependency. Betting the request box on a private
API fails all three.

---

## What would unblock it

Any one of these, in preference order:

1. **Streamlit publishes the v2 component API** (a public `st.component`, or
   `st._bidi_component` documented and stable). Then the request field
   becomes a shadow-root component, Tab is handled in its own keydown
   handler, and focus order is unaffected — the design section 4 asks for,
   with no DOM archaeology.
2. **`streamlit` is pinned** in `requirements.txt` to a version whose
   component API has been checked on the server. A pinned private API is a
   known risk rather than an unknown one; it would still need the fallback in
   point 3.
3. **A verified browser test exists.** This repository has no browser-level
   test harness, so a component's behaviour — does the iframe load, does the
   value round-trip, does focus leave correctly — cannot be established by
   anything in `scripts/verify/`. Adding one is a prerequisite for shipping
   frontend code that the Python tests cannot see.

## Update: what ships now

The `Use suggestion` button described below was replaced by a **Use template**
helper. The conclusion above is unchanged — Tab is still not intercepted, and
nothing claims it is — but the aid is now a template the user asks for rather
than a ghost value waiting to be accepted, which is a clearer interaction
than the one this audit was written to justify.

## What ships instead

* the suggestion is the `placeholder` of the native `st.text_area`, so it is
  grey, is not the user's request, and disappears as soon as they type;
* one `Use suggestion` button is rendered **only while the field is empty**,
  so typed text is never overwritten;
* accepting writes the canonical request key (`agent_original_request`) — the
  same key typing writes — and reruns. It does not analyse, call the LLM,
  build a plan or execute;
* Tab keeps its normal meaning everywhere on the page. Nothing is
  intercepted, globally or locally.

Section 4 permits a visible control "unless absolutely necessary"; this audit
is the record of that necessity.
