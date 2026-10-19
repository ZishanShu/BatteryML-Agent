"""A claim-by-claim review sheet for one recorded LLM explanation (EXPL-02).

Input: a record written by :mod:`.explanation_record`. Output: one row per
sentence of the model's answer, each with

* the model's words, verbatim;
* the evidence it can be checked against — lines of the prompt the model
  was **actually sent** that contain the same numbers or terms;
* an automatic finding;
* two empty columns, ``human_judgement`` and ``author_notes``, which this
  module never fills.

**What the automatic checks are.** String and number matching against the
sent prompt, nothing more. They answer "does this number appear in what the
model was given, exactly, rounded, or ×100?", "is this metric quoted with a
unit the context gives it?", "does this sentence define the task beyond what
the context states?", and "does it assert a mechanism, cause or
generalisation the context contains no evidence for?". A sentence with no
automatic finding is **not** thereby correct — the result label says
``NO_AUTOMATIC_FINDING``, never "pass" — and a flagged sentence is not
thereby wrong. Every judgement is the author's.

**What it checks against.** Only the messages the record says were sent.
Anything the model was not given (the label definition in
``batteryml/label/rul.py``, the dataset paper, the predictions themselves)
is not evidence for its claims; where a sentence relies on such knowledge
the finding says so, and may name where an author can look it up, marked
as not sent to the model.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FLAG = "FLAG"
NOTE = "NOTE"
NONE_FOUND = "NO_AUTOMATIC_FINDING"

#: The columns of the sheet, in order. The last two are the author's.
COLUMNS = (
    "claim_id",
    "section",
    "model_text",
    "evidence_in_sent_context",
    "automatic_result",
    "automatic_findings",
    "human_judgement",
    "author_notes",
)

#: Answer sections in the order the analysis prompt asks for them.
SECTIONS = ("summary", "interpretation", "diagnostics", "benchmark", "next_steps")

_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![A-Za-z_\d.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"'(\[])|(?<=[。！？；])")
_METRIC_VALUE = re.compile(
    r"\b(RMSE|MAE|MAPE)\b[^0-9\n]{0,30}?([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*(%|percent|per cent)?",
    re.IGNORECASE,
)

#: Terms that define what is being predicted. The context names the task and
#: the label annotator class; it does not define the label.
TASK_DEFINITION_TERMS = (
    "remaining useful life", "remaining life", "remaining cycles", "cycle life", "lifetime", "life time",
    "end of life", "end-of-life", "eol", "80%", "80 %", "knee", "state of health", "capacity fade",
    "capacity threshold", "nominal capacity", "failure", "until the cell",
)

MECHANISM_TERMS = (
    "mechanism", "lithium plating", "plating", "sei", "solid electrolyte", "electrolyte", "loss of lithium",
    "lli", "loss of active", "active material", "impedance", "internal resistance", "temperature",
    "fast charg", "charging protocol", "charging polic", "chemistry", "cathode", "anode", "graphite",
    "lfp", "lifepo", "particle", "crack", "dendrite",
)
CAUSAL_TERMS = (
    "because", "due to", "caused by", "driven by", "results from", "resulting from", "attributable",
    "explains why", "the reason", "stems from", "leads to", "is why",
)
GENERALISATION_TERMS = (
    "generaliz", "generalis", "other dataset", "other chemistr", "other cell", "real-world", "real world",
    "in practice", "deploy", "robust", "reliable", "trustworthy", "accurate enough", "state-of-the-art",
    "state of the art", "outperform", "superior", "best ", "competitive", "production", "in the field",
    "industr", "any battery", "all batteries", "unseen",
)
HYPOTHESIS_MARKERS = (
    "hypothes", "may ", "might", "could", "possibl", "likely", "perhaps", "plausib", "suggest", "would need",
    "to confirm", "check",
)

#: Where an author can look up what the model was not given. Never sent.
AUTHOR_REFERENCES = {
    "RUL prediction": "batteryml/label/rul.py (RULLabelAnnotator: label definition and EOL threshold)",
    "SOH estimation": "batteryml/label/soh.py and docs/SOH_PROTOCOL.md (SOH label definition)",
}


@dataclass
class ContextFacts:
    """What the model was sent, in checkable form."""

    prompt_text: str
    system_text: str
    lines: List[str]
    numbers: List[Tuple[str, float, str]] = field(default_factory=list)  # (token, value, line)
    metrics: Dict[str, Tuple[float, str]] = field(default_factory=dict)  # name -> (value, unit)
    task: str = ""
    label_annotator: Optional[str] = None
    benchmark_comparable: Optional[bool] = None
    benchmark_supplied: bool = False


def _to_float(token: str) -> Optional[float]:
    text = token.replace(",", "").rstrip("%")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _decimals(token: str) -> int:
    text = token.replace(",", "").rstrip("%").lower()
    if "e" in text:
        return 12
    return len(text.split(".", 1)[1]) if "." in text else 0


def _rounds_to(context_value: float, token: str) -> bool:
    """Whether ``token`` is ``context_value`` rounded to the token's decimals."""
    value = _to_float(token)
    if value is None:
        return False
    places = _decimals(token)
    return abs(round(context_value, places) - value) <= 0.5 * 10 ** (-places) + 1e-12


def context_facts(record: Mapping[str, Any]) -> ContextFacts:
    """The checkable facts of the messages the record says were sent."""
    messages = ((record.get("request") or {}).get("messages")) or []
    system = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")
    prompt = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
    lines = [line for line in prompt.splitlines() if line.strip()]
    facts = ContextFacts(prompt_text=prompt, system_text=system, lines=lines)
    for line in lines:
        for match in _NUMBER.finditer(line):
            value = _to_float(match.group())
            if value is not None:
                facts.numbers.append((match.group(), value, line.strip()))

    analysis = (record.get("request") or {}).get("analysis_context") or {}
    identity = analysis.get("identity") or {}
    facts.task = str(identity.get("task") or "")
    facts.label_annotator = (analysis.get("protocol") or {}).get("label_annotator")
    # Only metrics that appear in the sent prompt count as sent.
    for row in analysis.get("metrics") or []:
        name = str(row.get("name") or "").upper()
        value = row.get("value")
        # v1 renders "- RMSE = <repr>"; v2 sends JSON, where a float is its repr too.
        if isinstance(value, (int, float)) and (
            f"{row.get('name')} = {value!r}" in prompt or f'"value": {value!r}' in prompt
        ):
            facts.metrics[name] = (float(value), str(row.get("unit") or ""))
    block = analysis.get("benchmark") or {}
    if analysis.get("context_version") == "v2":
        # v2 never sends a difference and never marks a reference comparable.
        facts.benchmark_supplied = block.get("reference") is not None and '"reference": {' in prompt
        facts.benchmark_comparable = (block.get("comparability_status") == "COMPARABLE") if facts.benchmark_supplied else None
    else:
        facts.benchmark_supplied = block.get("record") is not None and "benchmark reference" in prompt
        facts.benchmark_comparable = bool(block.get("comparable")) if facts.benchmark_supplied else None
    return facts


def split_claims(parsed: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """(section, sentence) pairs, in the answer's own order."""
    claims: List[Tuple[str, str]] = []
    for section in SECTIONS:
        value = parsed.get(section)
        items: Sequence[Any] = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            for sentence in _SENTENCE_END.split(text):
                sentence = sentence.strip()
                if sentence:
                    claims.append((section, sentence))
    for key, value in parsed.items():
        if key not in SECTIONS and str(value or "").strip():
            claims.append((str(key), str(value).strip()))
    return claims


def _lines_with(facts: ContextFacts, needles: Iterable[str]) -> List[str]:
    found: List[str] = []
    for needle in needles:
        for line in facts.lines:
            if needle and needle.lower() in line.lower() and line.strip() not in found:
                found.append(line.strip())
    return found


def _check_number(token: str, facts: ContextFacts, *, percent_context: bool) -> Tuple[str, str, Optional[str]]:
    """(result, finding, evidence line) for one number in a claim."""
    bare = token.replace(",", "").rstrip("%")
    value = _to_float(token)
    is_percent = token.endswith("%") or percent_context
    for ctx_token, _ctx_value, line in facts.numbers:
        if ctx_token.rstrip("%") == bare:
            return NONE_FOUND, f"{token}: appears verbatim in the sent context.", line
    if value is None:
        return NOTE, f"{token}: not a readable number.", None
    mape = facts.metrics.get("MAPE")
    if mape is not None and is_percent:
        if _rounds_to(mape[0] * 100.0, bare):
            return NOTE, (
                f"{token}: MAPE {mape[0]!r} ×100 = {mape[0] * 100.0!r}%, shown to {_decimals(bare)} decimal(s); "
                "conversion consistent with the sent ratio."
            ), f"MAPE = {mape[0]!r} [{mape[1]}]"
        if _rounds_to(mape[0], bare):
            return FLAG, (
                f"{token}: equals the MAPE ratio {mape[0]!r} but is written as a percent — the ×100 "
                "conversion is missing."
            ), f"MAPE = {mape[0]!r} [{mape[1]}]"
    for ctx_token, ctx_value, line in facts.numbers:
        if ctx_value == value:
            return NONE_FOUND, f"{token}: same value as {ctx_token} in the sent context (formatting differs).", line
    for ctx_token, ctx_value, line in facts.numbers:
        if _decimals(bare) < _decimals(ctx_token) and _rounds_to(ctx_value, bare):
            return FLAG, (
                f"{token}: rounded from {ctx_token} in the sent context; the system prompt asked for values "
                "verbatim and at full precision."
            ), line
    derived = _derived(value, bare, facts)
    if derived:
        return FLAG, f"{token}: not in the sent context; matches {derived} (a derived number — check it).", None
    return FLAG, f"{token}: not found in the sent context, exactly, rounded or as MAPE×100.", None


def _derived(value: float, token: str, facts: ContextFacts) -> Optional[str]:
    """A simple arithmetic origin over the context's metric/benchmark numbers."""
    pool = [(t, v) for t, v, line in facts.numbers if any(k in line for k in ("=", "min=", "max=", "count="))]
    pool = pool[:60]
    for (ta, a), (tb, b) in ((x, y) for x in pool for y in pool if x is not y):
        candidates = [(a - b, f"{ta} − {tb}")]
        if b:
            candidates += [(a / b, f"{ta} / {tb}"), (a / b * 100.0, f"{ta} / {tb} × 100"),
                           ((a - b) / b * 100.0, f"({ta} − {tb}) / {tb} × 100")]
        for candidate, label in candidates:
            if math.isfinite(candidate) and _rounds_to(candidate, token):
                return label
    return None


def _check_metric_pairs(sentence: str, facts: ContextFacts) -> List[Tuple[str, str]]:
    """A value written right after a metric name must be that metric's value."""
    findings: List[Tuple[str, str]] = []
    for match in _METRIC_VALUE.finditer(sentence):
        name, token, percent = match.group(1).upper(), match.group(2).rstrip(".,"), match.group(3)
        if name not in facts.metrics:
            findings.append((FLAG, f"{name} = {token}: the sent context gives no {name} value."))
            continue
        value, unit = facts.metrics[name]
        if percent and name == "MAPE":
            ok = _rounds_to(value * 100.0, token)
            findings.append((NOTE if ok else FLAG, (
                f"MAPE written as {token}%: sent ratio {value!r} ×100 = {value * 100.0!r}; "
                + ("consistent." if ok else "does not match.")
            )))
        elif percent:
            findings.append((FLAG, f"{name} = {token}%: the sent context gives {name} in [{unit}], not percent."))
        elif token.replace(",", "") == repr(value) or _to_float(token) == value:
            findings.append((NONE_FOUND, f"{name} = {token}: matches the sent {name} exactly."))
        elif _rounds_to(value, token):
            findings.append((FLAG, f"{name} = {token}: the sent {name} is {value!r}; rounded."))
        else:
            findings.append((FLAG, f"{name} = {token}: the sent {name} is {value!r}; the values differ."))
    return findings


def _check_units(sentence: str, facts: ContextFacts) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    lowered = sentence.lower()
    for name, (_value, unit) in facts.metrics.items():
        for match in re.finditer(rf"\b{name}\b", sentence, re.IGNORECASE):
            window = lowered[match.end(): match.end() + 60]
            nxt = re.search(r"\b(rmse|mae|mape)\b", window)
            if nxt:
                window = window[: nxt.start()]
            if name in {"RMSE", "MAE"} and "cycle" in unit and re.search(r"\b(days?|hours?|ah|mah|soh)\b", window):
                findings.append((FLAG, f"{name}: a unit other than cycles appears; the sent unit is [{unit}]."))
            if name in {"RMSE", "MAE"} and "ratio" in unit and "cycle" in window:
                findings.append((FLAG, f"{name}: 'cycles' appears; the sent unit is [{unit}]."))
            if name == "MAPE" and "cycle" in window:
                findings.append((FLAG, "MAPE: 'cycles' appears; MAPE is a dimensionless ratio in the sent context."))
    return findings


def _contains(text: str, terms: Iterable[str]) -> List[str]:
    """Terms present in ``text``, matched at a word start.

    A term is a prefix ("generaliz" matches "generalization"), but it must
    start a word, and a term of four characters or fewer must also end one —
    otherwise "lli" would fire inside "intelligent" and "sei" inside "seismic".
    """
    lowered = f" {text.lower()} "
    found: List[str] = []
    for term in terms:
        core = term.strip()
        pattern = r"(?<![a-z0-9])" + re.escape(core)
        if len(core) <= 4 or term.endswith(" "):
            pattern += r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.append(core)
    return found


def check_claim(section: str, sentence: str, facts: ContextFacts) -> Dict[str, Any]:
    """The automatic findings and the evidence for one sentence."""
    findings: List[Tuple[str, str]] = []
    evidence: List[str] = []

    pair_spans = {m.group(2).rstrip(".,") for m in _METRIC_VALUE.finditer(sentence)}
    for result, text in _check_metric_pairs(sentence, facts):
        findings.append((result, text))
    for name, (value, _unit) in facts.metrics.items():
        if re.search(rf"\b{name}\b", sentence, re.IGNORECASE):
            evidence.extend(_lines_with(facts, [f"- {name} =", f'"value": {value!r}']))

    for match in _NUMBER.finditer(sentence):
        token = match.group()
        if token.rstrip("%").replace(",", "") in {p.replace(",", "") for p in pair_spans}:
            continue  # already checked against its metric
        following = sentence[match.end(): match.end() + 12].lower()
        percent_context = following.lstrip().startswith(("percent", "per cent"))
        result, text, line = _check_number(token, facts, percent_context=percent_context)
        findings.append((result, text))
        if line and line not in evidence:
            evidence.append(line)

    findings.extend(_check_units(sentence, facts))

    task_terms = _contains(sentence, TASK_DEFINITION_TERMS)
    if task_terms:
        unsent = [t for t in task_terms if t.lower() not in facts.prompt_text.lower()]
        reference = AUTHOR_REFERENCES.get(facts.task)
        text = (
            f"Task-definition wording ({', '.join(task_terms)}). The sent context states only "
            f"task={facts.task!r} and label annotator={facts.label_annotator!r}; it does not define the label."
        )
        if unsent:
            text += f" Not in the sent context: {', '.join(unsent)}."
        if reference:
            text += f" Author reference (not sent to the model): {reference}."
        findings.append((FLAG if unsent else NOTE, text))
        evidence.extend(_lines_with(facts, ["- task:", "label annotator"]))

    mechanism = _contains(sentence, MECHANISM_TERMS)
    causal = _contains(sentence, CAUSAL_TERMS)
    general = _contains(sentence, GENERALISATION_TERMS)
    hedged = bool(_contains(sentence, HYPOTHESIS_MARKERS))
    if mechanism:
        unsent = [t for t in mechanism if t.lower() not in facts.prompt_text.lower()]
        if unsent:
            findings.append((FLAG, (
                f"Physical/mechanistic terms not in the sent context: {', '.join(unsent)}. "
                f"Labelled as a hypothesis: {'yes' if hedged else 'no'}."
            )))
    if causal:
        findings.append((FLAG if not hedged else NOTE, (
            f"Causal wording ({', '.join(causal)}). The system prompt requires causes to be labelled as "
            f"hypotheses with a check. Labelled as a hypothesis: {'yes' if hedged else 'no'}."
        )))
    if general:
        findings.append((FLAG, (
            f"Generalisation/quality wording ({', '.join(general)}). The sent context describes one run "
            "on one split and contains no evidence beyond it."
        )))

    if section == "benchmark" or re.search(r"\b(benchmark|published|reference|paper)\b", sentence, re.IGNORECASE):
        if not facts.benchmark_supplied:
            findings.append((NOTE, "Benchmark statement; the sent context supplied NO published reference."))
        elif facts.benchmark_comparable is False and re.search(r"\b(better|worse|lower|higher|than|differ)", sentence, re.IGNORECASE):
            findings.append((FLAG, "Comparative wording against a reference the sent context marks NOT DIRECTLY COMPARABLE."))
        evidence.extend(_lines_with(facts, ["benchmark", "COMPARABLE", "source:"]))

    results = [result for result, _ in findings]
    overall = FLAG if FLAG in results else NOTE if NOTE in results else NONE_FOUND
    seen: List[str] = []
    for line in evidence:
        if line not in seen:
            seen.append(line)
    return {
        "automatic_result": overall,
        "automatic_findings": [f"[{result}] {text}" for result, text in findings],
        "evidence_in_sent_context": seen,
    }


def build_checklist(record: Mapping[str, Any]) -> Dict[str, Any]:
    """The review sheet for one record; human columns empty."""
    facts = context_facts(record)
    parsed = (record.get("response") or {}).get("parsed")
    rows: List[Dict[str, Any]] = []
    if isinstance(parsed, Mapping):
        for index, (section, sentence) in enumerate(split_claims(parsed), start=1):
            checked = check_claim(section, sentence, facts)
            rows.append(
                {
                    "claim_id": f"C{index:02d}",
                    "section": section,
                    "model_text": sentence,
                    **checked,
                    "human_judgement": "",
                    "author_notes": "",
                }
            )
    call = record.get("call") or {}
    result = record.get("result") or {}
    return {
        "record_id": record.get("record_id"),
        "outcome": record.get("outcome"),
        "project": result.get("project"),
        "run_id": result.get("run_id"),
        "task": facts.task,
        "requested_model": call.get("requested_model"),
        "response_model": call.get("response_model"),
        "sent_metrics": {name: {"value": value, "unit": unit} for name, (value, unit) in facts.metrics.items()},
        "benchmark_supplied": facts.benchmark_supplied,
        "benchmark_comparable": facts.benchmark_comparable,
        "rows": rows,
        "human_review_status": "NOT STARTED",
    }


def checklist_csv(sheet: Mapping[str, Any]) -> str:
    """The rows as CSV (UTF-8; list cells joined with ' | ')."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in sheet.get("rows") or []:
        writer.writerow(
            {
                **row,
                "evidence_in_sent_context": " | ".join(row.get("evidence_in_sent_context") or []),
                "automatic_findings": " | ".join(row.get("automatic_findings") or []),
            }
        )
    return buffer.getvalue()


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def checklist_markdown(sheet: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    """A human-readable draft: identity, what was sent, then one block per claim."""
    call = record.get("call") or {}
    files = record.get("files") or {}
    code = record.get("code") or {}
    lines = [
        f"# Explanation review sheet — DRAFT",
        "",
        f"- Record: `{record.get('record_id')}` ({record.get('action')})",
        f"- Outcome: **{record.get('outcome')}**",
        f"- Project / run: `{sheet.get('project')}` / `{sheet.get('run_id')}`",
        f"- Task (as sent): {sheet.get('task')!r}",
        f"- Requested model: `{call.get('requested_model')}`; response model: "
        f"`{call.get('response_model')}`" + (f" ({call.get('response_model_note')})" if call.get("response_model_note") else ""),
        f"- Generation settings sent: `{call.get('generation_settings_sent')}`, timeout {call.get('timeout_s')} s; "
        f"not sent (provider defaults): {', '.join(call.get('settings_not_sent_provider_defaults_apply') or [])}",
        f"- Called: {call.get('sent_at')} → {call.get('received_at')} (UTC)",
        f"- Code commit: `{code.get('commit')}`; tracked files modified: {code.get('tracked_files_modified')}",
        f"- Predictions: `{(files.get('predictions') or {}).get('path')}` sha256 "
        f"`{(files.get('predictions') or {}).get('sha256')}`",
        f"- Config: `{(files.get('config') or {}).get('path')}` sha256 `{(files.get('config') or {}).get('sha256')}`",
        "",
        "**Human review status: NOT STARTED.** The columns *Human judgement* and *Author notes* are the "
        "author's to fill. Automatic results are string/number matching against the prompt the model was "
        f"sent: `{NONE_FOUND}` means no pattern fired, not that the sentence is correct; `{FLAG}` means "
        "the author should look, not that it is wrong.",
        "",
        "## Metrics as sent to the model",
        "",
    ]
    metrics = sheet.get("sent_metrics") or {}
    if metrics:
        lines += ["| Metric | Value (verbatim) | Unit (as sent) |", "| --- | --- | --- |"]
        lines += [f"| {name} | `{entry['value']!r}` | {entry['unit']} |" for name, entry in metrics.items()]
    else:
        lines.append("No metric values were sent.")
    lines += [
        "",
        f"Benchmark reference supplied: {sheet.get('benchmark_supplied')}; comparable: {sheet.get('benchmark_comparable')}.",
        "",
        "## Claims",
        "",
    ]
    rows = sheet.get("rows") or []
    if not rows:
        lines.append("No answer was parsed for this record, so there are no claims to review. "
                     "See the record's `error` and `response.raw_content`.")
    for row in rows:
        lines += [
            f"### {row['claim_id']} · {row['section']} · {row['automatic_result']}",
            "",
            f"> {row['model_text']}",
            "",
            "**Evidence in the sent context:**",
            "",
        ]
        evidence = row.get("evidence_in_sent_context") or []
        lines += [f"- `{_cell(line)}`" for line in evidence] or ["- (none matched)"]
        lines += ["", "**Automatic findings:**", ""]
        lines += [f"- {_cell(item)}" for item in (row.get("automatic_findings") or [])] or ["- (none)"]
        lines += ["", "**Human judgement:** ", "", "**Author notes:** ", ""]
    return "\n".join(lines) + "\n"


CHECKLIST_STEM = "review_checklist"


def write_checklist(record: Mapping[str, Any], target_dir: Any, *, stem: str = CHECKLIST_STEM) -> Dict[str, str]:
    """Write ``<stem>.md`` and ``<stem>.csv`` next to a record; never overwrite.

    An existing sheet may already carry an author's judgements, so a second
    write to the same names is refused rather than silently replacing them.
    """
    from pathlib import Path

    folder = Path(str(target_dir))
    folder.mkdir(parents=True, exist_ok=True)
    sheet = build_checklist(record)
    targets = {"markdown": folder / f"{stem}.md", "csv": folder / f"{stem}.csv"}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite an existing review sheet: {', '.join(existing)}.")
    targets["markdown"].write_text(checklist_markdown(sheet, record), encoding="utf-8")
    # utf-8-sig so a spreadsheet application opens the Chinese/English text correctly.
    targets["csv"].write_text(checklist_csv(sheet), encoding="utf-8-sig")
    return {name: str(path) for name, path in targets.items()}
