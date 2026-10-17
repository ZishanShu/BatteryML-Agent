# LLM explanation records and review sheets

Spec: `EXPL-01` (record), `EXPL-02` (review sheet). Code:
`batteryml_web/agent/explanation_record.py`,
`batteryml_web/agent/explanation_checklist.py`, the Explain-metrics branch of
`render_review_results_page` in `batteryml_web/app.py`, and the optional
`exchange` recorder of `batteryml_web/agent/llm_client.run_llm_prompt`.

This document is written from the source code `[STATIC]`. No record has been
produced from a real provider in the development environment, which has
neither an API key nor battery data.

## 1. What Review Results → Explain metrics does

| Step | Where | What happens |
| --- | --- | --- |
| Button | `app.py`, `render_review_results_page`, key `review_explain_metrics` | Only when the LLM state allows an action (`llm_action_allowed`); otherwise the settings dialog opens and nothing is sent. |
| Context assembly | `app.py` `_selected_analysis_context` → `agent/result_analysis.build_analysis_context` | Built from the **selected run's** `run.json` (dataset, task, model key, config path, seed, device, status, saved metrics) and its **selected** prediction artifact (diagnostics via `agent/prediction_diagnostics.artifact_diagnostics`), plus the config's protocol fields (`read_protocol`) and the benchmark registry (`core/benchmarks.assess_comparability`). |
| Prompt | `agent/result_analysis.build_result_analysis_prompt` | Plain text, sections in this order: RUN identity; metrics verbatim with units; protocol; prediction diagnostics; benchmark; platform warnings; the five-part instruction and the JSON schema. |
| System prompt | `agent/result_analysis.ANALYSIS_SYSTEM_PROMPT` | Numbers only from the context, verbatim, never recomputed/rounded/rescaled; claims graded FACT / DERIVED / HYPOTHESIS; benchmarks only as supplied; no ranking; no smoke-run performance claims. |
| Call | `agent/result_agent.explain_run_result` → `agent/llm_client.run_llm_prompt` | One `POST <Base URL>/chat/completions` with `{"model", "messages": [system, user], "stream": false, "temperature": 0.2}`, timeout 60 s. No `max_tokens`, `top_p`, `seed`, `response_format` or other settings are sent, so the provider's defaults apply to them. |
| Parsing | `agent/result_agent._extract_json` | The reply must be a JSON object (a code fence is tolerated). |
| Display | `agent/result_analysis.render_analysis_markdown` → `render_llm_output_panel` | Five headed sections; empty ones are omitted. |

### What the model receives

- Run identity: project name, run id, dataset, task label (for example
  `RUL prediction`), model **key** (for example `xgb`), config path, seed,
  device, run status, and the smoke-run sentence if the run used `--epochs 1`.
- Metrics exactly as saved in `run.json`, at full precision (`repr`), each
  with the unit the platform assigns: RUL RMSE and MAE `[cycles]`, MAPE
  `[ratio (multiply by 100 for percent)]`; SOH MAE and RMSE `[SOH ratio]`.
- Protocol from the executed config: train/test splitter class, cell data
  path, feature extractor class, label annotator class, model class.
- Diagnostics from the selected prediction artifact: predicted and observed
  counts, finite/non-finite counts, distinct values, constant flag, min and
  max; sample count; distinct cells; the artifact's task metadata.
- The benchmark block: the published BatteryML reference with its source,
  subset, metric, unit and protocol, and either the run value and the
  difference (when comparable) or the reason it is not comparable.

### What the model does not receive

The predictions or labels themselves (only the summary statistics above),
any log, the raw or processed battery data, the dataset's description, and
the **definition of the label** — the prompt names the annotator class
(`RULLabelAnnotator`) but not what it computes (`batteryml/label/rul.py`:
cycles until the largest discharge capacity first falls to 80 % of nominal).
Anything the answer says about those comes from the model, not the context.

## 2. The record

Every Explain-metrics click that reaches the request path writes:

```
runs/web_projects/<project>/llm_explanations/<run id>/<record id>/
    record.json            # the record (schema batteryml-agent/llm-explanation-record@1)
    review_checklist.md    # the review sheet, readable
    review_checklist.csv   # the same rows, for a spreadsheet (UTF-8 with BOM)
```

The record is written for a failed call too: the request may not have been
sent, the provider may have returned an error, or the reply may not be valid
JSON. A failure is never replaced by an answer. A record is never
overwritten. `runs/` is not in git.

| Section | Fields |
| --- | --- |
| `result` | project, project root, run id, run dir, page identity key, dataset, task, model key, seed, device, run status, the commit recorded **when the run was trained** (`run_git_commit`), the run command, the metrics in `run.json` |
| `code` | commit of the checkout that **sent the request**, and whether tracked files were modified (`git status --porcelain`) |
| `files` | path, size, modification time and SHA-256 of `run.json`, the config named in `run.json`, every `config_*.yaml` BatteryML saved in the run directory, and the selected prediction artifact |
| `call` | start/finish of the click, request sent/received time, elapsed seconds, failure stage, HTTP status, provider label, endpoint (query string and userinfo removed), **requested model**, **response model** (the response's own `model` field, or a note that the provider did not return one), response id, `system_fingerprint`, finish reason, token usage, the generation settings **as sent**, the timeout, the settings **not** sent |
| `request` | the exact `messages` array sent, and the deterministic analysis context the user message was built from |
| `response` | the raw `message.content`, the parsed JSON, the markdown shown on the page |
| `error` | stage, message, the provider's error body (redacted) |
| `human_review` | `{"status": "NOT STARTED"}` — never changed by the software |

**Never stored:** the API key and every request header. The client copies
the JSON body only, removes the key and anything key-shaped from provider
error text, and the record is refused before writing if the key appears in it.

## 3. The review sheet

One row per sentence of the answer, with the columns
`claim_id, section, model_text, evidence_in_sent_context, automatic_result,
automatic_findings, human_judgement, author_notes`. The last two are empty.

Automatic checks compare the sentence with the **user message that was
sent**, and nothing else:

| Check | Finding |
| --- | --- |
| A value right after `RMSE`/`MAE`/`MAPE` | exact match, rounded (the system prompt asked for verbatim values), different, or quoted as percent where the sent unit is not percent |
| MAPE as a percent | consistent with the sent ratio × 100 at the shown precision, or the ×100 conversion is missing |
| Any other number | verbatim in the context; same value, different formatting; rounded from a context value; a simple difference/ratio of context numbers (marked derived); or not found |
| Units | a non-cycle unit next to an RUL RMSE/MAE; `cycles` next to MAPE or an SOH metric |
| Task definition | wording such as "remaining useful life", "cycle life", "end of life", "80%": the context names only the task and the annotator class; terms absent from the context are flagged, with a pointer for the author to the label code (marked *not sent to the model*) |
| Mechanism | physical/chemical terms (plating, SEI, LLI, chemistry, temperature, …) absent from the context; whether the sentence is labelled as a hypothesis |
| Causal wording | "because", "due to", … — flagged unless hedged as a hypothesis |
| Generalisation | "generalize", "robust", "reliable", "real-world", "outperform", "best", … |
| Benchmark | a benchmark statement when no reference was supplied, or comparative wording against a reference marked not comparable |

`automatic_result` is `FLAG`, `NOTE` or `NO_AUTOMATIC_FINDING`. The last
means no pattern fired; it is not a judgement that the sentence is correct.

## 4. Producing a case for the manuscript

1. Use a **completed, non-smoke** MATR · RUL prediction · XGBoost run with
   `configs/baselines/sklearn/xgb/matr_1.yaml` (README §7.4). That config's
   splitter is the MATR1 subset, for which the benchmark block supplies a
   comparable published reference.
2. Connect the LLM (README §12). Open **Review Results**, select that run,
   click **Explain metrics** once.
3. The page shows `Explanation record (answer): <path>`, or
   `(failed)` with the reason. Keep the record either way.
4. Review `review_checklist.md`/`.csv`: fill **Human judgement** and
   **Author notes**. Only the author changes `human_review`.
5. To rebuild a sheet into a new folder (never over an annotated one):

   ```bash
   python -m scripts.explanation_review --list runs/web_projects/<project>
   python -m scripts.explanation_review --show <record.json>
   python -m scripts.explanation_review --record <record.json> --out runs/explanation_review/<name>
   ```

Checks: `tests/test_explanation_record.py`; `RUN-01` in
`python -m scripts.verify.run_all` includes the record contract.
