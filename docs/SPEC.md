# `docs/SPEC.md` — the standard

`CLAUDE.md` §2 names this file as the single source of truth for what "done"
means, and cites stable requirement IDs throughout. **It did not exist**; that
is priority 3 in `CLAUDE.md` §11. This is the minimum restoration: it establishes
the ID scheme, records the requirements that today's code actually implements,
and gives later work somewhere to add to. It does **not** attempt to
reverse-engineer a full specification for the whole platform — inventing
requirements to match existing code would produce a document that always says
everything is done.

## How to use it

* Work is scoped by ID. Before writing code, state which IDs you are
  implementing.
* After writing code, set each ID's status and name the implementing file.
* Only the user may set `VERIFIED`, and only after pasting back real output from
  the server (`CLAUDE.md` §0).
* A requirement that is missing gets a new ID and a rationale, never a silent
  implementation.
* A requirement that turns out to be wrong gets an amendment proposal, never a
  quiet substitution.

## Status values

| Status | Meaning |
| --- | --- |
| `NOT-STARTED` | No implementation exists. |
| `PARTIAL` | Some of it exists; the gap is stated in the row. |
| `IMPLEMENTED-STATIC` | Implemented and established by reading the code. Never run against real data by the author. |
| `BLOCKED` | Cannot proceed; the blocker is stated. |
| `VERIFIED` | The user ran it on the server and pasted back the output. **Only the user sets this.** |

## Two ID namespaces, deliberately separate

| Namespace | What it identifies | Where it lives |
| --- | --- | --- |
| `VIZ-nn`, and future feature prefixes | a **requirement** — something the platform must do | this file |
| `ENV-01`, `DATA-01`, `CFG-01`, `VAL-01`, `SPLIT-01`, `REG-01`, `EXT-01`, `CLI-01`, `FIG-01`, `DEV-01`, `INTENT-01` | a **check** — something the verification harness establishes | `docs/VERIFY.md` |

They are the same shape, so keep the prefixes disjoint: the figure *requirements*
are `VIZ-01`…, and the figure *check* that verifies them is `FIG-01`. Nothing is
both.

---

## VIZ — visualisation

Rationale for the group: the protocol's outputs are numbers, and a reviewer
cannot judge a battery-ageing prediction from three aggregate metrics alone. A
parity plot shows whether the error is uniform or concentrated in the long-lived
cells; a per-cell error chart shows whether one outlier carries the RMSE. The
`figures/` directory is already part of the reproducibility contract
(`CLAUDE.md` §8) and, before this group, nothing wrote into it.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `VIZ-01` | Figure filenames are declared in one place, are safe and unique, and are treated as part of the manuscript: adding one is a code change, renaming one is a code **and** paper change. Every generated figure is recorded in `figure_manifest.json` with the artifact it came from, that artifact's SHA-256, the git commit, the seed, the label space and unit, and the figure's own SHA-256. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/contract.py`, `batteryml_web/viz/render.py` |
| `VIZ-02` | Predictions are plotted in the label space the recorded metrics were computed in. The label transformation carried by the artifact is inverted, and the result is proved correct by recomputing the recorded metrics with BatteryML's own evaluator and refusing to draw anything if they disagree. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/loaders.py` (`load_prediction_artifact`, `verify_recorded_scores`) |
| `VIZ-03` | A per-cell figure names the right cell. The NaN-label mask `Task.build` applies is reproduced rather than assumed, and the recomputed labels are cross-checked against the artifact's own. Where identity cannot be established the figure is skipped with a stated reason and never drawn against guessed labels. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/loaders.py` (`resolve_test_cell_ids`) |
| `VIZ-04` | Result figures — parity, residuals, per-cell absolute error, and a metric summary — are generated offline from a prediction artifact, with no API key, no Streamlit and no network (Route B). Figures never compute an aggregate metric; they display the evaluator's values verbatim. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/figures.py`, `batteryml_web/viz/cli.py` |
| `VIZ-05` | Data-side figures — capacity fade, cycle-life distribution, train/test split balance, single-cell detail, feature matrix — generated from `data/processed/<DATASET>` before any training. | `NOT-STARTED` | — |
| `VIZ-06` | A Streamlit page displays the generated figures, and offers an interactive layer for exploring individual cells. The page degrades to the static figures when the interactive library is absent, so Route B never depends on it. | `NOT-STARTED` | — |
| `VIZ-07` | `report.html` assembles the run's figures, metrics and provenance into the single reviewer-facing artifact `CLAUDE.md` §8 names. Nothing writes it today. | `NOT-STARTED` | — |
| `VIZ-08` | Cross-model comparison over several runs of one dataset, read from their manifests. Introduces the first categorical palette, which must be validated before it ships. | `NOT-STARTED` | — |

### Notes on the VIZ group

* `VIZ-02` and `VIZ-03` exist as separate requirements because they are separate
  failure modes with the same symptom — a figure that looks entirely reasonable
  and is wrong. `VIZ-02` puts the axes in the wrong space; `VIZ-03` puts the
  right numbers under the wrong names.
* The check that establishes `VIZ-01`…`VIZ-04` is `FIG-01`
  (`scripts/verify/check_figure_contract.py`).
* No requirement in this group may be marked `VERIFIED` from this environment:
  it has no numpy, no matplotlib and no torch, so no figure has ever been drawn
  here. The command block that would verify them is in `docs/VISUALIZATION.md`.

---

## EXEC — execution runtime (smoke runs, devices, failures)

Rationale for the group: the platform's one irreversible action is starting a
BatteryML run. Everything about *how* that run is started — which device, how
many epochs, what happens when it fails — must be decided by the backend and
displayed honestly, because this is exactly where a plausible-looking shortcut
(a retried OOM, a mutated config, a fabricated percentage) would silently
change what the manuscript reports.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `EXEC-01` | A smoke run is one epoch of the model's **own formal config**. The only override is the planner's `--epochs 1`; no config file is written, copied or mutated, and no batch size, support size or learning rate is touched. A smoke run therefore exercises the same code path a benchmark run does. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/run_policy.py` (`SMOKE_EPOCHS`, `smoke_epochs_for`), `batteryml_web/agent/execution.py` (`build_execution_plan`) |
| `EXEC-02` | Every surface that offers or reports a smoke run shows the sentence "1-epoch smoke test — workflow validation only; not benchmark performance." A smoke metric is never presented as benchmark performance. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/run_policy.py` (`SMOKE_NOTICE`), `batteryml_web/ui/agent_workflow.py` |
| `EXEC-03` | BatLiNet's smoke run defaults to and recommends CPU, with the reason stated. An explicit GPU choice is always honoured; the recommendation never selects a device on the user's behalf. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/run_policy.py` (`SMOKE_CPU_RECOMMENDED_MODELS`, `smoke_device_recommendation`) |
| `EXEC-04` | Selectable devices are enumerated from the machine (`nvidia-smi`), never from a hardcoded GPU count, and each is shown with its observed free/total memory. Unknown memory is displayed as unknown, never as zero. When enumeration fails, an explicit GPU choice remains possible and the enumeration error is shown beside it. | `IMPLEMENTED-STATIC` | `batteryml_web/core/device_inventory.py`, `batteryml_web/ui/agent_workflow.py` (`device_options`) |
| `EXEC-05` | A failed run is never retried — not on another GPU, not on the same one. A CUDA out-of-memory failure is named as a shared-memory condition, states the CPU-smoke and larger-GPU alternatives, quotes the run's own error line, and leaves the full stdout/stderr (traceback included) available for inspection. No metric is ever derived from a failed run. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/run_policy.py` (`classify_execution_failure`), `batteryml_web/agent/executor.py` |
| `EXEC-06` | A run reports progress while it runs. The backend executes the same validated plan through `subprocess.Popen` (argv list, `shell=False`) and emits structured events — `stage` / `training_progress` / `output` / `completed` — on the caller's own thread. The executor stays Streamlit-independent; the page renders the events and holds no execution logic. stdout and stderr are preserved in full, carriage returns included, so a tqdm bar and a traceback both survive. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/streaming.py`, `batteryml_web/ui/agent_workflow.py` (`_LiveProgress`) |
| `EXEC-07` | A percentage is displayed only where the run itself printed a counter. The real tqdm lines the installed models emit are recognised (`Training <Model>: …`, and the upstream `Traning: …` typo in `batteryml/models/nn_model.py`), tqdm's stderr carriage returns included. Where no counter exists, a stage name and an elapsed clock are shown and **no** bar is drawn. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/streaming.py` (`parse_progress_line`, `classify_stage`), `batteryml_web/ui/agent_workflow.py` (`progress_caption`) |
| `EXEC-08` | Progress output is never a source of scientific truth. Metrics come only from `build_result_context`, which reads BatteryML's own saved `scores`. A failed run ends at stage `Failed` with its return code and stderr intact and no metric of any kind. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/streaming.py`, `batteryml_web/agent/results.py` |

The check that establishes `EXEC-01`…`EXEC-08` is `DEV-01`
(`scripts/verify/check_runtime_contract.py`); the unit tests are
`tests/test_agent_smoke.py`, `tests/test_agent_streaming.py` and the
milestone-A/B classes in `tests/test_agent_web_ui.py`. None of them may be marked `VERIFIED` from this
environment: it has no torch, no GPU and no Streamlit, so no run has ever
started here.

---

## UI — entry points and feedback

Rationale for the group: the platform has two legitimate ways in — the
deterministic quick start a reviewer follows with no API key (Route B), and
the natural-language agent (Route A). Presenting the agent as "step 3 of 4"
made it look like a mandatory stage of the manual route, and buried the
product's main entry point inside a sequence. These requirements fix where
each route is offered, not what either one does.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `UI-01` | Home offers Quick Start **and** the Agent Workflow at the same visual hierarchy: a free-text request ("Describe what you want to do...", placeholder "Use CPTransformer on MATR to predict battery lifetime.") and an **Analyze with Agent** button. Quick Start's behaviour is unchanged. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`render_home_agent_entry`) |
| `UI-02` | Home records the request in session state and navigates; it resolves no intent, reads no readiness, builds no plan and executes nothing. The Agent Workflow page analyses the handed-over request once on arrival through the existing backend; execution still requires the explicit Run button there. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py`, `batteryml_web/ui/agent_workflow.py` (`_analyze_request`) |
| `UI-03` | Page and section headings use native Streamlit primitives with `anchor=False`, so no heading anchor, link icon or raw markup is exposed in the rendered page. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py`, `batteryml_web/ui/agent_workflow.py` |
| `UI-04` | The sidebar shows the Agent Workflow as an alternative intelligent entry point in its own block, not as a numbered step of the protocol workflow. The manual route stays numbered 1–3 and remains fully usable with no LLM. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/workflow_sidebar.py` |
| `UI-05` | Creating a project confirms what was created — the project name and its workspace, drawn from the directory that was actually created — and the confirmation itself is transient (see `UI-09`). Project filesystem semantics are unchanged. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`_ensure_project`, `_render_project_created_notice`) |
| `UI-06` | The Agent Workflow entry is written exactly `Agent Workflow` wherever it appears — Home, sidebar, page heading, navigation buttons. No emoji, no caret, no numbering, and no copy explaining that it is an alternative route: one destination, one name. Quick Start and Agent Workflow remain equal primary Home entries. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/workflow_sidebar.py` (`nav_label`), `batteryml_web/app.py`, `batteryml_web/ui/agent_workflow.py` |
| `UI-07` | The request is supplied through **one** control: a single text area, no tabs and no example gallery. While it is empty it shows one suggestion as grey placeholder text, derived from the backend catalog and — where readiness is known — naming a model that is actually runnable; no model or dataset name is written into the UI. Superseded in part by `RUN-13`, which governs accepting that suggestion. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/catalog.py` (`build_request_suggestion`), `batteryml_web/ui/agent_workflow.py` (`request_suggestion`) |
| `UI-08` | The Streamlit developer toolbar is not offered to end users: `client.toolbarMode = "viewer"` in `.streamlit/config.toml`. Configured, not scripted — no JavaScript intercepts keystrokes. | `IMPLEMENTED-STATIC` | `.streamlit/config.toml` |
| `UI-09` | Success is transient and failure is persistent. An ordinary successful action reports through one `st.toast`, emitted once per occurrence; Streamlit reruns do not repeat it, and a signature keyed on the action's *subject* keeps a device or checkbox change from toasting. Failures stay on the page as `st.error` / `st.warning`, and a failure is never rendered in a success box. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/feedback.py` |

The tests for this group are `tests/test_web_navigation.py` (source- and
AST-level, so Home's boundary is checked without importing torch or
Streamlit), `tests/test_ui_feedback.py`, and the handover, request-tab and
feedback classes in `tests/test_agent_web_ui.py`. Nothing here may be marked
`VERIFIED` from this environment: no Streamlit page has been rendered in it,
and `UI-08` in particular can only be confirmed by looking at the running
app's toolbar.

---

## INTENT — understanding the request

Rationale for the group: the deterministic phrase table is precise and
rigid. It cannot read "Can you run a model for cycle aging on the MATR
cells?", an informal model name, or mixed Chinese/English. An LLM can — and
an LLM can also confidently name a dataset, a model or a config file that
does not exist. These requirements put the LLM where it helps (reading the
sentence) and keep every identifier it produces subject to the backend.

The order is fixed and must not be shortened:

    LLM interpretation -> structured candidate intent -> canonical catalog
    -> readiness -> validator -> user confirmation -> execution

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `INTENT-01` | The catalog offered to the LLM is built from the live backend — `ConfigRegistry` for models and their dataset/task coverage, the platform dataset list plus `list_dataset_readiness` for datasets, the three canonical task labels, and `readiness.EXTERNAL_MODELS` for external runners. No second, hand-maintained catalog exists, and no config path ever appears in it. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/catalog.py` |
| `INTENT-02` | The LLM is asked for structured JSON only: dataset, task, requested_model, needs_clarification, clarification_question, suggested_model_keys, user_facing_rationale. Hidden reasoning is never requested; the rationale is short, user-facing and length-capped. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/llm_intent.py` (`SYSTEM_PROMPT`, `build_intent_prompt`) |
| `INTENT-03` | Every identifier the LLM returns is resolved against the catalog before use. An unknown dataset, task or model key becomes `None` and is recorded under `rejected`; a config path, command, device, epoch count or metric is discarded and reported. A reply that is not valid JSON, or not an object, is discarded whole. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/llm_intent.py` (`validate_llm_intent`, `extract_json_object`) |
| `INTENT-04` | The deterministic resolver stays the fast path — a request it fully resolves costs no LLM call — and the fallback: when no LLM is configured, or the call or parse fails, its answer is returned and the page says plainly that advanced natural-language understanding is unavailable. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/llm_intent.py` (`resolve_workflow_intent_with_llm`), `batteryml_web/ui/agent_workflow.py` (`_render_intent`) |
| `INTENT-05` | Recommendations are filtered by readiness, not by the LLM. A suggestion that is not runnable for the resolved dataset and task is separated out and may be *mentioned* with its backend reason (PBT, NEEDS_ASSETS) but never offered as runnable. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/llm_intent.py` (`filter_suggestions`) |
| `INTENT-06` | A requested model that is blocked, or that is unknown to the catalog, is never silently replaced. The reason is stated, runnable alternatives are listed, `requires_confirmation` is set, and the user selects a replacement explicitly. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/llm_intent.py` (`resolve_model_request`), `batteryml_web/ui/agent_workflow.py` (`_render_model_request`) |
| `INTENT-07` | An unresolved dataset or task produces one concise clarification question. An explicit request is never defaulted to MATR. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/llm_intent.py`, `batteryml_web/agent/intent.py` |
| `INTENT-08` | Common Chinese and mixed-language phrasings resolve, in the **offline** route as well: the task phrase table carries the Chinese spellings, so "帮我用 MATR 做电池寿命预测" resolves to MATR + RUL prediction with no API key. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/intent.py`, `batteryml_web/agent/catalog.py` (`TASK_ALIASES`) |
| `INTENT-09` | Free-form LLM prose can never trigger a run. The intent layer imports no executor, calls no planner, and touches neither `subprocess` nor the filesystem; the canonical identifiers it produces still go through `build_agent_workflow_context`, `build_execution_plan`, the validator and the explicit Run button. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/llm_intent.py`, `scripts/verify/check_intent_catalog.py` |
| `INTENT-10` | A request that asks *which* model to use is recognised as a recommendation request — in English and in Chinese, deterministically, so the offline route behaves the same way. When an LLM is configured such a request reaches it even though the dataset and task already resolve, because choosing among runnable models is a judgement the phrase table cannot make. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/intent.py` (`detect_recommendation_request`), `batteryml_web/agent/llm_intent.py` |
| `INTENT-11` | The model section answers the request instead of listing the platform. A named runnable model is shown alone; a named blocked model is shown with its exact status and blockers, then runnable alternatives; an unregistered name is reported as such with the closest runnable models; a recommendation shows at most three, drawn from the agent's suggestions only where readiness keeps them. The full runnable and blocked catalogues live behind expanders. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/recommendation.py`, `batteryml_web/ui/agent_workflow.py` (`_render_model_presentation`) |
| `INTENT-12` | Nothing is ever substituted for a model the user asked for. Where the request named something blocked or unregistered, no model is preselected: the control opens on an empty choice and no plan exists — and therefore no Run — until the user selects one deliberately. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/recommendation.py` (`selectbox_options`, `NO_SELECTION`), `batteryml_web/ui/agent_workflow.py` |
| `INTENT-13` | No model name is written into recommendation logic. The presentation layer reads the readiness context and the catalog it is given, cannot consult readiness itself, and so cannot overrule it. `DEV`/`INTENT-01` fails if a model name appears there. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/recommendation.py`, `scripts/verify/check_intent_catalog.py` |

The check that establishes this group is `INTENT-01`
(`scripts/verify/check_intent_catalog.py`); the unit tests are
`tests/test_llm_intent.py` (57 cases, every LLM answer stubbed),
`tests/test_model_presentation.py` (22 cases) and the LLM-intent class in
`tests/test_agent_web_ui.py`. Nothing here may be
marked `VERIFIED` from this environment: no LLM endpoint has been called
from it.

---

## GPU — devices and parallelism

Rationale for the group: the development server has eight GPUs, which makes
two mistakes tempting. The first is hardcoding "8". The second is offering a
multi-GPU run that does not exist, on the theory that 8 × 2 GB free adds up
to 16 GB. The audit behind this group is `docs/GPU_MULTI_GPU_AUDIT.md`.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `GPU-01` | The machine's GPUs are enumerated at runtime — index, name, total/used/free memory — and any single one may be selected. The count is discovered from the output, never assumed; unknown memory is shown as unknown. | `IMPLEMENTED-STATIC` | `batteryml_web/core/device_inventory.py` (see also `EXEC-04`) |
| `GPU-02` | A run is sent to exactly one torch device. `cpu`, `cuda`, `cuda:<index>` and `mps` are accepted; a multi-device string such as `cuda:0,1` is refused at the planner boundary with the reason, rather than failing inside a run that has already started. | `IMPLEMENTED-STATIC` | `batteryml_web/core/device_inventory.py` (`validate_device`), `batteryml_web/agent/execution.py` |
| `GPU-03` | No multi-GPU Run option is exposed while none exists. The web layer contains no `DataParallel`, `DistributedDataParallel`, `init_process_group` or `torchrun` path, and the DEV-01 check fails if one appears. | `IMPLEMENTED-STATIC` | `scripts/verify/check_runtime_contract.py`, `tests/test_agent_smoke.py` |
| `GPU-04` | The reason multi-GPU is absent is written down, with the code it follows from and the work a real implementation would take: `DataBundle.to(device)` places the whole dataset on one device, so neither DataParallel nor DDP relieves the per-GPU memory limit; BatLiNet's inter-cell support sampling has no defined distributed semantics; DDP changes effective batch size and determinism. | `IMPLEMENTED-STATIC` | `docs/GPU_MULTI_GPU_AUDIT.md` |
| `GPU-05` | Device selection is a structure — `{"mode": "cpu" \| "single_gpu" \| "multi_gpu", "gpu_ids": [...]}` — separate from the `--device` string BatteryML receives. CPU and one GPU produce a device string; a multi-GPU selection produces none, so `"cuda:0,1"` cannot be constructed. | `IMPLEMENTED-STATIC` | `batteryml_web/core/device_inventory.py` (`build_device_selection`, `device_string_for_selection`) |
| `GPU-06` | The page offers CPU, one GPU or several. Compute device is an explicit CPU/GPU choice (`RUN-04`); in GPU mode a multiselect lists the enumerated devices with index, name, total/used/free memory and utilization. Single-GPU and CPU behaviour is unchanged. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py`, `batteryml_web/core/device_inventory.py` |
| `GPU-07` | Whether a multi-GPU selection may execute is a backend capability (`supports_multi_gpu`, `multi_gpu_strategy`), and no model carries it today. A multi-GPU selection for a model without it disables Run with "True multi-GPU execution is not yet supported for this model/runtime." — and the selection is never quietly reduced to a single GPU. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/run_policy.py` (`multi_gpu_capability`, `execution_blockers`) |

`GPU-05`…`GPU-07` prepare for a real multi-GPU implementation without
claiming one: the selection shape and the capability gate are what a DDP
milestone would fill in, and `DEV-01` fails if a model is registered as
capable while no DDP path exists.

`GPU-04` is a documentation requirement on purpose: the decision *not* to
build something is the part of an audit that gets lost, and the next person
to look at eight idle GPUs needs the reasoning, not just the absence.

---

## RUN — runtime context, provenance and honest reporting

Rationale for the group: the previous milestones made the platform *able* to
run something. This one makes it able to say, truthfully and before the fact,
**what** it is about to run and **where** the result will land — and to stop
short of saying anything it has not observed.

Three failures motivate it. A project could be loaded on Home while the run
was written under the default project, because the page carried a second
workspace rule with its own fallback path; the two answers only had to
disagree once for a result to be attributed to the wrong experiment. A
progress line with a remaining time is read as a measurement, so inventing
one from a constant rate is the same class of error as inventing a metric.
And "Explain computed result" was disabled with a caption telling the user to
configure an assistant the page offered no way to configure — a dead end
that made the LLM route look broken rather than optional.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `RUN-01` | The loaded project is the only source of an agent run's workspace: `runs/web_projects/<project>/agent_<dataset>_<model>[_smoke]`. No second rule, no fallback path written into the page. A project root that would leave `runs/` is brought back inside it, and an unnameable project raises rather than silently becoming the default. | `IMPLEMENTED-STATIC` | `batteryml_web/core/project_context.py` (`resolve_project`, `agent_workspace`, `safe_segment`) |
| `RUN-02` | Provenance — Project, Run directory, Dataset, Task, Model, Config, Seed, Epochs, Device — is shown **before** the run is started and preserved **with** the result, so the two can never disagree. The run directory is read out of the plan's own argv, because the argv is what executes. A value the backend does not have renders as "Not available", never as a blank cell. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/provenance.py` (`build_run_context`, `context_rows`), `batteryml_web/ui/agent_workflow.py` (`_render_run_context`) |
| `RUN-03` | Paths are shown repo-relative where the file is inside the repository; the absolute form stays available in a details view rather than being discarded. | `IMPLEMENTED-STATIC` | `batteryml_web/core/project_context.py` (`display_path`), `batteryml_web/agent/provenance.py` (`artifact_display`) |
| `RUN-04` | Compute device is an explicit choice — CPU or GPU — not the absence of one. CPU is the default and needs no GPU interaction. The GPU control appears only in GPU mode, and its entries come from the runtime inventory: index, name, free memory and utilization exactly as the driver reported them, with any figure the driver did not report simply omitted. No GPU count, index, name, memory figure or utilization is written into the source. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py`, `batteryml_web/core/device_inventory.py` (`gpu_option_label`) |
| `RUN-05` | An estimated time to completion is shown only where the run itself has produced enough observations to support one — a counter, and more than one measured interval. Until then the line says `estimating…`. Where no counter exists at all, no ETA and no bar are shown. A rate is never assumed from a constant. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/progress_estimate.py` (`EpochTimer`, `progress_line`, `ESTIMATING_LABEL`) |
| `RUN-06` | A metric is displayed at reading scale beside its exact stored value. The compact form is presentation only: the canonical value from `build_result_context` is never rounded, replaced or re-derived, and MAPE's percent form is a supplementary reading of the stored ratio. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/formatting.py` (`compact_metric`, `full_precision`, `metric_display`) |
| `RUN-07` | User-facing settings live in the sidebar, in the application's own UI. Streamlit's developer menu is not product UI and is not relied on (see `UI-08`); no script drags, hides or rewrites Streamlit's own DOM elements. The top chrome is reduced through documented, stable `data-testid` selectors only. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`render_sidebar_settings`), `batteryml_web/app.py` |
| `RUN-08` | "Explain computed result" is always clickable. With no assistant configured it opens the LLM configuration form (Cancel / Save settings / Save & explain) instead of dead-ending; it never calls anything before the user saves. Both entry points — sidebar and prompt — read and write the one existing credential store, so there is no second settings system. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py`, `batteryml_web/ui/llm_state.py`, `batteryml_web/ui/agent_workflow.py` (`_explain_result`) |
| `RUN-09` | The API key is entered masked (`type="password"`), is reduced to a boolean anywhere the page may describe the settings, and is never written to a caption, toast, log, debug view or repository file. It is held in session state for the session only — the mechanism the app already used — and this milestone adds no new persistence. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`llm_settings_summary`, `render_llm_settings_form`) |
| `RUN-10` | Creating or loading a project confirms which project is now current — `Project "<name>" created and loaded` / `Project "<name>" loaded` — once, transiently (see `UI-09`), and the current project stays visible in the sidebar for the rest of the session. Loading a project clears the previous project's plan, result and run context rather than leaving them attributed to the new one. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py`, `batteryml_web/ui/workflow_sidebar.py` (`current_project_label`) |
| `RUN-11` | The LLM settings form offers **Save** and **Connect**, in that order, wherever settings are edited — the sidebar and the prompt opened from "Explain computed result" alike. There is no primary Cancel / Save settings pair. Save stores the entered values through the existing credential store and makes no network call. Connect stores the same values through the same function and then runs the connection test; success sets status Connected and reports transiently, failure leaves an actionable persistent error. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`render_llm_settings_form`, `SAVE_LABEL`, `CONNECT_LABEL`) |
| `RUN-12` | There is one connection implementation. Connect calls `agent.llm_client.test_llm_connection` — the platform's existing test — and no page opens a socket, builds a request or parses a response of its own. The API key is password-masked, reduced to a boolean wherever the settings are described, and never appears in a caption, toast, log, error message or toast signature. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`connect_llm`, `llm_settings_signature`), `batteryml_web/app.py` (`render_inline_llm_settings`) |
| `RUN-13` | Accepting the request suggestion fills the box and nothing else: it writes the one canonical request key (`agent_original_request`) and the box's own widget key, and does not analyse, call the LLM, build a plan or execute. Analyze stays a separate explicit action. Acceptance is refused whenever the field already holds text, re-checked at the moment it is applied. **Tab is not intercepted** — see `docs/STREAMLIT_KEYBOARD_AUDIT.md` — and the page never claims that it is. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py` (`accept_suggestion`, `ACCEPT_SUGGESTION_KEY`), `docs/STREAMLIT_KEYBOARD_AUDIT.md` |
| `LIFE-01` | Every explicit execution gets a unique, readable, filesystem-safe run id (`run_<date>_<time>_<model>_<dataset>_seed<n>_<suffix>`), minted when the run is created and never during Analyze. Two runs of an identical configuration in the same second still differ. Model and dataset alone are never a run identity. | `IMPLEMENTED-STATIC` | `batteryml_web/core/run_identity.py` |
| `LIFE-02` | A run writes to `runs/web_projects/<project>/agent_<dataset>_<model>/<run_id>/`, so repeated runs never overwrite one another, and the loaded project stays authoritative (`RUN-01`). Artifact discovery is unchanged — it walks a workspace recursively — so older workspaces without a run level remain readable with no migration. The executed plan points at that directory, so the plan reviewed is the plan that runs. | `IMPLEMENTED-STATIC` | `batteryml_web/core/project_context.py` (`run_workspace`) |
| `LIFE-03` | Each run carries a `run.json` manifest with its identity, configuration, lifecycle timestamps, return code, artifacts, metrics when completed and git commit. It is written atomically (temp file in the same directory, fsync, `os.replace`), so an interrupted write never leaves corrupt JSON. Fields are named explicitly, so no credential has a route into the file. | `IMPLEMENTED-STATIC` | `batteryml_web/core/run_manifest.py` |
| `LIFE-04` | Status is one of PLANNED / RUNNING / COMPLETED / FAILED / CANCELLED / INTERRUPTED, recorded rather than inferred from a directory's existence, with checked transitions: nothing reaches COMPLETED except from RUNNING, and a terminal run cannot be reopened. A manifest left at RUNNING by a process that no longer tracks it reports INTERRUPTED — never COMPLETED — and the stored file is not rewritten by the reporting. | `IMPLEMENTED-STATIC` | `batteryml_web/core/run_manifest.py` (`ALLOWED_TRANSITIONS`, `reconcile_status`) |
| `LIFE-05` | Cancel terminates the run's real process. The child is started in its own session, so the cancel signals exactly that process group — SIGTERM, then SIGKILL after a grace period — and never a name pattern (`pkill`/`killall` appear nowhere on the run path). Output already produced is kept, artifacts are labelled partial, no result context is built and no metric is offered. | `IMPLEMENTED-STATIC` | `batteryml_web/core/run_manager.py` (`cancel`, `_signal_process`) |
| `LIFE-06` | Execution runs on the manager's thread, not the Streamlit script's, which is the only reason Cancel can be clicked at all. The page polls structured state through `st.fragment(run_every=…)` — a bounded timed refresh, not a rerun loop. The manager imports no Streamlit. | `IMPLEMENTED-STATIC` | `batteryml_web/core/run_manager.py`, `batteryml_web/ui/agent_workflow.py` (`_render_active_run`) |
| `LIFE-07` | A project-scoped Run History lists every recorded run newest-first with its id, start, dataset, task, model, seed, epochs, device, status, duration and metrics — every value read from that run's manifest, none written into the page. Opening a run shows its provenance and outcome and never re-runs it. One project never lists another's runs. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/run_history.py`, `batteryml_web/app.py` (`_render_project_run_history`) |
| `LIFE-08` | History is filesystem state, so a browser refresh or an application restart does not lose it. Live process state and persisted manifest state are distinguished: a run this process no longer tracks is reported as such rather than shown with a spinner forever. | `IMPLEMENTED-STATIC` | `batteryml_web/core/run_manifest.py` (`list_runs`), `batteryml_web/ui/run_history.py` |
| `SCI-01` | Scientific views are built by a UI-independent adapter from the canonical saved predictions artifact, read through the platform's existing CPU-safe unpickler. Streamlit never parses the artifact, and the CUDA-remapping loader has exactly one implementation. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py`, `batteryml_web/agent/results.py` (`load_prediction_payload`) |
| `SCI-02` | For a scalar-label run the page shows observed-vs-predicted with a y=x reference, residuals against observed, and a per-sample table. Values are inverse-transformed exactly as `DataBundle.evaluate` does before scoring, so the axes are in the same space as the reported metrics, and the figure says so. Residuals are a display of two stored arrays, never a recomputed score. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`build_rul_view`), `batteryml_web/ui/run_history.py` (`render_rul_figures`) |
| `SCI-03` | An artifact that cannot support a view produces the sentence "This run does not contain the values required for this visualization." and no figure. Ground truth is never derived, predictions are never plotted against themselves, and a sample identifier is shown only where the artifact carries one. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`UNAVAILABLE_MESSAGE`, `sample_ids`) |
| `SCI-04` | Trajectory views accept only a genuine sequence-per-sample schema. A scalar RUL output is refused rather than reinterpreted as a trajectory. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`detect_schema`, `build_trajectory_view`) |
| `SCI-05` | Reproducibility metadata records the Python version, platform and an allow-listed set of package versions via `importlib.metadata`, plus the git commit, seed, config, workspace and timestamps. Environment variables are never iterated, so no credential has a route into it. A dataset fingerprint is file list and sizes only — contents are not hashed, and that is stated rather than implied. | `IMPLEMENTED-STATIC` | `batteryml_web/core/reproducibility.py` |
| `SCI-06` | A completed run exports a zip containing its README, manifest, metrics, environment, git commit, config snapshot, logs and (where the values are paired) a predictions CSV. The raw dataset is never included and checkpoints are excluded by default; anything omitted is named in the README with the path where it remains. Files are streamed into the archive, and only files under that run's directory are considered — a bundle for one run cannot contain another's. | `IMPLEMENTED-STATIC` | `batteryml_web/core/reproducibility.py` (`write_bundle`) |
| `SCI-07` | The artifact browser lists only files that exist, and every page-supplied path is resolved and proved inside the run directory with `relative_to` (not `startswith`) before any read. `..`, absolute paths, a sibling sharing a name prefix, and symlinks pointing out of the run are all refused; an escaping symlink is not even listed. | `IMPLEMENTED-STATIC` | `batteryml_web/core/artifact_browser.py` |
| `ROB-01` | Failures are classified into one explicit vocabulary — CUDA OOM, unavailable CUDA device, dataset unavailable, missing model assets, invalid config, process failure, cancelled, the four LLM categories, and `unclassified`. Execution failures are not re-derived: the existing `classify_execution_failure` verdict is mapped into the shape, so two classifiers cannot drift apart. Anything unrecognised is `unclassified`, which is an answer, not a guess. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/failure_taxonomy.py` |
| `ROB-02` | A failed run shows a plain statement and a suggested action first, with return code, argv, stderr and the log path behind "Show technical details" — kept, never removed. The page states outright that nothing was retried and no device was changed. No recommendation is ever carried out automatically. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py` (`_render_execution_status`) |
| `ROB-03` | Anything key-shaped is redacted from every message, evidence line and recommended action before display. Full logs stay in the run directory. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/failure_taxonomy.py` (`redact`) |
| `ROB-04` | An LLM failure is never a run failure. A failed explanation is reported beside the result with its own category and action; the run's status and its canonical metrics are untouched, because they come from BatteryML's saved scores. Intent resolution falls back to the deterministic resolver and says that it did. An unconfigured assistant is reported as a state, not a fault. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/failure_taxonomy.py` (`classify_llm_failure`), `batteryml_web/agent/llm_intent.py`, `batteryml_web/ui/agent_workflow.py` |
| `ROB-05` | Training is never retried automatically, on any device. An LLM authentication failure is explicitly not retried either, because repeating a rejected credential can lock an account. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/run_policy.py`, `batteryml_web/agent/failure_taxonomy.py` |
| `ROB-06` | Changing the request, project, model, seed, epochs, device or config clears the previous run's result, explanation, provenance and reserved run id, so a stale metric is never displayed beside a new selection. Completed runs remain reachable through Run History, which renders from a manifest and never re-runs anything. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py` (`_analyze_request`), `batteryml_web/ui/run_history.py` |
| `ROB-07` | After a restart, completed, failed and cancelled runs remain discoverable from their manifests, and a run whose process cannot be verified is reported INTERRUPTED — never COMPLETED. No pid read from a manifest is ever signalled: the only pid used comes from a live process handle this process created. | `IMPLEMENTED-STATIC` | `batteryml_web/core/run_manifest.py`, `batteryml_web/core/run_manager.py` |
| `ROB-08` | Resume is audited and **not implemented**. `NNModel.load_checkpoint` restores weights only — no optimizer state, no epoch counter, no RNG state — sklearn/XGBoost models have no equivalent, and the web CLI exposes no resume flag. No Resume control is offered merely because checkpoint files exist. | `IMPLEMENTED-STATIC` | `docs/RESUME_AUDIT.md` |

The check that establishes this group is `RUN-01`
(`scripts/verify/check_run_context.py`); the tests are
`tests/test_run_context.py` (the four backend modules, without Streamlit)
and, in `tests/test_agent_web_ui.py`, the classes
`TestComputeDeviceSelector`, `TestProjectDeterminesTheWorkspaceInTheUi`,
`TestExplainOpensConfigurationInsteadOfDeadEnding` and
`TestNothingRuntimeIsHardcoded`. The last is written to fail if anyone later
writes a GPU count, a GPU name, a device string, a project path or a metric
value into the UI source, and its scan is proved non-vacuous by mutation.

`RUN-04` deliberately does **not** claim multi-GPU execution. A multi-GPU
selection is still preserved as `{"mode": "multi_gpu", "gpu_ids": [...]}`,
still produces no device string, and still disables Run — see `GPU-05`…
`GPU-07`. **True multi-GPU execution remains unimplemented.**

`RUN-13` is the one requirement in this document that records a capability
the platform deliberately does **not** have. Tab-key acceptance was specified
and was not built, because no stable API in the installed Streamlit supports
it; `docs/STREAMLIT_KEYBOARD_AUDIT.md` carries the evidence and the three
conditions that would unblock it. A visible accept control is the documented
substitute, never a silent one.

Most rows in this group still need a rendered app before they can be marked
`VERIFIED`. `RUN-11`, `RUN-13` and the `UI-07` request area are the
exception: they were exercised in a headless Chromium against the running
app during development — no tabs rendered; one text area carrying the derived
suggestion as its placeholder; Tab moving focus without filling; the accept
control filling the box and then disappearing; typed text surviving a click
on it; Save and Connect present with one masked field. Those observations
were made in the authoring container, not on the server, so the rows stay
`IMPLEMENTED-STATIC` until the user reproduces them there.

---

## Where the moving parts are described

`docs/AGENT_WORKFLOW.md` walks the whole path from a sentence to a validated
run, says which module owns which decision, and carries the command block the
user runs on the server to turn any of these rows into `VERIFIED`.
`docs/GPU_MULTI_GPU_AUDIT.md` carries the device and parallelism audit behind
the `GPU` group.

---

## Amendments

Record here any requirement found to be wrong or impossible, with the proposed
change, rather than implementing something different.

*(none yet)*
