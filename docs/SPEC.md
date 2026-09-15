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
| `ENV-01`, `DATA-01`, `CFG-01`, `VAL-01`, `SPLIT-01`, `REG-01`, `EXT-01`, `CLI-01`, `FIG-01`, `DEV-01` | a **check** — something the verification harness establishes | `docs/VERIFY.md` |

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
| `UI-05` | Creating a project shows an explicit success state naming the project and its workspace, drawn from the directory that was actually created. Project filesystem semantics are unchanged. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`_ensure_project`, `_render_project_created_notice`) |

The tests for this group are `tests/test_web_navigation.py` (source- and
AST-level, so Home's boundary is checked without importing torch or
Streamlit) and the handover class in `tests/test_agent_web_ui.py`. Nothing
here may be marked `VERIFIED` from this environment: no Streamlit page has
been rendered in it.

---

## Amendments

Record here any requirement found to be wrong or impossible, with the proposed
change, rather than implementing something different.

*(none yet)*
