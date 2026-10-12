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
| `VIZ-06` | A Streamlit page displays the generated figures, and offers an interactive layer for exploring individual cells. The page degrades to the static figures when the interactive library is absent, so Route B never depends on it. | `PARTIAL` | Review Results / Predictions / Agent Workflow draw the `VIZ-09` panels with a cell selector and a zoom toggle (`batteryml_web/ui/run_history.py`). No hover layer exists; the static figures are the only layer. |
| `VIZ-07` | `report.html` assembles the run's figures, metrics and provenance into the single reviewer-facing artifact `CLAUDE.md` §8 names. Nothing writes it today. | `NOT-STARTED` | — |
| `VIZ-08` | Cross-model comparison over several runs of one dataset, read from their manifests. Introduces the first categorical palette, which must be validated before it ships. | `NOT-STARTED` | — |
| `VIZ-09` | The website result figures and their export come from one function, `build_panels`, which returns each figure with the exact arrays given to its builder; the export's CSV is those arrays. The selected run's own artifact is the only input; a run with several artifacts asks rather than taking the first. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/result_figures.py`, `batteryml_web/ui/run_history.py`; tests `tests/test_result_figures.py` |
| `VIZ-10` | SOH across observed cycles shows **one** cell: recorded cycle on x, observed (solid) and estimated (dashed) SOH on y, in recorded order, no smoothing/interpolation/forced range. Default cell = first ID in ascending order (outcome-blind); the selector lists only real cells and is keyed to the artifact path and revision, so a cell never survives a run change. Repeated (cell, cycle) records are flagged, not averaged. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/result_figures.py` (`soh_cell_series`, `build_panels`), `batteryml_web/viz/figures.py` (`soh_cell_figure`) |
| `VIZ-11` | Label space: the recorded inverse transformation is applied exactly once, with a status; a missing record or a failing inverse refuses the physical-unit view instead of plotting transformed values. Non-finite rows are omitted through one shared mask, counted, and never used to recompute saved metrics. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`_invert`, `_paired_series`) |
| `VIZ-12` | Each view states what a value is, from the recorded label annotator: "Battery lifetime (cycles)" for whole-cell `RULLabelAnnotator`, "Remaining useful life (cycles)" only for genuine cycle-wise RUL, "SOH (ratio)" with its reference-capacity mode. SOH residual ×100 is labelled percentage points. Values above 1.0 SOH are not called invalid. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`target_spec`, `RANGE_NOTE`) |
| `VIZ-13` | Full-range parity/residual/cycle views are the default and keep every finite value; a labelled zoomed view changes only the viewport (1st–99th percentile, padded) and states how many points fall outside. A per-cell MAE/RMS-residual table is a display diagnostic and never replaces the saved scores. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/result_figures.py` (`zoom_window`, `residual_zoom`, `per_cell_errors`) |
| `VIZ-14` | `python -m scripts.export_result_figures --artifact <exact.pkl> --output <new dir> [--cell-id]` writes separate SVG+PNG panels, `plotted_samples.csv`, per-cell CSVs, `figure_sources.json` (artifact SHA-256, run.json/config-copy provenance, smoke evidence, transformation and parameters, units, counts, cell rule, limits, saved metrics verbatim, command, git commit, versions) and `figure_notes.md`. Read-only `--inspect` / `--list-cells`. Never trains, evaluates, downloads or searches; refuses a non-empty output directory. | `IMPLEMENTED-STATIC` | `scripts/export_result_figures.py`, `batteryml_web/viz/result_export.py`, `batteryml_web/agent/artifact_provenance.py`. Run on synthetic fixtures only; no real artifact was available. |

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

## SOH — state-of-health estimation

Rationale for the group: SOH is the second scientific task the platform
supports, and it is the one most easily confused with the first. "State of
health" appears inside RUL's own vocabulary — `eol_soh` is the threshold an RUL
label uses — and before this group the platform advertised 42 lifetime configs
as SOH estimation on the strength of that substring. The requirements below
exist to keep the two tasks apart at every layer: the label, the sample, the
split, the config, the artifact, the figure and the agent.

SOH is also the platform's first **cycle-level** task. One cell yields one
sample per observed cycle, which breaks the one-row-per-cell assumption the
pipeline was built on, and creates a leakage route (splitting expanded rows
instead of cells) that produces a good-looking score and no visible symptom.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `SOH-01` | SOH estimation is a distinct task: the health of a cell **at an observed cycle**, from information available at or before it. It is not next-cycle capacity forecasting, not RUL, not lifetime, and not degradation-trajectory forecasting. A config's task is read from its label annotator, never from a substring — in particular `eol_soh`, an RUL end-of-life threshold, never implies SOH. | `IMPLEMENTED-STATIC` | `batteryml_web/core/config_registry.py` (`_infer_task`), `docs/SOH_PROTOCOL.md` |
| `SOH-02` | The canonical label is a ratio, `SOH(cell, t) = Q_discharge(t) / Q_ref(cell)`, with `Q_ref` fixed for the whole run by `reference_capacity_mode` and applied to every cell. No per-cell fallback: a cell that cannot satisfy the configured mode is excluded and counted, never relabelled under another. Invalid cycles are dropped, never interpolated; ratios above 1.0 are kept, never clipped. A UI may display a percentage; the stored value stays a ratio. | `IMPLEMENTED-STATIC` | `batteryml/label/soh.py` (`CycleSOHLabelAnnotator`, `reference_capacity`) |
| `SOH-03` | One cell yields one sample per observed cycle the sample policy accepts, and every row carries its `(cell_id, cycle_index)`. The label annotator owns the cycle list and the feature extractor is asked for exactly those cycles, so feature rows and label rows align by construction. Existing one-sample-per-cell RUL workflows are unchanged. | `IMPLEMENTED-STATIC` | `batteryml/data/sample_index.py`, `batteryml/task.py` (`_build_cycle_level`), `batteryml/{feature,label}/base.py` |
| `SOH-04` | Cells are split before they are expanded into cycles, checked rather than assumed: `assert_disjoint_cells` runs on the loaded cells before any expansion and names every cell found in more than one split. A test constructs the leakage-prone alternative — expand all cells, then split rows — and proves it is rejected. | `IMPLEMENTED-STATIC` | `batteryml/data/sample_index.py` (`assert_disjoint_cells`), `tests/test_soh_core.py` |
| `SOH-05` | The SOH feature is the voltage curve observed at cycle `t`: a fixed-width resampling plus five summary scalars, reading cycles `<= t` only. Discharge capacity is deliberately excluded — it is the label's numerator, and a full discharge measurement already answers the question. Causality is proved by mutating every later cycle and requiring the row to be unchanged, with a companion test that mutates cycle `t` and requires it to change. | `IMPLEMENTED-STATIC` | `batteryml/feature/soh_cycle.py`, `tests/test_soh_pipeline.py` |
| `SOH-06` | SOH eligibility is decided by opening the processed cells and counting what a run would get, through the same annotator and extractor. Verdicts are SOH-specific, so a dataset can be ready for RUL and not for SOH. A dataset with nothing on disk reports zero and says so; no count is ever fabricated. | `IMPLEMENTED-STATIC` | `batteryml_web/core/soh_readiness.py`, `scripts/audit_soh_readiness.py` |
| `SOH-07` | The prediction artifact carries `sample_index` (one `{cell_id, cycle_index}` per prediction) and `task_metadata` (label annotator, reference-capacity mode, sample policy, feature extractor). Canonical metrics are MAE and RMSE on the ratio scale; MAPE is not inherited from the RUL set. The loader validates the recorded identity before anything plots it. | `IMPLEMENTED-STATIC` | `batteryml/pipeline.py` (`_task_metadata`), `batteryml_web/viz/loaders.py`, `batteryml_web/agent/execution.py` (`metrics_for_task`) |
| `SOH-08` | The SOH result views are the existing parity and residual figures plus `soh_vs_cycle`, which groups measured and estimated SOH by real `cell_id` from the saved artifact and states how many of how many cells it shows. Every point is an at-cycle estimate; it is never presented as a trajectory forecast. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/figures.py` (`soh_vs_cycle_figure`), `batteryml_web/viz/contract.py`, `batteryml_web/viz/cli.py` |
| `SOH-09` | No dataset is claimed SOH-ready and no dataset SOH config exists until `scripts/audit_soh_readiness.py` is run on real processed data. The legacy `configs/soh/dummy/*` examples are marked `legacy_example: true` and withheld from the runnable catalog rather than advertised as SOH support. | `IMPLEMENTED-STATIC` | `configs/soh/dummy/*.yaml`, `batteryml_web/core/config_registry.py` (`_parse_entry`) |
| `SOH-10` | Agent intent maps SOH language (English and Chinese) to SOH estimation and lifetime, RUL and trajectory language away from it. "End of life at 80% SOH" is the `eol_soh` decoy in natural language and resolves to RUL. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/intent.py`, `tests/test_soh_agent_results.py` |
| `SOH-11` | The first real SOH route is MATR + Ridge Regression at `configs/soh/ridge/matr.yaml`, with `min_cycle: 10`, `cycle_stride: 5` and `reference_capacity_mode: first_valid_discharge_capacity`, on the canonical cell-level `MATRPrimaryTestTrainTestSplitter` (41 train / 42 test cells, 83 of MATR's 180). The MATR RUL menu is unchanged. (Amended by `SOH-13`: Ridge is no longer the only entry the SOH catalog offers.) Labels are audited before training and the artifact after it, by `scripts/audit_soh_labels.py` and `scripts/inspect_soh_artifact.py`. | `IMPLEMENTED-STATIC` | `configs/soh/ridge/matr.yaml`, `scripts/audit_soh_labels.py`, `scripts/inspect_soh_artifact.py`, `tests/test_soh_matr_config.py` |
| `SOH-12` | A cycle-level SOH result is presented as one, in every place a result is drawn. The schema is decided from what the artifact records about itself, never from the numbers — an SOH scalar per cycle is structurally identical to an RUL scalar per cell — and from `RES-01` onward that means the recorded task *and* the recorded per-row identity, not the identity alone — so each row names its cell and its cycle, the axes say SOH (ratio), and per-cell curves are drawn beside the parity scatter. Beside them is a display-only range check: how many estimates fall below 0, above 1.0 and above 1.1, the extremes, and the ten largest absolute residuals. It counts and reports; it never clips, rescales, drops or reorders a prediction, and the values plotted are the values the canonical metrics were computed from. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`SOH_SCHEMA`, `build_soh_view`, `soh_range_diagnostics`, `largest_residuals`), `batteryml_web/ui/run_history.py` (`render_soh_figures`), `tests/test_soh_result_views.py` |
| `SOH-13` | Beside Ridge, the SOH catalog offers every estimator that passed a compatibility audit against the real `SklearnModel.fit`/`predict`/`evaluate` path on a cycle-level bundle of the route's own 37-wide shape: Lasso, Elastic Net, Random Forest and XGBoost. Each has a config under `configs/soh/` that is byte-identical to the Ridge config outside its `model:` block — same split, reference convention, sample policy, feature extractor, transformations and metrics — so two SOH numbers differ only by the estimator. No config declares a hyperparameter and none pins `random_state`, so `--seed` keeps its meaning. MLP was audited and rejected: it is an `NNModel` requiring a two-dimensional per-sample feature geometry a flat SOH row does not have, and the feature is not reshaped to suit it. These are compatible baselines, not a ranking. | `IMPLEMENTED-STATIC` | `configs/soh/{lasso,elastic_net,rf,xgb}/matr.yaml`, `tests/test_soh_baseline_pack.py`, `docs/SOH_PROTOCOL.md` |

### Notes on the SOH group

* `SOH-01` and `SOH-04` are the two that would fail silently. A mislabelled
  task runs the wrong science and reports it under the right name; a leaked
  cell inflates the score and looks entirely normal. Both are enforced by
  tests that construct the failure deliberately rather than by inspection.
* No SOH requirement may be marked `VERIFIED` from this environment. It has no
  `data/`, so no dataset has been shown SOH-eligible and no SOH run has
  executed on real cells. The command block is in `docs/SOH_PROTOCOL.md`.
* This group implements an SOH **route**, not an SOH benchmark: one feature
  path, one label convention, and models chosen for being task-agnostic rather
  than for SOH performance. `docs/SOH_PROTOCOL.md` states the limitations.
* The upstream core changes it required are `UP-04`, `UP-05` and `UP-06` in
  `docs/UPSTREAM_PATCHES.md`.

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
| `EXEC-09` | Run Model runs on the device the user selected, and only there. A device this machine cannot provide (e.g. `cuda` without CUDA) is explained beside the selector and on the Command & Run tab, keeps the command showing the requested device, and disables Run BatteryML until the user explicitly chooses another; it is never replaced by `cpu`. Same contract as the Quick experience (`QX-03`): `quick_experience.cuda_available` / `device_problem` over `device_inventory.validate_device`. No widget key is written after its widget is drawn. Rationale: the earlier page showed "Falling back to CPU" and executed `--device cpu` for an explicit `cuda` choice. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`render_run_model_page`); tests `tests/test_quick_experience_fixes.py` (`TestRunModelDevicePage`) |
| `EXEC-10` | Run Model's Task holds canonical labels only (`config_task`: "RUL prediction" / "SOH estimation"); the session default is RUL. The radio opens on the task the user last chose **on Run Model** (kept under its own key, which outlives the radio's widget state and Home's rewrite of `selected_task`); until there is one, on the session's task — the one Home's Quick panel shows — and on RUL when neither names a task. A model with no config for the chosen task is cleared before the model selector is drawn, and the config is looked up from (dataset, task, model), so the task on screen, the config shown on both tabs and the config in the command always agree. No widget key is written after its widget is drawn. Rationale: the radio compared the stored task with the short label `"RUL"`, so any full label — written by Home and by Run Model itself — opened it on SOH. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`_run_model_initial_task`, `_remember_run_model_task`, `render_run_model_page`); tests `tests/test_quick_experience_fixes.py` (`TestRunModelTaskPage`) |
| `EXEC-11` | A GPU is offered as usable only if PyTorch sees it **and** the installed PyTorch build can run kernels on it. The build's own architecture list decides (PyTorch's rule: a compiled `sm_XY` of the GPU's major version); a PyTorch too old to list it (< 1.7) is judged by the first CUDA toolkit each GPU generation needs (Ampere 8.x → 11.0). The GPU's compute capability is read from `nvidia-smi` (argv, no CUDA context); when it cannot be read, nothing is invented. An unusable GPU is reported with the reason and blocks Run on Quick experience, Run Model and Agent Workflow; the CPU is never substituted (Agent's former "Runs will use the CPU" fallback is removed). Rationale: server acceptance — PyTorch 1.6 / CUDA 10.2 on RTX 3090 (sm_86) was offered as cuda. | `IMPLEMENTED-STATIC` | `batteryml_web/core/device_inventory.py` (`query_compute_capabilities`, `gpu_build_incompatibilities`), `batteryml_web/core/quick_experience.py` (`cuda_available`), `batteryml_web/ui/agent_workflow.py` (`gpu_mode_block`, `block_plan`); tests `tests/test_server_acceptance_fixes.py` |

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
| `UI-10` | One visual language on every page, the mentor's (`codex/ui-optimization` Home): the canvas, the 1480px content width, ink headings and muted body copy, the cobalt primary action, rounded selection controls, calm disclosures and tables. It is one site-wide stylesheet injected after the surface roles on every run; Home adds only its own components (hero, start paths, step overview). Streamlit's own widget accent (tabs, radios, toggles, focus) is the same cobalt, set as the theme's `primaryColor`. Status messages and metric values keep their own colour. Selectors are stable `data-testid`/`data-baseweb`/role attributes and the app's own names, never generated classes; the sidebar keeps its own sheet. Rationale: the mentor's sheet shipped only with Home, so every other page rendered Streamlit's defaults (measured in a browser). | `IMPLEMENTED-STATIC` | `batteryml_web/ui/home_style.py` (`SITE_STYLE_CSS`, `HOME_STYLE_CSS`), `batteryml_web/app.py`, `.streamlit/config.toml`; tests `tests/test_site_style.py` |
| `UI-11` | The interaction hierarchy is visible at a glance: primary is the one filled cobalt control; secondary (Download, Refresh, Explain) an outlined white control; navigation a cobalt wash with no outline; a disabled control of any role is neutral, dashed, unlifted and `not-allowed`, with its reason beneath it; only an enabled control reacts to the pointer. Information — cards, read-out tiles, disclosures — keeps a flatter, passive shape (no hover, no shadow) in the same palette, so it never resembles a control. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/home_style.py`; tests `tests/test_site_style.py`, `tests/test_interaction_hierarchy.py` |
| `UI-12` | One design system, named: colour roles (primary, primary-hover, primary-muted, background, surface, surface-secondary, surface-selected, border, border-subtle, text-primary/secondary/muted, focus-ring, success, warning, error), type (page/section/subsection title, body, label, caption, metric, monospace), geometry (page width, card/button/input radius, control height, section/component gap, card padding) and interaction states (hover, selected, disabled, loading, completed, failed) are `--bml-*` tokens derived from the mentor's seven palette values, and the shared rules use the tokens. One heading scale on every page — page title 2rem/700 (`st.header`), section 1.5rem/650 (`st.subheader`), subsection 1.25rem/600 (`####`, the mentor's component-heading size) — with Home's hero, start paths and step overview keeping the mentor's own sizes. One type family, the theme's, never overridden; code and run ids in the theme's monospace, wrapped. The LLM settings form takes the same input and button styling in the sidebar Settings block and in a dialog (colour, border and radius only; the sidebar's spacing is untouched). Stop run is an interrupting control (red hairline on white), a run status a passive pill. Rationale: page titles measured 36px/600 beside 28px sections and 24px subsections, so Review Results stacked three oversized headings; the sidebar Settings inputs measured 0px radius with Streamlit's grey border. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/home_style.py` (`SITE_STYLE_CSS`); tests `tests/test_design_system.py`, `tests/test_site_style.py`; browser check `scripts/ui_audit/viewport_audit.py` |
| `UI-13` | Review Results reads as one hierarchy — page title, then the sections Run history, Selected result (tabs Summary / Metrics / Predictions / Logs / Files, then the Benchmark reference subsection) and Optional LLM interpretation, in that order — and the Run History table is compact: a short run id (the minted id's unique suffix), the start time with its recorded time zone written out (`2026-10-12 17:29 UTC`; no conversion to a local zone), one `model · dataset · task` cell, seed, device, status, duration and metrics. The shortening is display-only: the "Open a run" selector's value, the session state and every section below use the canonical run id; the run's details show the full id in a copyable code block and the start time exactly as recorded; nothing on disk is rewritten. A run's status is a passive pill (`.bml-status`), never a control. Rationale: the table printed `run_20261012_172911_elastic_net_matr_seed0_f9b0a1` and `2026-10-12T17:29:11+00:00` across eleven columns, and the LLM heading sat above the result tabs it did not describe. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/run_history.py` (`short_run_id`, `format_timestamp`, `experiment_summary`, `status_badge`, `history_row`, `run_option_label`), `batteryml_web/app.py` (`render_review_results_page`, `render_benchmark_panel`); tests `tests/test_review_results_presentation.py` |
| `UI-14` | LLM settings are one session-level store (`llm_state.py`) reached from every page, and Results never blocks on them. Review Results shows the live status (`LLM assistant status: …`) apart from any stored output. **Configure LLM** opens the shared settings body in a small dismissible dialog (`settings_panel.render_requested_llm_settings`; the same `render_llm_settings_form` call site the sidebar uses, its own widget namespace, the one store). **Summarize results / Explain metrics** call the provider only with a verified connection (`is_llm_connected`); otherwise they open the same dialog and send nothing. Arriving at Review Results through an intentional "view results" of a run that just completed (Quick "Open Review Results" after a successful run, Run Model "Review Results →") opens Settings once per session, and only when nothing is configured; a saved-but-untested (Case C) or failed (Case D) configuration shows its saved fields with the key masked and a short redacted reason, and is never retried automatically. A dialog request is consumed when it opens, so no rerun reopens it; a successful Connect closes it, keeps the selected result and sends no request. One click sends at most one request. A failed request is shown as the outcome of that request — never as the live status — and is dropped once a later connection succeeds (connection serial); an explanation stays bound to its run and the configuration that produced it. Changing provider, URL, model or key invalidates verification; an unchanged Save keeps it. Connect is the form's primary action, Save secondary. Run Model's "Review Results →" is drawn outside the Run branch: inside it, the button vanished on the rerun its own click caused. Rationale: "Configure LLM" showed a sentence pointing at the sidebar, which a collapsed sidebar (narrow windows) hides; Summarize/Explain called the provider with untested settings; an old 401 stayed on screen after a successful Connect. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`request_llm_settings`, `render_requested_llm_settings`, `consume_results_intent`, `arrival_reveal_due`, `llm_action_allowed`, `_render_settings_body`), `batteryml_web/ui/llm_state.py` (`is_llm_connected`, `connection_serial`), `batteryml_web/app.py` (`render_review_results_page`, `render_llm_output_panel`, `_record_review_output`); tests `tests/test_llm_results_flow.py` |
| `UI-15` | Agent Workflow tells data readiness apart from combination availability. When the analysed dataset + task (+ named model) cannot run, "Backend recommended next steps" opens on a deterministic diagnosis — e.g. "HUST data is ready, but no registered SOH estimation configuration is currently available for HUST" — and three separate kinds of alternative: other runnable models for the same dataset and task, the same task on other datasets, other tasks on the same dataset. All of it comes from the configuration registry (which (dataset, task, model) triples have a run config) and the readiness/validation layer (which of those are READY with a validated config); a dataset that is RAW_ONLY or MISSING contributes nothing runnable, and no configuration is ever invented. An action ("Use Elastic Net Regression on MATR for SOH estimation", "Change task to RUL prediction") is offered only for a combination with a runnable model; it replaces the request (visibly), clears stale model and configuration choices and re-analyses deterministically — it never runs anything. Works with no LLM. A model written without separators ("ElasticNet") is recognised. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/alternatives.py`, `batteryml_web/ui/agent_workflow.py` (`_render_next_steps`, `_apply_alternative`, `_analyze_request`), `batteryml_web/agent/catalog.py`, `batteryml_web/agent/llm_intent.py`; tests `tests/test_agent_alternatives.py`; verify `INTENT-01` |
| `UI-16` | "Explain this workflow" responds where it is pressed. No call on render. With a connected LLM: a spinner, exactly one request through the shared client, and the explanation directly beneath the button (not in a closed panel at the foot of the page), kept across reruns and dropped when the analysed request changes. Without a connected LLM: a plain statement beneath the button, an "Open LLM settings" entry and the shared settings dialog — nothing sent. On failure: the classified, redacted error beneath the button; nothing retried; the deterministic analysis intact. A workflow that cannot run is explained too: the prompt carries the page's own context, the requested model's status and the backend alternatives verbatim, and the system prompt forbids presenting it as runnable, inventing configs, quoting benchmarks or offering to start training. Rationale: measured, the click stored its text 34 elements below the button inside a collapsed expander with no progress shown, and with no LLM only a grey caption appeared. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py` (`_render_workflow_explanation`, `_explain_workflow`, `workflow_identity`), `batteryml_web/agent/workflow.py` (`explain_workflow_state`); tests `tests/test_agent_alternatives.py` |

The tests for this group are `tests/test_web_navigation.py` (source- and
AST-level, so Home's boundary is checked without importing torch or
Streamlit), `tests/test_ui_feedback.py`, and the handover, request-tab and
feedback classes in `tests/test_agent_web_ui.py`. Nothing here may be marked
`VERIFIED` from this environment: no Streamlit page has been rendered in it,
and `UI-08` in particular can only be confirmed by looking at the running
app's toolbar.

---

## QX — Home Quick experience (the mentor's UI on the current backend)

Rationale for the group: the Quick experience panel (merged from
`codex/ui-optimization`, PR #13) is the reviewer's first route. Review found
that its state logic could run a different protocol than the one on screen
(an RUL config under "SOH estimation"), that an uploaded YAML was written over
a same-named file and could be run unchecked, and that a CPU-only machine
crashed the page. These requirements fix behaviour only; the panel's layout,
labels and flow are the mentor's and are kept.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `QX-01` | The task decides the models offered, the model decides the configs offered; a config that trains another task or belongs to another model is never kept selected — it is cleared and the page says so. What may run is decided by one gate, evaluated when the panel is drawn **and again immediately before launch**: the config's task, established by its registered label annotator, must equal the task on screen; every class must be registered (`config_validator`); the dataset must be one the panel offers; data must be present; the device must exist. An invalid plan disables Run and lists its reasons, and the launcher refuses an invalid plan. | `IMPLEMENTED-STATIC` | `batteryml_web/core/quick_experience.py` (`config_choices`, `reconcile_config`, `build_quick_plan`, `plan_command`), `batteryml_web/app.py` (`_render_home_quickstart_panel`, `_run_home_experiment`); tests `tests/test_quick_experience_fixes.py` (`TestPlanGate`, `TestQuickExperiencePage`) |
| `QX-02` | An uploaded config is accepted only if: its name is a plain `.yaml`/`.yml` file name; it is at most 256 KiB; it parses with `yaml.safe_load` into a mapping (language tags refused); it passes `config_validator`; its task is established by its label annotator and offered by the page; its dataset is offered; it sets no file/directory option other than `train_test_split.cell_data_path`, which must resolve inside `data/processed`; and it sets no device option. It is then stored once, normalised, as `<project>/uploads/configs/<random id>.yaml` with exclusive create — the original name never chooses a path and nothing is overwritten; a symlinked upload directory is refused. **Apply selects; it never runs.** A refused upload leaves the previous selection unchanged and lists the reasons. | `IMPLEMENTED-STATIC` | `batteryml_web/core/config_upload.py`, `batteryml_web/app.py`; tests `TestUpload` |
| `QX-03` | A machine without CUDA renders the panel normally (no `nvidia-smi`, no GPU, and a failing availability probe all mean "not available"). Choosing `cuda` there is explained and blocks Run until the user chooses `cpu`; the requested device is never silently replaced. | `IMPLEMENTED-STATIC` | `batteryml_web/core/quick_experience.py` (`cuda_available`, `device_problem`); tests `TestDevice`, `TestQuickExperiencePage.test_no_cuda_renders_and_explains` |
| `QX-04` | A Quick experience run is recorded through the shared page-run lifecycle (`LIFE-11`): its own `<project>/quickstart/<run_id>/` folder, RUNNING before start, COMPLETED/FAILED with return code, artifacts and only BatteryML's saved metrics; a failure keeps its reason. The metrics Home shows are read from that run's own `run.json` (written through the CPU-safe artifact loader), so Home, Run History and Review Results agree; the page never unpickles an artifact itself. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`_run_home_experiment`), `batteryml_web/core/quick_experience.py` (`run_metrics`); tests `test_completed_and_failed_runs_are_recorded_and_selectable` |
| `QX-05` | "← Choose another start path" returns to the start-path choice and stays there; the Quick experience default applies once per session, not on every rerun. No run, project or LLM setting is touched. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py`; test `test_choose_another_start_path` |
| `QX-06` | A failed dataset-manifest refresh after a download or preprocessing is shown on the page with the command and its error; it is never reported as success. | `IMPLEMENTED-STATIC` | `batteryml_web/core/quick_experience.py` (`sync_manifests`), `batteryml_web/app.py` (`_refresh_data_manifests`); tests `test_manifest_sync_failure_is_reported`, `test_manifest_sync_error_is_visible` |
| `QX-07` | There is one rule for which task a config trains: the exact registered label annotator class (`RULLabelAnnotator` → RUL; `SOHLabelAnnotator`, `CycleSOHLabelAnnotator` → SOH), cross-checked against an optional explicit `task:` of an exact known spelling. A disagreement or an unknown value is an error, never a guess; `eol_soh` and file names are never signals. The validator, the registry, the config semantics and the Quick experience gate all use it. The result views keep their own artifact-side table (`result_views._ANNOTATOR_FAMILIES`) and are checked to agree with it for every known and unknown annotator and every repository config; they differ only for a declared `task:` that this rule refuses, which no platform run route can execute. | `IMPLEMENTED-STATIC` | `batteryml_web/core/config_task.py`; `config_validator.py`, `config_registry.py`, `config_semantics.py`; tests `TestTaskResolution`, `TestResultViewsTaskAgreement`, `tests/test_soh_task_inference.py` |
| `QX-08` | "Raw dataset" offers every dataset the readiness layer knows (`list_dataset_readiness`: platform datasets plus anything on disk), with no list of names in the page. The default protocol's dataset (MATR) is first and the default; datasets the config registry has a protocol config for follow; the rest stay listed, labelled "no run configuration", and when selected say why Quick experience cannot run them and offer Prepare Data — no data action or run is offered for them. Selecting a dataset never downloads, preprocesses or runs anything. The dataset drives the task options (those it has configs for), then model, then config; a task, model or config the new dataset does not offer is cleared before its widget is drawn and the page says so. Uploads are accepted and offered only for the dataset on screen, and the execution gate allows only that dataset, so a config of the previous dataset can never run. Rationale: the selector was the hardcoded list `["MATR"]`, an enabled control with one option. | `IMPLEMENTED-STATIC` | `batteryml_web/core/quick_experience.py` (`dataset_choices`, `task_options`, `quick_limitation`, `reconcile_selection`), `batteryml_web/app.py` (`_quick_dataset_choices`, `_render_home_quickstart_panel`); tests `tests/test_quick_dataset_selection.py` |
| `QX-09` | Step 1 reports the selected dataset's readiness from the readiness layer: raw files, processed files, their paths, and the verdict (`READY`/`RAW_ONLY`/`INCOMPLETE_PROCESSED`/`MISSING`) with its reason verbatim. Raw files alone never read as ready. Each data action is in one of three states: available (a button — Download is secondary, Process primary), unavailable (a disabled button with its reason underneath), or already done (a one-line ✓ note and no button), so a finished step never looks like a broken control. Running and failure use the existing progress area and error message. | `IMPLEMENTED-STATIC` | `batteryml_web/core/quick_experience.py` (`download_action`, `process_action`), `batteryml_web/app.py`; tests `TestActionStates`, `TestQuickDatasetPage` |
| `QX-10` | With no project selected, a Quick experience run and its data-action logs go to the same default project Run Model and Review Results use (`resolve_project` → `runs/web_projects/default`), so a fresh session's Review Results lists them. When the current project has no runs, Run History names the other projects that do (with counts) and how to open them; it never shows another project's runs as this one's. Rationale: server acceptance — Quick runs were written to `runs/web_projects/quick_experience`, which no other page read, so Review Results opened on "No run is selected". | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`_data_action_root`), `batteryml_web/ui/run_history.py` (`projects_with_runs`); tests `tests/test_server_acceptance_fixes.py` |

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
| `RUN-11` | The LLM settings form offers **Save** and **Connect**, in that order, wherever settings are edited — the sidebar and the prompt opened from "Explain computed result" alike (since `UI-14`, also the Settings dialog opened from Review Results; Connect is the primary-styled action). There is no primary Cancel / Save settings pair. Save stores the entered values through the existing credential store and makes no network call. Connect stores the same values through the same function and then runs the connection test; success sets status Connected and reports transiently, failure leaves an actionable persistent error. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`render_llm_settings_form`, `SAVE_LABEL`, `CONNECT_LABEL`) |
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
| `LIFE-09` | Running the same plan twice is ordinary. A terminal run never blocks the next one: the page releases it, a fresh run id and run directory are minted, the earlier run stays in history untouched, and the control reads **Run again** rather than looking like a first run. Terminal detection reads the recorded *status*, never the progress-event stream, so a model that reports no epochs — an sklearn fit such as Ridge — still ends, is still reaped and still frees the button. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py` (`_render_active_run`, `RUN_AGAIN_BUTTON_LABEL`), `tests/test_repeated_runs.py` |
| `LIFE-10` | A finished run is released before anything fallible reads its results, and the reaping is keyed on terminal status rather than on worker-thread liveness. Reading the artifacts first, and clearing the active run last, meant any failure in between reached the page error boundary with the run still marked active — and since the Run button is drawn only when no run is active, the page could never execute again. A failure to read results is reported as a failure to read results, never as a failed run. Likewise the "plan changed, drop stale results" guard compares the *experiment* (dataset, task, model, config, epochs, device, seed), not the argv, whose `--workspace` carries a run id re-minted on every rerun. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py` (`_finish_active_run`, `executed_plan_signature`), `batteryml_web/core/run_manager.py` (`forget`) |
| `LIFE-11` | A run started from **Run Model → Run BatteryML** or **Home → Run MATR Demo** is recorded exactly like an Agent Workflow run and is listed in Review Results. It gets its own folder, `runs/web_projects/<project>/run_model/<run_id>/` or `…/quickstart/<run_id>/`, which is also BatteryML's `--workspace`, so runs never overwrite each other (the old shared `<project>/workspace/` did). Its `run.json` uses the same schema (`build_manifest`), is written PLANNED → RUNNING before BatteryML starts — so a crash or a closed tab is reported INTERRUPTED, not lost — and COMPLETED or FAILED afterwards with return code, duration, `output.log`, the real artifact list (`discover_artifacts`) and only the metrics BatteryML saved (`build_result_context`). A failed run records no metrics; a metrics read failure is recorded as `metrics_error`, not hidden. The project comes from `resolve_project`, the same resolution Review Results uses. The Run Model page reserves the id before the click because it shows the exact command; the reservation follows the selection and is released once the run is recorded. | `IMPLEMENTED-STATIC` | `batteryml_web/core/direct_runs.py`, `batteryml_web/app.py` (`_begin_page_run`, `_finish_page_run`, `_run_model_workspace`); tests `tests/test_direct_runs.py`. Not run with real data in the authoring environment (no processed dataset), so the listing was not observed in a browser. |
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
| `UX-01` | The request area offers one text area and one **Use template** helper. There is no example gallery, no Examples tab and no keyboard hook. The placeholder shows the generic shape `Use [method] on [dataset] to [task].`; the helper fills it with values drawn from the live catalog and readiness, keeping a `[slot]` where the catalog cannot supply one rather than naming something this installation does not have. Home and the Agent Workflow page share the same helper and the same canonical request key. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/catalog.py` (`build_request_template`), `batteryml_web/ui/agent_workflow.py`, `batteryml_web/app.py` (`render_home_agent_entry`) |
| `UX-02` | LLM settings have exactly one store. Every surface renders the same form, and each form's widgets are re-seeded from that store whenever it changes elsewhere — per-surface widget keys are required (Streamlit rejects duplicates in one run) but must not become per-surface copies, which is what made the app ask for settings twice. Status has one vocabulary, and "Configured (not tested)" is never reported as "Connected". | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`_sync_widgets_from_store`), `batteryml_web/ui/llm_state.py` (`LLM_STATUS_LABELS`) |
| `UX-03` | Provider presets fill the Base URL for OpenAI, Gemini, Claude, Kimi, GLM and a custom option; the field stays editable and no model names are baked in. The help text states precisely what is supported — the OpenAI-compatible chat-completions protocol — and says provider-specific features beyond it are not implemented. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/settings_panel.py` (`PROVIDER_PRESETS`, `SUPPORTED_API_NOTE`) |
| `UX-04` | A failed Connect is classified rather than shown raw: network/unreachable, authentication, timeout, invalid response and unknown are distinguished, and an unfinished form is not reported as a connection failure at all because nothing was sent. `Network is unreachable` — the observed symptom — classifies as connectivity and names a locked-down server as the likely cause. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/failure_taxonomy.py` (`_TIMEOUT_MARKERS`), `batteryml_web/ui/settings_panel.py` |
| `UX-05` | A BatteryML prediction `.pkl` is described rather than refused: output kind, prediction count, whether ground truth is stored, the seed, and whether saved metrics are present — then the observed-vs-predicted, residual and table views. Reading goes through the one CPU-safe unpickler; no metric is computed or restated here. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/artifact_preview.py` |
| `UX-06` | An unrecognised `.pkl` is neither parsed nor called "not implemented": it reports that its structure is not a supported prediction-view format and shows only what it can read — top-level type, keys, types and shapes. A dict that merely has a `prediction` key but no usable shape counts as unrecognised. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/artifact_preview.py` (`describe_structure`, `_is_prediction_payload`) |
| `UX-07` | The Files view groups discovered artifacts by category with size and modified time, and states Project / Workspace / Run directory explicitly rather than leaving them to be inferred from a path. No group is created for files that do not exist, and every path shown comes from discovery inside that run's own directory. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/run_history.py` (`group_artifacts`, `render_run_location`, `render_file_browser`) |
| `UX-08` | Opening a page from the sidebar shows it from the top. The previous page's scroll offset is never carried across a page switch, on any entry (Home, Agent Workflow, Prepare Data, Run Model, Review Results) and by any route into it (a sidebar button, Prev/Next, the bottom navigation, or a direct `current_step` assignment). A rerun *within* a page does not scroll, so a button press does not move the reader. The reset is injected before the page body, so a page that raises part-way through still opens at the top. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/scroll.py` (`page_changed`, `render_scroll_to_top`), `batteryml_web/app.py` |
| `UX-09` | Prepare Data is one linear flow, not a Preprocessing / Feature Inspection split: dataset selector → readiness, raw and processed paths → preprocess action → latest preprocessing log → feature/artifact inspection. One dataset selector drives all of it, so the page cannot act on one dataset while reporting another. Nothing the two tabs offered was dropped, and an empty dataset list no longer abandons the rest of the page. | `IMPLEMENTED-STATIC` | `batteryml_web/app.py` (`render_prepare_data_page`, `_selected_dataset_index`) |
| `UX-10` | The sidebar's vertical spacing is tightened with a layout-only stylesheet: padding, margin and gap, scoped to the sidebar by stable `data-testid` selectors, with nothing hidden and no generated class names relied on. One block gap sets the sidebar's whole vertical rhythm — measured in a browser, every top-level block is separated by it and nothing else — and it is held at or above `0.25rem`, below which the controls stop reading as separate rows. Streamlit's sidebar heading overflows its own container by ~16px, so the heading padding may not be zeroed and the brand rule's negative margin must cancel exactly one block gap: that pairing, not either number, is what keeps the title off its own caption, and it holds at any gap. Every value was established by rendering the page, not by eye. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/styles.py` (`SIDEBAR_COMPACT_CSS`, `render_layout_css`), `batteryml_web/ui/workflow_sidebar.py` |
| `UX-11` | The sidebar draws **exactly one** horizontal rule, immediately above the **Protocol Workflow** heading — the one place a boundary exists, between the two ways in (Home, Agent Workflow) and the numbered route. The rules that used to sit above Prev/Next, above the settings block and above the status line separated a list from its own controls, and a single-line control from a single-line readout; vertical rhythm does that work now. A navigation entry carries a caption only where the caption says something its label cannot: **Home** carries none, the Protocol Workflow heading carries none, and **Agent Workflow** keeps the one that states the execution contract (nothing runs until Run is pressed). The grey secondary lines are set at `0.82rem`, scoped to the sidebar so `st.caption` elsewhere in the application is untouched, with type size and line-height only — no margin, height or overflow, which is what previously clipped a wrapped caption behind the button beneath it. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/workflow_sidebar.py` (`render_sidebar_workflow`), `batteryml_web/ui/styles.py` (`SIDEBAR_COMPACT_CSS`) |
| `UX-12` | A running run shows **one structured state**, not a log. Stage names are the tqdm descriptions BatteryML actually prints (`Reading train data`, `Reading test data`, `Extracting {train,test} cycle samples`, `Training`/the upstream `Traning` typo), so the four data phases are reported separately instead of collapsing into one "Loading data". A progress bar and a percentage are drawn **only** from a parsed `current/total`, recomputed as the exact ratio rather than tqdm's rounded percent, and a counter is attributed to the stage that printed it. A stage with no denominator gets a spinner and an elapsed time and never a fabricated percentage. An ETA appears only once the run has produced enough completed units to time one, is a moving average of the most recent ten, and is discarded when the stage changes because two stages do not share a unit of work. Two boundaries BatteryML announces with no output — training→evaluating and dataset-built→training — are inferred from the CLI's own call order and from nothing else, which is what keeps a one-shot sklearn fit from waiting for an epoch counter that never comes. The periodic heartbeat is flagged in the event rather than recognised by its text, and is kept out of the live tail and `run.log`; the raw stdout/stderr stays complete and one click away under a disclosure. Terminal state is read from the RunManager's status, never from the event stream, so a run that emits no progress at all still releases the page. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/streaming.py`, `batteryml_web/agent/progress_estimate.py` (`EpochTimer`), `batteryml_web/core/run_manager.py`, `batteryml_web/ui/agent_workflow.py` (`progress_view`), `docs/RUN_PROGRESS.md` |
| `UX-13` | The request boxes on Home and on Agent Workflow never trigger Streamlit's "created with a default value but also had its value set via the Session State API" warning. Each box is seeded through its own widget key before it is built and is never given a non-empty `value=`; with a user key, `value` is not part of the widget's identity in Streamlit 1.64 (`key_as_main_identity`), so nothing else changes. | `IMPLEMENTED-STATIC` | `batteryml_web/ui/agent_workflow.py` (`render_agent_workflow`), `batteryml_web/app.py` (`render_home_agent_entry`); tests `TestRequestAreaIsOneControl` in `tests/test_agent_web_ui.py`. `[VERIFIED]` in the authoring container only: **Use template** on both pages, zero warnings in the server log. |

The check that establishes this group is `RUN-01`
(`scripts/verify/check_run_context.py`); the tests are
`tests/test_run_context.py` (the four backend modules, without Streamlit)
and, in `tests/test_agent_web_ui.py`, the classes
`TestComputeDeviceSelector`, `TestProjectDeterminesTheWorkspaceInTheUi`,
`TestExplainOpensConfigurationInsteadOfDeadEnding` and
`TestNothingRuntimeIsHardcoded`. The last is written to fail if anyone later
writes a GPU count, a GPU name, a device string, a project path or a metric
value into the UI source, and its scan is proved non-vacuous by mutation.

`UX-08`–`UX-10` are covered by `tests/test_ui_layout.py`. Two of their
claims are about what a browser does rather than what the source says, so
they were checked by rendering the app with Playwright and measuring: that
run is what caught the reset being skipped by a page that raises, and what
established the two sidebar values `UX-10` now pins. The measurements
(sidebar content height 1202px unstyled → 627px, every navigation landing
at offset 0) are `[VERIFIED]` in this container only; the Streamlit version
on the server is not pinned by this repository, so re-checking the sidebar
there is worth one look.

## ACQ — data acquisition and dataset discovery

Rationale for the group: the protocol's first data step is a download, and a
fresh-environment replay of the README showed it could leave the data folder
in a state that every later step misreports. Upstream's `download_file`
opens the target before the request, so any failure leaves a 0-byte or
truncated file that the next run skips; preprocessing then fails with h5py's
"file signature not found", far from the cause. Separately, the readiness
list showed a `BatteryLife MISSING` row on every fresh clone that no action
could clear.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `ACQ-01` | `download` never leaves a broken file. The request is sent and its HTTP status checked before any file is created; data goes to `<name>.part` and is renamed to `<name>` only when the byte count matches the expected size (`DOWNLOAD_LINKS` length, else an unencoded `Content-Length`); any failure, Ctrl+C included, deletes the `.part` and exits 1 with one line naming the file, the URL and the fix. An existing file is kept only when its size matches; a 0-byte file is removed and fetched again, a short one is replaced once a complete copy exists. Upstream is not edited (`CLAUDE.md` §4): the CLI wrapper installs the replacement into `batteryml.preprocess` before `bin/batteryml.py` imports it. TLS behaviour (`verify=False`) is kept from upstream on purpose. | `IMPLEMENTED-STATIC` | `batteryml_web/core/safe_download.py`, `batteryml_web/core/run_batteryml_cli.py`; check `CLI-01`; tests `tests/test_safe_download.py`. `[VERIFIED]` in the authoring container only for the failure path: `download MATR` against a blocked host exited 1 and left the folder empty. A successful download was not possible there. |
| `ACQ-02` | Dataset discovery lists datasets, not the folders that hold them: `BatteryLife` and `BatteryLife_Raw` under `data/raw` are skipped (`RAW_CONTAINER_DIRS`); the datasets inside them are still assessed, and naming a container explicitly still assesses it. `tools/sync_data_and_config_manifests.py` no longer creates `data/raw/BatteryLife/v12` at import; it is created only when a BatteryLife dataset is actually linked into it. | `IMPLEMENTED-STATIC` | `batteryml_web/core/readiness.py` (`list_dataset_readiness`), `tools/sync_data_and_config_manifests.py`; tests `TestContainerFoldersAreNotDatasets` in `tests/test_batterylife_readiness.py`, `tests/test_fresh_clone_startup.py`. `[VERIFIED]` in the authoring container only: the README §10 command printed no `BatteryLife` row and `data/raw` stayed empty after the tool ran. |
| `ACQ-03` | A network or download error names the host that actually failed — the safe downloader's own `DownloadError` text, or the host of the failed request — and never an assumed one; URL credentials and secret query values are redacted from every such message. The partial-download guarantees of `ACQ-01` are unchanged. | `IMPLEMENTED-STATIC` | `batteryml_web/core/safe_download.py` (`redact`, `request_host`), `batteryml_web/core/run_batteryml_cli.py`, `batteryml_web/core/quick_experience.py` (`download_failure_host`); tests `TestHelpers` in `tests/test_quick_experience_fixes.py` |

## Result schemas (`RES-nn`)

Which figures a saved artifact earns. A figure is a claim about work that
happened, and the artifact — not the model's name, not the row count — is what
says which claim is true. Full rationale in `docs/RESULT_SCHEMAS.md`.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `RES-01` | A result artifact is classified from two independent things, never one standing in for the other: the **task** it recorded (`task_metadata`'s declared task, else its label annotator — never a model class name, because `RidgeRULPredictor` trains the SOH route) and the **granularity** of its rows. The categories stay semantically separate: `rul_scalar`, `rul_cyclewise`, `soh_cycle`, `soh_scalar`, `trajectory`, `unknown`. A sequence per row is a trajectory whatever the recorded task says. An unidentified task can never produce `rul_cyclewise`. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`task_family`, `schema_evidence`, `detect_schema`) |
| `RES-02` | A result is cycle-level only on positive evidence: a well-formed `sample_index` of matching length, **no** row carrying the `WHOLE_CELL_CYCLE` sentinel, a repeated cell id, and at least two distinct cycles for some cell. Shape alone is never sufficient — "more rows than cells" and "a `sample_index` exists" are both satisfied by whole-cell RUL artifacts, which write one index record per cell with `cycle_index = -1`. Each rejected rule has a fixture that would pass it. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`schema_evidence`), `tests/fixtures/synthetic_result_artifacts.py`, `tests/test_rul_cycle_views.py` |
| `RES-03` | **Cycle-wise RUL curves are shown only for artifacts that contain genuine per-cell, per-cycle RUL estimates.** The view returns per-cell entries rather than a flat series, so two cells cannot be joined into one line by default; cycles are sorted ascending within a cell, carrying observed and predicted with them; and every value is one the artifact stores — `true_rul = final_lifetime - cycle` is never reconstructed, and an artifact with no observed labels draws nothing and says why. A `rul_scalar` run gets no warning, only a collapsed caption stating that one estimate per cell means no cycle-wise curve. | `IMPLEMENTED-STATIC` | `batteryml_web/agent/result_views.py` (`build_rul_cycle_view`, `RUL_SCALAR_CURVE_NOTE`), `batteryml_web/ui/run_history.py` (`render_rul_cycle_figure`) |
| `RES-04` | Every schema dispatch site handles every category, so a working result cannot silently lose its figures by falling through to "cannot be visualized". | `IMPLEMENTED-STATIC` | `batteryml_web/ui/run_history.py`, `batteryml_web/ui/agent_workflow.py`, `batteryml_web/agent/artifact_preview.py` |

### Notes on the RES group

* Audited, not assumed: all 170 RUL configs across 23 models use
  `RULLabelAnnotator`, which subclasses `BaseLabelAnnotator` rather than
  `BaseCycleLabelAnnotator`, so `Task.is_cycle_level` is False and every RUL
  run takes the `SampleIndex.for_whole_cells` branch. The only registered
  cycle-level annotator is `CycleSOHLabelAnnotator`.
* Therefore: **cycle-wise RUL visualization is implemented, but no current
  production RUL artifact emits the required per-cell, per-cycle RUL schema.**
  The capability is dormant against real artifacts and exercised only by the
  synthetic fixtures. No RUL model training was changed to make a curve
  appear.
* That conclusion is asserted by
  `tests/test_rul_cycle_views.py::TestTheProductionRulArtifactAudit`, so it
  fails a test rather than going quietly stale. The day a cycle-level RUL
  annotator is added, that failure is the signal the feature can come out of
  dormancy.
* `RES-01` amends the `sample_index`-only rule `SOH-12` described. Under that
  rule a whole-cell RUL artifact — which now carries an index of `-1`
  sentinels — classified as a cycle-level SOH result.


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
the `GPU` group. `docs/RUN_PROGRESS.md` carries the progress contract behind
`UX-12` — which line of BatteryML each stage name comes from, and which two
boundaries are inferred and why. `docs/RESULT_SCHEMAS.md` carries the schema
rule behind the `RES` group, including the four tempting classifiers it
refuses and the fixture that would fool each one.

---

## Amendments

Record here any requirement found to be wrong or impossible, with the proposed
change, rather than implementing something different.

### `SOH-12` — "the schema is decided from the recorded `sample_index`"

**Found wrong.** `SOH-12` said a result's schema is decided from the presence
of a `sample_index`. That was true when only a cycle-level task wrote one.
`Task.build` now writes `SampleIndex.for_whole_cells` for a whole-cell run
too (`batteryml/task.py:102`), stamping every row with
`cycle_index = -1` — so under the old rule **every RUL artifact classified as
a cycle-level SOH result**, and would have been drawn with SOH axes and
per-cell curves.

**Amended, not replaced.** `RES-01` and `RES-02` keep the part of `SOH-12`
that was right — the schema comes from what the artifact records about itself,
never from the numbers — and fix the part that was not: the decision now reads
the recorded *task* and the *granularity* separately, and treats the
whole-cell sentinel as what it is. `SOH-12`'s text has been updated to point
at `RES-01`. Its other claims (per-cell curves, the range check, nothing
clipped) are unchanged.

### `SOH-11` — "it is the only entry the SOH catalog offers"

**Superseded, not wrong.** True when written; `SOH-13` adds the four audited
baselines beside Ridge. `SOH-11`'s text now says so rather than standing as a
claim the catalog no longer satisfies.
