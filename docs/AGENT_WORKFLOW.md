# The Agent Workflow

How a sentence becomes a validated BatteryML run, which module owns which
decision, and exactly what the user must check on the server.

Per `CLAUDE.md` §0, nothing here was established by running the pipeline.
Every claim is `[STATIC]` — established by reading the code — unless the
user has pasted back real output. The verification block at the end is what
turns any of it into `[VERIFIED]`.

---

## The pipeline

```
natural-language request
  -> intent resolution        deterministic first, LLM-assisted when needed
  -> canonical catalog        every identifier must already exist
  -> readiness                which models can run on this dataset/task
  -> execution plan           config + argv, built by the backend
  -> deterministic validator  the same gate a manual run goes through
  -> explicit Run button      the only way anything starts
  -> streaming executor       argv list, shell=False, live progress events
  -> BatteryML
  -> saved predictions/scores
  -> verified result context  metrics read from BatteryML's own artifact
  -> optional LLM explanation prose only, never a number
```

Two properties hold at every step, and the tests and checks exist to keep
them holding:

* **The LLM never decides anything.** It maps words onto identifiers that
  already exist and writes prose. Readiness, validation, planning and
  metrics are the backend's.
* **Nothing runs without an explicit click.** Analysis is not execution.

## Modules

| Module | Owns | Never does |
| --- | --- | --- |
| `batteryml_web/agent/recommendation.py` | which models the page names, given the request | name a model of its own, or consult readiness |
| `batteryml_web/ui/feedback.py` | transient success feedback, deduplicated | report a failure |
| `batteryml_web/agent/intent.py` | deterministic dataset/task phrase matching, English and Chinese | guess a default dataset |
| `batteryml_web/agent/catalog.py` | the canonical menu, derived from `ConfigRegistry` + readiness | carry a config path |
| `batteryml_web/agent/llm_intent.py` | the LLM call, and the refusal of everything it invents | plan or execute |
| `batteryml_web/agent/prompts.py` | the compact readiness context | invent a runnable model |
| `batteryml_web/agent/execution.py` | the plan: validated config, argv, epochs, device | start a process |
| `batteryml_web/agent/executor.py` | the pre-execution gate, blocking run | retry |
| `batteryml_web/agent/streaming.py` | the same run with progress events | read a metric from stdout |
| `batteryml_web/agent/run_policy.py` | what a smoke run is; what a failure means | choose a device for you |
| `batteryml_web/core/device_inventory.py` | the machine's real GPUs | assume a GPU count |
| `batteryml_web/agent/results.py` | metrics, from the saved `scores` only | recompute a metric |
| `batteryml_web/core/project_context.py` | which project a run belongs to, and its workspace under `runs/web_projects/` | let a run escape `runs/`, or silently fall back to the default project |
| `batteryml_web/agent/provenance.py` | the run's own account of itself, read from the plan's argv | invent a field the plan does not carry |
| `batteryml_web/agent/progress_estimate.py` | an ETA, once the run has earned one | assume a rate, or guess before two intervals are timed |
| `batteryml_web/ui/formatting.py` | metrics at reading scale | change the stored value |
| `batteryml_web/ui/settings_panel.py` | the LLM settings form, in the sidebar and in the Settings dialog a page opens (Configure LLM, Summarize/Explain without a connection, once after viewing a completed run's results with nothing configured — `UI-14`) | display, log or persist the API key |
| `batteryml_web/ui/agent_workflow.py` | rendering all of the above | any of the above |

## Two entry points

* **Home** (`batteryml_web/app.py`) offers Quick Start and, at the same
  level, a free-text box with **Analyze with Agent**. Home records the
  request and navigates; it resolves nothing itself.
* **Agent Workflow** analyses the handed-over request on arrival and shows
  how it was understood. The request itself is supplied through two tabs —
  **Describe request** and **Examples**, where the examples are built from
  this installation's catalog and only fill the box.

The model section answers the request rather than listing the platform: a
named runnable model is shown alone; a named blocked model is shown with its
exact blockers and runnable alternatives; an unregistered name is reported
as such; a recommendation shows at most three, filtered by readiness. The
full runnable and blocked catalogues live behind expanders. Where the
request named something blocked or unregistered, nothing is preselected and
no plan exists until the user chooses a model deliberately.

The sidebar shows the agent as an alternative intelligent entry point, not
as a numbered step of the manual protocol route. The manual route
(Prepare Data → Run Model → Review Results) works with no API key and is
the reviewer's reproduction path.

## Smoke runs

A smoke run is **one epoch of the model's own formal config**. The only
override is `--epochs 1`. No config is written or mutated; batch size,
support size and learning rate are exactly what the config declares. The
surface always shows:

> 1-epoch smoke test — workflow validation only; not benchmark performance.

BatLiNet's smoke run defaults to and recommends CPU (a 1-epoch CPU run of
`configs/external/batlinet/matr_1.yaml` was observed by the user at about
6.8 s, while the shared server's GPUs had roughly 2 GB free each and the
CUDA attempt died with OOM). An explicit GPU choice is always honoured.

## Devices

Devices are enumerated from `nvidia-smi` at runtime; the count is
discovered, never assumed, and each device shows its observed free/total
memory and utilization.

The compute control is a GPU multiselect: nothing ticked means CPU, one
means that GPU, several mean several. The choice is kept as a structure —
`{"mode": "cpu" | "single_gpu" | "multi_gpu", "gpu_ids": [...]}` — separate
from the `--device` string BatteryML receives. CPU and one GPU produce a
device string; a multi-GPU selection produces none, so `cuda:0,1` cannot be
built, and `validate_device` would refuse it anyway.

Whether a multi-GPU selection may run is a backend capability
(`supports_multi_gpu`, `multi_gpu_strategy`). No model carries it today, so
a multi-GPU choice disables Run with the reason rather than being downgraded
to a single GPU. The audit behind that is `docs/GPU_MULTI_GPU_AUDIT.md`.

## The run lifecycle, and running twice

Pressing **Run** consumes the reserved run id, writes the manifest, starts the
worker and stores the run id in `agent_active_run_id`. That key is what the
page branches on: while it is set, the block shows live progress and **Stop
run**; while it is clear, the block shows **Run** (or **Run again**, once this
page has executed something).

Two ordering rules make a second run possible, and both were repairs:

1. **A terminal run is released before anything fallible reads its results.**
   `_finish_active_run` clears `agent_active_run_id`, reaps the handle and
   records the status *first*, then reads the artifacts inside a guard. The
   original order was the reverse, and any failure in between — an
   unreadable predictions artifact, a value the state serializer refused —
   reached the page error boundary with the run still marked active. Every
   later script run re-entered the same failing branch, so the Run button
   never came back. A failure to read results is now reported as exactly
   that, beside an intact run directory.

2. **The staleness guard compares the experiment, not the argv.** The plan's
   argv carries `--workspace <agent workspace>/<run_id>`, and the reserved
   run id is re-minted on the rerun that follows every launch. Comparing
   argv strings therefore reported "the plan changed" on every rerun after a
   run, and deleted the finished run's results before the page could draw
   them. `executed_plan_signature` compares dataset, task, model, config,
   epochs, device and seed instead.

`RunManager.forget` reaps on *terminal status*, not on thread liveness: the
worker publishes its status and result together and then spends a moment
writing the manifest, and a page reacting to that snapshot arrives inside
exactly that window. Gating on `is_alive()` made the reap a silent no-op
there.

Terminal detection reads the recorded status and never the progress-event
stream. An sklearn fit such as Ridge reports no epochs at all, so a lifecycle
that waited for a progress event would never notice that it had finished.

A second run of the same plan is an ordinary act: a new run id, a new
directory, the earlier run untouched in Run History.

## Terminal state never depends on the output pipes

A run ends when the *process* ends. The streaming loop used to end only when
both pipes reached EOF, with no timeout and no look at the child, and a pipe
closes when the last holder of its write end closes it — which is not always
the child. Anything it spawned that inherited stdout or stderr keeps the pipe
open after the child is gone, and the loop then waited for an EOF that was
never coming: the manifest stayed `RUNNING` for the life of the server
process while the job had already finished. Reproduced with a child that
spawns a sleeper and exits 0.

The loop now also watches `proc.poll()`. Once the child has exited *and*
nothing has arrived for `POST_EXIT_DRAIN_S` (5s), it stops waiting and the
outcome is decided by the return code, which is already available. The drain
is measured from the last byte received, not from the exit, so a large
buffered tail still arriving keeps pushing the deadline out and is never
truncated — and a healthy run, which EOFs within milliseconds, never spends
the window at all. When the window is used, the result carries a
`stream_note` saying the capture may be short; the run's own outcome is not
in doubt.

Progress parsing is presentation. A missing tqdm line, a malformed counter
and a run that prints nothing at all each reach a terminal state exactly as
a chatty one does.

## One command builder, two routes

`build_run_command` in `batteryml_web/agent/execution.py` is the only place a
BatteryML argv is constructed. The Run Model page used to spell its own out
and omitted `--metric`, so the same config scored `MAE,RMSE` through the
Agent and fell back to the CLI default `RMSE,MAE,MAPE` through Run Model —
the two routes disagreed about what a result was. They cannot drift again
without changing the one function.

The routes still differ in what they are *for*, and that is deliberate: the
Agent Workflow runs through `RunManager` into a per-run directory with a
manifest and a lifecycle; Run Model runs synchronously into the project
workspace. The command they build for a given config, task, device and seed
is identical.

## Failures

A failed run is never retried. A CUDA out-of-memory failure is named as a
shared-memory condition, quotes the run's own error line, offers the CPU
smoke and any GPU that actually reports more free memory, and leaves the
full stdout/stderr — traceback included — in the debug expander. No metric
is ever produced from a failed run.

---

## Verification on the server

The author of this branch cannot run any of it. These are the commands that
establish it, with what to expect.

### 1. Static checks — no data, no GPU needed

```bash
conda run -n batteryml python -m unittest discover -s tests -p "test_*.py"
conda run -n batteryml python -m scripts.verify.run_all
```

Expect: the unit suite passes. In `run_all`, `DEV-01` and `INTENT-01` pass;
`DATA-01` and `REG-01` are the known failures on this branch and must not be
"fixed" here. `DEV-01`'s evidence block records the machine's real GPU
inventory — check that it lists the eight RTX 3090s with plausible free
memory.

If `DEV-01` fails, read its message: it names which contract drifted.

### 2. The BatLiNet CPU smoke run, through the plan the UI builds

```bash
conda run -n batteryml python - <<'PY'
from batteryml_web.agent.execution import build_execution_plan
plan = build_execution_plan(
    "MATR", "RUL prediction", "batlinet",
    device="cpu", seed=0, epochs=1,
    workspace="runs/web_projects/default/agent_matr_batlinet_smoke",
)
print(plan["config_path"])
print(plan["can_execute"])
print(" ".join(plan["command"]))
PY
```

Expect exactly:

* `configs/external/batlinet/matr_1.yaml`
* `True`
* a command ending `--train --eval --device cpu --seed 0 --epochs 1`

If `can_execute` is False, the reason is in `plan["reason"]` — usually
`data/processed/MATR` is missing, which is a preprocessing problem, not a
web-layer one.

### 3. The same run, executed with live progress

```bash
conda run -n batteryml python - <<'PY'
from batteryml_web.agent.execution import build_execution_plan
from batteryml_web.agent.streaming import execute_plan_streaming
plan = build_execution_plan(
    "MATR", "RUL prediction", "batlinet",
    device="cpu", seed=0, epochs=1,
    workspace="runs/web_projects/default/agent_matr_batlinet_smoke",
)
seen = []
def on_event(e):
    if e["type"] in ("stage", "training_progress") and e["message"] != "Running.":
        seen.append((e["type"], e["stage"], e["current"], e["total"]))
result = execute_plan_streaming(plan, on_event=on_event)
print("returncode:", result["returncode"], "stage:", result["final_stage"])
for row in seen[:20]:
    print(row)
PY
```

Expect: `returncode: 0`, `stage: Complete`, and among the events at least one
`('training_progress', 'Training', 1, 1)`. The stages should progress
`Loading data` → `Training` → `Evaluating`. **Paste the event list back** —
that is what establishes `EXEC-06` and `EXEC-07`.

If the events show only `Loading data` and then `Complete`, the tqdm format
has changed and `parse_progress_line` needs the new shape; nothing is broken
scientifically, but the bar will not move.

### 4. Metrics come from the artifact, not the output

```bash
conda run -n batteryml python - <<'PY'
from batteryml_web.agent.results import build_result_context
ctx = build_result_context(
    "runs/web_projects/default/agent_matr_batlinet_smoke",
    dataset="MATR", task="RUL prediction", model="batlinet",
    config_path="configs/external/batlinet/matr_1.yaml",
    run_status="SUCCESS", smoke_run=True,
)
print(ctx["success"], ctx["metrics"])
for w in ctx["warnings"]:
    print("-", w)
PY
```

Expect the RMSE/MAE/MAPE that BatteryML itself saved, and the smoke-run
warning among the warnings. The numbers must match the `scores` in the
predictions pickle; they are never parsed from stdout.

### 5. The UI

```bash
conda run -n batteryml streamlit run batteryml_web/app.py
```

Check by hand, and report what you see:

| # | Check |
| --- | --- |
| A | Home shows Quick Start **and** an Agent Workflow box at the same level; the `Agent Workflow` heading renders as a heading, with no anchor icon, `[svg]` or stray markup (see J: no emoji) |
| B | "Use CPTransformer on MATR to predict battery lifetime." → MATR / RUL prediction, CPTransformer selectable, PBT listed as blocked, Run is the only way to start |
| C | "Recommend a model to predict battery lifetime on MATR." → only READY models are recommended |
| D | "帮我用 MATR 做电池寿命预测" → MATR + RUL prediction (this one works with no API key) |
| E | Selecting BatLiNet with the smoke box ticked defaults the device to CPU and states why; the plan shows `Epochs override: 1` |
| F | While a run is going, the stage and elapsed time update; a bar appears only once real epoch numbers arrive |
| G | A MICH request reports RAW_ONLY with the backend's own text and offers no `data/raw/MICH` preprocessing command |
| H | The result panel shows metrics only for a successful run, with the smoke warning visible |
| I | Creating a project shows a transient toast, with the project name and workspace path left on the page as plain text — no permanent green box |
| J | The Agent Workflow entry reads exactly `Agent Workflow` on Home, in the sidebar, as the page heading and on the navigation button — no emoji, no caret — and the sidebar carries no "Intelligent entry point" copy |
| K | The request area is one text area with no tabs and no example gallery; while empty it shows one grey suggestion, and `Use suggestion` fills the box and does nothing else |
| L | The Streamlit toolbar offers no "Clear cache" or developer entries (`.streamlit/config.toml`, `toolbarMode = "viewer"`) — this is the one item only the running app can confirm |
| M | Ticking two or more GPUs leaves the selection as chosen, shows "True multi-GPU execution is not yet supported for this model/runtime.", and renders Run disabled; ticking one GPU still runs |

Runtime context, provenance and settings (the `RUN` group). These twenty are
the acceptance list for that milestone; every one of them needs the running
app, so none can be established from the authoring environment:

| # | Check |
| --- | --- |
| N | **Compute device** is an explicit CPU/GPU control, visible before any GPU interaction, opening on CPU |
| O | In CPU mode there is no GPU control on the page at all, and a run starts with `--device cpu` |
| P | Switching to GPU reveals exactly as many options as the machine has GPUs — count them against `nvidia-smi -L` |
| Q | Each GPU option reads `GPU <i> · <name> · <free> free · <util>% util` with the driver's own figures; a figure the driver did not report is absent, not zero |
| R | On a machine with no readable `nvidia-smi`, GPU mode offers at most the single manual `cuda:0` override, shows the enumeration error, and warns that its availability is unknown |
| S | Ticking one GPU sends `--device cuda:<i>` — the index you ticked, not always `0` |
| T | Ticking two GPUs keeps both ticked, disables Run with the multi-GPU message, and no `cuda:0,1` string appears anywhere, including the plan's command |
| U | Create a project: the toast reads `Project "<name>" created and loaded`, exactly once, and the sidebar then shows that project under **Current project** |
| V | Load a different project: the toast reads `Project "<name>" loaded`, once, and the previous run's plan and result are cleared rather than shown under the new project |
| W | With project `alpha` loaded, the Run directory reads `runs/web_projects/alpha/agent_<dataset>_<model>`; load `beta` and it changes to `beta` |
| X | The run context (Project, Run directory, Dataset, Task, Model, Config, Seed, Epochs, Device) is shown **before** you press Run |
| Y | After the run finishes, the same nine fields are shown with the same values — compare them line by line against what you saw in X |
| Z | Paths in the run context are repo-relative; "Exact paths" expands to the absolute form actually on disk |
| AA | Confirm the run directory in the provenance block is the directory that actually appeared on the filesystem |
| AB | Metrics are readable at a glance — no oversized or clipped digits — and the exact stored value is visible beside each compact one |
| AC | MAPE's percent form matches the stored ratio ×100; the stored ratio is still what `metrics.csv` carries |
| AD | During training, the progress line says `estimating…` until at least two epochs have been timed, then shows an ETA that tracks the observed rate. A run with no counter shows a stage and an elapsed clock and no bar |
| AE | Settings are in the left sidebar; the top chrome is reduced but the sidebar still opens and closes normally |
| AF | With no assistant configured, **Explain computed result** is clickable and opens the LLM configuration form (Cancel / Save settings / Save & explain); nothing is sent before you save |
| AG | The API key field is masked, and the key appears nowhere on the page, in any toast, in the debug expander, or in any file under `runs/` — grep for it to be sure |
| AH | LLM settings show **Save** and **Connect** — no Cancel / Save settings — in the sidebar and in the form Explain opens |
| AI | Save stores the values and reports "LLM settings saved" without contacting the endpoint (watch the network, or save with the endpoint down) |
| AJ | Connect against a working endpoint reports "LLM connected" and the status becomes Connected; against a wrong one it leaves a persistent error naming the reason, with no key in it |
| AK | Connect from the Explain form closes it and leaves the result on screen; the explanation appears only when you press Explain again |
| AL | Pressing Tab in the request box moves focus and does **not** fill it — the suggestion is accepted only by `Use suggestion` (docs/STREAMLIT_KEYBOARD_AUDIT.md says why) |
| AM | Type into the request box while `Use suggestion` is still visible, then click it: your text must survive |

Run lifecycle, results and recovery. These need the real server and a real
run; none may be marked VERIFIED from unit tests.

| # | Check |
| --- | --- |
| AN | Start a 1-epoch BatLiNet CPU smoke run; the page stays responsive and shows a Run ID before it starts |
| AO | While it runs, the stage, counter, elapsed clock and (once earned) the ETA update without you touching the page |
| AP | Start another suitable run and press **Cancel run**; confirm with `ps` that the training process is really gone |
| AQ | The cancelled run shows "Cancelled by user" and **no metrics** |
| AR | Refresh the browser; both runs are still in Run History with their real statuses |
| AS | Run the same model twice; the two runs have different Run IDs and different run directories |
| AT | Switch project and confirm the history and the run directories are isolated |
| AU | Open an older completed run from history; its result appears without re-running it |
| AV | A completed RUL run shows observed-vs-predicted and residuals, if its artifact carries observed labels |
| AW | Spot-check three rows of the prediction table against the saved artifact |
| AX | Download the run bundle; confirm it contains the manifest, metrics, environment and logs |
| AY | Grep the bundle for your API key and for `data/raw` — neither may appear |
| AZ | Set a wrong LLM key and press Connect: an actionable error, and the computed workflow still works |
| BA | With the LLM unreachable, a request still resolves deterministically and says assistance was unavailable |
| BB | Force a CUDA OOM: a clear message, a suggested action, no retry and no device switch |
| BC | Restart Streamlit: completed history is still visible, and any run that was active is INTERRUPTED, not COMPLETED |

Compute infrastructure. Resume is implemented and unit-proved; DDP
orchestration is implemented and launch-proved on CPU/gloo, but **no model is
marked DDP-capable**, so BF–BI are the audit that would lift that gate rather
than a check of a shipped feature.

| # | Check |
| --- | --- |
| BD | Train a small NN model (CNN/LSTM/MLP/Transformer) far enough to write a resumable checkpoint, then confirm `python -c "from batteryml_web.core.training_checkpoint import classify_checkpoint; print(classify_checkpoint('<path>'))"` reports `resumable` with the right epoch |
| BE | Point the same call at an **older** `.ckpt` from a previous run: it must report `weights_only`, and the UI must not offer Resume for it |
| BF | Confirm the two-rank launch on real GPUs: `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m batteryml_web.core.distributed_runner --config <cfg> --workspace <dir> --seed 0 --describe-only` — expect two JSON lines, `world_size: 2`, ranks 0 and 1, `backend: nccl`, devices `cuda:0` and `cuda:1`, exactly one `is_primary: true` |
| BG | Repeat BF with `CUDA_VISIBLE_DEVICES=5,6`; the ranks must still report `cuda:0` / `cuda:1` (the renumbering is the point) and `nvidia-smi` must show the load on GPUs 5 and 6 |
| BH | While BF runs, confirm with `nvidia-smi` that two worker processes occupy the two selected GPUs and nothing else on the box was touched |
| BI | Read `third_party/BatteryLife/models/CPTransformer.py` and confirm nothing in its forward/loss assumes a global batch, a fixed rank count, or cross-sample state — the audit `docs/DDP_CONTRACT.md` §6 needs before CPTransformer may be enabled |

Resume from the web. One cheap NN model; do not start a long benchmark.

| # | Check |
| --- | --- |
| BJ | Run a supported NN model (CNN/LSTM/MLP/Transformer) on a small config with `target epochs = 3` and a `checkpoint_freq` that writes at least once before the end |
| BK | Confirm a `*.resume.ckpt` appeared in the run directory, and that `classify_checkpoint` reports it `resumable` with the expected epoch |
| BL | Open Run History, select that run, expand **Resume training**: it shows the last checkpoint epoch and a target field. A BatLiNet or XGBoost run must show no Resume at all |
| BM | Set target total epochs to 5 and press **Resume** |
| BN | In the new run's log, confirm training begins at `checkpoint_epoch + 1` — not at 0, and not running 5 more on top |
| BO | Confirm the source run is unchanged (same status, same metrics, same files) and the new run shows "Resumed from &lt;run_id&gt; at epoch N" |
| BP | While the resumed run is going, press **Stop run** and confirm it becomes CANCELLED with no metrics; then repeat and let one finish to confirm a normal result |

**True multi-GPU execution is still not implemented.** T checks that the
platform says so rather than attempting one.

### 6. If a GPU run OOMs

That is the expected behaviour on the shared server, and it is what the
message is for. Confirm that the page says CUDA out of memory on the device
you picked, offers CPU and any GPU with more free memory, states that
nothing was retried, and still shows the full traceback in the debug
expander.

## When a request cannot run (UI-15, UI-16)

"Dataset READY" describes the data only. Whether *dataset + task + model*
can run also needs a registered run configuration that passes readiness and
validation. When the analysed combination cannot run, **Backend recommended
next steps** (deterministic, no LLM) says why and lists, from the
configuration registry and readiness only:

1. other runnable models for the same dataset and task;
2. the same task on other datasets (registered pairs only; a dataset that is
   not READY is named but never offered);
3. other tasks on the same dataset.

Each runnable alternative has an action that rewrites the request and
re-analyses it; nothing runs until **Run**. Example, with the registry in
this repository: `Use ElasticNet on HUST to predict soh.` → HUST has no
registered SOH estimation configuration; SOH estimation is registered for
MATR only (Elastic Net, Lasso, Random Forest, Ridge, XGBoost), offered when
MATR is READY; RUL prediction is registered for HUST.

**Explain this workflow** is the optional LLM explanation of the same
facts. It needs a connected LLM (shared settings), sends one request per
click, shows the answer directly beneath the button and may explain a
workflow that cannot run — never as a plan to execute.

