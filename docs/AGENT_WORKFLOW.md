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
| `batteryml_web/ui/agent_workflow.py` | rendering all of the above | any of the above |

## Two entry points

* **Home** (`batteryml_web/app.py`) offers Quick Start and, at the same
  level, a free-text box with **Analyze with Agent**. Home records the
  request and navigates; it resolves nothing itself.
* **Agent Workflow** analyses the handed-over request on arrival, shows how
  it was understood, what was refused, which models are runnable and which
  are blocked, then builds a plan for the selected model.

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
memory. A run goes to exactly one torch device — `cuda:0,1` is refused at
the planner boundary. There is no multi-GPU run and none is advertised; the
reasoning is in `docs/GPU_MULTI_GPU_AUDIT.md`.

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
| A | Home shows Quick Start **and** an Agent Workflow box at the same level; the `🤖 Agent Workflow` heading renders as a heading, with no anchor icon, `[svg]` or stray markup |
| B | "Use CPTransformer on MATR to predict battery lifetime." → MATR / RUL prediction, CPTransformer selectable, PBT listed as blocked, Run is the only way to start |
| C | "Recommend a model to predict battery lifetime on MATR." → only READY models are recommended |
| D | "帮我用 MATR 做电池寿命预测" → MATR + RUL prediction (this one works with no API key) |
| E | Selecting BatLiNet with the smoke box ticked defaults the device to CPU and states why; the plan shows `Epochs override: 1` |
| F | While a run is going, the stage and elapsed time update; a bar appears only once real epoch numbers arrive |
| G | A MICH request reports RAW_ONLY with the backend's own text and offers no `data/raw/MICH` preprocessing command |
| H | The result panel shows metrics only for a successful run, with the smoke warning visible |
| I | Creating a project shows "✅ Project created successfully" with the project name and workspace path |

### 6. If a GPU run OOMs

That is the expected behaviour on the shared server, and it is what the
message is for. Confirm that the page says CUDA out of memory on the device
you picked, offers CPU and any GPU with more free memory, states that
nothing was retried, and still shows the full traceback in the debug
expander.
