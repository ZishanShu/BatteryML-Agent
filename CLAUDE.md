# CLAUDE.md — BatteryML Protocol Platform

> Repository root file. Claude Code reads this at the start of every session.
> Fill in the `<<< FILL IN >>>` markers before first use.

---

## 0. EXECUTION ENVIRONMENT — READ THIS FIRST

**You cannot run this pipeline. Assume you have no datasets, no GPU, and no working
BatteryML runtime.** The raw and processed battery data, the trained models, and the
compute live on a separate machine that you have no access to. You can read and write
source code in this repository; you cannot observe the behaviour of that code on real
data.

Three consequences, all non-negotiable:

1. **Never state or imply that you ran, tested, verified, or confirmed anything by
   execution.** If you did not observe an output, you did not verify it.
2. **Every factual claim you write — in code comments, docs, commit messages, or chat
   — must carry a status tag:**
   - `[VERIFIED]` — you observed the actual output. In this environment, almost nothing
     qualifies. Reserve it for things you genuinely ran (e.g. a pure-Python unit test
     on a committed fixture, if the environment allows even that).
   - `[STATIC]` — established by reading the code. Cite file and line.
   - `[ASSUMED]` — believed but unchecked. Say what would check it.
   A claim without a tag is a defect.
3. **Verification is performed by the user on the server.** Your job is to make that
   verification cheap and complete: for every change, you produce the exact command
   block the user should run and the exact output they should expect. The user pastes
   the real output back; only then may anything become `[VERIFIED]`.

Because you cannot test, correctness must come from **specification conformance and
defensive construction**, not from trial and error. Write code as if you get one shot.

## 1. What this repository is for

Reference implementation for a **Nature Protocols manuscript**: a step-by-step,
reproducible protocol for **battery aging prediction** (RUL and SOH), with
**BatteryML as the computational backbone**.

The paper is not "here is our software." It is "here is the standardized protocol for
the field, and here is the platform that executes it." Judge every decision against:

> Can an experimental battery scientist, on a laptop, follow the numbered steps and
> reproduce the reported numbers?

Three routes must all work:

- **Route A — LLM-guided workflow**: optional agent assists schema mapping, config
  generation, troubleshooting, result explanation.
- **Route B — Quick-start without LLM**: fully deterministic, no API key, offline.
  The reviewer's reproduction path. Must never break.
- **Route C — Extension**: adding a new dataset or a new model.

## 2. The standard is `docs/SPEC.md`

`docs/SPEC.md` is the single source of truth for what "done" means. Every requirement
in it has a stable ID (`EXEC-01`, `ART-03`, `UI-12`, …) and a status field.

Rules:

- Work is scoped by spec IDs, not by vague goals. Before writing code, state which IDs
  you are implementing.
- After writing code, update each ID's status to one of `IMPLEMENTED-STATIC`,
  `PARTIAL`, `BLOCKED`, or `NOT-STARTED`, with the implementing file path. Only the
  user may set `VERIFIED`.
- If you discover a requirement that is missing from the spec, add it as a new ID with
  a rationale rather than silently implementing it.
- If a spec item turns out to be wrong or impossible, say so and propose an amendment.
  Do not implement something different and call it done.

## 3. Hard architectural boundaries (never violate)

| Responsibility | Owner |
| --- | --- |
| Preprocessing, feature extraction | BatteryML |
| Model training, prediction | BatteryML |
| Metric computation (RMSE / MAE / MAPE) | BatteryML evaluator |
| Schema mapping *suggestions* | LLM agent (suggest only) |
| Config *drafting* | LLM agent (suggest only) |
| Error diagnosis, result explanation | LLM agent |
| Permission to execute anything | Validators |

**The LLM must never:** produce a numeric prediction, metric, or benchmark value;
modify raw data; write an unvalidated config to disk; bypass or retry around a
validator; apply a fix without explicit user confirmation.

**Agent output path is always:** `suggest → user review → validator → execute`.

If the agent is unavailable (no key, offline, rate-limited), every LLM-touched page
must remain fully usable by hand. Route B is a requirement, not a fallback.

## 4. Do not modify upstream BatteryML core

- New code goes in `batteryml_web/`.
- Extend through BatteryML's registry / extension points.
- If a core change seems unavoidable, **stop and ask**. If approved, record it in
  `docs/UPSTREAM_PATCHES.md` with rationale and a minimal diff.

## 5. Correct BatteryML CLI usage (from observed behaviour on the server)

Always go through the wrapper. Never run `python bin/batteryml.py` directly — it
shadows the real `batteryml` package (`ModuleNotFoundError: ... 'batteryml' is not a
package`).

Preprocess:

```bash
python batteryml_web/core/run_batteryml_cli.py preprocess \
  MATR data/raw/MATR data/processed/MATR
```

Train + evaluate — `config_path` is a **positional** argument, and `--train --eval`
are **mandatory** (without them BatteryML prints the workspace path and exits):

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace runs/native_matr_xgb_test \
  --train --eval --device cpu --seed 0
```

Wrong: `run --config configs/...` → `unrecognized arguments: --config`.

Data flow — training reads **processed** data, never raw:

```
data/raw/<DATASET> → preprocess → data/processed/<DATASET> → run → outputs
```

`AssertionError: data/processed/MATR` means preprocessing has not been run. It does
**not** mean the web layer or agent is broken.

These five facts are `[VERIFIED]` by the user. Treat them as ground truth and never
"simplify" them away.

## 6. Coding rules for an untestable environment

You get one shot, so:

- **Fail loudly, never silently.** No bare `except`, no `except: pass`, no default
  value substituted for a missing input. Every error message names the offending file,
  field, or command, and states the corrective action.
- **Validate at the boundary.** Anything entering from disk, the UI, or the LLM is
  checked against an explicit schema before use.
- **No hidden state.** Functions take explicit arguments; paths are parameters, not
  globals. This is what makes the code checkable without running the full pipeline.
- **Keep the risky part small.** Isolate the code that actually touches BatteryML and
  the filesystem into thin, obvious functions; keep logic in pure functions that could
  be unit-tested on a fixture.
- **Type hints everywhere**, so static checking substitutes for the tests you cannot run.
- **No unpinned new dependencies.** If a new package is genuinely required, justify it
  and pin it; prefer the standard library.
- **Every new module ships with**: (a) the code, (b) a corresponding check in
  `scripts/verify/`, (c) a docs entry, (d) updated spec IDs.

## 7. Mock data policy

- **No mock data, synthetic metrics, or placeholder numbers anywhere in the pipeline or
  in anything a user or reviewer could mistake for a result.**
- **Never** create an empty `data/processed/<DATASET>` to silence an assertion.
- The single exception: small, clearly labelled fixtures under `tests/fixtures/`, used
  only by unit tests, named with a `synthetic_` prefix, and physically incapable of
  being picked up by the dataset registry. A fixture may never satisfy a missing
  dataset, appear in `metrics.csv`, or reach a report.

## 8. Reproducibility contract (artifact filenames are part of the paper)

Every run, in both routes, writes to `runs/web_projects/<project_name>/`:

```
project_state.json
config.yaml
schema_mapping.json          # if a custom dataset was ingested
run_metadata.json            # git commit, package versions, seed, device,
                             # config hash, dataset fingerprint, wall-clock timing
environment_check.json
dataset_status.csv
feature_summary.json         # feature set, dimensionality, source cycles
training_log.txt
metrics.csv
predictions/
figures/
checkpoints/
report.html
```

**The manuscript quotes these filenames verbatim in its per-phase "Main outputs" lists.
Renaming one desynchronizes the code from the paper — flag it explicitly, never do it
silently.**

Determinism: `--seed` is always passed and always recorded. Identical config + seed +
data fingerprint must produce identical `metrics.csv`. If a model is inherently
non-deterministic, record that fact in `run_metadata.json` rather than hiding it.

## 9. Delivery workflow

Changes reach the server through GitHub, so:

- One branch per milestone: `milestone/<id>-<slug>`. Never commit to main directly.
- Commits are small and single-purpose; the message says what changed and which spec
  IDs it touches.
- Every milestone ends with a handover block containing: files changed, spec IDs moved
  and their new status, the **exact command block for the user to run on the server**,
  the **expected output** for each command, what to do if it differs, and an explicit
  list of what remains `[ASSUMED]`.
- Never mark a milestone complete on your own authority. Completion requires the user
  pasting back real output.

## 10. Working style

- **Plan before editing.** Anything beyond a one-line fix: state the plan and wait.
- **One milestone per session.** No opportunistic refactoring of adjacent code.
- Prefer reading code over assuming. If unsure whether something is integrated, grep
  and show the evidence.
- When something cannot be done, say so plainly instead of stubbing it.
- Repo-relative paths only, never `/Users/<name>/...`.

## 11. Project facts

- Repo: https://github.com/ZishanShu/BatteryML-Agent.git
- Default demo configuration: dataset **MATR**, task **RUL**, model **XGBoost**,
  device **cpu**, seed **0**.
- Target manuscript: Nature Protocols (2–4 h from setup to first prediction).

### Server environment (measured 2026-09-01)

> The dataset inventory previously recorded here was written from a **different
> machine** and did not describe this server. Every line below is now tagged with
> how it was established. Re-measure with `python -m scripts.verify.run_all` and
> read `runs/verify/dataset_status.csv` rather than trusting this list.

- Interpreter: Python 3.10.20, conda env `batteryml`. All 16 declared packages
  installed; `batteryml` importable. 8 × NVIDIA RTX 3090 visible.
  `[VERIFIED]` — ENV-01, user-pasted output.
- Dataset state, `data/processed/`:
  | Dataset | Processed | Raw | How established |
  | --- | --- | --- | --- |
  | `MATR` | populated, ready | present | `[VERIFIED]` SPLIT-01 + four real train/eval runs |
  | `HUST` | populated, ready | present | `[VERIFIED]` SPLIT-01 (`hust` preset ready) |
  | `SNL` | **directory exists, empty** | present | `[VERIFIED]` DATA-01; preprocessing ran and produced nothing (sweep) |
  | `HNEI` | **directory exists, empty** | **absent** | `[VERIFIED]` DATA-01 + sweep `skipped_no_raw` |
  | `UL_PUR` | **directory exists, empty** | **absent** | `[VERIFIED]` DATA-01 + sweep `skipped_no_raw` |
  | `OX` | **directory exists, empty** | not measured | `[VERIFIED]` DATA-01; raw state `[ASSUMED]` |
  | `CALCE` | absent | not measured | `[VERIFIED]` SPLIT-01 lists it missing and DATA-01 does not list it empty |
  | `RWTH` | absent | not measured | as `CALCE` |
- Split presets ready: `clo`, `hust`, `matr_1`, `matr_2`. Not ready: `cruh`,
  `crush`, `mix`, `snl`. `[VERIFIED]` — SPLIT-01.
- Verified-working path: MATR + `matr_1` train/eval on cpu, seed 0, confirmed for
  XGBoost's four siblings (Ridge, KernelRidge, ExtraTrees, Huber). Identical
  config + seed reproduces `metrics.csv` bit-for-bit after clearing `cache/`.
  `[VERIFIED]` — user-pasted output.
- **`cache/` is not invalidated by re-preprocessing.** `build_dataset` hashes only
  config fields (`batteryml/pipeline.py:186`), never the data. Clear `cache/`
  before trusting any number after a preprocessing run. `[STATIC]`,
  see `docs/DEBUGGING.md` §7.

### Current top-3 priorities

1. Complete the Nature Protocols protocol so **Route A (LLM-guided), Route B
   (quick-start, offline, no API key), and Route C (extension)** all work end-to-end.
2. Confirm and lock the MATR + XGBoost demo (preprocess → train → eval) as the
   reproducible Route B baseline.
3. `docs/SPEC.md` does not exist yet in this repo — it is cited throughout
   CLAUDE.md as the single source of truth for spec IDs. Decide whether to restore
   it or rework the spec-ID workflow before deep platform work.
