# Run lifecycle — when a run is finished, and who says so

A run has exactly one outcome, and it is recorded in two places: in memory,
for the page that is watching it, and in `run.json`, which outlives the
server. This document is the rule that keeps those two from disagreeing.

Implemented in `batteryml_web/core/run_manager.py` (`_execute`,
`_run_to_completion`, `_finalize`), with the durable half in
`batteryml_web/core/run_manifest.py`.

## The contract

> **`snapshot()["finished"]` becomes true only after the manifest is already
> terminal.**

Everything else follows from that one ordering. A consumer that sees
`finished` may poll, may call `forget()`, may re-read the manifest, may
restart the server — and it finds one settled state, not two halves of one.

A run reaches exactly one of `COMPLETED`, `FAILED`, `CANCELLED`,
`INTERRUPTED`. `INTERRUPTED` is never written by a run about itself: it is
what `reconcile_status` *reports* for a manifest left non-terminal with no
live process, and the stored file is not rewritten.

Progress output has no part in this. An sklearn fit such as Ridge prints no
tqdm bar at all, so terminal detection reads the status and never the event
stream. Parsing progress is a display concern; see `docs/RUN_PROGRESS.md`.

## What went wrong, and how it was measured

Two defects, both reproduced before repair.

**The ordering was inverted.** `_execute` published the in-memory terminal
status and wrote the manifest afterwards. Measured with a poll that does not
sleep: the manifest was still `RUNNING` at the instant `finished` first
became true in **20 of 20** repetitions — typically for about 2ms, once for
446ms on an idle machine.

That window was not academic, because of what the page does inside it.
`_finish_active_run` reacts to `finished` by calling `forget()`, which drops
the run from `live_run_ids()`, and then reruns the script; Run History then
renders `list_runs(..., live_run_ids=...)`. A manifest still saying `RUNNING`
that is no longer live is exactly what `reconcile_status` treats as an
abandoned job — so a run that had just exited 0 was shown as **INTERRUPTED,
"not known to have finished"**. Through the real `forget`/`list_runs` path
that reproduced **15 of 15**.

The test that caught this polled every 20ms, which slept through the usual
2ms window and only noticed the occasional long one: it failed 3 times in 25
isolated runs, and more under load. The flake was the mild symptom.

**The worker was unguarded.** `execute_plan_streaming` was called with no
`try`. Anything it raised killed the worker thread with the status still
`RUNNING` — and `live_run_ids()` reads the status, not the thread, so the
manager went on reporting the run as live, the page polled a job whose
worker was dead, and orphan recovery never applied because the manager
insisted it was alive. **This one had no time bound at all.** The same shape
applied one step earlier: `run_dir.mkdir` ran outside the guard too, and an
unwritable parent stranded the run at `PLANNED`.

## The ordering now

```
child exits
  -> streaming drains the pipes and returns a result
  -> _finalize:
       1. write stdout.log / stderr.log
       2. update run.json to the terminal status      <-- durable first
       3. publish handle.status + handle.result       <-- `finished` here
       4. on_finished callback
```

Step 3 is in a `finally`, so a failed manifest write cannot strand the run at
`RUNNING` — that would trade a millisecond window for the permanent one this
also fixes. When there is no manifest, or it is already terminal,
`update_manifest` raises and is ignored: in neither case is anything left on
disk claiming `RUNNING`.

Every path out of `_execute`, exception or not, reaches exactly one
`_finalize`. An exception the executor never recovered from is recorded
`FAILED` with the exception text and **no return code** — no process outcome
was observed, and claiming an exit code would invent one.

## Real-server acceptance

Run on the server, from the repository root. Nothing below trains: one epoch
on the smallest configured CNN is enough to exercise start, exit, drain and
finalization.

```bash
# 1. Start the app and note the project you run in.
python -m streamlit run batteryml_web/app.py

# 2. In the UI: Run Model -> MATR / RUL / CNN, device cpu, seed 0, epochs 1.
#    Press Run. Copy the run id shown on the page.
RUN_ID=<paste the run id>
PROJECT=runs/web_projects/<your project>

# 3. Wait for the child to exit. It is in its own process group, so this
#    names only this run's tree:
watch -n 2 "pgrep -af run_batteryml_cli.py || echo 'no BatteryML child running'"

# 4. The durable record. This is the authority once the process is gone.
cat "$PROJECT/$RUN_ID/run.json" | python -m json.tool | \
  grep -E '"status"|"return_code"|"finished_at"|"elapsed_seconds"'
```

Expected, for a run that completed:

```
"status": "COMPLETED",
"return_code": 0,
"finished_at": "....Z",
"elapsed_seconds": ...,
```

`"status"` must be one of `COMPLETED`, `FAILED`, `CANCELLED`. It must **not**
be `RUNNING` or `PLANNED` once the child is gone — that is the defect this
document is about.

Then, in the UI: Review Results → Run History shows this run at the same
status, **not** "INTERRUPTED". That is the user-visible half, and it is the
one that was wrong.

### Inspecting a run that still looks stuck

```bash
python -m scripts.inspect_run_state "$PROJECT" --no-outcome-only
```

It prints every run whose manifest never reached a terminal status. It reads
only, and it does not claim to know whether a process is alive — liveness
lives in the server process that started the run. Run it while the server is
up and a listed run may simply be in flight; run it after that server has
stopped and a listed run is one that will never report.
