# Run progress — what the page may say a run is doing

A progress display is a claim about work that has happened. This document is
the rule the run page follows: **every stage name and every percentage traces
to something BatteryML printed**, and where nothing was printed the page says
so rather than filling the gap.

Implemented in `batteryml_web/agent/streaming.py` (parsing and events),
`batteryml_web/core/run_manager.py` (the live tail) and
`batteryml_web/ui/agent_workflow.py` (`progress_view` and the rendering).
There is one executor; this is a reading of its output, not a second one.

## The bars BatteryML actually prints

Copied from the source, with the line each is emitted at. Nothing else is
recognised.

| Output | Source | Stage |
| --- | --- | --- |
| `Reading train data` | `batteryml/task.py:79` | Reading train data |
| `Reading test data` | `batteryml/task.py:81` | Reading test data |
| `Extracting train cycle samples` | `batteryml/task.py:150` | Extracting train samples |
| `Extracting test cycle samples` | `batteryml/task.py:150` | Extracting test samples |
| `Traning` | `batteryml/models/nn_model.py:90` | Training |
| `Training <Model>` | e.g. `batteryml_web/models/batlinet.py:856` | Training |
| `Seed is set to …` | `batteryml/pipeline.py:285` | Loading data (no counter) |
| `Load datasets from cache …` | `batteryml/pipeline.py:256` | Loading data (no counter) |

`Traning` is an upstream typo. It is real output, so it is recognised;
"fixing" it by refusing to match it would lose every NN training bar.

These four data stages used to collapse into one "Loading data", so a run that
spent a minute reading cells and a minute expanding them into cycle samples
reported the same three words for both.

## Percentages come from counters, and only from counters

A tqdm line is matched on its full bar-and-counter shape, so a sentence that
merely contains `20/41` is not mistaken for progress. The fraction is
recomputed as `current / total`, not taken from tqdm's rounded percent: 20/41
draws at 48.78%, not 48%.

Where there is no counter there is no bar. The page draws a spinner and an
elapsed time — `Evaluating · elapsed 00:04` — and no number. The stages that
never carry a counter are declared in `INDETERMINATE_STAGES`, so the page can
choose a spinner immediately instead of waiting for a bar that is not coming.

A counter belongs to the stage that printed it. A finished `42/42` from
extraction is never redrawn as a training bar sitting at 100%.

## Two inferred boundaries, and why each follows

BatteryML prints nothing at either, so each is read from the CLI's own call
order rather than from output. They are the only inferences in the module.

* **Training → Evaluating.** `bin/batteryml.py:115 run()` calls
  `pipeline.train()` then `pipeline.evaluate()`, and the wrapper always passes
  both `--train` and `--eval` (`CLAUDE.md` §5). A full training counter means
  evaluation is what remains.
* **Extracting test samples → Training.** `Task.build` reads train cells,
  reads test cells, extracts train samples, extracts test samples, and
  returns; `Pipeline.train` then calls `model.fit`. A full *test extraction*
  counter means the dataset is built and fitting is what remains.

Each is announced in its own event, carrying no counter.

## The sklearn case

Ridge, Lasso, Elastic Net, Random Forest and XGBoost fit in one shot and print
nothing at all while doing it. That is not a defect and nothing waits for an
epoch counter that will never arrive. A SOH Ridge run reports:

```
Reading train data        41/41
Reading test data         42/42
Extracting train samples  41/41
Extracting test samples   42/42
Training                  spinner · elapsed …
Completed
```

`[VERIFIED]` against a real `bin/batteryml.py run --train --eval` on a
synthetic 12-cell fixture: the four data bars parse, and the run reaches
Training from the extraction counter alone. The recorded output is pinned in
`tests/test_agent_streaming.py::TestAgainstRecordedRealOutput` — real tqdm
draws `█`, which is why the recorded line is used rather than a hand-written
one.

A run that emits no progress events must still end. Terminal detection reads
the RunManager's **status**, never the event stream, so a silent sklearn fit
cannot leave the page polling a finished run.

## ETA

Only from observations the run produced. `EpochTimer` records how long each
newly completed unit took and averages the most recent ten, so a run that
speeds up or slows down is tracked rather than averaged flat. Below two
completed units it returns None and the line says `estimating…`.

The timer is scoped to the stage: arriving in a new stage discards the
previous stage's per-unit durations. Reading 41 cells and extracting samples
from 41 cells are not the same unit of work, so an ETA for the second computed
from the rate of the first would be a number with nothing behind it.

An indeterminate stage gets no ETA from timing at all.

## The heartbeat, and the `Running.` spam

While a run is quiet the executor emits a heartbeat every 0.2 s so an elapsed
time can tick. It repeats the last observed stage and counter and advances
neither.

It used to be appended to the live tail and to `run.log` like any other
output, which is why the run page showed a code box filling with hundreds of
identical `Running.` lines — the run's real output pushed out of the window
the tail keeps. Heartbeats now carry a `heartbeat: True` flag (a flag, not a
string to match), and `RunManager` skips them for both the tail and the log.

The primary view is **one structured state**: a bar with its stage, counter,
percentage, elapsed and ETA, or a spinner with its stage and elapsed. The raw
stdout/stderr stays one click away under *Technical log (raw output)*, and the
complete output is preserved in the run's logs regardless — carriage returns
included.

## Event shape

```python
{"type": "stage" | "training_progress" | "output" | "completed",
 "stage": str,
 "current": int | None,
 "total": int | None,
 "fraction": float | None,
 "message": str,
 "heartbeat": bool,
 "indeterminate": bool}
```

`fraction` is filled only when the run printed a counter. No metric is ever
read out of the output: scientific results come only from the canonical
`scores` BatteryML's own evaluator saved (`CLAUDE.md` §3), and
`tests/test_agent_streaming.py` asserts that no metric name is compiled into a
regex anywhere in the module.
