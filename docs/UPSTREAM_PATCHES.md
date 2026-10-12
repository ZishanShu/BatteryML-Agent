# Upstream patches to `batteryml/`

`CLAUDE.md` §4 says new code goes in `batteryml_web/` and that a change to
upstream core must be recorded here with a rationale and a minimal diff.
This file is that record.

---

## UP-01 — `NNModel.fit` gains resume support and a per-epoch hook

| Field | Value |
| --- | --- |
| File | `batteryml/models/nn_model.py` |
| Function | `NNModel.fit` |
| Added | `resume_state=None`, `on_epoch_end=None` |
| Milestone | Training resume (A) |
| Basis | `[VERIFIED]` — `tests/test_training_resume.py` trains a model twice and compares parameters |

### Why it could not go in `batteryml_web/`

Resume needs two things that exist only inside the training loop:

1. **the optimizer actually in use.** It is a local variable created at
   `nn_model.py:56` and *rebuilt* at `:66` whenever a non-finite loss
   triggers `reset_parameters`. It is never stored on `self`, so no external
   caller can reach the object whose state must be saved.
2. **the epoch counter.** `range(self.train_epochs)` starts at 0 with no way
   to begin elsewhere, so a continuation would always restart — the exact
   fiction this milestone removes.

Subclassing would have meant copying the whole loop, which then drifts from
upstream and quietly diverges in scientific behaviour. Extending it is the
smaller and safer change.

### The diff, in words

* `fit(...)` takes `resume_state` and `on_epoch_end`, both defaulting to
  `None`, so **every existing call site behaves exactly as before**.
* when `resume_state` is given: the optimizer's `state_dict` is loaded, and
  the loop runs `range(start_epoch, self.train_epochs)` instead of
  `range(self.train_epochs)`.
* after each epoch, `on_epoch_end(epoch, {"optimizer", "model",
  "total_epochs"})` is called when supplied. This is how a caller in
  `batteryml_web/` can persist a full resumable checkpoint without upstream
  having to know the checkpoint format.

### What was deliberately not changed

* no scheduler and no AMP scaler were introduced — the loop has neither, and
  adding them would change training;
* the learning rate, batch size, optimizer choice, shuffling and the
  `reset_parameters` behaviour on a non-finite loss are untouched;
* `dump_checkpoint` / `load_checkpoint` keep their weights-only meaning, so
  existing checkpoints stay readable for inference and warm start.

**No scientific semantics changed.** With `resume_state=None` the loop is
byte-for-byte the previous behaviour, and the resume path is proved to
reproduce uninterrupted training on a deterministic model.

---

## UP-02 — `Pipeline.train` distinguishes a resumable checkpoint from a warm start

| Field | Value |
| --- | --- |
| File | `batteryml/pipeline.py` |
| Function | `Pipeline.train`, plus module-level helpers |
| Milestone | Resume from the web |
| Basis | `[STATIC]` — read from the source; the resume *mechanics* are `[VERIFIED]` by `tests/test_training_resume.py`, but this wiring has not been executed on real data |

### Why it could not go in `batteryml_web/`

`--ckpt-to-resume` already existed and already reached `Pipeline.train`, but
it led to `_prepare_model` → `model.load_checkpoint(path)` — the weights-only
path — and `fit` was called with no resume state. The decision of *which*
kind of checkpoint a path is, and the wiring of the state into `fit`, happen
inside `train()`. Nothing outside it can intervene without replacing the
execution entry point, which would mean a second execution system for
resumed runs.

### The diff, in words

* `train()` classifies `ckpt_to_resume`. A **resumable** payload is loaded,
  its model state applied to the unwrapped module, its RNG restored, and it
  is passed to `fit(resume_state=…)`. A **weights-only** file takes the
  pre-existing `_prepare_model` warm-start path, unchanged.
* an `on_epoch_end` callback writes a resumable checkpoint at the model's
  own `checkpoint_freq` — no extra writes, the existing ones simply carry
  optimizer, epoch and RNG state too.
* `_supported_fit_kwargs` passes the new kwargs only to a `fit` that accepts
  them, so **BatLiNet, which defines its own training loop, is called
  exactly as before**.
* `_unwrap` peels `torch.compile`'s `_orig_mod.` and DDP's `module.` before
  saving or loading, so a checkpoint is not tied to how the model was
  wrapped.

### Where the schema lives

These helpers use `batteryml.training_checkpoint` (UP-03) — an ordinary
module-level import of a sibling in the training layer.

An earlier revision of this patch imported
`batteryml_web.core.training_checkpoint` lazily, inside a `try`, so that
standalone upstream degraded to a no-op. That was a dependency from core
toward the web application layer, which `CLAUDE.md` §3 forbids. It has been
removed: the module moved down into `batteryml/`, the web layer re-exports
it, and there is still exactly one schema. With the schema beside the code
that reads and writes it, no `try` and no `None` degradation is needed, so
the three `if schema is None` branches are gone as well.

### What was deliberately not changed

* no change to the optimizer, learning rate, batch size, shuffling, epoch
  count semantics, or the `reset_parameters` behaviour on a non-finite loss;
* `dump_checkpoint` / `load_checkpoint` keep their weights-only meaning;
* the two training vocabularies are kept apart: the checkpoint records
  `model_class` (`CNNRULPredictor`) and never `model_key` (`cnn`), so a
  legitimate resume is not rejected for a naming difference.

**No scientific semantics changed.** With no `ckpt_to_resume`, `train()` is
the previous behaviour.

---

## UP-03 — `batteryml/training_checkpoint.py` (new file in the upstream tree)

| Field | Value |
| --- | --- |
| File | `batteryml/training_checkpoint.py` (moved, via `git mv`, from `batteryml_web/core/training_checkpoint.py`) |
| Milestone | Remove the web dependency from the training checkpoint core |
| Basis | `[VERIFIED]` — `tests/test_training_resume.py` and `tests/test_resume_launch.py` import the old path and pass unchanged; `tests/test_core_independence.py` asserts `batteryml/` contains no `batteryml_web` import |

### Why it is here and not in `batteryml_web/`

The resumable-checkpoint format belongs to training. `Pipeline.train` both
writes these files (`on_epoch_end`) and reads them back (`_load_resume_state`),
so as long as the schema lived in the web layer, core BatteryML had to import
the web application to resume its own training. That is the dependency UP-02
recorded as a concession; this patch removes it by moving the module down
rather than by duplicating it.

The module was already neutral — no Streamlit, no `batteryml_web` import,
`torch` imported lazily — so the move is a relocation, not a rewrite.

### The diff, in words

* `batteryml_web/core/training_checkpoint.py` → `batteryml/training_checkpoint.py`,
  contents unchanged apart from the module docstring;
* `batteryml_web/core/training_checkpoint.py` is now a compatibility
  re-export listing the same 20 public names in `__all__`. The re-exported
  objects are *the same objects* (`shim.classify_checkpoint is
  core.classify_checkpoint`), so there is one schema, one serializer and one
  validator, and every existing web-layer caller and test keeps its import
  path;
* `batteryml/pipeline.py` imports the sibling directly and drops the lazy
  `_resume_schema()` indirection together with its three `None` branches.

### What was deliberately not changed

* **the checkpoint format is untouched** — same `SCHEMA_KEY`, same
  `SCHEMA_VERSION = 1`, same payload layout, so checkpoints written before
  this change remain readable and resumable;
* no resume test was edited; they still import
  `batteryml_web.core.training_checkpoint`;
* the web UI stays capability-driven — it asks the schema whether a file is
  resumable, exactly as before;
* no circular import: `batteryml/__init__.py` does not import `pipeline`, and
  `training_checkpoint` imports nothing from `batteryml`.

**No scientific semantics changed.** This patch moves a file and deletes an
import.


---

## UP-04 — cycle-level samples: many rows per cell, each with an identity

| Field | Value |
| --- | --- |
| Files | `batteryml/data/sample_index.py` (new), `batteryml/data/databundle.py`, `batteryml/feature/base.py`, `batteryml/label/base.py`, `batteryml/label/soh.py`, `batteryml/task.py` |
| Milestone | SOH route, commit 1 |
| Basis | `[VERIFIED]` — `tests/test_soh_core.py` (33 tests) and the full suite (789 tests) |

### Why it could not go in `batteryml_web/`

SOH estimation is a cycle-level task: one cell yields one sample per observed
cycle. Every layer of the pipeline assumed one sample per cell —
`feature/base.py` and `label/base.py` both `torch.stack` a single row per
cell, `task.py` pairs them positionally, and `Dataset` holds two tensors with
no identity, so the row index *is* the cell.

Expanding cells into cycle rows inside `batteryml_web/` would have meant a
second execution path for SOH runs: a different assembler, a different
dataset object and a different route into the model. That is the parallel
system `CLAUDE.md` §3 and the milestone brief both rule out. The extension
belongs where the assembly already happens.

### The diff, in words

* **`SampleIndex`** (new) records `(cell_id, cycle_index)` per row, with the
  length checked against the tensors it describes. It is not a tensor and
  never reaches a model, so a cell id cannot be mistaken for a feature.
  `WHOLE_CELL_CYCLE = -1` marks a whole-cell sample such as RUL.
* **`BaseCycleLabelAnnotator`** and **`BaseCycleFeatureExtractor`** are new
  subclasses of the existing bases. The annotator owns which cycles are
  sampled and the extractor is asked for exactly those, in that order — one
  owner, so feature rows and label rows align by construction rather than by
  two components agreeing independently.
* **`Task.build`** switches to the cycle-level path when its label annotator
  is a `BaseCycleLabelAnnotator`, concatenating per-cell blocks instead of
  stacking per-cell rows, and filtering the sample index by the same NaN mask
  as the tensors. Pairing a cycle-level annotator with a whole-cell extractor
  raises, naming both.
* **`assert_disjoint_cells`** is called in `Task.build` on the loaded train
  and test cells, before any expansion. The split was already by cell; this
  checks the result rather than trusting the ordering to stay correct.
* **`Dataset` / `DataBundle`** take an optional sample index. Omitted, they
  behave exactly as before.

### What was deliberately not changed

* **every existing RUL workflow is untouched**: `BaseLabelAnnotator` and
  `BaseFeatureExtractor` keep their contract and their code, the whole-cell
  path in `Task.build` is the original three lines, and no existing config,
  extractor or annotator was modified;
* no model changed. `SklearnModel` was audited and has no task-specific
  assumption, so a cycle-level dataset reaches it as an ordinary tabular one;
* no metric, transformation or splitter changed.

**No scientific semantics changed for RUL.** With a whole-cell annotator the
pipeline is the previous behaviour plus an identity record that nothing reads.

---

## UP-05 — SOH labels: one reference-capacity convention per run

| Field | Value |
| --- | --- |
| File | `batteryml/label/soh.py` |
| Milestone | SOH route, commit 1 |
| Basis | `[VERIFIED]` — `tests/test_soh_core.py::TestReferenceCapacity`, exact ratios on synthetic cells |

### The defect this removes

The original annotator chose the reference capacity **per cell**: nominal
capacity when the cell carried one, otherwise that cell's first-cycle
discharge capacity. Two cells in the same dataset could therefore be labelled
under two different definitions of SOH, with nothing recording which — and the
labels differ by whatever ratio nominal and first-cycle capacity happen to
have (a factor of two in the test fixture).

### The diff, in words

* `reference_capacity_mode` is explicit, validated against
  `REFERENCE_CAPACITY_MODES`, and applies to every cell in the run. The
  default is `first_valid_discharge_capacity`. There is no fallback between
  modes: a cell that cannot satisfy the configured mode is **excluded and
  counted**, never relabelled under another.
* `mode='absolute'` is gone. It returned capacity in Ah while the same
  parameter's other value returned a ratio. Canonical SOH is a ratio.
* `CycleSOHLabelAnnotator` (new) is the canonical annotator:
  `SOH(cell, t) = Q_discharge(t) / Q_ref(cell)` for each cycle the sample
  policy accepts. Invalid cycles are dropped and counted, never interpolated;
  ratios above 1.0 are kept, never clipped.
* The sample policy (`min_cycle`, `max_cycle`, `cycle_stride`) is
  configuration, reported by `sample_policy()` for provenance.

### What was deliberately not changed

* `SOHLabelAnnotator` still exists and still produces one label per cell, for
  the `configs/soh/dummy/*` examples, which are now marked legacy in-file;
* the `soh_filepath` path still reads a supplied label file — but a missing
  file now raises instead of silently falling back to computed labels, which
  would have been two different label sources under one config.

---

## UP-06 — the prediction artifact carries sample identity and label conventions

| Field | Value |
| --- | --- |
| File | `batteryml/pipeline.py` (`Pipeline.evaluate`, `_task_metadata`) |
| Milestone | SOH route, commit 2 |
| Basis | `[VERIFIED]` — `tests/test_soh_pipeline.py::TestSOHTrainEvaluateArtifact` reads the written `.pkl` back |

### Why it could not go in `batteryml_web/`

`Pipeline.evaluate` is what writes `predictions_seed_*.pkl`. For a cycle-level
task the row index identifies nothing — one cell contributed many rows — so a
prediction could not be traced to the cell and cycle it was made for, and
"SOH vs cycle" could not be drawn at all. The information exists only inside
`evaluate`, at the moment the artifact is assembled.

### The diff, in words

* when the evaluated `Dataset` carries a `sample_index` (UP-04), the artifact
  gains `sample_index`: one `{cell_id, cycle_index}` per prediction, in row
  order;
* it also gains `task_metadata`: the label annotator, the reference-capacity
  mode and sample policy as configured, and the feature extractor. An SOH
  number is meaningless without the reference capacity it is a ratio of, so
  the convention travels with the predictions rather than living only in a
  config file that may be edited afterwards;
* `_task_metadata` reads the config as configured. Nothing is inferred and
  nothing is defaulted, so the artifact cannot claim a convention the run did
  not use.

### What was deliberately not changed

* a whole-cell task writes exactly the previous four keys — `sample_index` is
  absent, not `None`, because there row order *is* the cell order and nothing
  was lost;
* the existing keys, the filename and the pickle format are untouched, so
  every existing reader keeps working.

---

## UP-04 — `batteryml/preprocess`: partial-download handling and a completeness check

| Field | Value |
| --- | --- |
| Files | `batteryml/preprocess/download.py`, `batteryml/preprocess/__init__.py`, `batteryml/preprocess/preprocess_MATR.py` |
| Added | `is_download_complete(filename)`; `.part` download in `download_file`; a named error in `load_batch` |
| Origin | commit `f058d51` on `codex/ui-optimization` (repository owner), kept when that branch was merged with `main` |
| Basis | `[STATIC]` — read from the diff; not executed against a real download in the authoring environment |

### Why it is upstream

The Quick experience controls on the home page
(`batteryml_web/app.py`, `_raw_download_progress`) ask whether each MATR batch
file is *usable*, not merely present: a truncated HDF5 file can be gigabytes
long. `is_download_complete` opens the HDF5 header (`'batch' in file`) to
answer that, and it sits beside `DOWNLOAD_LINKS`, which names the files. The
owner chose to keep it there; this entry records that decision as `CLAUDE.md`
§4 requires.

### The diff, in words

* `is_download_complete(filename)`: missing or 0-byte → False; a
  `MATR_batch_*.mat` file → True only if `h5py` can open it and it contains
  `batch`; any other file with bytes → True.
* `download_file`: an existing target is skipped only when
  `is_download_complete` says so (otherwise it is deleted and fetched again);
  the body is written to `<name>.part` and renamed onto the final name only
  after the transfer finished; on a request/OS/value error the `.part` file
  is removed and the error re-raised.
* `__init__.py` exports `is_download_complete`.
* `preprocess_MATR.load_batch`: an `OSError` from `h5py.File` is re-raised as
  `OSError("MATR batch file is incomplete or corrupt: <name>. Delete it and
  download the file again.")`.

### What was deliberately not changed

* `download_file`'s parameters, so the platform's run-time replacement below
  still matches it (`CLI-01` checks this);
* `verify=False` on the request;
* the platform's own download path: the CLI wrapper still installs
  `batteryml_web/core/safe_download.download_file` before handing over, so
  `batteryml download` through the wrapper does not run this body. It applies
  to code that imports upstream `download_file` directly
  (e.g. `scripts/reproduce_batteryml/prepare_data.py`).

**No scientific semantics changed.** Only download bookkeeping and an error
message; preprocessing output for a complete file is unchanged.

---

## Not a patch: `download_file` is replaced at run time

Recorded here because it changes what an upstream function does when it is
called through this platform, even though **no file under `batteryml/` or
`bin/` was edited**.

* **Upstream behaviour (before UP-04).** `batteryml/preprocess/download.py`
  `download_file` opened the target file before sending the request. Any
  failure left a 0-byte or truncated file under the final name, and the next
  run printed `[INFO] … already exists. Skip it.` and kept it. UP-04 has since
  changed the upstream body as well; the wrapper still installs its own.
* **What the platform does.** `batteryml_web/core/run_batteryml_cli.py` calls
  `install_safe_download()` before `runpy.run_path("bin/batteryml.py")`. That
  sets `batteryml.preprocess.download_file` to
  `batteryml_web/core/safe_download.download_file`, and `bin/batteryml.py`
  picks it up through its own `from batteryml.preprocess import …
  download_file`. Same parameters; `.part` file, size check, atomic rename;
  spec `ACQ-01`.
* **Kept from upstream on purpose.** `verify=False` on the request. Changing
  TLS verification could break a dataset host that works today, and none of
  the hosts could be reached from the authoring environment to check.
* **Guarded by.** `CLI-01` fails if the wrapper stops installing it before the
  hand-over, or if its parameters drift from upstream's;
  `tests/test_safe_download.py` asserts this fix never edits the upstream
  file and that any upstream edit to it is recorded here (UP-04).

