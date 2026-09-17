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
