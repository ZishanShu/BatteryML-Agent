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
