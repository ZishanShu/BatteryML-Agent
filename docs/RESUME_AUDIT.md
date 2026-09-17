# Checkpoint / resume: audit result

Milestone C7 asks whether a genuinely safe, model-agnostic resume path
already exists that could be exposed, or whether Resume must be documented as
not implemented.

**Result: RESUME IS NOT IMPLEMENTED, and no Resume control is exposed.**

A checkpoint file existing is not a resume contract. Exposing a button
because `*.ckpt` is on disk would offer a scientifically different run under
the name of a continued one.

Every finding is `[STATIC]` — read from the source at the cited lines in this
repository. Nothing here was observed by running a training job.

---

## Per framework

| Question | NN models (`batteryml/models/nn_model.py`) | sklearn / XGBoost (`batteryml/models/sklearn_model.py`) | External models (`batteryml_web/models/`) |
| --- | --- | --- | --- |
| Resume supported safely? | **No** | **No** | **No** |
| Checkpoint format | `torch.save` of a `state_dict` | framework-specific dump | not applicable |
| CLI / config path | none exposed by the web wrapper | none | none |
| Optimizer state | **not saved** | not applicable | — |
| Epoch state | **not saved** | not applicable | — |
| RNG state | **not saved** | **not saved** | — |

## What actually exists

`Pipeline.train` and `Pipeline.evaluate` accept `ckpt_to_resume`
(`batteryml/pipeline.py:34`, `:83`, `:125-132`), and when it is given
`_prepare_model` calls `model.load_checkpoint(path)`.

For a neural model, `load_checkpoint` is
(`batteryml/models/nn_model.py:110-111`):

```python
def load_checkpoint(self, path: str):
    self.load_state_dict(torch.load(path))
```

That is **weights only**.

## Why that is not resume

Loading weights and calling it a resumed run changes the science in three
ways, none of them visible in the result:

1. **The optimizer restarts.** Adam's moment estimates are gone, so the
   first steps after "resuming" behave like the first steps of a fresh run,
   not like step *n* of the original. The loss trajectory differs.
2. **The epoch counter restarts.** Any learning-rate schedule keyed on the
   epoch restarts with it, so a "resumed" run trains at the wrong point on
   its own schedule.
3. **Determinism is lost.** `--seed` is recorded and re-applied from the
   start (CLAUDE.md §8), but RNG state at the checkpoint is not saved. An
   identical config plus seed plus data would no longer reproduce an
   identical `metrics.csv` for a run that had been resumed — which is the
   reproducibility contract this platform exists to keep.

There is also no route to it from the platform: the web wrapper
(`batteryml_web/core/run_batteryml_cli.py`) exposes no `--ckpt_to_resume`
flag, so resume is currently unreachable from any UI path.

## What a real implementation would need

Out of scope for this milestone, and it belongs upstream rather than in the
web layer:

1. checkpoints that carry optimizer state, epoch index and RNG state, not
   just weights;
2. a model-agnostic contract, since sklearn and XGBoost models have no
   equivalent of a mid-training checkpoint at all — "resume" cannot mean the
   same thing for them;
3. a decision, recorded in the spec, about what a resumed run's
   `run_manifest.json` claims: it is not the same experiment as an
   uninterrupted run with the same seed, and it must not claim to be.

Until those exist, an INTERRUPTED or CANCELLED run is re-run from the
beginning. That is slower and it is honest.
