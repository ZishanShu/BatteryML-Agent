# SOH route — audit of what BatteryML already has

Step 0 of the SOH milestone: inspect the actual repository before writing any
SOH code, and prefer completing BatteryML's intended design over inventing a
parallel one.

Every claim below is `[STATIC]` — read from the source in this checkout, at
commit `138c62a` — unless marked otherwise. Nothing here was executed against
real battery data: this container has **no `data/` directory**, so no claim
about a dataset's SOH eligibility can be made here at all.

**Conclusion up front: the upstream SOH route is a placeholder, and two of its
gaps cannot be closed without a decision that is not mine to make.** They are
listed in "Blocking questions" at the end. Implementation stops there.

---

## 1. Component status

| Component | Path | Status | What is actually there |
| --- | --- | --- | --- |
| SOH label annotator | `batteryml/label/soh.py` | **partial** | Real code, registered, but one label per *cell* at one fixed cycle. Reference capacity is chosen per cell, not per dataset (§2). |
| SOH label registration | `batteryml/label/__init__.py:5` | complete | `SOHLabelAnnotator` is imported and registered. |
| SOH models | `batteryml/models/soh_predictors/` | **placeholder** | Contains `__init__.py` and nothing else. No model. Its docstring defines SOH as "the discharge capacity of the next cycle" — a *third* definition (§2). |
| SOH configs | `configs/soh/dummy/` | **placeholder** | Two configs, both named `dummy`, both using `DummyRULPredictor` (`sklearn.dummy.DummyRegressor`, predicts the training mean). |
| SOH example | `soh_example.ipynb` | **placeholder** | See §3. Its second half trains on `random.uniform`-generated labels. |
| SOH feature extractor | — | **missing** | No SOH-specific extractor exists. The configs reuse `VarianceModelFeatureExtractor`, an RUL feature set. |
| Cell-level split | `batteryml/train_test_split/` | complete | Splitting is already by cell, upstream, for every splitter (§5). |
| Cycle-level sample generation | — | **missing** | The pipeline is one-sample-per-cell end to end (§4). |
| Per-sample identity (cell_id, cycle) | — | **missing** | `Dataset` holds two tensors and nothing else (§6). |
| Metrics | `batteryml/data/databundle.py:100` | complete | `RMSE`, `MAE`, `MAPE` available; inverse-transform applied before scoring. |
| Web task vocabulary | `batteryml_web/core/readiness.py:51` | complete | `TASK_SOH = "SOH estimation"` exists and is distinct from RUL and trajectory. |
| Agent SOH intent | `batteryml_web/agent/intent.py:50-142` | complete | English and Chinese SOH phrases already resolve to `TASK_SOH`. |
| Web SOH config discovery | `batteryml_web/core/config_registry.py:186` | **broken** | Misclassifies 42 RUL configs as SOH (§7). |
| SOH readiness | `batteryml_web/core/readiness.py` | **missing** | Readiness is dataset-level; there is no SOH-specific eligibility test. |
| SOH result views | `batteryml_web/viz/loaders.py:70` | **partial** | Knows an SOH axis label. No SOH-vs-cycle view, because no artifact carries a cycle. |

---

## 2. What SOH definition does BatteryML intend?

**There is no single one. The repository contains three, and they disagree.**

**(a) `batteryml/label/soh.py:56-66` — ratio at a fixed cycle**

```python
Qd = max(cell_data.cycle_data[self.cycle_index-1].discharge_capacity_in_Ah)
if self.mode == 'relative':
    nominal_capacity_in_Ah = 1
    if hasattr(cell_data, 'nominal_capacity_in_Ah') and cell_data.nominal_capacity_in_Ah:
        nominal_capacity_in_Ah = cell_data.nominal_capacity_in_Ah      # convention A
    else:
        nominal_capacity_in_Ah = max(cell_data.cycle_data[0].discharge_capacity_in_Ah)   # convention B
    label = Qd / nominal_capacity_in_Ah
else:
    label = Qd                                                          # not a ratio at all
```

So `mode='relative'` is a ratio, `mode='absolute'` is capacity in Ah, and the
denominator of the ratio is **chosen per cell at runtime**: nominal capacity if
the cell happens to carry one, otherwise that cell's first-cycle discharge
capacity. Two cells in the same dataset can therefore be labelled under two
different conventions, and nothing records which. This is exactly the silent
mixing the milestone brief forbids.

There is also a `TODO` in that branch asking whether the first-cycle fallback
should be a mean of the first five cycles — upstream itself is undecided.

**(b) `batteryml/models/soh_predictors/__init__.py:4`** — "estimate the state of
health (i.e., **the discharge capacity of the next cycle**)". That is
next-cycle forecasting, not current-cycle SOH.

**(c) `configs/baselines/**/*.yaml`** — `RULLabelAnnotator(eol_soh: 0.9)` uses
"SOH" to mean the end-of-life *threshold* for an RUL label, not a target.

**Per-cycle capacity lives in** `CycleData.discharge_capacity_in_Ah`
(`batteryml/data/battery_data.py:25`), a list per cycle; the annotator takes its
`max()`. Nominal capacity, when present, is `BatteryData.nominal_capacity_in_Ah`
(`:89`).

---

## 3. How `soh_example.ipynb` constructs targets

Sixteen cells. It is a demonstration of the label *plumbing*, not a working SOH
route:

1. Cells 3-7 run `configs/soh/dummy/matr_1.yaml`: `SOHLabelAnnotator`
   (`cycle_index: 300`, `mode: relative`) with `DummyRULPredictor`. The reported
   `RMSE: Train 0.032, test 0.026` is the score of a model that predicts the
   training mean — a spread statistic, not a result.
2. Cells 9-15 demonstrate loading labels from a file. The file is **generated
   with `random.uniform(0.0005, 0.001)` degradation rates** in cell 10, so the
   reported `RMSE ... test 0.043` is a dummy model fitted to random numbers.

It also shows the feature/label timing upstream actually uses:
`critical_cycles: [2, 9, 99]` with `cycle_index: 300` — i.e. features from
cycles ≤ 99 predicting SOH at cycle 300. That is **forecasting SOH 200 cycles
ahead**, not "estimate SOH at the observed cycle from information available at
that cycle", which is what this milestone defines as the canonical task. The
existing example cannot be reused as-is for the canonical semantics.

---

## 4. Is there an intended cycle-sampling representation? No.

The whole pipeline is one row per cell:

- `batteryml/feature/base.py:17-21` — `torch.stack([process_cell(c) for c in cells])`
- `batteryml/label/base.py:12-14` — `torch.stack([...]).float().view(-1)`
- `batteryml/task.py:56-72` — extracts, masks NaN labels, builds a `DataBundle`

A cycle-level SOH sample set `(cell, cycle t) -> SOH_t` does not fit this: it
needs one cell to produce *many* rows. Doing that means changing
`feature/base.py`, `label/base.py` and `task.py` — **upstream core**, which
`CLAUDE.md` §4 says to stop and ask about.

One thing already works in our favour: `Task.build()` splits **cells** before
touching features (`task.py:47`), so the cell-level-split-before-cycle-expansion
rule the brief makes a hard requirement is the direction upstream already
points. The NaN-label mask at `task.py:63-67` is also already the "exclude
invalid samples explicitly" mechanism.

---

## 5. Which models are actually compatible?

Audited rather than assumed:

- `batteryml/models/sklearn_model.py:14-41` — `fit()` does
  `model.fit(feature.view(n, -1), label)` and `predict()` returns a flat
  tensor. **No RUL-specific assumption anywhere**: no lifetime, no EOL
  threshold, no cycle semantics. It is a plain tabular regressor.
- `XGBoostRULPredictor` (`rul_predictors/xgb.py`) and the other sklearn
  wrappers are three-line subclasses of it. The `RUL` in the class name is
  **naming only**.
- `DummyRULPredictor` is `sklearn.dummy.DummyRegressor` — fine as a floor to
  compare against, never as the baseline.

So a scientifically valid SOH baseline is available today at the model layer.
The obstacle is the sample representation (§4), not the model.

`batteryml/models/soh_predictors/` exists, empty, which is where an
SOH-named model would go if we want provenance to stop saying "RULPredictor"
for an SOH run (brief §17).

---

## 6. Prediction artifact cannot identify a sample

`pipeline.evaluate` (`batteryml/pipeline.py:129-138`) writes
`{'prediction', 'scores', 'data', 'seed'}`, and `data` is a `DataBundle` whose
`Dataset` holds only `feature` and `label` (`databundle.py:11-24`). There is no
`cell_id` and no `cycle_index` on a sample.

The brief's §11 artifact (`cell_id`, `cycle_index`, `observed_soh`,
`predicted_soh`) and its §12 "SOH vs cycle" view therefore cannot be built on
the current artifact at all — not as a UI gap, but because the information is
discarded before the artifact is written.

---

## 7. Live defect found: 42 RUL configs are advertised as SOH

`batteryml_web/core/config_registry.py:189` classifies a config as SOH when the
string `soh` appears anywhere in it:

```python
if any(x in hay or x in label_name for x in ("soh", "state of health")):
    return "SOH estimation"
```

Every RUL baseline config carries `label: {name: RULLabelAnnotator, eol_soh: 0.9}`,
and `eol_soh` contains `soh`. Measured in this checkout `[VERIFIED]`:

```
classified as SOH estimation: 44
  genuinely SOHLabelAnnotator:  2     (both configs/soh/dummy/*, DummyRULPredictor)
  actually RUL (eol_soh param): 42
```

So the platform today offers a user 44 "SOH estimation" configs of which 42 are
lifetime prediction wearing an SOH label, and the remaining 2 are dummies. This
is precisely the "do not relabel battery-life prediction as SOH" prohibition,
already happening, and it must be fixed before any SOH model catalog (brief
§15) can be trusted.

---

## 8. Dataset eligibility cannot be assessed here

`data/` does not exist in this container. Which datasets have enough cells with
enough valid `discharge_capacity_in_Ah` cycles to support SOH labels is an
empirical question about the processed files on the server. The brief is
explicit that MATR/HUST must not be assumed. An eligibility *checker* can be
written and unit-tested against synthetic fixtures here; the *answer* has to
come from a run on the server.

---

## Blocking questions

Two decisions are needed before implementation continues. Both are flagged by
the brief itself, and guessing either one would produce a scientifically
mislabelled pipeline.

**Q1 — Reference-capacity convention (brief §2 says STOP if undefined).**
Upstream mixes nominal capacity and first-cycle capacity per cell. One
convention must be fixed and recorded in config, provenance and result
metadata. The options:

- **(A) Nominal capacity only** — textbook SOH; cells without a nominal
  capacity are excluded and reported, not silently switched.
- **(B) First valid measured discharge capacity** — applies to every dataset
  uniformly, self-consistent, but is a "relative to this cell's own start"
  health, not an absolute one.
- **(C) Mean of first N valid cycles** (the upstream `TODO`'s suggestion) —
  as (B) but less sensitive to a single noisy first cycle.
- **(D) Configurable with no default**, the config must name one, and mixing
  within a dataset is a hard error.

**Q2 — Cycle-level samples require changing upstream core (`CLAUDE.md` §4
says stop and ask).** The canonical task in the brief (§1, §4) is per-cycle;
upstream is strictly one sample per cell. Options:

- **(A) Extend the upstream contracts** — let `process_cell` return an
  `(n_samples, ...)` block plus a sample-identity record, concatenate instead
  of stack in `feature/base.py`, `label/base.py`, `task.py`, and carry
  `cell_id`/`cycle_index` into `DataBundle`. This is the design that makes
  §4, §5, §11 and §12 all work, and would be recorded in
  `docs/UPSTREAM_PATCHES.md`. It is the largest change.
- **(B) Keep one sample per cell** at a configured cycle, as upstream does.
  No core change, but it is not the task the brief defines, and "SOH vs
  cycle" (§12C) would show one point per cell.
- **(C) A `batteryml_web`-side sample builder** that expands cells into cycle
  rows and calls the model layer directly. Avoids touching core, but builds
  the parallel execution system the brief and `CLAUDE.md` both warn against.

My recommendation is **Q1 → (D) with (B) as the config default**, since it is
the only convention every dataset can satisfy uniformly and it makes the mixing
impossible rather than merely discouraged; and **Q2 → (A)**, because it is the
only option that delivers the task as specified without a second pipeline.
Both need your agreement before I write the code.
