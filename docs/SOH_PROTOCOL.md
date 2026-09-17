# SOH estimation — what it means here

This is the protocol for the SOH route in this repository: what the task is,
what the label is a ratio of, what the model may see, and what the route does
not claim.

## SOH is not RUL, lifetime, or a trajectory forecast

| | Question it answers |
| --- | --- |
| **SOH estimation** (this document) | *How healthy is this cell **right now**, at the cycle I am observing?* |
| RUL prediction | *How many cycles are left before end of life?* |
| Battery lifetime prediction | *How long will this cell last in total?* |
| Degradation trajectory forecasting | *What will the capacity be over the next N cycles?* |

They are separate tasks with separate labels, and the platform keeps them
separate. In particular, `eol_soh` in an RUL config is the **end-of-life
threshold** — the SOH value at which a cell counts as dead — and a config that
carries it is an RUL config. Task classification reads the label annotator, not
substrings; see `batteryml_web/core/config_registry.py::_infer_task` and
`tests/test_soh_task_inference.py`.

## The canonical task

> Estimate a cell's state of health at an **observed cycle `t`**, using only
> information available at or before `t`.

One sample is one `(cell, cycle t)` pair. A cell contributes many samples.

This is not next-cycle capacity forecasting. The placeholder docstring in
`batteryml/models/soh_predictors/__init__.py` that called SOH "the discharge
capacity of the next cycle" describes a different task and is historical.
Neither is `soh_example.ipynb` the protocol: it labels one cycle per cell,
predicts cycle 300 from cycles ≤ 99 (a forecast), trains `DummyRULPredictor`,
and its second half fits random labels. It is a demonstration of label
plumbing.

## The label

```
SOH(cell, t) = Q_discharge(cell, t) / Q_ref(cell)
```

`Q_discharge(cell, t)` is the maximum of that cycle's
`discharge_capacity_in_Ah` trace — BatteryML's own per-cycle convention.

**The canonical value is a ratio**, e.g. `0.973`. A UI may render it as
`97.3%`; the stored value, the metric and the artifact stay on the ratio scale.

### Reference capacity — one convention per run

`Q_ref` is fixed for the whole run by `reference_capacity_mode` and applies to
every cell. There is no per-cell fallback and no automatic switching.

| Mode | `Q_ref` | Note |
| --- | --- | --- |
| `first_valid_discharge_capacity` *(default)* | the cell's first cycle whose discharge capacity is valid | Every dataset can satisfy it. Health relative to the cell's own start. |
| `nominal_capacity` | `BatteryData.nominal_capacity_in_Ah` | Absolute. A cell without one is **excluded**, never switched to another mode. |
| `first_n_valid_mean` | mean over the first `reference_n_cycles` valid cycles | Less sensitive to one noisy first cycle. |

This replaces the previous behaviour, which used nominal capacity when a cell
happened to carry one and the first cycle otherwise — labelling two cells in
one dataset under two different definitions of SOH with nothing recording
which. See `docs/UPSTREAM_PATCHES.md` UP-05.

The mode travels with the run: it is in the config, in the run provenance and
in the prediction artifact's `task_metadata`.

### What is excluded, and why you will know

Nothing is interpolated, substituted or clipped.

| Situation | Outcome |
| --- | --- |
| No valid reference capacity | the whole cell is excluded, counted as `no_valid_reference_capacity` |
| A cycle with missing / zero / negative discharge capacity | that cycle is dropped, counted as `invalid_discharge_capacity` |
| A cycle with no readable voltage trace | that sample is dropped by `Task.build` |
| SOH above 1.0 early in life | **kept** — it is a real measurement |

## What the model may see

`SOHVoltageCurveFeatureExtractor` (`batteryml/feature/soh_cycle.py`) is the one
canonical feature path.

For the sample at cycle `t`, per cycle in the history window ending at `t`:

* the cycle's `voltage_in_V` trace resampled to `n_voltage_points` (default 32);
* five summary scalars: `voltage_mean`, `voltage_min`, `voltage_max`,
  `current_abs_mean`, `duration_in_s`.

Row width is fixed and reported by `feature_dim`; the default is 37.

**Capacity is deliberately not a feature.** The label is
`Q_discharge(t) / Q_ref`, so feeding `Q_discharge(t)` in would make the target
an arithmetic function of the input — and would answer a question nobody asks,
because anyone who has run a full discharge to measure the capacity already
knows the SOH. The point is to read health off the shape of the measured curve.

**Causality.** The row for cycle `t` reads cycles `≤ t` and nothing after.
`tests/test_soh_pipeline.py` proves this by mutation: every cycle after `t` is
replaced with wildly different data and the row must be bit-identical. A
companion test mutates cycle `t` itself and requires the row to change, so the
first test cannot pass vacuously.

## Cells are split before they are expanded into cycles

This is the hard invariant.

```
cells  ->  cell-level train/test split  ->  per-split cycle expansion
```

Never: expand every cell into cycle rows, then split rows. That would put one
cell's early cycles in train and its late cycles in test, and the score would
measure memorisation of that cell while looking entirely normal.

`Task.build` calls `assert_disjoint_cells` on the loaded cells before any
expansion, and names every offending cell if it ever fails.
`tests/test_soh_core.py::TestCellLevelSplitInvariant` builds the leakage-prone
alternative on purpose and proves it is rejected.

## Sample policy

Configuration, not science. Defaults are a starting point, not a claim.

| Field | Meaning |
| --- | --- |
| `min_cycle` | skip a formation period |
| `max_cycle` | cap the observation window; `null` means to the end of the cell |
| `cycle_stride` | subsample cycles |

## Metrics

**MAE** and **RMSE**, on the ratio scale.

MAPE is not inherited from the RUL pipeline. A UI may annotate `MAE 0.021` as
`= 2.1 percentage points`; that is display only and the stored metric is
unchanged.

## Models

Any registered BatteryML regressor whose contract is task-agnostic.
`SklearnModel` was audited for this: `fit` is
`model.fit(feature.view(n, -1), label)` and `predict` returns a flat tensor,
with no lifetime, end-of-life or cycle semantics anywhere. `Ridge`, `XGBoost`
and their siblings qualify — the `RUL` in `XGBoostRULPredictor` is naming only.

The UI shows the generic display name (`XGBoost`, `Ridge Regression`), never
"XGBoost RUL", for an SOH run.

### The compatible baselines

Five estimators have a config under `configs/soh/`. They are **compatible SOH
baselines**, not a ranking: no controlled comparison has been run, and nothing
here says which of them estimates SOH better.

| Model | Class | Config | Why it qualifies |
| --- | --- | --- | --- |
| Ridge Regression | `RidgeRULPredictor` | `configs/soh/ridge/matr.yaml` | L2-penalized linear. The first route, run on the server. |
| Lasso Regression | `LassoRULPredictor` | `configs/soh/lasso/matr.yaml` | Pure L1; selects a subset of the 37 dimensions. Extension model. |
| Elastic Net Regression | `ElasticNetRULPredictor` | `configs/soh/elastic_net/matr.yaml` | L1+L2; stabilises the choice among correlated neighbouring voltage points. |
| Random Forest | `RandomForestRULPredictor` | `configs/soh/rf/matr.yaml` | First non-linear estimator: says whether the linear error is the feature's limit or the model's. |
| XGBoost | `XGBoostRULPredictor` | `configs/soh/xgb/matr.yaml` | Boosted trees, built against the running residual. |

Every one of them reaches the data through `SklearnModel`, which is the whole
reason a cycle-level SOH bundle can be handed to them unchanged.

**Each config is byte-identical to the Ridge config outside its `model:`
block** — same split, same reference convention, same sample policy, same
feature extractor, same transformations. That is what makes two SOH numbers
comparable at all, and `tests/test_soh_baseline_pack.py` compares the configs
field by field rather than re-asserting the expected values one at a time.

None of the four added configs declares a hyperparameter: library defaults
only, matching every corresponding config in `configs/baselines/`. A value
chosen by looking at a score would make the pack a badly-run comparison
instead of a compatibility statement.

`random_state` is deliberately not pinned in any of them. The seed reaches
sklearn through `batteryml.pipeline.set_seed`, called immediately before
`fit`; pinning `random_state` in the config would fix the estimator's
randomness regardless of `--seed` and make the recorded seed untrue.

### Reproducibility, measured

`[VERIFIED]` in a development container, on a synthetic bundle of the same
37-wide cycle-level shape — **not** on MATR:

| Model | Two runs at seed 0 | Seed 0 vs seed 1 | Fit time at route scale |
| --- | --- | --- | --- |
| Ridge | identical | identical (no stochastic component) | < 1 s |
| Lasso | identical | identical (`selection='cyclic'`, unshuffled KFold) | < 1 s |
| Elastic Net | identical | identical (same reason) | < 1 s |
| Random Forest | identical | **different** | ~18 s |
| XGBoost | identical | identical | < 1 s |

Random Forest is the only entry whose result the seed actually moves, which is
exactly why its `random_state` must stay unpinned. XGBoost at its defaults has
no stochastic component at all — `subsample` and `colsample_bytree` are 1.0 and
the histogram method is deterministic — so its result is seed-invariant. That
is a property to record, per `CLAUDE.md` §8, not one to paper over.

### Audited and rejected

| Model | Why not |
| --- | --- |
| MLP (`MLPRULPredictor`) | An `NNModel`, not a `SklearnModel`. Its constructor requires `input_height` and `input_width`, and `forward` unpacks `B, _, H, W = feature.size()`. A flat 37-wide SOH row has no such geometry, and `fit` raises `ValueError: not enough values to unpack (expected 4, got 2)`. Reshaping the SOH feature to suit the model would be changing the science to fit the model. |

Deep sequence models (CNN, LSTM, Transformer, BatLiNet, CPTransformer) are out
of scope here for the same class of reason and were not audited.

## Which datasets are eligible

**Do not assume any.** Eligibility is an empirical question about the processed
files, and it is answered by opening them:

```bash
python -m scripts.audit_soh_readiness
python -m scripts.audit_soh_readiness --min-cycle 10 --cycle-stride 5 \
    --json runs/verify/soh_readiness.json
```

A dataset is `SOH_READY` only when enough cells each yield enough cycle-level
samples through the same annotator and extractor a run would use. Being ready
for RUL says nothing about being ready for SOH.

### What the server audit found

Run on the real processed data. These are the user's observed numbers, not
this repository's claim:

| Dataset | Eligible cells | SOH samples (default) | SOH samples (min_cycle 10, stride 5) | Status |
| --- | --- | --- | --- | --- |
| MATR | 180 / 180 | 142,035 | 28,156 | `SOH_READY` |
| HUST | 77 / 77 | 146,120 | 29,119 | `SOH_READY` |

`reference_capacity_mode: first_valid_discharge_capacity`, no exclusions
reported for either dataset.

**MATR is the first route**, for cell-level generalisation: 180 independent
cells against HUST's 77. HUST is the intended second dataset; no HUST config
exists yet.

## Before training: audit the labels

Eligibility says a dataset *can* produce labels. This says what they are:

```bash
python -m scripts.audit_soh_labels --dataset MATR --min-cycle 10 --cycle-stride 5
python -m scripts.audit_soh_labels --dataset MATR --min-cycle 10 --cycle-stride 5 \
    --per-cell --json runs/verify/soh_labels_matr.json
```

It reports the SOH distribution (min, p1, p5, median, p95, p99, max), counts of
NaN / infinite / non-positive labels, how many exceed 1.0, 1.05 and 1.10, and
the reference-capacity range. It exits non-zero when the distribution looks
structurally wrong.

**SOH is never clipped to [0, 1].** Under a first-cycle reference, a ratio
above 1.0 early in life is a real measurement — capacity activation in the
first cycles is a known effect. The audit surfaces a suspicious distribution;
it never tidies one.

Stop and report rather than training if the audit shows a large NaN or `<= 0`
count, a median far from ~1.0, a max far above ~1.2, or a p1 near 0.

## The first real config

`configs/soh/ridge/matr.yaml` — MATR, SOH estimation, Ridge regression,
`min_cycle: 10`, `cycle_stride: 5`,
`reference_capacity_mode: first_valid_discharge_capacity`.

### The split

`MATRPrimaryTestTrainTestSplitter`: the canonical MATR-1 partition every MATR
RUL baseline in `configs/baselines/` uses, so an SOH result here concerns the
same cells those results concern.

| | Cells |
| --- | --- |
| train | 41 |
| test | 42 |
| **covered** | **83 of the 180 processed MATR cells** (batches 1-2; upstream removes the b2c1 outlier) |

There is **no validation split**: BatteryML's splitter contract is
`(train, test)`, and inventing a third split would mean a splitter this
repository would then have to justify. To use 178 of the 180 cells instead,
change the splitter to `MATRCLOTestTrainTestSplitter` (108 train / 70 test) —
one line, and it is also cell-disjoint.

The cell partition is checked, not assumed: `Task.build` calls
`assert_disjoint_cells` on the loaded cells *before* any expansion into
cycles.

### Why Ridge

`sklearn.linear_model.Ridge` through `RidgeRULPredictor`, whose `RUL` is
naming only. Fast, deterministic, and task-agnostic, which is what a first
end-to-end proof needs. It is not a claim that Ridge is a good SOH estimator.

## The server smoke

```bash
conda run -n batteryml python batteryml_web/core/run_batteryml_cli.py run \
    configs/soh/ridge/matr.yaml \
    --workspace runs/soh_smoke_matr_ridge \
    --train --eval \
    --device cpu --seed 0 \
    --metric MAE,RMSE
```

`--metric` is a real CLI flag (`bin/batteryml.py`, default `RMSE,MAE,MAPE`).
It is passed explicitly because MAPE is not a canonical SOH metric: it is
informative for a cycle count of order 100 and close to meaningless for a
ratio of order 1, where it merely restates MAE. The Agent's execution plan
passes the same list for an SOH task.

Ridge on roughly 28k x 37 is cheap; this is a smoke, not a benchmark run.

### The other four baselines

Same command, one word changed in two places. Run them one at a time and read
each artifact before starting the next: four runs that all failed the same way
are four times the output and none of the information.

```bash
conda run -n batteryml python batteryml_web/core/run_batteryml_cli.py run \
    configs/soh/lasso/matr.yaml \
    --workspace runs/soh_smoke_matr_lasso \
    --train --eval \
    --device cpu --seed 0 \
    --metric MAE,RMSE

conda run -n batteryml python batteryml_web/core/run_batteryml_cli.py run \
    configs/soh/elastic_net/matr.yaml \
    --workspace runs/soh_smoke_matr_elastic_net \
    --train --eval \
    --device cpu --seed 0 \
    --metric MAE,RMSE

conda run -n batteryml python batteryml_web/core/run_batteryml_cli.py run \
    configs/soh/rf/matr.yaml \
    --workspace runs/soh_smoke_matr_rf \
    --train --eval \
    --device cpu --seed 0 \
    --metric MAE,RMSE

conda run -n batteryml python batteryml_web/core/run_batteryml_cli.py run \
    configs/soh/xgb/matr.yaml \
    --workspace runs/soh_smoke_matr_xgb \
    --train --eval \
    --device cpu --seed 0 \
    --metric MAE,RMSE
```

`--device cpu` for all four. None of these estimators uses a GPU: the three
sklearn ones cannot, and XGBoost is not built here for it.

What to expect, and what would mean something is wrong:

* The reading stages should report the **same** `41` train and `42` test cells
  as the Ridge run, and the artifact the **same 6116 test predictions across
  42 cells**. A different count means a config diverged from the protocol —
  stop and diff it against `configs/soh/ridge/matr.yaml`, rather than
  reporting the metric.
* `Random Forest` is the slow one, of order tens of seconds to fit against
  well under a second for the rest. It is the only one for which a progress
  stage sits still for a noticeable time.
* `Lasso` needs the CLI wrapper, as every invocation does anyway: it is an
  extension model, and `bin/batteryml.py` on its own would raise
  `KeyError: LassoRULPredictor is not registered to Models!`.
* Metrics are MAE and RMSE on the ratio scale. Do not compare them to the
  Ridge numbers as a ranking — one run each, one seed, no repeats.

## After the run: inspect the artifact

```bash
python -m scripts.inspect_soh_artifact runs/soh_smoke_matr_ridge \
    --config configs/soh/ridge/matr.yaml --rows 10
```

Reports the task and reference convention the artifact records, the prediction
and unique-test-cell counts, observed and predicted SOH ranges, the recorded
MAE and RMSE, a cross-check that the inverse transform reproduces them, and
the first rows as `cell_id, cycle, observed, predicted, residual`.

**Workflow validation, not a quality judgement.** One Ridge run says the route
works end to end. It says nothing about how well SOH can be estimated.

## The prediction artifact

`predictions_seed_<seed>_<timestamp>.pkl` carries, in addition to the existing
`prediction`, `scores`, `data` and `seed`:

| Key | Contents |
| --- | --- |
| `sample_index` | one `{cell_id, cycle_index}` per prediction, in row order |
| `task_metadata` | `label_annotator`, `reference_capacity_mode`, the sample policy, `feature_extractor` |

Without `sample_index` a cycle-level prediction could not be traced to a cell
at all, because the row number identifies nothing once a cell contributes many
rows.

## Reading the result in the web app

An SOH artifact holds one scalar per *cycle sample*. Structurally that is
indistinguishable from an RUL artifact's one scalar per cell, so the view
layer decides which it is from what the artifact records about itself and
never from the numbers (`batteryml_web/agent/result_views.py`,
`detect_schema`). It reads two independent things — the **task**, from the
`task_metadata` the evaluation step writes, and the **granularity**, from the
`sample_index` — and never lets one stand in for the other. An SOH result
becomes the SOH view, whose rows name their cell and cycle and whose axes say
*SOH (ratio)*. See `docs/RESULT_SCHEMAS.md` for the full table.

Three things are drawn:

* **Observed vs estimated SOH**, with the `y = x` reference line.
* **SOH across cycles**, for the first few test cells, observed and estimated
  together — a parity scatter answers "is it calibrated", only a per-cell
  curve answers "does it track degradation".
* **The estimate range check**, below.

### The range check is a reading, not a correction

SOH is a ratio: a physically meaningful estimate lies above 0 and at or below
1.0 (slightly above 1.0 is ordinary early-life behaviour, so 1.1 is where the
report calls a value implausible). A linear model fitted on a 37-dimensional
voltage-curve feature can and does produce estimates well outside that range,
and an average error hides them completely.

So the app **counts them and says so**: how many fall below 0, above 1.0 and
above 1.1, the observed minimum and maximum, the fraction outside the range,
and the ten largest absolute residuals with the cell and cycle each belongs
to.

Nothing is clipped, rescaled, dropped or reordered. The arrays plotted and
counted are the arrays the canonical MAE and RMSE were computed from, value
for value — which is the point: a diagnostic that quietly repaired the
predictions would make the figures agree with a metric that no longer
described them.

## Limitations

* **This is a route, not a benchmark.** One feature path, one label
  convention, and models chosen for being task-agnostic rather than for SOH
  performance. No claim is made about accuracy, and none should be read from
  the numbers a first run produces.
* Eligibility, sample counts and any metric depend on the processed data on
  the machine that runs it. Nothing here has been run against a real dataset;
  the synthetic fixtures under `tests/fixtures/` exist only to test the code
  and can never reach a result.
* Only `first_valid_discharge_capacity` has been exercised end to end. The
  other two modes are implemented and unit-tested but have not been used for a
  full run.
* The feature path uses voltage, current and time. Temperature and internal
  resistance are present in `CycleData` for some datasets and are not used.
