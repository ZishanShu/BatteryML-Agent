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

## Which datasets are eligible

**Do not assume any.** Eligibility is an empirical question about the processed
files, and it is answered by opening them:

```bash
python -m scripts.audit_soh_readiness
python -m scripts.audit_soh_readiness --min-cycle 10 --cycle-stride 5 \
    --json runs/verify/soh_readiness.json
```

It prints one row per dataset:

```
Dataset | Cells | Eligible cells | SOH samples | Status
```

A dataset is `SOH_READY` only when enough cells each yield enough cycle-level
samples through the same annotator and extractor a run would use. Being ready
for RUL says nothing about being ready for SOH.

## The config

Written **after** the audit names an eligible dataset, one config for one
dataset and one model:

```yaml
# configs/soh/<model>/<dataset>.yaml
task: 'SOH estimation'
model:
    name: 'RidgeRULPredictor'          # task-agnostic; displayed as "Ridge Regression"
train_test_split:
    name: '<DATASET>TrainTestSplitter'  # a cell-level splitter
    cell_data_path: 'data/processed/<DATASET>'
feature:
    name: 'SOHVoltageCurveFeatureExtractor'
    n_voltage_points: 32
    n_history_cycles: 1
label:
    name: 'CycleSOHLabelAnnotator'
    reference_capacity_mode: 'first_valid_discharge_capacity'
    min_cycle: 10
    max_cycle: null
    cycle_stride: 1
feature_transformation:
    name: 'ZScoreDataTransformation'
label_transformation:
    name: 'ZScoreDataTransformation'
```

Note the label transformation is a plain Z-score, **not** the
`LogScale + ZScore` sequence the RUL configs use: a log of a ratio near 1.0
compresses exactly the range SOH lives in, and is undefined at 0.

`configs/soh/dummy/*` are marked `LEGACY EXAMPLE` in-file. They use
`DummyRULPredictor` and one label per cell; they are not this task.

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
