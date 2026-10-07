# Result schemas — which figures a saved artifact earns

Every prediction artifact this platform writes is, structurally, a list of
numbers. What those numbers *mean* decides which figures are honest to draw
from them, and the meaning is not in the numbers. This document is the rule
the view layer follows, and `batteryml_web/agent/result_views.py` is where it
is implemented.

The claim a figure makes is the point. A "remaining useful life vs cycle"
curve asserts that the run estimated remaining life for one cell at several
observed cycles. Most BatteryML lifetime models never do that — they read a
cell's early cycles and produce one number for it — and drawing a curve
through one point per cell, or through several cells' points joined together,
would be a picture of something nobody measured.

## The two questions, answered separately

`detect_schema` reads two independent things and never lets one stand in for
the other.

**Which task**, from `task_metadata` — written beside the predictions by
`Pipeline.evaluate`. Its `task` field when the config declared one, and
otherwise the **label annotator's class name**, which is the component the run
actually trained against.

> A model's name is never consulted. `RidgeRULPredictor` trains the SOH route
> in `configs/soh/ridge/matr.yaml`, and `XGBoostRULPredictor` the XGBoost one:
> the `RUL` in those class names is naming only.

**At what granularity**, from `sample_index` — the per-row `(cell_id,
cycle_index)` record. A row that is about a *cell* rather than a cycle carries
`cycle_index = -1` (`SampleIndex.WHOLE_CELL_CYCLE`), which is what
`SampleIndex.for_whole_cells` writes for every whole-cell run.

A result is **cycle-level** only when all of the following hold:

1. a well-formed `sample_index` exists, of the same length as the predictions;
2. **no** row carries the `-1` whole-cell sentinel;
3. at least one cell id appears more than once;
4. some cell has at least **two distinct** cycle indices.

## The categories

| Task | Cycle-level | Schema | Views |
| --- | --- | --- | --- |
| RUL | no | `rul_scalar` | parity, residual, table |
| RUL | yes | `rul_cyclewise` | parity, residual, table, **RUL vs cycle** |
| SOH | yes | `soh_cycle` | SOH parity, SOH residual, SOH vs cycle, range check |
| SOH | no | `soh_scalar` | parity, residual, table |
| unknown | no | `rul_scalar` | parity, residual, table |
| unknown | yes | `soh_cycle` | as above |
| — | — | `trajectory` | trajectory views |
| — | — | `unknown` | structural / file preview only |

A sequence per row is a trajectory whatever the recorded task says, and is
never re-read as a series of per-cycle estimates.

The two "unknown" rows preserve the contract that predates `task_metadata`, so
artifacts already on disk keep reading as they did. Note what they cannot
produce: an unidentified task never yields `rul_cyclewise`. **Cycle-wise RUL
curves are shown only for artifacts that contain genuine per-cell, per-cycle
RUL estimates.**

## Four ways to get this wrong

Each of these is a real classifier someone might write, each is refused, and
each has a fixture in `tests/fixtures/synthetic_result_artifacts.py` that
would pass it.

| Tempting rule | Why it is wrong |
| --- | --- |
| `if rows > cells: cyclewise` | A whole-cell run writes one index record per cell; and rows can repeat a cell with no cycle identity at all (`repeated_rows_without_cycle_identity_payload`). |
| `if sample_index exists: cyclewise` | **Every** run writes one now. A RUL artifact's records all carry `cycle_index = -1`. |
| `if model_name contains "RUL": rul` | `RidgeRULPredictor` trains the SOH route. |
| `if cycles are present: rul_cyclewise` | An SOH artifact is structurally identical — two cells, several real cycles each, one scalar per row. Only the recorded task separates them. |

One cycle per cell is also not a curve: `single_cycle_per_cell_payload` has
real cycle indices and is still `rul_scalar`, because one point per cell shows
no trend and joining two cells' single points would draw a line between
different batteries.

## The cycle-wise RUL data contract

`build_rul_cycle_view(payload)` returns:

```python
{"available": bool,
 "reason": str,              # why not, when unavailable
 "cells": [
     {"cell_id": str,
      "cycles":    [int, ...],    # ascending
      "observed":  [float, ...],  # true remaining life, as stored
      "predicted": [float, ...]}, # estimated remaining life, as stored
     ...
 ],
 "count": int, "n_cells": int,
 "inverse_transformed": bool, "note": str}
```

Three properties hold by construction:

* **Per-cell entries, not a flat series.** There is no top-level `cycles` or
  `observed` array, so joining two cells into one line would take deliberate
  effort rather than being the default.
* **Sorted by cycle within a cell**, carrying observed and predicted with
  them. Storage order is the order the extractor produced, and a line drawn in
  that order would zig-zag.
* **Saved values only.** `true_rul = final_lifetime - cycle` is never
  reconstructed, from anything. An artifact with no observed labels draws
  nothing and says so.

## What the production models emit today

Audited, not assumed — and asserted in
`tests/test_rul_cycle_views.py::TestTheProductionRulArtifactAudit`, so this
section fails a test rather than going quietly stale.

| | Finding |
| --- | --- |
| RUL configs | 170, across 23 models |
| Label annotator used by all of them | `RULLabelAnnotator` |
| Is it cycle-level? | No — it subclasses `BaseLabelAnnotator`, not `BaseCycleLabelAnnotator`, so `Task.is_cycle_level` is False and the run takes the `SampleIndex.for_whole_cells` branch |
| The only registered cycle-level annotator | `CycleSOHLabelAnnotator` |

So: **every** current production RUL model — CPTransformer, BatLiNet, the
sklearn family, the neural models — emits one scalar per cell.

> **Cycle-wise RUL visualization is implemented, but no current production RUL
> artifact emits the required per-cell, per-cycle RUL schema.**

The capability is therefore dormant against real artifacts and exercised only
by the synthetic fixtures. Nothing about RUL model training was changed to
make a curve appear, and nothing should be. When a cycle-level RUL annotator
is added, the audit test above fails and says so — that is the signal the
feature can come out of dormancy.

### What a `rul_scalar` run shows instead

Not a warning: nothing is wrong with such a run, and there is nothing for the
reader to fix. The curve section collapses to one caption:

> This run produces one RUL/lifetime estimate per cell, so a cycle-wise RUL
> curve is not available.

## Fixtures

`tests/fixtures/synthetic_result_artifacts.py`, per `CLAUDE.md` §7: built in
memory, `synthetic_` prefix on the module and every cell id, never written
under `runs/` or `data/`, and incapable of reaching `metrics.csv` or a report.
The numbers mean nothing; they exist so the arrays have a length.

| Builder | Expected schema |
| --- | --- |
| `scalar_rul_payload` | `rul_scalar`, no curve |
| `cyclewise_rul_payload` | `rul_cyclewise`, 2 cell groups, cycles sorted |
| `repeated_rows_without_cycle_identity_payload` | not `rul_cyclewise` |
| `single_cycle_per_cell_payload` | not `rul_cyclewise` |
| `soh_cyclewise_payload` | `soh_cycle`, never RUL |
| `soh_scalar_payload` | `soh_scalar` |
| `trajectory_payload` | `trajectory`, never `rul_cyclewise` |
| `trajectory_payload_with_cycle_index` | `trajectory` — the adversarial case, where every non-shape signal points at cycle-wise RUL |

## Rows that cannot be plotted, and what a value is

Two rules added with the website figure rework (`docs/VISUALIZATION.md`):

* A single NaN or inf no longer makes a result "unknown". The schema is
  decided from the row shape; the views then drop rows whose observed **or**
  predicted value is non-finite through one shared mask, report how many, and
  filter cell and cycle identity with the same mask.
* Every view carries `target`: the quantity and unit read from the recorded
  label annotator — "Battery lifetime (cycles)" for `RULLabelAnnotator`,
  "SOH (ratio)" for the SOH annotators, "Remaining useful life (cycles)" only
  for `rul_cyclewise`. Nothing is rescaled for display.

