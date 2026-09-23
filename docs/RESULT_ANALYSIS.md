# Result analysis — what the assistant may say about a run

The result explanation used to be a narrator. Its prompt carried the
dataset, task, model, metrics and a file list, and closed by asking the
model to "state which model/dataset/task ran … restate the metrics above
exactly … list the generated artifacts". It did that accurately and
answered none of the questions a battery scientist actually has.

This document is the rule for what replaced it. Implemented in
`batteryml_web/core/benchmarks.py` (the published values and the gate),
`batteryml_web/agent/result_analysis.py` (the deterministic context and the
prompt) and `batteryml_web/agent/result_agent.py` (the call).

## The deterministic context

The assistant reasons from facts the backend already computes, never from
its own recollection:

| Group | Source |
| --- | --- |
| identity (project, run id, dataset, task, model, config, seed, device) | the selected result and its manifest |
| metrics, **with units** | the run's own saved scores |
| prediction diagnostics (count, finite/non-finite, distinct, constant, min/max, cells) | `prediction_diagnostics.artifact_diagnostics` on the **selected** artifact |
| protocol (split class, cell data, feature extractor, label annotator) | the config the run executed |
| smoke status | `--epochs 1` in the executed argv |
| benchmark reference | `core/benchmarks.py`, gated |

Units are part of the fact. "RMSE 0.081" means nothing until it says
whether that is cycles or a capacity ratio — the two differ by three orders
of magnitude, and confusing them is the specific error the benchmark gate
exists to prevent.

**One artifact.** Diagnostics come from the selected artifact and no other.
Reaching for a different file to enrich a prompt is the substitution that
made Review Results describe the wrong run.

## The published benchmark

### Source, and what was verified

The table is transcribed from this repository's own `README.md`, section
*"Benchmark result of RUL(Remain Useful Life) task"*.
`tests/test_benchmark_registry.py` re-parses that README on every test run
and fails if a single cell drifts.

The README names the task and the seed protocol but **not the metric**.
`result.ipynb` — the notebook in this repository that generated the table —
does: every cell comes from `extract_scores(...)['RMSE']`. Its dataset keys
are exactly `['clo','cruh','crush','snl','matr_1','matr_2','hust','mix']`,
the config basenames, which is what ties the MATR1/MATR2 columns to
`matr_1.yaml` and `matr_2.yaml` rather than to a guess. `baseline.ipynb`
corroborates that independently: its dummy regressor scores
`test_RMSE 398.82` and `test2_RMSE 510.63`, matching the README's MATR1
`398` and MATR2 `510`.

The unit is cycles: RUL labels are cycle counts (`RULLabelAnnotator`), and
`docs/VISUALIZATION.md` records that the stored RMSE is a number of cycles.
Lower is better.

### What could not be verified

* **The paper itself.** arxiv.org and openreview.net are both blocked by
  this environment's egress proxy and no PDF is vendored, so nothing here
  rests on the paper's own text.
* **The meaning of `>1000`.** Evidently a censoring threshold for a
  diverged run, but nothing local says whether it means strictly greater
  than 1000 or a failure to converge. It is therefore stored with **no
  central value at all** — never as the number 1000 — and a censored cell
  can never produce a difference. A run scoring 900 must not look like an
  improvement on a cell the paper declined to score.
* **The seed count per row.** `result.ipynb` groups `rf` with the
  single-seed sklearn baselines and formats that group without a
  dispersion, yet the README's Random forest row reads `168±9`. The saved
  notebook therefore does not reproduce that row and cannot be the
  authority on seeds. The README's own sentence is used instead — "the
  error mean across ten seeds … standard deviation as subscript" — so a
  cell with a dispersion is a ten-seed mean and a cell without one is a
  single run.

### MATR1 vs MATR2

Read from the run's **split class**, never from the dataset name and never
from the model:

| splitter | subset |
| --- | --- |
| `MATRPrimaryTestTrainTestSplitter` | MATR1 |
| `MATRSecondaryTestTrainTestSplitter` | MATR2 |
| `MATRCLOTestTrainTestSplitter` | CLO |
| `HUSTTrainTestSplitter` | HUST |
| `SNLTrainTestSplitter` | SNL |
| `CRUHTrainTestSplitter` | CRUH |
| `CRUSHTrainTestSplitter` | CRUSH |
| `MIX100TrainTestSplitter` | MIX |

`MATRTrainTestSplitter`, the generic parent, is deliberately absent. A run
using it is on neither published subset, and defaulting to MATR1 because it
is listed first would attach the run to a protocol it did not use.

### Model names

Mapped explicitly, on the platform's own `model_key`, never by string
similarity. `cptransformer`, `batlinet`, `PBT` and `BatteryMFormer` have no
row: CPTransformer is not the paper's "Transformer" and BatLiNet is not its
"MLP", however similar the names read. The platform has both a
`transformer` key and a `cptransformer` key, so a fuzzy match would have
found one.

## The comparability gate

A difference is computed only when **all** of these hold: the task is RUL
prediction, the split class identifies a published subset, the model has a
published row, the metric is RMSE in cycles, the run is not a smoke run,
the reference is not censored, and the run has a saved value. Every failing
condition is reported, not just the first.

A reference may still be *shown* when arithmetic is refused — a smoke run
on a published protocol is the case that motivates it. **Except on a task
mismatch**, where no record is supplied at all: a ridge/MATR1 SOH run
resolves the same model and the same split as the published ridge/MATR1 RUL
row, so a lookup that ignored the task would put "116 cycles" on screen
beside an SOH RMSE of 0.081. Labelling that pair "not comparable" is not
enough; showing them together is itself the error.

## What the four cases do

Produced by the shipped code, with metrics stood in:

| case | reference supplied | comparable | difference |
| --- | --- | --- | --- |
| SOH Ridge on MATR | none | no | — |
| Ridge on `matr_1`, trained | 116 cycles | **yes** | +14.4 cycles |
| CNN on `matr_1`, 1 epoch | 102 ± 94 cycles | no (smoke) | — |
| CPTransformer on `matr_1` | none | no (no published baseline) | — |

## Real-server acceptance

Run on the server, from the repository root, with the LLM Assistant
configured in Settings. Each case is: run it, open Review Results, select
the run, read the **Published BatteryML reference** panel, then press
**Explain metrics**.

The panel is deterministic and is drawn whether or not an explanation is
requested, so the first four checks need no API call.

### CASE 1 — SOH Ridge on MATR

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/soh/ridge/matr.yaml \
  --workspace runs/web_projects/<project>/soh_ridge_matr \
  --train --eval --device cpu --seed 0
```

Expect: the panel says **no published reference**, naming the task as the
reason. **No cycle count appears anywhere on the page.** The explanation
interprets MAE/RMSE as SOH ratios, comments on the prediction spread, and
proposes next steps. It must not mention 116, or any other RUL number.

### CASE 2 — a RUL run on a published protocol

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/ridge/matr_1.yaml \
  --workspace runs/web_projects/<project>/ridge_matr1 \
  --train --eval --device cpu --seed 0
```

Expect: the panel shows **Ridge / MATR1 / RMSE / 116 cycles**, the run's own
RMSE, and a signed difference. The explanation discusses the gap. Confirm
the subset reads MATR1 and not MATR2 — and run `matr_2.yaml` too, which
must show **184** instead.

### CASE 3 — 1-epoch CNN smoke

Use the Agent's smoke option, or `--epochs 1`. Expect: the panel shows the
CNN/MATR1 reference `102 ± 94` **with no difference**, and the reason
"1-epoch smoke run". The explanation must call it workflow validation, must
not rank it, and must not state a gap.

### CASE 4 — an external model

Run CPTransformer on MATR. Expect: **no reference**, reason "no directly
corresponding published BatteryML baseline". It must not be compared with
the paper's Transformer row.

### Inspecting what the backend decided

```bash
python - <<'PY'
from batteryml_web.agent.result_analysis import read_protocol
from batteryml_web.core import benchmarks as bm
protocol = read_protocol("configs/baselines/sklearn/ridge/matr_1.yaml")
print(protocol["splitter"], "->", bm.subset_for_splitter(protocol["splitter"]))
print(bm.lookup("ridge", "MATR1"))
PY
```
