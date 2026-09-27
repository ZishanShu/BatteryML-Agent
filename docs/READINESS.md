# Backend readiness layer (`batteryml_web/core/readiness.py`)

One module that answers, from real files and live registries only:

* Is this **dataset** usable for a run right now?
* Is this **model** runnable on this dataset right now?

It is the state source for the current validators, the current backend and the
future LLM agent. The agent must read this state instead of guessing from
filenames, and it must never manufacture readiness that this layer does not
report.

## Dataset states

| Status | Meaning | Current examples (2026-09, server filesystem) |
| --- | --- | --- |
| `READY` | `data/processed/<DATASET>/` holds `*.pkl` files | MATR (180), HUST (77) |
| `RAW_ONLY` | raw data present, no usable processed output | CALCE, RWTH, SDU, Tongji, XJTU |
| `INCOMPLETE_PROCESSED` | processed directory exists but holds no `*.pkl` | HNEI, OX, SNL, UL_PUR |
| `MISSING` | nothing usable locally (declared datasets report this too) | MICH, MICH_EXP, Stanford_2, ISU_ILCC |

`RAW_ONLY` and `MISSING` are **states, not platform failures**. The empty
processed directories flagged under `INCOMPLETE_PROCESSED` are dangerous
rather than merely unfinished: `BaseTrainTestSplitter` only asserts existence
(`batteryml/train_test_split/base.py:25`), so an empty directory yields an
empty cell list instead of failing loudly.

```python
from batteryml_web.core.readiness import assess_dataset, list_dataset_readiness

assess_dataset("CALCE").to_dict()
# {'dataset': 'CALCE', 'status': 'RAW_ONLY', 'raw_available': True,
#  'processed_available': False, 'processed_sample_count': 0, ...,
#  'recommended_action': 'Run preprocessing before model execution.'}
```

`list_dataset_readiness()` reports the union of the declared platform datasets
(`PLATFORM_DATASETS`) and everything found under `data/raw` / `data/processed`,
so nothing on disk is silently left out; `_`-prefixed internal directories are
ignored.

Readiness is **not cached**. It is recomputed from the filesystem on every
call, so a dataset downloaded while the app is running shows its new status on
the next page read — no restart. (The two `lru_cache`s in this package cover
the YAML config catalog and the BatteryML model registries, neither of which
depends on whether data is on disk.)

### Recognised raw layouts

A dataset's raw data is looked for in three places, best first. A layout
counts only when its **signature** holds; a directory existing is never
enough.

| Layout | Path | Signature | Preprocessable by BatteryML |
| --- | --- | --- | --- |
| native | `data/raw/<NAME>/` | directory holding ≥1 regular file | only if BatteryML registers a preprocessor for `<NAME>` |
| BatteryLife, extracted | `data/raw/BatteryLife/v11/<NAME>/` | directory holding ≥1 regular file | no |
| BatteryLife, archive | `data/raw/BatteryLife/v11/<NAME>.zip` | `zipfile.is_zipfile` is true | no |

The signatures are deliberately narrow in one direction and deliberately
loose in the other:

* An **empty** directory stays `MISSING` — that is the state a failed
  download leaves behind, and calling it raw data would send someone to
  preprocess nothing. A tree of empty subdirectories does not qualify either,
  because the count is over regular files.
* A file wearing the directory's name, and a `.zip` that is not a readable
  archive, both stay `MISSING`. A truncated download says so in the reason,
  which is a different finding from "nothing was ever downloaded".
* An extracted directory is **not** matched against an expected file list.
  `third_party/BatteryLife` is absent from this checkout, so the contents of
  a v11 dataset cannot be proven here; requiring a guessed filename would
  trade one misclassification for a stricter one. Non-empty is what can be
  established, and it separates downloaded from not-downloaded.

The set of preprocessable datasets is read from
`batteryml.preprocess.SUPPORTED_SOURCES["DATASETS"]`, not listed in the web
layer, so it cannot drift from what the CLI can actually preprocess.

### Detected is not runnable

`RAW_ONLY` says raw data was found. It does **not** say a run can start, and
for a BatteryLife layout it does not even say preprocessing is possible:
none of `ISU_ILCC`, `MICH`, `MICH_EXP`, `SDU`, `Stanford_2`, `Tongji`,
`XJTU` has a BatteryML preprocessor. Those report `RAW_ONLY` with
`preprocessing_supported: False`, and nothing offers a preprocess command
for them — the Prepare Data panel says the source is not wired to the
BatteryML preprocessing route instead.

The same distinction applies to a native path: `data/raw/XJTU/` is the native
shape, but there is no XJTU preprocessor, so it is detected and still not
preprocessable. Dataset presence and route runnability are two questions and
stay two fields.

### Why downloaded data used to report MISSING

The alternate-raw branch this replaced recognised **one** shape — a
`.zip` at `data/raw/BatteryLife/v11/<NAME>.zip` — for a hardcoded **three**
datasets (`MICH`, `MICH_EXP`, `Stanford_2`), and only when the caller passed
no `raw_root`. So:

* a dataset that had been downloaded **and extracted** — the normal end
  state, and the state the external-model readiness path already requires of
  its `root_path` — reported `MISSING`;
* so did one whose archive had been deleted after extraction;
* so did `ISU_ILCC`, `SDU`, `Tongji` and `XJTU` however much was on disk,
  because they were not on the list;
* and any caller naming its own raw directory skipped BatteryLife detection
  altogether.

To see what the platform sees on a machine, read-only:

```bash
python scripts/inspect_dataset_readiness.py            # every declared dataset
python scripts/inspect_dataset_readiness.py MATR XJTU  # just these
python scripts/inspect_dataset_readiness.py --json     # machine-readable
```

It prints each raw candidate path, whether its signature holds, the processed
directory and the resulting status. It reads only — nothing is downloaded,
extracted, copied, renamed or preprocessed.

## Model states

| Status | Meaning |
| --- | --- |
| `READY` | registered (or its external runner is wired) and at least one validated config's data is present |
| `NEEDS_ASSETS` | source exists; a pretrained checkpoint / PCA / LLM asset is missing |
| `NEEDS_DATA` | the required dataset or a config for the requested dataset is missing |
| `DEPENDENCY_UNAVAILABLE` | the runtime (conda env, batteryml import) is not usable |
| `UNREGISTERED` | no config resolves to the model, so the pipeline cannot build it |
| `CONFIG_INVALID` | a config exists but fails the deterministic validator for non-data reasons |

Two backends, one report shape (`ModelReadiness.to_dict()`):

* **Registry models** (`assess_registry_model`) — everything running through
  the BatteryML pipeline (XGBoost ... CPTransformer, BatLiNet). READY requires
  that a config passes `config_validator.load_and_validate_config_file` with
  data required — the same gate a run goes through — and that the config's
  model class is in the live `MODELS` registry, populated via
  `register_extension_models()` exactly as `run_batteryml_cli.py` does.
  A model with no config for the requested dataset is `NEEDS_DATA` for that
  dataset, never READY by borrowing another dataset's config.
* **External models** (`assess_external_model`) — PBT and BatteryMFormer,
  declared in `EXTERNAL_MODELS` with their fixed task semantics:
  PBT → `RUL prediction` (scalar lifetime), BatteryMFormer →
  `Degradation trajectory forecasting` (never a scalar RUL model, never
  through `RULLabelAnnotator`). Readiness is filesystem inspection of
  `third_party/` plus, for PBT, a cheap conda environment check
  (`probe_environment=True` additionally probes `from models import PBT`
  inside pbt311 through `external_runners.pbt.PBTRunner`). Nothing in
  `third_party/` is modified or executed by a readiness check.

Current verified results:

| Model | Status | Blocking inputs |
| --- | --- | --- |
| CPTransformer | `READY` | — |
| BatLiNet | `READY` | — |
| PBT | `NEEDS_ASSETS` | checkpoint archive not unpacked, PCA, LLM checkpoint, PBT-format dataset (pbt311 env itself is available) |
| BatteryMFormer | `NEEDS_DATA` | BatteryLife v11 unpacked data, `processed_SOH` (Qwen3 embeddings and split JSONs are present) |

An external model whose status is not READY is **correct**, not a failure to
be worked around by substituting another model, another dataset, or placeholder
assets (CLAUDE.md §7).

## Agent-facing context

```python
from batteryml_web.core.readiness import get_workflow_context

ctx = get_workflow_context(dataset="MATR", task="RUL prediction")
# {
#   "dataset": {...},                  # DatasetReadiness for MATR
#   "task": "RUL prediction",
#   "available_models": [...],         # READY models for this dataset/task
#   "blocked_models": [...],           # the rest, each with `missing` + action
#   "recommended_next_steps": [...],   # ordered, actionable
# }
```

Task labels fold case-insensitively: `RUL` / `RUL prediction`, `SOH` /
`SOH estimation`, `trajectory` / `Degradation trajectory forecasting`.
For the trajectory task only external models are listed: the registry models
are scalar RUL/SOH predictors by construction.

The five agent responsibilities (workflow guidance, readiness explanation,
config suggestion, troubleshooting, result interpretation) all consume this
state. What the agent may *not* do: generate executable Python, write an
unvalidated config, decide that a missing file exists, or produce a number
that BatteryML did not compute.

## Tests

`tests/test_readiness.py` (stdlib `unittest`, no pytest needed, no GPU, no
downloads; registry-backed cases skip themselves when `batteryml` cannot be
imported):

```bash
python -m unittest tests.test_readiness -v
```

Covers: the four dataset states on temporary paths, READY CPTransformer and
BatLiNet against the live registries (the §11 regression check), the MATR
external configs against the execution gate, PBT/BatteryMFormer status on the
real tree plus synthetic READY trees, and the workflow context for MATR/RUL,
trajectory, and RAW_ONLY datasets.

## A config belongs to the data it loads

A config's dataset is read from `train_test_split.cell_data_path`, the path
`Task.build` actually loads. It is not read from the file's text.

It used to be: `_infer_dataset` searched `f"{path} {path.name} {text}"` for
each dataset name in turn, `MATR` first. Every CNN config names
`VoltageCapacityMatrixFeatureExtractor`, and *Matrix* contains **matr**, so
all eight CNN configs registered as MATR — including
`configs/baselines/nn_models/cnn/hust.yaml`, which loads
`data/processed/HUST`, and `snl.yaml`, which loads `data/processed/SNL`.

Two observed consequences:

* **MATR + CNN offered eight configs, five of which read another dataset's
  directory.** A plan built from one of those is labelled MATR and runs
  HUST; it is refused by the data-requiring validator, or dies on
  `AssertionError: data/processed/HUST`. Whether a CNN run worked depended
  on which config was selected, which is the "CNN sometimes cannot run"
  report.
* **HUST and SNL offered no CNN at all**, although each has a config written
  for it.

This is the same defect class as the `eol_soh` → SOH task mis-inference, and
the same repair: read structure, never substrings.

A config whose `cell_data_path` is a **list** spans several datasets
(`mix`, `cruh`, `crush`). It is not any one dataset's config and is no longer
offered under one — returning MATR for `mix.yaml` because MATR appears among
its six directories is exactly the mislabelling this rule stops. Those files
stay on disk and remain runnable from the CLI. The catalog went from 173
registered entries to 112; every entry removed was one of these, previously
mislabelled as MATR.

