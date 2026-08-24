# BatteryML Protocol Platform — Static Audit

**Method.** Read-only inspection of source. **No code in this repository was executed for
this audit**: there is no `data/` directory in the working tree, no trained models, and no
BatteryML runtime was invoked. Per `CLAUDE.md` §0, every claim below is tagged:

- `[STATIC]` — established by reading the code, with `file:line`.
- `[ASSUMED]` — believed but unchecked; the check that would settle it is named.

There are **no `[VERIFIED]` claims in this document.** The five CLI facts in `CLAUDE.md` §5
are treated as ground truth supplied by the user, not re-derived here.

Audit date: 2026-08-24. Commit at audit time: `89c3621` on branch `claude/new-session-3c9hqv`.

---

## Executive summary (read this first)

Two independent, statically provable defects make the web platform's main workflow
unreachable, and one silent-corruption bug sits in the shared feature path:

1. **`get_config_registry` is used but never imported** — `batteryml_web/app.py:838` and
   `:1075`. Guaranteed `NameError`. This kills both the *Prepare Data* and *Run Model*
   pages. `[STATIC]`
2. **`ConfigEntry` has no `.model` attribute** but `app.py:842` and `:1075` read
   `entry.model`. Guaranteed `AttributeError` even if (1) were fixed. `[STATIC]`
3. **`smooth()` discards its own result** — `batteryml/feature/severson.py:43-56` computes
   outlier replacement into `res` and then `return meds`. Every feature extractor in the
   repository is affected. This does not crash; it silently changes the numbers. `[STATIC]`

Additionally, four modules are unimportable due to names that do not exist
(§F.4), no artifact from `CLAUDE.md` §8 is produced under its specified name except
`training_log.txt`, `config.yaml` and `project_state.json` (§G), and the dataset-readiness
signal shown in the UI reflects **raw** data presence while the run requires **processed**
data (§H.4).

---

## A. Repository map

### A.1 Tree to depth 3

```
.
├── batteryml/                     UPSTREAM. The BatteryML package (Microsoft, MIT).
│   ├── data/                      BatteryData/CycleData containers + DataBundle + metrics
│   │   └── transformation/        ZScore / LogScale / Sequential feature+label transforms
│   ├── feature/                   Feature extractors (Severson family + QV matrix)
│   ├── label/                     RUL and SOH label annotators
│   ├── models/                    Model base classes
│   │   ├── rul_predictors/        14 registered RUL models
│   │   └── soh_predictors/        EMPTY — no SOH model exists
│   ├── preprocess/                10 dataset/cycler preprocessors + downloader
│   ├── train_test_split/          8 registered splitters
│   ├── utils/                     Registry + YAML config loader
│   └── visualization/             plot_helper.py (not referenced by the web layer)
├── batteryml_web/                 OURS. Streamlit platform.
│   ├── agent/                     LLM client + prompts + 4 agent stubs
│   ├── core/                      Runner, registries, validators, readers (see A.3)
│   ├── docs/                      schema_reference.md (field spec, prose only)
│   ├── pages/                     EMPTY (.gitkeep only)
│   ├── templates/configs/         EMPTY (.gitkeep only)
│   └── ui/                        Sidebar, progress, LLM state; 4 empty placeholder files
├── bin/batteryml.py               UPSTREAM. argparse CLI: download / preprocess / run
├── configs/                       UPSTREAM. 114 YAML configs (112 RUL + 2 SOH) + 3 cycler
├── scripts/reproduce_batteryml/   OURS. 4 standalone reproduction scripts
├── tools/                         OURS. sync_data_and_config_manifests.py
├── image/                         README assets
└── *.ipynb                        UPSTREAM notebooks (baseline, result, soh_example)
```
`[STATIC]` — from `find`, and from the MIT/Microsoft copyright headers present on every
file under `batteryml/`, `bin/` and `configs/` and absent from every file under
`batteryml_web/`, `scripts/` and `tools/`.

### A.2 Upstream vs. ours

| Boundary | Evidence |
| --- | --- |
| Upstream: `batteryml/`, `bin/`, `configs/`, notebooks | `# Copyright (c) Microsoft Corporation.` header on essentially every module; `setup.py:12` names Microsoft and `url=https://github.com/microsoft/BatteryML` |
| Ours: `batteryml_web/`, `scripts/reproduce_batteryml/`, `tools/`, `CLAUDE.md`, `README_BATTERYML_REPRODUCE.md` | no upstream header; all use `from __future__ import annotations` + PEP-604 typing, a style absent upstream |

**`CLAUDE.md` §4 ("do not modify upstream BatteryML core") appears to be respected**: I found
no `batteryml_web` import inside `batteryml/`, and no `docs/UPSTREAM_PATCHES.md` exists.
`[STATIC]` — grep over `batteryml/` for `batteryml_web` returns nothing.

### A.3 Entry points

| Entry point | File | Notes |
| --- | --- | --- |
| `batteryml` console script | `setup.py:22` → `bin.batteryml:main` | Installing the package puts `bin/` on the path; per `CLAUDE.md` §5 this shadows the real package |
| CLI wrapper (the sanctioned path) | `batteryml_web/core/run_batteryml_cli.py:1-22` | inserts repo root at `sys.path[0]` (`:11`) then `runpy.run_path` on `bin/batteryml.py` (`:14-20`) |
| Streamlit app | `batteryml_web/app.py` (1480 lines) | launched as `streamlit run batteryml_web/app.py`; inserts `REPO_ROOT` and `APP_DIR` into `sys.path` at `:44-47` |
| Reproduction scripts | `scripts/reproduce_batteryml/{prepare_data,manual_train_eval,explicit_matr_xgb,check_dataset_status}.py` | bypass the CLI entirely, import `batteryml.pipeline` directly |
| Manifest builder | `tools/sync_data_and_config_manifests.py` | writes `data/metadata/{dataset_manifest,model_config_manifest}.json` |

### A.4 Python version and dependencies

- **No Python version is pinned anywhere.** `setup.py:29` declares only
  `"Programming Language :: Python :: 3"`; there is no `python_requires`, no
  `pyproject.toml`, no `.python-version`, no CI matrix. `[STATIC]`
- The codebase uses PEP-604 unions (`Path | None`) at runtime in dataclass fields, e.g.
  `batteryml_web/core/config_registry.py:52-59`, and `Path.unlink(missing_ok=True)`
  (`batteryml_web/core/config_validator.py:60`), so **Python ≥ 3.8 with
  `from __future__ import annotations` is required and 3.10+ is the safe floor**. `[ASSUMED]`
  — settled by `python -c "import sys; print(sys.version)"` on the server.
- `requirements.txt` (14 lines) pins only `numpy>=1.24,<2.0.0`. Everything else —
  `pandas, addict, fire, tqdm, scipy, scikit-learn, pyyaml, numba, matplotlib, h5py,
  openpyxl, requests, xgboost` — is **unpinned**. `[STATIC]`
- **Three imported packages are absent from `requirements.txt`:** `[STATIC]`
  - `torch` — imported at `batteryml/pipeline.py:7`, `batteryml/task.py:4`,
    `batteryml/data/databundle.py:4`, and `batteryml_web/app.py:13`. Without it,
    *nothing* runs.
  - `streamlit` — imported at `batteryml_web/app.py:12`, `ui/progress_utils.py:9`,
    `ui/llm_state.py:5`, `ui/workflow_sidebar.py:3`.
  - `xgboost` is present, but `sklearn` is declared as `scikit-learn` (correct).
- `requirements.txt` has **no trailing newline** after `xgboost` (the file ends
  `xgboost` immediately followed by EOF), which is cosmetic but was visible when
  concatenating. `[STATIC]`

---

## B. Dataset integration matrix

Full table in `docs/INTEGRATION_MATRIX.md`. Summary of what reading the source establishes:

- **8 dataset preprocessors are registered**: CALCE, HNEI, HUST, MATR, OX, RWTH, SNL,
  UL_PUR (`batteryml/preprocess/__init__.py:3-10`, each class carrying
  `@PREPROCESSORS.register()`).
- **2 cycler preprocessors are registered**: ARBIN, NEWARE
  (`preprocess_arbin.py:17`, `preprocess_neware.py:18`).
- **`SUPPORTED_SOURCES` advertises 6 cyclers that have no implementation.**
  `batteryml/preprocess/__init__.py:20` lists `BATTERYARCHIVE, BIOLOGIC, INDIGO, LANDT,
  MACCOR, NOVONIX` as valid `argparse` choices (`bin/batteryml.py:34-36` builds the choice
  list from this dict), but no `*Preprocessor` class of those names exists. The failure is
  `KeyError: BIOLOGICPreprocessor is not registered to Preprocessors!` from
  `batteryml/utils/registry.py:59`, raised only after the CLI has already accepted the
  argument and created the output directory (`bin/batteryml.py:100-101`). `[STATIC]`
  This matters because `batteryml_web/core/custom_uploads.py:13` offers exactly
  `["ARBIN", "BIOLOGIC", "LANDT", "MACCOR", "NEWARE"]` in the custom-upload UI — **three of
  the five choices cannot work.** `[STATIC]`
- **`DOWNLOAD_LINKS` covers only 4 of the 8 datasets**: MATR, HUST, CALCE, RWTH
  (`batteryml/preprocess/download.py:12-48`). HNEI, SNL, UL_PUR and OX have preprocessors
  but no download entry, so `batteryml download SNL` is rejected by argparse
  (`bin/batteryml.py:26-28` derives choices from `DOWNLOAD_LINKS.keys()`). Our own
  `scripts/reproduce_batteryml/check_dataset_status.py:35-39` documents that SNL/HNEI/UL_PUR
  come from Battery Archive and must be fetched manually — consistent with the gap. `[STATIC]`
- **OX is orphaned.** `OXPreprocessor` is complete (`preprocess_OX.py:16-45`) but no
  download link, no config, no splitter and no UI path references OX. Its `cell_id` also
  omits the dataset prefix used by every other preprocessor (`preprocess_OX.py:71` passes
  `cell_id=name` where `name` is the bare file stem, versus `CALCE_{cell}` at
  `preprocess_CALCE.py:35`). `[STATIC]`
- **`download_file` disables TLS verification** — `requests.get(url, stream=True,
  verify=False)` at `batteryml/preprocess/download.py:72`. `[STATIC]`

---

## C. Model / method integration matrix

Full table in `docs/INTEGRATION_MATRIX.md`. What reading establishes:

- **14 models are registered**, all under `batteryml/models/rul_predictors/`, all imported
  (hence registered) via `rul_predictors/__init__.py:4-17`.
- **`batteryml/models/soh_predictors/` contains only a docstring** — 4 lines, no class, no
  import (`soh_predictors/__init__.py`). **There is no SOH model in this repository.**
  `[STATIC]`
- **The two SOH configs use a RUL model.** `configs/soh/dummy/matr_1.yaml` and
  `matr_1_with_soh_filepath.yaml` both set `model: name: 'DummyRULPredictor'`. SOH support
  is therefore label-annotator-only: `SOHLabelAnnotator` exists
  (`batteryml/label/soh.py:15`) and is registered (`:14`), but the only "SOH model" is a
  `sklearn.dummy.DummyRegressor` (`rul_predictors/dummy.py:13`). `[STATIC]`
- **Two registered models are wired into no config at all**: `SVMRULPredictor`
  (`rul_predictors/svm.py:11`) and `ElasticNetRULPredictor` (`rul_predictors/elastic_net.py:11`).
  Grepping every YAML under `configs/` for those class names returns nothing. `[STATIC]`
- **BatLiNet is absent.** The only occurrences of the string are in
  `batteryml_web/app.py.bak*` and `batteryml_web/app_old.py:1531`, which themselves say
  *"Experimental, not yet implemented"*. No `BatLiNet` class, config or registry entry
  exists in live code. `[STATIC]`
- **No config references a class that does not exist.** I resolved the `model`,
  `train_test_split`, `feature`, `label`, `feature_transformation` and
  `label_transformation` names in all 114 configs against the `@*.register()` decorators;
  every name resolves. `[STATIC]`
- **`GaussianProcessRULPredictor` silently discards its hyperparameters.**
  `rul_predictors/gaussian_process.py:13-16` accepts `*args, **kwargs` and then constructs
  `GaussianProcessRegressor(kernel)` with a hardcoded `DotProduct() + RBF()`, passing
  neither. Anything a user writes under `model:` in a `gpr/*.yaml` is ignored without a
  warning. `RidgeRULPredictor` (`ridge.py:14`) likewise accepts `*args` and drops it.
  `[STATIC]`
- **`TransformerRULPredictor.forward` prints the feature shape on every forward pass** —
  `rul_predictors/transformer.py:35`. This floods `training_log.txt` and is scraped by the
  metrics regex (§F.6). `[STATIC]`

---

## D. Feature pipeline

### D.1 What exists

Four registered extractors, in two families:

| Extractor | File:line | Output |
| --- | --- | --- |
| `VarianceModelFeatureExtractor` | `feature/variance_model.py:12` | 1 scalar: `['Variance']` |
| `DischargeModelFeatureExtractor` | `feature/discharge_model.py:12` | 6 scalars |
| `FullModelFeatureExtractor` | `feature/full_model.py:12` | 9 scalars |
| `VoltageCapacityMatrixFeatureExtractor` | `feature/voltage_capacity_matrix.py:15` | `(n_cycles_kept, 1000)` matrix |

The first three subclass `SeversonFeatureExtractor` (`feature/severson.py:68`) and call
`get_features(cell, [names])` (`severson.py:79`). `[STATIC]`

### D.2 Exact dimensionality and cycle window

**Severson family** (`severson.py:82-107`): with `critical_cycles` defaulting to `[1, 9, 99]`
(`severson.py:74`) and the SOH configs overriding it to `[2, 9, 99]`:

- `early_cycle = cycle_data[critical_cycles[1]]`, `late_cycle = cycle_data[critical_cycles[2]]`
  (`severson.py:83-84`) — i.e. **ΔQ is computed between 0-indexed cycles 9 and 99**, so
  1-indexed cycles **10 and 100**. This matches Severson et al.'s "cycle 100 minus cycle 10".
- `diff_qdlin = late_qdlin - early_qdlin` (`severson.py:91`), smoothed (`:92-93`), NaNs
  dropped (`:95`), and `ValueError('Qdlin is all nan!')` raised if ≤1 point survives (`:96-97`).
- Output length = `len(feature_lists)` → **1, 6 or 9 scalars per cell**.

**Voltage-capacity matrix** (`voltage_capacity_matrix.py:43-80`), which is what
`configs/baselines/sklearn/xgb/matr_1.yaml` uses:

- `diff_base: 8`, `max_cycle_index: 98`, `cycles_to_keep: 98`, `use_precalculated_qdlin: True`.
- `cycles_to_keep` is an **int**, converted to `[98]` at `voltage_capacity_matrix.py:37-38`,
  so exactly one cycle survives the filter at `:60-62`.
- The feature is therefore **`Qdlin(cycle 98) − Qdlin(cycle 8)`, both 0-indexed** — i.e.
  1-indexed cycles **99 and 9**, a one-cycle offset from the Severson family's 100−10 and
  from the "QV 100-10" description in the advisor feedback. **Flagging this explicitly**:
  the two feature families in this repository difference *different* cycle pairs. `[STATIC]`
- Shape is `torch.stack(feature)` over the kept cycles (`:79`) → **`(1, 1000)`**, flattened
  to 1000 columns by `SklearnModel.fit` (`models/sklearn_model.py:18`).

**Voltage-axis discretization** (`severson.py:16-27`):
`_get_Qdlin` masks to discharge (`I < -0.1 A`, `severson.py:24-25`), then
`interpolate(V, Q, 1000, xs=min_V, xe=max_V)` builds
`np.linspace(min_voltage_limit_in_V, max_voltage_limit_in_V, 1000)` (`:20`) and reverses it
(`:26`). So **the grid is 1000 points, linear in voltage, spanning each cell's own declared
voltage limits** — it is *not* a shared grid across datasets. Cells from different datasets
with different `min/max_voltage_limit_in_V` therefore have non-comparable feature axes; the
`mix.yaml` and `cruh.yaml` configs mix exactly such cells. `[STATIC]` This is a scientific
point worth raising in the spec, not a crash.

### D.3 Two dead configuration parameters

- `VoltageCapacityMatrixFeatureExtractor.interp_dim` is stored at
  `voltage_capacity_matrix.py:25` and **never read** — `process_cell` (`:43-80`) never
  references `self.interp_dim`.
- `SeversonFeatureExtractor.interp_dims` is stored at `severson.py:76` and **never read**.
- Both are dead because `interpolate(...)` is called with a **hardcoded literal `1000`** at
  `severson.py:24-25`. The SOH configs set `interp_dims: 1000`, which happens to coincide,
  so the setting is silently inert rather than silently wrong today — but any user who
  changes it gets no effect and no warning. `[STATIC]`

### D.4 Separability from training

**Feature extraction is fused into dataset construction, not separable.**
`Task.build()` (`batteryml/task.py:44-73`) loads cells, extracts features and annotates
labels in one method; `Pipeline.train` calls `build_dataset` (`pipeline.py:57`), which calls
`Task.build()` (`pipeline.py:222`). There is **no feature-only entry point and no feature
artifact on disk** — the only persistence is an opaque whole-dataset pickle at
`cache/battery_cache_<sha256[:32]>.pkl` (`pipeline.py:205-207, 231-233`). `[STATIC]`

### D.5 Does any artifact record the feature set used?

**No.** `feature_summary.json` is never written — grep for the string across the repository
returns only `CLAUDE.md`. The cache filename hashes the feature config
(`pipeline.py:198-204` over `CONFIG_FIELDS[1:]`) but is not human-readable and is not copied
into any run directory. The config *is* copied to the workspace as
`config_{timestamp}.yaml` (`pipeline.py:65-68`), which is the only surviving record of which
feature configuration produced a run. `[STATIC]`

### D.6 The `smooth()` defect

```
batteryml/feature/severson.py:42-56
@njit
def smooth(x, window_size=10, sigma=3):
    res = np.empty_like(x)
    meds = np.empty_like(x)
    for i in range(len(x)):            # rolling median
        ...
        meds[i] = np.median(x[low: high])
    base = np.std(np.abs(x - meds))
    for i in range(len(x)):            # conditional outlier replacement -> res
        if np.abs(meds[i] - x[i]) > base * sigma:
            res[i] = meds[i]
        else:
            res[i] = x[i]
    return meds                        # <-- res is computed and thrown away
```

`res` is fully computed and never returned; `meds` — the raw rolling median — is returned
instead. The `sigma` parameter has **no effect on the return value**. Consequences: the
signal is median-filtered with a 21-sample window everywhere instead of having only
>3σ outliers replaced, which flattens genuine structure in Qdlin. `smooth` is called at
`severson.py:92` (Severson family) and `voltage_capacity_matrix.py:48, 71, 75` (QV matrix),
so **every feature in the repository passes through it**. `[STATIC]`

This is the single most dangerous line in the audit: it does not raise, and the numbers it
produces are plausible. **Whether the published BatteryML benchmark numbers were produced
with this same bug is unknown to me** — if they were, "fixing" it would desynchronize this
repository from the upstream reference values. `[ASSUMED]` — settled only by comparing a
MATR/XGBoost run before and after a one-line change, which is a spec decision, not a fix to
make now.

---

## E. Task coverage

See `docs/INTEGRATION_MATRIX.md` for the grid. Counts established by parsing all 114 configs:

- **RUL: 112 configs** = 14 model directories × 8 split presets
  (`matr_1, matr_2, clo, hust, snl, cruh, crush, mix`). The grid is **completely full for
  the 14 model directories that have configs**; there are 0 gaps within it.
- **The gaps are at the model axis, not the dataset axis**: `SVMRULPredictor` and
  `ElasticNetRULPredictor` have 0 configs → **2 × 8 = 16 empty cells**.
- **SOH: 2 configs**, both MATR, both `DummyRULPredictor`. Against the same 8 split
  presets and 14 models that would be **112 cells, of which 2 are filled and 110 empty** —
  and the 2 that are filled use a dummy regressor. `[STATIC]`
- Six datasets (CALCE, RWTH, UL_PUR, HNEI, SNL, HUST) are reachable **only** through the
  composite presets `cruh` / `crush` / `mix` / `hust` / `snl`; there is **no
  single-dataset config for CALCE, RWTH, UL_PUR or HNEI**. `[STATIC]`

---

## F. Web platform state

### F.1 Page inventory

Navigation is a 4-entry list, `NAV_STEPS` in `ui/workflow_sidebar.py:5-36`:
`home → prepare_data → run_model → review_results`. **There is no standalone LLM page and
no standalone troubleshooting page in `app.py`** — advisor items 6 and 7 appear already
addressed structurally. (`app_old.py` and the `.bak*` files still contain them; those files
are dead.) `[STATIC]`

| Page | Function | Real | Placeholder / broken |
| --- | --- | --- | --- |
| Home | `app.py:342-503` | Preflight checklist over `data/raw/MATR`, `data/processed/MATR`, config path (`:421-427`); "Run MATR Demo" builds a correct `run … --train --eval --device cpu --seed 0` command (`:465-484`) | "Test connection" (`:412-413`) reports "connected" from `is_llm_configured()`, which only checks that three text boxes are non-empty (`ui/llm_state.py:31-37`) — **no network call is made** |
| Prepare Data | `app.py:604-836` | Preprocess button issues a correct `preprocess <DS> <raw> <processed>` command (`:698`) | **Dead**: `info["raw_path"]` (`:684`), `info["processed_path"]` (`:685,715`), `info["processed_files"]` (`:718`) are keys `_dataset_card` never returns (`:538-556`) → `KeyError`; and `:838` → `NameError` (§F.3) |
| Run Model | `app.py:1043-1258` | Command string at `:1176` is correct and complete | **Dead**: `_discover_run_configs()` at `:1046` → `:838` `NameError`; `entry.model` at `:842` and `:1075` → `AttributeError`; `info['raw_files']` at `:1060` → `KeyError` |
| Review Results | `app.py:1260-1435` | 5 tabs; lists searched paths on every empty state — good practice | Predictions tab cannot ever plot (§F.6); falls back to a hardcoded run directory (§H.5) |

The four files `ui/components.py`, `ui/landing.py`, `ui/styles.py`, `ui/workspace.py` are
**0 bytes**, as are `pages/.gitkeep` and `templates/configs/.gitkeep`. `[STATIC]`

### F.2 Quick-start trace, launch → report

Reading `render_home_page` (`app.py:342-503`) end to end:

1. `streamlit run batteryml_web/app.py` → module import → `sys.path` gets `REPO_ROOT` and
   `APP_DIR` (`:44-47`) → `import torch` (`:13`). **Break point 1**: `torch` is not in
   `requirements.txt` (§A.4).
2. Preflight table (`:421-427`) checks three paths exist. **Break point 2**: it checks
   `data/processed/MATR` *exists*, not that it is non-empty. An empty directory passes.
   `[STATIC]`
3. `preflight_ok` (`:440`) gates the Run button (`:462`).
4. `run_command_with_progress` (`ui/progress_utils.py:31-110`) spawns the subprocess and
   streams `stdout` line by line into `log_box` (`:69-72`) — **this genuinely streams**.
5. `process.wait()` (`:74`), then the remaining progress steps are replayed with
   `time.sleep(0.03)` (`:79-81`). **Break point 3**: the stages "Training/evaluation
   running", "Collect outputs" are displayed *after* the process has already exited. The
   progress bar is theatre, not status. `[STATIC]`
6. Log written to `<project>/logs/matr_demo.log` (`:100-103`), session state set, jump to
   `review_results` (`app.py:495-497`).
7. **Break point 4**: the Home quick-start **never calls `_collect_run_outputs`** (compare
   `app.py:1202`, where the Run Model page does). So the quick-start produces **no
   `runs/web_projects/<name>/` artifacts at all** beyond the log — no `config.yaml`, no
   `project_state.json` update. `[STATIC]`
8. Review Results reads metrics by regex-scraping the log (§F.6).

**No `report.html` is produced on this path, or on any live path** (§G).

### F.3 The two fatal name errors

```
app.py:838   registry = get_config_registry(str(REPO_ROOT))
app.py:1075  available_models = sorted({entry.model for entry in get_config_registry(...)})
```

`get_config_registry` is defined at `batteryml_web/core/config_registry.py:178`. Grepping
`app.py` for `config_registry` returns **only lines 838 and 1075** — there is no import
statement, module-level or function-local. Guaranteed `NameError: name
'get_config_registry' is not defined`. `[STATIC]`

Even with the import added, `ConfigEntry` (`config_registry.py:51-63`) is a frozen dataclass
whose fields are `dataset, task, model_key, model_display, split_preset, split_label,
config_path, metadata`. **There is no `model` field**, so `entry.model` at `app.py:842` and
`:1075` raises `AttributeError`. `[STATIC]`

### F.4 Four modules that cannot be imported

| Module | Bad import | Why it fails |
| --- | --- | --- |
| `core/batteryml_runner.py:9` | `from .results_reader import ensure_metrics_csv_from_log` | `results_reader.py` defines no such name (its 8 defs are at `:28,43,76,91,108,136,148,160`) |
| `core/report_builder.py:8` | `from .results_reader import list_figures, read_log, read_metrics, read_predictions` | none of the four exist in `results_reader.py` |
| `agent/schema_agent.py:6` | `from .llm_client import LLMClient` | `llm_client.py` defines only `run_llm_prompt` (`:16`), `test_llm_connection` (`:49`), `save_llm_output` (`:64`) — **no `LLMClient` class** |
| `agent/result_agent.py:6` | `from .llm_client import LLMClient` | same |

All four are `ImportError` at module load. `[STATIC]` They are currently invisible because
**nothing imports them** — `app.py` imports only `custom_uploads` (`:16`), `results_reader`
(`:26`), `agent.llm_client` (`:15`) and the three `ui` modules (`:27-29`).

Two more modules use a **top-level `core` package name that only resolves if
`batteryml_web/` itself is on `sys.path`**: `agent/prompts.py:5`
(`from core.schema_validator import ...`) and `agent/troubleshoot_agent.py:7-8`. `app.py:44-47`
happens to insert `APP_DIR`, so they would resolve *if* imported after `app.py` — but
`import batteryml_web.agent.prompts` from anywhere else fails. `[STATIC]`

Net effect: of 13 modules in `batteryml_web/core/`, **`app.py` uses 2**. `catalog.py` (455
lines — a more complete, `Enum`-typed registry with `TaskType`, `Status`, device inference)
and `config_registry.py` (179 lines) are **two parallel implementations of the same
concept**, and the app reaches for the broken one. `[STATIC]`

### F.5 Where the LLM is invoked, and whether a validator gates it

Live call sites, all in `app.py`, all going through `run_llm_prompt`
(`agent/llm_client.py:16-46`):

| Site | Line | What the agent may do | Validator? |
| --- | --- | --- | --- |
| Explain this config | `1108-1114` | free text | none — output goes straight to a display panel and to `agent_outputs/config_explanation.md` |
| Suggest model/config | `1136-1143` | free text | **none** |
| Explain this command | `1180` | free text | none |
| Explain last error | `1232` | free text | none |
| Suggest next action | `1245` | free text | none |
| Summarize results | `1310-1316` | free text | none |
| Explain metrics | `1338-1344` | free text | none |

**Assessment against `CLAUDE.md` §3.** The boundary is currently held by *architecture, not
by validation*: no call site parses the model's reply, and no call site writes a config or
touches data — every response is passed to `_set_llm_output` (`app.py:908-913`) which only
displays it and optionally saves a `.md`. So the agent cannot today reach execution.
But the required `suggest → user review → validator → execute` chain **does not exist**:
there is no diff, no confirmation step, and no validator on the agent path. The three
modules that *would* have provided structured, schema-checked output — `schema_agent`,
`result_agent`, `config_agent` — are respectively unimportable, unimportable, and a
literal stub returning `{"config": {}, "source": "placeholder"}`
(`agent/config_agent.py:6-7`). `[STATIC]`

Three call sites (`1180`, `1232`, `1245`) call `run_llm_prompt(prompt)` **without**
`base_url`/`api_key`/`model`, which are keyword-only and have no defaults
(`llm_client.py:16-23`) → `TypeError`. `[STATIC]`

**API key handling** (`CLAUDE.md` §3 / prompt 10 item 5): the key lives in
`st.session_state["llm_api_key"]` (`ui/llm_state.py:26, 60`), is sent as a Bearer header
(`llm_client.py:32`), and is **not** written by `save_llm_output`, which records only model
and base URL (`llm_client.py:68-70`). No `.env`, no on-disk persistence. That part is clean.
`[STATIC]`

**No interaction log exists.** Nothing writes prompt + retrieved context + raw response +
validation result + user decision to the run directory; only the final assistant text is
saved to `agent_outputs/*.md`. `[STATIC]`

### F.6 Are run outputs collected into a stable structure?

`_collect_run_outputs` (`app.py:295-317`) is the only collector, and it is called from
exactly one place (`app.py:1202`, the Run Model page — which is dead per §F.3).

It copies subdirectories `metrics/, predictions/, figures/, checkpoints/, reports/` from the
workspace (`app.py:310-318`). **BatteryML writes none of those directories.** Reading the
writers:

- `pipeline.py:120-124` writes `predictions_seed_{seed}_{ts}.pkl` **flat** in the workspace.
- `models/sklearn_model.py:24-26` writes `{timestamp}.ckpt` **flat**, then
  `models/base.py:41-43` copies it to `latest.ckpt` **flat**.
- `pipeline.py:65-68` writes `config_{ts}.yaml` **flat**.
- Nothing anywhere writes a figure.

So `_collect_run_outputs` will find zero subdirectories and copy nothing but
`config.yaml`, `training_log.txt` and a two-key `project_state.json`. `[STATIC]`

**Metrics never reach a file.** `metrics.csv` is written by nothing (grep across the repo
finds the string only in `CLAUDE.md`, `results_reader.py` glob patterns, and the dead
`ensure_metrics_csv_from_log` import). Instead `render_review_results_page` falls back to
`load_metrics_from_logs` (`app.py:1276-1277`), which regex-scrapes the training log
(`results_reader.py:108-134`). Reading the two patterns against BatteryML's actual output —
`print(scores)` at `pipeline.py:113` emits a Python dict repr like
`{'RMSE': 158.4, 'MAE': 96.2, 'MAPE': 0.19}` — pattern `[1]` (`RMSE\s*[:=]`) **cannot match**
because a quote sits between `RMSE` and `:`; pattern `[0]` (the brace pattern +
`ast.literal_eval`) **does** match. So metric display plausibly works, but only by the
second of two patterns and only while the key order is RMSE→MAE→MAPE. `[ASSUMED]` — settled
by running the demo and reading the Metrics tab.

**The predictions plot can never render.** `pipeline.py:115-124` pickles
`{'prediction', 'scores', 'data', 'seed'}`. `load_predictions`
(`results_reader.py:160-190`) looks for a `y_true`-like key among
`["y_true","true","gt","labels","label"]` — **none of which is present**; ground truth lives
inside the `data` DataBundle. So `y_true is None`, and control falls to the
`"scores" in payload` branch (`:187-188`), returning a DataFrame **of the three metric
scores**. The Predictions tab (`app.py:1385-1394`) then looks for `y_true`/`y_pred` columns,
does not find them, and shows *"Prediction file found but parser not implemented for this
format."* `[STATIC]` This is advisor item 2 — prediction curves and the truth-vs-prediction
gap — and it is not merely missing, it is structurally unreachable with the current reader.

---

## G. Reproducibility gap

For each filename `CLAUDE.md` §8 says the manuscript quotes verbatim:

| Artifact | Produced? | Writing code / evidence |
| --- | --- | --- |
| `project_state.json` | **Partial** | `app.py:271` and `app.py:297` write it, but with only `{project_name, mode}` / `{project_name, workspace_dir}`. `core/state.py:53-58` has a richer version — **unused by `app.py`**. No "missing artifact" record. |
| `config.yaml` | **Partial** | `app.py:299-300`, only via `_collect_run_outputs`, i.e. only on the dead Run Model path. Upstream separately writes `config_{ts}.yaml` (`pipeline.py:65-68`) — a *different* name. |
| `schema_mapping.json` | **No** | `report_builder.py:38` *reads* it; nothing writes it. `standardize_dataset` writes `column_provenance.json` + `dataset_profile.json` instead (`standardize_dataset.py:36-48`). |
| `run_metadata.json` | **No** | grep finds the string only in `CLAUDE.md`. No git commit, package versions, config hash, dataset fingerprint or timing is recorded anywhere. |
| `environment_check.json` | **No** | `gpu_checker.get_gpu_readiness_report()` (`gpu_checker.py:173-207`) computes much of the content but **returns a dict and writes no file**, and is imported by nothing but `core/__init__.py:3`. |
| `dataset_status.csv` | **No** | `scripts/reproduce_batteryml/check_dataset_status.py` prints a status table to stdout; it writes no CSV. |
| `feature_summary.json` | **No** | nothing computes it (§D.5). |
| `training_log.txt` | **Yes-ish** | `app.py:301-302` copies the run log to this name — again only via `_collect_run_outputs`. The live quick-start writes `logs/matr_demo.log` instead (`app.py:464`). |
| `metrics.csv` | **No** | §F.6. |
| `predictions/` | **No** | BatteryML writes `predictions_seed_*.pkl` flat (`pipeline.py:120-124`); `_collect_run_outputs` looks for a directory (`app.py:310`). |
| `figures/` | **No** | nothing in the live path writes any figure. `batteryml/visualization/plot_helper.py` exists (123 lines) and is imported by nothing outside the notebooks. |
| `checkpoints/` | **No** | `.ckpt` files land flat (`sklearn_model.py:24-26`). |
| `report.html` | **No** | `report_builder.generate_html_report` exists (`report_builder.py:29-80`) but the module is **unimportable** (§F.4) and nothing imports it. |

**Score: 0 of 13 produced reliably under the specified name; 3 partially, and only on a
code path that cannot currently be reached.** `[STATIC]`

**Determinism.** `--seed` is accepted (`bin/batteryml.py:71`) and applied
(`pipeline.py:33`, `set_seed` at `:238-244` seeds `random`, `numpy`, `torch`, and sets
`cudnn.deterministic`). But: the seed is **not recorded in any artifact**;
`NNModel.fit` takes `seed: int = 0` (`nn_model.py:48`) and `Pipeline.train` calls
`model.fit(dataset, timestamp=ts)` **without passing it** (`pipeline.py:70`), so every NN
checkpoint is named `..._seed_0_...` regardless of `--seed`; and sklearn models write
`{timestamp}.ckpt` with **no seed in the filename at all** (`sklearn_model.py:24`), which
means the `skip_if_executed` guard `glob(f'*seed_{seed}_*.ckpt')` (`pipeline.py:41-43`) can
never fire for them. `[STATIC]`

---

## H. Fragility list — ranked

Ranked by *probability × silence*. Silent-wrong outranks loud-crash.

**1. `smooth()` returns the wrong array — every feature in the repo is affected.**
`batteryml/feature/severson.py:56` returns `meds`, discarding the `res` computed at `:51-55`;
`sigma` is inert. Called at `severson.py:92`, `voltage_capacity_matrix.py:48,71,75`. No
exception, plausible-looking numbers. Highest risk to the manuscript because it is invisible
and it changes the science. `[STATIC]`

**2. Missing config sections are silently replaced by empty dicts.**
`batteryml/utils/config.py:34-36` **prints** `"Missing expected config fields: [...]"` and
continues — the `raise ValueError` on line 36 is commented out. Because the config is an
`addict.Dict` (`:75`), `config[field]` on a missing key returns a fresh empty `Dict` rather
than raising, the emptiness check at `:39-41` only rejects `None`/empty `str`/`list`, and
`:44-46` yields `{}`. `DATA_TRANSFORMATIONS.build({})` then returns `None`
(`registry.py:47-49`), so **the transformation is silently skipped**.
This is not hypothetical: **38 of 114 configs omit `feature_transformation`** — all 32
`nn_models/*` configs (plausibly deliberate) plus `sklearn/{xgb,rf,ridge,pcr,plsr,gpr}/mix.yaml`,
which are **inconsistent with their own siblings** (`sklearn/{dummy,variance_model,
discharge_model,full_model}/mix.yaml` all include it). Those six run **without Z-score
feature normalisation** while every other sklearn config normalises. `[STATIC]`

**3. `Registry.build` returns `None` for a config with no `name`.**
`batteryml/utils/registry.py:47-49`: `if name is None: return`. A typo like `mdoel:` yields a
`None` model, and the failure surfaces far away as
`AttributeError: 'NoneType' object has no attribute 'workspace'` at `pipeline.py:128`.
`error_handle='ignore'` is used for the feature/label/transform builds
(`task.py:28-36` calls `.build(...)` with the default `'raise'` for the splitter only).
`[STATIC]`

**4. UI "Ready" means *raw* data exists, not *processed*.**
`_dataset_card` (`app.py:537-556`) sets `"ready": path.exists()` where `path` is
`entry["path"]` from the manifest, defaulting to `data/raw/{name}` (`:539`); and
`"processed_exists"` is `str(path).startswith(REPO_ROOT/"data"/"processed") and path.exists()`
(`:553`), which is False for any raw path. `render_run_model_page` gates the Run button on
`readiness[ds]["ready"]` (`app.py:1188`). A user with raw-but-unprocessed data therefore sees
**"Ready"**, is allowed to launch, and gets `AssertionError: data/processed/MATR` from
`batteryml/train_test_split/base.py:25`. Precisely the confusion `CLAUDE.md` §5 warns about.
`[STATIC]`

**5. Review Results silently falls back to a hardcoded foreign run directory.**
`app.py:1267`: `project_dir = _project_root() or Path("runs/native_matr_xgb_test")`. With no
project loaded, the page reads metrics, predictions and logs **from a different run** and
labels them as the current result, with no banner distinguishing them. For a platform whose
output goes into a paper, showing one run's numbers under another run's heading is the worst
possible failure mode. `[STATIC]`

**6. Four modules raise `ImportError` on import; two more need `batteryml_web/` on `sys.path`.**
§F.4. The moment anyone wires `report_builder` in to produce `report.html`, or wires
`batteryml_runner` in as the single execution path (`CLAUDE.md` §3 / prompt 4 item 1), the
app stops starting. `[STATIC]`

**7. `run_dry_run` reports success without running anything.**
`core/batteryml_runner.py:60-70` writes a text file and unconditionally returns
`{"success": True, ...}`. Nothing was validated, nothing was executed. If the benchmark
runner's `--dry-run` (prompt 6 item 5) is built on this, it will green-light broken cells.
`[STATIC]`

**8. `config_validator` demotes the one error that matters to a warning, and rejects almost
every real config.** `core/config_validator.py:49-52`: a missing processed dataset appends to
`warnings`, not `errors`, so `valid` stays `True` (`:64`). Meanwhile `SUPPORTED_MODELS` is
`{"XGBoostRULPredictor", "DummyRULPredictor"}` (`:10-13`) and `SUPPORTED_SPLITS` is
`{"MATRPrimaryTestTrainTestSplitter"}` (`:14-16`), so **112 of the 114 configs in the
repository are rejected as "Unsupported"**. The validator both under- and over-blocks.
`[STATIC]`

**9. `config_registry` silently drops 24 configs and misattributes the composite ones.**
`_parse_entry` returns `None` when the model cannot be inferred (`config_registry.py:95-96`).
`MODEL_DISPLAY_NAMES` keys use spaces — `"discharge model"`, `"full model"`,
`"variance model"` (`:26-28`) — while the directories use underscores
(`configs/baselines/sklearn/variance_model/`), so the substring test at `:118` never matches
and **all 24 Severson-family configs vanish from the registry without a message**.
Separately, `_infer_dataset` (`:99-108`) returns the first name found in
`path + filename + full text`, so `cruh.yaml` (which references CALCE, RWTH, UL_PUR, HNEI) is
labelled **RWTH**, and `mix.yaml` is labelled **MATR**. `[STATIC]`

**10. Censored RUL labels are treated as observed, and short-lived cells are dropped
without a count.** `batteryml/label/rul.py:31-32`: a cell that never reaches EOL gets
`label = last_cycle + 1` — a right-censored value fed to the regressor as if measured.
`:34-35`: any cell with `RUL ≤ min_rul_limit` (default 100) becomes `NaN`, and
`task.py:63-68` then silently masks it out of both features and labels. **The number of test
cells actually evaluated is therefore config-dependent and recorded nowhere**, so two runs
quoting "MATR test RMSE" may be over different test sets. `[STATIC]`

**Runners-up** (real, but louder or narrower): `bin/batteryml.py:121` unpacks
`model, dataset = pipeline.train(...)` while `pipeline.py:47` can `return` bare `None` →
`TypeError` whenever `--skip_if_executed true` hits an existing checkpoint;
`nn_model.py:86` calls `link_latest_checkpoint(latest)` with `latest=None` when
`checkpoint_freq` is `None` → `TypeError` in `shutil.copyfile`;
`severson.py:18` returns `np.zeros(interp_dims)` for a short cycle — a silent all-zero
feature; `download.py:72` uses `verify=False`; `custom_uploads.py:36-51` calls
`extractall` with no path-traversal check (zip-slip);
`results_reader.py:103,131,140,188` and `config_registry.py:88-89` swallow bare
`except Exception` around parsing.

---

## Housekeeping observations

Not defects in behaviour, but they will be visible to a Nature Protocols reviewer who
clones the repository:

- **7 backup files are committed**: `batteryml_web/app.py.bak`, `.bak_metrics_*` ×2,
  `.bak_fix_line969_*` ×3, `.bak_widget_keys_*`, plus `app_old.py` (1932 lines) — together
  **~9,700 lines of dead code**, more than the live app. They also contain the only
  BatLiNet references, which is how a reader would wrongly conclude BatLiNet is integrated.
- Two `.DS_Store` files are committed (`configs/.DS_Store`, `configs/soh/.DS_Store`).
- `app.py:1437` defines `render_sidebar_workflow`, **shadowing** the function imported at
  `app.py:29` from `ui/workflow_sidebar.py`.
- `render_llm_output_panel("review_llm_output", ...)` is called three times in three tabs
  (`app.py:1359, 1420, 1433`); if it creates keyed widgets this is a duplicate-key error.
- No `tests/`, no `scripts/verify/`, no `.github/` — there is currently **no automated check
  of any kind**, which is why the `NameError` at `app.py:838` survives in the tree.

---

## What I could not determine by reading

Listed so the spec does not assume them:

1. Whether any dataset is preprocessed on the server (`data/` is `.gitignore`d and absent here).
2. Whether the published BatteryML benchmark numbers were produced with the `smooth()` bug
   (§D.6) — this decides whether fixing it is a correction or a divergence.
3. The real on-disk layout BatteryML leaves in a workspace after a successful run. I inferred
   it from `pipeline.py:120-124`, `sklearn_model.py:24-26` and `base.py:41-43`; a directory
   listing of a real run would confirm it and is the single highest-value piece of evidence
   for milestone M1's output collector.
4. The installed Python, `torch` and `streamlit` versions.
5. Whether the `mix`/`cruh`/`crush` presets have ever been run (they need 4–6 datasets
   preprocessed simultaneously).

---

## Consolidated confirming commands

The single block to run on the server is at the end of `docs/INTEGRATION_MATRIX.md`.
