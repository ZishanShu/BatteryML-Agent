# BatteryML Integration Matrix

Tables B, C and E from `docs/AUDIT.md`, separated so they can be adapted for the
manuscript's supplementary material.

**Evidence class: every row is `[STATIC]` — established by reading source, not by
execution.** Nothing here was run. The "Confirming command" column is the exact command
that would move a row to `VERIFIED`; until its output is pasted back, no row is verified.

Static status vocabulary:

| Status | Meaning |
| --- | --- |
| `CODE-COMPLETE` | A full path exists in source with no gap visible by reading |
| `CODE-PARTIAL` | Something specific is missing — the gap is named in the row |
| `STUB` | The name appears (choice list, docstring, comment) with no implementation |
| `ABSENT` | Not present in live code at all |

Audit commit: `89c3621`.

---

## B. Dataset integration matrix

| Dataset | Downloader (file:line) | Preprocessor (file:line) | Configs referencing it | In web UI? | Static status | Confirming command |
| --- | --- | --- | --- | --- | --- | --- |
| **MATR** | `preprocess/download.py:13-22` (4 `.mat`) | `preprocess_MATR.py:16-17` | 42 direct (14 model dirs × `matr_1`,`matr_2`,`clo`) + 14 via `mix` + 2 SOH | Yes — hardcoded quick-start default, `app.py:40,63` | `CODE-COMPLETE` | `python batteryml_web/core/run_batteryml_cli.py preprocess MATR data/raw/MATR data/processed/MATR && ls data/processed/MATR \| wc -l` |
| **HUST** | `download.py:23-27` (1 `.zip`) | `preprocess_HUST.py:19-20` | 14 (`hust.yaml` in every model dir) + `mix` | Only if `dataset_manifest.json` lists it | `CODE-COMPLETE` | `python batteryml_web/core/run_batteryml_cli.py preprocess HUST data/raw/HUST data/processed/HUST && ls data/processed/HUST \| wc -l` |
| **CALCE** | `download.py:28-42` (13 `.zip`) | `preprocess_CALCE.py:22-23` | 0 direct; only via `cruh`/`crush`/`mix` | Only via manifest | `CODE-PARTIAL` — **no single-dataset config**; `preprocess_CALCE.py:57-59` `continue`s past any cell whose extracted folder yields no `txt/xlsx/xls`, silently reducing the cell count | `python batteryml_web/core/run_batteryml_cli.py preprocess CALCE data/raw/CALCE data/processed/CALCE && ls data/processed/CALCE \| wc -l` (expect 13 if none skipped) |
| **RWTH** | `download.py:43-46` (1 `.zip`) | `preprocess_RWTH.py:19-20` | 0 direct; only via `cruh`/`crush`/`mix` | Only via manifest | `CODE-PARTIAL` — no single-dataset config | `python batteryml_web/core/run_batteryml_cli.py preprocess RWTH data/raw/RWTH data/processed/RWTH && ls data/processed/RWTH \| wc -l` |
| **SNL** | **none** — absent from `DOWNLOAD_LINKS` | `preprocess_SNL.py:16-17` | 14 (`snl.yaml`) + `crush` | Only via manifest | `CODE-PARTIAL` — preprocessor complete, **download path missing**; raw must be fetched by hand from Battery Archive (our own `scripts/reproduce_batteryml/check_dataset_status.py:36` says so) | `python batteryml_web/core/run_batteryml_cli.py preprocess SNL data/raw/SNL data/processed/SNL && ls data/processed/SNL \| wc -l` |
| **HNEI** | **none** | `preprocess_HNEI.py:16-17` | 0 direct; only via `cruh`/`crush`/`mix` | Only via manifest | `CODE-PARTIAL` — no download link, no single-dataset config | `python batteryml_web/core/run_batteryml_cli.py preprocess HNEI data/raw/HNEI data/processed/HNEI && ls data/processed/HNEI \| wc -l` |
| **UL_PUR** | **none** | `preprocess_UL_PUR.py:16-17` | 0 direct; only via `cruh`/`crush`/`mix` | Only via manifest | `CODE-PARTIAL` — no download link, no single-dataset config | `python batteryml_web/core/run_batteryml_cli.py preprocess UL_PUR data/raw/UL_PUR data/processed/UL_PUR && ls data/processed/UL_PUR \| wc -l` |
| **OX** | **none** | `preprocess_OX.py:15-16` | **0** | **No** | `CODE-PARTIAL` — fully implemented but orphaned: no link, no config, no splitter, no UI. Also the only preprocessor that does **not** prefix `cell_id` with the dataset name (`preprocess_OX.py:71` vs `preprocess_CALCE.py:35`) | `python batteryml_web/core/run_batteryml_cli.py preprocess OX data/raw/OX data/processed/OX` — expect success but nothing can consume the output |
| **ARBIN** (cycler) | n/a | `preprocess_arbin.py:17-18` | `configs/cyclers/arbin.yaml`, `arbin_metadata.yaml` | Yes — `custom_uploads.py:13` | `CODE-COMPLETE` | `python batteryml_web/core/run_batteryml_cli.py preprocess ARBIN --config configs/cyclers/arbin.yaml <raw> <out>` |
| **NEWARE** (cycler) | n/a | `preprocess_neware.py:18-19` | `configs/cyclers/neware.yaml` | Yes — `custom_uploads.py:13` | `CODE-COMPLETE` | `python batteryml_web/core/run_batteryml_cli.py preprocess NEWARE --config configs/cyclers/neware.yaml <raw> <out>` |
| **BATTERYARCHIVE** | n/a | **none** | none | no | `STUB` — listed as a CLI choice at `preprocess/__init__.py:20` only | `python batteryml_web/core/run_batteryml_cli.py preprocess BATTERYARCHIVE /tmp/x /tmp/y` → expect `KeyError: BATTERYARCHIVEPreprocessor is not registered` |
| **BIOLOGIC** | n/a | **none** | none | **Yes — offered in the custom-upload UI**, `custom_uploads.py:13` | `STUB` | `python batteryml_web/core/run_batteryml_cli.py preprocess BIOLOGIC /tmp/x /tmp/y` → expect `KeyError` |
| **INDIGO** | n/a | **none** | none | no | `STUB` | as above |
| **LANDT** | n/a | **none** | none | **Yes — offered in the custom-upload UI** | `STUB` | as above |
| **MACCOR** | n/a | **none** | none | **Yes — offered in the custom-upload UI** | `STUB` | as above |
| **NOVONIX** | n/a | **none** | none | no | `STUB` | as above |

**Row-level notes.**
- "In web UI?" is qualified for most datasets because `_ui_dataset_options()`
  (`app.py:558-561`) derives the entire dataset list from
  `data/metadata/dataset_manifest.json`, which is **not in the repository** (`data` is
  `.gitignore`d). On a fresh clone the list is empty. Run
  `python tools/sync_data_and_config_manifests.py` first.
- Three of the five formats offered for custom upload (BIOLOGIC, LANDT, MACCOR) have no
  preprocessor. This is a user-facing promise the code cannot keep.

---

## C. Model / method integration matrix

All 14 registered models are RUL models under `batteryml/models/rul_predictors/`. Device
column reads: `cpu` for `SklearnModel` subclasses (`models/sklearn_model.py:14-40` moves the
dataset to CPU for both `fit` and `predict` regardless of `--device`), `cpu|cuda` for
`NNModel` subclasses (`models/nn_model.py:98-99` delegates to `nn.Module.to`).

| Model | Implementation (file:line) | Registered? | Config template(s) | Tasks | Device | In UI? | Static status | Confirming command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **XGBoost** | `rul_predictors/xgb.py:10-14` | Yes | `configs/baselines/sklearn/xgb/*.yaml` (8) | RUL | cpu | Yes — quick-start default | `CODE-COMPLETE` | `python batteryml_web/core/run_batteryml_cli.py run configs/baselines/sklearn/xgb/matr_1.yaml --workspace runs/v_xgb --train --eval --device cpu --seed 0` |
| **Random Forest** | `random_forest.py:10-14` | Yes | `sklearn/rf/*.yaml` (8) | RUL | cpu | Yes | `CODE-COMPLETE` | `... run configs/baselines/sklearn/rf/matr_1.yaml --workspace runs/v_rf --train --eval --device cpu --seed 0` |
| **Ridge** | `ridge.py:10-14` | Yes | `sklearn/ridge/*.yaml` (8) | RUL | cpu | Yes | `CODE-PARTIAL` — `:14` accepts `*args` and drops it silently | `... run configs/baselines/sklearn/ridge/matr_1.yaml --workspace runs/v_ridge --train --eval --device cpu --seed 0` |
| **PCR** | `pcr.py:12-16` | Yes | `sklearn/pcr/*.yaml` (8) | RUL | cpu | Yes | `CODE-COMPLETE` | `... run configs/baselines/sklearn/pcr/matr_1.yaml --workspace runs/v_pcr --train --eval --device cpu --seed 0` |
| **PLSR** | `plsr.py:10-14` | Yes | `sklearn/plsr/*.yaml` (8) | RUL | cpu | Yes | `CODE-COMPLETE` | `... run configs/baselines/sklearn/plsr/matr_1.yaml --workspace runs/v_plsr --train --eval --device cpu --seed 0` |
| **Gaussian Process** | `gaussian_process.py:11-16` | Yes | `sklearn/gpr/*.yaml` (8) | RUL | cpu | Yes | `CODE-PARTIAL` — **`*args/**kwargs` accepted then discarded**; kernel hardcoded `DotProduct()+RBF()` at `:14-15`. Config hyperparameters have no effect. | `... run configs/baselines/sklearn/gpr/matr_1.yaml --workspace runs/v_gpr --train --eval --device cpu --seed 0` |
| **Linear Regression** | `linear_regression.py:10-14` | Yes | 24 configs — `sklearn/{variance_model,discharge_model,full_model}/*.yaml` | RUL | cpu | **No** — dropped by the registry, see note | `CODE-PARTIAL` | `... run configs/baselines/sklearn/variance_model/matr_1.yaml --workspace runs/v_var --train --eval --device cpu --seed 0` |
| **Dummy** | `dummy.py:10-14` | Yes | `sklearn/dummy/*.yaml` (8) + `configs/soh/dummy/*.yaml` (2) | RUL, "SOH" | cpu | Yes | `CODE-COMPLETE` (as a baseline) | `... run configs/baselines/sklearn/dummy/matr_1.yaml --workspace runs/v_dummy --train --eval --device cpu --seed 0` |
| **MLP** | `mlp.py:26-60` | Yes | `nn_models/mlp/*.yaml` (8) | RUL | cpu\|cuda | Yes | `CODE-PARTIAL` — config omits `feature_transformation` (see E note) | `... run configs/baselines/nn_models/mlp/matr_1.yaml --workspace runs/v_mlp --train --eval --device cpu --seed 0` |
| **CNN** | `cnn.py:60-100` | Yes | `nn_models/cnn/*.yaml` (8) | RUL | cpu\|cuda | Yes | `CODE-PARTIAL` — `BatteryDropout.forward` (`cnn.py:16-19`) is `pass`, returning `None`; the class is defined but unused | `... run configs/baselines/nn_models/cnn/matr_1.yaml --workspace runs/v_cnn --train --eval --device cpu --seed 0` |
| **LSTM** | `lstm.py:11-39` | Yes | `nn_models/lstm/*.yaml` (8) | RUL | cpu\|cuda | Yes | `CODE-COMPLETE` | `... run configs/baselines/nn_models/lstm/matr_1.yaml --workspace runs/v_lstm --train --eval --device cpu --seed 0` |
| **Transformer** | `transformer.py:11-47` | Yes | `nn_models/transformer/*.yaml` (8) | RUL | cpu\|cuda | Yes | `CODE-PARTIAL` — **`print(feature.shape)` on every forward pass** (`:35`) floods the log the metrics parser reads | `... run configs/baselines/nn_models/transformer/matr_1.yaml --workspace runs/v_tf --train --eval --device cpu --seed 0` |
| **SVM** | `svm.py:10-14` | Yes | **none** | RUL | cpu | No | `CODE-PARTIAL` — **importable and registered, wired into zero configs** | none exists; a config must be written first |
| **Elastic Net** | `elastic_net.py:10-14` | Yes | **none** | RUL | cpu | No | `CODE-PARTIAL` — **importable and registered, wired into zero configs** | none exists; a config must be written first |
| **Any SOH model** | — | — | — | — | — | — | **`ABSENT`** — `batteryml/models/soh_predictors/__init__.py` is a 4-line docstring with no class and is imported by nothing | n/a |
| **BatLiNet** | — | — | — | — | — | — | **`ABSENT`** — occurs only in `app.py.bak*` and `app_old.py:1531`, both saying "not yet implemented" | n/a |

**Registry note (affects the "In UI?" column).** `config_registry._parse_entry`
(`config_registry.py:80-97`) discards any config whose model it cannot infer. The lookup keys
at `config_registry.py:26-28` are `"discharge model"`, `"full model"`, `"variance model"` —
**with spaces**, while the directories are `discharge_model/`, `full_model/`,
`variance_model/` — with underscores. The substring test at `:118` therefore never matches,
`_infer_model` returns `(None, None)`, and **all 24 Linear-Regression configs are silently
dropped from the UI**.

**Config integrity check.** Every `model`, `train_test_split`, `feature`, `label`,
`feature_transformation` and `label_transformation` name in all 114 configs resolves to a
registered class. **No config references a non-existent class.**

---

## E. Task coverage grid

### E.1 RUL — 112 configs

Rows are model directories, columns are split presets. `✓` = a config file exists.

| Model | matr_1 | matr_2 | clo | hust | snl | cruh | crush | mix |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| xgb | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| rf | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| ridge | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| pcr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| plsr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| gpr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| dummy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| variance_model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| discharge_model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| full_model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| mlp | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† |
| cnn | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† |
| lstm | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† |
| transformer | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† | ✓† |
| **svm** | — | — | — | — | — | — | — | — |
| **elastic_net** | — | — | — | — | — | — | — | — |

`*` — **missing `feature_transformation`, inconsistently with its own siblings.** Six configs:
`sklearn/{xgb,rf,ridge,pcr,plsr,gpr}/mix.yaml`. `sklearn/{dummy,variance_model,
discharge_model,full_model}/mix.yaml` all include it. Per `utils/config.py:34-36` the
omission produces a `print` and then **no feature normalisation**, silently. These six cells
are not comparable to the rest of the `mix` column.

`†` — all 32 `nn_models/*` configs omit `feature_transformation`. Uniform across the family,
so plausibly deliberate — but it is nowhere documented, and it means **NN and sklearn results
in the same table are computed on differently scaled features**.

**Gap count (RUL): 16** — the two `svm` and `elastic_net` rows × 8 presets. Within the 14
model directories that have configs, the grid is complete: **112/112**.

### E.2 Which dataset each preset actually needs

From `cell_data_path` in the configs; **every cell listed must be preprocessed before the
preset can run**, because `train_test_split/base.py:25` asserts each path exists.

| Preset | Splitter class | Processed datasets required |
| --- | --- | --- |
| `matr_1` | `MATRPrimaryTestTrainTestSplitter` (`MATR_split.py:31-32`) | MATR |
| `matr_2` | `MATRSecondaryTestTrainTestSplitter` (`MATR_split.py:65-66`) | MATR |
| `clo` | `MATRCLOTestTrainTestSplitter` (`MATR_split.py:89-90`) | MATR |
| `hust` | `HUSTTrainTestSplitter` (`HUST_split.py:11-12`) | HUST |
| `snl` | `SNLTrainTestSplitter` (`SNL_split.py:8-9`) | SNL |
| `cruh` | `CRUHTrainTestSplitter` (`CRUH_split.py:8-9`) | CALCE + RWTH + UL_PUR + HNEI |
| `crush` | `CRUSHTrainTestSplitter` (`CRUSH_split.py:8-9`) | CALCE + RWTH + UL_PUR + SNL + HNEI |
| `mix` | `MIX100TrainTestSplitter` (`MIX100_split.py:8-9`) | CALCE + RWTH + UL_PUR + HNEI + MATR + HUST |

**Train/test leakage check.** `CRUH_split.py:52-57`, `CRUSH_split.py`, `MIX100_split.py:157-162`
and `HUST_split.py` all build `train = [f for f in file_list if f.stem not in test_ids]`,
so train and test are complements by construction — **no overlap is possible**.
`MATR_split.py:14-25` maps explicit `train_ids` and `test_ids` lists through a path map and
asserts both resolve completely (`:24-25`); I checked the `matr_1` lists at `MATR_split.py:34-58`
and **found no id present in both**. `[STATIC]`

### E.3 SOH — 2 configs

| Model | matr_1 | matr_2 | clo | hust | snl | cruh | crush | mix |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| dummy (`DummyRULPredictor`) | ✓✓ | — | — | — | — | — | — | — |
| every other model | — | — | — | — | — | — | — | — |

Both files are MATR (`configs/soh/dummy/matr_1.yaml`, `matr_1_with_soh_filepath.yaml`); they
differ only in whether `soh_filepath` is set (`label/soh.py:17-35` reads a `.json` or `.pkl`
SOH dictionary when it is, and falls back to computing `Qd(cycle_index)/nominal_capacity` from
the cycle data when it is not, `soh.py:44-63`).

**Gap count (SOH): 110 of 112 cells empty, and the 2 filled cells use a dummy regressor.**
The `configs/soh/dummy/matr_1_with_soh_filepath.yaml` config points at
`data/example_soh_dict.json`, which is not in the repository.

**SOH is effectively unimplemented**: the label annotator exists, no model does.

---

## Consolidated confirming-command block

Run from the repository root on the server, in order. Paste the whole transcript back.
Nothing here trains a model except section 4; sections 1–3 are cheap.

```bash
# ============================================================
# 0. Environment — settles §A.4 (unpinned/missing dependencies)
# ============================================================
python -c "import sys; print('python', sys.version)"
python -c "import torch, numpy, sklearn, xgboost, pandas, scipy, numba, h5py, yaml, addict; \
print('torch', torch.__version__); print('numpy', numpy.__version__); \
print('sklearn', sklearn.__version__); print('xgboost', xgboost.__version__); \
print('pandas', pandas.__version__); print('scipy', scipy.__version__); \
print('numba', numba.__version__); print('h5py', h5py.__version__)"
python -c "import streamlit; print('streamlit', streamlit.__version__)"
python -c "import torch; print('cuda_available', torch.cuda.is_available())"
python -c "import batteryml; print('batteryml OK', batteryml.__file__)"

# ============================================================
# 1. CLI contract — confirms CLAUDE.md §5 still holds
# ============================================================
python batteryml_web/core/run_batteryml_cli.py --help
python batteryml_web/core/run_batteryml_cli.py run --help
python batteryml_web/core/run_batteryml_cli.py preprocess --help

# ============================================================
# 2. Registry contents — settles table C in one shot
# ============================================================
python - <<'PY'
import batteryml, batteryml.preprocess, batteryml.models.rul_predictors
import batteryml.feature, batteryml.label, batteryml.train_test_split
import batteryml.data.transformation
from batteryml.builders import (MODELS, PREPROCESSORS, FEATURE_EXTRACTORS,
                                LABEL_ANNOTATORS, TRAIN_TEST_SPLITTERS,
                                DATA_TRANSFORMATIONS)
for r in (MODELS, PREPROCESSORS, FEATURE_EXTRACTORS, LABEL_ANNOTATORS,
          TRAIN_TEST_SPLITTERS, DATA_TRANSFORMATIONS):
    print(f"{r.name}: {sorted(r.class_mapping)}")
PY

# ============================================================
# 3. Data presence — settles table B and CLAUDE.md §11
# ============================================================
ls -la data/raw/ 2>/dev/null || echo "NO data/raw"
ls -la data/processed/ 2>/dev/null || echo "NO data/processed"
for d in CALCE HNEI HUST MATR RWTH SNL UL_PUR OX; do
  r=$(ls data/raw/$d 2>/dev/null | wc -l)
  p=$(ls data/processed/$d/*.pkl 2>/dev/null | wc -l)
  echo "$d raw_files=$r processed_pkl=$p"
done
python scripts/reproduce_batteryml/check_dataset_status.py

# ============================================================
# 4. The one real run — MATR / RUL / XGBoost / cpu / seed 0
#    Confirms the CLAUDE.md §5 path AND gives me the real
#    workspace layout, which is my single biggest [ASSUMED].
# ============================================================
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace runs/audit_matr_xgb --train --eval --device cpu --seed 0 \
  2>&1 | tee /tmp/audit_matr_xgb.log
echo "----- WORKSPACE LAYOUT (this is what I need most) -----"
find runs/audit_matr_xgb -type f | sort
echo "----- CACHE -----"
ls -la cache/ 2>/dev/null || echo "no cache dir"
echo "----- PREDICTION PICKLE KEYS -----"
python - <<'PY'
import pickle, glob
for f in sorted(glob.glob('runs/audit_matr_xgb/predictions_seed_0_*.pkl')):
    with open(f,'rb') as fh: obj = pickle.load(fh)
    print(f, '->', type(obj), list(obj) if isinstance(obj, dict) else '')
    if isinstance(obj, dict):
        print('  scores:', obj.get('scores'))
        p = obj.get('prediction')
        print('  prediction type/shape:', type(p), getattr(p, 'shape', None))
PY

# ============================================================
# 5. Web layer — confirms the two fatal name errors (§F.3)
#    Expect: SUCCESS, then NameError, then AttributeError.
# ============================================================
python -c "import batteryml_web.app" 2>&1 | tail -5
python - <<'PY'
from pathlib import Path
from batteryml_web.core.config_registry import get_config_registry
reg = get_config_registry(str(Path('.').resolve()))
print('configs parsed into registry:', len(reg.entries), 'of 114 on disk')
print('ConfigEntry fields:', list(reg.entries[0].__dataclass_fields__) if reg.entries else 'none')
try:
    reg.entries[0].model
except AttributeError as e:
    print('CONFIRMED AttributeError:', e)
import collections
print('datasets inferred:', collections.Counter(e.dataset for e in reg.entries))
PY

# ============================================================
# 6. The four unimportable modules (§F.4) — expect 4 ImportErrors
# ============================================================
for m in batteryml_web.core.batteryml_runner batteryml_web.core.report_builder \
         batteryml_web.agent.schema_agent batteryml_web.agent.result_agent; do
  echo "--- $m"; python -c "import $m" 2>&1 | tail -2
done

# ============================================================
# 7. The smooth() defect (§D.6, §H.1) — pure function, no data
#    Expect: "sigma has no effect: True" and a non-zero max diff.
# ============================================================
python - <<'PY'
import numpy as np
from batteryml.feature.severson import smooth
rng = np.random.default_rng(0)
x = np.sin(np.linspace(0, 12, 1000)) + 0.01*rng.standard_normal(1000)
x[500] += 5.0                      # a single obvious outlier
a, b = smooth(x, sigma=3), smooth(x, sigma=1000)
print('sigma has no effect:', np.array_equal(a, b))
print('max |smooth(x) - x| :', float(np.abs(a - x).max()))
print('outlier at 500 -> x=%.3f smoothed=%.3f' % (x[500], a[500]))
PY
```

**How to read the output.**
- Section 4's `find` listing is the highest-value item in this block: it replaces my
  inferred workspace layout with fact and unblocks milestone M1's output collector.
- Section 5's first line may fail with a Streamlit *session-state* error rather than
  importing cleanly — `app.py:71` touches `st.session_state` at module scope, which is only
  legal inside a Streamlit script run. **That is not the defect under test**; ignore it. The
  defects under test are in the second command: a confirmed `AttributeError` on
  `entry.model`, and the parsed-config count. **If that count is 90 rather than 114, the 24
  silently dropped Linear-Regression configs are confirmed.** The two `NameError`s at
  `app.py:838`/`:1075` live inside function bodies and will only surface when those pages
  render, so confirm them by opening the app and clicking *Prepare Data* and *Run Model*.
- Section 7 confirms or refutes finding H.1 without any data. If `sigma has no effect: True`
  prints, the bug is real, and the next decision — whether to fix it — belongs in `docs/SPEC.md`,
  not in a patch.

After pasting, say: *"Update `docs/INTEGRATION_MATRIX.md` from this real output. Mark rows
VERIFIED or BROKEN accordingly and do not change any row the output does not cover."*
