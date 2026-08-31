# Adding a model (Route C)

How to add a model to this platform without touching `batteryml/`.

`CLAUDE.md` §4 forbids modifying upstream BatteryML core, but a model has to
reach BatteryML's `MODELS` registry to be usable. `batteryml_web/models/` is the
extension point that bridges the two.

Everything below is `[STATIC]` unless marked otherwise: established by reading
the code, not by running the pipeline. The command block at the end is what
turns it into `[VERIFIED]` on the server.

---

## How registration actually reaches the pipeline

```
run_batteryml_cli.py                     ← the wrapper, per CLAUDE.md §5
  ├─ sys.path.insert(repo_root)
  ├─ register_extension_models()         ← imports batteryml_web/models/<name>.py
  │                                         so its @MODELS.register() fires
  └─ runpy.run_path(bin/batteryml.py)
        └─ Pipeline → MODELS.build({'name': '<X>RULPredictor', ...})
```

`bin/batteryml.py` imports only `batteryml.*`, so without the wrapper's call a
model defined outside `batteryml/` never registers, and the run fails inside
`MODELS.build()` with a bare `KeyError: <X> is not registered to Models!`
(`batteryml/utils/registry.py:59-60`).

The Streamlit UI shells out to the same wrapper (`batteryml_web/app.py:698`,
`:1176`), so the CLI and the UI are both covered.

**What does not see extension models:** the `batteryml` console script that
`setup.py` installs (`entry_points` → `bin.batteryml:main`), and
`run_all_rul_baseline.sh`. Both execute `bin/batteryml.py` directly, bypassing
the wrapper. Use `python batteryml_web/core/run_batteryml_cli.py run ...`.

---

## The four steps

Worked example: adding `ExtraTreesRegressor`, the natural sibling of the
existing Random Forest baseline. No new dependency — scikit-learn is already in
`requirements.txt`.

### 1. The model module

`batteryml_web/models/extra_trees.py`:

```python
"""Extra Trees RUL predictor.

Extension model: registered into BatteryML's MODELS registry without
modifying batteryml/ (CLAUDE.md §4).
"""

from sklearn.ensemble import ExtraTreesRegressor

from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel


@MODELS.register()
class ExtraTreesRULPredictor(SklearnModel):
    def __init__(self, workspace: str = None, **kwargs):
        SklearnModel.__init__(self, workspace)
        self.model = ExtraTreesRegressor(**kwargs)
```

Three constraints, all load-bearing:

| Constraint | Why | Evidence |
| --- | --- | --- |
| The **class name** is what a config's `model.name` must say. | `Registry.register()` keys on `cls.__name__` when no explicit name is given. | `batteryml/utils/registry.py:36-37` |
| The **`RULPredictor` / `SOHPredictor` suffix** is functional, not cosmetic. | `REG-01` derives the task from it, and `catalog.py:103-110` matches on it. | `scripts/verify/check_registry_integrity.py`, `batteryml_web/core/catalog.py:103-110` |
| Do **not** write `*args`. | `Registry.build()` passes only keyword arguments — `{k: v for k, v in config.items() if k != 'name'}`. A `*args` parameter is always empty. | `batteryml/utils/registry.py:53-57` |

Some existing core models (`svm.py`, `linear_regression.py`) do carry `*args`.
That is a known dead parameter on your blocked list; do not copy the pattern
into new code.

**Base class contract:**

| Base | You must | Evidence |
| --- | --- | --- |
| `SklearnModel` | set `self.model` to any estimator with `.fit(X, y)` / `.predict(X)`. `fit`, `predict`, and pickle checkpointing come for free. Features arrive flattened via `feature.view(len(feature), -1)`. | `batteryml/models/sklearn_model.py:13-51` |
| `NNModel` | implement `forward(self, feature, label, return_loss=False)`, returning the loss when `return_loss=True` and predictions otherwise. Call `NNModel.__init__(self, **kwargs)` first. | `batteryml/models/nn_model.py:25`, `:65` |

Inheriting from one of these two is also what lets `REG-01` derive device
support (`SklearnModel` → `["cpu"]`, `NNModel` → `["cpu", "cuda"]`). A model
that inherits neither makes `REG-01` fail with
`device support is not derivable`.

### 2. Declare it

`batteryml_web/models/__init__.py`:

```python
EXTENSION_MODULES: tuple[str, ...] = ("extra_trees",)
```

Not optional and not inferred. A module file present but unlisted is never
imported, so its registration never fires. `register_extension_models()`
refuses to start while an unlisted file exists, and names it; `REG-01` reports
the same condition. Prefix a file with `_` if it is a shared helper rather than
a model — those are ignored.

### 3. The eight configs

The directory name **is** the model key, so it must match what you register in
step 4:

```bash
mkdir -p configs/baselines/sklearn/extra_trees
for f in clo cruh crush hust matr_1 matr_2 mix snl; do
  sed "s/name: 'RandomForestRULPredictor'/name: 'ExtraTreesRULPredictor'/" \
    configs/baselines/sklearn/rf/$f.yaml > configs/baselines/sklearn/extra_trees/$f.yaml
done
```

Copy from the closest existing sibling and **change only `model.name`**. In
particular, carry `feature_transformation` over exactly as the template has it,
including its absence in `mix.yaml` — whether a config declares one is a
blocked decision, not something to normalize while adding a model.

A neural-network model goes under `configs/baselines/nn_models/<name>/`
instead.

### 4. Register the display name

`batteryml_web/core/config_registry.py`, three tables:

```python
MODEL_DISPLAY_NAMES = { ...  "extra trees": "Extra Trees Regression", }
MODEL_KEY_ALIASES   = { ...  "extra trees": "extra_trees", }
MODEL_CLASS_KEYS    = { ...  "extratreesrulpredictor": "extra trees", }
```

Write the spaced form. `normalize_model_token()` folds the directory name
`extra_trees` to `extra trees` before the lookup, which is what makes the
underscore/space distinction stop mattering. Without these three lines the
configs still run from the CLI, but the model does not appear in the UI.

---

## Verify

```bash
# Extension point consistent; class registered, unique, devices derivable
python -m scripts.verify.run_all --only REG-01

# All configs parse and name a registered implementation
python -m scripts.verify.run_all --only CFG-01

# UI registry sees it
python -c "
import sys;sys.path.insert(0,'.')
from batteryml_web.core.config_registry import ConfigRegistry
from pathlib import Path; import collections
r=ConfigRegistry(Path('.')).build()
print('parsed:',len(r.entries))
print(sorted(collections.Counter(e.model_key for e in r.entries).items()))"

# Actually run it (server only)
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/extra_trees/matr_1.yaml \
  --workspace runs/extra_trees_matr1 --train --eval --device cpu --seed 0
```

Each model added should move `REG-01`'s `MODELS` count by 1 and `CFG-01`'s
config count by 8.

`REG-01`'s report records, per model, an `origin` of `upstream` or `extension`
and the defining `path:line`, so it is always visible which models come from
where.

---

## The execution gate needs no edit

`batteryml_web/core/config_validator.py` used to whitelist two model names and
one splitter name, so it accepted **4 of 130** configs and would have refused
every new model. It now resolves each declared name against the live
`batteryml.builders` registries, populating them the way a run does — importing
`batteryml`, then calling `register_extension_models()`. **A model added
through the four steps above is accepted by the gate automatically.**

`VAL-01` (`python -m scripts.verify.run_all --only VAL-01`) runs the real
validator over every config and reports what it would refuse, so a drift
between the gate and the registries is caught rather than discovered at run
time.

Two things to know about it:

* It resolves names **live**, so it needs `batteryml` importable. That is not a
  limitation: it runs immediately before a run, and if the import fails the run
  cannot happen either. The failure is reported with the import error, never
  passed over silently.
* A declared `cell_data_path` that is missing or holds no `*.pkl` is an
  **error**, because the run provably cannot succeed. Pass `require_data=False`
  to check a config's structure before preprocessing; the same findings come
  back as warnings.

---

## Known blockers

### A third pattern table in `catalog.py`

`batteryml_web/core/catalog.py:113-116` keeps its own `NN_MODEL_PATTERNS` and
`SKLEARN_PATTERNS` lists for device inference, independent of both
`config_registry.py` and `REG-01`. `SVM` and `ElasticNet` happen to already be
in `SKLEARN_PATTERNS`; a new model may need adding. Consolidating these three
sources of model metadata is worth doing, but is a separate design decision.

### New third-party dependencies

A model needing LightGBM, CatBoost or similar brings a new dependency.
`CLAUDE.md` §6 requires justifying it and pinning it, and `requirements.txt`
records the convention for versions this repository cannot observe.
