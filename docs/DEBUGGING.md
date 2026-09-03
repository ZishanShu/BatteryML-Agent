# Debugging this repository locally

Aimed at import errors and environment problems — the failures that account for
almost everything you hit before a run ever starts.

The method is three questions, always in this order. Skipping straight to
question 3 is how an afternoon disappears.

> 1. **Which interpreter is running?**
> 2. **What is installed in it?**
> 3. **Which import actually failed, and why?**

---

## 0. One command first

```bash
python -m scripts.verify.check_environment
```

This is written to run when nothing else does: it imports only the standard
library, so it works with no pyyaml, no scikit-learn, no torch. It prints the
interpreter, every distribution `requirements.txt` declares with its installed
version or `NOT INSTALLED`, whether `batteryml` imports, and whether a GPU is
visible. It writes `runs/verify/environment_check.json` and
`package_versions.txt`.

`python -m scripts.verify.run_all --only ENV-01` does the same inside the
harness.

If that answers your question, stop reading.

---

## 1. Which interpreter is running?

Most "it works in one shell and not another" problems end here.

```bash
which python                 # what the shell resolves
python -c "import sys; print(sys.executable)"
python -c "import sys; print(sys.prefix)"     # the env root
conda env list                                # * marks the active one
```

**`pip` and `python` can disagree.** `pip install X` may install into a
different environment than the `python` you then run. Always prefer:

```bash
python -m pip install X      # guaranteed to be this interpreter's pip
python -m pip --version      # prints the pip AND the python it belongs to
```

### Which env actually has the dependencies?

If you are not sure which environment BatteryML was ever working in:

```bash
for e in $(conda env list | awk '/^[a-zA-Z]/{print $1}'); do
  printf '%-20s ' "$e"
  conda run -n "$e" python -c \
    "import yaml,sklearn,torch,xgboost;print('yaml sklearn torch xgboost all present')" 2>&1 | tail -1
done
```

The env that prints all four is the one to use. If none does, none of them can
run BatteryML.

---

## 2. What is installed in it?

```bash
python -m pip list                     # everything
python -m pip show scikit-learn        # one package: version, location
python -m pip check                    # broken dependency graph
python -m pip install -r requirements.txt
```

### The install name is not the import name

This is the single most common trap, and it is what makes `pip install yaml`
fail with *"Could not find a version that satisfies the requirement yaml"*.
The name in `import` and the name on PyPI are unrelated strings.

| Code says | `pip install` wants |
| --- | --- |
| `import yaml` | **`pyyaml`** |
| `import sklearn` | **`scikit-learn`** |
| `import torch` | `torch` |
| `import xgboost` | `xgboost` |
| `import addict` | `addict` |

`requirements.txt` always lists **install names**, which is why
`pip install -r requirements.txt` is safer than installing packages one at a
time from the names in a traceback.

To go the other way — you have an import name and want the distribution:

```bash
python -c "
import importlib.metadata as m
print(m.packages_distributions().get('sklearn'))"     # -> ['scikit-learn']
```

---

## 3. Which import failed, and why?

Read the traceback **bottom-up**. The last line is the error; the lines above
are the chain that led there. In particular the *first* import in a chain to
fail is the one to fix, not the module that happened to trigger the chain.

```bash
# Does one module import at all, and from where?
python -c "import yaml; print(yaml.__file__)"

# Where would Python look?
python -c "import sys; print('\n'.join(sys.path))"

# What is the whole import chain, and what is slow?
python -X importtime -c "import batteryml" 2>&1 | tail -30
```

### Follow the chain rather than guessing

`import batteryml` reaches scikit-learn through four hops:

```
batteryml/__init__.py:13            from .models import CNNRULPredictor
  models/__init__.py:4                from .rul_predictors import CNNRULPredictor
    rul_predictors/__init__.py         imports pcr, svm, plsr, elastic_net,
                                       random_forest, linear_regression, ridge,
                                       gaussian_process
      each of those                    from sklearn... import ...
```

So **`import batteryml` cannot succeed without scikit-learn**. If you see
`No module named 'sklearn'` from an extension model in
`batteryml_web/models/`, the extension is not the cause — it simply imports
sklearn a few lines earlier than core does.

Trace any chain yourself with:

```bash
grep -rn "^from \|^import " batteryml/models/rul_predictors/__init__.py
grep -h "^from sklearn" batteryml/models/rul_predictors/*.py
```

### Namespace packages, and why `scripts.verify` can vanish

A directory without `__init__.py` is a *namespace package*. A regular package
(with `__init__.py`) found anywhere on `sys.path` **beats every namespace
portion**, even one earlier in the path. So if any installed distribution
provides a top-level `scripts` package, the repository's own `scripts/` is
shadowed and you get:

```
ModuleNotFoundError: No module named 'scripts.verify'
```

Note the shape of that message: `scripts` resolved, `scripts.verify` did not.
That is the signature of shadowing, as opposed to `No module named 'scripts'`,
which means it was not found at all.

`scripts/__init__.py` now exists to prevent this. To check what a name resolves
to:

```bash
python -c "import scripts; print(scripts.__file__, scripts.__path__)"
```

### `python -m X` vs `python path/to/x.py`

They put different things on `sys.path[0]`:

| Command | `sys.path[0]` |
| --- | --- |
| `python -m pkg.mod` | the current working directory |
| `python path/to/script.py` | **the script's directory**, not the cwd |

That is why `python -m scripts.verify.run_all` must be run from the repository
root, and why a helper script placed elsewhere cannot import the repo unless it
adds the root itself. `scripts/preprocess_sweep.py` does exactly that, at the
top of the file.

---

## 4. Reproducing a broken environment without breaking yours

To confirm a fix handles a missing package, block the import instead of
uninstalling anything:

```python
# block.py
import os, runpy, sys
sys.path.insert(0, os.getcwd())

class Blocker:
    def __init__(self, names): self.names = set(names)
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)

blocked = sys.argv[1].split(",")
for name in [n for n in sys.modules if n.split(".")[0] in blocked]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker(blocked))

assert sys.argv[2] == "-m"
target = sys.argv[3]
sys.argv = [target] + sys.argv[4:]
runpy.run_module(target, run_name="__main__", alter_sys=True)
```

```bash
python block.py yaml,sklearn,torch -m scripts.verify.run_all
```

This is how the harness's behaviour under a missing pyyaml was verified without
uninstalling it.

---

## 5. Reading this project's own error messages

The messages are written to be actionable; the useful information is usually
already in them.

| Message | Means | Do |
| --- | --- | --- |
| `AssertionError: data/processed/MATR` | preprocessing has not been run for that dataset | `python scripts/preprocess_sweep.py --only MATR` |
| `<X> is not registered to Models!` | the config names a model whose module never imported | `python -m scripts.verify.run_all --only REG-01` |
| `error: ... module file(s) ... are not listed in EXTENSION_MODULES` | a model file exists but was not declared | add it to `EXTENSION_MODULES` in `batteryml_web/models/__init__.py` |
| `error: Extension model module ... needs a package that is not installed` | environment, not code | `pip install -r requirements.txt` |
| `unrecognized arguments: --config` | the config path is positional | see `CLAUDE.md` §5 |
| `ModuleNotFoundError: ... 'batteryml' is not a package` | `bin/batteryml.py` was run directly | go through `batteryml_web/core/run_batteryml_cli.py` |

The CLI wrapper's exit code tells you which layer failed:

| Code | Meaning |
| --- | --- |
| 0 | the underlying CLI succeeded |
| 1 | the CLI itself failed and printed its own error |
| 2 | a wrapper precondition failed: extension registration, or an unimplemented `input_type` |
| 3 | preprocessing ran but produced no BatteryData |

---

## 6. When the environment is fine and the failure is real

Only at this point are the pipeline-level tools worth reaching for.

```bash
# Full report; writes runs/verify/verification_report.json
python -m scripts.verify.run_all

# One check at a time
python -m scripts.verify.run_all --only CFG-01
python -m scripts.verify.run_all --list

# What would run, without running it
python scripts/preprocess_sweep.py --dry-run

# Would the execution gate accept this config, and why not?
python -c "
import sys; sys.path.insert(0,'.')
from batteryml_web.core.config_validator import load_and_validate_config_file as v
r = v('configs/baselines/sklearn/xgb/matr_1.yaml')
print('valid:', r['valid'])
for e in r['errors']: print('  error:', e)
for w in r['warnings']: print('  warn :', w)"
```

Every check writes structured evidence to `runs/verify/`; when a message is not
enough, the JSON has the detail behind it.

---

## 7. "I re-preprocessed the data but the numbers did not change"

`build_dataset` caches the assembled dataset to
`cache/battery_cache_<hash>.pkl`, and the hash is computed **from config fields
only** — `train_test_split`, `feature`, `label`, `feature_transformation`,
`label_transformation` (`batteryml/pipeline.py:186`, hashed at `:191-193`). `[STATIC]`

The contents of `data/processed/` are **not** part of that hash. So
re-running preprocessing does not invalidate the cache: a subsequent run
silently loads the old dataset and reports metrics computed from data that is
no longer on disk. The only visible sign is a line in the log:

```
Load datasets from cache cache/battery_cache_0ff5ff401e0feff22f54360eae76bc3f.pkl.
```

Excluding `model` from the hash is deliberate and correct — every model on a
given split then shares one assembled dataset, which is why a second model on
`matr_1` starts instantly. The missing piece is a fingerprint of the data
itself.

**After any preprocessing run, clear the cache before trusting a number:**

```bash
rm -rf cache/
```

Two things follow from this that are worth deciding rather than living with:

* `CLAUDE.md` §8 requires `run_metadata.json` to record a **dataset
  fingerprint**. Nothing computes one yet. The same fingerprint would fix this
  cache, so the two should be designed together.
* Until then, a reported metric is only trustworthy if `cache/` was empty when
  the run started, or if the log line above is absent.

To see what a cache entry corresponds to:

```bash
ls -la cache/
python -c "
import pickle, sys
d = pickle.load(open(sys.argv[1], 'rb'))
print('train cells:', len(d['raw_data']['train_cells']))
print('test cells :', len(d['raw_data']['test_cells']))
print('first train:', d['raw_data']['train_cells'][0])" cache/battery_cache_<hash>.pkl
```
