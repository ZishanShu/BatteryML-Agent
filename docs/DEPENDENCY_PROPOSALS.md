# Dependency proposals

`CLAUDE.md` §6: *"No unpinned new dependencies. If a new package is genuinely
required, justify it and pin it; prefer the standard library."*

One section per proposed dependency, each ending in a recommendation. **Nothing
here has been added.** Each needs your decision first, and — see the blocker
immediately below — the package must be installed on the server *before* the
model module lands in the repository.

---

## Blocker that applies to both proposals

The extension point fails loudly by design. `register_extension_models()`
raises `ExtensionRegistrationError` when a declared module cannot be imported,
and `run_batteryml_cli.py` turns that into `exit 2`
(`batteryml_web/models/__init__.py`, `batteryml_web/core/run_batteryml_cli.py`).

So committing `batteryml_web/models/lightgbm_model.py` while `lightgbm` is not
installed **breaks every CLI invocation**, including `preprocess`, including
`--help`, and including Route B — the offline reproduction path that
`CLAUDE.md` §1 says must never break. It is not a soft failure confined to that
one model.

That gives a required order:

1. you install the package on the server and paste back the version;
2. it is pinned in `requirements.txt`;
3. only then does the model module land.

There is no way to do step 3 first without risking Route B.

### The design gap this exposes

The extension point has no notion of an *optional* model. Every declared module
is a hard requirement. If either package is added, it is worth deciding whether
to also add an `OPTIONAL_EXTENSION_MODULES` tuple whose import failure
downgrades to a warning, so a reviewer without LightGBM installed still gets a
working Route B and simply does not see that model.

I have not implemented that — strictness was a deliberate choice
(`CLAUDE.md` §6, "fail loudly, never silently"), and relaxing it is a decision
about what the platform guarantees, not a detail. Flagging it because it is the
difference between "one more baseline" and "one more thing a reviewer can fail
to install".

---

## Scale of the problem these would be applied to

Both proposals are gradient-boosting libraries, and both are usually justified
by performance on large datasets. That is not this problem. `[STATIC]`

| Quantity | Value | Source |
| --- | --- | --- |
| Training cells, MATR primary split | 41 | `batteryml/train_test_split/MATR_split.py:33-42` |
| Training cells, largest preset (`mix`) | order 10^2 | `MIX100_split.py`, 137 held-out ids out of the six pooled datasets |
| Feature dimension `D` | 1000 | `VoltageCapacityMatrixFeatureExtractor(interp_dim=1000)`, and the configs set `cycles_to_keep: 98` — an `int`, folded to `[98]` at `voltage_capacity_matrix.py:34-35` and applied at `:60-61`, so exactly one cycle is kept |

So: `N ≈ 41`, `D = 1000`. Wide and very short. Keep that in mind below — it is
what makes both cases weaker than they first look.

`[ASSUMED]` for the `use_precalculated_qdlin: True` path: `D` is then the
length of the stored `Qdlin` array rather than `interp_dim`. Confirm from a real
run if it matters.

---

## LightGBM

### What it would add

Leaf-wise tree growth, histogram-based split finding, native categorical
handling, and multi-threaded training that is typically several times faster
than XGBoost's.

### The case against

**XGBoost is already a dependency** and already occupies the
gradient-boosted-trees slot in the grid (`configs/baselines/sklearn/xgb/`,
8 presets). Adding LightGBM adds a second member of the same family, not a new
family — unlike, say, the robust regressor or the instance-based baseline added
today, each of which answers a question nothing else in the grid answers.

Every advantage listed above is a large-data advantage. At `N = 41` there is no
speed problem to solve: XGBoost fits these splits in well under a second.
`[ASSUMED]` — plausible from the data scale, not measured here.

Worse, LightGBM's defaults are actively wrong at this scale.
`min_child_samples = 20` means a split is only made when both children keep at
least 20 samples; with 41 training rows that permits essentially one split
before the constraint binds, so the default model is close to a stump. Making
it useful requires setting `min_child_samples`, `num_leaves` and probably
`min_split_gain` — i.e. hyperparameter choices that determine the reported
numbers, which is exactly the kind of decision that should not be smuggled in
alongside "add a library".

### Install cost

Small — a manylinux wheel of a few MB, no build step on standard platforms.
Low risk for a reviewer following the protocol.

### Recommendation

**Skip**, unless you specifically want a second gradient-boosting entry for the
paper's comparison table. The scientific gap it fills is already filled, and it
cannot be added honestly without also making tuning decisions.

If you do want it: `LGBMRegressor` wraps exactly like the five added today, and
its configs would live in `configs/baselines/sklearn/lightgbm/`, following the
precedent that `xgb/` sits under `sklearn/` despite not being scikit-learn.

```
# requirements.txt
lightgbm       # PIN: fill from environment_check.json -> packages[lightgbm].installed
```

---

## CatBoost

### What it would add

Ordered boosting, symmetric (oblivious) trees, and defaults tuned to work
without search.

### The case for

Stronger than LightGBM's, and for a reason specific to this problem. Ordered
boosting exists to reduce the target leakage that ordinary gradient boosting
suffers when each tree is fitted on residuals computed from the same rows it
was grown on. That bias is worst when `N` is small — which is precisely the
regime here (`N ≈ 41`). It is one of the few boosting variants whose stated
motivation matches this dataset's shape rather than fighting it.

Its defaults are also far more usable at small `N` than LightGBM's, so it can
be added without the tuning decisions that make LightGBM awkward.

### The case against

The wheel is large — on the order of 100 MB, against a few MB for LightGBM.
For a Nature Protocols submission promising setup-to-first-prediction in 2–4
hours on a laptop, every dependency is an install-failure surface for a
reviewer, and this is the heaviest one on the table.

It also still competes with XGBoost for the same slot in the grid, even if it
brings a genuinely different algorithm to it.

### Install cost

High relative to everything else in `requirements.txt`. Wheels exist for the
usual platforms, so no build step, but the download dominates a fresh install.

### Recommendation

**Defensible, and better than LightGBM if you want exactly one of them** — the
ordered-boosting argument is real at this data scale, not borrowed from a
large-data setting.

Conditional on one thing: it should be **optional**, not a Route B requirement.
That means resolving the `OPTIONAL_EXTENSION_MODULES` design gap above first,
so a reviewer without CatBoost still gets a working offline reproduction. Adding
it as a hard dependency would trade a reproduction path the paper depends on for
one more baseline, which is the wrong trade.

```
# requirements.txt (or a requirements-optional.txt, per the decision above)
catboost       # PIN: fill from environment_check.json -> packages[catboost].installed
```

---

## Summary

| Package | Fills a gap in the grid? | Suits `N ≈ 41`? | Install cost | Recommendation |
| --- | --- | --- | --- | --- |
| lightgbm | No — XGBoost holds that slot | No; defaults near-degenerate, needs tuning | Low | Skip |
| catboost | Same slot, genuinely different algorithm | Yes — ordered boosting targets small `N` | High (~100 MB) | Add only as an optional dependency |

If you want neither, nothing needs doing: the five models added today are pure
scikit-learn and introduce no new dependency at all.

### If you decide to proceed

Send me, for each package you want:

1. the output of `pip show <package> | head -2` on the server, so it can be
   pinned to what you actually have rather than a guessed number;
2. your decision on `OPTIONAL_EXTENSION_MODULES` — required, or optional with a
   downgraded failure.

Then the model module, its eight configs, and the registry entry follow the
same four steps as the five added today.
