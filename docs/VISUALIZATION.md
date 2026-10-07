# `batteryml_web/viz/` — result figures

Implements `VIZ-01` … `VIZ-04` (`docs/SPEC.md`). Verified by `FIG-01`
(`scripts/verify/check_figure_contract.py`).

Nothing in this package trains, evaluates, or produces a metric. It reads a
prediction artifact BatteryML already wrote and draws it.

---

## What it produces

Given one evaluated run, into the directory you pass as `--output-dir`:

```
figures/
  figure_manifest.json
  results/
    parity_<DATASET>_<MODEL>_seed<SEED>.png
    residuals_<DATASET>_<MODEL>_seed<SEED>.png
    error_by_cell_<DATASET>_<MODEL>_seed<SEED>.png
    metrics_summary_<DATASET>_<MODEL>_seed<SEED>.png
```

**These filenames are part of the manuscript** (`CLAUDE.md` §8). They are
declared once, in `batteryml_web/viz/contract.py`, and every producer and
checker reads them from there. Renaming one is a code change *and* a paper
change; `FIG-01` fails if two figures collide or if a name stops matching the
contracted pattern.

Point `--output-dir` at `<workspace>/figures` and the existing collection step
picks the figures up: `batteryml_web/app.py:302-311` already copies a
`figures/` subdirectory out of the workspace into
`runs/web_projects/<project>/figures/`. No change to the frozen `app.py` is
needed for the figures to reach a project directory.

| Figure | Form | Why this form |
| --- | --- | --- |
| `parity` | scatter on a shared, equal-aspect scale with the identity line and a ±10 % band | The reader's job is to see the departure from `y = x`. Unequal axes would misstate its size. |
| `residuals` | signed error against the measured value, zero baseline | Separates a constant offset from a fan that widens with cycle life — which a parity plot cannot show. |
| `error_by_cell` | horizontal bars, sorted worst first, sequential colour | Shows whether one outlier carries the RMSE. Horizontal because a test split runs to dozens of named cells. |
| `metrics_summary` | a row of stat tiles | Deliberately **not** a bar chart: RMSE and MAE are in cycles while MAPE is a dimensionless ratio, so one shared scale would misstate their relative size. |

---

## The three things this package exists to get right

Each is a way to produce a figure that looks entirely reasonable and is wrong.

### 1. The label space (`VIZ-02`)

`predictions_seed_*.pkl` stores the prediction and the test labels in the
**transformed** space. The demo config stacks `LogScale → ZScore`
(`configs/baselines/sklearn/xgb/matr_1.yaml:16-21`), so those tensors are
z-scores of order 1 — while the recorded RMSE is a number of cycles of order
100, because `DataBundle.evaluate` inverse-transforms before scoring
(`batteryml/data/databundle.py:88-92`).

Plotting the raw tensors gives axes running roughly −2 … 2 that silently
disagree with `metrics.csv`.

`load_prediction_artifact` inverse-transforms both arrays using the
transformation the artifact itself carries, then calls BatteryML's own evaluator
on the result and compares against the recorded `scores`. A disagreement beyond
`--score-tolerance` (default `1e-4` relative) **aborts**: no figure is written.
The comparison is recorded in the manifest under `score_consistency_check`.

### 2. Cell alignment (`VIZ-03`)

`Task.build` drops cells whose label is NaN from the feature and label tensors
(`batteryml/task.py:62-68`) but keeps the full cell list on the task
(`:53-54`). So `test_cells[i]` is **not** row `i` of `test_data.label` whenever
any cell was dropped, and a per-cell chart built on that assumption mislabels
every bar.

`resolve_test_cell_ids` never assumes. It re-derives the mask by rebuilding the
config's label annotator, applies it, and then cross-checks: the labels
recomputed from the cells must match the artifact's own labels. Where they do
not, or where identity cannot be established at all, `error_by_cell` is
**skipped with a stated reason** and the reason is written into the manifest's
`skipped` array.

For MATR + RUL the mask is expected to keep every cell (`min_rul_limit=100`,
`batteryml/label/rul.py:34-36`), so the two counts should agree — but "expected"
is why the check exists rather than why it is skipped.

### 3. Cell identity is not always available (`VIZ-03`)

| Writer | Contents | Cell IDs |
| --- | --- | --- |
| `Pipeline.evaluate` (`batteryml/pipeline.py:113-122`) | `prediction`, `scores`, `data`, `seed` | **no** |
| `manual_train_eval.py` (`scripts/reproduce_batteryml/manual_train_eval.py:85-88`) | the same **plus** `raw_data{train_cells, test_cells}` | yes |

So for the normal CLI path, identity comes from rebuilding the splitter the
config names — deterministic, and it reads only filenames
(`batteryml/train_test_split/MATR_split.py:15-22`). That needs `--config`.
Without it the three other figures are still drawn; only `error_by_cell` is
skipped.

---

## Running it

### Route B, offline, no API key

```bash
# 1. Train and evaluate (CLAUDE.md §5 — config is positional, --train --eval are mandatory)
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace runs/native_matr_xgb_test \
  --train --eval --device cpu --seed 0

# 2. Draw the figures from the artifact that produced
python -m batteryml_web.viz.cli results \
  --prediction "$(ls -t runs/native_matr_xgb_test/predictions_seed_0_*.pkl | head -1)" \
  --config configs/baselines/sklearn/xgb/matr_1.yaml \
  --output-dir runs/native_matr_xgb_test/figures

# 3. Verify the contract and the manifest
python -m scripts.verify.run_all --only FIG-01 \
  --output-dir runs/native_matr_xgb_test/figures
```

Step 1 prints the workspace path and exits without `--train --eval`. If it says
`Skip evaluation for ... as the prediction exists`, an artifact from an earlier
run is already there and step 2 will use it — that is fine, and the seed in the
filename says which run it was.

### Options worth knowing

| Flag | Effect |
| --- | --- |
| `--format {png,svg,pdf}` | `png` at `--dpi 300` for the UI and for raster figures; `svg` or `pdf` for a vector figure in the manuscript. |
| `--theme {light,dark}` | The surface the figure is drawn against. Dark is a selected palette, not an inverted one. Manuscript figures are `light`. |
| `--verify-cell-alignment` | Load the test cells from disk to prove the per-cell mapping even when the counts already imply it. Costs one pass over the split's `.pkl` files. |
| `--band 0` | Drop the parity tolerance band. |
| `--strict` | Exit 1 if any contracted figure was skipped. Use it in a pipeline that must produce all four. |
| `--score-tolerance` | Raise only when you have established that a mismatch is float32 rounding. Raising it to hide a real mismatch defeats `VIZ-02`. |

---

## Determinism

`CLAUDE.md` §8 requires identical inputs to produce identical artifacts.

* Matplotlib stamps a creation date into SVG and PDF output and salts generated
  SVG element ids. Both are suppressed by `render._deterministic_output`, which
  sets `SOURCE_DATE_EPOCH=0` and `rcParams["svg.hashsalt"]` around the save and
  restores both afterwards. `SOURCE_DATE_EPOCH` is matplotlib's documented
  switch for reproducible output; it is used in preference to passing
  `Date: None` through `metadata`, because whether the backends read an explicit
  `None` as "omit" or as a literal has varied between versions and this
  environment cannot test which. The only `metadata` passed is a constant
  `Creator` / `Software` string, so a matplotlib upgrade does not change the
  bytes for that reason alone.
* Figures use matplotlib's bundled DejaVu Sans rather than a system font, which
  would resolve differently on every machine.
* `figure_manifest.json` carries **no** wall-clock timestamp, deliberately — one
  would make every rerun differ and destroy the check below. Run timing belongs
  in `run_metadata.json`.

To check it on the server, generate twice and compare:

```bash
python -m batteryml_web.viz.cli results \
  --prediction "$(ls -t runs/native_matr_xgb_test/predictions_seed_0_*.pkl | head -1)" \
  --config configs/baselines/sklearn/xgb/matr_1.yaml \
  --output-dir /tmp/figs_a

python -m batteryml_web.viz.cli results \
  --prediction "$(ls -t runs/native_matr_xgb_test/predictions_seed_0_*.pkl | head -1)" \
  --config configs/baselines/sklearn/xgb/matr_1.yaml \
  --output-dir /tmp/figs_b

diff <(cd /tmp/figs_a && sha256sum results/*.png) \
     <(cd /tmp/figs_b && sha256sum results/*.png) && echo "byte-identical"
```

`FIG-01` performs the same comparison in memory on every run, so a regression is
caught by the harness rather than by a reviewer.

---

## Adding a figure

1. Add a builder to `figures.py`. Pure: arrays in, `Figure` out, no I/O, no
   `pyplot`, no `rcParams`.
2. Add a `FigureSpec` to `contract.py`. That fixes its filename — decide it once,
   because the paper will quote it.
3. Add a dispatch branch in `cli.py`. `FIG-01` reads the branches with `ast` and
   fails if a declared figure has none.
4. Add a smoke case to `_smoke_render` in `scripts/verify/check_figure_contract.py`,
   or `FIG-01` reports that the figure ships unexercised.
5. Add a row to `docs/SPEC.md` and to the table above.

### Colour

Single hue plus neutrals, from the validated reference palette: blue
categorical slot 1 for marks, its sequential ramp for magnitude, and the
chart-chrome neutrals for grid, axes and text. There is no categorical
adjacency to validate because no figure here puts two categorical series on one
axis.

`VIZ-08` (cross-model comparison) will be the first that does. Run the palette
validator on the hue order before it ships; do not eyeball it.

---

## What is not established from this environment

`CLAUDE.md` §0: this environment has no numpy, no matplotlib and no torch, so
**no figure in this package has ever been rendered here.** Specifically:

* `[STATIC]` — the contract, the loader logic, the trap handling, and every
  matplotlib call, established by reading the code against the matplotlib and
  BatteryML sources cited above.
* `[VERIFIED]` — only these, observed in the authoring session: the filename
  contract renders and matches its pattern; `slugify` strips a path traversal;
  the contract and the CLI dispatch agree; `FIG-01` reports SKIP with the
  missing package named; the CLI's `--help` works and its
  missing-dependency path exits 1 with an actionable message.
* `[ASSUMED]` — that matplotlib PNG output is byte-identical across two renders
  in one process. `FIG-01` is exactly the check that settles it; it has not been
  run anywhere with matplotlib installed.
* `[ASSUMED]` — the layout: margins, the equal-aspect parity box, and the
  per-cell figure's height scaling were chosen by calculation, not by looking at
  a rendered figure. Open the four PNGs and check for clipped labels.

---

## Website result figures and `scripts/export_result_figures.py`

Implements `VIZ-09` … `VIZ-14` (`docs/SPEC.md`). Separate from the offline
`viz.cli results` route above: that one draws the four contracted RUL figures
from an artifact plus its config; this one is what Review Results, the
Predictions tab and the Agent Workflow page show, and its export.

### One data path

```
selected run (run.json) ─► exact artifact path
  ─► agent/results.load_prediction_payload        (CPU-safe unpickler, the only reader)
  ─► agent/result_views.build_rul_view / build_soh_view
        _invert: recorded label_transformation.inverse_transform, applied once;
                 status inverted / none_recorded / metadata_missing / inverse_failed
                 (the last two refuse a physical-unit view instead of guessing)
        _paired_series: one finite mask over observed AND predicted; omitted rows counted
        target_spec: quantity + unit from the recorded label annotator
  ─► viz/result_figures.build_panels              (panel data + matplotlib Figure)
        ├─► ui/run_history.render_*_figures      st.pyplot — the page
        └─► viz/result_export.build_export       SVG/PNG/CSV/JSON/MD — the export
```

`build_panels` returns the figure *and* the exact arrays given to its builder,
so `plotted_samples.csv` is what was drawn, not a recomputation.

### What each panel means

| Panel | x | y | One point/line | Units |
| --- | --- | --- | --- | --- |
| `soh_cycle_<cell>` | recorded `cycle_index` | observed (solid) and estimated (dashed) SOH | one recorded cycle of **one** cell, in recorded cycle order | ratio (capacity / reference capacity) |
| `parity` | observed target | predicted target | one stored row: a cell (lifetime) or a cell × cycle (SOH) | lifetime: cycles; SOH: ratio |
| `residuals` | observed target | predicted − observed | as parity | same unit; SOH ×100 = percentage points |
| `*_zoom` | as above | as above | same points; viewport only | as above |

* **Lifetime, not RUL.** `RULLabelAnnotator` counts cycles from the first
  recorded cycle to end of life, so a whole-cell value is the cell's cycle
  life. The axis says "Battery lifetime (cycles)". Only a genuine cycle-wise
  RUL artifact is called remaining useful life.
* **Default SOH cell** = first cell ID in ascending text order among the
  plotted samples. Independent of error; recorded in every export. Another
  real cell can be chosen in the page or with `--cell-id`.
* **Zoom** = 1st–99th percentile of the plotted values, padded 5 %. It changes
  only the visible range; the count of points outside is printed on the
  figure and recorded. The full-range view is the default.
* **Per-cell summary** = each cell's mean absolute and RMS residual over its
  plotted samples: a display diagnostic. It weights cells equally; the saved
  scores weight samples equally. Neither replaces the other.
* **SOH > 1.0** is shown as stored and counted; it is not called invalid.
  Below 0 is counted as not physical.
* **Non-finite rows** are dropped from the figures through one shared mask and
  counted; the saved metrics are never recomputed on the reduced set.

### Export command

```bash
# read-only: schema, counts, cell IDs, transformation, smoke evidence
python -m scripts.export_result_figures --artifact <EXACT.pkl> --inspect
python -m scripts.export_result_figures --artifact <EXACT.pkl> --list-cells

# write the export into a NEW directory (runs/ is git-ignored)
python -m scripts.export_result_figures \
  --artifact <EXACT.pkl> \
  --output runs/figure_exports/<NEW_DIRECTORY> \
  [--cell-id <REAL_CELL_ID>] [--dpi 300]
```

Writes `parity`, `parity_zoom`, `residuals`, `residuals_zoom` (and, for
cycle-level SOH, `soh_cycle_<cell>`, `soh_cycle_<cell>_zoom`) as `.svg` +
`.png`; `plotted_samples.csv`; `soh_cycle_<cell>.csv` and `per_cell_errors.csv`
for SOH; `figure_sources.json`; `figure_notes.md`. It refuses an output
directory that exists and is not empty, an unknown cell ID, and any artifact
that is not a scalar-per-sample result. The page's **Export figures** expander
shows the same command and offers the same files as a zip.

`figure_sources.json` records the artifact path and SHA-256, everything
`run.json` and the `config_<ts>.yaml` copy beside it record (run id, model,
split, seed, smoke flag, epochs override), the transformation chain and its
fitted parameters, units, counts, the selected cell and rule, every panel's
limits and outside counts, the saved metrics verbatim, the export command,
the git commit (with an uncommitted-changes flag) and library versions. No
environment variables are read.

**Provenance trap, recorded not fixed:** `Pipeline.train` copies the
*original* config before applying `--epochs`, so a 1-epoch smoke run's config
copy still shows the full epoch count. Only `run.json` records the override.
A legacy CLI workspace has no `run.json`; its smoke status is "not recorded",
and a directory named `..._smoke_...` is not taken as evidence.

### Verification status

* `[VERIFIED]` here, synthetic fixtures only: `tests/test_result_figures.py`
  (28 tests, two of them proved non-vacuous by mutation), the export CLI on
  pickled fixtures built with real BatteryML LogScale/ZScore transformations,
  and a Chromium check of Review Results with two synthetic runs (desktop and
  720 px).
* Not inspected here: any real artifact. The MATR Ridge SOH and MATR CNN
  smoke artifacts live only on the server; use `--inspect` there first.
* Layout change shared with the offline route: `style.add_titles` now pads the
  title 18 pt above a subtitle (was 14 pt, which let them touch).
* The offline `viz/loaders.py` still unpickles with plain `pickle.load`, not
  the CPU-safe loader; a CUDA-origin artifact needs a GPU there. Not changed.
