# Fig. 3 standalone panels (`scripts/paper_figures`)

Scripts that turn **existing BatteryML prediction artifacts** into separate,
editable Fig. 3 panels. Each panel is its own SVG, meant to be placed by hand
in Illustrator / PowerPoint. These scripts never train, re-run inference,
download data, delete runs or change a prediction file. They also never draw a
combined A–E figure.

Every panel writes four files into the output directory:

| File | Content |
| --- | --- |
| `<stem>.svg` | vector panel; text stays editable (`svg.fonttype = none`, DejaVu Sans, no embedded fonts or bitmaps) |
| `<stem>.png` | preview (300 dpi by default) |
| `<stem>_source_data.csv` | the exact numbers drawn, one row per mark |
| `<stem>_metadata.json` | sources (run_id, artifact path, SHA-256), metric and unit, aggregation, error-bar definition, filters, missing/non-finite handling, the command used, and the code version |

Existing files are never overwritten. A rerun into the same directory stops
with an error, so use a new directory each time.

## 1. List candidate runs (read-only)

```bash
python -m scripts.paper_figures.collect_fig3_data list-runs --runs-root runs \
    --output runs/paper_figures/fig3_candidates.csv
```

This prints every `predictions_seed_*.pkl` sorted by path, with its run_id,
model, task, protocol (from the split class in the config copy), seed and smoke
status (from `run.json`). Nothing is unpickled and nothing is chosen: rows are
sorted by path, never by date or error.

## 2. Write the selection

Copy `fig3_selection.example.json` to `fig3_selection.json` and list the exact
artifacts:

- **`model_order`** fixes each model's colour (validated 8-slot palette) and
  legend position in **every** panel. More than eight models is refused.
- **`runs[]`** needs `id`, `model` and `artifact`. `dataset` and `protocol` are
  optional; when given, they are cross-checked against `run.json` and the
  config copy, and a contradiction is refused.
- **Different hyperparameter configurations of the same model** must have
  different `model` names. They are never averaged together.
- **`published_reference.include: true`** adds the BatteryML README RMSE table
  as a separate source.
- **`latency_csv`** and **`radar.metrics_csv` / `radar.config`** point to
  measured inputs. See the templates below.

## 3. Collect, then draw

```bash
python -m scripts.paper_figures.collect_fig3_data collect \
    --selection scripts/paper_figures/fig3_selection.json \
    --output runs/paper_figures/fig3_data_v1

# all panels the data supports (separate files + fig3_status.md)
python -m scripts.paper_figures.plot_fig3_all \
    --data runs/paper_figures/fig3_data_v1 --output runs/paper_figures/fig3_v1

# or one panel at a time
python -m scripts.paper_figures.fig3a_lifetime_heatmap --data ... --output ... [--metric MAE] [--source published]
python -m scripts.paper_figures.fig3b_soh_error        --data ... --output ... [--scale pp] [--metric cell-macro]
python -m scripts.paper_figures.fig3c_multicriteria_radar --data ... --output ...
python -m scripts.paper_figures.fig3d_inference_latency   --data ... --output ... [--yscale log]
python -m scripts.paper_figures.fig3e_lifetime_error_distribution --data ... --output ... \
    [--dataset MATR --protocol MATR1] [--seed 0] [--yscale log]
```

Every panel accepts `--width --height --font-size --line-width --dpi` and
`--allow-smoke`.

**How `collect` reads the data.** `collect` reads each artifact through the
platform's existing loader (`batteryml_web.agent.results.load_prediction_payload`)
and builds the same view the website uses (`batteryml_web.agent.result_views`):

- the recorded label transformation is inverted **once**;
- one finite mask is applied, and dropped pairs are counted in `omitted_nonfinite`;
- cell identity is filtered by the same mask.

The saved BatteryML scores are copied verbatim.

**Trusted artifacts only.** The loader only remaps CUDA tensors to CPU; it does
not restrict what a pickle may contain. Only point the selection at artifacts
produced by your own runs.

## Panels

| Panel | Reads | Value drawn | Refused when |
| --- | --- | --- | --- |
| A `fig3a_lifetime_heatmap` | `runs.csv` (task_kind = lifetime) | saved RMSE (default) or MAE in cycles; mean ± sample SD (n−1) over seeds of one configuration; "1 run" otherwise; NA for no run | mixed configurations, repeated or unrecorded seeds, or a column that mixes label annotator / unit / split / weighting |
| A `…_published_reference` | `published_rmse.csv` | README RMSE; ">1000" kept as censored | MAE requested (the table has RMSE only) |
| B `fig3b_soh_error` | `runs.csv` (task_kind = soh) | cycle-pooled SOH MAE (saved score); `--scale pp` = ×100; `--metric cell-macro` is a separate statistic and file | same as A |
| C `fig3c_multicriteria_radar` | `radar_metrics.csv` + `radar_config.json` | score = (v−lo)/(hi−lo), inverted for lower_is_better, with fixed bounds; no composite score | fewer than 3 dimensions, a missing value, bad or missing bounds, a value outside the bounds, a subjective dimension, or a unit mismatch |
| D `fig3d_inference_latency` | `latency.csv` | measured inference latency; SD only for mean with repeats ≥ 2 | training/CLI/load time, mixed hardware or timing protocol, mixed units |
| E `fig3e_lifetime_error_distribution` | `samples.csv` | \|predicted − observed\| per test cell, in cycles | several protocols without `--dataset/--protocol`, several seeds without `--seed`, or different cell-ID sets |

**Target naming.** The whole-cell `RULLabelAnnotator` target counts cycles to
end of life from the first cycle. Panels A and E therefore call it **battery
lifetime (cycle life)**, not remaining useful life.

**Smoke runs.** Smoke status comes from `run.json`, never from a directory name.

- By default, smoke runs are excluded.
- With `--allow-smoke`, the panel carries the line *"workflow-validation example
  (includes smoke runs; not benchmark performance)"* and the metadata records it.

**Status table.** `plot_fig3_all` writes `fig3_status.md` with one line per
panel: `| Panel | CREATED / MISSING_DATA / INSUFFICIENT_DATA / INVALID_COMPARISON | Output / Reason |`.
A panel the data cannot support is listed with its reason and not drawn.

## Templates

- `fig3_selection.example.json`: selection skeleton. Its paths are
  placeholders, and `collect` refuses them.
- `fig3c_radar_config.template.json`: dimension definitions. Its `null`
  bounds are refused until you fix them in advance.
- `fig3c_radar_metrics.template.csv`: `model,dimension,value,unit,source,n_repeats`.
- `fig3d_latency.template.csv`: one row per model. Every row must record the
  same device, hardware, batch size, threads, warm-up, timing boundary,
  inclusions and GPU sync. `measurement_kind` must be `inference`.

## Tests

```bash
python -m unittest tests.test_paper_figures -v
```

The tests use synthetic artifacts only (`synthetic_` cell ids in a temporary
directory, real BatteryML label transformations). No training and no dataset
access.
