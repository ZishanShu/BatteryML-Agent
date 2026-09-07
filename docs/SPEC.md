# `docs/SPEC.md` — the standard

`CLAUDE.md` §2 names this file as the single source of truth for what "done"
means, and cites stable requirement IDs throughout. **It did not exist**; that
is priority 3 in `CLAUDE.md` §11. This is the minimum restoration: it establishes
the ID scheme, records the requirements that today's code actually implements,
and gives later work somewhere to add to. It does **not** attempt to
reverse-engineer a full specification for the whole platform — inventing
requirements to match existing code would produce a document that always says
everything is done.

## How to use it

* Work is scoped by ID. Before writing code, state which IDs you are
  implementing.
* After writing code, set each ID's status and name the implementing file.
* Only the user may set `VERIFIED`, and only after pasting back real output from
  the server (`CLAUDE.md` §0).
* A requirement that is missing gets a new ID and a rationale, never a silent
  implementation.
* A requirement that turns out to be wrong gets an amendment proposal, never a
  quiet substitution.

## Status values

| Status | Meaning |
| --- | --- |
| `NOT-STARTED` | No implementation exists. |
| `PARTIAL` | Some of it exists; the gap is stated in the row. |
| `IMPLEMENTED-STATIC` | Implemented and established by reading the code. Never run against real data by the author. |
| `BLOCKED` | Cannot proceed; the blocker is stated. |
| `VERIFIED` | The user ran it on the server and pasted back the output. **Only the user sets this.** |

## Two ID namespaces, deliberately separate

| Namespace | What it identifies | Where it lives |
| --- | --- | --- |
| `VIZ-nn`, and future feature prefixes | a **requirement** — something the platform must do | this file |
| `ENV-01`, `DATA-01`, `CFG-01`, `VAL-01`, `SPLIT-01`, `REG-01`, `EXT-01`, `CLI-01`, `FIG-01` | a **check** — something the verification harness establishes | `docs/VERIFY.md` |

They are the same shape, so keep the prefixes disjoint: the figure *requirements*
are `VIZ-01`…, and the figure *check* that verifies them is `FIG-01`. Nothing is
both.

---

## VIZ — visualisation

Rationale for the group: the protocol's outputs are numbers, and a reviewer
cannot judge a battery-ageing prediction from three aggregate metrics alone. A
parity plot shows whether the error is uniform or concentrated in the long-lived
cells; a per-cell error chart shows whether one outlier carries the RMSE. The
`figures/` directory is already part of the reproducibility contract
(`CLAUDE.md` §8) and, before this group, nothing wrote into it.

| ID | Requirement | Status | Implementation |
| --- | --- | --- | --- |
| `VIZ-01` | Figure filenames are declared in one place, are safe and unique, and are treated as part of the manuscript: adding one is a code change, renaming one is a code **and** paper change. Every generated figure is recorded in `figure_manifest.json` with the artifact it came from, that artifact's SHA-256, the git commit, the seed, the label space and unit, and the figure's own SHA-256. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/contract.py`, `batteryml_web/viz/render.py` |
| `VIZ-02` | Predictions are plotted in the label space the recorded metrics were computed in. The label transformation carried by the artifact is inverted, and the result is proved correct by recomputing the recorded metrics with BatteryML's own evaluator and refusing to draw anything if they disagree. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/loaders.py` (`load_prediction_artifact`, `verify_recorded_scores`) |
| `VIZ-03` | A per-cell figure names the right cell. The NaN-label mask `Task.build` applies is reproduced rather than assumed, and the recomputed labels are cross-checked against the artifact's own. Where identity cannot be established the figure is skipped with a stated reason and never drawn against guessed labels. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/loaders.py` (`resolve_test_cell_ids`) |
| `VIZ-04` | Result figures — parity, residuals, per-cell absolute error, and a metric summary — are generated offline from a prediction artifact, with no API key, no Streamlit and no network (Route B). Figures never compute an aggregate metric; they display the evaluator's values verbatim. | `IMPLEMENTED-STATIC` | `batteryml_web/viz/figures.py`, `batteryml_web/viz/cli.py` |
| `VIZ-05` | Data-side figures — capacity fade, cycle-life distribution, train/test split balance, single-cell detail, feature matrix — generated from `data/processed/<DATASET>` before any training. | `NOT-STARTED` | — |
| `VIZ-06` | A Streamlit page displays the generated figures, and offers an interactive layer for exploring individual cells. The page degrades to the static figures when the interactive library is absent, so Route B never depends on it. | `NOT-STARTED` | — |
| `VIZ-07` | `report.html` assembles the run's figures, metrics and provenance into the single reviewer-facing artifact `CLAUDE.md` §8 names. Nothing writes it today. | `NOT-STARTED` | — |
| `VIZ-08` | Cross-model comparison over several runs of one dataset, read from their manifests. Introduces the first categorical palette, which must be validated before it ships. | `NOT-STARTED` | — |

### Notes on the VIZ group

* `VIZ-02` and `VIZ-03` exist as separate requirements because they are separate
  failure modes with the same symptom — a figure that looks entirely reasonable
  and is wrong. `VIZ-02` puts the axes in the wrong space; `VIZ-03` puts the
  right numbers under the wrong names.
* The check that establishes `VIZ-01`…`VIZ-04` is `FIG-01`
  (`scripts/verify/check_figure_contract.py`).
* No requirement in this group may be marked `VERIFIED` from this environment:
  it has no numpy, no matplotlib and no torch, so no figure has ever been drawn
  here. The command block that would verify them is in `docs/VISUALIZATION.md`.

---

## Amendments

Record here any requirement found to be wrong or impossible, with the proposed
change, rather than implementing something different.

*(none yet)*
