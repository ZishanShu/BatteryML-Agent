# `scripts/verify/` — the verification harness

Independent checks that inspect the repository, the environment and the
filesystem. They are the cheap half of the verification loop described in
`CLAUDE.md` §0: they establish what can be established without running the
pipeline, so that the expensive half — the user running the real thing on the
server — starts from a known-good baseline.

**What these checks never do:** train, evaluate, download, install, or write
anything under `data/`. Nothing they report is a claim about pipeline behaviour
on real data.

For diagnosing an import or environment failure rather than running the
checks, see `docs/DEBUGGING.md`.

## Running

```bash
# everything
python -m scripts.verify.run_all

# one check, by id or by name
python -m scripts.verify.run_all --only CFG-01
python -m scripts.verify.run_all --only config_validity

# several
python -m scripts.verify.run_all --only ENV-01 --only DATA-01

# list what exists, run nothing
python -m scripts.verify.run_all --list

# write artifacts into a project workspace instead of runs/verify/
python -m scripts.verify.run_all --output-dir runs/web_projects/<project_name>
```

Each check is also runnable on its own:

```bash
python -m scripts.verify.check_environment
python -m scripts.verify.check_dataset_status --output-dir runs/verify
```

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | No check returned FAIL (PASS and SKIP only). |
| 1 | At least one check returned FAIL. |
| 2 | `--only` matched no check — a usage error, not a result. |

`SKIP` does not fail the run: a check that cannot observe anything because the
data is absent has not found a defect. Pass `--strict` to treat SKIP as failure
where the data is expected to be present (i.e. on the server, after
preprocessing).

## The checks

| ID | Name | What it establishes | Artifacts |
| --- | --- | --- | --- |
| `ENV-01` | `environment` | Python version against the minimum; every distribution declared in `requirements.txt`, installed or not, with the installed version; `batteryml` importable; GPU visible via torch and `nvidia-smi`. | `environment_check.json`, `package_versions.txt` |
| `CLI-01` | `cli_contract` | That `run_batteryml_cli.py run` still takes the config **positionally** and still offers `--train --eval --device --seed --workspace`, and offers no `--config`. Inspects `--help` only. Also, read from source with `ast`: the wrapper calls `install_safe_download()` before `runpy.run_path`, and `safe_download.download_file` takes upstream's parameters (spec `ACQ-01`). | — |
| `DATA-01` | `dataset_status` | Per dataset: raw present, file count and size; processed present and non-empty; processed older than raw. | `dataset_status.csv`, `dataset_status.json` |
| `CFG-01` | `config_validity` | Every config under `configs/` parses, declares `model` / `train_test_split` / `feature` / `label` each with a `name`, and names only registered implementations. Declared `cell_data_path`s are checked for existence. | `config_validity.json` |
| `VAL-01` | `validator_contract` | Runs the real `config_validator.validate_config` over every run config and reports which the execution gate would refuse. Structure only (`require_data=False`), so it needs no data and writes nothing. | `validator_contract.json` |
| `SPLIT-01` | `split_requirements` | For each of the eight presets, which processed datasets it requires and which are missing — derived from two independent sources and cross-checked. | `split_requirements.json` |
| `REG-01` | `registry_integrity` | Every registered name resolves and is unique; the `batteryml_web/models/` extension point is self-consistent; every `input_type` the CLI advertises has a preprocessor; per-model origin, task and device support. | `registry_integrity.json` |
| `DEV-01` | `runtime_contract` | The run-time contract the web layer depends on: device enumeration goes through an argv list with `shell=False` and derives the GPU count from `nvidia-smi` rather than a constant; a smoke run is exactly `--epochs 1` on the model's own formal config, with no config written or mutated; BatLiNet's smoke run recommends CPU; a device string is a single torch device (`cuda:0,1` is refused), a multi-GPU *selection* produces no device string and blocks Run, no model is registered as multi-GPU capable, and no DataParallel/DDP/torchrun path is exposed anywhere in the web layer; a CUDA OOM is recognised, never retried, and yields no metric; the streaming executor stays Streamlit-independent, recognises the real tqdm lines the models emit, and fabricates no percentage. The machine's real GPU snapshot is recorded as evidence — its absence is never a failure. | `runtime_contract.json` |
| `INTENT-01` | `intent_catalog` | That the catalog handed to the LLM is derived from `ConfigRegistry`, the platform dataset list and `readiness.EXTERNAL_MODELS` rather than hand-written; that it carries no config path (a path in the prompt is a path the model can echo back); that every alias it advertises folds back onto its own model key; that an invented dataset, task, model or config path is rejected and recorded by `validate_llm_intent`; and that the intent layer neither plans nor executes. | `intent_catalog.json` |
| `RUN-01` | `run_context` | Where a run goes and what it may claim about itself: the loaded project is the only source of an agent workspace, two projects cannot produce the same one, and a project root outside `runs/` is brought back inside it; the provenance block reads the run directory out of the plan's own argv, names every contracted field and states "Not available" rather than leaving a blank; an ETA appears only after more than one measured interval and says `estimating…` until then; a compact metric never replaces the exact stored value; and no GPU name, device string, GPU count, project path or API key is written into the UI source (docstrings excluded — prose is documentation, a literal in an expression is behaviour). Pure and static: no GPU, no data, no Streamlit runtime, no LLM. | `run_context.json` |
| `FIG-01` | `figure_contract` | Every contracted figure has a unique id, an existing builder in `batteryml_web/viz/figures.py`, a filename that is safe and unique, and a dispatch branch in the viz CLI. Each is then rendered twice to memory and the two byte strings compared, which is what establishes the determinism `CLAUDE.md` §8 requires. Audits `figure_manifest.json` against the files on disk when one is present. Also builds the website/export result panels (`viz/result_figures.build_panels`) from constant literals, renders each SVG twice and compares the bytes, checks the default SOH cell rule, and checks that the zoomed view leaves the plotted arrays unchanged. | `figure_contract.json` |

`run_all` additionally writes `verification_report.json` (the full structured
result of every check) and `verification_summary.txt` (the same, readable top to
bottom). Default output directory is `runs/verify/`, which is gitignored.

## The result contract

Every check returns exactly one `CheckResult`
(`scripts/verify/common.py`):

```python
CheckResult(
    check_id="CFG-01",        # stable, quoted in reports
    name="config_validity",
    status="PASS" | "FAIL" | "SKIP",
    message="one line the reader can act on",
    evidence={...},           # structured detail, serialized verbatim
    artifacts=["runs/verify/config_validity.json"],
    duration_s=0.31,
)
```

No check raises to the top level. `run_check()` in `common.py` is the single
barrier: an exception escaping a check becomes a `FAIL` result whose evidence
carries the exception type, message and full traceback. Nothing is swallowed
silently (`CLAUDE.md` §6).

`FAIL` messages name the offending file, field or command and state the
corrective action. `SKIP` messages say what could not be observed and why.

## Working without data, and without dependencies

The checks are **data-optional** by requirement: with no `data/` directory at
all, `DATA-01` and `SPLIT-01` report SKIP with a reason and still write their
artifacts, and nothing crashes.

They are also, deliberately, **dependency-tolerant**. `batteryml` cannot be
imported without torch, so every check that would prefer live registry data
falls back to reading the source with `ast`:

| Wanted | Preferred source | Fallback |
| --- | --- | --- |
| Registered class names | `batteryml.builders.<REGISTRY>.class_mapping`, after importing `batteryml` **and** `batteryml_web.models` | `@REGISTRY.register()` decorators parsed from the roots in `common.registration_source_roots()` |
| Model base chains (device support) | — | class definitions parsed from `batteryml/models/` and `batteryml_web/models/` |
| Advertised `input_type`s | `batteryml.preprocess.SUPPORTED_SOURCES` | `ast.literal_eval` of the same assignment in `batteryml/preprocess/__init__.py` |
| Split dataset requirements | — | cell-ID literals parsed out of the splitter classes, plus `cell_data_path` from the configs |

Both scan roots matter. A model registered through the extension point
(`batteryml_web/models/`, see `docs/EXTENDING_MODELS.md`) reaches the same
`MODELS` registry as a core one, so a check that looked only at `batteryml/`
would report it as unregistered — `CFG-01` would then fail every one of its
configs with a false `is not registered in MODELS`.

Every result records which source it used, in `evidence.registry_source` /
`evidence.dataset_source`, so a report is never ambiguous about what it read.

`CLI-01` is the one check that must start a subprocess. If the wrapper fails to
start because an import failed, `CLI-01` reports **SKIP** and points at
`ENV-01`: a missing dependency is the environment check's finding, not a
contract violation.

`VAL-01` is the one check with no dependency-free fallback, by design. It
exercises the real execution gate, which resolves names against the live
registries; a source-scanned approximation would test something other than what
gates a run. It asks `config_validator.registry_status()` once up front, so a
missing dependency reads as one SKIP rather than as 130 refused configs.

## Dependencies

Standard library, plus `pyyaml` — already in `requirements.txt`. In particular
the harness does **not** use pandas, so `ENV-01` can report that pandas is
missing rather than failing to import.

`FIG-01` is the one exception, and it is unavoidable: a check that draws a
figure needs the library that draws figures. It requires `numpy` and
`matplotlib`, both already declared in `requirements.txt`, and reports **SKIP**
with the missing package named when either is absent — so a machine without them
gets a stated gap, not a failure. It needs no torch, so it still runs where the
rest of BatteryML cannot be imported.

PEP 440 version comparison is deliberately not implemented: `ENV-01` reports the
declared specifier next to the installed version, and only judges an
unambiguous `==` pin. Taking on `packaging` to do better is not worth a new
dependency.

## Adding a check

1. Add `scripts/verify/check_<name>.py` exposing `CHECK_ID`, `CHECK_NAME`,
   `check(...) -> CheckResult` and a `main()` for standalone use.
2. Register it in the `CHECKS` tuple in `run_all.py`. The fourth element says
   whether the check takes `output_dir`.
3. Add a row to the table above.
4. Make sure it reports SKIP, not an exception, when its subject is absent.
