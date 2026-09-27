<div align="center">

# BatteryML-Agent

**A deployment and user guide for battery researchers.**
No machine-learning background required. No GPU required.

**English** · [简体中文](./README.zh-CN.md)

</div>

---

## What this is

**[BatteryML](https://github.com/microsoft/BatteryML)** (Microsoft, ICLR 2024)
is an open-source library for machine learning on battery degradation. It
defines the data format, the preprocessors, the train/test splits, the
models and the metrics. It is the scientific core, and every number this
platform reports comes from it.

**BatteryML-Agent** is a workflow layer on top of it. It adds:

- a **web interface** for the whole pipeline — inspect data, preprocess,
  configure a run, execute it, read the results;
- **dataset readiness** — a straight answer to "can I actually run this
  dataset yet, and if not, what is missing";
- a **fixed result layout** on disk, so a run is citable and reproducible;
- an **optional LLM assistant** that explains configs, errors and results.

It does **not** add new science. It does not change BatteryML's models,
metrics or splits.

### Two routes

| Route | What it is | Needs an API key? |
| --- | --- | --- |
| **Route B — deterministic** | The web UI, or one CLI command. Everything in §1–§12. | **No.** Works fully offline. |
| **Route A — LLM-assisted** | An optional assistant layered on top (§13). | Yes — and it is entirely optional. |

**Route B is the default and the one this guide follows.** If you never
configure an assistant, nothing is degraded: every page stays usable by
hand.

> **The assistant never produces a number.** It does not train, predict,
> compute a metric, or decide whether a dataset is ready. Those all come
> from BatteryML and from the deterministic readiness layer. The assistant
> only explains what is already there.

---

## Table of contents

1. [Before you start](#1-before-you-start)
2. [Get the code](#2-get-the-code)
3. [Create a Python environment](#3-create-a-python-environment)
4. [Install dependencies](#4-install-dependencies)
5. [Verify the environment](#5-verify-the-environment)
6. [Get battery data](#6-get-battery-data)
7. [Preprocess and register the data](#7-preprocess-and-register-the-data)
8. [Dataset readiness statuses](#8-dataset-readiness-statuses)
9. [Launch the web app](#9-launch-the-web-app)
10. [Run a smoke test](#10-run-a-smoke-test)
11. [Inspect your results](#11-inspect-your-results)
12. [Benchmark performance, and what is not](#12-benchmark-performance-and-what-is-not)
13. [Optional: the LLM assistant](#13-optional-the-llm-assistant)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations](#15-known-limitations)
16. [How this guide was verified](#16-how-this-guide-was-verified)

---

## 1. Before you start

### What you need

| Item | Requirement | How to check |
| --- | --- | --- |
| Operating system | Linux, macOS, or Windows (WSL2 recommended) | — |
| Python | **3.9 or newer** | `python3 --version` |
| Disk space | **~10 GB free** (raw MATR data alone is several GB) | `df -h .` |
| RAM | 8 GB minimum, 16 GB comfortable | — |
| GPU | **Not required.** CPU is the default; one GPU is supported. | — |
| Internet | Needed once, to install packages and download data | — |

### How long this takes

| Step | Roughly |
| --- | --- |
| §2–§5 — install and verify | 15–30 min |
| §6 — download MATR | 20–60 min, depending on your connection |
| §7 — preprocess MATR | 10–30 min |
| §9–§10 — first run | 1–5 min on CPU for a smoke run |

> These are estimates, not measurements — they depend on your network, disk
> and CPU. See §16 for exactly what was and was not timed.

### A note on the terminal

Every command below is typed into a terminal (Terminal on macOS, or
Ubuntu/WSL on Windows). Lines starting with `#` are comments — you do not
type those. Copy one block at a time and press Enter.

---

## 2. Get the code

```bash
git clone https://github.com/ZishanShu/BatteryML-Agent.git
cd BatteryML-Agent
```

Every remaining command assumes you are **inside the `BatteryML-Agent`
folder**. If you open a new terminal later, `cd` back into it first.

Check you are in the right place:

```bash
ls
```

You should see `batteryml`, `batteryml_web`, `configs`, `requirements.txt`,
among others. If you do not, you are in the wrong directory.

---

## 3. Create a Python environment

This keeps BatteryML's packages separate from the rest of your computer, so
nothing you already rely on gets upgraded or broken. **Pick one option.**

<details open>
<summary><b>Option A — conda (recommended if you already use Anaconda)</b></summary>

```bash
conda create -n batteryml python=3.10 -y
conda activate batteryml
```

You will need to run `conda activate batteryml` every time you open a new
terminal.

</details>

<details>
<summary><b>Option B — venv (built into Python, no extra install)</b></summary>

```bash
python3 -m venv .venv

# macOS / Linux:
source .venv/bin/activate

# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

You will need to re-run the `activate` line every time you open a new
terminal.

</details>

**Confirm it worked.** Your prompt should now show `(batteryml)` or
`(.venv)`. Check the Python version:

```bash
python --version
```

Expected: `Python 3.9.x` or newer. If it prints 3.8 or lower, go back and
create the environment with a newer Python.

---

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This pulls in numpy, pandas, scikit-learn, xgboost, torch, streamlit and a
few others. It is the slowest step — several minutes is normal, and `torch`
is the large one.

> **On a machine with no NVIDIA GPU**, the default `torch` wheel still works
> — it simply runs on CPU. You do not need CUDA for anything in this guide.

Confirm the key packages import:

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

Expected output:

```
all imports OK
```

If a line errors, re-run `pip install -r requirements.txt` and read the
error — it usually names a missing system library.

---

## 5. Verify the environment

Before touching any data, confirm the installation is sane:

```bash
python -m scripts.verify.run_all
```

This runs 13 independent checks and prints one line each. On a clean clone
with **no data yet**, the expected result is:

```
9 passed, 2 failed, 2 skipped of 13 check(s)
```

Once you have preprocessed a dataset (§7), `DATA-01` starts passing and it
becomes `10 passed, 2 failed, 1 skipped`.

**What matters at this stage:**

| Check | Should say | If it does not |
| --- | --- | --- |
| `CLI-01 cli_contract` | `[PASS]` | Your install is incomplete — redo §4 |
| `EXT-01 extension_contract` | `[PASS]` | Same |
| `CFG-01 config_validity` | `[PASS]` | Read the message; it names the config |
| `DATA-01 dataset_status` | `[SKIP]` before §6, `[PASS]` after §7 | It reports data, so it skips when there is none |
| `SPLIT-01 split_requirements` | `[SKIP]` | Normal — you have no processed data yet |

**The two failures you should expect on a clean clone**, neither of which
blocks anything in this guide:

- `ENV-01 environment` — reports `fire` as not installed. It is declared in
  `requirements.txt` but is not imported by any module in the repository.
- `REG-01 registry_integrity` — reports six cycler input types advertised by
  upstream BatteryML (`BATTERYARCHIVE`, `BIOLOGIC`, `INDIGO`, `LANDT`,
  `MACCOR`, `NOVONIX`) that have no registered preprocessor. That is an
  upstream gap, not a BatteryML-Agent regression.

You may also see `VAL-01` note that one config is *correctly refused*
because a third-party source is not vendored here. That is `CPTransformer`,
an optional deep model whose network code lives in a separate repository.
It costs you nothing — every other model runs normally. See §15.

Full check-by-check reference: [`docs/VERIFY.md`](./docs/VERIFY.md).
A machine-readable report is written to `runs/verify/verification_report.json`.

---

## 6. Get battery data

This guide uses **MATR** (the MIT/Toyota LFP dataset, 180 cells) — the
reference demo for this platform.

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

> **Always call the CLI through `batteryml_web/core/run_batteryml_cli.py`.**
> Running `python bin/batteryml.py` directly fails with
> `ModuleNotFoundError: ... 'batteryml' is not a package`, because the script
> name shadows the package it needs to import.

This downloads four `.mat` files (several GB total) and takes a while.
Confirm when it finishes:

```bash
ls -lh data/raw/MATR
```

Expected: `MATR_batch_20170512.mat`, `MATR_batch_20170630.mat`,
`MATR_batch_20180412.mat`, `MATR_batch_20190124.mat`.

**Datasets this command can download:** `MATR`, `HUST`, `CALCE`, `RWTH`.

**Datasets BatteryML can preprocess but not auto-download** — `HNEI`, `SNL`,
`UL_PUR`, `OX` — must be fetched by hand; see
[`dataprepare.md`](./dataprepare.md). Place them under `data/raw/<NAME>/`
before §7.

**Other datasets** listed by the readiness layer (`MICH`, `MICH_EXP`, `SDU`,
`Stanford_2`, `Tongji`, `XJTU`, `ISU_ILCC`) come from the BatteryLife
distribution and have no native BatteryML preprocessor. They can be
*recognised* as raw data but not preprocessed here. See §8 and §15.

---

## 7. Preprocess and register the data

Raw cycler files are not what the models read. This converts them into
BatteryML's uniform `BatteryData` format.

```bash
python batteryml_web/core/run_batteryml_cli.py preprocess \
  MATR data/raw/MATR data/processed/MATR
```

Confirm:

```bash
ls data/processed/MATR | wc -l
```

Expected: **180** — one `.pkl` file per cell.

Now build the manifests the web UI reads. This is what the **Dataset
Overview** page lists:

```bash
python tools/sync_data_and_config_manifests.py
```

It writes `data/metadata/dataset_manifest.json` and
`data/metadata/model_config_manifest.json` inside this checkout. Confirm
MATR is registered:

```bash
python -c "
import json
d = json.load(open('data/metadata/dataset_manifest.json'))
print('present datasets:', [x['name'] for x in d if x.get('status') == 'present'])
"
```

Expected: `present datasets: ['MATR']`, plus anything else you prepared.

> To describe a checkout other than the one holding the script, set
> `BATTERYML_ROOT=/path/to/checkout` before running it.

> **The data flow is always
> `data/raw/<NAME>` → preprocess → `data/processed/<NAME>` → run.**
> Training never reads raw files. If you later see
> `AssertionError: data/processed/MATR`, this step was not run — it does
> **not** mean the web app is broken.

---

## 8. Dataset readiness statuses

Readiness is computed from the filesystem alone, and is the same answer the
web UI shows. Inspect it at any time:

```bash
python -c "
from batteryml_web.core.readiness import list_dataset_readiness
for r in list_dataset_readiness():
    print(f'{r.dataset:<12} {r.status.value}')
"
```

There are exactly four states:

| Status | Meaning | What to do |
| --- | --- | --- |
| **`READY`** | Processed `.pkl` cells are present. | Run it. |
| **`RAW_ONLY`** | Recognised raw data is present, nothing processed yet. | Preprocess it (§7) — if a preprocessor exists for it (§15). |
| **`INCOMPLETE_PROCESSED`** | A processed directory exists but holds no usable `.pkl` cells. | Re-run preprocessing; the previous attempt did not finish. |
| **`MISSING`** | Neither recognised raw nor processed data. | Download or place the raw data (§6). |

Raw data is recognised in three shapes: BatteryML's native
`data/raw/<NAME>/`, an extracted BatteryLife distribution at
`data/raw/BatteryLife/v11/<NAME>/`, and the archive `<NAME>.zip` beside it.
An archive that carries the `.zip` name but not valid zip data — an
interrupted download — is deliberately **not** counted as present.

<details>
<summary><b>Example — one real machine's readiness (not a universal expectation)</b></summary>

This is what one configured research server reported. Your machine will
differ entirely; it depends only on what you have downloaded.

```
READY                 MATR (180 processed cells), HUST (77 processed cells)
RAW_ONLY              CALCE, MICH, MICH_EXP, RWTH, SDU, Stanford_2, Tongji, XJTU
INCOMPLETE_PROCESSED  HNEI, SNL, UL_PUR
MISSING               ISU_ILCC
```

</details>

---

## 9. Launch the web app

```bash
python -m streamlit run batteryml_web/app.py
```

Open the URL it prints — normally <http://localhost:8501>.

> **On a remote server?** Add `--server.port 8501 --server.address 0.0.0.0`,
> or forward the port over SSH from your laptop:
> `ssh -L 8501:localhost:8501 you@server`, then open `localhost:8501`
> locally.

### The deterministic workflow (Route B)

1. **Prepare Data** → **Dataset Overview** → **Refresh datasets**. MATR
   should appear as **READY**.
2. **Run Model** → choose **Dataset**, **Task** (RUL prediction or SOH
   estimation), **Model**, **Device**, **Seed**.
3. Read the **command preview** on screen. That is the exact command that
   will run — nothing is hidden or added.
4. Press the dark **Run BatteryML** button. Progress streams live.
5. **Review Results** → metrics table, observed-vs-predicted plot, residuals
   and the prediction table.

The dark filled button is always the main action on a page; outlined buttons
are supporting actions; flat panels are information only.

**Devices.** The normal UI supports **CPU or one GPU** (`cpu`, or a single
`cuda:N`). Multi-GPU and distributed training are not supported product
features — see §15.

The same run from the command line, without a browser:

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace runs/native_matr_xgb \
  --train --eval --device cpu --seed 0
```

**Three things that are easy to get wrong:**

- The config path is **positional**. `run --config configs/...` fails with
  `unrecognized arguments: --config`.
- `--train --eval` are **mandatory**. Without them BatteryML prints the
  workspace path and exits without doing anything.
- Always pass `--seed`, so the run is reproducible.

---

## 10. Run a smoke test

A smoke test answers one question: **does the execution path work end to
end on this machine?** It is not a measurement of model quality.

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/nn_models/cnn/matr_1.yaml \
  --workspace runs/smoke_matr_cnn \
  --train --eval --device cpu --seed 0 --epochs 1
```

`--epochs 1` is what makes it a smoke test. Swap `--device cpu` for
`--device cuda:0` if you have one GPU.

A successful smoke run ends with metrics and a full set of predictions. For
reference, one real single-epoch run of exactly this configuration on
`cuda:0` produced:

```
RMSE  381.609375
MAE   254.52743530273438
MAPE  0.34497949481010437
42 predictions, 42 distinct values
```

> ### ⚠ These numbers are **not** benchmark performance
>
> One epoch trains almost nothing. A smoke run tells you the pipeline is
> wired correctly — data loads, features build, the model trains, metrics
> are computed, artifacts are written. **It tells you nothing about how well
> the model predicts.** Do not quote, compare or publish a smoke-run number.
> For what counts as benchmark performance, see §12.

---

## 11. Inspect your results

Every run — from the web UI or the CLI — writes to
`runs/web_projects/<project_name>/`:

```
project_state.json
config.yaml
schema_mapping.json          # only if you ingested a custom dataset
run_metadata.json            # git commit, package versions, seed, device,
                             # config hash, dataset fingerprint, timing
environment_check.json
dataset_status.csv
feature_summary.json
training_log.txt
metrics.csv                  # <- your RMSE / MAE / MAPE
predictions/
figures/
checkpoints/
report.html
```

**`metrics.csv` holds the numbers you would cite.** `run_metadata.json`
holds everything needed to reproduce them.

In the UI, **Review Results** shows the metrics table, observed-vs-predicted,
residuals and the prediction table. A result with one lifetime estimate per
cell shows exactly those views — no cycle-wise curve is drawn for it,
because none exists.

**Reproducibility.** The same config + same seed + same data fingerprint
produces an identical `metrics.csv`. Where a model is inherently
non-deterministic, that fact is recorded in `run_metadata.json` rather than
hidden.

---

## 12. Benchmark performance, and what is not

Upstream BatteryML publishes a benchmark table for the RUL task. It lives in
**[`README_BATTERYML.md`](./README_BATTERYML.md)** — the preserved upstream
README — and that is the reference to cite and compare against.

Reading it correctly:

- the metric is **RMSE**, the unit is **cycles**, and **lower is better**;
- **MATR1 and MATR2 are different protocols** — different train/test splits.
  A MATR1 result may not be compared with a MATR2 number;
- an **SOH** result is never comparable to an **RUL** benchmark;
- a **smoke run is never comparable to anything** (§10).

As one concrete anchor: for **MATR1 + XGBoost**, the published RMSE is
**334 cycles**. A properly trained run of that exact configuration landing
near that value suggests your pipeline is wired correctly. A run using a
different split, a different task, or one epoch does not.

The platform enforces this for you: the result view refuses to show a
published benchmark alongside your result unless the protocol actually
matches.

---

## 13. Optional: the LLM assistant

**Skip this entirely if you only want predictions.** Everything above works
without it, offline, with no account anywhere.

The assistant explains configs, diagnoses errors and interprets results in
plain language. It never computes a number and never decides readiness.

Open **Settings** in the sidebar and fill in:

| Field | What to put |
| --- | --- |
| **Provider** | A preset, or **Custom OpenAI-compatible** to type your own endpoint |
| **Base URL** | Your provider's OpenAI-compatible endpoint |
| **Model** | A model name your key is authorized to use |
| **API key** | Your key — held in this browser session only |

Then press **Save** (stores it for this session), and **Connect** (tests it).

Any OpenAI-compatible endpoint works. The dropdown offers presets — OpenAI,
Gemini, Claude, Kimi / Moonshot, GLM / Zhipu — purely as a convenience that
fills in the Base URL for you; a preset never locks the field, and
**Custom OpenAI-compatible** accepts any gateway you have access to. Naming
a preset is not an endorsement, and none is required.

**About your API key:**

- It is held **in the browser session only**.
- It is **never** written to any repository file, config, project metadata,
  `run_metadata.json`, run history, log, exported result bundle, or error
  message shown to you.
- It disappears when the session ends. Re-enter it next time.
- **Never paste an API key into a file you commit to git.**

If the assistant is unavailable — no key, no network, provider rate-limited
— every page stays fully usable by hand. That is a requirement of the
design, not a fallback.

---

## 14. Troubleshooting

| What you see | What it means | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: ... 'batteryml' is not a package` | You ran `python bin/batteryml.py` directly | Use `python batteryml_web/core/run_batteryml_cli.py ...` |
| `unrecognized arguments: --config` | The config path is positional | `run configs/....yaml`, not `run --config configs/....yaml` |
| Run prints a workspace path and exits immediately | `--train --eval` missing | Add both flags |
| `AssertionError: data/processed/MATR` | Preprocessing was never run | Do §7. This is **not** a web-app bug |
| Web UI says "No datasets are available yet" | Manifests not rebuilt since you preprocessed | Run `python tools/sync_data_and_config_manifests.py`, then press **Refresh datasets** |
| A dataset shows `RAW_ONLY` and will not preprocess | No native preprocessor for it | See §8 and §15 |
| A dataset shows `INCOMPLETE_PROCESSED` | Preprocessing did not finish | Re-run §7 for that dataset |
| `[FAIL] ENV-01 ... not installed: fire` | Declared but unused dependency | Expected on a clean clone (§5); `pip install fire` silences it |
| `[FAIL] REG-01 ... have no registered preprocessor` | Upstream BatteryML gap | Expected (§5); not a BatteryML-Agent regression |
| Browser shows nothing on a remote server | Streamlit bound to localhost | SSH port forwarding, or `--server.address 0.0.0.0` |
| `CUDA out of memory` | Model too large for your GPU | Use `--device cpu`, or a smaller model. Nothing here needs a GPU |

Still stuck? Run `python -m scripts.verify.run_all` and open an issue with
its output plus `runs/verify/verification_report.json`. That report names
the failing check and the corrective action.

---

## 15. Known limitations

Stated plainly, because finding these out mid-project is worse than reading
them now.

- **Some models need third-party source you must supply.** `CPTransformer`
  wraps a network whose code lives in a separate repository under
  `third_party/`, which is not distributed here. It is skipped
  automatically and does not block anything else; its config is reported as
  unavailable rather than broken.
- **Some datasets can only ever be `RAW_ONLY` here.** The BatteryLife
  datasets (`MICH`, `MICH_EXP`, `SDU`, `Stanford_2`, `Tongji`, `XJTU`,
  `ISU_ILCC`) are recognised as raw data but have no native BatteryML
  preprocessor, so they cannot reach `READY` through this pipeline.
- **Six cycler formats are advertised upstream but unimplemented**:
  `BATTERYARCHIVE`, `BIOLOGIC`, `INDIGO`, `LANDT`, `MACCOR`, `NOVONIX`.
  `REG-01` reports this on every run (§5).
- **A one-epoch run is a smoke test, not a result** (§10).
- **Not every dataset × model × task combination is runnable.** The Run
  Model page only offers combinations whose config and data actually exist;
  a combination that is missing either is reported as blocked, with the
  reason, rather than silently failing at execution time.
- **The normal UI is CPU or a single GPU.** Multi-GPU and distributed
  (DDP) training are **not** supported product features and are not
  exposed; do not plan around them.
- **The LLM assistant depends on network and provider access.** It is
  optional, it can be unavailable, and nothing in Route B depends on it.
- **Windows is not verified.** Commands here were exercised on Linux only.

---

## 16. How this guide was verified

Being explicit, because a guide that overstates its testing wastes your
time.

**Executed in a clean Linux container (Python 3.11) with no datasets:**

- `pip install -r requirements.txt` and the import check (§4)
- `python -m scripts.verify.run_all` (§5) — in a **fresh worktree with no
  `data/` and no `third_party/`**: **9 passed, 2 failed, 2 skipped**, the
  two failures being `ENV-01` (`fire`) and `REG-01` (upstream preprocessor
  gaps), exactly as §5 describes
- `python batteryml_web/core/run_batteryml_cli.py run --help` on a clone
  with no `third_party/` — starts and prints usage
- the readiness command in §8 — ran, exit 0, printed the four-status
  vocabulary
- the smoke-run command in §10 — parsed and reached
  `AssertionError: data/processed/MATR`, which confirms the **syntax**
  including `--epochs`, and nothing more
- `python tools/sync_data_and_config_manifests.py` (§7) — wrote 15 dataset
  entries and 180 model-config entries inside the checkout
- `python -m streamlit run batteryml_web/app.py` (§9) — started, and the
  pages named above rendered in a real browser
- the repository's own suite: **1355 tests, no failures**

**Reported from a configured research server, not run here:**

- MATR `READY` with 180 processed cells, HUST `READY` with 77
- `python -m unittest -v tests.test_agent_config_selection` — 12/12 pass
- the single-epoch MATR + CNN smoke run in §10, including its metrics and
  its 42 distinct predictions

**Not executed anywhere in preparing this guide:**

- The MATR download (§6) and preprocessing (§7). No dataset was downloaded
  in this environment. The four file names and the count of 180 come from
  the dataset registry and project records.
- A full, multi-epoch train/eval cycle. The 334-cycle figure in §12 is the
  **published upstream benchmark**, not a number measured here.
- All timings in §1 — estimates.
- macOS and Windows.

If a step behaves differently on your machine, that is worth an issue — the
gap is in this guide, not in you.

---

## Further reading

| Document | What it covers |
| --- | --- |
| [`README_BATTERYML.md`](./README_BATTERYML.md) | Upstream BatteryML: the paper, the datasets, the full benchmark tables |
| [`dataprepare.md`](./dataprepare.md) | Manual data preparation, dataset by dataset |
| [`docs/VERIFY.md`](./docs/VERIFY.md) | What each verification check proves |
| [`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) | Adding your own model |
| [`docs/AGENT_WORKFLOW.md`](./docs/AGENT_WORKFLOW.md) | How the optional assistant is wired |
| [`docs/SOH_PROTOCOL.md`](./docs/SOH_PROTOCOL.md) | The SOH estimation protocol |

## Citation

BatteryML-Agent builds on BatteryML. If you use it, please cite:

```bibtex
@inproceedings{zhang2024batteryml,
  title={Battery{ML}: An Open-source Platform for Machine Learning on Battery Degradation},
  author={Han Zhang and Xiaofan Gui and Shun Zheng and Ziheng Lu and Yuqi Li and Jiang Bian},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
