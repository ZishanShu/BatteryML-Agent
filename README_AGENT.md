<div align="center">

# BatteryML-Agent — Deployment Guide

**A step-by-step guide for battery researchers.**
No machine-learning background required. No GPU required.

**English** · [简体中文](./README_AGENT.zh-CN.md)

</div>

---

## What you are installing

BatteryML-Agent is a web interface and workflow layer on top of
[BatteryML](https://github.com/microsoft/BatteryML) (ICLR 2024). You give it
battery cycling data; it predicts **RUL** (remaining useful life) or **SOH**
(state of health) and writes every result to disk in a fixed, citable layout.

There are two ways to use it, and **you never need the second one**:

| Route | What it is | Needs an API key? |
| --- | --- | --- |
| **Route B — Quick start** | Click through the web UI, or run one CLI command. Fully deterministic. | **No.** Works offline. |
| **Route A — LLM-guided** | An optional assistant that explains configs, errors and results. | Yes, but entirely optional. |

**This guide covers Route B.** The assistant is a bonus at the very end
(§10). If you never configure it, nothing breaks — every page stays usable
by hand.

> **The assistant can never produce a number.** It does not train, predict,
> or compute a metric. Every value you see comes from BatteryML itself.

---

## Table of contents

1. [Before you start](#1-before-you-start)
2. [Get the code](#2-get-the-code)
3. [Create a Python environment](#3-create-a-python-environment)
4. [Install dependencies](#4-install-dependencies)
5. [Two one-time fixes](#5-two-one-time-fixes-important)
6. [Health check](#6-health-check)
7. [Download battery data](#7-download-battery-data)
8. [Preprocess the data](#8-preprocess-the-data)
9. [Run your first prediction](#9-run-your-first-prediction)
10. [Optional: the LLM assistant](#10-optional-the-llm-assistant)
11. [Where your results are](#11-where-your-results-are)
12. [Troubleshooting](#12-troubleshooting)
13. [How this guide was verified](#13-how-this-guide-was-verified)

---

## 1. Before you start

### What you need

| Item | Requirement | How to check |
| --- | --- | --- |
| Operating system | Linux, macOS, or Windows (WSL2 recommended) | — |
| Python | **3.9 or newer** | `python3 --version` |
| Disk space | **~10 GB free** (raw MATR data alone is several GB) | `df -h .` |
| RAM | 8 GB minimum, 16 GB comfortable | — |
| GPU | **Not required.** The default demo runs on CPU. | — |
| Internet | Needed once, to download packages and data | — |

### How long this takes

| Phase | Roughly |
| --- | --- |
| Steps 2–6 (install + health check) | 15–30 min |
| Step 7 (download MATR) | 20–60 min, depending on your connection |
| Step 8 (preprocess MATR) | 10–30 min |
| Step 9 (first prediction) | 1–5 min on CPU |

> Download and preprocessing times are estimates, not measurements — they
> depend on your network and disk. See §13.

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

Every remaining command in this guide assumes you are **inside the
`BatteryML-Agent` folder**. If you open a new terminal later, `cd` back into
it first.

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
few others. It is the slowest install step — several minutes is normal, and
`torch` is the large one.

> **On a machine with no NVIDIA GPU**, the default `torch` wheel still works
> — it simply runs on CPU. You do not need CUDA for the demo in this guide.

Confirm the key packages imported correctly:

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

Expected output:

```
all imports OK
```

If any line errors, re-run the `pip install -r requirements.txt` and read
the error — it usually names a missing system library.

---

## 5. Two one-time fixes (important)

A fresh clone has two rough edges that will stop you. Both are one line.
**Do these before going further**, or Steps 6–9 will fail with confusing
errors.

### Fix 1 — the optional CPTransformer model blocks the CLI

The repository lists a model (`cptransformer`) whose source code lives in a
separate project that is *not* included in this clone. Because of that, the
BatteryML command line refuses to start at all:

```
error: Could not import extension model module batteryml_web.models.cptransformer:
FileNotFoundError: BatteryLife CPTransformer source not found: .../third_party/BatteryLife/models/CPTransformer.py
```

Unless you specifically need CPTransformer, disable it:

```bash
# Rename the module so it is treated as a helper, not a model
mv batteryml_web/models/cptransformer.py batteryml_web/models/_cptransformer.py

# Remove it from the registered model list
python - <<'PY'
from pathlib import Path
p = Path("batteryml_web/models/__init__.py")
s = p.read_text()
p.write_text(s.replace('    "cptransformer",\n', ''))
print("cptransformer disabled")
PY
```

Verify the command line now starts:

```bash
python batteryml_web/core/run_batteryml_cli.py run --help
```

Expected: a usage message beginning
`usage: BatteryML command line utilities. run [-h] ...`, and **no** error
line. Every other model — XGBoost, Random Forest, Ridge, SVM, the variance
and discharge models, BatLiNet — stays available.

### Fix 2 — the dataset manifest tool points at someone else's machine

`tools/sync_data_and_config_manifests.py` currently hard-codes an absolute
path from the original author's server. On your machine it would write to
the wrong place, and the web UI would show **no datasets forever**. Point it
at your own clone:

```bash
python - <<'PY'
from pathlib import Path
p = Path("tools/sync_data_and_config_manifests.py")
s = p.read_text()
old = 'ROOT = Path("/root/share/szs/BatteryML").resolve()'
new = 'ROOT = Path(__file__).resolve().parents[1]'
assert old in s, "already patched, or the file changed — check line 8 by hand"
p.write_text(s.replace(old, new, 1))
print("manifest tool now uses this repository as its root")
PY
```

> If the command prints an `AssertionError`, open the file and look at line
> 8. Either someone already applied this fix, or the line moved.

---

## 6. Health check

Before touching any data, confirm the installation is sane:

```bash
python -m scripts.verify.run_all
```

This runs 13 independent checks and prints one line each. On a **fresh
install with no data yet**, this is normal and expected:

```
[SKIP] DATA-01 dataset_status: data does not exist, so no dataset has raw or processed files...
[SKIP] SPLIT-01 split_requirements: data/processed does not exist...
...
X passed, Y failed, 3 skipped of 13 check(s)
```

**What matters at this stage:**

| Check | Should say | If it does not |
| --- | --- | --- |
| `ENV-01 environment` | `[PASS]`, or `[FAIL]` naming a missing package | Install what it names, then re-run |
| `CLI-01 cli_contract` | `[PASS]` | You skipped **Fix 1** in §5 |
| `EXT-01 extension_contract` | `[PASS]` | You skipped **Fix 1** in §5 |
| `DATA-01`, `SPLIT-01`, `VAL-01` | `[SKIP]` | Normal — you have no data yet |

Full check-by-check reference: [`docs/VERIFY.md`](./docs/VERIFY.md).
A machine-readable report is written to `runs/verify/verification_report.json`.

---

## 7. Download battery data

We use **MATR** (the MIT/Toyota LFP dataset, 180 cells) — the reference demo
for this platform.

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

> **Always call the CLI through `batteryml_web/core/run_batteryml_cli.py`.**
> Running `python bin/batteryml.py` directly fails with
> `ModuleNotFoundError: ... 'batteryml' is not a package`, because the script
> name shadows the package it needs to import.

This downloads four `.mat` files (several GB total) and will take a while.
Confirm when it finishes:

```bash
ls -lh data/raw/MATR
```

Expected: four files named `MATR_batch_20170512.mat`,
`MATR_batch_20170630.mat`, `MATR_batch_20180412.mat`,
`MATR_batch_20190124.mat`.

**Datasets you can download this way:** `MATR`, `HUST`, `CALCE`, `RWTH`.

**Datasets BatteryML can preprocess but not auto-download**
(`HNEI`, `SNL`, `UL_PUR`, `OX`) must be fetched by hand — see
[`dataprepare.md`](./dataprepare.md) — then placed in
`data/raw/<NAME>/` before Step 8.

---

## 8. Preprocess the data

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

Now build the manifests the web UI reads (this is why Fix 2 mattered):

```bash
python tools/sync_data_and_config_manifests.py
```

This writes `data/metadata/dataset_manifest.json` and
`data/metadata/model_config_manifest.json`. Confirm MATR is registered as
present:

```bash
python -c "
import json
d = json.load(open('data/metadata/dataset_manifest.json'))
print('present datasets:', [x['name'] for x in d if x.get('status') == 'present'])
"
```

Expected: `present datasets: ['MATR']` (plus anything else you prepared).

> **The data flow is always
> `data/raw/<NAME>` → preprocess → `data/processed/<NAME>` → run.**
> Training never reads raw files. If you later see
> `AssertionError: data/processed/MATR`, it means this step was not run —
> it does **not** mean the web app is broken.

---

## 9. Run your first prediction

Two ways. They do the same thing.

<details open>
<summary><b>Option A — the web interface (recommended for first-time users)</b></summary>

```bash
python -m streamlit run batteryml_web/app.py
```

Open the URL it prints — normally <http://localhost:8501>.

> **On a remote server?** Add `--server.port 8501 --server.address 0.0.0.0`,
> or forward the port over SSH from your laptop:
> `ssh -L 8501:localhost:8501 you@server`, then open `localhost:8501` locally.

Then, in the browser:

1. **Prepare Data** (left sidebar) → **Dataset Overview** tab → click
   **Refresh datasets**. MATR should appear with status **READY**.
2. **Run Model** → choose **Dataset: MATR**, **Task: RUL prediction**,
   **Model: XGBoost**, **Device: cpu**, **Seed: 0**.
3. Review the command preview shown on screen — this is the exact command
   that will run; nothing is hidden.
4. Press the dark **Run BatteryML** button. Progress streams live.
5. **Review Results** → metrics table, observed-vs-predicted plot, residuals
   and the prediction table.

The dark filled button is always the main action on a page; outlined buttons
are supporting actions; flat panels are information only.

</details>

<details>
<summary><b>Option B — one command line (no browser)</b></summary>

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace runs/native_matr_xgb_test \
  --train --eval --device cpu --seed 0
```

**Three things that are easy to get wrong:**

- `config_path` is **positional**. `run --config configs/...` fails with
  `unrecognized arguments: --config`.
- `--train --eval` are **mandatory**. Without them BatteryML prints the
  workspace path and exits without doing anything.
- Always pass `--seed` so the run is reproducible.

</details>

### Is my number right?

For **MATR1 + XGBoost**, the published benchmark RMSE is **334 cycles**
(source: BatteryML README, "Benchmark result of RUL task", Zhang et al.,
ICLR 2024). A result in that neighbourhood means your pipeline is wired
correctly.

Do not compare across different splits — MATR1 and MATR2 are different test
sets, and an SOH result is never comparable to an RUL benchmark. The web UI
enforces this for you and refuses to show a benchmark when the protocol does
not match.

---

## 10. Optional: the LLM assistant

**Skip this entirely if you only want predictions.** Everything above works
without it.

The assistant explains configs, diagnoses errors, and interprets results in
plain language. It never computes a number.

1. In the sidebar, open **Settings**.
2. Fill in **Base URL**, **Model**, and **API key** for any
   OpenAI-compatible endpoint.
3. Press **Save** (stores it), then **Connect** (tests it).

**About your API key:**

- It is kept **in your browser session only**.
- It is never written to any repository file, config, run manifest, log,
  exported bundle, or error message.
- It disappears when the session ends. Re-enter it next time.

Never paste an API key into a file you commit to git.

---

## 11. Where your results are

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

**Reproducibility:** the same config + same seed + same data fingerprint
produces an identical `metrics.csv`. Where a model is inherently
non-deterministic, that fact is recorded in `run_metadata.json` rather than
hidden.

---

## 12. Troubleshooting

| What you see | What it means | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: ... 'batteryml' is not a package` | You ran `python bin/batteryml.py` directly | Use `python batteryml_web/core/run_batteryml_cli.py ...` |
| `error: Could not import extension model module ... cptransformer` | Missing optional third-party source | Apply **Fix 1** in §5 |
| `unrecognized arguments: --config` | `config` is positional | `run configs/....yaml`, not `run --config configs/....yaml` |
| Run prints a workspace path and exits immediately | `--train --eval` missing | Add both flags |
| `AssertionError: data/processed/MATR` | Preprocessing was never run | Do §8. This is **not** a web-app bug |
| Web UI says "No datasets are available yet" | Manifests missing or written elsewhere | Apply **Fix 2** in §5, then re-run `python tools/sync_data_and_config_manifests.py` |
| `[FAIL] ENV-01 ... not installed: fire` | A declared package is missing | `pip install fire`, or `pip install -r requirements.txt` again |
| Browser shows nothing on a remote server | Streamlit bound to localhost | Use SSH port forwarding, or `--server.address 0.0.0.0` |
| `CUDA out of memory` | Model too large for your GPU | Add `--device cpu`. The demo does not need a GPU |

Still stuck? Run `python -m scripts.verify.run_all` and open an issue with
its output plus `runs/verify/verification_report.json`. That report names
the failing check and the corrective action.

---

## 13. How this guide was verified

Being explicit about this, because a guide that overstates its testing wastes
your time.

**Verified by actually running it** in a clean Linux container with Python
3.11:

- `pip install -r requirements.txt` and the import check (§4)
- Both fixes in §5, including the before/after of
  `run_batteryml_cli.py run --help` (exit 2 → exit 0)
- `python -m scripts.verify.run_all` (§6)
- `python tools/sync_data_and_config_manifests.py` after Fix 2 — produced
  15 dataset entries and 180 model-config entries (§8)
- `python -m streamlit run batteryml_web/app.py`, and every page named in
  §9 rendered in a real browser

**Not executed here, and stated from the code and project records:**

- The MATR download (§7) and preprocessing (§8). No dataset was downloaded
  in this environment. File names and the expected count of 180 come from
  the dataset registry and project records, not from a run.
- A full train/eval cycle and the resulting `metrics.csv`. The 334-cycle
  figure is the **published benchmark**, not a number measured here.
- All timings in §1 are estimates.
- Windows and macOS. Commands were exercised on Linux only.

If a step behaves differently on your machine, that is worth an issue — the
gap is in this guide, not in you.

---

## Further reading

| Document | What it covers |
| --- | --- |
| [`README.md`](./README.md) | Upstream BatteryML: the paper, datasets, full benchmark tables |
| [`dataprepare.md`](./dataprepare.md) | Manual data preparation for every dataset |
| [`docs/VERIFY.md`](./docs/VERIFY.md) | What each verification check proves |
| [`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) | Adding your own model |
| [`docs/AGENT_WORKFLOW.md`](./docs/AGENT_WORKFLOW.md) | How the assistant is wired |
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
