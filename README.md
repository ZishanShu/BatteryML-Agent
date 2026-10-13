<div align="center">

# BatteryML-Agent — User Guide

**English** · [简体中文](./README.zh-CN.md)

Previous guides (archived, V1): [English](./README_v1.md) · [简体中文](./README_v1.zh-CN.md)

</div>

BatteryML-Agent is a web application and command-line toolkit for predicting
battery **remaining useful life (RUL)** and **state of health (SOH)**. It is
built on [BatteryML](https://github.com/microsoft/BatteryML): BatteryML does
all the preprocessing, training, prediction and scoring, and this repository
adds a step-by-step website, safety checks around every run, and an optional
AI assistant that explains things but never computes a number.

This guide assumes **no prior experience** with conda, Python or Streamlit.
Every command can be copied as it is. Every button is named exactly as it
appears on screen.

> **Tip.** If you only want to see a first result, read §2, §3, §4 (MATR
> only), §6 and §7. That path needs no API key and no GPU.

---

## Contents

1. [What BatteryML-Agent is](#1-what-batteryml-agent-is)
2. [Requirements and installation](#2-requirements-and-installation)
3. [The conda / Python environment](#3-the-conda--python-environment)
4. [Datasets: download and preprocessing](#4-datasets-download-and-preprocessing)
5. [Dataset readiness statuses](#5-dataset-readiness-statuses)
6. [Starting the website](#6-starting-the-website)
7. [Home page and Quick experience, step by step](#7-home-page-and-quick-experience-step-by-step)
8. [Prepare Data and Dataset Overview](#8-prepare-data-and-dataset-overview)
9. [Run Model: choosing a configuration](#9-run-model-choosing-a-configuration)
10. [Uploading your own YAML configuration safely](#10-uploading-your-own-yaml-configuration-safely)
11. [Agent Workflow and model recommendations](#11-agent-workflow-and-model-recommendations)
12. [Optional LLM assistant: Save, Connect, Explain](#12-optional-llm-assistant-save-connect-explain)
13. [CPU and GPU](#13-cpu-and-gpu)
14. [Review Results: run history, results and metrics](#14-review-results-run-history-results-and-metrics)
15. [Scientific figures and figure export](#15-scientific-figures-and-figure-export)
16. [Benchmark comparison and smoke-test limits](#16-benchmark-comparison-and-smoke-test-limits)
17. [Command line: copyable commands](#17-command-line-copyable-commands)
18. [Troubleshooting and known limitations](#18-troubleshooting-and-known-limitations)
19. [Reproducibility, verification status and citation](#19-reproducibility-verification-status-and-citation)

---

## 1. What BatteryML-Agent is

### 1.1 What it does for you

You give it battery cycling data (for example the public **MATR** dataset of
180 LFP/graphite cells). It then:

1. converts the raw files into BatteryML's format (**preprocessing**);
2. trains a model you choose (for example **XGBoost**) and evaluates it on a
   held-out test set (**run**);
3. shows the scores (RMSE, MAE, MAPE), the predictions and scientific figures,
   and lets you download everything (**review**).

Two prediction tasks are supported:

| Task | What is predicted | Unit of the error |
| --- | --- | --- |
| **RUL prediction** | how many cycles a cell has left | cycles |
| **SOH estimation** | the remaining capacity as a fraction of the initial capacity | capacity ratio |

### 1.2 Three ways to use it

| Route | For whom | Needs an API key? |
| --- | --- | --- |
| **Quick experience / Protocol Workflow** (§7–§9, §14) | Everyone. Fully deterministic and works offline. This is the reference route. | No |
| **Agent Workflow** (§11) | Describe the job in one sentence; the website works out the dataset, task and model and checks that they can run. | No (an LLM is optional) |
| **Command line** (§17) | Scripts, servers, batch jobs. | No |

### 1.3 Architecture, in one picture

```
            ┌──────────────────────── your browser ───────────────────────┐
            │  Home · Agent Workflow · Prepare Data · Run Model · Review  │
            └───────────────────────────────┬─────────────────────────────┘
                                            │  (Streamlit website: batteryml_web/app.py)
     optional LLM ─ explains only ─┐        ▼
                                   │   validators  ── refuse anything that cannot run safely
                                   │        │
                                   └──▶ suggestions are shown to you; nothing runs without your click
                                            ▼
             batteryml_web/core/run_batteryml_cli.py   (the one wrapper around BatteryML)
                                            ▼
                    BatteryML: preprocess · train · predict · evaluate (RMSE / MAE / MAPE)
                                            ▼
      data/raw/<DATASET> → data/processed/<DATASET> → runs/web_projects/<project>/…
```

Rules the code enforces:

- **All numbers come from BatteryML.** Predictions, metrics and benchmark
  values are never produced by the LLM.
- **Nothing runs until you press a Run button.** Analysing, recommending and
  explaining never start training.
- **Training reads processed data, never raw data.** Preprocess first (§4).

### 1.4 Folder layout

| Folder | Content |
| --- | --- |
| `batteryml/` | The BatteryML library (upstream code, not modified for this platform). |
| `batteryml_web/` | The website, the safety checks and the command-line wrapper. |
| `configs/` | Ready-made run configurations (YAML files), one per dataset × model. |
| `data/raw/<DATASET>/` | Raw files you download. Not stored in git. |
| `data/processed/<DATASET>/` | Preprocessed files (one `.pkl` per cell). Not stored in git. |
| `runs/` | Everything the runs write. Not stored in git. |
| `scripts/` | Health check, readiness inspection, figure export and other tools. |
| `docs/` | Technical documentation. |

---

## 2. Requirements and installation

### 2.1 What you need

| Item | Requirement |
| --- | --- |
| Operating system | **Linux x86_64** (tested). macOS: not tested. Windows: use [WSL2 with Ubuntu](https://learn.microsoft.com/windows/wsl/install) and follow the Linux steps inside it (not tested). |
| Disk space | About **12 GB** for the software (measured in the V1 test install: Miniconda + environment 8.0 GB, pip download cache 3.5 GB), plus room for the battery data. |
| Internet | During installation and data download. Hosts used: `repo.anaconda.com`, `pypi.org`, `files.pythonhosted.org`, `github.com`, `data.matr.io`. |
| GPU | **Not required.** Everything works on CPU. An NVIDIA GPU can be used (§13). |
| Admin rights | Not required. Everything installs into your home folder. |
| API key | **Not required.** Only the optional LLM assistant (§12) uses one. |

### 2.2 Check the machine

```bash
git --version
curl --version
```

Both must print a version. If one is missing, install it with your system's
package manager (Ubuntu: `sudo apt-get install -y git curl`).

Only if the machine has an NVIDIA GPU:

```bash
nvidia-smi
```

A table appears; note the `CUDA Version` in its top-right corner (needed in
§3.3). "Command not found" means no usable NVIDIA driver; everything will run
on CPU, which is fine.

### 2.3 Install Miniconda

Miniconda provides `conda`, which creates an isolated Python environment, so
nothing else on your machine is changed.

> Already have conda or Anaconda? Run `conda --version`. If it prints a
> version, skip to **step 4** below.

1. Download the installer:

   ```bash
   cd ~
   curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

   Other platforms use another file name in the URL:
   `Miniconda3-latest-Linux-aarch64.sh` (Linux ARM),
   `Miniconda3-latest-MacOSX-arm64.sh` (Apple Silicon),
   `Miniconda3-latest-MacOSX-x86_64.sh` (Intel Mac).

2. Install it into `~/miniconda3` without questions:

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
   ```

   The output ends with `installation finished.`

3. Connect conda to your terminal, then **close the terminal and open a new
   one** (required):

   ```bash
   $HOME/miniconda3/bin/conda init bash
   ```

   On macOS (zsh) use `$HOME/miniconda3/bin/conda init zsh`. In the new
   terminal the prompt starts with `(base)` and `conda --version` prints a
   version.

4. Accept the Anaconda Terms of Service. Without this, creating the
   environment fails with `CondaToSNonInteractiveError`:

   ```bash
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   ```

   Expected: `accepted Terms of Service for https://repo.anaconda.com/pkgs/main`
   and the same line for `…/pkgs/r`.

### 2.4 Download the code

```bash
cd ~
git clone https://github.com/ZishanShu/BatteryML-Agent.git
cd BatteryML-Agent
ls
```

The list includes `batteryml`, `batteryml_web`, `configs`, `requirements.txt`
and `README.md`.

> **Important:** every command in this guide is run **inside the
> `BatteryML-Agent` folder**. In every new terminal, run `cd ~/BatteryML-Agent`
> first.

---

## 3. The conda / Python environment

### 3.1 Create and activate the environment

```bash
conda create -n batteryml python=3.10 -y
conda activate batteryml
python --version
```

The prompt now starts with `(batteryml)` and Python prints `Python 3.10.x`.
Use Python 3.10: `requirements.txt` needs `numpy<2.0`, which has no builds for
Python 3.13.

> Every time you open a new terminal: `cd ~/BatteryML-Agent` and
> `conda activate batteryml`. Most "module not found" errors come from
> forgetting this.

### 3.2 Install PyTorch

```bash
python -m pip install --upgrade pip
pip install torch
```

This is the biggest download (about 5.5 GB on disk in the V1 test install).
The same package works with and without a GPU. Check it:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

| Output | Meaning |
| --- | --- |
| `… False` on a machine **without** a GPU | Correct. Everything runs on CPU. |
| `… True` on a machine **with** a GPU | Correct. The GPU can be used. |
| `… False` on a machine **with** a GPU | The NVIDIA driver is older than PyTorch's CUDA version. See §3.3. |

A warning `Failed to initialize NumPy: No module named 'numpy'` can appear at
this point. It is expected and disappears after §3.4.

### 3.3 Only for a GPU that printed `False`

Compare the second number (PyTorch's CUDA version, for example `13.0`) with
the `CUDA Version` from `nvidia-smi`. If the driver's number is lower, update
the driver, or install a PyTorch build that matches it. Take the exact
command from <https://pytorch.org/get-started/locally/> (Pip, Linux, your CUDA
version). It has this form (`cu126` = CUDA 12.6):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

### 3.4 Install the remaining packages

```bash
pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

Expected last line: `all imports OK`.

> **Do not run `pip install .`** The website and the command-line wrapper load
> BatteryML from this folder. An installed copy would not follow `git pull`.

Optional, frees about 3.5 GB: `pip cache purge`.

### 3.5 Run the health check

```bash
python -m scripts.verify.run_all
```

It runs 13 checks and prints one line per check, then a summary line. On a
fresh installation **without data**, the V1 guide recorded this summary:

```
10 passed, 1 failed, 2 skipped of 13 check(s)
```

| Check | Expected without data | Why |
| --- | --- | --- |
| `ENV-01` `CLI-01` `CFG-01` `VAL-01` `EXT-01` `DEV-01` `INTENT-01` `RUN-01` `UI-01` `FIG-01` | `[PASS]` | The installation is complete. |
| `DATA-01` `SPLIT-01` | `[SKIP]` | No data has been downloaded yet. |
| `REG-01` | `[FAIL]` | Expected. Upstream BatteryML lists six cycler formats without a preprocessor. It does not affect this guide. |

The command exits with code `1` because of `REG-01`; that is expected. Any
other `[FAIL]` names its problem; see §18. Once a `data/` folder exists,
`DATA-01` no longer says `[SKIP]`. The full report is written to
`runs/verify/verification_report.json`. What each check tests:
[`docs/VERIFY.md`](./docs/VERIFY.md).

---

## 4. Datasets: download and preprocessing

Data always flows in this order:

```
data/raw/<DATASET>  ──preprocess──▶  data/processed/<DATASET>  ──run──▶  results
```

You can do both steps **in the website** (§7 step 1, or §8) or **on the
command line** (below). They run the same BatteryML commands.

> After a download or preprocessing on the **command line**, open
> **Prepare Data → Dataset Overview** and click **Refresh datasets**, so that
> Prepare Data and Run Model list the dataset. The **Download …** and
> **Process …** buttons of Quick experience refresh this list automatically.

### 4.1 Which datasets can be downloaded automatically

| Dataset | Automatic download | Preprocessing |
| --- | --- | --- |
| `MATR` (reference dataset, 180 cells) | Yes | Yes |
| `HUST`, `CALCE`, `RWTH` | Yes | Yes |
| `HNEI`, `SNL`, `UL_PUR`, `OX` | **No.** Download by hand as described in [`dataprepare.md`](./dataprepare.md) and place the files in `data/raw/<NAME>/`. | Yes |

### 4.2 Download MATR

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

This downloads four large `.mat` files from `data.matr.io`; it takes a long
time. Running it again is always safe: complete files are skipped
(`[INFO] … already exists and is complete (… bytes). Skip it.`); missing or
incomplete files are fetched again. Files are written to a temporary `.part`
file first and renamed only when complete.

Check:

```bash
ls -lh data/raw/MATR
find data/raw/MATR -type f -size 0
```

The first command lists `MATR_batch_20170512.mat`, `MATR_batch_20170630.mat`,
`MATR_batch_20180412.mat` and `MATR_batch_20190124.mat`. The second prints
**nothing**.

If the machine cannot reach `data.matr.io`, the command stops with one line
`error: Could not download data/raw/MATR/MATR_batch_….mat` containing
`ProxyError`, `ConnectionError` or `403 Forbidden`, and writes nothing for
that file. Fix the network (try <https://data.matr.io> in a browser on the same
machine), then run the command again.

For `HUST`, `CALCE` or `RWTH`, replace `MATR` in **both** places of the command.

### 4.3 Preprocess MATR

```bash
python batteryml_web/core/run_batteryml_cli.py preprocess MATR data/raw/MATR data/processed/MATR
ls data/processed/MATR | wc -l
```

The count printed for MATR is `180` (one `.pkl` per cell; observed on the
project's server).

For another dataset, replace `MATR` in all three places. Many datasets at once:

```bash
python scripts/preprocess_sweep.py --dry-run
python scripts/preprocess_sweep.py --only HUST
```

`--dry-run` prints the command for each dataset and runs nothing (it still
writes its report, `logs/preprocess/sweep_report.csv`). Other options: `--skip`, `--force`,
`--timeout` (see `python scripts/preprocess_sweep.py --help`).

> Always use `batteryml_web/core/run_batteryml_cli.py`. Running
> `python bin/batteryml.py` directly fails with
> `ModuleNotFoundError: … 'batteryml' is not a package`.

### 4.4 Common data errors

| Message | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: Batch file not found: data/raw/MATR/…` | The download is incomplete. | Run §4.2 again, then §4.3. |
| `OSError: Unable to synchronously open file (file signature not found)` | A `.mat` file is empty or cut off. | Run §4.2 again (it replaces such files), then §4.3. |
| `AssertionError: data/processed/MATR` (during a run) | Preprocessing was never run. | Run §4.3. It does **not** mean the website is broken. |
| `KeyError: 'b1c1'` (during a run) | `data/processed/MATR` exists but is empty (preprocessing failed). | Fix the preprocessing error, run §4.3 again. |

---

## 5. Dataset readiness statuses

The website and the scripts classify every dataset into one of four states:

| Status | Meaning | What to do |
| --- | --- | --- |
| `READY` | Preprocessed `.pkl` files are present. | Nothing; it can be used for runs. |
| `RAW_ONLY` | Raw files are present; processed files are not. | Preprocess it (§4.3, or **Process … data** / **Run preprocessing** in the website). |
| `INCOMPLETE_PROCESSED` | `data/processed/<NAME>` exists but has no `.pkl` file; preprocessing failed or was interrupted. | Preprocess it again. |
| `MISSING` | Neither raw nor processed files. | Download it (§4.2), or place raw files by hand. |

Datasets you never downloaded show `MISSING`. That is normal.

Check readiness from the command line (read-only; changes nothing):

```bash
python -m scripts.inspect_dataset_readiness MATR
```

For each dataset it prints the raw and processed folders it looked at and a
`status` line, then a summary. Example on a machine **without** data:

```
MATR
    raw native                    absent  not present
        …/data/raw/MATR
    …
    processed                     absent  not present
        …/data/processed/MATR
    status                        MISSING

summary: MISSING 1
```

After §4.3 the status line reads `READY`. Without arguments it lists every
dataset; `--json` prints JSON instead.

Readiness describes **the data only**. A model also needs a registered,
validated configuration for that dataset and task (§9, §11).

---

## 6. Starting the website

```bash
cd ~/BatteryML-Agent
conda activate batteryml
python -m streamlit run batteryml_web/app.py
```

Use exactly `python -m streamlit run …` from the `BatteryML-Agent` folder.
(A plain `streamlit run …` can fail with
`ModuleNotFoundError: No module named 'batteryml_web'`, and
`.streamlit/config.toml` is only read from this folder.)

The terminal prints:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

Open `http://localhost:8501`. The browser tab is titled
**BatteryML Protocol Platform**. Stop the website with `Ctrl+C` in the
terminal.

Lines starting with a date that mention `use_container_width` or similar are
Streamlit deprecation warnings, not errors. Errors that matter appear on the
web page.

**On a remote server**, forward the port from **your own computer** (replace
`user@server`), keep that terminal open, and open `http://localhost:8501`
locally:

```bash
ssh -L 8501:localhost:8501 user@server
```

> The website has no login. Anyone who can reach port 8501 can use it. Prefer
> the SSH tunnel to opening the port to a network.

### 6.1 The sidebar (on every page)

| Element | What it is |
| --- | --- |
| **🔋 BatteryML / Protocol Navigator** | The brand tile. |
| **Current project** | The loaded project, or `(no project loaded)`. Runs made without a project go to `runs/web_projects/default`. |
| **🏠 Home** | Start page (§7). |
| **Agent Workflow** | "Describe a task in plain language; nothing runs until you press Run." (§11). |
| **Protocol Workflow** → **1. 🧪 Prepare Data**, **2. 🚀 Run Model**, **3. 📈 Review Results** | The manual protocol (§8, §9, §14). |
| **← Prev** / **Next →** | Move one step backward or forward in the protocol. |
| **Settings** | Opens the LLM settings (§12). |
| **LLM: …** status line | `Not connected`, `Configured / Not tested`, `Connected` or `Connection failed`. |

---

## 7. Home page and Quick experience, step by step

The Home page shows the title **BatteryML workspace** and the question
**How would you like to start?** with three cards:

| Card | Use it to |
| --- | --- |
| **⚡ Quick experience** ("Use the built-in MATR demo", "No API key required") | Get a first result with one dataset, one model, no project setup. |
| **✦ Create new project** | Make a named project folder for your own work. |
| **↗ Existing project** | Reopen a project you made earlier. |

Click a card to open its panel. **← Choose another start path** below the
panel closes it. Further down the page: **Your BatteryML workflow** (an
overview of the four steps), a note about the optional LLM if none is
configured, the **Agent Workflow** box (§11), and **Start workflow →**, which
opens **Prepare Data**.

### 7.1 Quick experience

Click **⚡ Quick experience**. The panel has four numbered steps. The default
is **MATR · RUL prediction · XGBoost · cpu**, using
`configs/baselines/sklearn/xgb/matr_1.yaml`.

#### Step 1 · Prepare data

1. **Where:** the **Raw dataset** drop-down. **Select:** `MATR` (default).
   A dataset with no run configuration shows `· no run configuration`; it
   cannot be run here (use **Open Prepare Data** instead).
2. **Read the status:** "Raw data" and "Processed data" each show their state,
   plus a line `Readiness: STATUS — reason` (statuses: §5). An unfinished
   download shows `Download incomplete (x/y files)`.
3. **If raw data is missing:** click **Download MATR data**.
   - **What happens:** BatteryML's download command runs with a progress bar.
     The log is saved to `runs/web_projects/<project>/logs/download_matr.log`.
   - **Expected:** the raw status changes to present, and the panel shows
     `✓ Raw files present — no download needed.`
   - **Errors:** a network failure shows the error on the page (see §4.2). For
     a dataset BatteryML cannot download, the panel says
     `BatteryML has no automatic download for X. Place its raw files under data/raw/X/, or see Prepare Data.`
4. **If processed data is missing:** click **Process MATR data**.
   - **What happens:** BatteryML's preprocessing runs; log in
     `runs/web_projects/<project>/logs/preprocess_matr.log`.
   - **Expected:** `✓ Processed data ready — no processing needed.` and
     readiness `READY`.
   - **Errors:** `FileNotFoundError` / `OSError` → the download is
     incomplete; download again (§4.4).

#### Step 2 · Configure model

| Control | What to choose |
| --- | --- |
| **Prediction task** | `RUL prediction` or `SOH estimation` (only tasks with a configuration for this dataset). |
| **Prediction model** | A model that has a configuration for this dataset and task, e.g. XGBoost. |
| **Protocol configuration** | The configuration file, shown as `name · path`. |
| **Compute device** | `cpu` or `cuda`. `cuda` shows `(not available on this machine)` when it cannot be used (§13). |
| **Upload YAML configuration (optional)** | Your own config file (§10). |

An info box confirms the choice:
`This configuration means: Dataset … · Task … · Prediction model … · Device … · Config file …`.
Read it before running.

#### Step 3 · Run training

1. **Read** "Run location": the folder the run will write.
2. If the data is not processed, the page warns
   `Process the dataset in step 1 before starting a model run.` and the run
   cannot start.
3. Click **Run experiment**.
   - **What happens:** BatteryML trains and evaluates with seed `0`. A progress
     bar shows the stages (initialize, check processed data, load
     configuration, training and evaluation, collect outputs).
   - **Output folder:** `runs/web_projects/<project>/quickstart/<run_id>/` with
     `run.json`, `run.log`, `output.log`, `config_<timestamp>.yaml` and
     `predictions_seed_0_<timestamp>.pkl`.
   - **Errors:** a failed run shows BatteryML's exit code and points to the
     run log. `AssertionError: data/processed/…` or `KeyError: 'b1c1'` →
     back to step 1.

#### Step 4 · Review results

The metrics BatteryML saved are shown. Click **Open Review Results** for the
full result (§14). This only opens the page; no LLM is needed and no LLM
dialog appears. The LLM settings dialog opens only if you later click
**Summarize results** or **Explain metrics** without a connected LLM (§12).

### 7.2 Create new project

1. Click **✦ Create new project**.
2. **Project name:** type a name (default `my_battery_project`).
3. **Projects base directory:** keep `runs/web_projects`.
4. Click **Create project and continue**. The sidebar's **Current project**
   shows the new name, and the workspace folder is shown.
5. Under **Workflow steps** open the step you need: **1 · Prepare data** →
   **Open Prepare Data**; **2 · Configure model** / **3 · Run training** →
   **Open Run Model**; **4 · Review results** → **Open Review Results**.

### 7.3 Existing project

1. Click **↗ Existing project**.
2. **Select existing project** from the list, then
   **Load project and continue**.
3. If the list is empty, the panel says `No existing project found yet…`;
   create one first (§7.2).

---

## 8. Prepare Data and Dataset Overview

Open **1. 🧪 Prepare Data** in the sidebar. The page shows
`Raw data → Preprocessing → Processed records → Feature inspection → Ready for model`
and has three tabs.

### 8.1 Tab "Dataset Overview"

1. Click **Refresh datasets**.
   - **What happens:** `tools/sync_data_and_config_manifests.py` re-reads the
     data folders and the configuration list (written to `data/metadata/`),
     and the page reloads.
   - **Expected:** every dataset that has files under `data/raw/<NAME>/` is
     listed. A dataset without raw files is not listed.
   - **Error:** if the tool fails, its error output is shown in red.
2. **Status cards** count datasets as **Ready**, **Preprocessing required**,
   **Processed incomplete** and **Missing** (= §5's `READY`, `RAW_ONLY`,
   `INCOMPLETE_PROCESSED`, `MISSING`).
3. The table lists every dataset with its paths and state. **Preparation
   Pipeline** shows the five stages.

If the page says `No datasets are available yet. Click Refresh datasets after generating manifests.`,
click **Refresh datasets**. If the message stays, no raw data is present yet:
download it (§4.2, or Quick experience step 1) and refresh again.

### 8.2 Tab "Preprocess & Inspect"

1. **Dataset:** choose a dataset. The page shows Selected dataset, Dataset
   readiness, Raw path, Processed path and Processed file count, and the exact
   command it will run.
2. Click **Run preprocessing** (or **Re-run preprocessing** if processed files
   exist).
   - The button is disabled with `Raw data missing` when there are no raw
     files, or with a warning when the raw source is not one BatteryML can
     preprocess.
   - **Log:** `runs/web_projects/<project>/logs/preprocess_<dataset>.log`,
     shown under **Latest preprocessing log**.
   - **Expected:** the processed file count grows (MATR: 180) and readiness
     becomes `READY`.
3. **Feature / artifact inspection** lets you look at the processed records.
4. **Go to Run Model** continues; **View Dataset Overview** goes back.

### 8.3 Tab "Custom Dataset / Schema Mapping" (your own cycler data)

1. **Dataset name:** a short name for your data.
2. **Source format:** `ARBIN`, `BIOLOGIC`, `LANDT`, `MACCOR` or `NEWARE`.
3. **Task type:** the task you intend to run.
4. **Upload training raw files** and **Upload validation/test raw files**;
   optionally **Optional cycler config file**.
5. Click **Save uploads**, then **Preprocess training split**,
   **Preprocess validation/test split** and **Validate processed outputs**.

Each preprocessing step writes its log next to the uploads
(`preprocess_train.log`, `preprocess_valid.log`). A run configuration for a
custom dataset is not created automatically; see
[`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) for extension.

---

## 9. Run Model: choosing a configuration

Open **2. 🚀 Run Model**. Use this page for full control over dataset, task,
model, device and seed.

If no dataset is listed yet, the page shows only
`No datasets are currently available. Prepare or upload a dataset first.` with
**Go to Prepare Data** and **Go to Home**. Get raw data first (§4, §7) and click
**Refresh datasets** (§8.1); then the four tabs below appear.

### 9.1 Tab "Configuration"

| Control | What to choose |
| --- | --- |
| **Dataset** | A dataset (only datasets with configurations are offered). |
| **Task** | `RUL prediction` or `SOH estimation`. |
| **Model** | A model with a configuration for this dataset and task. |
| **Device** | `cpu` or `cuda`. A problem with `cuda` is shown as a warning (§13). |
| **Seed** | An integer, default `0`. Same config + seed + data → same result. |

The right-hand column, **Selected config summary**, shows the configuration
file chosen for you: the **first registered configuration** for that dataset,
task and model. It cannot be picked by hand here (use §10 or the command line
for a different file). **Workspace dir** shows where the run will be written.

Configurations available in this repository (examples):

| Dataset · task | Models |
| --- | --- |
| MATR · RUL | 23 models, including `xgb`, `rf`, `ridge`, `lasso`, `elastic_net`, `svm`, `gpr`, `mlp`, `cnn`, `lstm`, `transformer`, `batlinet`, `cptransformer`, `dummy` |
| HUST · RUL, SNL · RUL | 21 models each (no `batlinet`, no `cptransformer`) |
| MATR · SOH | `elastic_net`, `lasso`, `rf`, `ridge`, `xgb` (`configs/soh/<model>/matr.yaml`) |

With a connected LLM (§12), **Explain this config** and **Suggest model/config**
describe the choice in words. They never change it.

### 9.2 Tab "Dataset Readiness"

Cards **Ready**, **Need preprocess** and **Missing raw**. **Select <dataset>**
picks a ready dataset; **Go Prepare Data** opens §8.

### 9.3 Tab "Command & Run"

1. Read the exact BatteryML command shown. (**Explain this command** needs an
   LLM.)
2. Warnings block the run when something is missing:
   `Processed dataset missing. Go to Prepare Data and run preprocessing first.`
   or `Config missing…`.
3. Click **Run BatteryML**.
   - **What happens:** BatteryML trains and evaluates with your seed and
     device. The website adds `--metric RMSE,MAE,MAPE` for RUL and
     `--metric MAE,RMSE` for SOH.
   - **Output:** `runs/web_projects/<project>/run_model/<run_id>/` with
     `run.json`, `output.log`, `config_<timestamp>.yaml` and
     `predictions_seed_<seed>_<timestamp>.pkl`. The latest output is also in
     `runs/web_projects/<project>/logs/run_model.log`.
   - **Expected:** a success message, **Review Results →**, and the
     **Output workspace** section.
4. A Run Model run cannot be stopped from the website (only Agent Workflow has
   **Stop run**). Stop the website with `Ctrl+C` if you must.

### 9.4 Tab "Logs / Debug"

Shows Last command, Return code, Log path and the end of the log. A crash of
the Python process (segmentation fault) is explained in plain words. With an
LLM, **Explain last error** and **Suggest next action** help diagnose; any fix
is up to you.

---

## 10. Uploading your own YAML configuration safely

Where: **Home → ⚡ Quick experience → Step 2 · Configure model →
Upload YAML configuration (optional)**.

1. Click the upload box and choose a `.yaml` or `.yml` file. A good starting
   point is a copy of a file from `configs/`, e.g.
   `configs/baselines/sklearn/xgb/matr_1.yaml`.
2. Click **Apply uploaded configuration**.
3. **What happens:** the file is checked. Applying only **selects** the
   configuration; it never starts a run.
4. **Expected:** the Protocol configuration and the info box switch to your
   file, and a caption `Uploaded configuration: …` appears. A validated copy
   is stored as `runs/web_projects/<project>/uploads/configs/<random id>.yaml`
   (your file name never chooses the path; nothing is overwritten).

The upload is refused, with the reasons listed under
`The uploaded configuration was not applied; the previous selection is unchanged.`, when:

| Rule | Example message |
| --- | --- |
| File name must be a plain `.yaml`/`.yml` name, no folders | `The file name … contains a path; upload a plain .yaml file.` |
| At most 256 KB, UTF-8 text, not empty | `The file is empty.` |
| Plain YAML only (no `!!python/…` tags) | `The file uses a YAML tag that is not allowed in a config …` |
| Must be a mapping of sections | `… a run config must be a mapping of sections (model, train_test_split, feature, label).` |
| Every class name must exist in BatteryML; the task must match the dataset on screen | the validator's message naming the field |
| No file/directory options except `train_test_split.cell_data_path`, which must stay inside `data/processed/` | `` `train_test_split.cell_data_path` = … resolves outside data/processed/ … `` |
| No device options (the device is chosen on the page) | `` `…` chooses a compute device; the device is selected on the page (Compute device) … `` |

Fix the file and upload again. On the command line you can run any
configuration file directly (§17); the same validators run in the website only.

---

## 11. Agent Workflow and model recommendations

Agent Workflow turns one sentence into a checked run plan. **No API key is
needed**: without an LLM the request is resolved deterministically against
the catalog of datasets, tasks and models. With an LLM configured, the LLM
may help read the request, but its answer is checked against the same
catalog.

### 11.1 Write the request

1. Open **Agent Workflow** in the sidebar (or type into the **Agent Workflow**
   box on Home and click **Analyze with Agent**).
2. In **Workflow request**, write a sentence following the pattern
   `Use [Model] on [dataset] to predict [task].`, for example
   `Use XGBoost on MATR to predict RUL.`
   While the box is empty, **Use template** fills in a sentence built from
   this installation's catalog (it never overwrites text you typed).
3. Click **Analyze**. Analysing never runs anything.

### 11.2 Read the analysis

**Workflow analysis** shows three values: **Dataset**, **Task** and
**Dataset readiness**, and a caption like
`N model(s) runnable, M blocked for this dataset and task.`

- **If the request can run**, **Backend recommended next steps** is a
  collapsed section you can open.
- **If the data is not ready**, a warning names the problem, for example
  `RUL prediction configurations are registered for MATR, but its data is MISSING; models run on processed data only. Prepare the dataset in Prepare Data first, then analyze again.`
  and `No model can be executed for this dataset/task.` Prepare the data (§7
  step 1 or §8), then click **Analyze** again.
- **If the combination has no configuration** (for example
  `Use ElasticNet on HUST to predict soh.`: there is no SOH configuration for
  HUST), the page explains why and lists alternatives, grouped as
  **<dataset> · <task>**, **<task> on other datasets** and
  **Other tasks on <dataset>**. An alternative whose model can run right now
  is a button, such as `Use Elastic Net Regression on MATR for SOH estimation`
  (this needs MATR to be `READY`). Clicking it **replaces the request** and
  analyses it again; nothing runs until you press Run. **Readiness details**
  lists the dataset states.
- **Explain this workflow** writes a plain-language explanation directly
  below the button. It needs a connected LLM; otherwise the page says so and
  offers **Open LLM settings**.

### 11.3 Choose and run

| Control | What to choose |
| --- | --- |
| **Model (runnable models only; blocked models cannot be selected)** | One of the runnable models. |
| **1-epoch smoke test — workflow validation only; not benchmark performance.** | Tick to train for one epoch only, to check the pipeline quickly (§16). |
| **Compute device** | **CPU** or **GPU**; with GPU, pick the **GPU**. |
| **Config — evaluation protocol (validated only)** | One of the validated configurations. |

Click **Run**.

- **What happens:** BatteryML trains and evaluates with seed `0` (fixed in
  this route). The live log is shown. **Stop run** stops it.
- **Output:** `runs/web_projects/<project>/agent_<dataset>_<model>/<run_id>/`
  (with `_smoke` appended for a smoke test), containing `run.json`, `run.log`,
  `stdout.log`, `stderr.log`, `config_<timestamp>.yaml` and
  `predictions_seed_0_<timestamp>.pkl`.
- **Expected:** the saved metrics; **Run again** repeats; with an LLM,
  **Explain computed result** describes the result in words.
- **Errors:** a dataset that is not `READY` blocks the run; preprocess it (§8).

---

## 12. Optional LLM assistant: Save, Connect, Explain

Skip this section if you have no API key. **Nothing else in this guide needs
one.** The assistant only explains and suggests; it never computes a
prediction, metric or benchmark value, never edits data, and never applies a
change without your click.

### 12.1 Configure it

1. Open **Settings** in the sidebar (or **Configure LLM** / **Open LLM
   settings** on a page; these open the **LLM settings** dialog).
2. **Provider:** `OpenAI`, `Gemini (OpenAI-compatible)`,
   `Claude (OpenAI-compatible)`, `Kimi / Moonshot`, `GLM / Zhipu`, or
   `Custom OpenAI-compatible`. A preset fills in the Base URL.
3. **Base URL:** your provider's OpenAI-compatible API address.
4. **API key:** your key (hidden while typing).
5. **Model:** a model name your key may use.
6. Click **Save** to store the settings without any network call
   (message `LLM settings saved. Connection not tested yet.`; sidebar
   `LLM: Configured / Not tested`), or **Connect** to store **and test** them.

**Expected after Connect:** the sidebar line changes from `LLM: Not connected`
to `LLM: Connected`. If the test fails, it shows `LLM: Connection failed`;
check the Base URL, key and model name, then **Connect** again.

The key lives **only in the open browser session**. It is not written to any
file, log, run record or export, and it is gone when the session ends. Never
put an API key into a file in this repository.

### 12.2 Where explanations are available

| Page | Buttons (need `Connected`) |
| --- | --- |
| Run Model | **Explain this config**, **Suggest model/config**, **Explain this command**, **Explain last error**, **Suggest next action** |
| Agent Workflow | **Explain this workflow**, **Explain computed result** |
| Review Results | **Summarize results**, **Explain metrics** (under **Optional LLM interpretation**) |

On Review Results the answer appears directly below the buttons, in a box
titled **🤖 LLM Assistant Output · Result interpretation** (not collapsed).
Without a connection, these buttons open the LLM settings dialog instead of
calling anything.

---

## 13. CPU and GPU

- **CPU always works** and is the default (`cpu` in Quick experience and Run
  Model, **CPU** in Agent Workflow, `--device cpu` on the command line).
- **GPU:** choose `cuda` (Quick experience, Run Model), **GPU** + a **GPU**
  number (Agent Workflow), or `--device cuda:0` on the command line.
- The website checks the GPU **before** running. If CUDA is unavailable, or the
  installed PyTorch was not built for your GPU's architecture (for example an
  old PyTorch on an RTX 3090), the GPU option is refused with the reason.
  **There is no automatic fallback to CPU**: choose CPU yourself.
- One device per run. Multi-GPU training is not offered in the website.
- Classical models (XGBoost, random forest, ridge, …) gain little from a GPU;
  neural networks (CNN, LSTM, Transformer) gain more.
- `CUDA out of memory` → use CPU or a smaller model.

---

## 14. Review Results: run history, results and metrics

Open **3. 📈 Review Results**. It lists the runs of the current project (or of
`runs/web_projects/default`) from Quick experience, Run Model and Agent
Workflow.

### 14.1 Run history

A compact table: **Run** (end of the run id), **Started**
(`YYYY-MM-DD HH:MM UTC`), **Experiment** (model · dataset · task), **Seed**,
**Device**, **Status**, **Duration**, **Metrics**.

Statuses: `COMPLETED`, `FAILED`, `RUNNING`, `PLANNED`, `CANCELLED`,
`INTERRUPTED`.

### 14.2 Open a run

1. **Open a run:** choose a run from the list.
2. **Run details · …** (expandable) shows a status pill, the full run id and a
   table: Run ID, Started, Project, Run directory, Dataset, Task, Model,
   Config, Seed, Epochs, Device, Status, Duration, Return code, Git commit.
3. **Metrics** shows exactly what BatteryML saved (RUL: RMSE, MAE, MAPE in
   cycles; SOH: MAE, RMSE). A failed run records no metrics.
4. **Scientific plots** (§15).
5. **Prepare run bundle** then **Download run bundle** downloads the whole run.
   **Artifacts** lists the files; **Download a file** / **Download selected
   file** downloads one.
6. **Resume training** appears only for neural-network models that saved a
   checkpoint.

### 14.3 Selected result

Tabs **Summary**, **Metrics**, **Predictions**, **Logs** and **Files**, then
**Benchmark reference** (§16) and **Optional LLM interpretation** (§12).

If the page says
`No runs recorded yet for this project (runs/web_projects/default). A run appears here as soon as it is started.`
and `No run is selected. Choose one in Run History above.`, the project has no
run yet: run one (§7, §9 or §11) and come back. Check that the sidebar shows
the project you ran in.

### 14.4 Reading the numbers

- **RMSE** and **MAE** are in the task's unit (cycles for RUL). Lower is better.
- **MAPE** is a relative error. Lower is better.
- A smoke-test result (one epoch) is **not** a performance result (§16).

From the command line, a project's runs and their states:

```bash
python -m scripts.inspect_run_state runs/web_projects/default
```

---

## 15. Scientific figures and figure export

Under **Scientific plots** for a run:

- **Prediction artifact:** the `.pkl` the figures are drawn from.
- **Axis range:** **Full range** or **Zoomed view**.
- **Cell** (SOH): choose a cell to plot its SOH over cycles.
- Parity plot (predicted vs. true) and residuals, a
  **Per-cell error summary (display diagnostic)** and an
  **Estimate range check**.

**Export figures** (expandable):

1. Read the equivalent command shown.
2. Click **Prepare figure package**, then
   **Download figure package (.zip)**.

The same export from the command line (replace the path with your file):

```bash
python -m scripts.export_result_figures --artifact runs/web_projects/default/quickstart/<run_id>/predictions_seed_0_<timestamp>.pkl --inspect
python -m scripts.export_result_figures --artifact runs/web_projects/default/quickstart/<run_id>/predictions_seed_0_<timestamp>.pkl --output runs/figure_exports/my_run
```

`--inspect` only describes the file. The export folder (must be new) holds
`parity.svg`/`.png`, `residuals.svg`/`.png`, their `_zoom` versions,
`plotted_samples.csv`, `figure_sources.json` and `figure_notes.md`; an SOH
result also gets `soh_cycle_<cell>.svg`/`.png`/`.csv` and `per_cell_errors.csv`.
Options: `--cell-id`, `--list-cells`, `--dpi`. The command only draws; it never
trains or evaluates. Details: [`docs/VISUALIZATION.md`](./docs/VISUALIZATION.md).

---

## 16. Benchmark comparison and smoke-test limits

**Benchmark reference** (Review Results) places a run beside the published
BatteryML RUL benchmark (RMSE in cycles, from
[`README_BATTERYML.md`](./README_BATTERYML.md)) **only when the comparison is
fair**. Otherwise it says why, with one of these reasons:

- the published benchmark covers RUL prediction only (every SOH run);
- the run's train/test split does not identify a published subset
  (`MATR1`, `MATR2`, …);
- the model has no corresponding published baseline;
- the metric or unit is not RMSE in cycles;
- it is a **1-epoch smoke run**;
- the published value is a threshold (`>1000`), not a number;
- the run saved no RMSE.

**Smoke tests** (Agent Workflow checkbox, or `--epochs 1` on the command line)
train for one epoch. They prove that loading, training and evaluation work on
your machine. **Their scores are not results**: do not report or compare
them.

---

## 17. Command line: copyable commands

Run these inside `BatteryML-Agent` with `conda activate batteryml`.

```bash
# Health check (13 checks)
python -m scripts.verify.run_all

# Download (MATR, HUST, CALCE, RWTH)
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR

# Preprocess
python batteryml_web/core/run_batteryml_cli.py preprocess MATR data/raw/MATR data/processed/MATR

# Readiness (read-only)
python -m scripts.inspect_dataset_readiness MATR

# Full run: XGBoost on MATR, CPU, seed 0
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/sklearn/xgb/matr_1.yaml --workspace runs/cli_matr_xgb --train --eval --device cpu --seed 0

# Smoke test: one epoch of CNN (not a result)
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/nn_models/cnn/matr_1.yaml --workspace runs/smoke_matr_cnn --train --eval --device cpu --seed 0 --epochs 1

# SOH run: XGBoost on MATR
python batteryml_web/core/run_batteryml_cli.py run configs/soh/xgb/matr.yaml --workspace runs/cli_matr_soh_xgb --train --eval --device cpu --seed 0

# Start the website
python -m streamlit run batteryml_web/app.py
```

Rules for `run`:

- The configuration file is **positional**. `run --config …` fails with
  `unrecognized arguments: --config`.
- `--train` and `--eval` are both required. Without them the command prints
  the workspace path and exits.
- Always give `--seed`.
- `--device cuda:0` uses the first GPU.
- Other options: `--metric`, `--epochs`, `--ckpt-to-resume`,
  `--skip_if_executed` (see `python batteryml_web/core/run_batteryml_cli.py run --help`).

The run ends by printing the scores, e.g. `{'RMSE': …, 'MAE': …, 'MAPE': …}`.
The workspace holds `config_<timestamp>.yaml` and
`predictions_seed_<seed>_<timestamp>.pkl`. To print the saved scores:

```bash
python -c "import pickle; print(pickle.load(open('runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl', 'rb'))['scores'])"
```

Preprocessing input types: datasets `CALCE HNEI HUST MATR OX RWTH SNL UL_PUR`;
cycler formats `ARBIN BATTERYARCHIVE BIOLOGIC INDIGO LANDT MACCOR NEWARE NOVONIX`
(some cycler formats have no preprocessor; see `REG-01` in §3.5).

---

## 18. Troubleshooting and known limitations

### 18.1 Troubleshooting

| Message or symptom | Cause | Fix |
| --- | --- | --- |
| `conda: command not found` | The terminal was not reopened after `conda init`. | Open a new terminal; if needed, run `$HOME/miniconda3/bin/conda init bash` again. |
| `CondaToSNonInteractiveError` | Terms of Service not accepted. | §2.3 step 4. |
| Prompt shows `(base)`, not `(batteryml)` | Environment not active. | `conda activate batteryml` |
| `ModuleNotFoundError: No module named 'numpy'` (or `torch`, `streamlit`) | Environment not active, or installation incomplete. | `conda activate batteryml`, then §3.2 and §3.4. |
| `ModuleNotFoundError: No module named 'batteryml_web'` | Started outside the folder, or with plain `streamlit run`. | `cd ~/BatteryML-Agent`; use `python -m streamlit run batteryml_web/app.py`. |
| `ModuleNotFoundError: … 'batteryml' is not a package` | `python bin/batteryml.py` was used. | Use `python batteryml_web/core/run_batteryml_cli.py …`. |
| `error: Could not download …` (`ProxyError`, `ConnectionError`, `403 Forbidden`) | Download server unreachable. | Fix the network; run the download again. |
| `FileNotFoundError: Batch file not found` / `OSError: … file signature not found` | Incomplete raw files. | Download again, then preprocess again (§4.4). |
| Dataset shows `INCOMPLETE_PROCESSED` | Preprocessing failed. | Preprocess again. |
| `AssertionError: data/processed/MATR` | Not preprocessed. | §4.3. |
| `KeyError: 'b1c1'` | Processed folder is empty. | Fix preprocessing, run it again. |
| `unrecognized arguments: --config` | `--config` in front of the file. | Remove `--config`. |
| Run prints a path and ends | `--train --eval` missing. | Add both. |
| Health check `[FAIL] REG-01` | Expected upstream gap. | Nothing. |
| `No datasets are available yet.` | The website has not re-read the data folders. | **Prepare Data → Refresh datasets**. |
| `No run is selected…` in Review Results | No run in this project yet. | Run one (§7 or §9). |
| Quick experience: `Process the dataset in step 1 before starting a model run.` | Data not processed. | Step 1 → **Process … data**. |
| Run Model: `Processed dataset missing…` | Data not processed. | §8.2. |
| Run Model: `No datasets are currently available. Prepare or upload a dataset first.` | No raw data, or the list was not refreshed. | Get raw data (§4), then **Prepare Data → Refresh datasets**. |
| GPU option refused | No CUDA, or PyTorch not built for this GPU. | Choose CPU, or fix PyTorch (§3.3, §13). |
| `CUDA out of memory` | Model does not fit on the GPU. | Choose CPU. |
| LLM buttons open the settings dialog | LLM not connected. | §12, or ignore; nothing else needs it. |
| `LLM: Connection failed` | Wrong Base URL, key or model, or no network. | Correct them, click **Connect**. |
| Browser cannot open `localhost:8501` | Website runs on a remote server. | SSH tunnel (§6). |

### 18.2 Known limitations

- **CPTransformer** needs code from a separate repository placed in
  `third_party/BatteryLife/`, which is not included; **PBT** needs
  `third_party/PBT` assets, which are not included. All other models work
  without them. The health check (`VAL-01`) mentions this.
- BatteryLife datasets (`MICH`, `MICH_EXP`, `SDU`, `Stanford_2`, `Tongji`,
  `XJTU`, `ISU_ILCC`) have no BatteryML preprocessor: they can be detected
  (`RAW_ONLY`) but not preprocessed here.
- `HNEI`, `SNL`, `UL_PUR` and `OX` must be downloaded by hand.
- Not every dataset × task × model has a configuration; the website offers only
  combinations that exist. SOH configurations exist for MATR only.
- Quick experience and Run Model runs cannot be stopped from the website; only
  Agent Workflow has **Stop run**.
- One device per run; no multi-GPU training in the website.
- In Agent Workflow and Quick experience the seed is fixed at `0`; use Run
  Model or the command line for other seeds.
- No login: protect port 8501 (§6).
- Tested on Linux only.

### Updating to a newer version

Stop the website (`Ctrl+C`), then:

```bash
cd ~/BatteryML-Agent
conda activate batteryml
git pull
pip install -r requirements.txt
python -m scripts.verify.run_all
```

### Starting over

Removes only the environment; code, data and runs stay:

```bash
conda deactivate
conda env remove -n batteryml -y
```

Then repeat from §3.1.

---

## 19. Reproducibility, verification status and citation

### 19.1 What makes a run reproducible

- Every run records its command, configuration, **seed**, **device**, git
  commit, status, duration and saved metrics in `run.json`, next to the exact
  configuration BatteryML used (`config_<timestamp>.yaml`).
- Same configuration + seed + processed data → the same metrics for
  deterministic models. Neural networks on GPU may differ slightly between
  runs.
- To repeat a website run on the command line, copy the command shown on the
  Run Model **Command & Run** tab (or the command in `run.json`).
- Report the dataset, task, model, configuration file, seed, device and the
  git commit (`git rev-parse HEAD`) with your numbers.

### 19.2 What was verified for this guide

| Item | Status |
| --- | --- |
| Installation steps §2–§3 (Miniconda, environment, PyTorch CPU, requirements, health check), on Linux x86_64 without GPU | [VERIFIED] in the V1 install test; tested versions: conda 26.7.1, Python 3.10.21, torch 2.14.0+cu130, numpy 1.26.4, scipy 1.15.3, pandas 2.3.3, numba 0.67.0, scikit-learn 1.7.2, xgboost 3.2.0, h5py 3.16.0, matplotlib 3.10.9, streamlit 1.64.0 |
| `python -m scripts.verify.run_all` in a fresh clone of the current `main` | [VERIFIED] `10 passed, 1 failed, 2 skipped of 13 check(s)`, exit code `1` (documentation environment: Linux, Python 3.11, no data, no GPU) |
| `--help` of every command in §17; `run` without data (`AssertionError: data/processed/MATR`), with `--config` (`unrecognized arguments: --config`) and without `--train --eval` (prints the workspace path); `inspect_dataset_readiness` and `preprocess_sweep.py --dry-run` without data | [VERIFIED] in the documentation environment |
| `download MATR` | [VERIFIED] only the failure path: `data.matr.io` was blocked (`ProxyError … 403 Forbidden`), one `error:` line, nothing written. A successful download was not possible here |
| Starting the website from a fresh clone with `python -m streamlit run batteryml_web/app.py`; the labels of Home, the three start panels, Quick experience steps 1–4, Prepare Data, Run Model's empty state, Agent Workflow analysis, Review Results' empty state and the LLM settings fields | [VERIFIED] in a browser, without data |
| Run Model's four tabs, Review Results with a run, figures, run bundles, YAML upload, LLM explanations | [STATIC] read from the current source code (`batteryml_web/`); they need data or an API key to reach |
| MATR preprocessing gives 180 files | Observed on the project's server, not in the documentation environment |
| A complete training run, metrics, figures | Not executed in the documentation environment (no data) |
| GPU, real LLM providers, macOS, Windows/WSL2 | Not verified |

### 19.3 Further reading

| Document | Content |
| --- | --- |
| [`README_BATTERYML.md`](./README_BATTERYML.md) | The original BatteryML README: paper, datasets, benchmark tables |
| [`dataprepare.md`](./dataprepare.md) | Manual download of every dataset |
| [`docs/VERIFY.md`](./docs/VERIFY.md) | What each health check tests |
| [`docs/READINESS.md`](./docs/READINESS.md) | Readiness rules |
| [`docs/AGENT_WORKFLOW.md`](./docs/AGENT_WORKFLOW.md) | Agent Workflow internals |
| [`docs/SOH_PROTOCOL.md`](./docs/SOH_PROTOCOL.md) | SOH task protocol |
| [`docs/VISUALIZATION.md`](./docs/VISUALIZATION.md) | Figures and export |
| [`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) | Adding your own model |
| [`README_v1.md`](./README_v1.md) / [`README_v1.zh-CN.md`](./README_v1.zh-CN.md) | Archived V1 installation guides |

### 19.4 Citation

BatteryML-Agent builds on BatteryML. If you use it, please cite:

```bibtex
@inproceedings{zhang2024batteryml,
  title={Battery{ML}: An Open-source Platform for Machine Learning on Battery Degradation},
  author={Han Zhang and Xiaofan Gui and Shun Zheng and Ziheng Lu and Yuqi Li and Jiang Bian},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
