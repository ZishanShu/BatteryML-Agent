<div align="center">

# BatteryML-Agent — Installation Guide

**English** · [简体中文](./README.zh-CN.md)

</div>

BatteryML-Agent is a web application for battery lifetime (RUL) and
state-of-health (SOH) prediction, built on
[BatteryML](https://github.com/microsoft/BatteryML). This guide installs it
from an empty machine to a running website, one command at a time.

When you finish, you will have:

- a conda environment called `batteryml` with BatteryML and all its dependencies;
- the MATR battery dataset downloaded and preprocessed;
- the website running at `http://localhost:8501`;
- one model trained and evaluated on MATR, with its results on disk and on screen.

Follow the steps **in order**. Every step ends with a check: do not go on until
the check passes. Every command is run from a terminal (Linux / macOS
Terminal; on Windows use WSL2 — see §0).

---

## Steps

0. [What you need](#0-what-you-need)
1. [Check the machine](#1-check-the-machine)
2. [Install Miniconda](#2-install-miniconda)
3. [Download the code](#3-download-the-code)
4. [Create the Python environment](#4-create-the-python-environment)
5. [Install PyTorch](#5-install-pytorch)
6. [Install the remaining dependencies](#6-install-the-remaining-dependencies)
7. [Run the health check](#7-run-the-health-check)
8. [Download the MATR dataset](#8-download-the-matr-dataset)
9. [Preprocess the dataset](#9-preprocess-the-dataset)
10. [Check dataset readiness](#10-check-dataset-readiness)
11. [Start the website](#11-start-the-website)
12. [Your first run in the website](#12-your-first-run-in-the-website)
13. [Running from the command line instead](#13-running-from-the-command-line-instead)
14. [Where the results are](#14-where-the-results-are)
15. [Optional: connect an LLM assistant](#15-optional-connect-an-llm-assistant)
16. [Every time you come back](#16-every-time-you-come-back)
17. [Updating to a newer version](#17-updating-to-a-newer-version)
18. [Troubleshooting](#18-troubleshooting)
19. [Known limitations](#19-known-limitations)
20. [Tested versions and what was verified](#20-tested-versions-and-what-was-verified)

---

## 0. What you need

| Item | Requirement |
| --- | --- |
| Operating system | **Linux x86_64** (tested). macOS: not tested. Windows: install [WSL2 with Ubuntu](https://learn.microsoft.com/windows/wsl/install) and follow the Linux steps inside it — not tested. |
| Disk space | **12 GB** for the software (measured: Miniconda + environment 8.0 GB, pip download cache 3.5 GB), **plus** space for the raw and processed battery data. |
| Internet | Needed during installation and data download. Hosts used: `repo.anaconda.com`, `pypi.org`, `files.pythonhosted.org`, `github.com`, `data.matr.io`. |
| GPU | **Not required.** Everything runs on CPU. An NVIDIA GPU is used if present (§5). |
| Admin rights | Not required. Everything installs into your home directory. |

---

## 1. Check the machine

```bash
git --version
curl --version
```

**Check:** both print a version. If `git` is missing, install it with your
system's package manager (on Ubuntu: `sudo apt-get install -y git`). If
`curl` is missing, install it the same way (`sudo apt-get install -y curl`).

Optional — only if the machine has an NVIDIA GPU:

```bash
nvidia-smi
```

**Check:** a table appears. Note the **`CUDA Version`** printed in its top-right
corner; you need it in §5. If the command is not found, the machine has no
usable NVIDIA driver and everything will run on CPU — that is fine.

---

## 2. Install Miniconda

Miniconda provides `conda`, which creates an isolated Python environment for
BatteryML so nothing else on your machine is changed.

> **Already have conda or Anaconda?** Run `conda --version`. If it prints a
> version, skip 2.1–2.3 and go straight to **2.4**.

### 2.1 Download the installer

```bash
cd ~
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

On other platforms, replace the file name in the URL:

| Platform | File name |
| --- | --- |
| Linux, ARM (aarch64) | `Miniconda3-latest-Linux-aarch64.sh` |
| macOS, Apple Silicon (M1/M2/M3/M4) | `Miniconda3-latest-MacOSX-arm64.sh` |
| macOS, Intel | `Miniconda3-latest-MacOSX-x86_64.sh` |

**Check:** `ls -lh Miniconda3-latest-*.sh` shows a file of roughly 120–190 MB.

### 2.2 Run the installer

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

`-b` installs without questions; `-p` sets the install folder to
`~/miniconda3`.

**Check:** the output ends with `installation finished.`

### 2.3 Connect conda to your terminal

```bash
$HOME/miniconda3/bin/conda init bash
```

(On macOS, whose default shell is zsh, run `$HOME/miniconda3/bin/conda init zsh` instead.)

Now **close the terminal and open a new one.** This step is required: the
change only takes effect in a new terminal.

**Check:** in the new terminal, the prompt starts with `(base)`, and

```bash
conda --version
```

prints a version (the test install printed `conda 26.7.1`).

### 2.4 Accept the Anaconda Terms of Service

Current Miniconda versions refuse to create environments until you accept
the Terms of Service of the default channels. Without this step, §4 fails
with `CondaToSNonInteractiveError: Terms of Service have not been accepted`.

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**Check:** the two lines printed are
`accepted Terms of Service for https://repo.anaconda.com/pkgs/main` and
`accepted Terms of Service for https://repo.anaconda.com/pkgs/r`.

---

## 3. Download the code

```bash
cd ~
git clone https://github.com/ZishanShu/BatteryML-Agent.git
cd BatteryML-Agent
```

**Check:**

```bash
ls
```

The list includes `batteryml`, `batteryml_web`, `configs`, `requirements.txt`
and `README.md`.

> **Important:** from here on, every command in this guide is run **inside the
> `BatteryML-Agent` folder**. Whenever you open a new terminal, run
> `cd ~/BatteryML-Agent` first.

---

## 4. Create the Python environment

```bash
conda create -n batteryml python=3.10 -y
conda activate batteryml
```

**Check:** the prompt now starts with `(batteryml)` instead of `(base)`, and

```bash
python --version
```

prints `Python 3.10.x`.

Python 3.10 is the version this guide was tested with. Do not use Python 3.13:
`requirements.txt` requires `numpy<2.0`, which has no Python 3.13 builds.

---

## 5. Install PyTorch

PyTorch is installed on its own first, so you can check the GPU before
installing everything else.

```bash
python -m pip install --upgrade pip
pip install torch
```

This is the largest download of the whole installation (in the test install:
PyTorch 2.14.0 plus its NVIDIA CUDA libraries, about 5.5 GB on disk, 222 s on
a fast connection). The same package works on machines **with and without**
a GPU.

**Check:**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

| What it prints | Meaning |
| --- | --- |
| `2.14.0+cu130 13.0 False` on a machine **without** a GPU | Correct. Everything will run on CPU. |
| `… True` on a machine **with** a GPU | Correct. The GPU can be used. |
| `… False` on a machine **with** a GPU | Your NVIDIA driver is older than the CUDA version PyTorch was built for. See below. |

Two lines may appear **above** the version line:
`UserWarning: Failed to initialize NumPy: No module named 'numpy'` and
`cpu = _conversion_method_template(device=torch.device("cpu"))`. This is
expected at this point — numpy is installed in §6. Ignore it; it disappears
after §6.

**Only if you have a GPU and it printed `False`:** compare the second number
(the CUDA version PyTorch was built for, e.g. `13.0`) with the `CUDA Version`
that `nvidia-smi` showed in §1. If `nvidia-smi` shows a lower number, either
update the NVIDIA driver, or replace PyTorch with a build for your CUDA
version. Get the exact command from the selector at
<https://pytorch.org/get-started/locally/> (choose *Pip*, *Linux*, your CUDA
version). It has this form — `cu126` means CUDA 12.6:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Then run the check above again.

---

## 6. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

This installs numpy, scipy, pandas, numba, scikit-learn, xgboost, h5py,
matplotlib, streamlit and the other packages BatteryML needs (53 s in the
test install).

If you are logged in as `root`, pip prints a warning that starts with
`WARNING: Running pip as the 'root' user`. Inside the conda environment this
warning is harmless.

**Check:**

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

prints `all imports OK`.

> **Do not run `pip install .`** The upstream BatteryML README asks for it, but
> BatteryML-Agent does not need it: the website and the command-line tool
> always load BatteryML from this folder. A `pip install .` would copy a
> snapshot of the code into the environment, and that copy would **not** be
> updated when you later run `git pull`.

Optional — free 3.5 GB of disk by deleting pip's download cache:

```bash
pip cache purge
```

---

## 7. Run the health check

```bash
python -m scripts.verify.run_all
```

It runs 13 checks and prints one line per check. On a new installation
without data, the last line is exactly:

```
10 passed, 1 failed, 2 skipped of 13 check(s)
```

**This result is correct.** The command also exits with code `1` because of
the one failure; that is expected. What each line should say:

| Check | Expected | Why |
| --- | --- | --- |
| `ENV-01` `CLI-01` `CFG-01` `VAL-01` `EXT-01` `DEV-01` `INTENT-01` `RUN-01` `UI-01` `FIG-01` | `[PASS]` | The installation is complete. |
| `DATA-01` `SPLIT-01` | `[SKIP]` | No data has been downloaded yet (§8). |
| `REG-01` | `[FAIL]` | Upstream BatteryML lists six cycler formats (`BATTERYARCHIVE`, `BIOLOGIC`, `INDIGO`, `LANDT`, `MACCOR`, `NOVONIX`) that have no preprocessor. It does not affect anything in this guide. |

The first time it runs, two lines starting with a date may appear between
the checks: `INFO - Failed to extract font properties from …` and
`INFO - generated new fontManager`. Matplotlib is building its font cache;
this happens once and is not an error.

If any check other than `REG-01` says `[FAIL]`, read its message — it names
the problem — and see §18. The full report is written to
`runs/verify/verification_report.json`.

---

## 8. Download the MATR dataset

MATR contains 180 LFP/graphite cells and is the reference dataset of this
guide.

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

This downloads four large `.mat` files from `data.matr.io`. It takes a long
time; leave it running.

If the machine cannot reach `data.matr.io`, the command stops with one line
that starts with `error: Could not download data/raw/MATR/MATR_batch_….mat`
and contains `ProxyError`, `ConnectionError` or `403 Forbidden`, and it
writes **nothing** for that file. Fix the network first (open
<https://data.matr.io> in a browser on the same machine; a university or
company proxy may block it), then run the same command again.

Running the command again is always safe. Files that are already complete
are skipped (`[INFO] … already exists and is complete (… bytes). Skip it.`);
files that are missing, empty or cut off are downloaded again.

> Always start BatteryML through `batteryml_web/core/run_batteryml_cli.py`.
> Running `python bin/batteryml.py` directly fails with
> `ModuleNotFoundError: No module named 'batteryml.…'; 'batteryml' is not a package`.

**Check:**

```bash
ls -lh data/raw/MATR
find data/raw/MATR -type f -size 0
```

The first command lists exactly these four files, each much larger than zero:
`MATR_batch_20170512.mat`, `MATR_batch_20170630.mat`,
`MATR_batch_20180412.mat`, `MATR_batch_20190124.mat`.
The second command must print **nothing**.

> **If a file is missing or the second command printed a file name:** run the
> download command above again; it replaces empty and incomplete files. Only
> if it then stops with `… already exists (… bytes), but whether it is
> complete cannot be checked` (the server could not be reached to compare
> sizes), delete the folder and download again:
>
> ```bash
> rm -rf data/raw/MATR
> python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
> ```
>
> Folders downloaded with an older version of this repository can still hold
> an empty or cut-off file: that version created each file *before*
> downloading it and then skipped it on the next run. The download command
> now writes to a temporary `.part` file and renames it only when it is
> complete, so it no longer leaves broken files behind.

Other datasets:

- `HUST`, `CALCE` and `RWTH` can be downloaded the same way — replace `MATR`
  in both places of the command.
- `HNEI`, `SNL`, `UL_PUR` and `OX` cannot be downloaded by this command.
  Download them by hand as described in [`dataprepare.md`](./dataprepare.md)
  and put the files in `data/raw/<NAME>/`.

---

## 9. Preprocess the dataset

The models do not read the raw files. Preprocessing converts them into
BatteryML's own format, one `.pkl` file per cell.

```bash
python batteryml_web/core/run_batteryml_cli.py preprocess MATR data/raw/MATR data/processed/MATR
```

**Check:**

```bash
ls data/processed/MATR | wc -l
```

prints `180`.

If preprocessing stops with
`FileNotFoundError: Batch file not found: data/raw/MATR/MATR_batch_….mat`,
the download in §8 is incomplete: redo §8, then run this step again.

If it stops with `OSError: Unable to synchronously open file (file signature
not found)`, one of the `.mat` files is empty or cut off — a file left by an
older version of the download command (see the box in §8). Run the download
command in §8 again (it replaces such files), then this step again.

A failed preprocessing run leaves an **empty** `data/processed/MATR` folder
behind. That is why §10 then reports `INCOMPLETE_PROCESSED`, and a model run
fails with `KeyError: 'b1c1'`. Running this step again successfully fixes both.

---

## 10. Check dataset readiness

```bash
python -c "
from batteryml_web.core.readiness import list_dataset_readiness
for r in list_dataset_readiness():
    print(f'{r.dataset:<12} {r.status.value}')
"
```

It prints one line per dataset. **Check:** the line for MATR is

```
MATR         READY
```

The four possible states:

| State | Meaning | What to do |
| --- | --- | --- |
| `READY` | Preprocessed `.pkl` files are present. | Nothing. It can be used. |
| `RAW_ONLY` | Raw files are present, preprocessed files are not. | Run §9 for this dataset. |
| `INCOMPLETE_PROCESSED` | `data/processed/<NAME>` exists but contains no `.pkl` files — preprocessing failed or was interrupted. | Run §9 again. |
| `MISSING` | Neither raw nor preprocessed files. | Run §8 for this dataset. |

Datasets you did not download are listed as `MISSING`. That is normal.

---

## 11. Start the website

```bash
python -m streamlit run batteryml_web/app.py
```

Start it from the `BatteryML-Agent` folder — the settings in
`.streamlit/config.toml` are only read from there.

**Check:** the terminal prints

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

Open `http://localhost:8501` in a browser. The page title is
**LLM-guided BatteryML Protocol Platform**.

The website runs until you press `Ctrl+C` in this terminal (it prints
`Stopping...`). Closing the terminal stops it too.

While you use the website, this terminal prints lines starting with a date,
such as ``Please replace `use_container_width` with `width`.`` or
``Please replace `st.components.v1.html` with `st.iframe`.``. These are
Streamlit deprecation warnings, not errors; the website keeps working. Errors
that matter are shown on the web page itself.

### If the code runs on a remote server

The browser runs on your own computer, so forward the port. On **your own
computer**, open a new terminal and run (replace `user@server` with how you log
in to the server):

```bash
ssh -L 8501:localhost:8501 user@server
```

Keep that terminal open and open `http://localhost:8501` in your local
browser.

> The website has no login. Anyone who can reach port 8501 can use it. Prefer
> the SSH tunnel above over opening the port to a network.

---

## 12. Your first run in the website

The website has a sidebar on the left. The steps below use the exact names of
its buttons and tabs.

### 12.1 Create a project

1. On the **Home** page, under **Create New Project**, keep the default
   **Project name** `my_battery_project` (or type your own).
2. Keep **Projects base directory** as `runs/web_projects`.
3. Click **Create New Project**.

**Check:** the sidebar shows your project under **Current project**, and the
folder `runs/web_projects/my_battery_project/` now exists. All runs of this
project are saved inside it.

### 12.2 Register the data with the website

1. In the sidebar, click **1. 🧪 Prepare Data**.
2. On the **Dataset Overview** tab, click **Refresh datasets**.

**Check:** a card for **MATR** appears. Its last two lines read
`Status: READY` and **Ready**. (Clicking **Refresh datasets** runs
`tools/sync_data_and_config_manifests.py`, which writes the two files in
`data/metadata/` that the website reads. Click it again whenever you add or
preprocess a dataset.)

If the card instead ends with `Status: INCOMPLETE_PROCESSED` and
**Processed incomplete**, or `Status: RAW_ONLY` and
**Preprocessing required**, go back to §9, then click **Refresh datasets**
again.

### 12.3 Train and evaluate a model

Use the **Agent Workflow** page: it shows live progress and has **Stop run**.
Its runs appear on the **Review Results** page (so do runs from the Run Model
page and the Home page's **Run MATR Demo**, §12.5).
It does **not** need an API key — the request is interpreted by fixed rules.

1. In the sidebar, click **Agent Workflow**.
2. In **Workflow request**, type:

   ```text
   Use XGBoost on MATR to predict RUL.
   ```

   (**Use template** fills in an example of the form
   `Use [Model] on [dataset] to predict [task].`, for example
   `Use CNN on HUST to predict battery lifetime.`. If you click it, replace
   the text with the line above.)
3. Click **Analyze**. Under **Workflow analysis** the page shows
   **Dataset** `MATR`, **Task** `RUL prediction` and **Dataset readiness**
   `READY`, followed by the models that can run.

   If it shows **Dataset readiness** `INCOMPLETE_PROCESSED` (or `RAW_ONLY`)
   and the message `xgb cannot run for this dataset and task.`, MATR has not
   been preprocessed: go back to §9, then **Prepare Data → Refresh datasets**,
   then click **Analyze** again. No model can be selected until then.
4. In **Model (runnable models only; blocked models cannot be selected)**, choose **XGBoost**.
5. In **Config — evaluation protocol (validated only)**, choose
   `configs/baselines/sklearn/xgb/matr_1.yaml`. The list also offers
   `clo.yaml` and `matr_2.yaml`: they are different test splits of the same
   MATR data, so pick one and keep it when comparing runs.
6. Leave **1-epoch smoke test — workflow validation only; not benchmark
   performance.** unticked. (It only matters for neural-network models; see §13.)
7. Under **Compute device**, choose **CPU**, or **GPU** if you have one.
8. Click **Run**. Progress is shown live on the page. While it runs, **Stop run**
   cancels it.

**Check:** the run finishes with a success message and shows RMSE, MAE and
MAPE.

### 12.4 Look at the results

1. In the sidebar, click **3. 📈 Review Results**.
2. Under **Run history**, choose your run in **Open a run** (the newest run is
   selected by default).
3. Use the tabs **Summary**, **Metrics**, **Predictions**, **Logs** and
   **Files**.

The **Predictions** tab shows observed vs. predicted values, the residuals
and the prediction table.

### 12.5 The Run Model page

The sidebar entry **2. 🚀 Run Model** runs the same BatteryML command through
a form: tabs **Configuration** (choose **Dataset**, **Task**, **Model**,
**Device**, **Seed**), **Dataset Readiness**, **Command & Run** (shows the
exact command, then **Run BatteryML**) and **Logs / Debug**.

Each run started there gets its own folder (§14) and is listed on
**Review Results** like an Agent Workflow run. The **Workspace dir** and the
command shown before you press **Run BatteryML** already contain that run's
folder, `runs/web_projects/<project>/run_model/<run_id>`. Unlike Agent
Workflow, this page has no **Stop run** button: the run ends when BatteryML
finishes.

---

## 13. Running from the command line instead

Everything the website runs is one BatteryML command. You can run it
directly (with the `batteryml` environment active, inside the
`BatteryML-Agent` folder).

**A full run** — trains and evaluates XGBoost on MATR, on CPU:

```bash
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/sklearn/xgb/matr_1.yaml --workspace runs/cli_matr_xgb --train --eval --device cpu --seed 0
```

**A smoke test** — one training epoch of the CNN model, only to confirm that
training works on this machine:

```bash
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/nn_models/cnn/matr_1.yaml --workspace runs/smoke_matr_cnn --train --eval --device cpu --seed 0 --epochs 1
```

Use `--device cuda:0` instead of `--device cpu` to use the first GPU.

Three rules for this command:

- The config file is given **without** `--config` in front of it. `run --config …`
  fails with `unrecognized arguments: --config`.
- `--train` and `--eval` are both required. Without them the command only
  prints the workspace path and does nothing.
- Always give `--seed`, so the run can be repeated with the same result.

When the run ends, the last line printed is the scores, for example
`{'RMSE': …, 'MAE': …, 'MAPE': …}`.

> **A smoke-test score is not a result.** One epoch trains almost nothing; it
> only shows that data loading, training and evaluation work. Do not compare
> it with anything. The published BatteryML benchmark (RMSE in cycles; lower
> is better; MATR1 and MATR2 are different test sets) is in
> [`README_BATTERYML.md`](./README_BATTERYML.md).

---

## 14. Where the results are

BatteryML does not write a metrics spreadsheet. It prints the scores at the
end of the run and saves them, together with the predictions, in one `.pkl`
file.

**Runs from Agent Workflow** (§12.3):

```
runs/web_projects/<project>/agent_<dataset>_<model>/<run_id>/
├── run.json                                  run record: command, status, seed, device, git commit, scores
├── run.log   stdout.log   stderr.log         everything the run printed
├── config_<timestamp>.yaml                   the config as BatteryML used it
└── predictions_seed_<seed>_<timestamp>.pkl   predictions and scores
```

**Runs from the Run Model page or the Home page's Run MATR Demo button**
(§12.5):

```
runs/web_projects/<project>/run_model/<run_id>/    Run Model page
runs/web_projects/<project>/quickstart/<run_id>/   Run MATR Demo
├── run.json                                  run record: command, status, seed, device, git commit, scores
├── output.log                                everything the run printed
├── config_<timestamp>.yaml                   the config as BatteryML used it
└── predictions_seed_<seed>_<timestamp>.pkl   predictions and scores
```

The pages also still write `runs/web_projects/<project>/logs/run_model.log`
(Run Model) and `runs/web_projects/<project>/logs/matr_demo.log` (demo), which
hold the most recent run's output.

**Runs from the command line with `--workspace <folder>`** (§13):

```
<folder>/
├── config_<timestamp>.yaml
└── predictions_seed_<seed>_<timestamp>.pkl
```

To print the scores saved in a `.pkl` file (run inside the `BatteryML-Agent`
folder; replace the path with your file):

```bash
python -c "import pickle; print(pickle.load(open('runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl', 'rb'))['scores'])"
```

To export the result figures of one `.pkl` file for a paper — separate SVG and
PNG panels, the exact plotted values as CSV, and `figure_sources.json`
recording where every number came from — first look at what it holds
(`--inspect` writes nothing), then export into a **new** folder. The command
only draws; it never trains or evaluates.

```bash
python -m scripts.export_result_figures --artifact runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl --inspect
python -m scripts.export_result_figures --artifact runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl --output runs/figure_exports/cli_matr_xgb
```

The folder then contains `parity.svg`/`.png`, `residuals.svg`/`.png`, their
`_zoom` versions, `plotted_samples.csv`, `figure_sources.json` and
`figure_notes.md`; an SOH result also gets `soh_cycle_<cell>.svg`/`.png`/`.csv`
and `per_cell_errors.csv`. The same export is in the website under
**Export figures** below each result. Details: `docs/VISUALIZATION.md`.

---

## 15. Optional: connect an LLM assistant

Skip this if you do not have an API key. Steps 1–14 do not use it.

With a key, the website can additionally explain configs, errors and
results in plain language. It never computes numbers; all numbers come from
BatteryML.

1. In the sidebar, open **Settings**.
2. **Provider**: choose a preset (OpenAI, Gemini, Claude, Kimi / Moonshot,
   GLM / Zhipu), or **Custom OpenAI-compatible** for any other
   OpenAI-compatible service. A preset only fills in the Base URL.
3. **Base URL**: your provider's API address (filled in by the preset).
4. **API key**: your key.
5. **Model**: a model name your key is allowed to use.
6. Click **Save**, then **Connect**.

**Check:** the status line at the bottom of the sidebar changes from
**LLM: Not connected** to **LLM: Connected**.

The API key is kept only in the open browser session. It is not written to
any file, log, run record or export, and it is gone when the session ends.
Never put an API key in a file inside this repository.

---

## 16. Every time you come back

After a restart, or in any new terminal:

```bash
cd ~/BatteryML-Agent
conda activate batteryml
python -m streamlit run batteryml_web/app.py
```

Then open `http://localhost:8501` (on a remote server, start the SSH tunnel
from §11 first). Your projects, data and runs are still on disk.

---

## 17. Updating to a newer version

```bash
cd ~/BatteryML-Agent
conda activate batteryml
git pull
pip install -r requirements.txt
python -m scripts.verify.run_all
```

Stop the website with `Ctrl+C` before updating and start it again afterwards.

`git pull` prints `Updating …` and `Fast-forward` and a list of changed files
(or `Already up to date.`). `pip install -r requirements.txt` prints
`Requirement already satisfied` for everything that did not change.

Read the health check as in §7: `REG-01` stays `[FAIL]` (expected). Once you
have data, `DATA-01` and `SPLIT-01` no longer say `[SKIP]`. Any other `[FAIL]`
line names its problem; see §18.

---

## 18. Troubleshooting

| Message or symptom | Cause | Fix |
| --- | --- | --- |
| `conda: command not found` | The terminal was not reopened after §2.3. | Close the terminal, open a new one. If still missing, run `$HOME/miniconda3/bin/conda init bash` again. |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | §2.4 was skipped. | Run the two commands in §2.4. |
| The prompt shows `(base)`, not `(batteryml)` | The environment is not active. | `conda activate batteryml` |
| `ModuleNotFoundError: No module named 'numpy'` (or `torch`, `streamlit`, …) | The environment is not active, or §5/§6 did not finish. | `conda activate batteryml`, then repeat §5 and §6. |
| `ModuleNotFoundError: … 'batteryml' is not a package` | `python bin/batteryml.py` was run directly. | Use `python batteryml_web/core/run_batteryml_cli.py …` |
| `ModuleNotFoundError: No module named 'batteryml_web'` | The command was run outside the `BatteryML-Agent` folder. | `cd ~/BatteryML-Agent` and run it again. |
| `torch.cuda.is_available()` prints `False` on a GPU machine | NVIDIA driver older than PyTorch's CUDA version. | §5, last part. |
| `[INFO] data/raw/MATR/… already exists. Skip it.` (without `and is complete`) | The folder was downloaded with an older version, which skipped broken files. | Update (§17), then run §8 again. |
| `error: Could not download …` with `ProxyError`, `ConnectionError` or `403 Forbidden` during §8 | The machine cannot reach the download server. Nothing was written. | Fix the network, then run §8 again. |
| `error: … already exists (… bytes), but whether it is complete cannot be checked` | The server could not be reached to compare sizes. | If the earlier download ended without an error, the file is complete: go on to §9. Otherwise fix the network and run §8 again. |
| `FileNotFoundError: Batch file not found: data/raw/MATR/…` | The download is incomplete. | §8 again, then §9. |
| `OSError: Unable to synchronously open file (file signature not found)` in §9 | A `.mat` file is empty or cut off (left by an older version of the download). | Run §8 again, then §9. |
| MATR shows `INCOMPLETE_PROCESSED` | Preprocessing failed or was interrupted. | §9 again. |
| `AssertionError: data/processed/MATR` when running a model | §9 was not run. | Run §9. |
| `KeyError: 'b1c1'` when running a model | `data/processed/MATR` exists but is empty — §9 failed. | Fix the cause of the §9 error, run §9 again. |
| `unrecognized arguments: --config` | `--config` was typed in front of the config file. | Remove `--config`. |
| The run only prints a path and ends | `--train --eval` are missing. | Add both. |
| Health check: `[FAIL] REG-01 …` and exit code 1 | Upstream gap, expected. | Nothing. See §7. |
| Health check: `[FAIL] DATA-01 … processed directory/directories exist but contain no *.pkl file: MATR` | §9 failed and left an empty `data/processed/MATR`. | Fix the cause of the §9 error, run §9 again. |
| Website: **No datasets are available yet.** | The website has not re-read the data folder. | **Prepare Data → Refresh datasets**. |
| Website: **No run is selected. Choose one in Run History above.** | No run has been started in this project yet. | Run a model (§12.3 or §12.5), then open **Review Results** again. |
| Browser cannot open `localhost:8501` from your own computer | The website runs on a remote server. | Use the SSH tunnel in §11. |
| `CUDA out of memory` | The model does not fit on the GPU. | Choose **CPU** / `--device cpu`. |

**Starting over completely** — deletes the environment only; your code, data
and runs stay:

```bash
conda deactivate
conda env remove -n batteryml -y
```

Then repeat from §4.

---

## 19. Known limitations

- **CPTransformer** needs source code from a separate repository placed in
  `third_party/BatteryLife/`, which is not part of this repository. Without
  it, this one model is unavailable; all other models work. The health check
  (`VAL-01`) mentions it.
- The BatteryLife datasets (`MICH`, `MICH_EXP`, `SDU`, `Stanford_2`, `Tongji`,
  `XJTU`, `ISU_ILCC`) have no BatteryML preprocessor. They can be detected
  (`RAW_ONLY`) but not preprocessed here.
- Runs started on the **Run Model** page or with **Run MATR Demo** cannot be
  stopped from the website; only **Agent Workflow** has **Stop run**.
- Only one device per run: CPU, or one GPU. Multi-GPU training is not
  supported in the website.
- Not every dataset × model × task combination has a config. The website only
  offers combinations that exist.
- Tested on Linux only.

---

## 20. Tested versions and what was verified

Installed by following this guide from §1 to §7 on a Linux x86_64 machine
with no GPU, starting from an empty home directory:

| Component | Version |
| --- | --- |
| conda | 26.7.1 |
| Python | 3.10.21 |
| torch | 2.14.0+cu130 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| numba | 0.67.0 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |
| h5py | 3.16.0 |
| matplotlib | 3.10.9 |
| streamlit | 1.64.0 |

What was actually executed, and what was not:

| Step | Status |
| --- | --- |
| §1 `git`/`curl` check, §2 Miniconda (download, install, `conda init`, reopened terminal, Terms of Service), §3 clone, §4 environment, §5 `pip install torch` and the check (CPU), §6 requirements and import check, §7 health check (`10 passed, 1 failed, 2 skipped`) | Executed exactly as written, twice: the second time by copying every command block of this file, unchanged, into a new terminal with an empty home directory. |
| §8 download command | Executed; the test machine's network policy refused the connection to `data.matr.io` (`403 Forbidden`), so no data was downloaded. With the current download command that failure exits with one `error:` line and leaves the folder empty; an empty file left by the older version is removed and fetched again. A successful download could not be run here. |
| §9 preprocess, §10 readiness | Executed without data: the error messages for missing raw files (`FileNotFoundError`) and for an empty raw file (`OSError … file signature not found`), and the `INCOMPLETE_PROCESSED` state, were observed (the `OSError` with a 0-byte file from the older download command). The count `180` and `MATR READY` come from a separate server that holds the processed MATR data. |
| §11 start the website, §12.1 create a project, §12.2 Refresh datasets, page and button names in §12 | Executed in a browser against this installation. |
| §12.3–12.4 a run and its results in the website, §13 a completed run | Not executed here (no data). §12.3 steps 1–3 (request, **Use template**, **Analyze**, and the no-data message) were executed in the browser; the labels in steps 4–8 were taken from the code. The CLI commands in §13 were checked as far as the data step: they stop with `AssertionError: data/processed/MATR` when that folder does not exist, and with `KeyError: 'b1c1'` when it exists but is empty. A one-epoch MATR CNN run completed on the separate server. That Run Model and **Run MATR Demo** runs are listed in Review Results (§12.5) is covered by unit tests only; no such run could be started here. |
| §5 GPU case, the `--index-url` command, `nvidia-smi` | Not executed (no GPU on the test machine; `download.pytorch.org` was blocked). |
| §15 LLM assistant | Fields, buttons and the six provider presets checked in the browser; no connection was made. |
| §16 coming back, §17 updating, §18 "Starting over" | Executed exactly as written (website started and stopped with `Ctrl+C`; `git pull` fast-forwarded; `conda env remove` left only `base`). |
| macOS, Windows/WSL2 | Not tested. |

---

## Further reading

| Document | Content |
| --- | --- |
| [`README_BATTERYML.md`](./README_BATTERYML.md) | The original BatteryML README: paper, datasets, benchmark tables |
| [`dataprepare.md`](./dataprepare.md) | Manual download instructions for every dataset |
| [`docs/VERIFY.md`](./docs/VERIFY.md) | What each health-check line tests |
| [`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) | Adding your own model |

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
