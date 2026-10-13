<div align="center">

# BatteryML-Agent — 使用指南

[English](./README.md) · **简体中文**

旧版指南（已归档，V1）：[English](./README_v1.md) · [简体中文](./README_v1.zh-CN.md)

</div>

BatteryML-Agent 是一个用于预测电池**剩余使用寿命（RUL）**和**健康状态（SOH）**
的网页应用和命令行工具集，构建在 [BatteryML](https://github.com/microsoft/BatteryML)
之上。预处理、训练、预测和打分全部由 BatteryML 完成；本仓库在此之上提供：
一个分步操作的网站、每次运行前的安全检查，以及一个可选的 AI 助手——它只负责解释，
从不计算任何数字。

本指南假设你**没有** conda、Python 或 Streamlit 的使用经验。每条命令都可以原样复制，
每个按钮都按它在屏幕上显示的名字书写（网站界面为英文，因此按钮名保留英文）。

> **提示。** 如果你只想先看到一个结果，请阅读 §2、§3、§4（只做 MATR）、§6 和 §7。
> 这条路径不需要 API key，也不需要 GPU。

---

## 目录

1. [BatteryML-Agent 是什么](#1-batteryml-agent-是什么)
2. [系统要求与安装](#2-系统要求与安装)
3. [conda / Python 环境](#3-conda--python-环境)
4. [数据集：下载与预处理](#4-数据集下载与预处理)
5. [数据集就绪状态](#5-数据集就绪状态)
6. [启动网站](#6-启动网站)
7. [首页与 Quick experience 分步指南](#7-首页与-quick-experience-分步指南)
8. [Prepare Data 与 Dataset Overview](#8-prepare-data-与-dataset-overview)
9. [Run Model：选择配置](#9-run-model选择配置)
10. [安全上传自己的 YAML 配置](#10-安全上传自己的-yaml-配置)
11. [Agent Workflow 与模型推荐](#11-agent-workflow-与模型推荐)
12. [可选的 LLM 助手：Save、Connect、Explain](#12-可选的-llm-助手saveconnectexplain)
13. [CPU 与 GPU](#13-cpu-与-gpu)
14. [Review Results：运行历史、结果与指标](#14-review-results运行历史结果与指标)
15. [科学可视化与图表导出](#15-科学可视化与图表导出)
16. [基准对比与冒烟测试的局限](#16-基准对比与冒烟测试的局限)
17. [命令行：可直接复制的命令](#17-命令行可直接复制的命令)
18. [故障排除与已知限制](#18-故障排除与已知限制)
19. [可复现性、验证状态与引用](#19-可复现性验证状态与引用)

---

## 1. BatteryML-Agent 是什么

### 1.1 它能为你做什么

你提供电池循环数据（例如公开的 **MATR** 数据集，包含 180 个 LFP/石墨电芯），它会：

1. 把原始文件转换成 BatteryML 的格式（**预处理**）；
2. 训练你选择的模型（例如 **XGBoost**），并在预留的测试集上评估（**运行**）；
3. 展示得分（RMSE、MAE、MAPE）、预测值和科学图表，并允许下载全部内容（**查看结果**）。

支持两种预测任务：

| 任务 | 预测什么 | 误差单位 |
| --- | --- | --- |
| **RUL prediction** | 电芯还剩多少个循环 | 循环数 |
| **SOH estimation** | 剩余容量占初始容量的比例 | 容量比 |

### 1.2 三种使用方式

| 路线 | 适合谁 | 需要 API key 吗？ |
| --- | --- | --- |
| **Quick experience / Protocol Workflow**（§7–§9、§14） | 所有人。完全确定性，可离线使用。这是参考路线。 | 不需要 |
| **Agent Workflow**（§11） | 用一句话描述任务；网站自动识别数据集、任务和模型，并检查能否运行。 | 不需要（LLM 可选） |
| **命令行**（§17） | 脚本、服务器、批量任务。 | 不需要 |

### 1.3 架构一览

```
            ┌──────────────────────── 你的浏览器 ─────────────────────────┐
            │  Home · Agent Workflow · Prepare Data · Run Model · Review  │
            └───────────────────────────────┬─────────────────────────────┘
                                            │  （Streamlit 网站：batteryml_web/app.py）
      可选 LLM ─ 只做解释 ─┐                ▼
                           │           校验器 ── 拒绝任何不能安全运行的请求
                           │                │
                           └──▶ 建议只展示给你；不点按钮就什么都不会运行
                                            ▼
             batteryml_web/core/run_batteryml_cli.py   （调用 BatteryML 的唯一封装）
                                            ▼
                    BatteryML：预处理 · 训练 · 预测 · 评估（RMSE / MAE / MAPE）
                                            ▼
      data/raw/<DATASET> → data/processed/<DATASET> → runs/web_projects/<project>/…
```

代码强制执行的规则：

- **所有数字都来自 BatteryML。** 预测值、指标和基准值从不由 LLM 生成。
- **不按 Run 按钮就什么都不会运行。** 分析、推荐和解释都不会启动训练。
- **训练读取的是预处理后的数据，而不是原始数据。** 必须先预处理（§4）。

### 1.4 目录结构

| 目录 | 内容 |
| --- | --- |
| `batteryml/` | BatteryML 库（上游代码，本平台未修改）。 |
| `batteryml_web/` | 网站、安全检查和命令行封装。 |
| `configs/` | 现成的运行配置（YAML 文件），每个 数据集 × 模型 一个。 |
| `data/raw/<DATASET>/` | 你下载的原始文件。不纳入 git。 |
| `data/processed/<DATASET>/` | 预处理后的文件（每个电芯一个 `.pkl`）。不纳入 git。 |
| `runs/` | 所有运行写出的内容。不纳入 git。 |
| `scripts/` | 健康检查、就绪状态检查、图表导出等工具。 |
| `docs/` | 技术文档。 |

---

## 2. 系统要求与安装

### 2.1 你需要准备什么

| 项目 | 要求 |
| --- | --- |
| 操作系统 | **Linux x86_64**（已测试）。macOS：未测试。Windows：请使用 [WSL2 + Ubuntu](https://learn.microsoft.com/windows/wsl/install)，并在其中按 Linux 步骤操作（未测试）。 |
| 磁盘空间 | 软件本身约 **12 GB**（V1 安装测试实测：Miniconda + 环境 8.0 GB，pip 下载缓存 3.5 GB），另需存放电池数据的空间。 |
| 网络 | 安装和下载数据时需要。用到的主机：`repo.anaconda.com`、`pypi.org`、`files.pythonhosted.org`、`github.com`、`data.matr.io`。 |
| GPU | **不需要。** 一切都可以在 CPU 上运行。可以使用 NVIDIA GPU（§13）。 |
| 管理员权限 | 不需要。所有内容都安装在你的主目录下。 |
| API key | **不需要。** 只有可选的 LLM 助手（§12）会用到。 |

### 2.2 检查机器

```bash
git --version
curl --version
```

两条命令都必须打印出版本号。如果缺少某个工具，请用系统的包管理器安装
（Ubuntu：`sudo apt-get install -y git curl`）。

仅当机器有 NVIDIA GPU 时：

```bash
nvidia-smi
```

会出现一个表格；记下右上角的 `CUDA Version`（§3.3 会用到）。如果提示
"command not found"，说明没有可用的 NVIDIA 驱动；一切都会在 CPU 上运行，这没有问题。

### 2.3 安装 Miniconda

Miniconda 提供 `conda`，用于创建一个隔离的 Python 环境，不会改动机器上的其他东西。

> 已经装有 conda 或 Anaconda？运行 `conda --version`。如果打印出版本号，直接跳到下面的**第 4 步**。

1. 下载安装程序：

   ```bash
   cd ~
   curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

   其他平台把 URL 中的文件名换成：
   `Miniconda3-latest-Linux-aarch64.sh`（Linux ARM）、
   `Miniconda3-latest-MacOSX-arm64.sh`（Apple Silicon）、
   `Miniconda3-latest-MacOSX-x86_64.sh`（Intel Mac）。

2. 无交互地安装到 `~/miniconda3`：

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
   ```

   输出的最后一行是 `installation finished.`

3. 把 conda 接入终端，然后**关闭终端并打开一个新终端**（必须）：

   ```bash
   $HOME/miniconda3/bin/conda init bash
   ```

   macOS（zsh）请用 `$HOME/miniconda3/bin/conda init zsh`。在新终端里，提示符以
   `(base)` 开头，`conda --version` 会打印版本号。

4. 接受 Anaconda 服务条款。不做这一步，创建环境时会报错 `CondaToSNonInteractiveError`：

   ```bash
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   ```

   预期输出：`accepted Terms of Service for https://repo.anaconda.com/pkgs/main`，
   以及针对 `…/pkgs/r` 的同样一行。

### 2.4 下载代码

```bash
cd ~
git clone https://github.com/ZishanShu/BatteryML-Agent.git
cd BatteryML-Agent
ls
```

列表中包含 `batteryml`、`batteryml_web`、`configs`、`requirements.txt` 和 `README.md`。

> **重要：** 本指南中的每条命令都在 **`BatteryML-Agent` 目录内**运行。每次打开新终端，
> 先运行 `cd ~/BatteryML-Agent`。

---

## 3. conda / Python 环境

### 3.1 创建并激活环境

```bash
conda create -n batteryml python=3.10 -y
conda activate batteryml
python --version
```

提示符变为以 `(batteryml)` 开头，Python 打印 `Python 3.10.x`。请使用 Python 3.10：
`requirements.txt` 要求 `numpy<2.0`，而它没有 Python 3.13 的版本。

> 每次打开新终端：`cd ~/BatteryML-Agent` 和 `conda activate batteryml`。
> 大多数"module not found"错误都源于忘了这一步。

### 3.2 安装 PyTorch

```bash
python -m pip install --upgrade pip
pip install torch
```

这是最大的一次下载（V1 安装测试中约占 5.5 GB 磁盘）。同一个包在有无 GPU 的机器上都能用。检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

| 输出 | 含义 |
| --- | --- |
| 在**没有** GPU 的机器上为 `… False` | 正确。一切在 CPU 上运行。 |
| 在**有** GPU 的机器上为 `… True` | 正确。可以使用 GPU。 |
| 在**有** GPU 的机器上为 `… False` | NVIDIA 驱动比 PyTorch 的 CUDA 版本旧。见 §3.3。 |

此时可能出现警告 `Failed to initialize NumPy: No module named 'numpy'`。这是预期的，完成 §3.4 后就会消失。

### 3.3 仅针对打印 `False` 的 GPU

把第二个数字（PyTorch 的 CUDA 版本，例如 `13.0`）与 `nvidia-smi` 显示的
`CUDA Version` 比较。如果驱动的版本更低，就升级驱动，或安装与之匹配的 PyTorch 版本。
从 <https://pytorch.org/get-started/locally/> 获取准确命令（Pip、Linux、你的 CUDA 版本），
形式如下（`cu126` = CUDA 12.6）：

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

### 3.4 安装其余依赖

```bash
pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

预期最后一行：`all imports OK`。

> **不要运行 `pip install .`** 网站和命令行封装都从本目录加载 BatteryML。
> 安装进环境的副本不会随 `git pull` 更新。

可选，释放约 3.5 GB：`pip cache purge`。

### 3.5 运行健康检查

```bash
python -m scripts.verify.run_all
```

它运行 13 项检查，每项打印一行，最后一行是汇总。在**没有数据**的全新安装上，
V1 指南记录的汇总为：

```
10 passed, 1 failed, 2 skipped of 13 check(s)
```

| 检查项 | 无数据时的预期 | 原因 |
| --- | --- | --- |
| `ENV-01` `CLI-01` `CFG-01` `VAL-01` `EXT-01` `DEV-01` `INTENT-01` `RUN-01` `UI-01` `FIG-01` | `[PASS]` | 安装完整。 |
| `DATA-01` `SPLIT-01` | `[SKIP]` | 尚未下载数据。 |
| `REG-01` | `[FAIL]` | 预期结果。上游 BatteryML 列出了六种没有预处理器的 cycler 格式，不影响本指南。 |

因为 `REG-01`，该命令以退出码 `1` 结束，这是预期的。其他任何 `[FAIL]` 都会写明问题，见 §18。一旦 `data/` 目录存在，`DATA-01` 就不再显示 `[SKIP]`。
完整报告写入 `runs/verify/verification_report.json`。每项检查的含义见
[`docs/VERIFY.md`](./docs/VERIFY.md)。

---

## 4. 数据集：下载与预处理

数据始终按以下顺序流动：

```
data/raw/<DATASET>  ──preprocess──▶  data/processed/<DATASET>  ──run──▶  结果
```

两步都可以**在网站里**完成（§7 第 1 步或 §8），也可以**在命令行**完成（见下文）。
它们运行的是同一套 BatteryML 命令。

> 在**命令行**下载或预处理之后，请打开 **Prepare Data → Dataset Overview** 并点击
> **Refresh datasets**，这样 Prepare Data 和 Run Model 才会列出该数据集。
> Quick experience 中的 **Download …** 和 **Process …** 按钮会自动刷新这个列表。

### 4.1 哪些数据集可以自动下载

| 数据集 | 自动下载 | 预处理 |
| --- | --- | --- |
| `MATR`（参考数据集，180 个电芯） | 支持 | 支持 |
| `HUST`、`CALCE`、`RWTH` | 支持 | 支持 |
| `HNEI`、`SNL`、`UL_PUR`、`OX` | **不支持。** 按 [`dataprepare.md`](./dataprepare.md) 手动下载，并把文件放到 `data/raw/<NAME>/`。 | 支持 |

### 4.2 下载 MATR

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

该命令从 `data.matr.io` 下载四个较大的 `.mat` 文件，耗时较长。重复运行总是安全的：
已完整的文件会被跳过（`[INFO] … already exists and is complete (… bytes). Skip it.`）；
缺失或不完整的文件会重新下载。文件先写入临时 `.part` 文件，完整后才重命名。

检查：

```bash
ls -lh data/raw/MATR
find data/raw/MATR -type f -size 0
```

第一条命令列出 `MATR_batch_20170512.mat`、`MATR_batch_20170630.mat`、
`MATR_batch_20180412.mat` 和 `MATR_batch_20190124.mat`。第二条命令**不输出任何内容**。

如果机器无法访问 `data.matr.io`，命令会停止并输出一行
`error: Could not download data/raw/MATR/MATR_batch_….mat`，其中包含 `ProxyError`、
`ConnectionError` 或 `403 Forbidden`，且不会为该文件写入任何内容。请先修复网络
（在同一台机器的浏览器里试试打开 <https://data.matr.io>），然后重新运行该命令。

对于 `HUST`、`CALCE` 或 `RWTH`，把命令中**两处** `MATR` 都替换掉。

### 4.3 预处理 MATR

```bash
python batteryml_web/core/run_batteryml_cli.py preprocess MATR data/raw/MATR data/processed/MATR
ls data/processed/MATR | wc -l
```

MATR 打印的数量是 `180`（每个电芯一个 `.pkl`；在项目服务器上观察到）。

其他数据集请把三处 `MATR` 都替换掉。一次处理多个数据集：

```bash
python scripts/preprocess_sweep.py --dry-run
python scripts/preprocess_sweep.py --only HUST
```

`--dry-run` 只打印每个数据集的命令，不运行任何东西（但仍会写出报告
`logs/preprocess/sweep_report.csv`）。其他选项：`--skip`、`--force`、`--timeout`
（见 `python scripts/preprocess_sweep.py --help`）。

> 始终通过 `batteryml_web/core/run_batteryml_cli.py` 调用。直接运行
> `python bin/batteryml.py` 会报错 `ModuleNotFoundError: … 'batteryml' is not a package`。

### 4.4 常见数据错误

| 报错信息 | 原因 | 解决办法 |
| --- | --- | --- |
| `FileNotFoundError: Batch file not found: data/raw/MATR/…` | 下载不完整。 | 重新执行 §4.2，再执行 §4.3。 |
| `OSError: Unable to synchronously open file (file signature not found)` | 某个 `.mat` 文件为空或被截断。 | 重新执行 §4.2（会替换此类文件），再执行 §4.3。 |
| `AssertionError: data/processed/MATR`（运行时） | 从未运行预处理。 | 执行 §4.3。这**不**代表网站坏了。 |
| `KeyError: 'b1c1'`（运行时） | `data/processed/MATR` 存在但为空（预处理失败）。 | 修复预处理错误，重新执行 §4.3。 |

---

## 5. 数据集就绪状态

网站和脚本把每个数据集归为以下四种状态之一：

| 状态 | 含义 | 该怎么做 |
| --- | --- | --- |
| `READY` | 已有预处理后的 `.pkl` 文件。 | 无需操作，可以直接用于运行。 |
| `RAW_ONLY` | 有原始文件，没有预处理文件。 | 进行预处理（§4.3，或网站中的 **Process … data** / **Run preprocessing**）。 |
| `INCOMPLETE_PROCESSED` | `data/processed/<NAME>` 存在但没有 `.pkl` 文件；预处理失败或被中断。 | 重新预处理。 |
| `MISSING` | 既没有原始文件也没有预处理文件。 | 下载（§4.2），或手动放入原始文件。 |

你没有下载过的数据集会显示 `MISSING`，这是正常的。

在命令行检查就绪状态（只读，不做任何修改）：

```bash
python -m scripts.inspect_dataset_readiness MATR
```

对每个数据集，它会打印所检查的原始和预处理目录以及一行 `status`，最后是汇总。
在**没有**数据的机器上的示例：

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

完成 §4.3 后，状态行显示 `READY`。不带参数时列出所有数据集；`--json` 输出 JSON。

就绪状态**只描述数据**。一个模型还需要针对该数据集和任务的、已注册并通过校验的配置（§9、§11）。

---

## 6. 启动网站

```bash
cd ~/BatteryML-Agent
conda activate batteryml
python -m streamlit run batteryml_web/app.py
```

请在 `BatteryML-Agent` 目录中准确使用 `python -m streamlit run …`。
（直接用 `streamlit run …` 可能报错 `ModuleNotFoundError: No module named 'batteryml_web'`，
而且 `.streamlit/config.toml` 只会从这个目录读取。）

终端打印：

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

打开 `http://localhost:8501`。浏览器标签页标题为 **BatteryML Protocol Platform**。
在终端按 `Ctrl+C` 停止网站。

以日期开头、提到 `use_container_width` 等内容的行是 Streamlit 的弃用警告，不是错误。
真正需要关注的错误会显示在网页上。

**在远程服务器上**，请在**你自己的电脑**上转发端口（把 `user@server` 换成你的登录方式），
保持该终端打开，然后在本地打开 `http://localhost:8501`：

```bash
ssh -L 8501:localhost:8501 user@server
```

> 网站没有登录功能。任何能访问 8501 端口的人都能使用它。优先使用 SSH 隧道，
> 不要把端口开放到网络上。

### 6.1 侧边栏（每个页面都有）

| 元素 | 说明 |
| --- | --- |
| **🔋 BatteryML / Protocol Navigator** | 品牌标识。 |
| **Current project** | 当前加载的项目，或 `(no project loaded)`。未加载项目时，运行写入 `runs/web_projects/default`。 |
| **🏠 Home** | 起始页（§7）。 |
| **Agent Workflow** | "Describe a task in plain language; nothing runs until you press Run."（用自然语言描述任务；按 Run 之前什么都不会运行。§11） |
| **Protocol Workflow** → **1. 🧪 Prepare Data**、**2. 🚀 Run Model**、**3. 📈 Review Results** | 手动协议流程（§8、§9、§14）。 |
| **← Prev** / **Next →** | 在协议中后退或前进一步。 |
| **Settings** | 打开 LLM 设置（§12）。 |
| **LLM: …** 状态行 | `Not connected`、`Configured / Not tested`、`Connected` 或 `Connection failed`。 |

---

## 7. 首页与 Quick experience 分步指南

首页显示标题 **BatteryML workspace** 和问题 **How would you like to start?**，下面有三张卡片：

| 卡片 | 用途 |
| --- | --- |
| **⚡ Quick experience**（"Use the built-in MATR demo"、"No API key required"） | 用一个数据集、一个模型、无需建项目，得到第一个结果。 |
| **✦ Create new project** | 为你自己的工作创建一个有名字的项目目录。 |
| **↗ Existing project** | 重新打开之前创建的项目。 |

点击卡片打开对应面板。面板下方的 **← Choose another start path** 用于关闭它。页面更下方依次是：
**Your BatteryML workflow**（四个步骤的概览）、未配置 LLM 时的可选 LLM 提示、
**Agent Workflow** 输入框（§11），以及 **Start workflow →**（打开 **Prepare Data**）。

### 7.1 Quick experience

点击 **⚡ Quick experience**。面板包含四个编号步骤。默认组合为
**MATR · RUL prediction · XGBoost · cpu**，使用
`configs/baselines/sklearn/xgb/matr_1.yaml`。

#### 第 1 步 · Prepare data

1. **位置：** **Raw dataset** 下拉框。**选择：** `MATR`（默认）。
   没有运行配置的数据集会显示 `· no run configuration`，不能在这里运行
   （请改用 **Open Prepare Data**）。
2. **查看状态：** "Raw data"和"Processed data"分别显示其状态，另有一行
   `Readiness: STATUS — reason`（状态含义见 §5）。下载未完成时显示 `Download incomplete (x/y files)`。
3. **如果缺少原始数据：** 点击 **Download MATR data**。
   - **会发生什么：** 运行 BatteryML 的下载命令并显示进度条。日志保存到
     `runs/web_projects/<project>/logs/download_matr.log`。
   - **预期结果：** 原始数据状态变为已存在，面板显示 `✓ Raw files present — no download needed.`
   - **错误：** 网络失败会在页面上显示错误（见 §4.2）。对于 BatteryML 无法下载的数据集，面板显示
     `BatteryML has no automatic download for X. Place its raw files under data/raw/X/, or see Prepare Data.`
4. **如果缺少预处理数据：** 点击 **Process MATR data**。
   - **会发生什么：** 运行 BatteryML 的预处理；日志在
     `runs/web_projects/<project>/logs/preprocess_matr.log`。
   - **预期结果：** `✓ Processed data ready — no processing needed.`，就绪状态为 `READY`。
   - **错误：** `FileNotFoundError` / `OSError` → 下载不完整；重新下载（§4.4）。

#### 第 2 步 · Configure model

| 控件 | 选择什么 |
| --- | --- |
| **Prediction task** | `RUL prediction` 或 `SOH estimation`（只列出该数据集有配置的任务）。 |
| **Prediction model** | 对该数据集和任务有配置的模型，例如 XGBoost。 |
| **Protocol configuration** | 配置文件，显示为 `名称 · 路径`。 |
| **Compute device** | `cpu` 或 `cuda`。无法使用时 `cuda` 显示 `(not available on this machine)`（§13）。 |
| **Upload YAML configuration (optional)** | 你自己的配置文件（§10）。 |

信息框会确认你的选择：
`This configuration means: Dataset … · Task … · Prediction model … · Device … · Config file …`。
运行前请先核对。

#### 第 3 步 · Run training

1. **查看** "Run location"：本次运行将写入的目录。
2. 如果数据尚未预处理，页面警告
   `Process the dataset in step 1 before starting a model run.`，且无法开始运行。
3. 点击 **Run experiment**。
   - **会发生什么：** BatteryML 以种子 `0` 进行训练和评估。进度条显示各阶段
     （初始化、检查预处理数据、加载配置、训练与评估、收集输出）。
   - **输出目录：** `runs/web_projects/<project>/quickstart/<run_id>/`，包含
     `run.json`、`run.log`、`output.log`、`config_<timestamp>.yaml` 和
     `predictions_seed_0_<timestamp>.pkl`。
   - **错误：** 运行失败会显示 BatteryML 的退出码并指向运行日志。
     `AssertionError: data/processed/…` 或 `KeyError: 'b1c1'` → 回到第 1 步。

#### 第 4 步 · Review results

页面显示 BatteryML 保存的指标。点击 **Open Review Results** 查看完整结果（§14）。
这只是打开页面，不需要 LLM，也不会弹出 LLM 对话框。只有之后在没有连接 LLM 的情况下点击
**Summarize results** 或 **Explain metrics** 时，才会弹出 LLM 设置对话框（§12）。

### 7.2 Create new project

1. 点击 **✦ Create new project**。
2. **Project name：** 输入名称（默认 `my_battery_project`）。
3. **Projects base directory：** 保持 `runs/web_projects`。
4. 点击 **Create project and continue**。侧边栏的 **Current project** 显示新名称，并显示工作目录。
5. 在 **Workflow steps** 下展开需要的步骤：**1 · Prepare data** → **Open Prepare Data**；
   **2 · Configure model** / **3 · Run training** → **Open Run Model**；
   **4 · Review results** → **Open Review Results**。

### 7.3 Existing project

1. 点击 **↗ Existing project**。
2. 在 **Select existing project** 中选择项目，然后点击 **Load project and continue**。
3. 如果列表为空，面板显示 `No existing project found yet…`；请先创建一个项目（§7.2）。

---

## 8. Prepare Data 与 Dataset Overview

在侧边栏打开 **1. 🧪 Prepare Data**。页面显示
`Raw data → Preprocessing → Processed records → Feature inspection → Ready for model`，
并有三个标签页。

### 8.1 标签页 "Dataset Overview"

1. 点击 **Refresh datasets**。
   - **会发生什么：** `tools/sync_data_and_config_manifests.py` 重新读取数据目录和配置列表
     （写入 `data/metadata/`），然后页面重新加载。
   - **预期结果：** 所有在 `data/raw/<NAME>/` 下有文件的数据集都会被列出。没有原始文件的数据集不会列出。
   - **错误：** 如果该工具运行失败，会以红色显示它的错误输出。
2. **状态卡片**按 **Ready**、**Preprocessing required**、**Processed incomplete**
   和 **Missing** 统计数据集（对应 §5 中的 `READY`、`RAW_ONLY`、`INCOMPLETE_PROCESSED`、`MISSING`）。
3. 表格列出每个数据集的路径和状态。**Preparation Pipeline** 显示五个阶段。

如果页面显示 `No datasets are available yet. Click Refresh datasets after generating manifests.`，
点击 **Refresh datasets**。如果该提示仍然存在，说明还没有原始数据：先下载（§4.2，或 Quick experience
第 1 步），再刷新一次。

### 8.2 标签页 "Preprocess & Inspect"

1. **Dataset：** 选择一个数据集。页面显示 Selected dataset、Dataset readiness、Raw path、
   Processed path 和 Processed file count，以及将要运行的准确命令。
2. 点击 **Run preprocessing**（若已有预处理文件则为 **Re-run preprocessing**）。
   - 没有原始文件时按钮被禁用并显示 `Raw data missing`；原始数据来源不是 BatteryML
     能预处理的类型时，按钮被禁用并给出警告。
   - **日志：** `runs/web_projects/<project>/logs/preprocess_<dataset>.log`，
     显示在 **Latest preprocessing log** 下。
   - **预期结果：** 预处理文件数增加（MATR：180），就绪状态变为 `READY`。
3. **Feature / artifact inspection** 用于查看预处理后的记录。
4. **Go to Run Model** 继续下一步；**View Dataset Overview** 返回。

### 8.3 标签页 "Custom Dataset / Schema Mapping"（你自己的 cycler 数据）

1. **Dataset name：** 为你的数据起一个简短名称。
2. **Source format：** `ARBIN`、`BIOLOGIC`、`LANDT`、`MACCOR` 或 `NEWARE`。
3. **Task type：** 你打算运行的任务。
4. **Upload training raw files** 和 **Upload validation/test raw files**；
   可选 **Optional cycler config file**。
5. 依次点击 **Save uploads**、**Preprocess training split**、
   **Preprocess validation/test split** 和 **Validate processed outputs**。

每个预处理步骤都会把日志写在上传文件旁边（`preprocess_train.log`、`preprocess_valid.log`）。
不会为自定义数据集自动生成运行配置；扩展方法见
[`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md)。

---

## 9. Run Model：选择配置

打开 **2. 🚀 Run Model**。这个页面可以完全控制数据集、任务、模型、设备和种子。

如果还没有列出任何数据集，页面只显示
`No datasets are currently available. Prepare or upload a dataset first.`，以及
**Go to Prepare Data** 和 **Go to Home**。请先获取原始数据（§4、§7）并点击
**Refresh datasets**（§8.1）；之后才会出现下面的四个标签页。

### 9.1 标签页 "Configuration"

| 控件 | 选择什么 |
| --- | --- |
| **Dataset** | 数据集（只列出有配置的数据集）。 |
| **Task** | `RUL prediction` 或 `SOH estimation`。 |
| **Model** | 对该数据集和任务有配置的模型。 |
| **Device** | `cpu` 或 `cuda`。`cuda` 有问题时会显示警告（§13）。 |
| **Seed** | 整数，默认 `0`。相同配置 + 种子 + 数据 → 相同结果。 |

右侧的 **Selected config summary** 显示为你选定的配置文件：该数据集、任务和模型的
**第一个已注册配置**。这里不能手动挑选（要用其他文件，请用 §10 或命令行）。
**Workspace dir** 显示本次运行的写入位置。

本仓库中可用的配置（示例）：

| 数据集 · 任务 | 模型 |
| --- | --- |
| MATR · RUL | 23 个模型，包括 `xgb`、`rf`、`ridge`、`lasso`、`elastic_net`、`svm`、`gpr`、`mlp`、`cnn`、`lstm`、`transformer`、`batlinet`、`cptransformer`、`dummy` |
| HUST · RUL、SNL · RUL | 各 21 个模型（没有 `batlinet` 和 `cptransformer`） |
| MATR · SOH | `elastic_net`、`lasso`、`rf`、`ridge`、`xgb`（`configs/soh/<model>/matr.yaml`） |

连接 LLM 后（§12），**Explain this config** 和 **Suggest model/config** 会用文字说明这个选择，
但从不修改它。

### 9.2 标签页 "Dataset Readiness"

卡片 **Ready**、**Need preprocess** 和 **Missing raw**。**Select <dataset>** 选择一个就绪的数据集；
**Go Prepare Data** 打开 §8。

### 9.3 标签页 "Command & Run"

1. 查看显示的准确 BatteryML 命令。（**Explain this command** 需要 LLM。）
2. 缺少东西时会有警告并阻止运行：
   `Processed dataset missing. Go to Prepare Data and run preprocessing first.`
   或 `Config missing…`。
3. 点击 **Run BatteryML**。
   - **会发生什么：** BatteryML 用你选的种子和设备进行训练和评估。网站会为 RUL 添加
     `--metric RMSE,MAE,MAPE`，为 SOH 添加 `--metric MAE,RMSE`。
   - **输出：** `runs/web_projects/<project>/run_model/<run_id>/`，包含
     `run.json`、`output.log`、`config_<timestamp>.yaml` 和
     `predictions_seed_<seed>_<timestamp>.pkl`。最近一次的输出也写在
     `runs/web_projects/<project>/logs/run_model.log`。
   - **预期结果：** 成功提示、**Review Results →** 按钮，以及 **Output workspace** 区域。
4. Run Model 的运行无法在网站中停止（只有 Agent Workflow 有 **Stop run**）。
   必要时在终端按 `Ctrl+C` 停止网站。

### 9.4 标签页 "Logs / Debug"

显示 Last command、Return code、Log path 和日志末尾。Python 进程崩溃（segmentation fault）
会用通俗语言说明。连接 LLM 后，**Explain last error** 和 **Suggest next action**
可协助诊断；是否修改由你决定。

---

## 10. 安全上传自己的 YAML 配置

位置：**Home → ⚡ Quick experience → 第 2 步 · Configure model →
Upload YAML configuration (optional)**。

1. 点击上传框，选择一个 `.yaml` 或 `.yml` 文件。推荐从 `configs/` 中复制一个文件开始修改，
   例如 `configs/baselines/sklearn/xgb/matr_1.yaml`。
2. 点击 **Apply uploaded configuration**。
3. **会发生什么：** 文件会被检查。应用只是**选中**这个配置，不会启动运行。
4. **预期结果：** Protocol configuration 和信息框切换到你的文件，并出现说明
   `Uploaded configuration: …`。通过校验的副本保存为
   `runs/web_projects/<project>/uploads/configs/<random id>.yaml`
   （路径不由你的文件名决定，也不会覆盖任何文件）。

在以下情况下，上传会被拒绝，原因列在
`The uploaded configuration was not applied; the previous selection is unchanged.` 下方：

| 规则 | 示例信息 |
| --- | --- |
| 文件名必须是普通的 `.yaml`/`.yml` 名称，不含目录 | `The file name … contains a path; upload a plain .yaml file.` |
| 最大 256 KB、UTF-8 文本、不能为空 | `The file is empty.` |
| 只接受普通 YAML（不能有 `!!python/…` 标签） | `The file uses a YAML tag that is not allowed in a config …` |
| 必须是由各部分组成的映射 | `… a run config must be a mapping of sections (model, train_test_split, feature, label).` |
| 每个类名都必须存在于 BatteryML 中；任务必须与页面上的数据集匹配 | 校验器给出的、指明字段的信息 |
| 除 `train_test_split.cell_data_path` 外不能有文件/目录选项，且该路径必须位于 `data/processed/` 内 | `` `train_test_split.cell_data_path` = … resolves outside data/processed/ … `` |
| 不能有设备选项（设备在页面上选择） | `` `…` chooses a compute device; the device is selected on the page (Compute device) … `` |

修改文件后重新上传。在命令行上可以直接运行任何配置文件（§17）；上述校验只在网站中进行。

---

## 11. Agent Workflow 与模型推荐

Agent Workflow 把一句话变成一个经过检查的运行计划。**不需要 API key**：没有 LLM 时，
请求会对照数据集、任务和模型目录进行确定性解析。配置了 LLM 时，LLM 可以帮助理解请求，
但它的结果会对照同一目录进行检查。

### 11.1 编写请求

1. 在侧边栏打开 **Agent Workflow**（或在首页的 **Agent Workflow** 输入框中输入，然后点击
   **Analyze with Agent**）。
2. 在 **Workflow request** 中按句式 `Use [Model] on [dataset] to predict [task].` 写一句话，
   例如 `Use XGBoost on MATR to predict RUL.`
   输入框为空时，**Use template** 会填入一句根据本机目录生成的句子（它从不覆盖你输入的文字）。
3. 点击 **Analyze**。分析永远不会运行任何东西。

### 11.2 阅读分析结果

**Workflow analysis** 显示三个值：**Dataset**、**Task** 和 **Dataset readiness**，
以及类似 `N model(s) runnable, M blocked for this dataset and task.` 的说明。

- **如果请求可以运行**，**Backend recommended next steps** 是一个可展开的折叠区域。
- **如果数据尚未就绪**，会有警告指出问题，例如
  `RUL prediction configurations are registered for MATR, but its data is MISSING; models run on processed data only. Prepare the dataset in Prepare Data first, then analyze again.`
  以及 `No model can be executed for this dataset/task.` 请先准备数据（§7 第 1 步或 §8），
  然后再次点击 **Analyze**。
- **如果该组合没有配置**（例如 `Use ElasticNet on HUST to predict soh.`：HUST 没有 SOH 配置），
  页面会说明原因并列出替代方案，分组为 **<dataset> · <task>**、**<task> on other datasets**
  和 **Other tasks on <dataset>**。其模型当前就能运行的替代方案会显示为按钮，例如
  `Use Elastic Net Regression on MATR for SOH estimation`（这要求 MATR 处于 `READY`）。
  点击后会**替换请求**并重新分析；在你按下 Run 之前什么都不会运行。**Readiness details**
  列出各数据集状态。
- **Explain this workflow** 会在按钮正下方写出通俗解释。它需要已连接的 LLM；
  否则页面会说明，并提供 **Open LLM settings**。

### 11.3 选择并运行

| 控件 | 选择什么 |
| --- | --- |
| **Model (runnable models only; blocked models cannot be selected)** | 一个可运行的模型。 |
| **1-epoch smoke test — workflow validation only; not benchmark performance.** | 勾选后只训练一个 epoch，用于快速检查流程（§16）。 |
| **Compute device** | **CPU** 或 **GPU**；选 GPU 时再选择 **GPU** 编号。 |
| **Config — evaluation protocol (validated only)** | 一个通过校验的配置。 |

点击 **Run**。

- **会发生什么：** BatteryML 以种子 `0`（此路线固定）进行训练和评估，页面显示实时日志。
  **Stop run** 可停止运行。
- **输出：** `runs/web_projects/<project>/agent_<dataset>_<model>/<run_id>/`
  （冒烟测试会追加 `_smoke`），包含 `run.json`、`run.log`、`stdout.log`、`stderr.log`、
  `config_<timestamp>.yaml` 和 `predictions_seed_0_<timestamp>.pkl`。
- **预期结果：** 显示保存的指标；**Run again** 可重复运行；连接 LLM 后，
  **Explain computed result** 用文字描述结果，直接显示在结果下方一个标题为
  **🤖 Agent explanation** 的框里（不是折叠的）。
- **错误：** 数据集不是 `READY` 时无法运行；请先预处理（§8）。

---

## 12. 可选的 LLM 助手：Save、Connect、Explain

如果没有 API key，请跳过本节。**本指南的其他部分都不需要它。**
助手只做解释和建议；它从不计算预测值、指标或基准值，从不修改数据，
也从不在你点击之前应用任何更改。

### 12.1 配置

1. 在侧边栏打开 **Settings**（或点击页面上的 **Configure LLM** / **Open LLM settings**，
   它们会打开 **LLM settings** 对话框）。
2. **Provider：** `OpenAI`、`Gemini (OpenAI-compatible)`、`Claude (OpenAI-compatible)`、
   `Kimi / Moonshot`、`GLM / Zhipu` 或 `Custom OpenAI-compatible`。预设只会填入 Base URL。
3. **Base URL：** 服务商的 OpenAI 兼容 API 地址。
4. **API key：** 你的密钥（输入时隐藏）。
5. **Model：** 你的密钥可以使用的模型名称。
6. 点击 **Save** 只保存设置、不发起网络请求（提示
   `LLM settings saved. Connection not tested yet.`；侧边栏显示 `LLM: Configured / Not tested`），
   或点击 **Connect** 保存**并测试**连接。

**点击 Connect 后的预期结果：** 侧边栏状态行从 `LLM: Not connected` 变为 `LLM: Connected`。
测试失败时显示 `LLM: Connection failed`；请检查 Base URL、密钥和模型名称，然后再次点击 **Connect**。

密钥**只保存在当前打开的浏览器会话中**。它不会写入任何文件、日志、运行记录或导出内容，
会话结束即消失。切勿把 API key 写进本仓库中的任何文件。

### 12.2 哪里可以使用解释功能

| 页面 | 按钮（需要 `Connected`） |
| --- | --- |
| Run Model | **Explain this config**、**Suggest model/config**、**Explain this command**、**Explain last error**、**Suggest next action** |
| Agent Workflow | **Explain this workflow**、**Explain computed result** |
| Review Results | **Summarize results**、**Explain metrics**（在 **Optional LLM interpretation** 下） |

在 Review Results 中，回答会直接显示在按钮下方一个标题为
**🤖 LLM Assistant Output · Result interpretation** 的框里（不是折叠的）。
未连接时，这些按钮会打开 LLM 设置对话框，而不会调用任何服务。

---

## 13. CPU 与 GPU

- **CPU 总是可用**，也是默认值（Quick experience 和 Run Model 中的 `cpu`，
  Agent Workflow 中的 **CPU**，命令行中的 `--device cpu`）。
- **GPU：** 选择 `cuda`（Quick experience、Run Model），**GPU** + **GPU** 编号（Agent Workflow），
  或命令行中的 `--device cuda:0`。
- 网站会在运行**之前**检查 GPU。如果 CUDA 不可用，或安装的 PyTorch 不是为你 GPU 的架构编译的
  （例如在 RTX 3090 上使用旧版 PyTorch），GPU 选项会被拒绝并说明原因。
  **不会自动回退到 CPU**：请自己选择 CPU。
- 每次运行只用一个设备。网站不提供多 GPU 训练。
- 经典模型（XGBoost、随机森林、ridge 等）从 GPU 获益很少；神经网络（CNN、LSTM、Transformer）获益更多。
- `CUDA out of memory` → 使用 CPU 或更小的模型。

---

## 14. Review Results：运行历史、结果与指标

打开 **3. 📈 Review Results**。它列出当前项目（或 `runs/web_projects/default`）中来自
Quick experience、Run Model 和 Agent Workflow 的运行。

### 14.1 运行历史（Run history）

一个紧凑表格：**Run**（运行 ID 末尾部分）、**Started**（`YYYY-MM-DD HH:MM UTC`）、
**Experiment**（模型 · 数据集 · 任务）、**Seed**、**Device**、**Status**、**Duration**、**Metrics**。

状态：`COMPLETED`、`FAILED`、`RUNNING`、`PLANNED`、`CANCELLED`、`INTERRUPTED`。

### 14.2 打开一次运行

1. **Open a run：** 从列表中选择一次运行。
2. **Run details · …**（可展开）显示状态标签、完整运行 ID 和一张表：Run ID、Started、Project、
   Run directory、Dataset、Task、Model、Config、Seed、Epochs、Device、Status、Duration、
   Return code、Git commit。
3. **Metrics** 显示的正是 BatteryML 保存的内容（RUL：RMSE、MAE、MAPE，单位为循环数；
   SOH：MAE、RMSE）。失败的运行不记录指标。
4. **Scientific plots**（§15）。
5. 先点 **Prepare run bundle**，再点 **Download run bundle** 下载整次运行。
   **Artifacts** 列出文件；**Download a file** / **Download selected file** 下载单个文件。
6. **Resume training** 只对保存了检查点的神经网络模型显示。

### 14.3 Selected result

标签页 **Summary**、**Metrics**、**Predictions**、**Logs** 和 **Files**，然后是
**Benchmark reference**（§16）和 **Optional LLM interpretation**（§12）。

如果页面显示
`No runs recorded yet for this project (runs/web_projects/default). A run appears here as soon as it is started.`
和 `No run is selected. Choose one in Run History above.`，说明该项目还没有运行：
先运行一次（§7、§9 或 §11）再回来。请确认侧边栏显示的是你运行时所在的项目。

### 14.4 如何解读数字

- **RMSE** 和 **MAE** 的单位与任务一致（RUL 为循环数）。越低越好。
- **MAPE** 是相对误差。越低越好。
- 冒烟测试（一个 epoch）的结果**不是**性能结果（§16）。

在命令行查看一个项目的运行及其状态：

```bash
python -m scripts.inspect_run_state runs/web_projects/default
```

---

## 15. 科学可视化与图表导出

在一次运行的 **Scientific plots** 下：

- **Prediction artifact：** 绘图所用的 `.pkl`。
- **Axis range：** **Full range** 或 **Zoomed view**。
- **Cell**（SOH）：选择一个电芯，绘制其 SOH 随循环的变化。
- 一致性图（预测值 vs. 真实值）和残差图、**Per-cell error summary (display diagnostic)**
  以及 **Estimate range check**。

**Export figures**（可展开）：

1. 查看显示的等效命令。
2. 点击 **Prepare figure package**，然后点击 **Download figure package (.zip)**。

在命令行中进行同样的导出（把路径换成你的文件）：

```bash
python -m scripts.export_result_figures --artifact runs/web_projects/default/quickstart/<run_id>/predictions_seed_0_<timestamp>.pkl --inspect
python -m scripts.export_result_figures --artifact runs/web_projects/default/quickstart/<run_id>/predictions_seed_0_<timestamp>.pkl --output runs/figure_exports/my_run
```

`--inspect` 只描述文件内容。导出目录（必须是新目录）包含 `parity.svg`/`.png`、
`residuals.svg`/`.png`、它们的 `_zoom` 版本、`plotted_samples.csv`、`figure_sources.json`
和 `figure_notes.md`；SOH 结果还会有 `soh_cycle_<cell>.svg`/`.png`/`.csv` 和 `per_cell_errors.csv`。
选项：`--cell-id`、`--list-cells`、`--dpi`。该命令只绘图，从不训练或评估。
详见 [`docs/VISUALIZATION.md`](./docs/VISUALIZATION.md)。

---

## 16. 基准对比与冒烟测试的局限

**Benchmark reference**（在 Review Results 中）**只有在对比公平时**才会把一次运行与已发表的
BatteryML RUL 基准（RMSE，单位为循环数，来自 [`README_BATTERYML.md`](./README_BATTERYML.md)）
并列。否则会说明原因，原因为以下之一：

- 已发表的基准只覆盖 RUL 预测（所有 SOH 运行都属此类）；
- 该运行的训练/测试划分无法对应到某个已发表子集（`MATR1`、`MATR2` 等）；
- 该模型没有对应的已发表基线；
- 指标或单位不是以循环数计的 RMSE；
- 这是一次 **1 个 epoch 的冒烟运行**；
- 已发表的值是一个阈值（`>1000`），而不是一个数字；
- 该运行没有保存 RMSE。

**冒烟测试**（Agent Workflow 中的复选框，或命令行中的 `--epochs 1`）只训练一个 epoch。
它证明数据加载、训练和评估在你的机器上能正常工作。**其得分不是结果**：不要报告或用于比较。

---

## 17. 命令行：可直接复制的命令

在 `BatteryML-Agent` 目录中、`conda activate batteryml` 之后运行。

```bash
# 健康检查（13 项）
python -m scripts.verify.run_all

# 下载（MATR、HUST、CALCE、RWTH）
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR

# 预处理
python batteryml_web/core/run_batteryml_cli.py preprocess MATR data/raw/MATR data/processed/MATR

# 就绪状态（只读）
python -m scripts.inspect_dataset_readiness MATR

# 完整运行：MATR 上的 XGBoost，CPU，种子 0
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/sklearn/xgb/matr_1.yaml --workspace runs/cli_matr_xgb --train --eval --device cpu --seed 0

# 冒烟测试：CNN 训练一个 epoch（不是结果）
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/nn_models/cnn/matr_1.yaml --workspace runs/smoke_matr_cnn --train --eval --device cpu --seed 0 --epochs 1

# SOH 运行：MATR 上的 XGBoost
python batteryml_web/core/run_batteryml_cli.py run configs/soh/xgb/matr.yaml --workspace runs/cli_matr_soh_xgb --train --eval --device cpu --seed 0

# 启动网站
python -m streamlit run batteryml_web/app.py
```

`run` 的规则：

- 配置文件是**位置参数**。`run --config …` 会报错 `unrecognized arguments: --config`。
- `--train` 和 `--eval` 都必须提供。缺少时命令只打印工作目录路径就退出。
- 始终提供 `--seed`。
- `--device cuda:0` 使用第一块 GPU。
- 其他选项：`--metric`、`--epochs`、`--ckpt-to-resume`、`--skip_if_executed`
  （见 `python batteryml_web/core/run_batteryml_cli.py run --help`）。

运行结束时会打印得分，例如 `{'RMSE': …, 'MAE': …, 'MAPE': …}`。工作目录中包含
`config_<timestamp>.yaml` 和 `predictions_seed_<seed>_<timestamp>.pkl`。打印保存的得分：

```bash
python -c "import pickle; print(pickle.load(open('runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl', 'rb'))['scores'])"
```

预处理的输入类型：数据集 `CALCE HNEI HUST MATR OX RWTH SNL UL_PUR`；
cycler 格式 `ARBIN BATTERYARCHIVE BIOLOGIC INDIGO LANDT MACCOR NEWARE NOVONIX`
（部分 cycler 格式没有预处理器；见 §3.5 的 `REG-01`）。

---

## 18. 故障排除与已知限制

### 18.1 故障排除

| 报错信息或现象 | 原因 | 解决办法 |
| --- | --- | --- |
| `conda: command not found` | `conda init` 后没有重新打开终端。 | 打开新终端；必要时再运行一次 `$HOME/miniconda3/bin/conda init bash`。 |
| `CondaToSNonInteractiveError` | 未接受服务条款。 | §2.3 第 4 步。 |
| 提示符显示 `(base)` 而不是 `(batteryml)` | 环境未激活。 | `conda activate batteryml` |
| `ModuleNotFoundError: No module named 'numpy'`（或 `torch`、`streamlit`） | 环境未激活，或安装不完整。 | `conda activate batteryml`，然后执行 §3.2 和 §3.4。 |
| `ModuleNotFoundError: No module named 'batteryml_web'` | 在目录外启动，或使用了普通的 `streamlit run`。 | `cd ~/BatteryML-Agent`；使用 `python -m streamlit run batteryml_web/app.py`。 |
| `ModuleNotFoundError: … 'batteryml' is not a package` | 使用了 `python bin/batteryml.py`。 | 改用 `python batteryml_web/core/run_batteryml_cli.py …`。 |
| `error: Could not download …`（`ProxyError`、`ConnectionError`、`403 Forbidden`） | 无法访问下载服务器。 | 修复网络后重新下载。 |
| `FileNotFoundError: Batch file not found` / `OSError: … file signature not found` | 原始文件不完整。 | 重新下载，再重新预处理（§4.4）。 |
| 数据集显示 `INCOMPLETE_PROCESSED` | 预处理失败。 | 重新预处理。 |
| `AssertionError: data/processed/MATR` | 未预处理。 | §4.3。 |
| `KeyError: 'b1c1'` | 预处理目录为空。 | 修复预处理问题后重新运行。 |
| `unrecognized arguments: --config` | 文件前加了 `--config`。 | 去掉 `--config`。 |
| 运行只打印一个路径就结束 | 缺少 `--train --eval`。 | 两个都加上。 |
| 健康检查 `[FAIL] REG-01` | 预期的上游缺口。 | 无需处理。 |
| `No datasets are available yet.` | 网站尚未重新读取数据目录。 | **Prepare Data → Refresh datasets**。 |
| Review Results 中显示 `No run is selected…` | 该项目还没有运行。 | 先运行一次（§7 或 §9）。 |
| Quick experience：`Process the dataset in step 1 before starting a model run.` | 数据未预处理。 | 第 1 步 → **Process … data**。 |
| Run Model：`Processed dataset missing…` | 数据未预处理。 | §8.2。 |
| Run Model：`No datasets are currently available. Prepare or upload a dataset first.` | 没有原始数据，或列表未刷新。 | 获取原始数据（§4），然后 **Prepare Data → Refresh datasets**。 |
| GPU 选项被拒绝 | 没有 CUDA，或 PyTorch 不是为该 GPU 编译的。 | 选择 CPU，或修复 PyTorch（§3.3、§13）。 |
| `CUDA out of memory` | 模型放不进 GPU。 | 选择 CPU。 |
| LLM 按钮打开的是设置对话框 | LLM 未连接。 | 见 §12，或忽略；其他功能都不需要它。 |
| `LLM: Connection failed` | Base URL、密钥或模型错误，或没有网络。 | 改正后点击 **Connect**。 |
| 浏览器打不开 `localhost:8501` | 网站运行在远程服务器上。 | 使用 SSH 隧道（§6）。 |

### 18.2 已知限制

- **CPTransformer** 需要把另一个仓库的代码放到 `third_party/BatteryLife/`，本仓库不包含；
  **PBT** 需要 `third_party/PBT` 中的资源，本仓库也不包含。其他模型都不依赖它们。
  健康检查（`VAL-01`）会提到这一点。
- BatteryLife 数据集（`MICH`、`MICH_EXP`、`SDU`、`Stanford_2`、`Tongji`、`XJTU`、`ISU_ILCC`）
  没有 BatteryML 预处理器：可以被识别（`RAW_ONLY`），但不能在这里预处理。
- `HNEI`、`SNL`、`UL_PUR` 和 `OX` 必须手动下载。
- 并非每个 数据集 × 任务 × 模型 都有配置；网站只提供存在的组合。SOH 配置只有 MATR。
- Quick experience 和 Run Model 的运行不能在网站中停止；只有 Agent Workflow 有 **Stop run**。
- 每次运行只用一个设备；网站不提供多 GPU 训练。
- 在 Agent Workflow 和 Quick experience 中种子固定为 `0`；需要其他种子请用 Run Model 或命令行。
- 没有登录功能：请保护好 8501 端口（§6）。
- 只在 Linux 上测试过。

### 更新到新版本

先停止网站（`Ctrl+C`），然后：

```bash
cd ~/BatteryML-Agent
conda activate batteryml
git pull
pip install -r requirements.txt
python -m scripts.verify.run_all
```

### 从头开始

只删除环境；代码、数据和运行结果都会保留：

```bash
conda deactivate
conda env remove -n batteryml -y
```

然后从 §3.1 重新开始。

---

## 19. 可复现性、验证状态与引用

### 19.1 运行为什么可复现

- 每次运行都会在 `run.json` 中记录命令、配置、**种子**、**设备**、git commit、状态、耗时和保存的指标，
  旁边是 BatteryML 实际使用的配置（`config_<timestamp>.yaml`）。
- 对确定性模型，相同的配置 + 种子 + 预处理数据 → 相同的指标。GPU 上的神经网络在不同运行之间可能略有差异。
- 要在命令行重复一次网站上的运行，复制 Run Model **Command & Run** 标签页中显示的命令
  （或 `run.json` 中的命令）。
- 报告结果时请同时给出数据集、任务、模型、配置文件、种子、设备和 git commit（`git rev-parse HEAD`）。

### 19.2 本指南的验证情况

| 项目 | 状态 |
| --- | --- |
| 安装步骤 §2–§3（Miniconda、环境、PyTorch CPU、依赖、健康检查），Linux x86_64 无 GPU | [VERIFIED] 于 V1 安装测试；测试版本：conda 26.7.1、Python 3.10.21、torch 2.14.0+cu130、numpy 1.26.4、scipy 1.15.3、pandas 2.3.3、numba 0.67.0、scikit-learn 1.7.2、xgboost 3.2.0、h5py 3.16.0、matplotlib 3.10.9、streamlit 1.64.0 |
| 在当前 `main` 的全新克隆中运行 `python -m scripts.verify.run_all` | [VERIFIED] `10 passed, 1 failed, 2 skipped of 13 check(s)`，退出码 `1`（文档编写环境：Linux、Python 3.11、无数据、无 GPU） |
| §17 中每条命令的 `--help`；无数据时的 `run`（`AssertionError: data/processed/MATR`）、带 `--config` 的 `run`（`unrecognized arguments: --config`）、不带 `--train --eval` 的 `run`（只打印工作目录路径）；无数据时的 `inspect_dataset_readiness` 和 `preprocess_sweep.py --dry-run` | [VERIFIED] 于文档编写环境 |
| `download MATR` | [VERIFIED] 仅失败路径：`data.matr.io` 被拦截（`ProxyError … 403 Forbidden`），输出一行 `error:`，未写入任何文件。此处无法完成一次成功下载 |
| 在全新克隆中用 `python -m streamlit run batteryml_web/app.py` 启动网站；首页、三个起始面板、Quick experience 第 1–4 步、Prepare Data、Run Model 的空状态、Agent Workflow 分析、Review Results 的空状态以及 LLM 设置字段的标签 | [VERIFIED] 在浏览器中，无数据 |
| Run Model 的四个标签页、有运行时的 Review Results、图表、运行打包、YAML 上传、LLM 解释 | [STATIC] 从当前源代码（`batteryml_web/`）读取；需要数据或 API key 才能到达 |
| MATR 预处理得到 180 个文件 | 在项目服务器上观察到，未在文档编写环境中验证 |
| 完整的训练运行、指标、图表 | 未在文档编写环境中执行（没有数据） |
| GPU、真实 LLM 服务、macOS、Windows/WSL2 | 未验证 |

### 19.3 延伸阅读

| 文档 | 内容 |
| --- | --- |
| [`README_BATTERYML.md`](./README_BATTERYML.md) | BatteryML 原始 README：论文、数据集、基准表 |
| [`dataprepare.md`](./dataprepare.md) | 各数据集的手动下载方法 |
| [`docs/VERIFY.md`](./docs/VERIFY.md) | 每项健康检查测试什么 |
| [`docs/READINESS.md`](./docs/READINESS.md) | 就绪状态规则 |
| [`docs/AGENT_WORKFLOW.md`](./docs/AGENT_WORKFLOW.md) | Agent Workflow 内部机制 |
| [`docs/SOH_PROTOCOL.md`](./docs/SOH_PROTOCOL.md) | SOH 任务协议 |
| [`docs/VISUALIZATION.md`](./docs/VISUALIZATION.md) | 图表与导出 |
| [`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) | 添加你自己的模型 |
| [`README_v1.md`](./README_v1.md) / [`README_v1.zh-CN.md`](./README_v1.zh-CN.md) | 已归档的 V1 安装指南 |

### 19.4 引用

BatteryML-Agent 构建于 BatteryML 之上。如果你使用了它，请引用：

```bibtex
@inproceedings{zhang2024batteryml,
  title={Battery{ML}: An Open-source Platform for Machine Learning on Battery Degradation},
  author={Han Zhang and Xiaofan Gui and Shun Zheng and Ziheng Lu and Yuqi Li and Jiang Bian},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
