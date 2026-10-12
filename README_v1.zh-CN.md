<div align="center">

# BatteryML-Agent — 安装指南

[English](./README_v1.md) · **简体中文**

> **已归档的 V1 指南。** 当前指南见 [README.zh-CN.md](./README.zh-CN.md)。本文件仅作参考保留，内容未改动。

</div>

BatteryML-Agent 是一个用于电池寿命（RUL）与健康状态（SOH）预测的网页应用，
构建在 [BatteryML](https://github.com/microsoft/BatteryML) 之上。
本指南带你从一台空机器开始，一条命令一条命令地把它装好，直到网站跑起来。

全部完成后，你会得到：

- 一个名为 `batteryml` 的 conda 环境，里面装好了 BatteryML 及其全部依赖；
- 已下载并预处理好的 MATR 电池数据集；
- 运行在 `http://localhost:8501` 的网站；
- 一个在 MATR 上训练并评估完成的模型，结果既保存在磁盘上，也显示在网页上。

请**严格按顺序**操作。每一步最后都有一个「检查」：检查没通过，就不要往下走。
所有命令都在终端里执行（Linux / macOS 的终端；Windows 请用 WSL2，见 §0）。

---

## 步骤

0. [你需要什么](#0-你需要什么)
1. [检查机器](#1-检查机器)
2. [安装 Miniconda](#2-安装-miniconda)
3. [下载代码](#3-下载代码)
4. [创建 Python 环境](#4-创建-python-环境)
5. [安装 PyTorch](#5-安装-pytorch)
6. [安装其余依赖](#6-安装其余依赖)
7. [运行健康检查](#7-运行健康检查)
8. [下载 MATR 数据集](#8-下载-matr-数据集)
9. [预处理数据集](#9-预处理数据集)
10. [检查数据集就绪状态](#10-检查数据集就绪状态)
11. [启动网站](#11-启动网站)
12. [在网站里完成第一次运行](#12-在网站里完成第一次运行)
13. [改用命令行运行](#13-改用命令行运行)
14. [结果在哪里](#14-结果在哪里)
15. [可选：连接 LLM 助手](#15-可选连接-llm-助手)
16. [以后每次重新打开](#16-以后每次重新打开)
17. [更新到新版本](#17-更新到新版本)
18. [常见问题排查](#18-常见问题排查)
19. [已知限制](#19-已知限制)
20. [测试版本与验证情况](#20-测试版本与验证情况)

---

## 0. 你需要什么

| 项目 | 要求 |
| --- | --- |
| 操作系统 | **Linux x86_64**（已测试）。macOS：未测试。Windows：先安装 [WSL2 + Ubuntu](https://learn.microsoft.com/windows/wsl/install)，然后在 WSL 里按 Linux 步骤操作——未测试。 |
| 磁盘空间 | 软件需要 **12 GB**（实测：Miniconda + 环境 8.0 GB，pip 下载缓存 3.5 GB），**另外**还要留出原始数据和预处理数据的空间。 |
| 网络 | 安装和下载数据时需要。会访问：`repo.anaconda.com`、`pypi.org`、`files.pythonhosted.org`、`github.com`、`data.matr.io`。 |
| GPU | **不需要。** 全部可在 CPU 上运行。如果有 NVIDIA GPU 会被用上（§5）。 |
| 管理员权限 | 不需要。所有东西都装在你的用户目录下。 |

---

## 1. 检查机器

```bash
git --version
curl --version
```

**检查：** 两条命令都打印出版本号。如果没有 `git`，用系统的包管理器安装
（Ubuntu 上：`sudo apt-get install -y git`）。如果没有 `curl`，同样安装
（`sudo apt-get install -y curl`）。

可选——仅当机器有 NVIDIA GPU 时：

```bash
nvidia-smi
```

**检查：** 显示出一张表格。记下表格右上角的 **`CUDA Version`**，§5 会用到。
如果提示找不到命令，说明这台机器没有可用的 NVIDIA 驱动，所有计算都会在 CPU 上进行——这完全没问题。

---

## 2. 安装 Miniconda

Miniconda 提供 `conda`，用它给 BatteryML 建一个独立的 Python 环境，
这样不会改动你机器上的其他任何东西。

> **已经装过 conda 或 Anaconda？** 运行 `conda --version`。
> 如果能打印出版本号，跳过 2.1–2.3，直接做 **2.4**。

### 2.1 下载安装程序

```bash
cd ~
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

其他平台请替换 URL 中的文件名：

| 平台 | 文件名 |
| --- | --- |
| Linux，ARM（aarch64） | `Miniconda3-latest-Linux-aarch64.sh` |
| macOS，Apple 芯片（M1/M2/M3/M4） | `Miniconda3-latest-MacOSX-arm64.sh` |
| macOS，Intel 芯片 | `Miniconda3-latest-MacOSX-x86_64.sh` |

**检查：** `ls -lh Miniconda3-latest-*.sh` 显示一个约 120–190 MB 的文件。

### 2.2 运行安装程序

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

`-b` 表示不询问直接安装；`-p` 把安装目录设为 `~/miniconda3`。

**检查：** 输出的最后一行是 `installation finished.`

### 2.3 让终端认识 conda

```bash
$HOME/miniconda3/bin/conda init bash
```

（macOS 的默认 shell 是 zsh，请改为运行 `$HOME/miniconda3/bin/conda init zsh`。）

然后**关闭当前终端，重新打开一个新的终端。** 这一步必须做：改动只在新终端里生效。

**检查：** 新终端的命令提示符以 `(base)` 开头，并且

```bash
conda --version
```

能打印出版本号（测试安装时打印的是 `conda 26.7.1`）。

### 2.4 接受 Anaconda 服务条款

新版 Miniconda 在你接受默认频道的服务条款之前，拒绝创建任何环境。
如果跳过这一步，§4 会报错
`CondaToSNonInteractiveError: Terms of Service have not been accepted`。

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**检查：** 打印出的两行分别是
`accepted Terms of Service for https://repo.anaconda.com/pkgs/main` 和
`accepted Terms of Service for https://repo.anaconda.com/pkgs/r`。

---

## 3. 下载代码

```bash
cd ~
git clone https://github.com/ZishanShu/BatteryML-Agent.git
cd BatteryML-Agent
```

**检查：**

```bash
ls
```

列表中包含 `batteryml`、`batteryml_web`、`configs`、`requirements.txt` 和 `README.md`。

> **重要：** 从这里开始，本指南的每一条命令都要在 **`BatteryML-Agent` 文件夹里**执行。
> 以后每次打开新终端，都先运行 `cd ~/BatteryML-Agent`。

---

## 4. 创建 Python 环境

```bash
conda create -n batteryml python=3.10 -y
conda activate batteryml
```

**检查：** 命令提示符现在以 `(batteryml)` 开头（不再是 `(base)`），并且

```bash
python --version
```

打印 `Python 3.10.x`。

本指南是用 Python 3.10 测试的。不要用 Python 3.13：
`requirements.txt` 要求 `numpy<2.0`，而它没有 Python 3.13 的安装包。

---

## 5. 安装 PyTorch

先单独安装 PyTorch，这样可以在装其他东西之前先确认 GPU 是否可用。

```bash
python -m pip install --upgrade pip
pip install torch
```

这是整个安装过程中最大的一次下载（测试安装中：PyTorch 2.14.0 加上它自带的 NVIDIA CUDA 库，
磁盘占用约 5.5 GB，在高速网络下用时 222 秒）。
这同一个安装包在**有 GPU 和没有 GPU** 的机器上都能用。

**检查：**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

| 打印结果 | 含义 |
| --- | --- |
| 在**没有** GPU 的机器上打印 `2.14.0+cu130 13.0 False` | 正确。全部在 CPU 上运行。 |
| 在**有** GPU 的机器上最后是 `… True` | 正确。可以使用 GPU。 |
| 在**有** GPU 的机器上最后是 `… False` | 你的 NVIDIA 驱动比 PyTorch 编译时用的 CUDA 版本旧。见下文。 |

版本号那一行**上方**可能会先出现两行：
`UserWarning: Failed to initialize NumPy: No module named 'numpy'` 和
`cpu = _conversion_method_template(device=torch.device("cpu"))`。在这一步这是正常的——
numpy 要到 §6 才会安装。忽略即可，做完 §6 后就不会再出现。

**仅当你有 GPU 却打印了 `False` 时：** 比较第二个数字（PyTorch 编译所用的 CUDA 版本，例如 `13.0`）
和 §1 中 `nvidia-smi` 显示的 `CUDA Version`。如果 `nvidia-smi` 显示的数字更小，
要么更新 NVIDIA 驱动，要么换成对应你 CUDA 版本的 PyTorch。
准确的命令请到 <https://pytorch.org/get-started/locally/> 的选择器里生成
（选 *Pip*、*Linux*、你的 CUDA 版本）。它的形式如下——`cu126` 表示 CUDA 12.6：

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

然后重新运行上面的检查命令。

---

## 6. 安装其余依赖

```bash
pip install -r requirements.txt
```

这会安装 numpy、scipy、pandas、numba、scikit-learn、xgboost、h5py、matplotlib、
streamlit 以及 BatteryML 需要的其他包（测试安装用时 53 秒）。

如果你是以 `root` 身份登录的，pip 会打印一条以
`WARNING: Running pip as the 'root' user` 开头的警告。在 conda 环境里，这条警告不影响使用。

**检查：**

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

打印 `all imports OK`。

> **不要运行 `pip install .`** 上游 BatteryML 的 README 要求这一步，但 BatteryML-Agent 不需要：
> 网站和命令行工具总是从这个文件夹加载 BatteryML。
> 执行 `pip install .` 会把代码的一份快照复制进环境，
> 而这份副本在你之后运行 `git pull` 时**不会**被更新。

可选——删除 pip 的下载缓存，释放 3.5 GB 磁盘空间：

```bash
pip cache purge
```

---

## 7. 运行健康检查

```bash
python -m scripts.verify.run_all
```

它会运行 13 项检查，每项打印一行。在还没有数据的新安装上，最后一行正好是：

```
10 passed, 1 failed, 2 skipped of 13 check(s)
```

**这个结果是正确的。** 由于那一项 failed，这条命令的退出码也是 `1`，这同样是预期中的。
每一行应该显示：

| 检查项 | 预期 | 原因 |
| --- | --- | --- |
| `ENV-01` `CLI-01` `CFG-01` `VAL-01` `EXT-01` `DEV-01` `INTENT-01` `RUN-01` `UI-01` `FIG-01` | `[PASS]` | 安装完整。 |
| `DATA-01` `SPLIT-01` | `[SKIP]` | 还没有下载数据（§8）。 |
| `REG-01` | `[FAIL]` | 上游 BatteryML 列出了六种没有预处理器的循环仪格式（`BATTERYARCHIVE`、`BIOLOGIC`、`INDIGO`、`LANDT`、`MACCOR`、`NOVONIX`）。它不影响本指南的任何步骤。 |

第一次运行时，检查结果之间可能出现两行以日期开头的信息：
`INFO - Failed to extract font properties from …` 和 `INFO - generated new fontManager`。
这是 Matplotlib 在建立字体缓存，只会出现一次，不是错误。

如果除 `REG-01` 以外还有 `[FAIL]`，请阅读它的提示信息——里面写明了问题所在——并参考 §18。
完整报告会写入 `runs/verify/verification_report.json`。

---

## 8. 下载 MATR 数据集

MATR 包含 180 个 LFP/石墨电芯，是本指南使用的参考数据集。

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

这会从 `data.matr.io` 下载四个很大的 `.mat` 文件，耗时较长，请让它一直运行。

如果这台机器连不上 `data.matr.io`，命令会停下并打印一行以
`error: Could not download data/raw/MATR/MATR_batch_….mat` 开头、包含 `ProxyError`、
`ConnectionError` 或 `403 Forbidden` 的信息，并且这个文件**什么都不会写入**。
先解决网络问题（在同一台机器的浏览器里打开 <https://data.matr.io> 试试；
学校或公司的代理可能会拦截它），然后再运行一次同样的命令。

再次运行这条命令总是安全的：已经完整的文件会被跳过
（`[INFO] … already exists and is complete (… bytes). Skip it.`）；缺失、为空或不完整的文件会重新下载。

> 始终通过 `batteryml_web/core/run_batteryml_cli.py` 启动 BatteryML。
> 直接运行 `python bin/batteryml.py` 会报错
> `ModuleNotFoundError: No module named 'batteryml.…'; 'batteryml' is not a package`。

**检查：**

```bash
ls -lh data/raw/MATR
find data/raw/MATR -type f -size 0
```

第一条命令正好列出下面四个文件，而且每个都远大于零：
`MATR_batch_20170512.mat`、`MATR_batch_20170630.mat`、
`MATR_batch_20180412.mat`、`MATR_batch_20190124.mat`。
第二条命令必须**什么都不打印**。

> **如果缺少某个文件，或者第二条命令打印出了文件名：** 再运行一次上面的下载命令，
> 它会替换为空或不完整的文件。只有当它停下并提示
> `… already exists (… bytes), but whether it is complete cannot be checked`
> （连不上服务器，无法比较文件大小）时，才删除整个文件夹后重新下载：
>
> ```bash
> rm -rf data/raw/MATR
> python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
> ```
>
> 用本仓库旧版本下载的文件夹里，仍可能有空的或不完整的文件：旧版本会在下载**之前**
> 就先创建文件，下次运行时又会跳过它。现在的下载命令先写入一个临时的 `.part` 文件，
> 只有下载完整后才改成正式文件名，所以不会再留下损坏的文件。

其他数据集：

- `HUST`、`CALCE`、`RWTH` 可以用同样的方式下载——把命令中两处 `MATR` 都换掉。
- `HNEI`、`SNL`、`UL_PUR`、`OX` 无法用这条命令下载。请按 [`dataprepare.md`](./dataprepare.md)
  中的说明手动下载，并把文件放到 `data/raw/<名称>/`。

---

## 9. 预处理数据集

模型不直接读取原始文件。预处理会把它们转换成 BatteryML 自己的格式，每个电芯一个 `.pkl` 文件。

```bash
python batteryml_web/core/run_batteryml_cli.py preprocess MATR data/raw/MATR data/processed/MATR
```

**检查：**

```bash
ls data/processed/MATR | wc -l
```

打印 `180`。

如果预处理中止并报错
`FileNotFoundError: Batch file not found: data/raw/MATR/MATR_batch_….mat`，
说明 §8 的下载不完整：重新做 §8，然后再运行这一步。

如果报错 `OSError: Unable to synchronously open file (file signature not found)`，
说明某个 `.mat` 文件是空的或不完整——这是旧版下载命令留下的文件（见 §8 的方框）。
再运行一次 §8 的下载命令（它会替换这类文件），然后再运行这一步。

预处理失败后会留下一个**空的** `data/processed/MATR` 文件夹。所以之后 §10 会显示
`INCOMPLETE_PROCESSED`，运行模型会报 `KeyError: 'b1c1'`。只要重新成功运行这一步，两者都会消失。

---

## 10. 检查数据集就绪状态

```bash
python -c "
from batteryml_web.core.readiness import list_dataset_readiness
for r in list_dataset_readiness():
    print(f'{r.dataset:<12} {r.status.value}')
"
```

每个数据集打印一行。**检查：** MATR 那一行是

```
MATR         READY
```

四种可能的状态：

| 状态 | 含义 | 该怎么做 |
| --- | --- | --- |
| `READY` | 预处理后的 `.pkl` 文件已存在。 | 无需操作，可以直接使用。 |
| `RAW_ONLY` | 有原始文件，但还没有预处理文件。 | 对该数据集执行 §9。 |
| `INCOMPLETE_PROCESSED` | `data/processed/<名称>` 存在但里面没有 `.pkl` 文件——预处理失败或被中断。 | 重新执行 §9。 |
| `MISSING` | 既没有原始文件也没有预处理文件。 | 对该数据集执行 §8。 |

你没下载的数据集会显示为 `MISSING`，这是正常的。

---

## 11. 启动网站

```bash
python -m streamlit run batteryml_web/app.py
```

请在 `BatteryML-Agent` 文件夹里启动——`.streamlit/config.toml` 中的设置只会从这里读取。

**检查：** 终端打印

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

在浏览器里打开 `http://localhost:8501`。页面标题是
**LLM-guided BatteryML Protocol Platform**。

网站会一直运行，直到你在这个终端里按 `Ctrl+C`（会打印 `Stopping...`）。关闭这个终端也会让网站停止。

使用网站时，这个终端会打印一些以日期开头的行，例如
``Please replace `use_container_width` with `width`.`` 或
``Please replace `st.components.v1.html` with `st.iframe`.``。这些是 Streamlit 的弃用警告，
不是错误，网站照常工作。真正需要处理的错误会显示在网页上。

### 如果代码运行在远程服务器上

浏览器运行在你自己的电脑上，所以需要转发端口。在**你自己的电脑上**打开一个新终端，运行
（把 `user@server` 换成你登录服务器的方式）：

```bash
ssh -L 8501:localhost:8501 user@server
```

保持这个终端开着，然后在本地浏览器中打开 `http://localhost:8501`。

> 网站没有登录功能。任何能访问 8501 端口的人都能使用它。
> 请优先使用上面的 SSH 隧道，而不是把端口直接开放到网络上。

---

## 12. 在网站里完成第一次运行

网站左侧有一个侧边栏。下面的步骤使用的都是网页上按钮和标签页的原文名称。

### 12.1 创建一个项目

1. 在 **Home** 页的 **Create New Project** 下，保留默认的 **Project name** `my_battery_project`（或输入你自己的名字）。
2. **Projects base directory** 保持为 `runs/web_projects`。
3. 点击 **Create New Project**。

**检查：** 侧边栏的 **Current project** 下显示出你的项目名，
并且文件夹 `runs/web_projects/my_battery_project/` 已经创建。这个项目的所有运行结果都会保存在里面。

### 12.2 让网站识别数据

1. 在侧边栏点击 **1. 🧪 Prepare Data**。
2. 在 **Dataset Overview** 标签页里，点击 **Refresh datasets**。

**检查：** 出现一张 **MATR** 的卡片，最后两行是 `Status: READY` 和 **Ready**。
（点击 **Refresh datasets** 会运行 `tools/sync_data_and_config_manifests.py`，
它会在 `data/metadata/` 中写入网站读取的两个文件。以后每次新增或预处理了数据集，都要再点一次。）

如果卡片最后两行是 `Status: INCOMPLETE_PROCESSED` 和 **Processed incomplete**，
或者 `Status: RAW_ONLY` 和 **Preprocessing required**，请回到 §9，然后再点一次 **Refresh datasets**。

### 12.3 训练并评估一个模型

使用 **Agent Workflow** 页：它会实时显示进度，并有 **Stop run** 按钮。
在这里发起的运行会出现在 **Review Results** 页（Run Model 页和 Home 页 **Run MATR Demo**
发起的运行也会出现，见 §12.5）。
它**不需要** API key——请求是按固定规则解析的。

1. 在侧边栏点击 **Agent Workflow**。
2. 在 **Workflow request** 中输入：

   ```text
   Use XGBoost on MATR to predict RUL.
   ```

   （点击 **Use template** 会按 `Use [Model] on [dataset] to predict [task].` 的格式填入一个例子，
   例如 `Use CNN on HUST to predict battery lifetime.`。如果点了它，请把文字换成上面那一行。）
3. 点击 **Analyze**。在 **Workflow analysis** 下，页面显示 **Dataset** `MATR`、
   **Task** `RUL prediction` 和 **Dataset readiness** `READY`，下面是可以运行的模型。

   如果显示的是 **Dataset readiness** `INCOMPLETE_PROCESSED`（或 `RAW_ONLY`），
   并出现 `xgb cannot run for this dataset and task.`，说明 MATR 还没有预处理：
   回到 §9，然后 **Prepare Data → Refresh datasets**，再点一次 **Analyze**。在此之前无法选择任何模型。
4. 在 **Model (runnable models only; blocked models cannot be selected)** 中选择 **XGBoost**。
5. 在 **Config — evaluation protocol (validated only)** 中选择
   `configs/baselines/sklearn/xgb/matr_1.yaml`。列表里还有 `clo.yaml` 和 `matr_2.yaml`：
   它们是同一份 MATR 数据的不同测试划分，选定一个，比较不同运行时保持一致。
6. **1-epoch smoke test — workflow validation only; not benchmark performance.**
   保持不勾选。（它只对神经网络模型有意义，见 §13。）
7. 在 **Compute device** 下选择 **CPU**，如果有 GPU 也可以选 **GPU**。
8. 点击 **Run**。页面会实时显示进度。运行过程中点击 **Stop run** 可以取消。

**检查：** 运行结束后显示成功信息，并给出 RMSE、MAE 和 MAPE。

### 12.4 查看结果

1. 在侧边栏点击 **3. 📈 Review Results**。
2. 在 **Run history** 下的 **Open a run** 中选择你的运行（默认选中最新的一次）。
3. 依次查看 **Summary**、**Metrics**、**Predictions**、**Logs**、**Files** 这几个标签页。

**Predictions** 标签页显示实测值与预测值对比、残差，以及预测表格。

### 12.5 Run Model 页

侧边栏里的 **2. 🚀 Run Model** 通过一个表单运行同样的 BatteryML 命令：
标签页 **Configuration**（选择 **Dataset**、**Task**、**Model**、**Device**、**Seed**）、
**Dataset Readiness**、**Command & Run**（显示完整命令，然后点击 **Run BatteryML**）
以及 **Logs / Debug**。

在这里发起的每次运行都有自己的文件夹（§14），并会像 Agent Workflow 的运行一样列在
**Review Results** 中。点击 **Run BatteryML** 之前显示的 **Workspace dir** 和命令里，
已经包含了这次运行的文件夹 `runs/web_projects/<project>/run_model/<run_id>`。
与 Agent Workflow 不同，这个页面没有 **Stop run** 按钮：运行会在 BatteryML 结束时结束。

---

## 13. 改用命令行运行

网站运行的每一个任务，本质上都是一条 BatteryML 命令。你也可以直接运行它
（需要已激活 `batteryml` 环境，并位于 `BatteryML-Agent` 文件夹内）。

**完整运行**——在 CPU 上用 MATR 训练并评估 XGBoost：

```bash
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/sklearn/xgb/matr_1.yaml --workspace runs/cli_matr_xgb --train --eval --device cpu --seed 0
```

**冒烟测试**——CNN 模型只训练 1 个 epoch，只用来确认这台机器上训练流程能跑通：

```bash
python batteryml_web/core/run_batteryml_cli.py run configs/baselines/nn_models/cnn/matr_1.yaml --workspace runs/smoke_matr_cnn --train --eval --device cpu --seed 0 --epochs 1
```

把 `--device cpu` 换成 `--device cuda:0` 即可使用第一张 GPU。

这条命令的三条规则：

- 配置文件前面**不要**写 `--config`。写成 `run --config …` 会报错
  `unrecognized arguments: --config`。
- `--train` 和 `--eval` 两个都必须写。缺了它们，命令只会打印 workspace 路径，什么也不做。
- 始终写上 `--seed`，这样运行才能重复得到同样的结果。

运行结束时，最后打印的一行就是分数，例如
`{'RMSE': …, 'MAE': …, 'MAPE': …}`。

> **冒烟测试的分数不是结果。** 一个 epoch 几乎什么都没训练到，它只说明数据加载、训练和评估能跑通。
> 不要拿它和任何东西比较。已发表的 BatteryML 基准（RMSE，单位是循环数；越低越好；
> MATR1 与 MATR2 是不同的测试集）在 [`README_BATTERYML.md`](./README_BATTERYML.md) 中。

---

## 14. 结果在哪里

BatteryML 不会生成指标表格文件。它在运行结束时打印分数，
并把分数和预测结果一起保存在一个 `.pkl` 文件里。

**来自 Agent Workflow 的运行**（§12.3）：

```
runs/web_projects/<project>/agent_<dataset>_<model>/<run_id>/
├── run.json                                  run record: command, status, seed, device, git commit, scores
├── run.log   stdout.log   stderr.log         everything the run printed
├── config_<timestamp>.yaml                   the config as BatteryML used it
└── predictions_seed_<seed>_<timestamp>.pkl   predictions and scores
```

（`run.json` 记录命令、状态、seed、设备、git commit 和分数；三个日志文件包含运行时打印的全部内容；
`config_<timestamp>.yaml` 是 BatteryML 实际使用的配置；`.pkl` 文件保存预测结果和分数。）

**来自 Run Model 页或 Home 页 Run MATR Demo 按钮的运行**（§12.5）：

```
runs/web_projects/<project>/run_model/<run_id>/    Run Model page
runs/web_projects/<project>/quickstart/<run_id>/   Run MATR Demo
├── run.json                                  run record: command, status, seed, device, git commit, scores
├── output.log                                everything the run printed
├── config_<timestamp>.yaml                   the config as BatteryML used it
└── predictions_seed_<seed>_<timestamp>.pkl   predictions and scores
```

（第一行是 Run Model 页的运行，第二行是 Run MATR Demo 的运行；`output.log` 包含运行时打印的全部内容，
其余文件的含义同上。）这两个页面仍然会写 `runs/web_projects/<project>/logs/run_model.log`（Run Model）
和 `runs/web_projects/<project>/logs/matr_demo.log`（演示），里面是最近一次运行的输出。

**来自命令行 `--workspace <folder>` 的运行**（§13）：

```
<folder>/
├── config_<timestamp>.yaml
└── predictions_seed_<seed>_<timestamp>.pkl
```

打印某个 `.pkl` 文件里保存的分数（在 `BatteryML-Agent` 文件夹中运行；把路径换成你的文件）：

```bash
python -c "import pickle; print(pickle.load(open('runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl', 'rb'))['scores'])"
```

如果要把某个 `.pkl` 的结果图导出用于论文——分开的 SVG 和 PNG 图、实际绘制数值的 CSV，
以及记录每个数字来源的 `figure_sources.json`——先查看它包含什么（`--inspect` 不写任何文件），
再导出到一个**新的**文件夹。这条命令只画图，从不训练或评估。

```bash
python -m scripts.export_result_figures --artifact runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl --inspect
python -m scripts.export_result_figures --artifact runs/cli_matr_xgb/predictions_seed_0_TIMESTAMP.pkl --output runs/figure_exports/cli_matr_xgb
```

导出的文件夹包含 `parity.svg`/`.png`、`residuals.svg`/`.png`、它们的 `_zoom` 版本、
`plotted_samples.csv`、`figure_sources.json` 和 `figure_notes.md`；SOH 结果还会有
`soh_cycle_<cell>.svg`/`.png`/`.csv` 和 `per_cell_errors.csv`。网站中每个结果下方的
**Export figures** 提供同样的导出。详见 `docs/VISUALIZATION.md`。

---

## 15. 可选：连接 LLM 助手

如果你没有 API key，跳过本节即可。第 1–14 步都不需要它。

有了 key，网站还可以用自然语言解释配置、报错和结果。它从不计算任何数字，所有数字都来自 BatteryML。

1. 在侧边栏打开 **Settings**。
2. **Provider**：选择一个预设（OpenAI、Gemini、Claude、Kimi / Moonshot、GLM / Zhipu），
   或选择 **Custom OpenAI-compatible** 以使用其他任何兼容 OpenAI 接口的服务。预设只是帮你填好 Base URL。
3. **Base URL**：你的服务商的 API 地址（选预设时会自动填入）。
4. **API key**：你的密钥。
5. **Model**：你的 key 有权限使用的模型名。
6. 先点击 **Save**，再点击 **Connect**。

**检查：** 侧边栏底部的状态行从 **LLM: Not connected** 变为 **LLM: Connected**。

API key 只保存在当前打开的浏览器会话中。它不会被写入任何文件、日志、运行记录或导出包，
会话结束后即消失。永远不要把 API key 写进这个仓库里的任何文件。

---

## 16. 以后每次重新打开

重启电脑后，或在任何新终端中：

```bash
cd ~/BatteryML-Agent
conda activate batteryml
python -m streamlit run batteryml_web/app.py
```

然后打开 `http://localhost:8501`（如果在远程服务器上，先按 §11 建立 SSH 隧道）。
你的项目、数据和运行结果都还在磁盘上。

---

## 17. 更新到新版本

```bash
cd ~/BatteryML-Agent
conda activate batteryml
git pull
pip install -r requirements.txt
python -m scripts.verify.run_all
```

更新前先用 `Ctrl+C` 停止网站，更新后再重新启动。

`git pull` 会打印 `Updating …`、`Fast-forward` 和改动的文件列表（或 `Already up to date.`）。
`pip install -r requirements.txt` 对没有变化的包会打印 `Requirement already satisfied`。

健康检查的结果按 §7 的方法看：`REG-01` 仍然是 `[FAIL]`（属预期）。有了数据之后，
`DATA-01` 和 `SPLIT-01` 不再是 `[SKIP]`。其他任何 `[FAIL]` 都会写明问题所在，见 §18。

---

## 18. 常见问题排查

| 报错或现象 | 原因 | 解决办法 |
| --- | --- | --- |
| `conda: command not found` | §2.3 之后没有重新打开终端。 | 关闭终端，重新打开。如果仍然没有，再运行一次 `$HOME/miniconda3/bin/conda init bash`。 |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | 跳过了 §2.4。 | 运行 §2.4 中的两条命令。 |
| 命令提示符显示 `(base)` 而不是 `(batteryml)` | 环境没有激活。 | `conda activate batteryml` |
| `ModuleNotFoundError: No module named 'numpy'`（或 `torch`、`streamlit` 等） | 环境没有激活，或 §5/§6 没有完成。 | `conda activate batteryml`，然后重做 §5 和 §6。 |
| `ModuleNotFoundError: … 'batteryml' is not a package` | 直接运行了 `python bin/batteryml.py`。 | 改用 `python batteryml_web/core/run_batteryml_cli.py …` |
| `ModuleNotFoundError: No module named 'batteryml_web'` | 命令不是在 `BatteryML-Agent` 文件夹里运行的。 | `cd ~/BatteryML-Agent` 后重新运行。 |
| 有 GPU 的机器上 `torch.cuda.is_available()` 打印 `False` | NVIDIA 驱动比 PyTorch 的 CUDA 版本旧。 | 见 §5 最后一部分。 |
| 出现 `[INFO] data/raw/MATR/… already exists. Skip it.`（没有 `and is complete`） | 文件夹是用旧版本下载的，旧版本会跳过损坏的文件。 | 先更新（§17），再重做 §8。 |
| §8 中出现 `error: Could not download …`，并带有 `ProxyError`、`ConnectionError` 或 `403 Forbidden` | 这台机器连不上下载服务器。没有写入任何文件。 | 先解决网络问题，再重做 §8。 |
| `error: … already exists (… bytes), but whether it is complete cannot be checked` | 连不上服务器，无法比较文件大小。 | 如果之前的下载没有报错就结束了，文件是完整的：直接做 §9。否则先解决网络问题，再重做 §8。 |
| `FileNotFoundError: Batch file not found: data/raw/MATR/…` | 下载不完整。 | 重做 §8，再做 §9。 |
| §9 中出现 `OSError: Unable to synchronously open file (file signature not found)` | 某个 `.mat` 文件是空的或不完整（旧版下载命令留下的）。 | 重做 §8，再做 §9。 |
| MATR 显示 `INCOMPLETE_PROCESSED` | 预处理失败或被中断。 | 重做 §9。 |
| 运行模型时报 `AssertionError: data/processed/MATR` | 没有执行 §9。 | 执行 §9。 |
| 运行模型时报 `KeyError: 'b1c1'` | `data/processed/MATR` 存在但是空的——§9 失败了。 | 解决 §9 报错的原因，重新执行 §9。 |
| `unrecognized arguments: --config` | 在配置文件前面写了 `--config`。 | 去掉 `--config`。 |
| 运行只打印一个路径就结束了 | 缺少 `--train --eval`。 | 两个都加上。 |
| 健康检查出现 `[FAIL] REG-01 …` 且退出码为 1 | 上游缺口，属预期。 | 无需处理。见 §7。 |
| 健康检查出现 `[FAIL] DATA-01 … processed directory/directories exist but contain no *.pkl file: MATR` | §9 失败，留下了一个空的 `data/processed/MATR`。 | 解决 §9 报错的原因，重新执行 §9。 |
| 网页显示 **No datasets are available yet.** | 网站还没有重新读取数据文件夹。 | **Prepare Data → Refresh datasets**。 |
| 网页显示 **No run is selected. Choose one in Run History above.** | 这个项目里还没有发起过任何运行。 | 运行一个模型（§12.3 或 §12.5），然后重新打开 **Review Results**。 |
| 在自己电脑上打不开 `localhost:8501` | 网站运行在远程服务器上。 | 使用 §11 中的 SSH 隧道。 |
| `CUDA out of memory` | 模型放不进 GPU 显存。 | 选择 **CPU** / `--device cpu`。 |

**完全从头再来**——只删除环境；你的代码、数据和运行结果都会保留：

```bash
conda deactivate
conda env remove -n batteryml -y
```

然后从 §4 重新开始。

---

## 19. 已知限制

- **CPTransformer** 需要把另一个仓库的源代码放到 `third_party/BatteryLife/`，而它不包含在本仓库中。
  没有它时，只有这一个模型不可用，其他所有模型都正常。健康检查（`VAL-01`）会提到它。
- BatteryLife 系列数据集（`MICH`、`MICH_EXP`、`SDU`、`Stanford_2`、`Tongji`、`XJTU`、`ISU_ILCC`）
  没有 BatteryML 预处理器。它们能被识别（`RAW_ONLY`），但无法在这里预处理。
- 在 **Run Model** 页或用 **Run MATR Demo** 发起的运行无法在网站上停止；只有 **Agent Workflow** 有 **Stop run**。
- 每次运行只能用一个设备：CPU，或一张 GPU。网站不支持多卡训练。
- 并非所有「数据集 × 模型 × 任务」的组合都有配置文件。网站只会提供实际存在的组合。
- 只在 Linux 上测试过。

---

## 20. 测试版本与验证情况

在一台没有 GPU 的 Linux x86_64 机器上，从一个空的用户目录开始，按本指南从 §1 执行到 §7 装好的版本：

| 组件 | 版本 |
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

哪些真正执行过、哪些没有：

| 步骤 | 情况 |
| --- | --- |
| §1 `git`/`curl` 检查，§2 Miniconda（下载、安装、`conda init`、重开终端、服务条款），§3 克隆，§4 环境，§5 `pip install torch` 及检查（CPU），§6 依赖与导入检查，§7 健康检查（`10 passed, 1 failed, 2 skipped`） | 完全按文中命令执行，共两遍：第二遍是把本文件中每个命令块原样复制到一个全新的终端（空的 home 目录）中执行。 |
| §8 下载命令 | 已执行；测试机器的网络策略拒绝了到 `data.matr.io` 的连接（`403 Forbidden`），所以没有下载到数据。在现在的下载命令下，这种失败只会打印一行 `error:` 并让文件夹保持为空；旧版本留下的空文件会被删除并重新下载。此处无法完成一次成功的下载。 |
| §9 预处理、§10 就绪状态 | 在无数据情况下执行：观察到了缺少原始文件时的报错（`FileNotFoundError`）、原始文件为空时的报错（`OSError … file signature not found`），以及 `INCOMPLETE_PROCESSED` 状态（`OSError` 是用旧版下载命令留下的 0 字节文件观察到的）。数量 `180` 和 `MATR READY` 来自另一台存有 MATR 预处理数据的服务器。 |
| §11 启动网站、§12.1 创建项目、§12.2 Refresh datasets、§12 中的页面与按钮名称 | 在浏览器中针对这次安装实际执行过。 |
| §12.3–12.4 在网站中运行并查看结果，§13 一次完整的运行 | 此处未执行（没有数据）。§12.3 的第 1–3 步（请求、**Use template**、**Analyze** 以及无数据时的提示）在浏览器中执行过；第 4–8 步中的名称取自代码。§13 的命令行检查到数据步骤为止：该文件夹不存在时报 `AssertionError: data/processed/MATR`，存在但为空时报 `KeyError: 'b1c1'`。一次 1 个 epoch 的 MATR CNN 运行在另一台服务器上完成过。Run Model 和 **Run MATR Demo** 的运行会列在 Review Results 中（§12.5）这一点只有单元测试覆盖；此处无法发起这样的运行。 |
| §5 的 GPU 情形、`--index-url` 命令、`nvidia-smi` | 未执行（测试机器没有 GPU；`download.pytorch.org` 被拦截）。 |
| §15 LLM 助手 | 在浏览器中核对了字段、按钮和六个服务商预设；没有实际建立连接。 |
| §16 再次使用、§17 更新、§18「从头再来」 | 完全按文中命令执行（网站启动后用 `Ctrl+C` 停止；`git pull` 快进更新；`conda env remove` 之后只剩 `base`）。 |
| macOS、Windows/WSL2 | 未测试。 |

---

## 延伸阅读

| 文档 | 内容 |
| --- | --- |
| [`README_BATTERYML.md`](./README_BATTERYML.md) | 原版 BatteryML README：论文、数据集、基准表格 |
| [`dataprepare.md`](./dataprepare.md) | 各数据集的手动下载说明 |
| [`docs/VERIFY.md`](./docs/VERIFY.md) | 健康检查每一行测试的内容 |
| [`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) | 接入你自己的模型 |

## 引用

BatteryML-Agent 构建于 BatteryML 之上。如果你使用了它，请引用：

```bibtex
@inproceedings{zhang2024batteryml,
  title={Battery{ML}: An Open-source Platform for Machine Learning on Battery Degradation},
  author={Han Zhang and Xiaofan Gui and Shun Zheng and Ziheng Lu and Yuqi Li and Jiang Bian},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
