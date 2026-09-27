<div align="center">

# BatteryML-Agent

**面向电池研究者的部署与使用指南。**
不需要机器学习背景，不需要 GPU。

[English](./README.md) · **简体中文**

</div>

---

## 这是什么

**[BatteryML](https://github.com/microsoft/BatteryML)**（Microsoft，ICLR 2024）
是一个用于电池退化机器学习的开源库。它定义了数据格式、预处理器、训练/测试划分、
模型与指标。它是科学内核——本平台报告的每一个数字都来自它。

**BatteryML-Agent** 是构建其上的工作流层。它增加了：

- 覆盖整条流程的**网页界面**——查看数据、预处理、配置运行、执行、阅读结果；
- **数据集就绪状态**——直接回答「这个数据集现在能不能跑，不能的话还缺什么」；
- 磁盘上**固定的结果目录结构**，使每次运行可引用、可复现；
- 一个**可选的 LLM 助手**，用于解释配置、报错与结果。

它**不增加任何新的科学内容**，不改动 BatteryML 的模型、指标或数据划分。

### 两条路线

| 路线 | 是什么 | 需要 API key 吗？ |
| --- | --- | --- |
| **路线 B — 确定性** | 网页界面，或一条命令行。即 §1–§12 的全部内容。 | **不需要。** 可完全离线运行。 |
| **路线 A — LLM 辅助** | 叠加在上面的可选助手（§13）。 | 需要——且完全可选。 |

**路线 B 是默认路线，也是本指南所讲的内容。** 即使你从不配置助手，
也不会有任何功能降级：每个页面都能手动操作。

> **助手永远不产生任何数字。** 它不训练、不预测、不计算指标，
> 也不判断数据集是否就绪。这些全部来自 BatteryML 和确定性的就绪判断层。
> 助手只负责解释已经存在的内容。

---

## 目录

1. [开始之前](#1-开始之前)
2. [获取代码](#2-获取代码)
3. [创建 Python 环境](#3-创建-python-环境)
4. [安装依赖](#4-安装依赖)
5. [验证环境](#5-验证环境)
6. [获取电池数据](#6-获取电池数据)
7. [预处理并登记数据](#7-预处理并登记数据)
8. [数据集就绪状态](#8-数据集就绪状态)
9. [启动网页应用](#9-启动网页应用)
10. [运行冒烟测试](#10-运行冒烟测试)
11. [查看结果](#11-查看结果)
12. [什么才算基准性能](#12-什么才算基准性能)
13. [可选：LLM 助手](#13-可选llm-助手)
14. [常见问题排查](#14-常见问题排查)
15. [已知限制](#15-已知限制)
16. [本指南的验证情况](#16-本指南的验证情况)

---

## 1. 开始之前

### 你需要准备

| 项目 | 要求 | 如何确认 |
| --- | --- | --- |
| 操作系统 | Linux、macOS，或 Windows（建议 WSL2） | — |
| Python | **3.9 或更高** | `python3 --version` |
| 磁盘空间 | **约 10 GB 可用**（仅 MATR 原始数据就有数 GB） | `df -h .` |
| 内存 | 最低 8 GB，16 GB 更从容 | — |
| GPU | **不需要。** 默认用 CPU；支持单张 GPU。 | — |
| 网络 | 需要一次，用于安装依赖和下载数据 | — |

### 大概需要多久

| 步骤 | 大致时间 |
| --- | --- |
| §2–§5 — 安装与验证 | 15–30 分钟 |
| §6 — 下载 MATR | 20–60 分钟，取决于网速 |
| §7 — 预处理 MATR | 10–30 分钟 |
| §9–§10 — 第一次运行 | CPU 上冒烟运行 1–5 分钟 |

> 这些是**估计值，不是实测值**，取决于你的网络、磁盘和 CPU。
> §16 说明了哪些计过时、哪些没有。

### 关于终端

下面每条命令都输入在终端里（macOS 的「终端」，Windows 上的 Ubuntu/WSL）。
以 `#` 开头的行是注释，**不用输入**。一次复制一个代码块，然后回车。

---

## 2. 获取代码

```bash
git clone https://github.com/ZishanShu/BatteryML-Agent.git
cd BatteryML-Agent
```

后面所有命令都默认你**位于 `BatteryML-Agent` 目录内**。
如果之后新开了终端，请先 `cd` 回到这个目录。

确认你在正确的位置：

```bash
ls
```

你应该看到 `batteryml`、`batteryml_web`、`configs`、`requirements.txt` 等。
如果没看到，说明目录不对。

---

## 3. 创建 Python 环境

这一步把 BatteryML 的依赖和你电脑上其他东西隔离开，
避免升级或破坏你已经在用的包。**二选一即可。**

<details open>
<summary><b>方案 A — conda（如果你已经在用 Anaconda，推荐）</b></summary>

```bash
conda create -n batteryml python=3.10 -y
conda activate batteryml
```

每次新开终端，都需要重新运行 `conda activate batteryml`。

</details>

<details>
<summary><b>方案 B — venv（Python 自带，无需额外安装）</b></summary>

```bash
python3 -m venv .venv

# macOS / Linux：
source .venv/bin/activate

# Windows PowerShell：
.venv\Scripts\Activate.ps1
```

每次新开终端，都需要重新运行上面的 `activate` 那一行。

</details>

**确认是否成功。** 命令行提示符前面应该出现 `(batteryml)` 或 `(.venv)`。
检查 Python 版本：

```bash
python --version
```

期望输出：`Python 3.9.x` 或更高。
如果显示 3.8 或更低，请退回去用更新的 Python 重建环境。

---

## 4. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

这会安装 numpy、pandas、scikit-learn、xgboost、torch、streamlit 等。
这是最慢的一步，花上几分钟是正常的，其中 `torch` 体积最大。

> **如果你的机器没有 NVIDIA 显卡**，默认的 `torch` 依然可用——它会在 CPU 上运行。
> 本指南中的任何步骤都不需要 CUDA。

确认关键依赖都能导入：

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

期望输出：

```
all imports OK
```

如果某一行报错，重新运行 `pip install -r requirements.txt` 并阅读报错，
它通常会指出缺少哪个系统库。

---

## 5. 验证环境

在接触任何数据之前，先确认安装是正常的：

```bash
python -m scripts.verify.run_all
```

它会运行 13 项独立检查，每项打印一行。
在全新克隆、**还没有数据**的情况下，期望结果是：

```
9 passed, 2 failed, 2 skipped of 13 check(s)
```

等你预处理完一个数据集（§7）之后，`DATA-01` 就会开始通过，
结果变为 `10 passed, 2 failed, 1 skipped`。

**此阶段你真正该关注的：**

| 检查项 | 应该是 | 如果不是 |
| --- | --- | --- |
| `CLI-01 cli_contract` | `[PASS]` | 安装不完整，重做 §4 |
| `EXT-01 extension_contract` | `[PASS]` | 同上 |
| `CFG-01 config_validity` | `[PASS]` | 阅读提示信息，它会指出是哪个配置 |
| `DATA-01 dataset_status` | §6 之前 `[SKIP]`，§7 之后 `[PASS]` | 它报告数据状态，没有数据时就跳过 |
| `SPLIT-01 split_requirements` | `[SKIP]` | 正常——你还没有预处理数据 |

**全新克隆上预期会出现的两个 failed**，它们都不影响本指南的任何步骤：

- `ENV-01 environment` —— 报告 `fire` 未安装。它在 `requirements.txt` 中声明，
  但仓库里没有任何模块 import 它。
- `REG-01 registry_integrity` —— 报告上游 BatteryML 声明了六种 cycler 输入格式
  （`BATTERYARCHIVE`、`BIOLOGIC`、`INDIGO`、`LANDT`、`MACCOR`、`NOVONIX`）
  却没有对应的预处理器。这是**上游的缺口**，不是 BatteryML-Agent 的功能退化。

你可能还会看到 `VAL-01` 提示有 1 个配置被**正确拒绝**，
原因是某个第三方源码没有包含在本仓库中。那是 `CPTransformer`——
一个可选的深度模型，其网络代码位于另一个仓库。
这对你没有任何影响，其他所有模型都能正常运行。详见 §15。

逐项说明见 [`docs/VERIFY.md`](./docs/VERIFY.md)。
机器可读的报告会写到 `runs/verify/verification_report.json`。

---

## 6. 获取电池数据

本指南使用 **MATR**（MIT/Toyota 的 LFP 数据集，180 个电芯）——本平台的参考示例。

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

> **一定要通过 `batteryml_web/core/run_batteryml_cli.py` 调用命令行。**
> 直接运行 `python bin/batteryml.py` 会报
> `ModuleNotFoundError: ... 'batteryml' is not a package`，
> 因为该脚本的文件名遮蔽了它自己要导入的同名包。

这会下载 4 个 `.mat` 文件（总计数 GB），需要一段时间。下载完成后确认：

```bash
ls -lh data/raw/MATR
```

期望看到：`MATR_batch_20170512.mat`、`MATR_batch_20170630.mat`、
`MATR_batch_20180412.mat`、`MATR_batch_20190124.mat`。

**这条命令能下载的数据集：** `MATR`、`HUST`、`CALCE`、`RWTH`。

**BatteryML 能预处理但不能自动下载的数据集**——`HNEI`、`SNL`、`UL_PUR`、`OX`
——需要手动获取，参见 [`dataprepare.md`](./dataprepare.md)，
放到 `data/raw/<名称>/` 下，再执行 §7。

**其他数据集**（就绪层会列出的 `MICH`、`MICH_EXP`、`SDU`、`Stanford_2`、
`Tongji`、`XJTU`、`ISU_ILCC`）来自 BatteryLife 数据分发，
没有对应的原生 BatteryML 预处理器。它们**能被识别为原始数据，但无法在这里预处理**。
详见 §8 与 §15。

---

## 7. 预处理并登记数据

原始循环仪文件不是模型直接读取的格式。
这一步把它们转换成 BatteryML 统一的 `BatteryData` 格式。

```bash
python batteryml_web/core/run_batteryml_cli.py preprocess \
  MATR data/raw/MATR data/processed/MATR
```

确认：

```bash
ls data/processed/MATR | wc -l
```

期望输出：**180**——每个电芯一个 `.pkl` 文件。

然后生成网页界面读取的清单文件。**Dataset Overview** 页面列出的就是它：

```bash
python tools/sync_data_and_config_manifests.py
```

它会在当前这个仓库内写出 `data/metadata/dataset_manifest.json` 和
`data/metadata/model_config_manifest.json`。确认 MATR 已登记：

```bash
python -c "
import json
d = json.load(open('data/metadata/dataset_manifest.json'))
print('present datasets:', [x['name'] for x in d if x.get('status') == 'present'])
"
```

期望输出：`present datasets: ['MATR']`，以及你准备好的其他数据集。

> 如果要描述**另一个**仓库副本（而不是脚本所在的这个），
> 运行前设置 `BATTERYML_ROOT=/path/to/checkout` 即可。

> **数据流向始终是
> `data/raw/<名称>` → 预处理 → `data/processed/<名称>` → 运行。**
> 训练永远不读原始文件。如果之后看到 `AssertionError: data/processed/MATR`，
> 说明这一步没有执行，**并不是**网页层坏了。

---

## 8. 数据集就绪状态

就绪状态完全由文件系统推断得出，与网页界面显示的是同一个答案。
你随时可以查看：

```bash
python -c "
from batteryml_web.core.readiness import list_dataset_readiness
for r in list_dataset_readiness():
    print(f'{r.dataset:<12} {r.status.value}')
"
```

一共只有四种状态：

| 状态 | 含义 | 该怎么做 |
| --- | --- | --- |
| **`READY`** | 已有处理好的 `.pkl` 电芯文件。 | 可以直接运行。 |
| **`RAW_ONLY`** | 已识别到原始数据，但还没有预处理产物。 | 执行预处理（§7）——前提是它有可用的预处理器（§15）。 |
| **`INCOMPLETE_PROCESSED`** | 存在 processed 目录，但里面没有可用的 `.pkl` 电芯。 | 重新预处理；上一次没有完成。 |
| **`MISSING`** | 既没有识别到原始数据，也没有预处理产物。 | 下载或放入原始数据（§6）。 |

原始数据有三种被识别的形态：BatteryML 原生的 `data/raw/<名称>/`、
已解压的 BatteryLife 分发 `data/raw/BatteryLife/v11/<名称>/`，
以及与之并列的压缩包 `<名称>.zip`。
一个只有 `.zip` 文件名、内容却不是有效压缩包的文件（即下载中断的残留）
会被**刻意判定为不存在**。

<details>
<summary><b>示例——某台真实机器的就绪状态（不是通用预期）</b></summary>

这是某台已配置好的研究服务器报告的结果。你的机器会完全不同，
它只取决于你下载了什么。

```
READY                 MATR（180 个处理后电芯）、HUST（77 个处理后电芯）
RAW_ONLY              CALCE、MICH、MICH_EXP、RWTH、SDU、Stanford_2、Tongji、XJTU
INCOMPLETE_PROCESSED  HNEI、SNL、UL_PUR
MISSING               ISU_ILCC
```

</details>

---

## 9. 启动网页应用

```bash
python -m streamlit run batteryml_web/app.py
```

打开它打印出的地址，通常是 <http://localhost:8501>。

> **在远程服务器上？** 加上 `--server.port 8501 --server.address 0.0.0.0`，
> 或者从本地笔记本做 SSH 端口转发：
> `ssh -L 8501:localhost:8501 你@服务器`，然后在本地打开 `localhost:8501`。

### 确定性工作流（路线 B）

1. **Prepare Data** → **Dataset Overview** → **Refresh datasets**。
   MATR 应该显示为 **READY**。
2. **Run Model** → 选择 **Dataset**、**Task**（RUL prediction 或 SOH estimation）、
   **Model**、**Device**、**Seed**。
3. 阅读页面上的**命令预览**。那就是即将执行的完整命令——没有隐藏或额外添加的内容。
4. 点击深色的 **Run BatteryML** 按钮。进度会实时刷新。
5. **Review Results** → 指标表、实测-预测对比图、残差图和预测表格。

界面约定：**深色实心按钮**永远是该页面的主操作，
**描边按钮**是辅助操作，**扁平面板**只是信息展示。

**设备。** 正常界面支持 **CPU 或单张 GPU**（`cpu`，或单个 `cuda:N`）。
多卡与分布式训练**不是**受支持的产品功能，详见 §15。

不开浏览器，用命令行完成同样的运行：

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace runs/native_matr_xgb \
  --train --eval --device cpu --seed 0
```

**三个最容易写错的地方：**

- 配置文件路径是**位置参数**。写成 `run --config configs/...` 会报
  `unrecognized arguments: --config`。
- `--train --eval` 是**必须的**。不加这两个参数，BatteryML 只会打印
  workspace 路径然后退出，什么也不做。
- 始终传入 `--seed`，这样结果才可复现。

---

## 10. 运行冒烟测试

冒烟测试只回答一个问题：**这台机器上，整条执行链路能不能跑通？**
它**不是**对模型质量的衡量。

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/nn_models/cnn/matr_1.yaml \
  --workspace runs/smoke_matr_cnn \
  --train --eval --device cpu --seed 0 --epochs 1
```

`--epochs 1` 正是让它成为冒烟测试的原因。
如果你有一张 GPU，可以把 `--device cpu` 换成 `--device cuda:0`。

成功的冒烟运行会输出指标和完整的预测结果。作为参考，
在 `cuda:0` 上用完全相同的配置做一次真实的单轮运行，得到：

```
RMSE  381.609375
MAE   254.52743530273438
MAPE  0.34497949481010437
42 predictions, 42 distinct values
```

> ### ⚠ 这些数字**不是**基准性能
>
> 一个 epoch 几乎训练不了什么。冒烟运行只告诉你流程接通了——
> 数据能加载、特征能构建、模型能训练、指标能计算、产物能写出。
> **它完全说明不了模型预测得有多好。**
> 不要引用、比较或发表冒烟运行的数字。
> 关于什么才算基准性能，见 §12。

---

## 11. 查看结果

每一次运行——无论来自网页界面还是命令行——都会写入
`runs/web_projects/<项目名>/`：

```
project_state.json
config.yaml
schema_mapping.json          # 仅在你导入了自定义数据集时生成
run_metadata.json            # git commit、包版本、seed、设备、
                             # 配置哈希、数据指纹、耗时
environment_check.json
dataset_status.csv
feature_summary.json
training_log.txt
metrics.csv                  # <- 你的 RMSE / MAE / MAPE
predictions/
figures/
checkpoints/
report.html
```

**`metrics.csv` 里就是你要引用的数字。**
`run_metadata.json` 记录了复现这些数字所需的全部信息。

在界面里，**Review Results** 会显示指标表、实测-预测对比图、残差图和预测表格。
如果结果是「每个电芯一个寿命估计值」这种形态，就只显示这几项——
**不会**为它画循环维度的曲线，因为那样的曲线并不存在。

**可复现性。** 相同配置 + 相同 seed + 相同数据指纹，
会产生完全一致的 `metrics.csv`。
如果某个模型本身具有不确定性，这一事实会被**如实记录**在
`run_metadata.json` 中，而不是被掩盖。

---

## 12. 什么才算基准性能

上游 BatteryML 发布了 RUL 任务的基准表格。它位于
**[`README_BATTERYML.md`](./README_BATTERYML.md)**（保留下来的上游 README），
那才是用于引用和比较的参考。

正确阅读方式：

- 指标是 **RMSE**，单位是**循环数（cycles）**，**越低越好**；
- **MATR1 与 MATR2 是不同的实验协议**——训练/测试划分不同。
  MATR1 的结果不能和 MATR2 的数字相比；
- **SOH** 的结果永远不能与 **RUL** 的基准相比；
- **冒烟运行的结果不能与任何东西相比**（§10）。

给一个具体的参照点：**MATR1 + XGBoost** 的已发表 RMSE 是 **334 个循环**。
如果你用完全相同的配置做一次正常训练，结果落在这个量级附近，
说明流程接通了。
而使用了不同划分、不同任务，或只跑了一个 epoch 的运行，则不能这样比较。

平台会替你把关：当实验协议不匹配时，结果页面会**拒绝**把已发表的基准值
与你的结果并列显示。

---

## 13. 可选：LLM 助手

**如果你只想要预测结果，可以完全跳过本节。**
上面所有内容都不依赖它，可离线运行，不需要任何账号。

助手用于解释配置、诊断报错、用自然语言解读结果。
它不计算任何数字，也不判断就绪状态。

在左侧栏打开 **Settings**，填写：

| 字段 | 填什么 |
| --- | --- |
| **Provider** | 选一个预设，或选 **Custom OpenAI-compatible** 自己填写端点 |
| **Base URL** | 你的服务商提供的 OpenAI 兼容端点 |
| **Model** | 你的 key 有权限使用的模型名 |
| **API key** | 你的密钥——仅保存在当前浏览器会话中 |

然后先点 **Save**（在本会话中保存），再点 **Connect**（测试连接）。

任何 OpenAI 兼容端点都可以使用。下拉框提供的预设
——OpenAI、Gemini、Claude、Kimi / Moonshot、GLM / Zhipu——
纯粹是为了方便，帮你自动填好 Base URL；预设**不会锁定**该字段，
选择 **Custom OpenAI-compatible** 可以填入任何你有权访问的网关。
列出某个预设不代表推荐，也不是必须使用其中之一。

**关于你的 API key：**

- 它**只保存在当前浏览器会话中**。
- 它**绝不会**被写入任何仓库文件、配置、项目元数据、`run_metadata.json`、
  运行历史、日志、导出的结果包，或展示给你的报错信息中。
- 会话结束即消失，下次需要重新输入。
- **永远不要**把 API key 粘贴进任何会被 git 提交的文件里。

如果助手不可用——没有 key、没有网络、服务商限流——
每个页面仍然完全可以手动操作。这是设计上的硬性要求，不是降级方案。

---

## 14. 常见问题排查

| 你看到的报错 | 含义 | 解决办法 |
| --- | --- | --- |
| `ModuleNotFoundError: ... 'batteryml' is not a package` | 你直接运行了 `python bin/batteryml.py` | 改用 `python batteryml_web/core/run_batteryml_cli.py ...` |
| `unrecognized arguments: --config` | 配置路径是位置参数 | 写 `run configs/....yaml`，不要写 `run --config ...` |
| 运行后只打印 workspace 路径就退出了 | 缺少 `--train --eval` | 补上这两个参数 |
| `AssertionError: data/processed/MATR` | 没有执行预处理 | 执行 §7。这**不是**网页层的 bug |
| 网页显示「No datasets are available yet」 | 预处理之后没有重新生成清单 | 运行 `python tools/sync_data_and_config_manifests.py`，再点 **Refresh datasets** |
| 某个数据集显示 `RAW_ONLY` 且无法预处理 | 它没有原生预处理器 | 见 §8 与 §15 |
| 某个数据集显示 `INCOMPLETE_PROCESSED` | 预处理没有完成 | 对该数据集重新执行 §7 |
| `[FAIL] ENV-01 ... not installed: fire` | 声明了但未被使用的依赖 | 全新克隆上属预期（§5）；`pip install fire` 可消除 |
| `[FAIL] REG-01 ... have no registered preprocessor` | 上游 BatteryML 的缺口 | 属预期（§5）；不是 BatteryML-Agent 的退化 |
| 远程服务器上浏览器打不开 | Streamlit 只绑定了 localhost | 用 SSH 端口转发，或加 `--server.address 0.0.0.0` |
| `CUDA out of memory` | 模型超出显存 | 改用 `--device cpu`，或换更小的模型。本指南不需要 GPU |

还是卡住？运行 `python -m scripts.verify.run_all`，
把输出和 `runs/verify/verification_report.json` 一起提 issue。
那份报告会指明是哪一项检查失败以及对应的修正动作。

---

## 15. 已知限制

如实写在这里，因为做到一半才发现这些，比现在读到要糟糕得多。

- **有些模型需要你自行提供第三方源码。** `CPTransformer` 封装的网络代码
  位于另一个仓库（`third_party/` 下），本仓库不包含。
  它会被**自动跳过**，不影响其他任何功能；它的配置会被报告为
  「不可用」而不是「有错」。
- **有些数据集在这里只能停留在 `RAW_ONLY`。** BatteryLife 系列数据集
  （`MICH`、`MICH_EXP`、`SDU`、`Stanford_2`、`Tongji`、`XJTU`、`ISU_ILCC`）
  能被识别为原始数据，但没有原生 BatteryML 预处理器，
  因此无法通过本流程达到 `READY`。
- **有六种 cycler 格式在上游被声明但未实现**：
  `BATTERYARCHIVE`、`BIOLOGIC`、`INDIGO`、`LANDT`、`MACCOR`、`NOVONIX`。
  `REG-01` 每次都会报告这一点（§5）。
- **一个 epoch 的运行是冒烟测试，不是结果**（§10）。
- **并非所有「数据集 × 模型 × 任务」的组合都可运行。** Run Model 页面
  只会提供配置与数据都确实存在的组合；缺任何一项的组合会被报告为
  blocked 并给出原因，而不是在执行时才悄悄失败。
- **正常界面只支持 CPU 或单张 GPU。** 多卡与分布式（DDP）训练
  **不是**受支持的产品功能，界面上也不提供，请不要据此做规划。
- **LLM 助手依赖网络与服务商访问。** 它是可选的、可能不可用，
  路线 B 的任何环节都不依赖它。
- **Windows 未经验证。** 本指南的命令只在 Linux 上实际执行过。

---

## 16. 本指南的验证情况

这里把话说清楚，因为一份夸大自己测试范围的教程只会浪费你的时间。

**在干净的 Linux 容器（Python 3.11、无任何数据集）中实际执行过：**

- `pip install -r requirements.txt` 以及 §4 的导入自检
- `python -m scripts.verify.run_all`（§5）——在**没有 `data/` 也没有
  `third_party/` 的全新 worktree** 中：**9 passed、2 failed、2 skipped**，
  两个 failed 正是 `ENV-01`（`fire`）与 `REG-01`（上游预处理器缺口），
  与 §5 描述完全一致
- 在没有 `third_party/` 目录的克隆上运行
  `python batteryml_web/core/run_batteryml_cli.py run --help`——可正常启动并打印用法
- §8 的就绪状态查询命令——已运行，退出码 0，输出了那四种状态的词汇表
- §10 的冒烟运行命令——已解析并运行到
  `AssertionError: data/processed/MATR`，
  这验证了包括 `--epochs` 在内的**命令语法**，仅此而已
- `python tools/sync_data_and_config_manifests.py`（§7）——
  在仓库内写出了 15 条数据集记录和 180 条模型配置记录
- `python -m streamlit run batteryml_web/app.py`（§9）——已启动，
  上文提到的各个页面在真实浏览器中渲染成功
- 仓库自带的测试套件：**1355 个测试，0 失败**

**由一台已配置好的研究服务器报告，并非在此执行：**

- MATR `READY`，180 个处理后电芯；HUST `READY`，77 个
- `python -m unittest -v tests.test_agent_config_selection`——12/12 通过
- §10 中 MATR + CNN 的单轮冒烟运行，包括其指标与 42 个互不相同的预测值

**在编写本指南的过程中，任何环境下都未执行过：**

- MATR 的下载（§6）与预处理（§7）。本环境没有下载任何数据集。
  那 4 个文件名与「180」这个数量来自数据集登记与项目记录。
- 完整的多轮训练/评估。§12 中的 334 个循环是**上游已发表的基准值**，
  不是这里实测出来的。
- §1 的所有时间——均为估计值。
- macOS 与 Windows。

如果某一步在你的机器上表现不同，欢迎提 issue——
那是本指南的疏漏，不是你的问题。

---

## 延伸阅读

| 文档 | 内容 |
| --- | --- |
| [`README_BATTERYML.md`](./README_BATTERYML.md) | 上游 BatteryML：论文、数据集、完整基准表 |
| [`dataprepare.md`](./dataprepare.md) | 各数据集的手动准备方法 |
| [`docs/VERIFY.md`](./docs/VERIFY.md) | 每一项验证检查证明了什么 |
| [`docs/EXTENDING_MODELS.md`](./docs/EXTENDING_MODELS.md) | 如何接入你自己的模型 |
| [`docs/AGENT_WORKFLOW.md`](./docs/AGENT_WORKFLOW.md) | 可选助手的实现方式 |
| [`docs/SOH_PROTOCOL.md`](./docs/SOH_PROTOCOL.md) | SOH 估计的实验协议 |

## 引用

BatteryML-Agent 构建于 BatteryML 之上。如果本项目对你有帮助，请引用：

```bibtex
@inproceedings{zhang2024batteryml,
  title={Battery{ML}: An Open-source Platform for Machine Learning on Battery Degradation},
  author={Han Zhang and Xiaofan Gui and Shun Zheng and Ziheng Lu and Yuqi Li and Jiang Bian},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
