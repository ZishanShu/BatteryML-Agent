<div align="center">

# BatteryML-Agent — 部署指南

**写给电池研究者的逐步部署教程。**
不需要机器学习背景，不需要 GPU。

[English](./README.md) · **简体中文**

</div>

---

## 你要安装的是什么

BatteryML-Agent 是建立在
[BatteryML](https://github.com/microsoft/BatteryML)（ICLR 2024）之上的网页界面与工作流层。
你提供电池循环数据，它预测 **RUL**（剩余使用寿命）或 **SOH**（健康状态），
并把每一次结果按固定、可引用的目录结构写到磁盘上。

有两种使用方式，而**第二种你完全可以不用**：

| 路线 | 是什么 | 需要 API key 吗？ |
| --- | --- | --- |
| **路线 B — 快速开始** | 点击网页界面，或运行一条命令行。完全确定性。 | **不需要。** 可离线运行。 |
| **路线 A — LLM 辅助** | 一个可选助手，解释配置、报错与结果。 | 需要，但完全可选。 |

**本指南只讲路线 B。** 助手放在最后（§9）作为附加内容。
即使你永远不配置它，也不会有任何功能失效——每个页面都能手动操作。

> **助手永远不会产生任何数字。** 它不训练、不预测、不计算指标。
> 你看到的每一个数值都来自 BatteryML 本身。

---

## 目录

1. [开始之前](#1-开始之前)
2. [获取代码](#2-获取代码)
3. [创建 Python 环境](#3-创建-python-环境)
4. [安装依赖](#4-安装依赖)
5. [健康检查](#5-健康检查)
6. [下载电池数据](#6-下载电池数据)
7. [预处理数据](#7-预处理数据)
8. [运行你的第一次预测](#8-运行你的第一次预测)
9. [可选：LLM 助手](#9-可选llm-助手)
10. [结果在哪里](#10-结果在哪里)
11. [常见问题排查](#11-常见问题排查)
12. [本指南的验证情况](#12-本指南的验证情况)

---

## 1. 开始之前

### 你需要准备

| 项目 | 要求 | 如何确认 |
| --- | --- | --- |
| 操作系统 | Linux、macOS，或 Windows（建议用 WSL2） | — |
| Python | **3.9 或更高** | `python3 --version` |
| 磁盘空间 | **约 10 GB 可用**（仅 MATR 原始数据就有数 GB） | `df -h .` |
| 内存 | 最低 8 GB，16 GB 更从容 | — |
| GPU | **不需要。** 默认示例在 CPU 上运行。 | — |
| 网络 | 需要一次，用于下载依赖包和数据 | — |

### 大概需要多久

| 阶段 | 大致时间 |
| --- | --- |
| 第 2–6 步（安装 + 健康检查） | 15–30 分钟 |
| 第 7 步（下载 MATR） | 20–60 分钟，取决于网速 |
| 第 8 步（预处理 MATR） | 10–30 分钟 |
| 第 9 步（第一次预测） | CPU 上 1–5 分钟 |

> 下载与预处理时间是**估计值，不是实测值**，取决于你的网络和磁盘。见 §12。

### 关于终端

下面每一条命令都输入在终端里（macOS 的「终端」，Windows 上的 Ubuntu/WSL）。
以 `#` 开头的行是注释，**不用输入**。
一次复制一个代码块，然后回车。

---

## 2. 获取代码

```bash
git clone https://github.com/ZishanShu/BatteryML-Agent.git
cd BatteryML-Agent
```

本指南后面所有命令都默认你**位于 `BatteryML-Agent` 目录内**。
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

**确认是否成功。** 你的命令行提示符前面应该出现 `(batteryml)` 或 `(.venv)`。
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

> **如果你的机器没有 NVIDIA 显卡**，默认的 `torch` 依然可以用——它会在 CPU 上运行。
> 本指南的示例不需要 CUDA。

确认关键依赖都能正常导入：

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch, streamlit; print('all imports OK')"
```

期望输出：

```
all imports OK
```

如果某一行报错，重新运行 `pip install -r requirements.txt` 并阅读报错信息，
它通常会指出缺少哪个系统库。

---

## 5. 健康检查

在接触任何数据之前，先确认安装是正常的：

```bash
python -m scripts.verify.run_all
```

它会运行 13 项独立检查，每项打印一行。
在**刚装好、还没有数据**的情况下，出现下面这些是正常且符合预期的：

```
[SKIP] SPLIT-01 split_requirements: data/processed does not exist...
...
X passed, Y failed, 1 skipped of 13 check(s)
```

**此阶段你真正该关注的：**

| 检查项 | 应该是 | 如果不是 |
| --- | --- | --- |
| `ENV-01 environment` | `[PASS]`，或 `[FAIL]` 并指出缺哪个包 | 按提示安装后重跑 |
| `CLI-01 cli_contract` | `[PASS]` | 安装不完整，重做第 4 步 |
| `EXT-01 extension_contract` | `[PASS]` | 同上 |
| `CFG-01 config_validity` | `[PASS]` | 阅读提示信息，它会指出是哪个配置 |
| `SPLIT-01` | `[SKIP]` | 正常——你还没有数据 |

> **有一条提示你会看到，可以忽略。** `VAL-01` 会报告有 1 个配置
> 「correctly refused because a third-party source is not vendored here」。
> 那是 `CPTransformer`——一个可选的深度模型，它的网络代码位于另一个仓库，
> 本仓库不包含。这对你没有任何影响：其他所有模型（XGBoost、随机森林、
> Ridge、SVM、CNN、variance / discharge 模型、BatLiNet）都能正常运行，
> 本指南的示例也不使用它。

逐项说明见 [`docs/VERIFY.md`](./docs/VERIFY.md)。
机器可读的报告会写到 `runs/verify/verification_report.json`。

## 6. 下载电池数据

我们使用 **MATR**（MIT/Toyota 的 LFP 数据集，180 个电芯）——本平台的参考示例。

```bash
python batteryml_web/core/run_batteryml_cli.py download MATR data/raw/MATR
```

> **一定要通过 `batteryml_web/core/run_batteryml_cli.py` 调用命令行。**
> 直接运行 `python bin/batteryml.py` 会报
> `ModuleNotFoundError: ... 'batteryml' is not a package`，
> 因为该脚本的文件名遮蔽了它自己要导入的同名包。

这会下载 4 个 `.mat` 文件（总计数 GB），需要一段时间。
下载完成后确认：

```bash
ls -lh data/raw/MATR
```

期望看到 4 个文件：`MATR_batch_20170512.mat`、`MATR_batch_20170630.mat`、
`MATR_batch_20180412.mat`、`MATR_batch_20190124.mat`。

**支持一键下载的数据集：** `MATR`、`HUST`、`CALCE`、`RWTH`。

**BatteryML 能预处理但不能自动下载的数据集**
（`HNEI`、`SNL`、`UL_PUR`、`OX`）需要你手动获取——
参见 [`dataprepare.md`](./dataprepare.md)——
然后放到 `data/raw/<名称>/` 下，再执行第 8 步。

---

## 7. 预处理数据

原始循环仪文件并不是模型直接读取的格式。
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

然后生成网页界面读取的清单文件——Dataset Overview 页面列出的就是它：

```bash
python tools/sync_data_and_config_manifests.py
```

它会写出 `data/metadata/dataset_manifest.json` 和
`data/metadata/model_config_manifest.json`。
确认 MATR 已被登记为 present：

```bash
python -c "
import json
d = json.load(open('data/metadata/dataset_manifest.json'))
print('present datasets:', [x['name'] for x in d if x.get('status') == 'present'])
"
```

期望输出：`present datasets: ['MATR']`（以及你准备好的其他数据集）。

> **数据流向始终是
> `data/raw/<名称>` → 预处理 → `data/processed/<名称>` → 运行。**
> 训练永远不读原始文件。
> 如果之后看到 `AssertionError: data/processed/MATR`，
> 说明这一步没有执行，**并不是**网页层或助手坏了。

---

## 8. 运行你的第一次预测

两种方式，效果相同。

<details open>
<summary><b>方案 A — 网页界面（首次使用推荐）</b></summary>

```bash
python -m streamlit run batteryml_web/app.py
```

打开它打印出的地址，通常是 <http://localhost:8501>。

> **在远程服务器上？** 加上 `--server.port 8501 --server.address 0.0.0.0`，
> 或者从本地笔记本做 SSH 端口转发：
> `ssh -L 8501:localhost:8501 你@服务器`，然后在本地打开 `localhost:8501`。

然后在浏览器里：

1. **Prepare Data**（左侧栏）→ **Dataset Overview** 标签页 → 点击
   **Refresh datasets**。MATR 应该出现，状态为 **READY**。
2. **Run Model** → 选择 **Dataset: MATR**、**Task: RUL prediction**、
   **Model: XGBoost**、**Device: cpu**、**Seed: 0**。
3. 查看页面上显示的命令预览——这就是即将执行的完整命令，没有任何隐藏。
4. 点击深色的 **Run BatteryML** 按钮。进度会实时刷新。
5. **Review Results** → 指标表、实测-预测对比图、残差图和预测表格。

界面约定：**深色实心按钮**永远是该页面的主操作，
**描边按钮**是辅助操作，**扁平面板**只是信息展示、不可点击。

</details>

<details>
<summary><b>方案 B — 一条命令行（不开浏览器）</b></summary>

```bash
python batteryml_web/core/run_batteryml_cli.py run \
  configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace runs/native_matr_xgb_test \
  --train --eval --device cpu --seed 0
```

**三个最容易写错的地方：**

- `config_path` 是**位置参数**。写成 `run --config configs/...` 会报
  `unrecognized arguments: --config`。
- `--train --eval` 是**必须的**。不加这两个参数，BatteryML 只会打印
  workspace 路径然后退出，什么也不做。
- 始终传入 `--seed`，这样结果才可复现。

</details>

### 我的结果对不对？

对于 **MATR1 + XGBoost**，已发表的基准 RMSE 是 **334 个循环**
（来源：BatteryML README「Benchmark result of RUL task」，Zhang et al., ICLR 2024）。
结果落在这个量级附近，说明你的流程接通了。

不要跨不同划分做比较——MATR1 和 MATR2 是不同的测试集；
SOH 的结果也永远不能和 RUL 的基准比较。
网页界面会替你把关：当实验协议不匹配时，它会拒绝显示基准值。

---

## 9. 可选：LLM 助手

**如果你只想要预测结果，可以完全跳过本节。** 上面所有内容都不依赖它。

助手用于解释配置、诊断报错、用自然语言解读结果。它不计算任何数字。

1. 在左侧栏打开 **Settings**。
2. 填入任意 OpenAI 兼容服务的 **Base URL**、**Model** 和 **API key**。
3. 先点 **Save**（保存），再点 **Connect**（测试连接）。

**关于你的 API key：**

- 它**只保存在当前浏览器会话中**。
- 它不会被写入任何仓库文件、配置、运行清单、日志、导出包或报错信息。
- 会话结束即消失，下次需要重新输入。

**永远不要**把 API key 粘贴进任何会被 git 提交的文件里。

---

## 10. 结果在哪里

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

**可复现性：** 相同配置 + 相同 seed + 相同数据指纹，
会产生完全一致的 `metrics.csv`。
如果某个模型本身具有不确定性，这一事实会被**如实记录**在
`run_metadata.json` 中，而不是被掩盖。

---

## 11. 常见问题排查

| 你看到的报错 | 含义 | 解决办法 |
| --- | --- | --- |
| `ModuleNotFoundError: ... 'batteryml' is not a package` | 你直接运行了 `python bin/batteryml.py` | 改用 `python batteryml_web/core/run_batteryml_cli.py ...` |
| `error: Could not import extension model module ... cptransformer` | 旧版本的代码；现在这不会再阻断 CLI | `git pull`；CPTransformer 是可选的，会被自动跳过 |
| `unrecognized arguments: --config` | `config` 是位置参数 | 写 `run configs/....yaml`，不要写 `run --config ...` |
| 运行后只打印 workspace 路径就退出了 | 缺少 `--train --eval` | 补上这两个参数 |
| `AssertionError: data/processed/MATR` | 没有执行预处理 | 执行 §7。这**不是**网页层的 bug |
| 网页显示「No datasets are available yet」 | 预处理之后没有重新生成清单 | 运行 `python tools/sync_data_and_config_manifests.py`，再点 **Refresh datasets** |
| `[FAIL] ENV-01 ... not installed: fire` | 声明的依赖没装上 | `pip install fire`，或重新执行 `pip install -r requirements.txt` |
| 远程服务器上浏览器打不开 | Streamlit 只绑定了 localhost | 用 SSH 端口转发，或加 `--server.address 0.0.0.0` |
| `CUDA out of memory` | 模型超出显存 | 加 `--device cpu`。本示例不需要 GPU |

还是卡住？运行 `python -m scripts.verify.run_all`，
把输出和 `runs/verify/verification_report.json` 一起提 issue。
那份报告会指明是哪一项检查失败以及对应的修正动作。

---

## 12. 本指南的验证情况

这里把话说清楚，因为一份夸大自己测试范围的教程只会浪费你的时间。

**在干净的 Linux 容器（Python 3.11）里真实运行并验证过：**

- `pip install -r requirements.txt` 以及 §4 的导入自检
- 在没有 `third_party/` 目录的克隆上运行
  `python batteryml_web/core/run_batteryml_cli.py run --help`——可正常启动并打印用法
- `python -m scripts.verify.run_all`（§5）：全新克隆、无数据时 13 项中有 10 项通过
- 运行 `python tools/sync_data_and_config_manifests.py`——
  生成了 15 条数据集记录和 180 条模型配置记录（§7）
- `python -m streamlit run batteryml_web/app.py`，
  并在真实浏览器中渲染了 §8 提到的每一个页面
- 仓库自带的测试套件：1340 个测试，0 失败

**未在此环境执行、依据代码与项目记录写出的部分：**

- MATR 的下载（§6）与预处理（§7）。本环境没有下载任何数据集。
  文件名和「180」这个数量来自数据集登记与项目记录，不是实跑得到的。
- 完整的训练/评估流程及其产生的 `metrics.csv`。
  334 个循环是**已发表的基准值**，不是这里实测出来的。
- §1 的所有时间估计。
- Windows 与 macOS。所有命令只在 Linux 上验证过。

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
| [`docs/AGENT_WORKFLOW.md`](./docs/AGENT_WORKFLOW.md) | 助手的实现方式 |
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
