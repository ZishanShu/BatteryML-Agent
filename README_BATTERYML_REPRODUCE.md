# BatteryML 代码分析与复现说明

这份文档用于说明当前仓库中 BatteryML 的数据、训练、评估流程，并配套 `scripts/reproduce_batteryml/` 下的几个脚本，帮助你在不直接依赖 `batteryml run` 这个封装命令的情况下复现主要操作。

## 1. BatteryML 的整体流程

BatteryML 把电池寿命预测任务拆成以下步骤：

```text
原始公开数据
  -> preprocess
统一 BatteryData .pkl
  -> train_test_split
训练/测试电池列表
  -> feature extractor
早期循环特征
  -> label annotator
RUL/SOH 标签
  -> DataBundle transformation
特征标准化、标签 log + 标准化
  -> model.fit
训练
  -> model.predict
预测
  -> DataBundle.evaluate
RMSE/MAE/MAPE
```

核心代码位置：

| 模块 | 作用 |
| --- | --- |
| `bin/batteryml.py` | CLI 入口，包含 `download`、`preprocess`、`run` |
| `batteryml/pipeline.py` | 读取 config、构造数据集、训练、评估 |
| `batteryml/task.py` | 加载 `.pkl` 电池、抽特征、打标签、组装 `DataBundle` |
| `batteryml/data/battery_data.py` | `BatteryData` 与 `CycleData` 统一数据格式 |
| `batteryml/feature/` | 特征提取器，例如 `VoltageCapacityMatrixFeatureExtractor` |
| `batteryml/label/` | 标签计算，例如 `RULLabelAnnotator` |
| `batteryml/train_test_split/` | 固定 benchmark 的 train/test 划分 |
| `batteryml/models/` | sklearn、XGBoost、神经网络模型封装 |
| `batteryml/preprocess/` | 各公开数据集原始文件到 `BatteryData` 的转换 |

## 2. 新增复现脚本

本次新增了以下脚本：

```text
scripts/reproduce_batteryml/
├── check_dataset_status.py
├── prepare_data.py
├── manual_train_eval.py
└── explicit_matr_xgb.py
```

### 2.1 `check_dataset_status.py`

检查 raw data、processed data 以及 benchmark 可运行状态。

```bash
python scripts/reproduce_batteryml/check_dataset_status.py
```

也可以指定目录：

```bash
python scripts/reproduce_batteryml/check_dataset_status.py \
  --raw-root data/raw \
  --processed-root data/processed
```

输出会告诉你：

- 每个数据集原始文件是否存在；
- 每个 `data/processed/<DATASET>` 目录下有多少 `.pkl`；
- `matr_1/matr_2`、`hust`、`snl`、`cruh`、`crush`、`mix` 是否具备 processed 数据。

### 2.2 `prepare_data.py`

不通过 `batteryml preprocess` CLI，直接调用仓库里的 preprocessor 类。

#### 使用本地已有 MATR 文件

当前你的仓库里已经有 4 个 MATR 原始 `.mat` 文件。如果它们还在 `data/` 根目录，可以运行：

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets MATR \
  --bootstrap-local-matr \
  --preprocess
```

默认行为是复制：

```text
data/MATR_batch_*.mat -> data/raw/MATR/MATR_batch_*.mat
```

如果你想移动而不是复制：

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets MATR \
  --bootstrap-local-matr \
  --move-local-matr \
  --preprocess
```

处理完成后会生成：

```text
data/processed/MATR/*.pkl
```

#### 下载并处理公开直链数据集

BatteryML 代码里有直链的数据集包括：

- `MATR`
- `HUST`
- `CALCE`
- `RWTH`

例如下载并处理 HUST：

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets HUST \
  --download \
  --preprocess
```

下载并处理多个数据集：

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets MATR HUST CALCE RWTH \
  --download \
  --preprocess
```

### 2.3 `manual_train_eval.py`

这个脚本复现 `batteryml run <config> --train --eval` 的主要行为，但不通过 CLI，而是直接调用：

- `load_config()`
- `build_dataset()`
- `MODELS.build()`
- `model.fit()`
- `model.predict()`
- `dataset.evaluate()`

运行 MATR1 XGBoost：

```bash
python scripts/reproduce_batteryml/manual_train_eval.py \
  --config configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace workspace/manual_xgb_matr_1 \
  --seed 0 \
  --device cpu
```

保存训练集预测结果：

```bash
python scripts/reproduce_batteryml/manual_train_eval.py \
  --config configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace workspace/manual_xgb_matr_1 \
  --save-train-predictions
```

如果跑神经网络配置，可以覆盖 epochs：

```bash
python scripts/reproduce_batteryml/manual_train_eval.py \
  --config configs/baselines/nn_models/mlp/matr_1.yaml \
  --workspace workspace/manual_mlp_matr_1 \
  --epochs 100 \
  --device cpu
```

### 2.4 `explicit_matr_xgb.py`

这个脚本更“展开”，不使用 `Pipeline`，也不使用 registry/builder，而是显式实例化：

- `MATRPrimaryTestTrainTestSplitter`
- `VoltageCapacityMatrixFeatureExtractor`
- `RULLabelAnnotator`
- `ZScoreDataTransformation`
- `LogScaleDataTransformation`
- `SequentialDataTransformation`
- `XGBoostRULPredictor`

运行 MATR1：

```bash
python scripts/reproduce_batteryml/explicit_matr_xgb.py \
  --processed-matr-dir data/processed/MATR \
  --split matr_1 \
  --workspace workspace/explicit_matr_xgb
```

运行 MATR2：

```bash
python scripts/reproduce_batteryml/explicit_matr_xgb.py \
  --processed-matr-dir data/processed/MATR \
  --split matr_2 \
  --workspace workspace/explicit_matr_xgb_matr2
```

这个脚本最适合你学习 BatteryML 的核心操作，因为训练过程基本没有隐藏在封装里。

## 3. 数据集下载与处理

### 3.1 MATR

你当前已经有 MATR 四个原始 `.mat` 文件：

```text
MATR_batch_20170512.mat
MATR_batch_20170630.mat
MATR_batch_20180412.mat
MATR_batch_20190124.mat
```

预处理脚本期望它们位于：

```text
data/raw/MATR/
```

处理命令：

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets MATR \
  --bootstrap-local-matr \
  --preprocess
```

或重新下载：

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets MATR \
  --download \
  --preprocess
```

### 3.2 HUST

BatteryML 内置下载链接。

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets HUST \
  --download \
  --preprocess
```

原始文件要求：

```text
data/raw/HUST/hust_data.zip
```

### 3.3 CALCE

BatteryML 内置下载链接。

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets CALCE \
  --download \
  --preprocess
```

注意：CALCE 预处理会读取大量 Excel/TXT 文件，耗时可能较长。

### 3.4 RWTH

BatteryML 内置下载链接。

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets RWTH \
  --download \
  --preprocess
```

原始文件要求：

```text
data/raw/RWTH/RWTH.zip
```

### 3.5 SNL / HNEI / UL_PUR

这三个数据集来自 Battery Archive，当前仓库没有直接下载链接。需要你到 Battery Archive 申请或手动下载，然后放置到：

```text
data/raw/SNL/
data/raw/HNEI/
data/raw/UL_PUR/
```

它们通常需要成对文件：

```text
*_timeseries.csv
*_cycle_data.csv
```

处理命令：

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets SNL HNEI UL_PUR \
  --preprocess
```

## 4. 各 benchmark 需要哪些 processed 数据

| Benchmark | 需要的 processed 数据 |
| --- | --- |
| `matr_1` / `matr_2` | `data/processed/MATR` |
| `hust` | `data/processed/HUST` |
| `snl` | `data/processed/SNL` |
| `cruh` | `data/processed/CALCE`, `data/processed/RWTH`, `data/processed/UL_PUR`, `data/processed/HNEI` |
| `crush` | `data/processed/CALCE`, `data/processed/RWTH`, `data/processed/UL_PUR`, `data/processed/SNL`, `data/processed/HNEI` |
| `mix` | `data/processed/CALCE`, `data/processed/RWTH`, `data/processed/UL_PUR`, `data/processed/HNEI`, `data/processed/MATR`, `data/processed/HUST` |

可以随时用下面命令检查：

```bash
python scripts/reproduce_batteryml/check_dataset_status.py
```

## 5. 推荐复现顺序

### 第一步：安装依赖

```bash
pip install -r requirements.txt
pip install .
```

如果要跑神经网络模型，还需要按你的机器环境安装 PyTorch。

### 第二步：处理 MATR

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets MATR \
  --bootstrap-local-matr \
  --preprocess
```

检查：

```bash
python scripts/reproduce_batteryml/check_dataset_status.py
```

### 第三步：显式复现 MATR1 XGBoost

```bash
python scripts/reproduce_batteryml/explicit_matr_xgb.py \
  --processed-matr-dir data/processed/MATR \
  --split matr_1 \
  --workspace workspace/explicit_matr_xgb
```

### 第四步：用通用手动脚本复现任意 config

```bash
python scripts/reproduce_batteryml/manual_train_eval.py \
  --config configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace workspace/manual_xgb_matr_1
```

### 第五步：扩展到其他数据集

```bash
python scripts/reproduce_batteryml/prepare_data.py \
  --datasets HUST CALCE RWTH \
  --download \
  --preprocess
```

SNL/HNEI/UL_PUR 需要你手动下载后再 preprocess。

## 6. 输出文件

训练脚本会在 workspace 下保存：

```text
*.ckpt                         # 模型 checkpoint
manual_config_*.yaml           # 复制的配置文件
manual_predictions_seed_*.pkl   # 预测、分数、DataBundle 等
explicit_*_predictions.pkl      # 显式 MATR 脚本输出
```

BatteryML 自身还会在 `cache/` 下缓存 `DataBundle`，例如：

```text
cache/battery_cache_*.pkl
```

如果你修改了特征、标签、划分或 transformation，又怀疑缓存影响结果，可以删除 `cache/` 后重跑。

## 7. 和官方 `batteryml run` 的对应关系

官方命令：

```bash
batteryml run configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace workspace/xgb_matr_1 \
  --train \
  --eval
```

等价于：

```bash
python scripts/reproduce_batteryml/manual_train_eval.py \
  --config configs/baselines/sklearn/xgb/matr_1.yaml \
  --workspace workspace/manual_xgb_matr_1
```

区别是后者把训练/评估过程写在脚本里，方便你调试和改代码。

## 8. 进一步修改建议

如果你想做自己的实验，优先改这些地方：

1. 改特征：`scripts/reproduce_batteryml/explicit_matr_xgb.py` 中的 `VoltageCapacityMatrixFeatureExtractor` 参数。
2. 改标签：替换 `RULLabelAnnotator` 或调整 `eol_soh`。
3. 改模型：把 `XGBoostRULPredictor` 换成 `RandomForestRULPredictor`、`RidgeRULPredictor` 等。
4. 改划分：替换 `MATRPrimaryTestTrainTestSplitter` 或自己写 splitter。
5. 改数据集：使用 `manual_train_eval.py --config <其他 yaml>`。
