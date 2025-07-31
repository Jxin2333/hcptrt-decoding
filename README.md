# HCPTRT-Decoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述

HCPTRT-Decoding是一个用于人类连接组项目任务响应测试(Human Connectome Project Task Response Test, HCPTRT)数据的fMRI解码工具包。该项目提供了多种机器学习和深度学习方法，用于从功能性磁共振成像(fMRI)数据中解码认知状态和任务信息。

主要功能包括：

- 基准模型评估：提供多种机器学习分类器（SVM、随机森林等）用于fMRI数据解码
- 图卷积神经网络(GCN)：利用脑区连接信息进行神经活动模式解码
- 时间序列分析：针对fMRI时间序列数据的特定基准测试方法
- 数据预处理工具：用于HCPTRT数据集的预处理和准备

## 项目结构

```
├── benchmark_models/         # 基准模型实现和评估
│   ├── benchmark_decoders.py # 多种解码器实现
│   ├── beta_maps_benchmark.py # 基于beta图的基准测试
│   ├── hcptrt_data_loader.py # 数据加载工具
│   ├── hcptrt_data_prep.py   # 数据预处理工具
│   └── visualization.py      # 可视化工具
├── gcn/                      # 图卷积神经网络实现
│   ├── conectomes_generator.py # 连接组生成器
│   ├── data_concat_windows_gcn.py # GCN数据窗口连接
│   ├── gcn_model.py          # GCN模型定义
│   ├── graph_construction.py  # 图构建工具
│   └── time_windows_dataset.py # 时间窗口数据集
├── timeseries_benchmark/     # 时间序列基准测试
│   ├── benchmark_utils.py    # 基准测试工具函数
│   ├── outputs/              # 输出结果目录
│   └── tbenchmark_all.ipynb  # 基准测试笔记本
├── requirements.txt          # 项目依赖
├── utils.py                  # 通用工具函数
└── LICENSE                   # MIT许可证
```

## 文件详细说明

### 项目根目录文件

#### utils.py
通用工具函数文件，包含：
- `sum_()`: 计算数组元素总和
- `between()`: 从字符串中提取特定部分，主要用于从BIDS格式文件名中提取会话和运行编号
- `printmd()`: 以Markdown格式打印文本

#### requirements.txt
项目依赖文件，列出了所有需要安装的Python包及其版本，包括：
- dypac (动态解析组件分析)
- keras 和 tensorflow (深度学习框架)
- matplotlib, plotly, seaborn (可视化库)
- nibabel, nilearn (神经影像处理库)
- numpy, pandas, scipy (数据处理库)
- scikit-learn (机器学习库)
- torch 相关包 (PyTorch深度学习框架及其几何扩展)

### benchmark_models 目录

#### benchmark_decoders.py
实现多种解码器模型用于fMRI数据分析，包括：
- 支持向量机(SVM)分类器
- 其他机器学习分类器(逻辑回归、随机森林、KNN等)
- 神经网络模型

该文件包含数据处理函数如`_generate_all_modality_files()`和模型实现如`_grid_svm_decoder()`。

#### beta_maps_benchmark.py
基于beta图(激活模式)的基准测试实现，用于评估不同解码方法在beta图上的表现。

#### hcptrt_data_loader.py
数据加载工具，负责从HCPTRT数据集加载和预处理fMRI数据。

#### hcptrt_data_prep.py
数据预处理工具，包含用于准备HCPTRT数据的函数，如时间序列提取、标准化等。

#### sanity_check_beta_map.py
用于验证生成的beta图是否正确的检查脚本，确保数据处理过程无误。

#### visualization.py
可视化工具，用于生成fMRI数据、解码结果和模型性能的可视化图表。

### gcn 目录 (图卷积神经网络)

#### conectomes_generator.py
连接组生成器，用于创建脑区之间的连接网络，作为GCN的输入。

#### data_concat_windows_gcn.py
处理和连接时间窗口数据，为GCN模型准备输入数据。

#### gcn_model.py
图卷积神经网络模型定义，包含：
- `train_loop()`: 训练循环函数
- `valid_test_loop()`: 验证和测试循环函数
- `GCN`: 主要的GCN模型类，使用ChebConv卷积层
- `GCN_2layers_tunning`: 可调参数的两层GCN模型

#### graph_construction.py
图构建工具，用于从fMRI数据创建图结构，定义节点(脑区)和边(连接)。

#### time_windows_dataset.py
时间窗口数据集类，用于处理时间序列fMRI数据的滑动窗口。

#### test_notebooks/
包含测试和示例Jupyter笔记本：
- `gcn_package_dypac.ipynb`: 使用dypac包的GCN实现示例
- `gcn_package_final.ipynb`: 最终的GCN实现和测试

### timeseries_benchmark 目录

时间序列基准测试目录，专门用于评估基于时间序列的fMRI解码方法。该目录包含了完整的数据处理流水线、基准测试工具和多个实验笔记本。

#### benchmark_utils.py
时间序列基准测试的核心工具函数库，包含294行代码，提供以下主要功能：

**数据处理函数：**
- `_between(value, before, after)`: 从BIDS格式文件名中提取会话和运行编号等信息
- `new_conditions(datapath, event, task_label)`: 重新标记试验类型，为工作记忆(wm)和关系(relational)任务创建更清晰的解码标签
- `conditions(event_file)`: 提取每个任务的目标条件，过滤掉不需要的条件如倒计时、注视点等

**Beta图生成：**
- `_generate_beta_maps(scans, confounds, events, conditions, mask, fname, task_label)`: 使用FirstLevelModel生成beta激活图，支持多种任务类型
- `postproc_task(subject, task_label, conditions, tpl_mask)`: 外部接口函数，执行完整的beta图生成流程

**解码评估：**
- `check_decoding(subject, task_dir, task_label, tpl_mask)`: 使用支持向量机(SVM)进行解码性能评估，包括5折交叉验证和权重图可视化

**辅助功能：**
- `printmd(string)`: 以Markdown格式显示文本
- 支持Params9混淆变量校正和灰质掩膜处理

#### outputs/
输出结果目录，存储预处理后的fMRI数据和对应标签文件：

**运动任务数据：**
- `motor_final_fMRI.npy`: 运动任务的最终fMRI时间序列数据（NumPy数组格式）
- `motor_final_labels.csv`: 运动任务标签文件（751个样本），包含以下条件：
  - `response_left_hand`: 左手响应
  - `response_right_hand`: 右手响应
  - `response_left_foot`: 左脚响应
  - `response_right_foot`: 右脚响应
  - `response_tongue`: 舌头响应

**工作记忆任务数据：**
- `wm_final_fMRI.npy`: 工作记忆任务的最终fMRI时间序列数据
- `wm_final_labels.csv`: 工作记忆任务标签文件（2038个样本），包含以下条件：
  - `0-Back_Face`: 0-back面孔任务
  - `0-Back_Body`: 0-back身体任务
  - `2-Back_Face`: 2-back面孔任务
  - `2-Back_Body`: 2-back身体任务

**受试者特定数据：**
- `sub-01_motor_fMRI2.npy`: 受试者01的运动任务fMRI数据
- `sub-01_wm_fMRI2.npy`: 受试者01的工作记忆任务fMRI数据

#### tbenchmark_all.ipynb
全面的时间序列基准测试主笔记本（1007行代码），包含完整的数据处理和模型评估流程：

**主要内容：**
- 数据准备和预处理流程
- 使用Params9和Params24混淆变量校正
- NiftiLabelsMasker进行脑区提取
- 多种机器学习模型比较（SVM、神经网络、MLP）
- 交叉验证和性能评估
- 结果可视化和统计分析

**技术特点：**
- 支持多受试者分析
- 集成sklearn和keras框架
- 包含混淆矩阵和分类报告
- 使用nilearn进行神经影像处理

#### All_tasks_hcptrt_extended_benchmark_model.ipynb
扩展基准模型笔记本（482行代码），专注于HCPTRT数据集的多任务解码评估：

**覆盖任务：**
- emotion（情绪）
- language（语言）
- gambling（赌博）
- social（社交）
- relational（关系）

**主要功能：**
- 自动化beta图生成流程
- 使用nilearn Decoder对象进行解码
- 5折交叉验证评估
- 权重图可视化分析
- 支持灰质掩膜处理

#### All_tasks_hcptrt_restriced_benchmark_model.ipynb
限制版基准模型笔记本（608行代码），提供更全面的HCPTRT任务集合测试：

**覆盖任务：**
- motor（运动）
- wm（工作记忆）
- language（语言）
- gambling（赌博）
- social（社交）
- relational（关系）
- emotion（情绪）

**特色功能：**
- 完整的7个HCPTRT任务评估
- 集成seaborn进行高级可视化
- 标准化的数据处理流程
- 批量任务处理能力
- 详细的性能指标报告

## 安装指南

### 环境要求

- Python 3.6+
- CUDA支持（用于PyTorch GPU加速，可选）

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/hcptrt-decoding.git
cd hcptrt-decoding
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

注意：某些PyTorch Geometric依赖项可能需要单独安装，请参考[PyTorch Geometric安装指南](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)。

## 使用方法

### 基准模型评估

```python
from benchmark_models import benchmark_decoders

# 示例代码
# 加载和准备数据
# 运行基准测试
```

### 图卷积神经网络(GCN)模型

```python
from gcn import gcn_model

# 示例代码
# 构建图
# 训练GCN模型
# 评估模型性能
```

### 时间序列基准测试

请参考`timeseries_benchmark`目录下的Jupyter笔记本示例。

## 数据集

本项目设计用于处理人类连接组项目任务响应测试(HCPTRT)数据集。您需要单独获取该数据集的访问权限。

## 贡献指南

欢迎提交问题报告和拉取请求。对于重大更改，请先开issue讨论您想要更改的内容。

## 许可证

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 致谢

- 感谢所有为本项目做出贡献的研究人员和开发者
- 特别感谢人类连接组项目(HCP)提供的宝贵数据资源