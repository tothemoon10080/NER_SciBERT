# NER_SciBERT
本项目是一个基于 SciBERT 的命名实体识别（NER）系统。它旨在提供一个高效、准确的NER模型，专门针对科学文献。

## 特点
1. 包含预训练脚本允许在专业领域论文上使用掩码语言建模(MLM)任务进一步预训练SciBert模型。
2. 从预训练任务中移除了下一句预测(NSP)，这会有效提高下流任务的性能。
3. 采用SciBERT-BiLSTM-CRF架构在NER任务中表现出色。

## 安装

要安装和运行 NER_SciBERT，请按照以下步骤操作：

1. 克隆仓库：`git clone [仓库链接](https://github.com/tothemoon10080/NER_SciBERT.git)`
2. 安装依赖：`pip install -r requirements.txt`

## 数据集
1. 使用了Kaggle数据集：[Mineral Processing Research](https://www.kaggle.com/datasets/tothemoon08/dataset-of-papers-in-mineral-processing
) 来预训练SciBERT模型
2. 为了创建 NER 数据集，我收集了约500篇与矿物加工高度相关的英文论文，对其中约100篇论文的摘要部分进行了实体标注。此数据集可在.data文件夹下找到。

## 使用

要使用 NER_SciBERT，运行以下脚本之一：
- 预训练：`python scripts/pretrain.py`
- 训练模型：`python scripts/train.py`
- 进行预测：`python scripts/predict.py`
- 评估模型：`python scripts/evaluate.py`

## 项目结构

<details>
<summary>项目结构</summary>

- NER_SciBERT/
  - data/
    - raw/
    - processed/
  - src/
    - models/
    - data/
    - utils/
  - scripts/
    - train.py
    - predict.py
    - evaluate.py
  - models/
  - logs/
  - requirements.txt
  - setup.py
  - README.md

</details>

## 贡献

欢迎任何形式的贡献，无论是新功能、文档还是问题修复。请先提交问题或拉取请求。

