# NER_SciBERT

本项目是一个基于 SciBERT 的命名实体识别（NER）系统。它旨在提供一个高效、准确的NER模型，专门针对科学文献。
并允许在专业领域论文上进一步预训练SciBert模型

## 安装

要安装和运行 NER_SciBERT，请按照以下步骤操作：

1. 克隆仓库：`git clone [仓库链接]`
2. 安装依赖：`pip install -r requirements.txt`
## 数据集

使用了Kaggle数据集来预训练SciBERT模型
https://www.kaggle.com/datasets/tothemoon08/dataset-of-papers-in-mineral-processing

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

