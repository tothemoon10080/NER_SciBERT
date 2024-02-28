<details>
<summary>中文描述</summary>

# NER_SciBERT

本项目是一个基于 SciBERT 的命名实体识别（NER）系统。它旨在提供一个高效、准确的NER模型，专门针对科学文献。

## 特点
1. 包含预训练脚本允许在专业领域论文上使用掩码语言建模(MLM)任务进一步预训练SciBert模型。
2. 从预训练任务中移除了下一句预测(NSP)，这会有效提高下流任务的性能。
3. 采用SciBERT-BiLSTM-CRF架构在NER任务中表现出色。
	
## 安装
要安装和运行 NER_SciBERT，请按照以下步骤操作：
1. 克隆仓库：`git clone https://github.com/tothemoon10080/NER_SciBERT.git`
2. 安装依赖：`pip install -r requirements.txt`

## 数据集
1. 使用了Kaggle数据集：[Mineral Processing Research](https://www.kaggle.com/datasets/tothemoon08/dataset-of-papers-in-mineral-processing
) 来预训练SciBERT模型
2. 为了创建 NER 数据集，收集了约500篇与矿物加工高度相关的英文论文，对其中约100篇论文的摘要部分进行了实体标注。此数据集可在.data文件夹下找到。

## 使用
要使用 NER_SciBERT，运行以下脚本：
- 预训练模型：`python scripts/pertraing.py`
- 微调模型：`python scripts/train.py`
- 使用模型预测：`python scripts/predict.py`

## 项目结构
<details>
<summary>项目结构</summary>

- NER_SciBERT/
  - data/
    - MLM/
    - NER/
  - src/
    - models/
      	- torchcrf/
    - data/
      - preprocess.py/
    - utils/
  - scripts/
    - train.py
    - pertraing.py
    - predict.py
  - requirements.txt
  - README.md

</details>

## 贡献
欢迎任何形式的贡献。

</details>

# NER_SciBERT
This project is a Named Entity Recognition (NER) system based on SciBERT. It aims to provide an efficient and accurate NER model specifically for scientific literature.

## Features
1. Includes a pre-training script that allows further pre-training of the SciBert model using the Masked Language Modeling (MLM) task on domain-specific papers.
2. The Next Sentence Prediction (NSP) task has been removed from the pre-training tasks, which effectively improves performance on downstream tasks.
3. Utilizes a SciBERT-BiLSTM-CRF architecture that performs excellently in NER tasks.

## Installation
To install and run NER_SciBERT, follow these steps:
1. Clone the repository: `git clone https://github.com/tothemoon10080/NER_SciBERT.git`
2. Install dependencies: `pip install -r requirements.txt`

## Dataset
1. The Kaggle dataset: [Mineral Processing Research](https://www.kaggle.com/datasets/tothemoon08/dataset-of-papers-in-mineral-processing) is used to pre-train the SciBERT model.
2. To create the NER dataset, approximately 500 English papers highly relevant to mineral processing were collected, and entity annotations were made on the abstract sections of about 100 papers. This dataset can be found in the .data folder.

## Usage
To use NER_SciBERT, run the following scripts:
- Pre-train model: `python scripts/pertraing.py`
- Fine-tune model: `python scripts/train.py`
- Use model for prediction: `python scripts/predict.py`

## Project Structure
<details>
<summary>Project Structure</summary>

- NER_SciBERT/
  - data/
    - MLM/
    - NER/
  - src/
    - models/
      	- torchcrf/
    - data/
      - preprocess.py/
    - utils/
  - scripts/
    - train.py
    - pertraing.py
    - predict.py
  - requirements.txt
  - README.md

</details>

## Contribution
Any form of contribution is welcome.

