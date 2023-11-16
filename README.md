NER_SciBERT/
│
├── data/                   # 数据文件夹
│   ├── raw/                # 原始数据
│   └── processed/          # 预处理后的数据
│
├── src/                    # 源代码文件夹
│   ├── models/             # 模型定义
│   ├── data/               # 数据加载和预处理代码
│   └── utils/              # 辅助功能代码
│
├── scripts/                # 脚本文件夹
│   ├──                     # 预训练模型的脚本
│   ├── train.py            # 训练模型的脚本
│   ├── predict.py          # 进行预测的脚本
│   └── evaluate.py         # 评估模型性能的脚本
│
├── models/                 # 训练后的模型和模型权重
│
├── logs/                   # 日志文件夹
│
├── requirements.txt        # 项目依赖项
│
├── setup.py                # 安装脚本
│
└── README.md               # 项目的README文件
