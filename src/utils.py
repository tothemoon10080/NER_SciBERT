# tokenizer的初始化和特殊标记的定义

# 特殊标记和标签
SENTENCE_SEP = "[SEP]"
DOCUMENT_SEP = "[DOC_SEP]"  # 表示文档的分割
SENTENCE_SEP_LABEL = "O"  # 可能需要根据标注方案进行调整
DOCUMENT_SEP_LABEL = "O"

# 标签到ID的映射
label_to_id = {
    'O': 0,
    'B-MIN': 1,
    'I-MIN': 2,
    'B-REA': 3,
    'I-REA': 4,
    'B-EQP': 5,
    'I-EQP': 6,
    'B-PRS': 7,
    'I-PRS': 8,
    'B-PAP': 9,
    'I-PAP': 10,
    'B-OCH': 11,
    'I-OCH': 12,
}