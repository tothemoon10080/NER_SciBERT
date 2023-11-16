import os
import re

# 设置句子分隔符和文档分隔符
SENTENCE_SEP = "[SEP]"
DOCUMENT_SEP = "[DOC_SEP]"

# 定义用于分割句子的函数
def split_sentences(text):
    # 这里简单地使用句号来分割句子。根据需要，可以使用更复杂的分割方法。
    sentences = re.split(r'(?<=[.!?]) +', text)
    # 移除空句子
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']
    return sentences

# 定义主处理函数
def preprocess_abstracts(folder_path):
    processed_texts = []  # 保存所有处理过的文本

    # 遍历文件夹内的所有.txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if text == '摘要未找到' or text == 'Unknown':
                    continue
                # 分割句子并添加句子分隔符
                sentences = split_sentences(text)
                processed_text = f" {SENTENCE_SEP} ".join(sentences) + f" {DOCUMENT_SEP}"
                processed_texts.append(processed_text)
    
    return processed_texts

# 指定论文摘要文件夹的路径
abstracts_folder_path = "data/raw/abstracts"
# 调用函数处理论文摘要，并获取处理后的数据
processed_data = preprocess_abstracts(abstracts_folder_path)

# 可以选择将处理后的数据保存到文件中，或直接用于模型预训练
# 例如，将处理后的数据保存到一个新的文件中
with open('preprocessed_abstracts.txt', 'w', encoding='utf-8') as outfile:
    outfile.write('\n'.join(processed_data))
