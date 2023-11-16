import os
import string
from collections import Counter
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
"""
此代码片段用于查找SciBERT词汇表中不存在的单词。

结果
[('HydroCopper™', 27), ('REFLUX™', 27), ('MAGCLA™', 25), ('BIOCOP™', 20), ('µl', 20), ('ϵ', 19), 
('µmol', 19), ('⩾0', 19), ('IsaMill™', 18), ('ϭ', 17), ('LeachWELL™', 17), ('BioLime™', 16), ('5´', 16), 
('3´', 16), ('⊙', 16), ('⩽', 15), ('‴', 15), ('µM', 14), ('V™', 14), ('Ⅰ', 14)]
"""
# 使用SciBERT的快速分词器
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)

# 定义一个更全面的标点符号和特殊字符集合
extended_punctuation = string.punctuation + "‘’“”•–—…‰℃ⅡⅣ∑∏√∫θ≤≥≠≈∞∝∈∉∂∇∅∩∪$€£¥¢₹₽₩©®™℠¼½¾³²₁₂₀§¶†‡※☆★♠♣♥♦αβγδεζηθικλμνξοπρστυφχψω→←↑↓↔↕↗↖↘↙µm″׳‑ϕɛµg¯ØµL‒∘ⅢⅥℓ"

def find_unk_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    tokenized = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenized.tokens()
    offsets = tokenized['offset_mapping']
    
    unk_words = []
    for token, offset in zip(tokens, offsets):
        if token == tokenizer.unk_token:
            start, end = offset
            unk_word = text[start:end]
            if unk_word.strip(extended_punctuation):  # 排除纯标点符号的单词
                unk_words.append(unk_word)

    return unk_words

def plot_unk_word_frequency(folder_path):
    unk_word_counter = Counter()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            unk_words = find_unk_words(file_path)
            unk_word_counter.update(unk_words)

    # 获取出现频率最高的前20个单词
    top_unk_words = unk_word_counter.most_common(20)
    print(top_unk_words)
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    words, counts = zip(*top_unk_words)  # 解包单词和频率
    plt.bar(words, counts)
    plt.xlabel('UNK Words')
    plt.ylabel('Frequency')
    plt.title('Top 20 UNK Words Frequency')
    plt.xticks(rotation=45)
    plt.show()

# 指定文件夹路径
folder_path = 'data/raw/Bodyset'
plot_unk_word_frequency(folder_path)
