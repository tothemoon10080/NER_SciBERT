import re
import os
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
import spacy
from spacy.lang.en import English

SENTENCE_SEP = "[SEP]"
DOCUMENT_SEP = "[DOC_SEP]"  # 表示文档的分割

# tokenizer的初始化和特殊标记的定义
    
def clean_text(text):
    """该函数通过移除电子邮件地址、引用编号以及特殊字符来清理文本。"""
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub(r'\[[^]]*\]', '', text)  # Remove citations
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()

def preprocess_abstracts(folder_path, max_length):
    """
    对指定文件夹内的所有文本文件进行处理，将每个文件中的文本分割成最大长度不超过 max_length 的片段。

    该函数首先使用 SpaCy 的句子分割器将文本分割成句子，然后尝试将句子合并成最大长度不超过 max_length 的片段。
    如果添加下一个句子会导致超过最大长度限制，它会将当前片段存储并开始新的片段。这种方法确保每个片段都在句子边界处结束，
    避免在句子中间截断文本。请注意，如果单个句子的长度超过了 max_length，这个句子会单独成为一个片段。

    参数:
    - folder_path: 包含文本文件的文件夹路径，函数将处理该文件夹下所有的 .txt 文件。
    - max_length: 每个片段的最大字符长度。

    返回:
    - processed_texts: 包含处理过的文本片段的列表。每个片段都不超过指定的最大长度，并且在句子边界处结束。

    注意：该函数使用的 SpaCy 模型需要事先加载，对于非英文文本，需要使用相应的语言模型。
    """
    nlp = English()
    nlp.add_pipe('sentencizer')
    processed_texts = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if text == '' or text == 'Unknown':
                    continue

                doc = nlp(text)
                current_chunk = ""
                current_length = 0  # 当前片段的单词数

                for sentence in doc.sents:
                    sentence_text = sentence.text.strip()
                    sentence_length = len(sentence_text.split())  # 句子的单词数

                    if current_length + sentence_length > max_length and current_chunk:
                        processed_texts.append(current_chunk)
                        current_chunk = sentence_text
                        current_length = sentence_length
                    else:
                        current_chunk += (" " + sentence_text if current_chunk else sentence_text)
                        current_length += sentence_length

                if current_chunk:  # 添加最后一个片段
                    processed_texts.append(current_chunk)

    return processed_texts

def mask_dataset(input_ids_np, attention_masks_np, tokenizer, mask_probability=0.15):
    """
    对给定的数据集应用掩码处理，模拟BERT的预训练过程中的掩码语言模型（MLM）任务。

    该函数通过随机选择一定比例的非特殊令牌并将其替换为[MASK]令牌，从而为BERT模型的MLM任务准备数据。
    特殊令牌（如[CLS]、[SEP]和[PAD]）不会被掩码。同时，函数还会创建一个标签数组，对于被掩码的令牌，
    其标签为原始的令牌ID，否则为-100（BERT模型中用于忽略的标签值）。

    参数:
    - input_ids_np: 包含令牌ID的NumPy数组。
    - attention_masks_np: 包含注意力掩码的NumPy数组，指示哪些令牌是有效的。
    - tokenizer: 使用的分词器实例，用于转换特殊令牌。
    - mask_probability: 每个令牌被替换为[MASK]的概率，默认为0.15。

    返回:
    - input_ids_tf: 包含更新后的令牌ID，一些令牌被替换为[MASK]的TensorFlow张量。
    - attention_masks_tf: 原始注意力掩码的TensorFlow张量。
    - labels_tf: 用于MLM任务的标签的TensorFlow张量，被掩码的令牌对应原始ID，其他为-100。

    该函数还会创建一个TensorFlow数据集（Dataset），包含用于模型训练的输入和标签。
    """
        
    labels_np = np.copy(input_ids_np)

    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

    for i in range(input_ids_np.shape[0]):
        probability_matrix = np.full(labels_np[i].shape, mask_probability)
        probability_matrix[input_ids_np[i] == tokenizer.pad_token_id] = 0
        masked_indices = np.random.rand(*labels_np[i].shape) < probability_matrix

        non_special_tokens_mask = np.isin(input_ids_np[i], special_tokens, invert=True)
        masked_indices &= non_special_tokens_mask

        input_ids_np[i, masked_indices] = mask_token_id
        labels_np[i, ~masked_indices] = -100

    # 转换为TensorFlow tensors
    input_ids_tf = tf.constant(input_ids_np)
    attention_masks_tf = tf.constant(attention_masks_np)
    labels_tf = tf.constant(labels_np)

    # 创建TensorFlow Dataset
    train_data = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_tf, 'attention_mask': attention_masks_tf}, labels_tf))

    return input_ids_tf, attention_masks_tf, labels_tf

#创建MLM数据集
def creat_mlm_dataset(data_dir, max_length=256, tokenizer=BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')):

    """
    创建用于BERT模型掩码语言模型（MLM）任务的数据集。

    该函数首先处理指定文件夹中的所有文本文件，将文本分割成不超过max_length的片段。然后，使用指定的分词器对这些文本片段进行编码，
    包括将文本转换为令牌ID，并生成相应的注意力掩码。编码后的文本和注意力掩码被转换为NumPy数组，以便进一步处理。

    最后，这些编码后的数据被传递给mask_dataset函数来生成最终的MLM任务数据集，包括输入ID、注意力掩码和MLM任务的标签。

    参数:
    - max_length: 每个文本片段的最大长度。默认值为256。
    - data_dir: 包含文本文件的文件夹路径。默认为'data/raw/abstracts'。
    - tokenizer: 用于编码文本的分词器。默认为SciBERT分词器。

    返回:
    - input_ids_tf: 用于MLM任务的输入令牌ID的TensorFlow张量。
    - attention_masks_tf: 对应的注意力掩码的TensorFlow张量。
    - labels_tf: MLM任务的标签的TensorFlow张量。

    注意：确保传递的分词器与用于最终模型训练的预训练模型兼容。
    """

    texts = preprocess_abstracts(data_dir,max_length)
    tokenized_texts = [tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True) for text in texts]

    # 将其转换为NumPy数组
    input_ids_np = np.array([tokenized['input_ids'] for tokenized in tokenized_texts])
    attention_masks_np = np.array([tokenized['attention_mask'] for tokenized in tokenized_texts])
    
    return mask_dataset(input_ids_np, attention_masks_np, tokenizer)