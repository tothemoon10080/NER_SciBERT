import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

def process_dataset(dataset_path, tokenizer, label_map, max_length=128):
    '''
    用于处理MAT数据集
    '''
    input_ids_list = []
    attention_masks_list = []
    tag_ids_list = []

    with open(dataset_path, 'r', encoding='utf-8') as file:
        temp_sentence = []
        temp_labels = []

        for line in file:
            if not line.strip():  # 跳过空行
                continue

            parts = line.strip().split()
            if len(parts) == 2:
                word, label = parts
                temp_sentence.append(word)
                temp_labels.append(label)

                if word.endswith('.'):
                    encoded_input = tokenizer.encode_plus(
                        temp_sentence,
                        is_split_into_words=True,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'  # 改为 'pt' 以返回 PyTorch 张量
                    )

                    # 将标签转换为列表并进行填充
                    tag_tensor = [label_map.get(l, label_map["O"]) for l in temp_labels]
                    padding_size = max_length - len(tag_tensor)
                    if padding_size > 0:
                        tag_tensor.extend([label_map["O"]] * padding_size)
                    else:
                        tag_tensor = tag_tensor[:max_length]

                    input_ids_list.append(encoded_input['input_ids'][0])
                    attention_masks_list.append(encoded_input['attention_mask'][0])
                    tag_ids_list.append(torch.tensor(tag_tensor, dtype=torch.int32))

                    temp_sentence = []
                    temp_labels = []

    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks_list)
    tag_ids = torch.stack(tag_ids_list)

    return input_ids, tag_ids, attention_masks


def process_dataset_conll(tokenizer, max_length=128):
    '''
    Process CoNLL dataset to prepare it for a model.

    Args:
    tokenizer: tokenizer to be used.
    max_length (int): Maximum length of the tokenized input.

    Returns:
    tuple: Lists of input_ids, attention_masks, and tag_ids.
    '''

    # Load dataset
    dataset = load_dataset("conll2003")

    # Initialize lists
    input_ids_list = []
    attention_masks_list = []
    tag_ids_list = []

    # Process each sentence in the dataset
    for sentence in dataset['train']:
        # Tokenize the sentence
        tokenized_input = tokenizer(sentence['tokens'], is_split_into_words=True, 
                                    max_length=max_length, truncation=True, padding='max_length')

        # Convert to tensor and append to the lists
        input_ids_tensor = torch.tensor(tokenized_input['input_ids'], dtype=torch.long)
        attention_mask_tensor = torch.tensor(tokenized_input['attention_mask'], dtype=torch.long)

        # Handle tag_ids: truncate if longer than max_length and pad if shorter
        tag_ids = sentence['ner_tags'][:max_length]  # Truncate if longer
        tag_ids += [0] * (max_length - len(tag_ids))  # Pad with 0s if shorter
        tag_ids_tensor = torch.tensor(tag_ids, dtype=torch.long)

        input_ids_list.append(input_ids_tensor)
        attention_masks_list.append(attention_mask_tensor)
        tag_ids_list.append(tag_ids_tensor)

    # Stack the tensors
    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks_list)
    tag_ids = torch.stack(tag_ids_list)

    return input_ids, tag_ids, attention_masks