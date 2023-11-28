import tensorflow as tf

def process_dataset(dataset_path, tokenizer, label_map, max_length=128):
    input_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    attention_masks = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    tag_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    index = 0

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
                        return_tensors='tf'
                    )

                    # 将标签转换为张量并进行填充
                    tag_tensor = tf.convert_to_tensor([label_map.get(l, label_map["O"]) for l in temp_labels], dtype=tf.int32)
                    padding_size = max_length - tf.shape(tag_tensor)[0]
                    if padding_size > 0:
                        tag_tensor = tf.pad(tag_tensor, [[0, padding_size]], constant_values=label_map["O"])
                    else:
                        tag_tensor = tag_tensor[:max_length]

                    input_ids = input_ids.write(index, encoded_input['input_ids'][0])
                    attention_masks = attention_masks.write(index, encoded_input['attention_mask'][0])
                    tag_ids = tag_ids.write(index, tag_tensor)

                    index += 1
                    temp_sentence = []
                    temp_labels = []

    input_ids = input_ids.stack()
    attention_masks = attention_masks.stack()
    tag_ids = tag_ids.stack()

    return input_ids, tag_ids, attention_masks
