import sys
import os
import tensorflow as tf

from pathlib import Path
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.append(str(project_dir / 'src'))

from models.scibertBilstmCrf import create_model
from data.NER_preprocess import process_dataset
from utils.F1Score import F1Score, F1ScoreCallback

# 标签到ID的映射
label_map  = {
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

model_path = os.path.join('models','saved_bert_model')
dataset_path = os.path.join('data','raw','Labeled_dataset.txt')
max_len = 64
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

input_ids, tag_ids, attention_masks = process_dataset(dataset_path, tokenizer, label_map,max_length=max_len)

# 将 TensorFlow 张量转换为 NumPy 数组
input_ids_np = input_ids.numpy()
attention_masks_np = attention_masks.numpy()
tag_ids_np = tag_ids.numpy()

# 使用 train_test_split 进行分割
train_inputs, validation_inputs, train_masks, validation_masks, train_tags, validation_tags = train_test_split(
    input_ids_np, attention_masks_np, tag_ids_np, random_state=42, test_size=0.1)

# 如果需要，将分割后的数据转换回 TensorFlow 张量
train_inputs = tf.convert_to_tensor(train_inputs)
validation_inputs = tf.convert_to_tensor(validation_inputs)
train_masks = tf.convert_to_tensor(train_masks)
validation_masks = tf.convert_to_tensor(validation_masks)
train_tags = tf.convert_to_tensor(train_tags)
validation_tags = tf.convert_to_tensor(validation_tags)


model_with_crf = create_model(max_len, label_map,model_path)

# 创建 F1 分数回调函数
# f1_callback = F1ScoreCallback(validation_data=([validation_inputs, validation_masks], validation_tags))
f1_metric = F1Score()

# 编译模型
model_with_crf.compile(optimizer='adam', metrics=[f1_metric])


# 训练模型时，确保传递一个列表作为模型的输入，列表中包含了input_ids和attention_masks
history = model_with_crf.fit(
    [train_inputs, train_masks],  # 传递两个输入：input_ids和attention_masks
    train_tags,
    validation_data=([validation_inputs, validation_masks], validation_tags),
    epochs=5,
    batch_size=32,
    # callbacks=[f1_callback]
)

print(history.history)
