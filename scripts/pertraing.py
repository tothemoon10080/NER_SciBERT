import os
import sys
from pathlib import Path
from transformers import TFBertForMaskedLM, BertTokenizer
import tensorflow as tf
from tqdm import tqdm


current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.append(str(project_dir / 'src'))

from data.MLM_preprocess import creat_mlm_dataset

def pertraining(data_dir,model_name = 'allenai/scibert_scivocab_uncased',save_model=False):
    """预训练BERT模型"""

    if not os.path.exists(data_dir):
        raise ValueError(f"{data_dir} 不存在，请检查后重试！")

    # model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForMaskedLM.from_pretrained(model_name, from_pt=True)

    # 示例数据集格式
    
    # input_ids_tf, attention_masks_tf, labels_tf 是预处理好的TensorFlow tensors

    input_ids_tf, attention_masks_tf, labels_tf = creat_mlm_dataset(data_dir, max_length=256, tokenizer=tokenizer)

    train_data = tf.data.Dataset.from_tensor_slices(({
        'input_ids': input_ids_tf, 
        'attention_mask': attention_masks_tf}, 
        labels_tf))

    train_data = train_data.shuffle(buffer_size=1024).batch(8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    epochs = 3

    for epoch in range(epochs):
        print(f"开始 Epoch {epoch + 1}")
        metric.reset_states()  # 重置指标状态
        # 初始化tqdm进度条并设置总步数
        progress_bar = tqdm(enumerate(train_data), total=len(train_data))

        for step, batch in progress_bar:
            input_data, labels = batch
            with tf.GradientTape() as tape:
                # 前向传播
                outputs = model(input_data, labels=labels)
                loss = outputs.loss

            # 反向传播和优化
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 更新评估指标和进度条
            metric.update_state(labels, outputs.logits)
            acc = metric.result().numpy()
            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.numpy()}, Accuracy: {acc}")
            
        # 打印每个epoch结束后的精度
        acc = metric.result().numpy()
        print(f"Epoch {epoch + 1} 结束，Accuracy: {acc}")

    if save_model:
        if not os.path.exists('models'):
            os.mkdir('models')
        model.save_pretrained('models/saved_bert_model')

data_dir=os.path.join('data', 'raw','Bodyset')
pertraining(data_dir,'allenai/scibert_scivocab_uncased',True)