import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score

from pathlib import Path
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight

current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.append(str(project_dir / 'src'))

from data.NER_preprocess import process_dataset
from models.model import SciBertBiLSTMCRF
from data.dataset import NERDataset
#代理cmd命令行
# set http_proxy=http://127.0.0.1:10809
# set https_proxy=http://127.0.0.1:10809
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

# bert_model = os.path.join('models','saved_bert_model')
bert_model='allenai/scibert_scivocab_uncased'

# dataset_path = os.path.join('data','raw','dataset.txt')
dataset_path = os.path.join('data','raw','dataset_v2.txt')
max_len = 64
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# 读取数据集
input_ids, tag_ids, attention_masks = process_dataset(dataset_path, tokenizer, label_map,max_length=max_len)
dataset = NERDataset(input_ids, tag_ids, attention_masks)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 实例化模型
model = SciBertBiLSTMCRF(bert_model, hidden_dim=128, num_tags=len(label_map))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
epochs = 5
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_accuracy = 0
    nb_train_steps = 0
    train_f1 = 0

    # 使用 tqdm 创建进度条
    progress_bar = tqdm(train_loader, desc="Epoch {:1d}".format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        # 直接从批次数据字典中提取并移动到设备
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['tag_ids'].to(device)
        model.zero_grad()

        # 在这里，model 会返回损失
        loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = loss.mean()  # 计算批次损失的平均值
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # 更新进度条
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    # 平均训练损失
    avg_train_loss = total_loss / len(train_loader)
    
    # 验证阶段
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    for batch in val_loader:
        # 直接从批次中提取张量并将它们移动到指定设备
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['tag_ids'].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_steps
    pred_tags = [p_i for p, l in zip(predictions, true_labels)
                    for p_i, l_i in zip(p, l) if l_i != -100]
    valid_tags = [l_i for l in true_labels
                    for l_i in l if l_i != -100]
    f1 = f1_score(valid_tags, pred_tags)

    print("Validation Accuracy: {}".format(eval_accuracy))
    print("Validation F1-Score: {}".format(f1))