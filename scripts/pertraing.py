"""
Author: Jiahao Xu
Date: 2024-2-6
Description: This Python script is designed to train a Masked Language Model (MLM) using the transformers library. 
             It includes functionality for loading and tokenizing text data from multiple text files, preparing the data 
             for MLM training, and training a BERT-like model on the processed data. It supports processing long text 
             documents by splitting them into smaller segments to fit the model's maximum input size, and padding shorter 
             segments as necessary.
"""
import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_paths, max_length=512):
        self.tokenizer = tokenizer
        self.file_paths = file_paths
        self.max_length = max_length
        self.samples = self.load_samples()

    def load_samples(self):
        samples = []
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                tokenized = self.tokenizer.encode_plus(
                    text, 
                    max_length=self.max_length, 
                    truncation=True, 
                    return_overflowing_tokens=True, 
                    return_tensors="pt"
                )
                samples.append({
                    "input_ids": tokenized["input_ids"][0],
                    "attention_mask": tokenized["attention_mask"][0]
                })

                # 处理可能连续溢出的令牌
                overflow_tokens = tokenized.get("overflowing_tokens")
                if overflow_tokens is not None:
                    overflow_tokens = overflow_tokens.squeeze().tolist()  # 确保是一维列表

                while overflow_tokens:
                    # 确保 overflow_tokens 是一个列表
                    if not isinstance(overflow_tokens, list):
                        overflow_tokens = [overflow_tokens]  # 将单个整数转换为列表

                    # 如果溢出令牌的长度小于15，则跳过处理
                    if len(overflow_tokens) < 15:
                        break  # 跳出循环

                    # 对当前批次的溢出令牌进行编码
                    tokenized_overflow = self.tokenizer.encode_plus(
                        overflow_tokens,
                        max_length=self.max_length,
                        truncation=True,
                        return_overflowing_tokens=True,
                        padding='max_length',
                        return_tensors="pt"
                    )
                    text1 = self.tokenizer.decode(tokenized_overflow["input_ids"][0])

                    samples.append({
                        "input_ids": tokenized_overflow["input_ids"][0],
                        "attention_mask": tokenized_overflow["attention_mask"][0]
                    })
                        

                    # 检查是否有新的溢出令牌
                    if "overflowing_tokens" in tokenized_overflow:
                        overflow_tokens = tokenized_overflow["overflowing_tokens"].squeeze().tolist()
                    else:
                        overflow_tokens = None  # 没有更多溢出令牌，结束循环

        return samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]




def load_data_as_dataset(data_dir, tokenizer):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    return TextDataset(tokenizer, file_paths)

def prepare_dataloader(dataset, tokenizer):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=data_collator)
    return dataloader


def train_mlm_model(data_dir, model_name='allenai/scibert_scivocab_uncased', max_length=512):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    
    dataset = load_data_as_dataset(data_dir, tokenizer)
    
    # 使用 prepare_dataloader 函数来准备数据加载器
    train_dataloader = prepare_dataloader(dataset, tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(3):  # 假设我们训练3个epoch
        total_loss = 0  # 初始化总损失
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        for step, batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)  # 从batch中获取attention_mask
            labels = batch['labels'].to(device)  # 注意：只有在进行有监督学习时才有labels

            optimizer.zero_grad()
            
            # 在模型的前向传播中传入attention_mask
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()  # 累加损失
            
            # 在进度条旁边更新损失信息
            progress_bar.set_postfix({'Loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)  # 计算平均损失
        progress_bar.set_description(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        progress_bar.close()  # 关闭当前进度条


    return model

data_dir = 'C:/Users/Jiahao/Desktop/BKY/ResearchPaperAutomator/data/Copper/txtPaper'
save_dir = 'C:/Users/Jiahao/Desktop/BKY/models'
model = train_mlm_model(data_dir)
print('Training finished! Do you want to save the model?')
save = input()
if save == 'yes':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    print('Model saved!')
else:
    print('Model not saved!')

