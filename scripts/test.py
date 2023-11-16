from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]

outputs = model(input_ids, labels=input_ids)
loss, prediction_scores = outputs[:2]
print(loss)