# 导入必要的库
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model_and_tokenizer(model_path, tokenizer_name='allenai/scibert_scivocab_uncased'):
    """
    加载微调后的模型和分词器。
    """
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # 加载微调后的模型
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    return model, tokenizer

def predict(texts, model, tokenizer):
    """
    对给定的文本列表进行预测。
    """
    # 使用分词器准备数据
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # 确保模型处于评估模式
    model.eval()

    # 使用不计算梯度的方式进行预测，以节省内存和计算资源
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # 获取预测结果
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    return predictions

def main():
    # 微调模型的路径
    model_path = 'path/to/your/fine-tuned/model'

    # 示例文本数据
    texts = ["This is a science text.", "This is another text related to biology."]

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path)

    # 进行预测
    predictions = predict(texts, model, tokenizer)

    # 打印预测结果
    for text, pred in zip(texts, predictions):
        print(f"Text: '{text}' - Predicted Label: {pred.item()}")

if __name__ == "__main__":
    main()
