import torch

def f1_score_pytorch(y_true, y_pred):
    # 将列表转换为PyTorch张量
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # 确保张量是长整型
    y_true = y_true.long()
    y_pred = y_pred.long()

    # 计算TP、FP和FN
    tp = (y_pred * y_true).sum().to(torch.float32)
    fp = (y_pred * (1 - y_true)).sum().to(torch.float32)
    fn = ((1 - y_pred) * y_true).sum().to(torch.float32)

    # 计算精确率和召回率
    epsilon = 1e-7  # 防止除以0
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.item()  # 将结果转换为标量

# 示例使用
# 假设valid_tags和pred_tags是包含真实标签和预测标签的列表
# 例如:
# valid_tags = [0, 1, 2, 1, 0]
# pred_tags = [0, 2, 1, 0, 0]