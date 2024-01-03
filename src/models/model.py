import torch
import torch.nn as nn
from transformers import AutoModel
from models.torchcrf import CRF

class SciBertBiLSTMCRF(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_tags):
        super(SciBertBiLSTMCRF, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags) 

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        emissions = self.hidden2tag(lstm_output)
        attention_mask_transposed = attention_mask.transpose(0, 1) # 交换 batch_size 和 seq_length 的维度
        emissions_transposed = emissions.transpose(0, 1)  # 交换 batch_size 和 seq_length 的维度
        # 交换 batch_size 和 seq_length 的维度  现在Sequence of tags of size ``(seq_length, batch_size)``.


        if labels is not None:
            labels_transposed = labels.transpose(0, 1).type(torch.long)
            loss = -self.crf(emissions_transposed, labels_transposed, mask=attention_mask_transposed)
            return loss
        else:
            return self.crf.decode(emissions=emissions_transposed, mask=attention_mask_transposed)
