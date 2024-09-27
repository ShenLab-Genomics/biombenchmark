import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DNABERTForSeqCls(nn.Module):
    def __init__(self, model):
        super(DNABERTForSeqCls, self).__init__()
        self.model = model

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]  # drop the first [CLS] token
        # mask = torch.ones_like(input_ids).to(input_ids.device)
        logits = self.model(input_ids).logits
        return logits


class RNAErnieForSeqCls(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        attn_mask = input_ids != 0
        input_ids = input_ids[:, :]
        logits = self.model(input_ids).logits
        # logits = self.classifier(logits)
        return logits


class DNABERT2ForSeqCls(nn.Module):
    def __init__(self, model):
        super(DNABERT2ForSeqCls, self).__init__()
        self.model = model

    def forward(self, input_ids):
        # print(input_ids, input_ids.shape)
        logits = self.model(input_ids).logits
        return logits


class RNABertForM6ACls(nn.Module):
    def __init__(self, bert, hidden_size=120, class_num=13):
        super(RNABertForM6ACls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, class_num)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        # print(input_ids, input_ids.shape)
        _, pooled_output = self.bert(input_ids)
        logits = self.classifier(pooled_output)
        return logits
