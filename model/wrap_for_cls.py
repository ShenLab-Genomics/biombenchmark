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


class RNAErnieForTokenCls(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        input_ids = input_ids[:, :]
        logits = self.model(input_ids).logits
        return logits
