import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DNABERTForSeqCls(nn.Module):
    def __init__(self, model):
        super(DNABERTForSeqCls, self).__init__()
        self.model = model

    def forward(self, input_ids):
        mask = torch.ones_like(input_ids).to(input_ids.device)
        logits = self.model(input_ids, attention_mask=mask).logits
        return logits
