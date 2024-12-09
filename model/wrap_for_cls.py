import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


class RNABertForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=120, class_num=13):
        super(RNABertForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, class_num)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids, return_embedding=False):
        _, pooled_output = self.bert(input_ids, attention_mask=input_ids > 0)
        if return_embedding:
            return pooled_output
        logits = self.classifier(pooled_output)
        return logits


class RNAMsmForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=768, class_num=13):
        super(RNAMsmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, class_num)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        # MSM use 1 as pad token, the attention mask is calculated automatically
        output = self.bert(input_ids, repr_layers=[10])
        representations = output["representations"][10][:, 0, 0, :]
        logits = self.classifier(representations)
        return logits


class RNAFmForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=640, class_num=13):
        super(RNAFmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, class_num)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    def forward(self, input_ids, return_embedding=False):
        # RNAFM use 1 as pad token, the attention mask is calculated automatically
        output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, 0, :]
        if return_embedding:
            return representations
        logits = self.classifier(representations)
        return logits


class DNABERTForSeqCls(nn.Module):
    def __init__(self, model):
        super(DNABERTForSeqCls, self).__init__()
        self.model = model

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        return logits


class RNAErnieForSeqCls(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        return logits


class DNABERT2ForSeqCls(nn.Module):
    def __init__(self, model):
        super(DNABERT2ForSeqCls, self).__init__()
        self.model = model

    def forward(self, input_ids):
        # DNABERT2 use 3 as pad token
        logits = self.model(input_ids, attention_mask=input_ids != 3).logits
        return logits


class SeqClsLoss(nn.Module):

    def __init__(self):
        super(SeqClsLoss, self).__init__()

    def forward(self, outputs, labels):
        # convert labels to int64
        loss = F.cross_entropy(outputs, labels)
        return loss


class NTForSeqCls(nn.Module):

    def __init__(self, model):
        super(NTForSeqCls, self).__init__()
        self.model = model

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 1).logits
        return logits
