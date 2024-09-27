import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RNAFmForTokenCls(nn.Module):
    def __init__(self, bert, hidden_size=640, num_labels=3):
        super(RNAFmForTokenCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pad = 6  # 511 -> 500

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, :, :]
        logits = self.classifier(representations)
        logits = logits[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits


class RNAMsmForTokenCls(nn.Module):
    def __init__(self, bert, hidden_size=768, num_labels=3):
        super(RNAMsmForTokenCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pad = 6  # 511 -> 500

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[10])
        representations = output["representations"][10][:, 0, :, :]
        logits = self.classifier(representations)
        logits = logits[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits


class SpliceBERTForTokenCls(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pad = 6  # 512 -> 500

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]  # drop the first [CLS]
        logits = self.model(input_ids).logits
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        # print(logits.shape)
        return logits


class DNABERTForTokenCls(SpliceBERTForTokenCls):
    pass


class RNAErnieForTokenCls(nn.Module):
    def __init__(self, model, hidden_size=768, num_labels=3) -> None:
        super().__init__()
        self.model = model
        self.pad = (512 - 500) // 2

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]  # drop the first [CLS]
        # print(input_ids.shape)
        # print(input_ids)
        logits = self.model(input_ids).logits
        # print(logits.shape)
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        # print(logits.shape)
        return logits


class MAMBAForTokenCls(nn.Module):
    def __init__(self, model, hidden_size=768, num_labels=15):
        super().__init__()
        self.model = model
        self.pad = (8194 - 500) // 2
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids):
        print(input_ids.shape)
        input_ids = input_ids[:, 1:]  # drop the first [CLS]
        logits = self.model(input_ids=input_ids).logits
        print(logits.shape)
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        print(logits.shape)
        logits = self.classifier(logits)
        return logits
