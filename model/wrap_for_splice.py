import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForMaskedLM, AutoModelForTokenClassification


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
        self.pad = 6  # 512 -> 500

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
        # self.classifier = nn.Linear(hidden_size, num_labels)
        self.pad = (510 - 500) // 2

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:-1]  # drop the first [CLS]
        # logits = self.model(input_ids, attention_mask=input_ids > 0)[
        #     'last_hidden_state']
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        # logits = self.classifier(logits).transpose(1, 2)
        return logits


class NTForTokenCls(nn.Module):
    def __init__(self, model, num_labels=3):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.classifier = nn.Linear(1024, 6 * num_labels)
        # according to the paper, each classifier head predicts 6 continental positions
        self.pad = (9000 - 500) // 2

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        torch_outs = self.model(
            input_ids, attention_mask=input_ids > 1, output_hidden_states=True)
        logits = torch_outs['hidden_states'][-1]
        logits = self.classifier(logits)
        # print('output: ', logits.shape)

        batch = logits.shape[0]
        logits = logits.reshape(batch, -1, self.num_labels)
        # print('output: ', logits.shape)
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        return logits


class NTForTokenClsShort(nn.Module):
    def __init__(self, model, num_labels=3):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.classifier = nn.Linear(1024, 6 * num_labels)
        # according to the paper, each classifier head predicts 6 continental positions
        self.pad = (510 - 500) // 2

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        torch_outs = self.model(
            input_ids, attention_mask=input_ids > 1, output_hidden_states=True)
        logits = torch_outs['hidden_states'][-1]
        logits = self.classifier(logits)
        # print('output: ', logits.shape)

        batch = logits.shape[0]
        logits = logits.reshape(batch, -1, self.num_labels)
        # print('output: ', logits.shape)
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
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


class RNABertForTokenCls(nn.Module):
    def __init__(self, bert, hidden_size=120, num_labels=3):
        super(RNABertForTokenCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pad = (512 - 500) // 2

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        # input is 512
        # split 0 to 440, 512-440 to 512
        bar = 440
        input1 = input_ids[:, :bar]
        input2 = input_ids[:, 512-bar+1:]

        output1, pooled_output = self.bert(
            input1, attention_mask=input1 > 0)
        output1 = self.classifier(output1[-1])

        output2, pooled_output = self.bert(
            input2, attention_mask=input2 > 0)
        output2 = self.classifier(output2[-1])

        # concatenate them but discard overlapping part
        logits = torch.cat(
            [output1[:, :bar, :], output2[:, -(512-bar+1):, :]], dim=1)

        # logits = torch.cat([output1, output2], dim=1)
        # print(logits.shape)

        logits = logits[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits
