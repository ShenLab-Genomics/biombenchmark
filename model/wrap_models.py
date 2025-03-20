import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ResBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        dilation=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
    ):
        super(ResBlock, self).__init__()
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_planes, out_planes,
                                kernel_size=3, padding=dilation, bias=False)

        if stride > 1 or out_planes != in_planes:
            self.downsample = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1,
                           stride=stride, bias=False),
                norm_layer(out_planes),
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


def create_1dcnn_for_emd(in_planes, out_planes):
    main_planes = 64
    dropout = 0.2
    emb_cnn = nn.Sequential(
        nn.Conv1d(in_planes, main_planes, kernel_size=3, padding=1),
        ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                 norm_layer=nn.BatchNorm1d),
        ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                 norm_layer=nn.BatchNorm1d),
        ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                 norm_layer=nn.BatchNorm1d),
        ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                 norm_layer=nn.BatchNorm1d),
        ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                 norm_layer=nn.BatchNorm1d),
        ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                 norm_layer=nn.BatchNorm1d),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(main_planes * 1, out_planes),
    )
    return emb_cnn


class RNAFmWarp(nn.Module):
    def __init__(self, bert, hidden_size=640, freeze_backbone=False):
        super().__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)


class RNAFmForSeqCls(RNAFmWarp):
    def __init__(self, bert, hidden_size=640, class_num=13, freeze_backbone=False):
        super().__init__(bert, hidden_size, freeze_backbone)
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids, return_embedding=False):
        # RNAFM use 1 as pad token, the attention mask is calculated automatically
        output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, 0, :]
        if return_embedding:
            return representations
        logits = self.classifier(representations)
        return logits


class RNAFmForTokenCls(RNAFmWarp):
    def __init__(self, bert, hidden_size=640, num_labels=3, freeze_backbone=False):
        super().__init__(bert, hidden_size, freeze_backbone)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pad = 6  # 511 -> 500

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, :, :]
        logits = self.classifier(representations)
        logits = logits[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits


class RNAFmForReg(RNAFmWarp):
    def __init__(self, bert, hidden_size=640, freeze_backbone=False):
        super().__init__(bert, hidden_size, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(640, 1)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        output = self.bert(input_ids, need_head_weights=False, repr_layers=[
            12], return_contacts=False)  # TODO:检查need_head_weights
        representations = output["representations"][12][:, :, :]
        representations = representations.transpose(1, 2)
        logits = self.predictor(representations).squeeze(-1)
        return logits

# -----------------

class RNABertWarp(nn.Module):
    def __init__(self, bert, freeze_backbone=False):
        super().__init__()
        self.bert = bert

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward_bert(self, input_ids):
        return self.bert(input_ids, attention_mask=input_ids > 0)

class RNABertForSeqCls(RNABertWarp):
    def __init__(self, bert, hidden_size=120, class_num=13):
        super().__init__(bert)
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids, return_embedding=False):
        _, pooled_output = self.forward_bert(input_ids)
        if return_embedding:
            return pooled_output
        logits = self.classifier(pooled_output)
        return logits

class RNABERTForTokenCls(RNABertWarp):
    def __init__(self, bert, hidden_size=120, num_labels=3):
        super().__init__(bert)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pad = (512 - 500) // 2

    def forward(self, input_ids):
        output, _ = self.forward_bert(input_ids)
        output = self.classifier(output[-1])
        logits = output[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits

class RNABERTForReg(RNABertWarp):
    def __init__(self, bert, hidden_size=120):
        super().__init__(bert)
        self.predictor = create_1dcnn_for_emd(hidden_size, 1)

    def forward(self, input_ids):
        _, pooled_output = self.forward_bert(input_ids)
        logits = self.predictor(pooled_output.transpose(1, 2)).squeeze(-1)
        return logits
# -----------------


class DNABERTWarp(nn.Module):
    def __init__(self, model, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            raise NotImplementedError


class DNABERTForSeqCls(DNABERTWarp):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        return logits


class DNABERTForTokenCls(DNABERTWarp):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.pad = 6  # 512 -> 500

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]  # drop the first [CLS]
        logits = self.model(input_ids).logits
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        return logits


class DNABERTForReg(DNABERTWarp):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(self.model.config.hidden_size, 1)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=input_ids > 0)[
                'last_hidden_state']
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
        return logits

# -----------------


class SpliceBERTForTokenCls(DNABERTForTokenCls):
    pass

class SpliceBERTForSeqCls(DNABERTForSeqCls):
    pass

class SpliceBERTForReg(DNABERTForReg):
    pass

# -----------------



# -----------------


class GENAWarp(nn.Module):
    def __init__(self, model, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            raise NotImplementedError


class GENAForSeqCls(GENAWarp):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        return logits


class GENAForTokenCls(GENAWarp):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.pad = 6  # 512 -> 500

    def forward(self, input_ids):
        logits = self.model(input_ids).logits
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        return logits


class GENAForReg(GENAWarp):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(self.model.config.hidden_size, 1)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=input_ids > 0)[
                'last_hidden_state']
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
        return logits
