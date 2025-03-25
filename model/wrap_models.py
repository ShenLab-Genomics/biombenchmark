import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification


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


class RNAFmWrap(nn.Module):
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


class RNAFmForSeqCls(RNAFmWrap):
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


class RNAFmForTokenCls(RNAFmWrap):
    def __init__(self, bert, hidden_size=640, class_num=3, freeze_backbone=False):
        super().__init__(bert, hidden_size, freeze_backbone)
        self.classifier = nn.Linear(hidden_size, class_num)
        self.pad = 6  # 511 -> 500

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, :, :]
        logits = self.classifier(representations)
        logits = logits[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits


class RNAFmForReg(RNAFmWrap):
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


class RNAMsmWrap(nn.Module):
    def __init__(self, bert, hidden_size=768, freeze_backbone=False):
        super().__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward_bert(self, input_ids):
        # MSM use 1 as pad token, the attention mask is calculated automatically
        output = self.bert(input_ids, repr_layers=[10])
        return output


class RNAMsmForSeqCls(RNAMsmWrap):
    def __init__(self, bert, hidden_size=768, class_num=13, freeze_backbone=False):
        super().__init__(bert, hidden_size, freeze_backbone)
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids):
        representations = self.forward_bert(
            input_ids)["representations"][10][:, 0, 0, :]
        logits = self.classifier(representations)
        return logits


class RNAMsmForTokenCls(RNAMsmWrap):
    def __init__(self, bert, hidden_size=768, class_num=3, freeze_backbone=False):
        super().__init__(bert, hidden_size, freeze_backbone)
        self.classifier = nn.Linear(hidden_size, class_num)
        self.pad = 6  # 512 -> 500

    def forward(self, input_ids):
        representations = self.forward_bert(
            input_ids)["representations"][10][:, 0, :, :]
        logits = self.classifier(representations)
        logits = logits[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits


class RNAMsmForReg(RNAMsmWrap):
    def __init__(self, bert, hidden_size=768, class_num=1, freeze_backbone=False):
        super().__init__(bert, hidden_size, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(hidden_size, class_num)

    def forward(self, input_ids):
        representations = self.forward_bert(input_ids)["representations"][10][:, 0, :, :].transpose(
            1, 2)
        logits = self.predictor(representations).squeeze(-1)
        return logits

# -----------------


class RNABertWrap(nn.Module):
    def __init__(self, bert, freeze_backbone=False):
        super().__init__()
        self.bert = bert
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward_bert(self, input_ids):
        # RNABert uses 0 as pad token
        return self.bert(input_ids, attention_mask=input_ids > 0)


class RNABertForSeqCls(RNABertWrap):
    def __init__(self, bert, hidden_size=120, class_num=13, freeze_backbone=False):
        super().__init__(bert, freeze_backbone)
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids, return_embedding=False):
        _, pooled_output = self.forward_bert(input_ids)
        if return_embedding:
            return pooled_output
        logits = self.classifier(pooled_output)
        return logits


class RNABERTForTokenCls(RNABertWrap):
    def __init__(self, bert, hidden_size=120, class_num=3, freeze_backbone=False):
        super().__init__(bert, freeze_backbone)
        self.classifier = nn.Linear(hidden_size, class_num)
        self.pad = (512 - 500) // 2

    def forward(self, input_ids):
        output, _ = self.forward_bert(input_ids)
        output = self.classifier(output[-1])
        logits = output[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        return logits


class RNABERTForReg(RNABertWrap):
    def __init__(self, bert, hidden_size=120, class_num=1, freeze_backbone=False):
        super().__init__(bert, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(hidden_size, class_num)

    def forward(self, input_ids):
        _, pooled_output = self.forward_bert(input_ids)
        logits = self.predictor(pooled_output.transpose(1, 2)).squeeze(-1)
        return logits
# -----------------


class DNABERTWrap(nn.Module):
    def __init__(self, model, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            # Huggingface BertModel
            for param in self.model.bert.parameters():
                param.requires_grad = False


class DNABERTForSeqCls(DNABERTWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        return logits


class DNABERTForTokenCls(DNABERTWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.pad = 6  # 512 -> 500

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]  # drop the first [CLS]
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        return logits


class DNABERTForReg(DNABERTWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(self.model.config.hidden_size, 1)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
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


class DNABERT2Wrap(nn.Module):
    def __init__(self, model, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.model.bert.parameters():
                param.requires_grad = False


class DNABERT2ForSeqCls(DNABERT2Wrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)

    def forward(self, input_ids):
        # DNABERT2 use 3 as pad token
        logits = self.model(input_ids, attention_mask=input_ids != 3).logits
        return logits


class DNABERT2ForReg(DNABERT2Wrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(768, 1)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        logits = self.model(input_ids, attention_mask=input_ids != 3)[0]
        logits = self.predictor(logits.transpose(1, 2)).squeeze(-1)
        return logits

# -----------------


class RNAErnieWrap(nn.Module):
    def __init__(self, model, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.model.bert.parameters():
                param.requires_grad = False


class RNAErnieForSeqCls(RNAErnieWrap):
    def __init__(self, model, freeze_backbone=False) -> None:
        super().__init__(model, freeze_backbone)

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        return logits


class RNAErnieForTokenCls(RNAErnieWrap):
    def __init__(self, model, freeze_backbone=False) -> None:
        super().__init__(model, freeze_backbone)
        self.pad = (510 - 500) // 2

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:-1]  # drop the first [CLS]
        logits = self.model(input_ids, attention_mask=input_ids > 0).logits
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        return logits


class RNAErnieForReg(RNAErnieWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(768, 1)

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 0)[
            'last_hidden_state']
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
        return logits

# -----------------


class NucleotideTransformerWrap(nn.Module):
    def __init__(self, model, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.model.bert.parameters():
                param.requires_grad = False


class NTForSeqCls(NucleotideTransformerWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)

    def forward(self, input_ids):
        logits = self.model(input_ids, attention_mask=input_ids > 1).logits
        return logits


class NTForTokenCls(NucleotideTransformerWrap):
    def __init__(self, model, class_num=3, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.class_num = class_num
        self.classifier = nn.Linear(1024, 6 * class_num)
        # according to the paper, each classifier head predicts 6 continental positions
        self.pad = (9000 - 500) // 2

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        torch_outs = self.model(
            input_ids, attention_mask=input_ids > 1, output_hidden_states=True)
        logits = torch_outs['hidden_states'][-1]
        logits = self.classifier(logits)

        batch = logits.shape[0]
        logits = logits.reshape(batch, -1, self.class_num)
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        return logits


class NTForReg(NucleotideTransformerWrap):
    def __init__(self, model, class_num=3, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(1024, class_num)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        torch_outs = self.model(
            input_ids, attention_mask=input_ids > 1, output_hidden_states=True)
        logits = torch_outs['hidden_states'][-1]
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
        return logits

# -----------------


class GENAWrap(nn.Module):
    def __init__(self, model, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            raise NotImplementedError

    def forward_model(self, input_ids):
        return self.model(input_ids, attention_mask=input_ids != 3)


class GENAForSeqCls(GENAWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)

    def forward(self, input_ids):
        logits = self.forward_model(input_ids).logits
        return logits


class GENAForTokenCls(GENAWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.pad = 6  # 512 -> 500

    def forward(self, input_ids):
        logits = self.forward_model(input_ids).logits
        logits = logits[:, self.pad:-self.pad, :].transpose(1, 2)
        return logits


class GENAForReg(GENAWrap):
    def __init__(self, model, freeze_backbone=False):
        super().__init__(model, freeze_backbone)
        self.predictor = create_1dcnn_for_emd(self.model.config.hidden_size, 1)

    def forward(self, input_ids):
        logits = self.forward_model(input_ids).logits
        logits = self.predictor(logits.transpose(1, 2)).squeeze(-1)
        return logits


class UTRLMWrap(nn.Module):
    pass


class UTRLMForSeqCls(UTRLMWrap):
    pass


class UTRLMForTokenCls(UTRLMWrap):
    pass


class UTRLMForReg(UTRLMWrap):
    pass
