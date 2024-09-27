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


class RNAFmForReg(nn.Module):
    def __init__(self, bert, hidden_size=640):
        super(RNAFmForReg, self).__init__()
        self.bert = bert
        self.predictor = create_1dcnn_for_emd(hidden_size, 1)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    def forward(self, input_ids):
        with torch.no_grad():
            output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, :, :].transpose(
            1, 2)
        # print(representations.shape)
        logits = self.predictor(representations).squeeze(-1)
        # logits = logits[:, 1 + self.pad:-self.pad, :].transpose(1, 2)
        # print(logits.shape)
        return logits


class PureReg(nn.Module):
    def __init__(self):
        super(PureReg, self).__init__()
        self.predictor = create_1dcnn_for_emd(4, 1)
        self.token_len = 100

    def forward(self, input_ids):
        # covert token for RNA-FM (20 tokens) to nest version (4 tokens A,U,C,G)
        nest_tokens = (input_ids[:, 1:-1] - 4)
        nest_tokens = torch.nn.functional.pad(
            nest_tokens, (0, self.token_len - nest_tokens.shape[1]), value=-2)
        token_padding_mask = nest_tokens.ge(0).long()
        one_hot_tokens = torch.nn.functional.one_hot(
            (nest_tokens * token_padding_mask), num_classes=4)
        one_hot_tokens = one_hot_tokens.float() * token_padding_mask.unsqueeze(-1)

        # print(one_hot_tokens.shape)
        logits = self.predictor(one_hot_tokens.permute(0, 2, 1))[:, 0]
        # print(logits.shape)
        return logits


class UTRLMForReg(nn.Module):
    def __init__(self, backbone):
        super(UTRLMForReg, self).__init__()

        self.model = backbone

    def forward(self, x):
        logits = self.model(x).logits
        # print(logits)
        return logits
