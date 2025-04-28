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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if isinstance(m.bias, nn.Parameter):
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class PreEncoder(nn.Module):
    def __init__(self, base_token=5):
        super(PreEncoder, self).__init__()
        self.base_token = base_token

    def forward(self, input_ids):
        # covert token for RNA-FM (20 tokens) to nest version (4 tokens A,U,C,G)
        # nest_tokens = (input_ids[:, 1:-1] - 4)

        # assume input_ids only contains token for A U C G
        nest_tokens = (input_ids[:, :] - 4)
        token_padding_mask = nest_tokens.ge(0).long()
        one_hot_tokens = torch.nn.functional.one_hot(
            (nest_tokens * token_padding_mask), num_classes=4)
        one_hot_tokens = one_hot_tokens.float() * token_padding_mask.unsqueeze(-1)
        one_hot_tokens = one_hot_tokens.permute(0, 2, 1)

        return one_hot_tokens


class RNAFmForReg(nn.Module):
    def __init__(self, bert, hidden_size=640):
        super(RNAFmForReg, self).__init__()
        self.bert = bert
        self.predictor = create_1dcnn_for_emd(640, 1)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]

        with torch.no_grad():
            self.bert.eval()
            output = self.bert(input_ids, need_head_weights=False, repr_layers=[
                12], return_contacts=False)

            representations = output["representations"][12][:, :, :]
            representations = representations.transpose(1, 2)

        logits = self.predictor(representations).squeeze(-1)
        return logits


class RNAErnieForReg(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.predictor = create_1dcnn_for_emd(768, 1)

    def forward(self, input_ids):
        with torch.no_grad():
            logits = self.bert(input_ids, attention_mask=input_ids > 0)[
                'last_hidden_state']
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
        return logits


class RNAErnieForRegAB(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids):
        logits = self.bert(input_ids, attention_mask=input_ids > 0)[
            'last_hidden_state']
        logits = self.classifier(logits[:, 0, :])
        return logits


class RNAMsmForReg(nn.Module):
    def __init__(self, bert, hidden_size=768, class_num=1):
        super(RNAMsmForReg, self).__init__()
        self.bert = bert
        self.predictor = create_1dcnn_for_emd(hidden_size, 1)
        # self.classifier = nn.Linear(hidden_size, class_num)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        with torch.no_grad():
            output = self.bert(input_ids, repr_layers=[10])
        representations = output["representations"][10][:, 0, :, :].transpose(
            1, 2)
        logits = self.predictor(representations).squeeze(-1)
        return logits


class RNABERTForReg(nn.Module):
    def __init__(self, bert, hidden_size=120):
        super(RNABERTForReg, self).__init__()
        self.bert = bert
        self.predictor = create_1dcnn_for_emd(hidden_size, 1)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        with torch.no_grad():
            encoded_layers, pooled_output = self.bert(
                input_ids, attention_mask=input_ids > 0, output_all_encoded_layers=False)
        logits = self.predictor(encoded_layers.transpose(
            1, 2)).squeeze(-1)
        return logits


class DNABERTForReg(nn.Module):
    def __init__(self, model, args):
        super(DNABERTForReg, self).__init__()
        self.model = model
        in_features = 768 if args.method == 'DNABERT' else 512
        self.predictor = create_1dcnn_for_emd(in_features, 1)

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        with torch.no_grad():
            # logits = self.model(input_ids, attention_mask=input_ids > 0)[
            #     'last_hidden_state']
            logits = self.model(input_ids,  attention_mask=input_ids > 0)[
                'last_hidden_state']
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
        return logits


class DNABERT2ForReg(nn.Module):
    def __init__(self, model):
        super(DNABERT2ForReg, self).__init__()
        self.predictor = create_1dcnn_for_emd(768, 1)
        self.model = model

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        with torch.no_grad():
            # logits = self.model(input_ids)[
            #     'last_hidden_state']
            logits = self.model(input_ids, attention_mask=input_ids != 3)[0]
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
        return logits


class NTForReg(nn.Module):
    def __init__(self, model):
        super(NTForReg, self).__init__()
        self.predictor = create_1dcnn_for_emd(1024, 1)
        self.model = model

    def forward(self, input_ids):
        input_ids = input_ids[:, 1:]
        with torch.no_grad():
            torch_outs = self.model(
                input_ids, attention_mask=input_ids > 1, output_hidden_states=True)
            logits = torch_outs['hidden_states'][-1]
        logits = self.predictor(logits.transpose(
            1, 2)).squeeze(-1)
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


class Optimus(nn.Module):
    def __init__(self, inp_len=50, nodes=40, layers=3, filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0, dropout3=0.2, base_token=5):
        super(Optimus, self).__init__()
        self.base_token = base_token

        self.model = nn.Sequential()
        # 1
        self.model.add_module('conv1', nn.Conv1d(
            4, nbr_filters, filter_len, padding='same', padding_mode='replicate'))
        self.model.add_module('relu1', nn.ReLU())
        # 2
        self.model.add_module('conv2', nn.Conv1d(
            nbr_filters, nbr_filters, filter_len, padding='same', padding_mode='replicate'))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('drop2', nn.Dropout(p=dropout1))
        # 3
        self.model.add_module('conv3', nn.Conv1d(
            nbr_filters, nbr_filters, filter_len, padding='same', padding_mode='replicate'))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('drop3', nn.Dropout(p=dropout2))

        self.model.add_module('flatten', nn.Flatten())

        # Add the fully connected layers
        self.model.add_module('fc1', nn.Linear(inp_len * nbr_filters, nodes))
        self.model.add_module('relu_fc', nn.ReLU())
        self.model.add_module('dropout_fc', nn.Dropout(p=dropout3))
        self.model.add_module('fc2', nn.Linear(nodes, 1))

    def forward(self, input_ids):
        # convert token from RNA-FM to one-hot encoding
        nest_tokens = (input_ids[:, 1:] - 4)  # drop the first [CLS] token
        token_padding_mask = nest_tokens.ge(0).long()
        one_hot_tokens = torch.nn.functional.one_hot(
            (nest_tokens * token_padding_mask), num_classes=4)
        one_hot_tokens = one_hot_tokens.float() * token_padding_mask.unsqueeze(-1)
        one_hot_tokens = one_hot_tokens.permute(0, 2, 1)

        logits = self.model(one_hot_tokens)[:, 0]
        return logits
