import os
import collections
import torch
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer
import torch.nn.functional as F


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class RNATokenizer(PreTrainedTokenizer):
    r"""
    """

    def __init__(
        self,
        vocab_file,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        print('vocab:', self.ids_to_tokens)
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=False,
            never_split=never_split,
        )
        super().__init__(
            do_lower_case=False,
            do_basic_tokenize=True,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # print('tokenize:',text)
        return self.basic_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string


class RNABertForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=120):
        super(RNABertForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, 13)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        _, pooled_output = self.bert(input_ids)
        logits = self.classifier(pooled_output)
        return logits


class RNAMsmForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=768):
        super(RNAMsmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, 13)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(
            path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[10])
        representations = output["representations"][10][:, 0, 0, :]
        logits = self.classifier(representations)
        return logits


class RNAFmForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=640):
        super(RNAFmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, 13)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[12])
        representations = output["representations"][12][:, 0, :]
        logits = self.classifier(representations)
        return logits


class SeqClsLoss(nn.Module):

    def __init__(self):
        super(SeqClsLoss, self).__init__()

    def forward(self, outputs, labels):
        # convert labels to int64
        loss = F.cross_entropy(outputs, labels)
        return loss
