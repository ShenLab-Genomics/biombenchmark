import torch
import torch.nn as nn
import pandas as pd

from .esm.data import *
from .esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from .esm import Alphabet

seed = 1337
torch.manual_seed(seed)

# global modelfile, layers, heads, embed_dim, batch_toks, inp_len, device
modelfile = 'model.pt'

inp_len = 50

# device = "cpu"

alphabet = Alphabet(mask_prob=0.0, standard_toks='AGCT')
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2,
                               'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}


class UTRLM(nn.Module):
    def __init__(self,
                 border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):

        super().__init__()

        self.layers = 6
        heads = 16
        embed_dim = 128
        batch_toks = 4096

        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = 40
        self.cnn_layers = 0
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = 0.5

        self.esm2 = ESM2_SISS(num_layers=self.layers,
                              embed_dim=embed_dim,
                              attention_heads=heads,
                              alphabet=alphabet)

        self.conv1 = nn.Conv1d(in_channels=self.embedding_size,
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding=self.border_mode)
        self.conv2 = nn.Conv1d(in_channels=self.nbr_filters,
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding=self.border_mode)

        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
        self.linear = nn.Linear(
            in_features=self.nbr_filters, out_features=self.nodes)
        self.output = nn.Linear(in_features=self.nodes, out_features=1)

    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation=True):

        x = self.esm2(tokens, [self.layers], need_head_weights,
                      return_contacts, return_representation)
        # x = self.esm2(tokens, [layers])

        x = x["representations"][self.layers][:, 0]
        x_o = x.unsqueeze(2)

        x = self.flatten(x_o)
        o_linear = self.fc(x)
        o_relu = self.relu(o_linear)
        o_dropout = self.dropout3(o_relu)
        o = self.output(o_dropout)
        return o

    def forward_backbone(self, tokens, need_head_weights=True, return_contacts=True, return_representation=True):
        x = self.esm2(tokens, [self.layers], need_head_weights,
                      return_contacts, return_representation)
        x = x["representations"][self.layers]
        return x


'''
def eval_step(dataloader, model, threshold=0.5):
    model.eval()
    logits_list = []
    ids_list, strs_list = [], []
    with torch.no_grad():
        for i, (ids, strs, _, toks, _, _) in enumerate(dataloader):
            ids_list.extend(ids)
            strs_list.extend(strs)
            # print(toks)
            logits = model(toks, return_representation=True,
                           return_contacts=True)

            logits = logits.reshape(-1)
            # y_prob = torch.sigmoid(logits)
            # y_pred = (y_prob > threshold).long()

            logits_list.extend(logits.tolist())
            # y_prob_list.extend(y_prob.tolist())
            # y_pred_list.extend(y_pred.tolist())

    # data_pred = pd.DataFrame({'ID':ids_list, 'Sequence':strs_list, "MRL":logits_list, "prob":y_prob_list, "pred":y_pred_list})
    data_pred = pd.DataFrame(
        {'ID': ids_list, 'Sequence': strs_list, "MRL": logits_list})
    return data_pred

'''
