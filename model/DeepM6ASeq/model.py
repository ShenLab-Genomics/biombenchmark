import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.nn import init


class ConvNet(torch.nn.Module):
    def __init__(self, output_dim, args, wordvec_len):
        super(ConvNet, self).__init__()

        args.filter_num = '256-128'
        args.filter_size = '10-5'
        args.pool_size = 0
        args.cnndrop_out = 0.5
        args.if_bn = 'Y'
        args.fc_size = 0
        args.rnn_size = 32

        self.conv = torch.nn.Sequential()
        self.args = args

        # cnn layers
        for i in range(len(args.filter_size.split('-'))):

            if i == 0:
                self.conv.add_module("conv_" + str(i+1),
                                     torch.nn.Conv1d(wordvec_len, int(args.filter_num.split('-')[i]),
                                                     kernel_size=int(
                                                         args.filter_size.split('-')[i]),
                                                     padding=int(int(args.filter_size.split('-')[i])/2)))

            else:
                self.conv.add_module("conv_" + str(i+1),
                                     torch.nn.Conv1d(int(args.filter_num.split('-')[i-1]), int(args.filter_num.split('-')[i]),
                                                     kernel_size=int(args.filter_size.split('-')[i])))

            # pool
            if args.pool_size != 0:
                self.conv.add_module(
                    "maxpool_" + str(i + 1), torch.nn.MaxPool1d(kernel_size=args.pool_size))

            # activation
            self.conv.add_module("relu_"+str(i+1), torch.nn.ReLU())

            # batch norm
            if args.if_bn == 'Y':
                self.conv.add_module("batchnorm_" + str(i + 1),
                                     torch.nn.BatchNorm1d(int(args.filter_num.split('-')[i])))

            # dropout
            self.conv.add_module("dropout_"+str(i+1),
                                 torch.nn.Dropout(args.cnndrop_out))

        # fc layer
        self.fc = torch.nn.Sequential()
        if args.fc_size > 0:
            self.fc.add_module("fc_1", torch.nn.Linear(
                int(args.filter_num.split('-')[-1]), int(args.fc_size)))
            self.fc.add_module("relu_1", torch.nn.ReLU())
            self.fc.add_module("fc_2", torch.nn.Linear(
                int(args.fc_size), output_dim))
        else:
            self.fc.add_module("fc_1", torch.nn.Linear(
                int(args.filter_num.split('-')[-1]), output_dim))

    def forward(self, x):
        x = x.float()
        x = x.transpose(1, 2)  # convert to 4*101
        x = self.conv.forward(x)
        x = torch.max(x, 2)[0]
        x = x.view(-1, int(self.args.filter_num.split('-')[-1]))

        return self.fc.forward(x)


class ConvNet_BiLSTM(torch.nn.Module):
    def __init__(self, output_dim, args, wordvec_len):
        super(ConvNet_BiLSTM, self).__init__()

        args.filter_num = '256-128'
        args.filter_size = '10-5'
        args.pool_size = 0
        args.cnndrop_out = 0.5
        args.if_bn = 'Y'
        args.fc_size = 0
        args.rnn_size = 32
        self.args = args

        self.conv = torch.nn.Sequential()

        # cnn layers
        for i in range(len(args.filter_size.split('-'))):

            if i == 0:
                self.conv.add_module("conv_" + str(i + 1),
                                     torch.nn.Conv1d(wordvec_len, int(args.filter_num.split('-')[i]),
                                                     kernel_size=int(
                                                         args.filter_size.split('-')[i]),
                                                     padding=int(
                                                         int(args.filter_size.split('-')[i]) / 2)
                                                     ))
            else:
                self.conv.add_module("conv_" + str(i + 1),
                                     torch.nn.Conv1d(int(args.filter_num.split('-')[i - 1]),
                                                     int(args.filter_num.split(
                                                         '-')[i]),
                                                     kernel_size=int(
                                                         args.filter_size.split('-')[i])
                                                     ))

            # pool

            if args.pool_size != 0:
                self.conv.add_module(
                    "maxpool_" + str(i + 1), torch.nn.MaxPool1d(kernel_size=args.pool_size))

            # activation
            self.conv.add_module("relu_" + str(i + 1), torch.nn.ReLU())

            # batchnorm
            if args.if_bn == 'Y':
                self.conv.add_module("batchnorm_" + str(i + 1),
                                     torch.nn.BatchNorm1d(int(args.filter_num.split('-')[i])))

            # dropout
            self.conv.add_module("dropout_" + str(i + 1),
                                 torch.nn.Dropout(args.cnndrop_out))

        self.lstm = torch.nn.LSTM(int(args.filter_num.split(
            '-')[-1]), args.rnn_size, 1, batch_first=True, bidirectional=True)

        # fc layer
        self.fc = torch.nn.Sequential()

        if args.fc_size > 0:
            self.fc.add_module("fc_1", torch.nn.Linear(
                args.rnn_size * 2, int(args.fc_size)))
            self.fc.add_module("relu_1", torch.nn.ReLU())
            self.fc.add_module("fc_2", torch.nn.Linear(
                int(args.fc_size), output_dim))
        else:
            self.fc.add_module("fc_1", torch.nn.Linear(
                args.rnn_size * 2, output_dim))

    def forward(self, x):
        x = x.float()
        # 2 for bidirection
        h0 = Variable(torch.zeros(1 * 2, x.size(0),
                      self.args.rnn_size)).to(x.device)
        c0 = Variable(torch.zeros(1 * 2, x.size(0),
                      self.args.rnn_size)).to(x.device)

        x = x.transpose(1, 2)  # convert to 4*101
        x = self.conv.forward(x)
        x = x.transpose(1, 2)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(torch.mean(out, 1))
        return out
