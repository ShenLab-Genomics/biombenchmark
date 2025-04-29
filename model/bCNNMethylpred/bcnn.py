import torch
import torch.nn as nn
import torch.nn.functional as F


class bCNN(nn.Module):
    def __init__(self):
        super(bCNN, self).__init__()
        input_length = 101

        # Branch 1 (input1)
        self.conv1_1 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        self.gn1_1 = nn.GroupNorm(4, 32)
        self.conv1_2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(16 * input_length, 24)  # 8*3 = 24

        # Branch 2 (input2)
        self.conv2_1 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.gn2_1 = nn.GroupNorm(4, 32)
        self.conv2_2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(16 * input_length, 24)

        # Branch 3 (input3)
        self.conv3_1 = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        self.gn3_1 = nn.GroupNorm(4, 32)
        self.conv3_2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(16 * input_length, 24)

        # Final layers
        self.dense_final = nn.Linear(72, 2)  # 24*3 = 72

        # Weight decay (L2 regularization)
        self.weight_decay = 1e-3
        self.bias_decay = 1e-4

    def forward(self, x):

        # Input shapes: x1:(B,4,41), x2:(B,16,41), x3:(B,3,41)
        # However, the source code swaped x2 and x3 before input them into the model, we follow the source code
        x = x.permute(0, 2, 1)
        x1 = x[:, 0:4, :]
        x3 = x[:, 4:7, :]
        x2 = x[:, 7:, :]

        # Branch 1
        x1 = F.relu(self.conv1_1(x1))
        x1 = self.gn1_1(x1)
        x1 = F.relu(self.conv1_2(x1))
        x1 = x1.flatten(1)
        x1 = self.dropout1(x1)
        x1 = F.relu(self.dense1(x1))

        # Branch 2
        x2 = F.relu(self.conv2_1(x2))
        x2 = self.gn2_1(x2)
        x2 = F.relu(self.conv2_2(x2))
        x2 = x2.flatten(1)
        x2 = self.dropout2(x2)
        x2 = F.relu(self.dense2(x2))

        # Branch 3
        x3 = F.relu(self.conv3_1(x3))
        x3 = self.gn3_1(x3)
        x3 = F.relu(self.conv3_2(x3))
        x3 = x3.flatten(1)
        x3 = self.dropout3(x3)
        x3 = F.relu(self.dense3(x3))

        # Concatenate and final layer
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dense_final(x)

        return x
