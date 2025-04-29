import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, f, k, p):
        super(DenseBlock, self).__init__()
        self.conv1 = Conv1DBlock(in_channels, f, k)
        self.conv2 = Conv1DBlock(f, f, k)
        self.conv3 = Conv1DBlock(f*2, f, k)
        self.conv4 = Conv1DBlock(f*2, f, k)
        self.pool = nn.MaxPool1d(p, stride=p)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(cat1)
        cat2 = torch.cat([x1, x3], dim=1)
        x4 = self.conv4(cat2)
        out = self.pool(x4)
        return out

class FCBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x

class ncRDense(nn.Module):
    def __init__(self, num_classes=13):
        super(ncRDense, self).__init__()
        
        # First branch
        self.conv1_1 = Conv1DBlock(4, 64, 5)
        self.pool1_1 = nn.MaxPool1d(2, stride=2)
        self.dense1_1 = DenseBlock(64, 128, 5, 4)
        self.dense1_2 = DenseBlock(128, 128, 5, 4)
        self.conv1_2 = Conv1DBlock(128, 64, 5)
        self.pool1_2 = nn.MaxPool1d(2, stride=2)
        
        # Second branch
        self.conv2_1 = Conv1DBlock(3, 64, 5)
        self.pool2_1 = nn.MaxPool1d(2, stride=2)
        self.dense2_1 = DenseBlock(64, 128, 5, 4)
        self.dense2_2 = DenseBlock(128, 128, 5, 4)
        self.conv2_2 = Conv1DBlock(128, 64, 5)
        self.pool2_2 = nn.MaxPool1d(2, stride=2)
        
        # Combined layers
        self.conv3 = Conv1DBlock(128, 64, 3)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        self.flatten = nn.Flatten()
        
        # FC layers
        self.fc1 = FCBlock(320, 256)  # 输入维度需要根据实际计算得到
        self.fc2 = FCBlock(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x1x2):
        x1x2 = x1x2.float()
        # x1: one-hot encoded sequence
        x1 = x1x2[:, :4, :]
        # x2: one-hot encoded structure
        x2 = x1x2[:, 4:, :]

        # First branch
        x1 = self.conv1_1(x1)
        x1 = self.pool1_1(x1)
        x1 = self.dense1_1(x1)
        x1 = self.dense1_2(x1)
        x1 = self.conv1_2(x1)
        x1 = self.pool1_2(x1)
        
        # Second branch
        x2 = self.conv2_1(x2)
        x2 = self.pool2_1(x2)
        x2 = self.dense2_1(x2)
        x2 = self.dense2_2(x2)
        x2 = self.conv2_2(x2)
        x2 = self.pool2_2(x2)
        
        # Combine branches
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        
        # FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x