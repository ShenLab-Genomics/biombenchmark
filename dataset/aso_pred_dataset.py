import os
import numpy as np
from torch.utils.data import Dataset
from Bio import SeqIO
import pandas as pd


class ASOTokenDataset(Dataset):
    # 把序列拼接成(ASO + [SEP] + Target)的形式，给序列模型用
    def __init__(self, fpath, regression=False) -> None:
        super(ASOTokenDataset).__init__()
        self.fpath = fpath
        self.feature_dict = {}
        self.label_dict = {'maxskip': []}
        self.load_normal_data()
        #
        if regression:
            self.Y = self.label_dict['maxskip']
        else:
            self.Y = self.label_dict['maxskip']
            self.load_classification_label()
        print(f'Load ASO data from {fpath}, ',
              f'Input size: {len(self.X)} ,', f'Label size: {self.Y.shape}')
        pass

    def load_normal_data(self):
        # 从表格中直接获取特征
        basic_df = pd.read_csv(self.fpath, sep='\t')

        aso_seq = basic_df['oligo_sequence']
        target_seq = basic_df['seq_around']

        self.X = [(aso + 'S' + target)
                  for aso, target in zip(aso_seq, target_seq)]
        self.label_dict['maxskip'] = basic_df['max_skip'].to_numpy()
        pass

    def load_classification_label(self):
        # 处理成分类任务
        label_cutoff = (25, 75)  # 百分比
        self.Y = np.array([1 if y >= label_cutoff[1] else 0 if y <=
                          label_cutoff[0] else 2 for y in self.Y])
        # index = np.where(self.Y != 2)
        index = (self.Y != 2)
        # self.X = np.array(self.X)[index]
        # self.Y = np.array(self.Y)[index]
        self.X = [self.X[i] for i in range(len(self.X)) if index[i]]
        self.Y = self.Y[index]
        pass

    def __getitem__(self, idx):
        seq = self.X[idx]
        label = self.Y[idx]
        # print(seq, label)
        return {"seq": seq, "label": label}

    def __len__(self):
        return len(self.Y)
