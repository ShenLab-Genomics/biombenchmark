import numpy as np
import pandas as pd
from sklearn import preprocessing
from dataset.base_dataset import BaseDataset


class MRLDatasetTemp(BaseDataset):
    # class MRLDatasetTemp():
    def __init__(self, fpath, split_name='train') -> None:
        super().__init__()
        self.fpath = fpath
        self.set_name = split_name
        self.seqs, self.scaled_rls = self._process_data()

    def _process_data(self):
        # adpated from https://github.com/ml4bio/RNA-FM/blob/edffd7a620153e201959c3b1682760086817cd9e/tutorials/utr-function-prediction/UTR-Function-Prediction.ipynb
        # 1.Filter Data
        # (1) Random Set
        src_df = pd.read_csv(self.fpath)
        src_df.loc[:, "ori_index"] = src_df.index
        random_df = src_df[src_df['set'] == 'random']
        # Filter out UTRs with too few less reads
        random_df = random_df[random_df['total_reads']
                              >= 10]    # 87000 -> 83919
        random_df.sort_values('total_reads', inplace=True, ascending=False)
        random_df.reset_index(inplace=True, drop=True)

        # (2) Human Set
        human_df = src_df[src_df['set'] == 'human']
        # Filter out UTRs with too few less reads
        human_df = human_df[human_df['total_reads'] >= 10]   # 16739 -> 15555
        human_df.sort_values('total_reads', inplace=True, ascending=False)
        human_df.reset_index(inplace=True, drop=True)

        # 2.Split Dataset
        # (1) Generate Random Test set
        # random_df_test = pd.DataFrame(columns=random_df.columns)
        random_df_test = []
        for i in range(25, 101):
            tmp = random_df[random_df['len'] == i].copy()
            tmp.sort_values('total_reads', inplace=True, ascending=False)
            tmp.reset_index(inplace=True, drop=True)
            random_df_test.append(tmp.iloc[:100])
            # random_df_test = pd.concat([random_df_test, random_df[random_df['len'] == i].iloc[:100]])
        random_df_test = pd.concat(random_df_test)

        # (2) Generate Human Test set
        # human_df_test = pd.DataFrame(columns=human_df.columns)
        human_df_test = []
        for i in range(25, 101):
            tmp = human_df[human_df['len'] == i].copy()
            tmp.sort_values('total_reads', inplace=True, ascending=False)
            tmp.reset_index(inplace=True, drop=True)
            human_df_test.append(tmp.iloc[:100])
            # human_df_test = pd.concat([human_df_test, human_df[human_df['len'] == i].iloc[:100]])
        human_df_test = pd.concat(human_df_test)

        # (3) Exclude Test data from Training data
        train_df = pd.concat([random_df, random_df_test]
                             ).drop_duplicates(keep=False)  # 76319

        # 3.Label Normalization (ribosome loading value)
        label_col = 'rl'
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(train_df.loc[:, label_col].values.reshape(-1, 1))
        train_df.loc[:, 'scaled_rl'] = self.scaler.transform(
            train_df.loc[:, label_col].values.reshape(-1, 1))
        random_df_test.loc[:, 'scaled_rl'] = self.scaler.transform(
            random_df_test.loc[:, label_col].values.reshape(-1, 1))
        human_df_test.loc[:, 'scaled_rl'] = self.scaler.transform(
            human_df_test.loc[:, label_col].values.reshape(-1, 1))

        # 4.Pickup Target Set
        if self.set_name == "train":
            set_df = train_df
        elif self.set_name == "valid":
            set_df = random_df_test
        else:
            set_df = human_df_test
        seqs = set_df['utr'].values
        scaled_rls = set_df['scaled_rl'].values
        # scaled_rls = set_df['rl'].values
        names = set_df["ori_index"].values

        print("Num samples of {} dataset: {} ".format(
            self.set_name, set_df["len"].shape[0]))
        return seqs, scaled_rls

    def __getitem__(self, index):
        seq_str = self.seqs[index].replace("T", "U")
        label = self.scaled_rls[index]

        return {"seq": seq_str, "label": label}

    def __len__(self):
        return len(self.seqs)


class MRLDataset(BaseDataset):
    def __init__(self, fpath, split_name='train') -> None:
        super().__init__()
        self.fpath = fpath
        mpra_data_varlen = pd.read_csv(fpath)

        if split_name == "train":
            train_data_100 = mpra_data_varlen[(mpra_data_varlen.set == "train") & (
                mpra_data_varlen.library == "random")]
        elif split_name == "test":
            train_data_100 = mpra_data_varlen[(mpra_data_varlen.set == "test") & (
                mpra_data_varlen.library == "human")]

        self.set_name = split_name
        self.seqs = train_data_100['utr'].values
        self.scaled_rls = train_data_100['rl'].values

        # if split_name == 'train':
        self.scaled_rls = preprocessing.StandardScaler().fit_transform(
            self.scaled_rls.reshape(-1, 1)).reshape(-1)

    def __getitem__(self, index):
        seq_str = self.seqs[index].replace("T", "U")
        label = self.scaled_rls[index]

        return {"seq": seq_str, "label": label}

    def __len__(self):
        return len(self.seqs)


if __name__ == "__main__":
    # data = MRLDataset("dataset/mrl_data/data_dict.pkl")
    # df = pd.read_csv("dataset/mrl_data/mpra_data_varlen.csv")
    # print(df.head())
    df = pd.read_csv("dataset/mrl_data/mpra_data_varlen.csv")
    test_data_100 = df[(df.set == "test") & (
        df.library == "human")]
    utrs = test_data_100['utr'].values
    seq_name = []
    with open('dataset/mrl_data/data_fasta.fa', 'w') as f:
        cnt = 0
        for utr in utrs:
            seq_name.append(str(cnt))
            f.write('>' + str(cnt) + '\n')
            f.write(utr + '\n')
            cnt += 1
    test_data_100['seq_name'] = seq_name
    test_data_100.to_csv("dataset/mrl_data/mpra_data_varlen_test.csv")
