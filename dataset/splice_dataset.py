import h5py
import torch
import numpy as np
from tqdm import tqdm
from dataset.base_dataset import BaseDataset
from dataset.splice_data.data_maker import IN_MAP, one_hot_encode
from transformers import BertTokenizer, BertTokenizerFast


class SpliceDataset(BaseDataset):
    def __init__(self, h5_filename) -> None:
        super().__init__()
        self.h5_filename = h5_filename
        self.h5f = h5py.File(self.h5_filename, "r")
        self.idx_to_key, self.num_examples = self.map_idx_to_key()

    def map_idx_to_key(self):

        num_examples = 0
        idx_to_key = {}

        for k in tqdm(sorted(self.h5f.keys(), key=lambda x: int(x[1:]))):
            assert k.startswith("X") or k.startswith("Y")

            if k.startswith("X"):
                for idx in range(self.h5f[k].shape[0]):
                    idx_to_key[idx + num_examples] = (k, idx)
                num_examples += self.h5f[k].shape[0]

        assert max(idx_to_key) == num_examples - 1
        return idx_to_key, num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        raise NotImplementedError


class SpliceNormalDataset(SpliceDataset):
    def __getitem__(self, idx):
        """
        Output:
            X: (4, context + center + context) shaped array
            Y: (dim, center) shaped array
        """
        Xk, idx = self.idx_to_key[idx]
        Yk = Xk.replace("X", "Y")
        X = self.h5f[Xk][idx]
        BASES = 'NACGT'
        X = ''.join(BASES[int(x)] for x in X)
        Y = torch.from_numpy(self.h5f[Yk][idx]).float()
        idx = torch.max(Y[3:, :], dim=0)[0] < 0.05  # desired
        Y[0, idx] = 1
        Y[1:, idx] = 0
        return (X, Y)


class SpTransformerDataset(SpliceDataset):
    def __init__(self, h5_filename) -> None:
        super().__init__(h5_filename)

    def __getitem__(self, idx):
        """
        Output:
            X: (4, context + center + context) shaped array
            Y: (dim, center) shaped array
        """
        Xk, idx = self.idx_to_key[idx]
        Yk = Xk.replace("X", "Y")
        # Turn X into one-hot encoding
        X = np.array(self.h5f[Xk][idx])
        X = one_hot_encode(X, IN_MAP)
        X = torch.from_numpy(X).float().transpose(0, 1)  # (L, 4) -> (4, L)
        #
        Y = torch.from_numpy(self.h5f[Yk][idx]).float()
        idx = torch.max(Y[3:, :], dim=0)[0] < 0.05  # desired
        Y[0, idx] = 1
        Y[1:, idx] = 0
        return (X, Y)


class SpliceBERTDataset(SpliceDataset):
    def __init__(self, h5_filename, tokenizer: BertTokenizer,
                 max_len: int, dnabert_k: int = None, shift=0, reverse=False) -> None:
        super().__init__(h5_filename)
        self.shift = shift
        self.reverse = reverse
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.k = dnabert_k

    def __getitem__(self, idx):
        # Turn one-hot encoding into normal sequence
        Xk, idx = self.idx_to_key[idx]
        Yk = Xk.replace("X", "Y")

        # Parse X into nucleotide string
        X = np.array(self.h5f[Xk][idx]).astype('int')
        start = (len(X) - self.max_len)//2

        if self.k is None:
            X = X[start:start+self.max_len]
        else:
            X = X[start-1:start+self.max_len+1]  # padding

        #
        Y = torch.from_numpy(self.h5f[Yk][idx]).float()
        idx = torch.max(Y[3:, :], dim=0)[0] < 0.05  # desired
        Y[0, idx] = 1
        Y[1:, idx] = 0

        #####
        BASES = 'NACGT'
        if self.k is None:
            X = ' '.join(BASES[x] for x in X)
            input_ids = torch.tensor(self.tokenizer.encode(X))
        else:
            X = ''.join(BASES[x] for x in X)
            X = self.seq_to_dnabert_kmers(X.upper(), k=self.k)
            input_ids = torch.tensor(self.tokenizer.encode(X))
        mask = torch.ones_like(input_ids)

        return input_ids, mask, Y

    def seq_to_dnabert_kmers(self, seq, k: int):
        kmers = list()
        for i in range(0, len(seq) - k + 1):
            kmers.append(seq[i:i+k])
        return ' '.join(kmers)


class DNABERT2Dataset(SpliceDataset):
    def __init__(self, h5_filename, tokenizer: BertTokenizer,
                 max_len: int) -> None:
        super().__init__(h5_filename)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Turn one-hot encoding into normal sequence
        Xk, idx = self.idx_to_key[idx]
        Yk = Xk.replace("X", "Y")

        # Parse X into nucleotide string
        X = np.array(self.h5f[Xk][idx]).astype('int')
        start = (len(X) - self.max_len)//2
        X = X[start:start+self.max_len]

        #
        Y = torch.from_numpy(self.h5f[Yk][idx]).float()
        idx = torch.max(Y[3:, :], dim=0)[0] < 0.05  # desired
        Y[0, idx] = 1
        Y[1:, idx] = 0

        BASES = 'NACGT'
        X = ' '.join(BASES[x] for x in X)
        input_ids = torch.tensor(self.tokenizer.encode(
            X,
            return_tensors="pt",
            padding="longest",
        ))
        mask = torch.ones_like(input_ids)
        return input_ids, mask, Y


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        'model/pretrained/SpliceBERT/models/SpliceBERT.510nt')
    ds = SpliceBERTDataset(
        '/public/home/shenninggroup/yny/code/biombenchmark/dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5',
        tokenizer=tokenizer,
        max_len=500)
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        'model/pretrained/DNABERT/DNABERT-2-117M')
    ds = DNABERT2Dataset('/public/home/shenninggroup/yny/code/biombenchmark/dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5',
                         tokenizer=tokenizer, max_len=500)
    dl = DataLoader(ds, batch_size=1)
    cnt = 0
    for idx, (input_id, mask, Y) in enumerate(dl):
        cnt += 1
        print(input_id)
        print(input_id.shape)
        if cnt > 5:
            break
