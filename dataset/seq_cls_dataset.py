import os
import numpy as np
from torch.utils.data import Dataset
from Bio import SeqIO


class SeqClsDataset(Dataset):
    def __init__(self, fasta_dir, prefix, seed=0, train=True):
        super(SeqClsDataset, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix

        file_name = "train.fa" if train else "test.fa"
        fasta = os.path.join(os.path.join(fasta_dir, prefix), file_name)
        records = list(SeqIO.parse(fasta, "fasta"))
        self.data = [(str(x.seq), x.description.split(" ")[1])
                     for x in records]
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        label = instance[1]
        # print(seq)
        return {"seq": seq, "label": label}

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = SeqClsDataset("seq_cls_data", "nRC", train=False)
    max_len = 0
    min_len = 1000000
    for i in range(data.__len__()):
        seq = data[i]['seq']
        max_len = max(max_len, len(seq))
        min_len = min(min_len, len(seq))
    print(min_len, max_len)
