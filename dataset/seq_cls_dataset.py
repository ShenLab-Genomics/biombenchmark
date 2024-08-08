import os
import numpy as np
from torch.utils.data import Dataset
from Bio import SeqIO


class SeqClsDataset(Dataset):
    def __init__(self, fasta_dir, prefix, tokenizer, seed=0, train=True):
        super(SeqClsDataset, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.tokenizer = tokenizer

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
        print(seq)
        return {"seq": seq, "label": label}

    def __len__(self):
        return len(self.data)
