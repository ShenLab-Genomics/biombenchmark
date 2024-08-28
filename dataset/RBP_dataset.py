import os
import numpy as np
from Bio import SeqIO
from dataset.base_dataset import BaseDataset


class RBPDataset(BaseDataset):
    def __init__(self, fasta_path, seed=0) -> None:
        super().__init__()

        self.fasta_path = fasta_path
        fasta = fasta_path
        records = list(SeqIO.parse(fasta, "fasta"))
        self.data = [(str(x.seq), x.description.split(":")[1])
                     for x in records]

        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, index):
        instance = self.data[index]
        seq = instance[0]
        label = int(instance[1])
        return {"seq": seq, "label": label}

    def __len__(self):
        return len(self.data)
