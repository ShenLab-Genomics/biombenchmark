import numpy as np
from Bio import SeqIO
from dataset.base_dataset import BaseDataset


class M6ADataset(BaseDataset):
    def __init__(self, fasta_dir, seed=0) -> None:
        super().__init__()
        records = list(SeqIO.parse(fasta_dir, "fasta"))
        self.data = [(str(x.seq), x.description.split(":")[-1])
                     for x in records]
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        label = instance[1]
        return {"seq": seq, "label": label}

    def __len__(self):
        return len(self.data)
