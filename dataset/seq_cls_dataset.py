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

class SeqClsDatasetOneHot(Dataset):
    def filterseq(self, seq):
        # 检查所有字符是否都是A、G、C、U，不符合的全都替换成N
        seq = [x if x in ['A', 'G', 'C', 'U'] else 'N' for x in seq]
        return ''.join(seq)

    def __init__(self, fasta_dir, prefix, seed=0, train=True,rnafold=False):
        super(SeqClsDatasetOneHot, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.rnafold = rnafold

        
        if rnafold:
            file_name = "train.fa" if train else "test.fa"
            fasta = os.path.join(os.path.join(fasta_dir, prefix), file_name)
            records = list(SeqIO.parse(fasta, "fasta"))
            labels = [x.description.split(" ")[1] for x in records]

            file_name = "train_fold.fa" if train else "test_fold.fa"
            fasta = os.path.join(os.path.join(fasta_dir, prefix), file_name)
            records = list(SeqIO.parse(fasta, "fasta"))
            self.data = []
            for idx,x in enumerate(records):
                seq = str(x.seq)
                seq = seq[:seq.rfind('(')]
                fasta_seq = self.filterseq(seq[:len(seq)//2])
                structure_seq = seq[len(seq)//2:]
                # label = x.id.split(":")[-1]
                label = labels[idx]

                self.data.append((fasta_seq, structure_seq, label))
        else:
            file_name = "train.fa" if train else "test.fa"
            fasta = os.path.join(os.path.join(fasta_dir, prefix), file_name)
            records = list(SeqIO.parse(fasta, "fasta"))
            self.data = [(str(x.seq), x.description.split(" ")[1])
                        for x in records]
            print(self.data[0])

        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        if self.rnafold:
            seq = instance[0]
            struct = instance[1] 
            label = instance[2]
            return {"seq": seq,"struct":struct, "label": label}
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = SeqClsDatasetOneHot("seq_cls_data", "mix_0.1", train=False,rnafold=True)
    max_len = 0
    min_len = 1000000
    for i in range(data.__len__()):
        seq = data[i]['seq']
        # print(seq)
        # print(data[i]['struct'])
        if 'M' in seq:
            print(seq)
            break
