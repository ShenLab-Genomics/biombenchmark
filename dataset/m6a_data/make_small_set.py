import pandas as pd
import numpy as np

import os
import numpy as np
from torch.utils.data import Dataset
from Bio import SeqIO


def solve_data(sp='test.fa', ratio=0.1):
    fasta = os.path.join('101bp/miCLIP', sp)
    records = list(SeqIO.parse(fasta, "fasta"))

    # data_101 = [(str(x.seq), x.description.split(" ")[1])
    #                 for x in records]
    data_101 = records

    fasta = os.path.join('509bp/miCLIP', sp)
    records = list(SeqIO.parse(fasta, "fasta"))
    # data_509 = [(str(x.seq), x.description.split(" ")[1])
    #                 for x in records]
    data_509 = records

    assert len(data_101) == len(data_509)
    print('total size:', len(data_101))

    np.random.seed(0)
    small_set_index = np.random.choice(len(data_101), int(ratio * len(data_101)), replace=False)

    data_101_small = [data_101[i] for i in small_set_index]
    data_509_small = [data_509[i] for i in small_set_index]

    print('small size:', len(data_101_small))

    SeqIO.write(data_101_small, f'101bp/miCLIP/{ratio}_{sp}', 'fasta-2line')
    SeqIO.write(data_509_small, f'509bp/miCLIP/{ratio}_{sp}', 'fasta-2line')


if __name__ == '__main__':
    # solve_data(sp='test.fa',ratio=0.1)
    # solve_data(sp='test.fa',ratio=0.5)
    solve_data(sp='train.fa',ratio=0.01)
    solve_data(sp='train.fa',ratio=0.1)
    solve_data(sp='train.fa',ratio=0.5)
