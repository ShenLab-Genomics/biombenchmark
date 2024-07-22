import h5py
import torch
from tqdm import tqdm
from dataset.base_dataset import BaseDataset


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
        Xk, idx = self.idx_to_key[idx]
        Yk = Xk.replace("X", "Y")
        X = torch.from_numpy(self.h5f[Xk][idx]).float()[:, :]
        Y = torch.from_numpy(self.h5f[Yk][0, idx]).float()
        idx = torch.max(Y[3:, :], dim=0)[0] < 0.05  # desired
        Y[0, idx] = 1
        Y[1:, idx] = 0
        return (X, Y)
