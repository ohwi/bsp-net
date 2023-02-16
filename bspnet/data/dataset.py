import h5py

from torch.utils.data import Dataset

import numpy as np


class BSP2dDataset(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            self.data = f["pixels"][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        data = np.transpose(data, (2, 0, 1))    # channel first
        return data
