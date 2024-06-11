import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class UARdataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label.iloc[item, 0]

        data = torch.from_numpy(data)
        data = data.permute(2, 0, 1)
        label = torch.from_numpy(np.array(label))

        return data, label

    def __len__(self):
        return len(self.data)
