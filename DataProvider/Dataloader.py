import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


class SequenceSleepDataset(Dataset):
    def __init__(self, data, label, window=10, normalize_method=None, resmaple=False):
        """
        :param data:
        :param label:
        :param window:
        :param normalize_method:
        """
        self.resample = resmaple
        self.data = data
        self.label = label
        self.window = window

        assert len(self.data) == len(self.label), "Error: Inconsistent length between data and label"
        self.x_end_index = len(self.data)
    

    def __getitem__(self, index):
        data = self.data[index: index + self.window].squeeze()
        label = self.label[index: index + self.window].squeeze()
        if self.resample:
            padding = np.zeros((21, 2, 750))
            data = np.concatenate((data, padding), axis=2)
        
        return data, label

    def __len__(self):
        return self.x_end_index - self.window
