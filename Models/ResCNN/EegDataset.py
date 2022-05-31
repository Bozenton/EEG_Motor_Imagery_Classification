import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.calc_hw import calc_hw


class EegDataset(Dataset):
    def __init__(self, data_path, label_path):
        assert os.path.exists(data_path), "Data path: {} does not exist".format(data_path)
        assert os.path.exists(label_path), "Label path {} does not exist".format(label_path)
        self.data_path = data_path
        self.label_path = label_path

        self.data = torch.load(self.data_path)
        self.label = torch.load(self.label_path)
        n_sample, n_channel, n_points = self.data.shape
        h, w = calc_hw(n_points)
        self.data = self.data.reshape([n_sample, n_channel, h, w])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx, ...]*1e5, self.label[idx]-1   # make index start from 0