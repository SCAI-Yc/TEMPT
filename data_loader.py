import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

class TrajDataset(Dataset):
    def __init__(self, dataset='CD', data_path_prefix="./data/",
                 data_path="", size=[10, 1], flag='train'):
        super().__init__()

        self.data_path_prefix = data_path_prefix
        self.flag = flag
        self.data_path = data_path

        self.data_path = "{}_demo_{}_".format(dataset, flag) + data_path    # add demo only for demo data.

        self.seq_len = size[0]
        self.pred_len = size[1]
        self.__init_data__()

    def __init_data__(self):
        # if self.flag == "train":
        dataset = np.load(self.data_path_prefix + self.data_path)

        self.pred_pairs = dataset[:, :, :2]

    def __getitem__(self, index):
        input_seq = self.pred_pairs[index, :self.seq_len, :2]
        pred_seq = self.pred_pairs[index, -self.pred_len:, :2]

        return input_seq, pred_seq
    
    def __len__(self):
        return len(self.pred_pairs)
    
