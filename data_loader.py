import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
# from mambaConfig import mambaConfig

class TrajDataset(Dataset):
    def __init__(self, dataset='CD', data_path_prefix="/data0/yuchen/mixer/ten_one_prediction_",
                 data_path="", size=[10, 1], flag='train'):
        super().__init__()

        self.data_path_prefix = data_path_prefix
        self.flag = flag
        self.data_path = data_path

        self.data_path = "{}_{}_".format(dataset, flag) + data_path

        self.seq_len = size[0]
        self.pred_len = size[1]
        # self.train_split = split[0] / sum(split)
        # self.test_split = split[1] / sum(split)
        # self.val_split = split[2] / sum(split)
        self.__init_data__()

    def __init_data__(self):
        # if self.flag == "train":
        dataset = np.load(self.data_path_prefix + self.data_path)

        # full_data = np.load(self.data_path)
        self.pred_pairs = dataset[:, :, :2]

    def __getitem__(self, index):
        input_seq = self.pred_pairs[index, :self.seq_len, :2]
        pred_seq = self.pred_pairs[index, -self.pred_len:, :2]

        return input_seq, pred_seq
    
    def __len__(self):
        return len(self.pred_pairs)
    
    

class MambaDataset(Dataset):
    def __init__(self, data_path, size=[400, 10], flag='train', dataset='XA', seed=2025):
        super().__init__()
        self.data_path = "/data1/jyc/mixer/{}_{}_".format(dataset, flag) + data_path
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.flag = flag

        self.__init_data__()

    def __init_data__(self):
        # if self.flag == "train":
        dataset = np.load(self.data_path)

        # full_data = np.load(self.data_path)
        self.pred_pairs = dataset[:, :, :2]

    def __getitem__(self, index):
        input_seq = self.pred_pairs[index, :self.seq_len, :2]
        pred_seq = self.pred_pairs[index, -self.pred_len:, :2]

        return input_seq, pred_seq
    
    def __len__(self):
        return len(self.pred_pairs)
