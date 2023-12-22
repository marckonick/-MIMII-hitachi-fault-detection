# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from torch.utils.data import Dataset



class labeled_dataset(Dataset):

      def __init__(self,X, Y):
        self.data = X
        self.labels = Y

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx])