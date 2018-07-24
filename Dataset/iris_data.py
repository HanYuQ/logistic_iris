# -*- coding:utf-8 -*-
import os
from torch.utils import data
import numpy as np
import torch as t


class Irisdata(data.Dataset):

    def __init__(self, root,Pattern='train'):
        '''
        Get images, divide into train/val set
        '''

        self.Pattern = Pattern
        self.images_root = root

        self._read_txt_file()


    def _read_txt_file(self):
        self.data_path = []
        self.label_path = []

        if self.Pattern == 'train':
            txt_file = self.images_root + "Iris_train.txt"
        if self.Pattern == 'val':
            txt_file = self.images_root + "Iris_val.txt"
        if self.Pattern == 'test':
            txt_file = self.images_root + "Iris_test.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.data_path.append(item[0])
                self.label_path.append(item[1])

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        label = self.label_path[index]
        label = np.array(int(label))
        label = t.from_numpy(label)
        Datas = self.data_path[index]
        Datas = Datas.split(',')
        Data = list(map(lambda x: float(x), Datas))
        Data = np.array(Data)
        Data = t.from_numpy(Data)

        return Data,label

    def __len__(self):
        return len(self.data_path)
