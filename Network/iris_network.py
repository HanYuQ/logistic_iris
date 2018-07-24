import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class Iris(nn.Module):
    def __init__(self,Factor,Classification):
        super(Iris,self).__init__()
        # self.Classifications = Classifications
        self.layer1 = nn.Sequential(
            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(Factor, Classification),
            nn.BatchNorm1d(Classification),
            nn.ReLU())

        # self.layer2 = nn.Sequential(
        #     nn.Linear(8, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU())
        #
        # self.layer3 = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU())
        #
        # self.layer4 = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU())
        #
        # self.layer5 = nn.Sequential(
        #     nn.Linear(16,Classification),
        #     nn.BatchNorm1d(Classification),
        #     nn.ReLU()
        # )


    def forward(self, x):
        # fc1 = self.layer1(x)
        # fc2 = self.layer2(fc1)
        # fc3 = self.layer3(fc2)
        # fc4 = self.layer4(fc3)
        # out = self.layer5(fc4)
        out = self.layer1(x)
        return out