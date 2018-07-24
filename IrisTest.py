from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn,optim
from torch.autograd import Variable
import torch
import time

from Dataset import Irisdata
from Network import Iris

if __name__ == '__main__':
    data_root = './Dataset/'
    save_root = './Model/'
    print("Loading dataset...")
    Test = 'test'
    test_data = Irisdata(data_root,Pattern=Test)
    batch_size_test = 1
    num_workers = 1
    # batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)

    test_dataloader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True, num_workers=num_workers)
    print('train dataset len: {}'.format(len(test_dataloader.dataset)))

    IrisNet = torch.load(save_root+'Iris_model.pth')

    for Data in test_dataloader:
        datas, labels = Data
        datas = datas.float()
        labels = labels.long()
        out = IrisNet(datas)
        print(out)
        out = torch.max(out,1)
        print('Predicted Value:{} ,actual value {}'.format(out[1],labels))