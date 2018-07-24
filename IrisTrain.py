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
    Train = 'train'
    Val = 'val'
    train_data = Irisdata(data_root,Pattern='train')
    val_data = Irisdata(data_root,Pattern='val')
    batch_size_train = 130
    batch_size_val = 10
    num_workers = 1
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-2
    num_epoches = 1000
    # batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)


    train_dataloader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    print('train dataset len: {}'.format(len(train_dataloader.dataset)))

    val_dataloader = DataLoader(val_data, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)
    print('val dataset len: {}'.format(len(val_dataloader.dataset)))

    # 输出数据格式
    # for batch_datas,batch_labels in train_dataloader:
    # print(batch_datas.size(),batch_labels.size())

    IrisNet = Iris(Factor=4,Classification=3)

    optimizer = optim.SGD(IrisNet.parameters(),lr=learning_rate)


    # input = Variable(torch.randn(5,3))
    # target = Variable(torch.FloatTensor(5).random_(3))
    # target = target.long()
    # out = criterion(input,target)
    # print(format(input))
    # print(format(target))
    # print(format(out))
    # print(format(input.size()))
    # print(format(target.size()))
    #

    for epoch in range(num_epoches):
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0
        for i,Data in enumerate(train_dataloader,1):
            datas,labels = Data
            datas = datas.float()
            labels = labels.long()
            datas = Variable(datas, volatile=True)
            labels = Variable(labels, volatile=True)
            #前向传播 Forward propagation
            out = IrisNet(datas)
            # print(format(out.size()))
            # print(format(labels.size()))

            #运算Loss
            loss = criterion(out,labels)   #运算Loss


            #反向传播 Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Loss:{}'.format(loss))



        IrisNet.eval()


        for Data in val_dataloader:
            datas,labels = Data
            datas = datas.float()
            labels = labels.long()
            datas = Variable(datas, volatile=True)
            labels = Variable(labels, volatile=True)
            #前向传播 Forward propagation
            out = IrisNet(datas)
            # print(format(out.size()))
            # print(format(labels.size()))

            #运算Loss
            loss = criterion(out,labels)   #运算Loss
            print('Val Loss:{}'.format(loss))

    torch.save(IrisNet, save_root+'Iris_model.pth')