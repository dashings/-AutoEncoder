#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from BasicModule import BasicModule
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from sklearn.decomposition import PCA

writer = SummaryWriter('./runs/exp1')
t.manual_seed(1)

root = r'D:\data'

batch_size = 32
data_spilt = 50000  # 截取的数据集大小，以减小计算量
inshape = (10, 10)

trainData = datasets.MNIST(root, download=True, transform=transforms.Compose([
    transforms.RandomCrop(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
]))
trainData = t.utils.data.random_split(
    trainData, [data_spilt, len(trainData) - data_spilt])[0]

train_loader = t.utils.data.DataLoader(
    trainData, batch_size=batch_size, shuffle=True)


class SparseAutoEncoder(BasicModule):
    def __init__(self, BETA=1, ROU=0.01, hiddenshape=300, USE_P=False):
        super(SparseAutoEncoder, self).__init__()
        self.model_name = 'SparseAutoEncoder'
        self.BETA = BETA  # 稀疏项考虑程度
        self.ROU = ROU  # 稀疏度
        self.USE_P = USE_P
        self.inshape = inshape[0] * inshape[1]
        self.hiddenshape = hiddenshape
        self.encoder = nn.Linear(self.inshape, self.hiddenshape)
        self.decoder = nn.Linear(self.hiddenshape, self.inshape)

    def forward(self, x):
        encode = t.sigmoid(self.encoder(x))
        decode = t.sigmoid(self.decoder(encode))
        return encode, decode

    def display_hidden(self, index):
        with t.no_grad():
            paras = [each for name,
                              each in self.encoder.named_parameters()]  # w,b
            w = paras[0]
            num = w[index, :]
            den = ((w[index, :] ** 2).sum()) ** 0.5
            plt.imshow((num / den).view(inshape).numpy())
            plt.show()

    def cal_hidden(self):
        with t.no_grad():
            paras = [each for name,
                              each in self.encoder.named_parameters()]  # w,b
            w = paras[0]
            out = t.Tensor(w.shape[0], 1, inshape[0], inshape[1])
            for i in range(w.shape[0]):
                num = w[i, :]
                den = ((w[i, :] ** 2).sum()) ** 0.5
                out[i, 0] = (num / den).view(inshape)
            return out

    def predict(self, x, load=None):
        '''输入单张图片预测dataset元组(Tensor[1,28,28],class)
        '''
        self.eval()
        with t.no_grad():
            if load is not None:
                self.load(load)
            x, c = x
            x = x.view(1, -1)
            result = self(x)[1]
            result = result.detach()
            return x.view(inshape).numpy(), result.view(inshape).numpy()

    def trainNN(self, lr=1, weight_decay=1e-5, epochs=5):
        self.train()
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        batch_loss = 0
        for epoch in range(epochs):
            for i, (bx, by) in enumerate(train_loader):
                bx = bx.view(bx.shape[0], -1)
                optimizer.zero_grad()

                criterion = nn.MSELoss()  # mse损失
                encode, decode = self(bx)
                p_head = encode.sum(dim=0, keepdim=True) / encode.shape[0]
                p = t.ones(p_head.shape) * self.ROU
                penalty = (p * t.log(p / p_head) + (1 - p) *
                           t.log((1 - p) / (1 - p_head))).sum() / p.shape[1]
                if self.USE_P:
                    loss = criterion(decode, bx) + \
                           self.BETA * penalty
                else:
                    loss = criterion(decode, bx)

                loss.backward()
                batch_loss += loss.item()
                optimizer.step()

            if epoch % 1 == 0:
                print('batch loss={}@epoch={}'.format(batch_loss, epoch))
            batch_loss = 0


    def showim(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        plt.show()