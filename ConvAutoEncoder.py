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
from torchvision.utils import make_grid, save_image

# 加噪声的SAE
t.manual_seed(1)

# root = ./AI/DataSets/MNIST'
root = r'D:\data'
batch_size = 32
data_spilt = 50000  # 截取的数据集大小，以减小计算量
inshape = (28, 28)

trainData = datasets.FashionMNIST(root, download=True,transform=transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5), (0.5))

]))
trainData = t.utils.data.random_split(
    trainData, [data_spilt, len(trainData) - data_spilt])[0]

train_loader = t.utils.data.DataLoader(
    trainData, batch_size=batch_size, shuffle=True)


class ConvAutoEncoder(BasicModule):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.model_name = 'ConvAutoEncoder'

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

    def predict(self, x, load=None):
        '''输入单张图片预测dataset元组(Tensor[1,28,28],class)
        '''
        self.eval()
        with t.no_grad():
            if load is not None:
                self.load(load)
            x, c = x
            x = t.unsqueeze(x, 0)  # (1,28,28) ->(1,1,28,28)
            result = self(x)[1]
            result = result.detach()
            return x.view(inshape).numpy(), result.view(inshape).numpy()

    def show(self):
        def showim(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
            plt.show()

        for i, (bx, by) in enumerate(train_loader):
            img_in = make_grid(bx, normalize=True)
            showim(img_in)
            self.eval()
            with t.no_grad():
                img_out = (self(bx)[1]).detach()
                img_out = make_grid(img_out, normalize=True)
                showim(img_out)
            break

    def trainNN(self, lr=1, weight_decay=1e-5, epochs=10):
        self.train()
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        batch_loss = 0
        for epoch in range(epochs):
            for i, (bx, by) in enumerate(train_loader):
                optimizer.zero_grad()

                criterion = nn.MSELoss()  # mse损失
                encode, decode = self(bx)

                loss = criterion(decode, bx)

                loss.backward()
                batch_loss += loss.item()
                optimizer.step()

            if epoch % 1 == 0:
                print('batch loss={}@epoch={}'.format(batch_loss, epoch))
            batch_loss = 0

