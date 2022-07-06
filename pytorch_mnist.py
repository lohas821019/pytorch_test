# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:31:02 2022

@author: Jason
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


EPOCH = 10                #全部data訓練10次
BATCH_SIZE = 50           #每次訓練隨機丟50張圖像進去
LR =0.001                 #learning rate
DOWNLOAD_MNIST = False    #第一次用要先下載data,所以是True
if_use_gpu = 1            #使用gpu


train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(), 
    #把灰階從0~255壓縮到0~1
    download= True
)

#看size
print(train_data.train_data.size())
print(train_data.train_labels.size())

plt.imshow(train_data.train_data[2].numpy(),cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()


train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True)
#shuffle是隨機從data裡讀去資料.

test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
    )

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).float(), requires_grad=False)
#requires_grad=False 不參與反向傳播,test data 不用做
test_y = test_data.test_labels

#%%

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(           
                in_channels=1,  
                out_channels=16, 
                kernel_size=5,   
                stride=1,        
                padding=2),
            nn.ReLU(),nn.MaxPool2d(kernel_size = 2))
        #以上為一層conv + ReLu + maxpool
        
        #快速寫法：
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),  #(32,14,14)
            nn.ReLU(), 
            nn.MaxPool2d(2)   #(32,7,7)
        )
        
        self.out = nn.Linear(32*7*7, 10) #10=0~9
       
    def forward(self,x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = x.view(x.size(0), -1)
       output = self.out(x)
       return output
       
cnn = CNN()
if if_use_gpu:
    cnn = cnn.cuda()

