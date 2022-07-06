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

if_use_gpu = 1            #使用gpu

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
       
model = CNN()
if if_use_gpu:
    model = model.cuda()

#%%

train_data_set = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(), 
    #把灰階從0~255壓縮到0~1
    download= False
)

#看size
print(train_data_set.train_data.size())
print(train_data_set.train_labels.size())


BATCH_SIZE = 50 

train_loader = Data.DataLoader(dataset = train_data_set, batch_size = BATCH_SIZE, shuffle=True)
#shuffle是隨機從data裡讀去資料.

test_data_set = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
    )

test_x = Variable(torch.unsqueeze(test_data_set.test_data, dim=1).float(), requires_grad=False)
#requires_grad=False 不參與反向傳播,test data 不用做
test_y = test_data_set.test_labels


#%%

LR =0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

EPOCH = 10                #全部data訓練10次
                 #learning rate
DOWNLOAD_MNIST = False    #第一次用要先下載data,所以是True

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
#決定跑幾個epoch,enumerate把load進來的data列出來成（x,y）

        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
#使用cuda加速        
        output = model(b_x)          #把data丟進網路中
        loss = loss_function(output, b_y)
        optimizer.zero_grad()      #計算loss,初始梯度
        loss.backward()            #反向傳播
        optimizer.step()       

        if step % 100 == 0:
            print('Epoch:', epoch, '|step:', step, '|train loss:%.4f'%loss.data)
        
        #每100steps輸出一次train loss
        
        
#%%

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pre_image(image_path,model):
   img = Image.open(image_path)
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([transforms.ToTensor(), 
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()  
      output =model(img_normalized)
     # print(output)
      index = output.data.cpu().numpy().argmax()
      classes = train_loader.dataset.classes
      class_name = classes[index]
      return class_name

