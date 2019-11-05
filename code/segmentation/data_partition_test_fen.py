from __future__ import print_function
from glob import glob
import torch.utils.data as data
import os
import cv2
from PIL import Image
#from utils import preprocess
import random
import numpy as np
import pandas as pd

save_root1='/home/cb/code/zidonjiashi/dataset/qiege/train1/src/'
save_root2='/home/cb/code/zidonjiashi/dataset/qiege/train1/label/'
save_root3='/home/cb/code/zidonjiashi/dataset/qiege/val1/src/'
save_root4='/home/cb/code/zidonjiashi/dataset/qiege/val1/label/'

CLASSES = ['background', 'tobacco', 'cron', 'barley-rice','building']#10类
#数据路径
imgpaths = sorted(glob('/home/cb/code/zidonjiashi/dataset/qiege/src' + '/*.jpg'))
maskpaths = sorted(glob('/home/cb/code/zidonjiashi/dataset/qiege/label_yuan' + '/*.png'))
    
print(len(imgpaths),len(maskpaths))
    
#获取四类label图像及mask路径
s1=[]
s2=[]
s3=[]
s4=[]
s5=[]
s6=[]
s7=[]
s8=[]
s9=[]
s0=[]
im1=[]
im2=[]
im3=[]
im4=[]
im5=[]
im6=[]
im7=[]
im8=[]
im9=[]
im0=[]
print("load dataset....")
for i in range(len(maskpaths)):
    img=cv2.imread(maskpaths[i])
    if img.max()==34:
        s9.append(maskpaths[i])
        im9.append(imgpaths[i])
    if img.max()==33:
        s8.append(maskpaths[i])
        im8.append(imgpaths[i])
    if img.max()==32:
        s7.append(maskpaths[i])
        im7.append(imgpaths[i])
    if img.max()==31:
        s6.append(maskpaths[i])
        im6.append(imgpaths[i])
    if img.max()==30:
        s5.append(maskpaths[i])
        im5.append(imgpaths[i])
    if img.max()==29:
        s4.append(maskpaths[i])
        im4.append(imgpaths[i])
    if img.max()==28:
        s3.append(maskpaths[i])
        im3.append(imgpaths[i])
    if img.max()==27:
        s2.append(maskpaths[i])
        im2.append(imgpaths[i])
    if img.max()==26:
        s1.append(maskpaths[i])
        im1.append(imgpaths[i])
    if img.max()==0:
        s0.append(maskpaths[i])
        im0.append(imgpaths[i])
    print(i+1,6987)
#打乱四类label图像路径
print(len(im0),0)
print(len(im1),1)
print(len(im2),2)
print(len(im3),3)
print(len(im4),4)
print(len(im5),5)
print(len(im6),6)
print(len(im7),7)
print(len(im8),8)
print(len(im9),9)

random_train = list(range(len(im0)))
random.seed(105)
random.shuffle(random_train)
im00 = [im0[idx] for idx in random_train]
s00 =[s0[idx] for idx in random_train]

random_train = list(range(len(im1)))
random.seed(105)
random.shuffle(random_train)
im11 = [im1[idx] for idx in random_train]
s11 =[s1[idx] for idx in random_train]
    
random_train = list(range(len(im2)))
random.seed(105)
random.shuffle(random_train)
im22 =[im2[idx] for idx in random_train]
s22 = [s2[idx] for idx in random_train]
    
random_train = list(range(len(im3)))
random.seed(105)
random.shuffle(random_train)
im33 = [im3[idx] for idx in random_train]
s33 = [s3[idx] for idx in random_train]
    
random_train = list(range(len(im4)))
random.seed(105)
random.shuffle(random_train)
im44 = [im4[idx] for idx in random_train]
s44 = [s4[idx] for idx in random_train]

random_train = list(range(len(im5)))
random.seed(105)
random.shuffle(random_train)
im55 = [im5[idx] for idx in random_train]
s55 =[s5[idx] for idx in random_train]

random_train = list(range(len(im6)))
random.seed(105)
random.shuffle(random_train)
im66 = [im6[idx] for idx in random_train]
s66 =[s6[idx] for idx in random_train]
    
random_train = list(range(len(im7)))
random.seed(105)
random.shuffle(random_train)
im77 =[im7[idx] for idx in random_train]
s77 = [s7[idx] for idx in random_train]
    
random_train = list(range(len(im8)))
random.seed(105)
random.shuffle(random_train)
im88 = [im8[idx] for idx in random_train]
s88 = [s8[idx] for idx in random_train]
    
random_train = list(range(len(im9)))
random.seed(105)
random.shuffle(random_train)
im99 = [im9[idx] for idx in random_train]
s99 = [s9[idx] for idx in random_train]
    
#获取训练和验证集
train_ratio = 0.85
    
train_image=im11[:int(len(im11)*train_ratio)]+im22[:int(len(im22)*train_ratio)]+im33[:int(len(im33)*train_ratio)]+im44[:int(len(im44)*train_ratio)]+im55[:int(len(im55)*train_ratio)]+im66[:int(len(im66)*train_ratio)]+im77[:int(len(im77)*train_ratio)]+im88[:int(len(im88)*train_ratio)]+im99[:int(len(im99)*train_ratio)]
train_mask=s11[:int(len(im11)*train_ratio)]+s22[:int(len(im22)*train_ratio)]+s33[:int(len(im33)*train_ratio)]+s44[:int(len(im44)*train_ratio)]+s55[:int(len(im55)*train_ratio)]+s66[:int(len(im66)*train_ratio)]+s77[:int(len(im77)*train_ratio)]+s88[:int(len(im88)*train_ratio)]+s99[:int(len(im99)*train_ratio)]
test_image=im11[int(len(im11)*train_ratio):]+im22[int(len(im22)*train_ratio):]+im33[int(len(im33)*train_ratio):]+im44[int(len(im44)*train_ratio):]+im55[int(len(im55)*train_ratio):]+im66[int(len(im66)*train_ratio):]+im77[int(len(im77)*train_ratio):]+im88[int(len(im88)*train_ratio):]+im99[int(len(im99)*train_ratio):]
test_mask=s11[int(len(im11)*train_ratio):]+s22[int(len(im22)*train_ratio):]+s33[int(len(im33)*train_ratio):]+s44[int(len(im44)*train_ratio):]+s55[int(len(im55)*train_ratio):]+s66[int(len(im66)*train_ratio):]+s77[int(len(im77)*train_ratio):]+s88[int(len(im88)*train_ratio):]+s99[int(len(im99)*train_ratio):]
print(len(train_image))
print(len(test_mask))
    
for i in range(int(len(train_image))):
    _img = Image.open(train_image[i]).convert('RGB')
    _target = Image.open(train_mask[i]).convert('L')
    img_name = '{0}'.format(i)
    label_name = '{0}'.format(i)
    _img.save(save_root1 +img_name+'.jpg')
    _target.save(save_root2 +label_name+ '.png')
print(1)
for i in range(int(len(test_mask))):
    _img = Image.open(test_image[i]).convert('RGB')
    _target = Image.open(test_mask[i]).convert('L')
    img_name = '{0}'.format(i)
    label_name = '{0}'.format(i)
    _img.save(save_root3 +img_name+ '.jpg')
    _target.save(save_root4 +label_name+ '.png')
print(2)
count = 0
for root,dirs,files in os.walk(save_root1):    #遍历统计
    for each in files:
        count += 1  #统计文件夹下文件个数
print (count)              #输出结果  


count = 0
for root,dirs,files in os.walk(save_root3):    #遍历统计
    for each in files:
        count += 1  #统计文件夹下文件个数
print (count)              #输出结果
