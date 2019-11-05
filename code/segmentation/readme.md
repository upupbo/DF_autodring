语义分割部分----baseline（https://github.com/upupbo/DF_autodring）

2.1图像预处理---图像扩增

将图片切割成512*512大小的图片，最后数据集的数量为2万张。

import numpy as np

import glob as glob

import cv2

from PIL import Image

import os

 

Image.MAX_IMAGE_PIXELS = 100000000000000000000000000000000000000

 

srcpath='/home/cb/code/zidonjiashi/dataset/zong/src/'

labelpath='/home/cb/code/zidonjiashi/dataset/zong/label_yuan/'

 

path_list=os.listdir(srcpath)

path_list.sort()

\#path_list=['000000.jpg']

 

for i,filename in enumerate(path_list):

  src = srcpath + filename[:-4] + '.jpg'

  label = labelpath + filename[:-4] + '.png'

  \#img = Image.open(src).convert('RGB')

  \#mask = Image.open(label)

  img_1=cv2.imread(src)

  img_1_mask=cv2.imread(label)

  \#img_1 = np.array(img)

  \#img_1_mask = np.array(mask)

  \#h,w = img_1_mask.shape

  h=720

  w=1280

  \#print(h,w)

  step=300

  outsize=512

  cx=0

  cy=0

  while cy+outsize < h:

​    cx=0

​    while cx+outsize < w:

​      img_s=img_1[cy:(outsize+cy),cx:(outsize+cx)]

​      img_m=img_1_mask[cy:(outsize+cy),cx:(outsize+cx)]

​      im_name='/home/cb/code/zidonjiashi/dataset/qiege/src/'+filename[:-4]+'_{}_{}.jpg'.format(cx,cy)

​      im_name1='/home/cb/code/zidonjiashi/dataset/qiege/label_yuan/'+filename[:-4]+'_{}_{}.png'.format(cx,cy)

​      cv2.imwrite(im_name,img_s)

​      cv2.imwrite(im_name1,img_m)

​      \#img=Image.fromarray(img)

​      \#img.save('/home/cb/code/zidonjiashi/dataset/qiege/src/'+filename[:-4]+'_{}_{}.jpg'.format(cx,cy))

​      \#img1=dslb.ReadAsArray(cx,cy,outsize+cx,outsize+cy)

​      \#img1=Image.fromarray(img1).convert('L')

​      \#img1.save('/home/cb/code/zidonjiashi/dataset/qiege/label_yuan/'+filename[:-4]+'_{}_{}.png'.format(cx,cy))

​      cx+=step

​    cy+=step

  print('final',i)

 

 

2.2图像预处理---划分训练，验证集

此程序能保证类别均匀地划分训练集和验证集

from __future__ import print_function

from glob import glob

import torch.utils.data as data

import os

import cv2

from PIL import Image

\#from utils import preprocess

import random

import numpy as np

import pandas as pd

 

save_root1='/home/cb/code/zidonjiashi/dataset/qiege/train1/src/'

save_root2='/home/cb/code/zidonjiashi/dataset/qiege/train1/label/'

save_root3='/home/cb/code/zidonjiashi/dataset/qiege/val1/src/'

save_root4='/home/cb/code/zidonjiashi/dataset/qiege/val1/label/'

 

CLASSES = ['background', 'tobacco', 'cron', 'barley-rice','building']#10类

\#数据路径

imgpaths = sorted(glob('/home/cb/code/zidonjiashi/dataset/qiege/src' + '/*.jpg'))

maskpaths = sorted(glob('/home/cb/code/zidonjiashi/dataset/qiege/label_yuan' + '/*.png'))

  

print(len(imgpaths),len(maskpaths))

  

\#获取四类label图像及mask路径

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

​    s9.append(maskpaths[i])

​    im9.append(imgpaths[i])

  if img.max()==33:

​    s8.append(maskpaths[i])

​    im8.append(imgpaths[i])

  if img.max()==32:

​    s7.append(maskpaths[i])

​    im7.append(imgpaths[i])

  if img.max()==31:

​    s6.append(maskpaths[i])

​    im6.append(imgpaths[i])

  if img.max()==30:

​    s5.append(maskpaths[i])

​    im5.append(imgpaths[i])

  if img.max()==29:

​    s4.append(maskpaths[i])

​    im4.append(imgpaths[i])

  if img.max()==28:

​    s3.append(maskpaths[i])

​    im3.append(imgpaths[i])

  if img.max()==27:

​    s2.append(maskpaths[i])

​    im2.append(imgpaths[i])

  if img.max()==26:

​    s1.append(maskpaths[i])

​    im1.append(imgpaths[i])

  if img.max()==0:

​    s0.append(maskpaths[i])

​    im0.append(imgpaths[i])

  print(i+1,6987)

\#打乱四类label图像路径

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

  

\#获取训练和验证集

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

for root,dirs,files in os.walk(save_root1):  #遍历统计

  for each in files:

​    count += 1 #统计文件夹下文件个数

print (count)       #输出结果 

 

 

count = 0

for root,dirs,files in os.walk(save_root3):  #遍历统计

  for each in files:

​    count += 1 #统计文件夹下文件个数

print (count)       #输出结果

​            2.2图像预处理---图像增强

在训练前对图像进行旋转，色调增强，随机裁剪等操作。

from PIL import Image,ImageFilter,ImageDraw,ImageEnhance

import random

import os

import numpy as np

from tqdm import tqdm

import cv2

from libtiff import TIFF

import scipy.misc

from scipy import misc

 

\#要裁剪图像的大小

img_w = 256 

img_h = 256 

\#读取路径下图片的名称

def file_name(file_dir):  

  L=[]  

  for root, dirs, files in os.walk(file_dir): 

​    for file in files:  

​      img_name = os.path.split(file)[1]  

​      L.append(img_name)

 

  return L 

image_sets=file_name('/home/dataset/rssrai2019_semantic/val/src/');#图片存贮路径

\#色调增强

def random_color(img):

  img = ImageEnhance.Color(img)

  img = img.enhance(2)

  return img

def data_augment(src_roi,label_roi):

  \#图像和标签同时进行90，180，270旋转

  if np.random.random() < 0.25:

​    src_roi=src_roi.rotate(90)

​    label_roi=label_roi.rotate(90)

  if np.random.random() < 0.25:

​    src_roi=src_roi.rotate(180)

​    label_roi=label_roi.rotate(180)

  if np.random.random() < 0.25:

​    src_roi=src_roi.rotate(270)

​    label_roi=label_roi.rotate(270)

  \#图像和标签同时进行竖直旋转  

  if np.random.random() < 0.25:

​    src_roi=src_roi.transpose(Image.FLIP_LEFT_RIGHT)

​    label_roi=label_roi.transpose(Image.FLIP_LEFT_RIGHT)

  \#图像和标签同时进行水平旋转

  if np.random.random() < 0.25:

​    src_roi=src_roi.transpose(Image.FLIP_TOP_BOTTOM)

​    label_roi=label_roi.transpose(Image.FLIP_TOP_BOTTOM)

  \#图像进行色调增强

  if np.random.random() < 0.25:

​    src_roi=random_color(src_roi)

  return src_roi,label_roi

  

\# image_num：增广之后的图片数据

def creat_dataset(image_num = 30000, mode = 'original'):

  print('creating dataset...')

  image_each = image_num / len(image_sets)

  g_count = 0

  for i in tqdm(range(len(image_sets))):

​    count = 0

​    src_img = TIFF.open('/home/dataset/rssrai2019_semantic/val/src/' + image_sets[i]) # 4 channels

​    src_img = src_img.read_image()

​    label_img = TIFF.open('/home/dataset/rssrai2019_semantic/val/label/' + image_sets[i]) # 4 channels

​    label_img = label_img.read_image()

​    

 

​    \#对图像进行随机裁剪，这里大小为256*256    

​    while count < image_each:

​      src_img = np.array(src_img)

​      label_img = np.array(label_img)

​      width0 = src_img.shape[0]

​      height0 = src_img.shape[1]

​      width1 = random.randint(0, width0 - img_w )

​      height1 = random.randint(0, height0 - img_h)

​      width2 = width1 + img_w

​      height2 = height1 + img_h 

​      

​      

​      label_img=Image.fromarray(label_img)

​      src_img=Image.fromarray(src_img)

 

​      src_roi=src_img.crop((width1, height1, width2, height2))

​      label_roi=label_img.crop((width1, height1, width2, height2))

​            

​      if mode == 'augment':

​        src_roi,label_roi = data_augment(src_roi,label_roi)

​      \#scipy.misc.toimage(src_roi).save('/home/dataset/rssrai2019_semantic/train/train_src/%d.tif' % g_count)

​      \#scipy.misc.toimage(label_roi).save('/home/dataset/rssrai2019_semantic/train/train_label/%d.tif' % g_count)

​      scipy.misc.imsave('/home/dataset/rssrai2019_semantic/val/train_src/%d.tif' % g_count,src_roi)

​      scipy.misc.imsave('/home/dataset/rssrai2019_semantic/val/train_label/%d.tif' % g_count,label_roi)

​      count += 1 

​      g_count += 1

​      print(count)

 

if __name__=='__main__': 

  creat_dataset(mode='augment')

 

​            2.3图像预测技巧---膨胀预测

from farmdataset1 import FarmDataset

import torch as tc

from osgeo import gdal

from torchvision import transforms

import png

import numpy as np

import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CUDA_VISIBLE_DEVICES = 1

\#torch.cuda.set_device(1)

 

 

 

use_cuda=True

model=tc.load('/home/cb/code/zidonjiashi/segmentation/deeplabv3+/model/model8') #torch.save(model,'./tmp/model{}'.format(epoch))

\#device = tc.device("cuda" if use_cuda else "cpu")

model=model.cuda()

model.eval()

ds=FarmDataset(istrain=False)

 

def createres(d,outputname):

  \#创建一个和ds大小相同的灰度图像BMP

  driver = gdal.GetDriverByName("BMP")

  \#driver=ds.GetDriver()

  od=driver.Create('/home/cb/code/zidonjiashi/segmentation/deeplabv3+/predict/'+outputname,d.RasterXSize,d.RasterYSize,1)

  return od

 

def createpng(height,width,data,outputname):

  w=png.Writer(width,height,bitdepth=8,greyscale=True)

  of=open('/home/cb/code/zidonjiashi/segmentation/deeplabv3+/predict/'+outputname,'wb')

  w.write_array(of,data.flat)

  of.close()

  return 

  

def modelpredict(img): #imgnumpy array

  x=tc.from_numpy(img/255.0).float()

  x=x.unsqueeze(0).cuda()

  r=model.forward(x)

  pr=tc.softmax(r[0],dim=0)

  r=tc.argmax(pr,0) #512*512

   r=r.byte().cpu().numpy()

  return r

  

def predict(d,outputname='tmp.bmp'):

  wx=d.RasterXSize  #width

  wy=d.RasterYSize  #height

  print(wx,wy)

  od=data=np.zeros((wy,wx),np.uint8)

  \#od=createres(d,outputname=outputname)

  \#ob=od.GetRasterBand(1) #得到第一个channnel

  blocksize=512

  step=256

  for cy in range(step,wy,step):

​    for cx in range(step,wx,step):

​      if (cx+step>wx and cy+step<=wy): 

​        img=d.ReadAsArray(wx-blocksize,cy-step,blocksize,blocksize)[0:3,:,:] #channel*h*w

​        if (img[0].sum()==0): continue

​        r=modelpredict(img)

​        cx=wx-step

​      if (cx+step<=wx and cy+step>wy):

​        img=d.ReadAsArray(cx-step,wy-blocksize,blocksize,blocksize)[0:3,:,:] #channel*h*w

​        if (img[0].sum()==0): continue

​        r=modelpredict(img)

​        cy=wy-step

​      if (cx+step>wx and cy+step>wy):

​        img=d.ReadAsArray(wx-blocksize,wy-blocksize,blocksize,blocksize)[0:3,:,:] #channel*h*w

​        if (img[0].sum()==0): continue

​        r=modelpredict(img)

​        cx=wx-step

​        cy=wy-step

​        print(r.shape)

​      if (cx+step<=wx and cy+step<=wy):

​         img=d.ReadAsArray(cx-step,cy-step,blocksize,blocksize)[0:3,:,:] #channel*h*w

​        if (img[0].sum()==0): continue

​        r=modelpredict(img)

​      if cy==step or cx==step:

​        od[cy-step:cy+step,cx-step:cx+step]=r

​      else:

​        \#od[cy-step//2:cy+step//2,cx-step//2:cx+step//2]=r[256:step+256,256:step+256]

​        od[cy-step//2:cy+step,cx-step//2:cx+step]=r[128:,128:]

​      print(cy,cx)

 

  \#del od

  createpng(wy,wx,od,outputname)

  return 

 

 

for i,col in enumerate(ds):

  print("start predict.....",i)

  j=7000 + i;

  \#predict(col[i],'image_3_predict.png')

  predict(col,'{}.png'.format(j))

 

​            2.4图像融合---模型融合

 

import os

import cv2

import torch

import random

from PIL import Image

import numpy as np

from utils.img_utils import random_scale, random_mirror_1, random_mirror_0, random_rotation, normalize, \

  generate_random_crop_pos, random_crop_pad_to_shape, random_gaussian_blur

 

 

def vote_fusion():

 

  data_path = '../MY/AgriculturalBrainAIChallenge/Datasets/vote'

  predict_3_1 = os.path.join(data_path, '1/image_3_predict.png') # cb 0.723

  predict_3_2 = os.path.join(data_path, '2/image_3_predict.png') # 2825 0.7277 resnet101 lovasz0.1

  predict_3_3 = os.path.join(data_path, '3/image_3_predict.png') # fusion 0.7250

 

  predict_4_1 = os.path.join(data_path, '1/image_4_predict.png')

  predict_4_2 = os.path.join(data_path, '2/image_4_predict.png')

  predict_4_3 = os.path.join(data_path, '3/image_4_predict.png')

 

  predict_3 = [predict_3_1, predict_3_2, predict_3_3]

  predict_4 = [predict_4_1, predict_4_2, predict_4_3]

  predict_ = [predict_3, predict_4]

 

  predict_path = '../MY/AgriculturalBrainAIChallenge/Datasets/predict/'

  predict = [os.path.join(predict_path, 'image_3_predict.png'), os.path.join(predict_path, 'image_4_predict.png')]

 

  for i in range(len(predict)):

​    print('the', i + 1, 'image')

​    result_list = []

​    for j in range(len(predict_[i])):

​      im = cv2.imread(predict_[i][j], 0)

​      result_list.append(im)

​    print('fusion number:', len(result_list))

​    \# each pixel

​    height, width = result_list[0].shape

​    vote_mask = np.zeros((height, width))

​    for h in range(height):

​      for w in range(width):

​        record = np.zeros((1, 5))

​        for n in range(len(result_list)):

​          mask = result_list[n]

​          pixel = mask[h, w]

​          \# print('pix:',pixel)

​          record[0, pixel] += 1

 

​        label = record.argmax()

​        \# print(label)

​        vote_mask[h, w] = label

 

​    cv2.imwrite(predict[i], vote_mask)

​    print('Write', predict[i])

 

 

 

if __name__ == '__main__':

  vote_fusion()