from PIL import Image,ImageFilter,ImageDraw,ImageEnhance
import random
import os
import numpy as np
from tqdm import tqdm
import cv2
from libtiff import TIFF
import scipy.misc
from scipy import misc

#要裁剪图像的大小
img_w = 256  
img_h = 256  
#读取路径下图片的名称
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:   
            img_name = os.path.split(file)[1]    
            L.append(img_name)

    return L  
image_sets=file_name('/home/dataset/rssrai2019_semantic/val/src/');#图片存贮路径
#色调增强
def random_color(img):
    img = ImageEnhance.Color(img)
    img = img.enhance(2)
    return img
def data_augment(src_roi,label_roi):
    #图像和标签同时进行90，180，270旋转
    if np.random.random() < 0.25:
       src_roi=src_roi.rotate(90)
       label_roi=label_roi.rotate(90)
    if np.random.random() < 0.25:
       src_roi=src_roi.rotate(180)
       label_roi=label_roi.rotate(180)
    if np.random.random() < 0.25:
       src_roi=src_roi.rotate(270)
       label_roi=label_roi.rotate(270)
    #图像和标签同时进行竖直旋转   
    if np.random.random() < 0.25:
       src_roi=src_roi.transpose(Image.FLIP_LEFT_RIGHT)
       label_roi=label_roi.transpose(Image.FLIP_LEFT_RIGHT)
    #图像和标签同时进行水平旋转
    if np.random.random() < 0.25:
       src_roi=src_roi.transpose(Image.FLIP_TOP_BOTTOM)
       label_roi=label_roi.transpose(Image.FLIP_TOP_BOTTOM)
    #图像进行色调增强
    if np.random.random() < 0.25:
       src_roi=random_color(src_roi)
    return src_roi,label_roi
    
# image_num：增广之后的图片数据
def creat_dataset(image_num = 30000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = TIFF.open('/home/dataset/rssrai2019_semantic/val/src/' + image_sets[i])  # 4 channels
        src_img = src_img.read_image()
        label_img = TIFF.open('/home/dataset/rssrai2019_semantic/val/label/' + image_sets[i]) # 4 channels
        label_img = label_img.read_image()
        

        #对图像进行随机裁剪，这里大小为256*256       
        while count < image_each:
            src_img = np.array(src_img)
            label_img = np.array(label_img)
            width0 = src_img.shape[0]
            height0 = src_img.shape[1]
            width1 = random.randint(0, width0 - img_w )
            height1 = random.randint(0, height0 - img_h)
            width2 = width1 + img_w
            height2 = height1 + img_h  
            
            
            label_img=Image.fromarray(label_img)
            src_img=Image.fromarray(src_img)

            src_roi=src_img.crop((width1, height1, width2, height2))
            label_roi=label_img.crop((width1, height1, width2, height2))
                       
            if mode == 'augment':
               src_roi,label_roi = data_augment(src_roi,label_roi)
            #scipy.misc.toimage(src_roi).save('/home/dataset/rssrai2019_semantic/train/train_src/%d.tif' % g_count)
            #scipy.misc.toimage(label_roi).save('/home/dataset/rssrai2019_semantic/train/train_label/%d.tif' % g_count)
            scipy.misc.imsave('/home/dataset/rssrai2019_semantic/val/train_src/%d.tif' % g_count,src_roi)
            scipy.misc.imsave('/home/dataset/rssrai2019_semantic/val/train_label/%d.tif' % g_count,label_roi)
            count += 1 
            g_count += 1
            print(count)

if __name__=='__main__':  
    creat_dataset(mode='augment')
