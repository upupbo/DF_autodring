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
#path_list=['000000.jpg']

for i,filename in enumerate(path_list):
    src = srcpath + filename[:-4] + '.jpg'
    label = labelpath + filename[:-4] + '.png'
    #img = Image.open(src).convert('RGB')
    #mask = Image.open(label)
    img_1=cv2.imread(src)
    img_1_mask=cv2.imread(label)
    #img_1 = np.array(img)
    #img_1_mask = np.array(mask)
    #h,w = img_1_mask.shape
    h=720
    w=1280
    #print(h,w)
    step=300
    outsize=512
    cx=0
    cy=0
    while cy+outsize < h:
        cx=0
        while cx+outsize < w:
            img_s=img_1[cy:(outsize+cy),cx:(outsize+cx)]
            img_m=img_1_mask[cy:(outsize+cy),cx:(outsize+cx)]
            im_name='/home/cb/code/zidonjiashi/dataset/qiege/src/'+filename[:-4]+'_{}_{}.jpg'.format(cx,cy)
            im_name1='/home/cb/code/zidonjiashi/dataset/qiege/label_yuan/'+filename[:-4]+'_{}_{}.png'.format(cx,cy)
            cv2.imwrite(im_name,img_s)
            cv2.imwrite(im_name1,img_m)
            #img=Image.fromarray(img)
            #img.save('/home/cb/code/zidonjiashi/dataset/qiege/src/'+filename[:-4]+'_{}_{}.jpg'.format(cx,cy))
            #img1=dslb.ReadAsArray(cx,cy,outsize+cx,outsize+cy)
            #img1=Image.fromarray(img1).convert('L')
            #img1.save('/home/cb/code/zidonjiashi/dataset/qiege/label_yuan/'+filename[:-4]+'_{}_{}.png'.format(cx,cy))
            cx+=step
        cy+=step
    print('final',i)


