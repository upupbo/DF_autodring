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
#torch.cuda.set_device(1)



use_cuda=True
model=tc.load('/home/cb/code/zidonjiashi/segmentation/deeplabv3+/model/model8')  #torch.save(model,'./tmp/model{}'.format(epoch))
#device = tc.device("cuda" if use_cuda else "cpu")
model=model.cuda()
model.eval()
ds=FarmDataset(istrain=False)

def createres(d,outputname):
    #创建一个和ds大小相同的灰度图像BMP
    driver = gdal.GetDriverByName("BMP")
    #driver=ds.GetDriver()
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
    r=tc.argmax(pr,0)  #512*512
    r=r.byte().cpu().numpy()
    return r
    
def predict(d,outputname='tmp.bmp'):
    wx=d.RasterXSize   #width
    wy=d.RasterYSize   #height
    print(wx,wy)
    od=data=np.zeros((wy,wx),np.uint8)
    #od=createres(d,outputname=outputname)
    #ob=od.GetRasterBand(1) #得到第一个channnel
    blocksize=512
    step=256
    for cy in range(step,wy,step):
        for cx in range(step,wx,step):
            if (cx+step>wx and cy+step<=wy):  
                img=d.ReadAsArray(wx-blocksize,cy-step,blocksize,blocksize)[0:3,:,:] #channel*h*w
                if (img[0].sum()==0): continue
                r=modelpredict(img)
                cx=wx-step
            if (cx+step<=wx and cy+step>wy):
                img=d.ReadAsArray(cx-step,wy-blocksize,blocksize,blocksize)[0:3,:,:] #channel*h*w
                if (img[0].sum()==0): continue
                r=modelpredict(img)
                cy=wy-step
            if (cx+step>wx and cy+step>wy):
                img=d.ReadAsArray(wx-blocksize,wy-blocksize,blocksize,blocksize)[0:3,:,:] #channel*h*w
                if (img[0].sum()==0): continue
                r=modelpredict(img)
                cx=wx-step
                cy=wy-step
                print(r.shape)
            if (cx+step<=wx and cy+step<=wy):
                img=d.ReadAsArray(cx-step,cy-step,blocksize,blocksize)[0:3,:,:] #channel*h*w
                if (img[0].sum()==0): continue
                r=modelpredict(img)
            if cy==step or cx==step:
                od[cy-step:cy+step,cx-step:cx+step]=r
            else:
                #od[cy-step//2:cy+step//2,cx-step//2:cx+step//2]=r[256:step+256,256:step+256]
                od[cy-step//2:cy+step,cx-step//2:cx+step]=r[128:,128:]
            print(cy,cx)

    #del od
    createpng(wy,wx,od,outputname)
    return 


for i,col in enumerate(ds):
	print("start predict.....",i)
	j=7000 + i;
	#predict(col[i],'image_3_predict.png')
	predict(col,'{}.png'.format(j))
