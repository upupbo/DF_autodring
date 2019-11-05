目标检测部分----baseline

数据预处理部分(附VOC,COCO数据集链接)

1.1赛题数据转VOC格式 and VOC标签转COCO标签

现在做人工智能比赛，一般都是需要预处理数据，并把数据转换成COCO格式或者VOC格式，接下来给大家分享如何将本赛题的检测数据转为VOC格式。

下图为此次竞赛给出的数据格式：

![img](file:///C:/Users/CEO/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

train.txt中的标签数据的具体说明如下：

![img](file:///C:/Users/CEO/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

如果检测目标有多个时，后面会用空格间隔，继续X1,Y1,X2,Y2 ,type，confidence这种格式。而type包含的类别编号对应的类别名称如下表所示，background 背景为0。

第一步：将train.txt按图像分成多个txt文件

Fen_hang.py:

f = open("train.txt","r")

lines = f.readlines()

for line in lines:

  line

  txt = line[0:6]

  \#print(txt)

  f=txt+'.txt'

  file = open(f, 'w')

  \#print(f)

  line=line[22:]

  \#print(line)

  for db in line.split():

​    \#print(db)

​    file.write(db[:-2])

​    file.write('\n')

效果如下：

![img](file:///C:/Users/CEO/AppData/Local/Temp/msohtmlclip1/01/clip_image005.png)

![img](file:///C:/Users/CEO/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)

第二步：将txt文件转换为xmlVOC类标签文件

Txt_xml.py:

\#! /usr/bin/python

\# -*- coding:UTF-8 -*-

import os, sys

import glob

from PIL import Image

 

\# VEDAI 图像存储位置

src_img_dir = "/home/cb/code/zidonjiashi/dataset/test/"#图片存放地址

\# VEDAI 图像的 ground truth 的 txt 文件存放位置

src_txt_dir = "/home/cb/code/zidonjiashi/txt_to_coco/weibiaoqian_test/"#txt存放地址

src_xml_dir = "/home/cb/code/zidonjiashi/txt_to_coco/anno1/"#生成xml存放地址

 

img_Lists = glob.glob(src_img_dir + '/*.jpg')

 

img_basenames = [] # e.g. 100.jpg

for item in img_Lists:

  img_basenames.append(os.path.basename(item))

 

img_names = [] # e.g. 100

for item in img_basenames:

  temp1, temp2 = os.path.splitext(item)

  img_names.append(temp1)

 

for img in img_names:

  im = Image.open((src_img_dir + '/' + img + '.jpg'))

  width, height = im.size

 

  \# open the crospronding txt file

  gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()

  \#gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

 

  \# write in xml file

  \#os.mknod(src_xml_dir + '/' + img + '.xml')

  xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')

  xml_file.write('<annotation>\n')

  xml_file.write('  <folder>VOC2007</folder>\n')

  xml_file.write('  <filename>' + str(img) + '.jpg' + '</filename>\n')

  xml_file.write('  <size>\n')

  xml_file.write('    <width>' + str(width) + '</width>\n')

  xml_file.write('     <height>' + str(height) + '</height>\n')

  xml_file.write('    <depth>3</depth>\n')

  xml_file.write('  </size>\n')

 

  \# write the region of image on xml file

  for img_each_label in gt:

​    spt = img_each_label.split(',') #这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。

​    xml_file.write('  <object>\n')

​    xml_file.write('    <name>' + str(spt[4]) + '</name>\n')

​    xml_file.write('    <pose>Unspecified</pose>\n')

​    xml_file.write('    <truncated>0</truncated>\n')

​    xml_file.write('    <difficult>0</difficult>\n')

​    xml_file.write('    <bndbox>\n')

​    xml_file.write('      <xmin>' + str(spt[0]) + '</xmin>\n')

​    xml_file.write('      <ymin>' + str(spt[1]) + '</ymin>\n')

​    xml_file.write('      <xmax>' + str(spt[2]) + '</xmax>\n')

​    xml_file.write('      <ymax>' + str(spt[3]) + '</ymax>\n')

​    xml_file.write('    </bndbox>\n')

​    xml_file.write('  </object>\n')

 

xml_file.write('</annotation>')

 

至此，生成xml标签文件如下，再按如下顺序存放，便成了VOC数据集。

![img](file:///C:/Users/CEO/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

![img](file:///C:/Users/CEO/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

![img](file:///C:/Users/CEO/AppData/Local/Temp/msohtmlclip1/01/clip_image011.png)

 

第三步：VOC标签转COCO标签

Voccoco.py:

import xml.etree.ElementTree as ET

import os

import json

 

coco = dict()

coco['images'] = []

coco['type'] = 'instances'

coco['annotations'] = []

coco['categories'] = []

 

category_set = dict()

image_set = set()

 

category_item_id = -1

image_id = 7000

annotation_id = 0

 

 

def addCatItem(name):

  global category_item_id

  category_item = dict()

  category_item['supercategory'] = 'none'

  category_item_id += 1

  category_item['id'] = category_item_id

  category_item['name'] = name

  coco['categories'].append(category_item)

  category_set[name] = category_item_id

  return category_item_id

 

 

def addImgItem(file_name, size):

  global image_id

  if file_name is None:

​    raise Exception('Could not find filename tag in xml file.')

  if size['width'] is None:

​    raise Exception('Could not find width tag in xml file.')

  if size['height'] is None:

​    raise Exception('Could not find height tag in xml file.')

  image_id += 1

  image_item = dict()

  image_item['id'] = image_id

  image_item['file_name'] = file_name

  image_item['width'] = size['width']

  image_item['height'] = size['height']

  coco['images'].append(image_item)

  image_set.add(file_name)

  return image_id

 

 

def addAnnoItem(object_name, image_id, category_id, bbox):

  global annotation_id

  annotation_item = dict()

  annotation_item['segmentation'] = []

  seg = []

  \# bbox[] is x,y,w,h

  \# left_top

  seg.append(bbox[0])

  seg.append(bbox[1])

  \# left_bottom

  seg.append(bbox[0])

  seg.append(bbox[1] + bbox[3])

  \# right_bottom

  seg.append(bbox[0] + bbox[2])

  seg.append(bbox[1] + bbox[3])

  \# right_top

  seg.append(bbox[0] + bbox[2])

  seg.append(bbox[1])

 

  annotation_item['segmentation'].append(seg)

 

  annotation_item['area'] = bbox[2] * bbox[3]

  annotation_item['iscrowd'] = 0

  annotation_item['ignore'] = 0

  annotation_item['image_id'] = image_id

  annotation_item['bbox'] = bbox

  annotation_item['category_id'] = category_id

  annotation_id += 1

  annotation_item['id'] = annotation_id

  coco['annotations'].append(annotation_item)

 

 

def parseXmlFiles(xml_path):

  n=0

  for f in os.listdir(xml_path):

​    if not f.endswith('.xml'):

​      continue

 

​    bndbox = dict()

​    size = dict()

​    current_image_id = None

​    current_category_id = None

​    file_name = None

​    size['width'] = None

​    size['height'] = None

​    size['depth'] = None

 

​    xml_file = os.path.join(xml_path, f)

​    print(xml_file)

​    print(n)

​    n=n+1;

 

​    tree = ET.parse(xml_file)

​    root = tree.getroot()

​    if root.tag != 'annotation':

​      raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

 

​    \# elem is <folder>, <filename>, <size>, <object>

​    for elem in root:

​      current_parent = elem.tag

​       current_sub = None

​      object_name = None

 

​      if elem.tag == 'folder':

​        continue

 

​      if elem.tag == 'filename':

​        file_name = elem.text

​        if file_name in category_set:

​          raise Exception('file_name duplicated')

 

​      \# add img item only after parse <size> tag

​      elif current_image_id is None and file_name is not None and size['width'] is not None:

​        if file_name not in image_set:

​          current_image_id = addImgItem(file_name, size)

​          print('add image with {} and {}'.format(file_name, size))

​        else:

​          raise Exception('duplicated image: {}'.format(file_name))

​           \# subelem is <width>, <height>, <depth>, <name>, <bndbox>

​      for subelem in elem:

​        bndbox['xmin'] = None

​        bndbox['xmax'] = None

​        bndbox['ymin'] = None

​        bndbox['ymax'] = None

 

​        current_sub = subelem.tag

​        if current_parent == 'object' and subelem.tag == 'name':

​          object_name = subelem.text

​          if object_name not in category_set:

​            current_category_id = addCatItem(object_name)

​          else:

​            current_category_id = category_set[object_name]

 

​        elif current_parent == 'size':

​          if size[subelem.tag] is not None:

​            raise Exception('xml structure broken at size tag.')

​          size[subelem.tag] = int(subelem.text)

 

​        \# option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>

​        for option in subelem:

​          if current_sub == 'bndbox':

​            if bndbox[option.tag] is not None:

​              raise Exception('xml structure corrupted at bndbox tag.')

​            bndbox[option.tag] = int(option.text)

 

​        \# only after parse the <object> tag

​        if bndbox['xmin'] is not None:

​          if object_name is None:

​            raise Exception('xml structure broken at bndbox tag')

​          if current_image_id is None:

​            raise Exception('xml structure broken at bndbox tag')

​          if current_category_id is None:

​            raise Exception('xml structure broken at bndbox tag')

​          bbox = []

​           \# x

​          bbox.append(bndbox['xmin'])

​          \# y

​          bbox.append(bndbox['ymin'])

​          \# w

​          bbox.append(bndbox['xmax'] - bndbox['xmin'])

​          \# h

​          bbox.append(bndbox['ymax'] - bndbox['ymin'])

​          print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,

​                                  bbox))

​          addAnnoItem(object_name, current_image_id, current_category_id, bbox)

 

 

if __name__ == '__main__':

  xml_path = '/home/cb/code/zidonjiashi/txt_to_coco/anno1/'  # 这是xml文件所在的地址

  json_file = '/home/cb/code/zidonjiashi/xml_to_json/wei_test/test.json'                   # 这是你要生成的json文件            

  parseXmlFiles(xml_path)                    # 只需要改动这两个参数就行了

  json.dump(coco, open(json_file, 'w'))

 

 

VOC,COCO数据集百度云链接：

链接：https://pan.baidu.com/s/1uwp_xlJNi1dXQ8SZbTEfig 

提取码：i6da 

 