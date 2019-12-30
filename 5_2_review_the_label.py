# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/2
@Desc  : 查看gt.txt是否有误
'''
# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/1
@Desc  : 
'''



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
import matplotlib.patches as patches
from PIL import Image
import random
import cv2

'''
coco data的数据格式为：class x y w h
左上角顶点为（0，0），水平向右为x轴正方向，竖直向下为y轴正方向，水平宽度为w，竖直高度为h
用PIL的Image读取图片的shape为row, col, channel
其中，calss为0-79的整数，x为bbox中心点横坐标值/col,y为bbox中心点纵坐标值/row，w为bbox的宽/col，h为bbox的高/row
'''
d= []
with open('E:/Datasets/MOT17Det/train/MOT17-05/gt/gt.txt') as f:
    for line in f.readlines():
        l = line.split(',')
        if l[0] == '1':
            x1, y1, w, h = map(int,l[2:6])
            d.append([x1, y1, w, h])
            print(x1, y1, w, h)
            print(line)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(100, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 80)]
classes = ['person']  # Extracts class labels from file

# img = np.array(Image.open(filename.replace('labels', 'img1') + '.jpg'))
img = cv2.imread('E:/Datasets/MOT17Det/train/MOT17-05/img1/000001.jpg')

w, h, c = img.shape
#d =[[1585, -1, 336, 578]]
d_3 = [[566,181,228,98]]
d_32 = [[-73,152,345,145]]
d_9 = [[325,-9,325,257]]
d_1 = [[259,204,19,47]]
for x, y, w, h in d_1:

    plot_one_box([x, y, x+w, y+h], img, label='0', color=colors[0])
cv2.imwrite('show.png',img)