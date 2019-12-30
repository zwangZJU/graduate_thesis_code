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

'''
coco data的数据格式为：class x y w h
左上角顶点为（0，0），水平向右为x轴正方向，竖直向下为y轴正方向，水平宽度为w，竖直高度为h
用PIL的Image读取图片的shape为row, col, channel
其中，calss为0-79的整数，x为bbox中心点横坐标值/col,y为bbox中心点纵坐标值/row，w为bbox的宽/col，h为bbox的高/row
'''
def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

filename = 'E:/Datasets/Caltech/Person/test/labels/set06_V000_I00269_usatest'

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 80)]
classes = ['person'] # Extracts class labels from file

img = np.array(Image.open(filename.replace('labels', 'images') + '.jpg'))

w, h, c = img.shape
print(w,h,c)
with open(filename+'.txt','r') as file:
    detections = []
    for i,line in enumerate(file):
        a = []
        for j, l in enumerate(line.split()):
            if j == 0:
                a.append(int(l))
            elif j%2==0:
                a.append(float(l)*w)
            else:
                a.append(float(l)*h)
        detection = a
        detections.append(detection)
    print(detections)
detections = np.array(detections)


plt.figure()
fig, ax = plt.subplots(1)
ax.imshow(img)


for cls_pred, x1, y1, x2, y2 in detections:

    color = colors[int(cls_pred)]
    # Create a Rectangle patch
    bbox = patches.Rectangle((x1-x2/2, y1-y2/2), x2, y2, linewidth=2,
                             edgecolor=color,
                             facecolor='none')
    # Add the bbox to the plot
    ax.add_patch(bbox)
    # Add label
    #plt.text(x1-x2/2, y1-y2/2, s=classes[int(cls_pred)], color='white', verticalalignment='top',bbox={'color': color, 'pad': 0})

# Save generated image with detections
plt.axis('off')
plt.gca().xaxis.set_major_locator(NullLocator())
plt.gca().yaxis.set_major_locator(NullLocator())
plt.savefig('show'+'.png', bbox_inches='tight', pad_inches=0.0)
plt.close()