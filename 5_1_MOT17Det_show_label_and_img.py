# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/2
@Desc  : 
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

filename = 'E:/Datasets/MOT17Det/train/MOT17-05/labels/000002'

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 80)]
classes = ['person'] # Extracts class labels from file

img = np.array(Image.open(filename.replace('labels', 'img1') + '.jpg'))

w, h, c = img.shape
print(w,h,c)
# with open(filename+'.txt','r') as file:
#     detections = []
#     for i,line in enumerate(file):
#         a = []
#         for j, l in enumerate(line.split()):
#             if j == 0:
#                 a.append(int(l))
#             elif j%2==0:
#                 a.append(float(l)*w)
#             else:
#                 a.append(float(l)*h)
#         detection = a
#         detections.append(detection)
#     print(detections)
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
import numpy as np
def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = np.clamp(coords[:, :4], min=0)
    return coords

plt.figure()
fig, ax = plt.subplots(1)
ax.imshow(img)
import cv2
import random
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

for cls, cx, cy, w, h in detections:

    plot_one_box([int(cx-w//2), int(cy-h//2), int(cx+w//2), int(cy+h//2)], img, label='person', color=colors[int(cls)])
cv2.imwrite('show.png',img)
# Save generated image with detections
# plt.axis('off')
# plt.gca().xaxis.set_major_locator(NullLocator())
# plt.gca().yaxis.set_major_locator(NullLocator())
# plt.savefig('show'+'.png', bbox_inches='tight', pad_inches=0.0)
# plt.close()