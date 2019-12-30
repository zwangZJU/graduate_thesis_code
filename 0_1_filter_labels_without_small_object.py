# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/10/9
@Desc  : 过滤标签中的小目标
'''



import cv2
import json

#

person = 0
head = 1

def transform(label_list_file, label_file, image_path, save_path):
    '''

    :param label_list_file: 存储所有的新label文件路径的文件
    :param label_file: 原始label的文件
    :param image_path: 所有图片所在的路径
    :param save_path: 存储单个label的路径
    :return:
    '''
    thresh = 1/20
    with open(label_file, 'r') as fr:
        for line in fr.readlines():
            # 将一行字符串数据转换成JSON格式
            data = json.loads(line)
            id = data['ID']
            print(id)

            # 保存单个label文件存储的路径到一个文件里
            with open(label_list_file,'a+') as f:
                f.write(save_path+id+'.txt'+'\n')

            with open(save_path+id+'.txt','a+') as fw:
                img_shape = cv2.imread(image_path+id+'.jpg').shape
                h = img_shape[0]
                w = img_shape[1]

                for boxes in data['gtboxes']:
                    try:
                        if boxes['extra']['box_id'] is not None:
                            pbox = boxes['vbox']
                            pbox[0] = (pbox[0]+pbox[2]/2) / w
                            pbox[1] = (pbox[1]+pbox[3]/2) / h
                            pbox[2] = pbox[2] / w
                            pbox[3] = pbox[3] / h
                            if pbox[2]>thresh and pbox[3]>thresh:
                                str_pbox = ' '.join(list(map(str, [person] + pbox)))
                                fw.write(str_pbox+'\n')
                    except:
                        pass

                    if boxes['head_attr']:
                        hbox = boxes['hbox']
                        hbox[0] = (hbox[0] + hbox[2] / 2) / w
                        hbox[1] = (hbox[1] + hbox[3] / 2) / h
                        hbox[2] = hbox[2] / w
                        hbox[3] = hbox[3] / h
                        if hbox[2]>thresh or hbox[3]>thresh:
                            str_hbox = ' '.join(list(map(str, [head] + hbox)))
                            fw.write(str_hbox+'\n')

def test_transform():
    obj = '{"ID": "273271,1f249000e0cb4b12", "gtboxes": [' \
          '{"tag": "person", "hbox": [1178, 203, 203, 292], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [1004, 182, 599, 1619], "vbox": [1066, 187, 434, 814], "extra": {"box_id": 0, "occ": 1}}, ' \
          '{"tag": "person", "hbox": [812, 232, 217, 310], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [669, 209, 493, 1595], "vbox": [719, 217, 437, 784], "extra": {"box_id": 1, "occ": 1}}, ' \
          '{"tag": "person", "hbox": [540, 249, 215, 270], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [122, 193, 680, 1616], "vbox": [126, 195, 641, 800], "extra": {"box_id": 2, "occ": 1}}, ' \
          '{"tag": "person", "hbox": [100, 193, 216, 288], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [-120, 158, 552, 1596], "vbox": [1, 160, 341, 840], "extra": {"box_id": 3, "occ": 1}}, ' \
          '{"tag": "mask", "hbox": [697, 415, 78, 160], "head_attr": {}, "fbox": [697, 415, 78, 160], "vbox": [697, 415, 78, 160], "extra": {"ignore": 1}}]}'

    data = json.loads(obj)
    id = data['ID']
    print(id)



    h = 416
    w = 416

    for boxes in data['gtboxes']:
        try:
            if boxes['extra']['box_id'] is not None:
                pbox = boxes['vbox']

                str_pbox = ' '.join(list(map(str, [person] + pbox)))
                print(str_pbox + '\n')
        except:
            pass

        if boxes['head_attr']:
            hbox = boxes['hbox']

            str_hbox = ' '.join(list(map(str, [head] + hbox)))
            print(str_hbox + '\n')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
import matplotlib.patches as patches
from PIL import Image
import random
# 展示图片及标注
def show_pic_and_coco_labels(file_name=None, root='E:/Datasets/CrowdHuman/train/'):


    if not file_name:
        index = random.randint(0, 15000)
        with open(root + 'labels_train.txt', 'r') as f:
            line = f.readlines()[index]
            file_name = line.split('/')[-1].split('.')[0]
            print(file_name)
            filename = root + 'images/' + file_name
            filelabel = root + 'labels/' + file_name

    else:
        filename = root + 'images/' + file_name
        filelabel = root + 'labels/' + file_name
    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]
    classes = [person, head]  # Extracts class labels from file

    img = np.array(Image.open(filename + '.jpg'))

    w, h, c = img.shape
    print(w, h, c)
    with open(filelabel + '.txt', 'r') as file:
        detections = []
        for i, line in enumerate(file):
            a = []
            for j, l in enumerate(line.split()):
                if j == 0:
                    a.append(int(l))
                elif j % 2 == 0:
                    a.append(float(l) * w)
                else:
                    a.append(float(l) * h)
            detection = a
            detections.append(detection)
        print(detections)
    detections = np.array(detections)

    plt.figure(figsize=(10, 10*h/w), dpi=300)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for cls_pred, x1, y1, x2, y2 in detections:
        color = colors[int(cls_pred)]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1 - x2 / 2, y1 - y2 / 2), x2, y2, linewidth=2,
                                 edgecolor=color,
                                 facecolor='none')
        # Add the bbox to the plot
        ax.add_patch(bbox)

        # Add label
        plt.text(x1 - x2 / 2, y1 - y2 / 2, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                 bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()
    # plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0.0)
    # plt.close()

def show_pic_and_odgt_labels():
    person = 0
    head = 1

    # obj_str='{"ID": "284193,faa9000f2678b5e", "gtboxes": [{"tag": "person", "hbox": [123, 129, 63, 64], "head_attr": {"ignore": 0, "occ": 1, "unsure": 0}, "fbox": [61, 123, 191, 453], "vbox": [62, 126, 154, 446], "extra": {"box_id": 0, "occ": 1}}, {"tag": "person", "hbox": [214, 97, 58, 74], "head_attr": {"ignore": 0, "occ": 1, "unsure": 0}, "fbox": [165, 95, 187, 494], "vbox": [175, 95, 140, 487], "extra": {"box_id": 1, "occ": 1}}, {"tag": "person", "hbox": [318, 109, 58, 68], "head_attr": {"ignore": 0, "occ": 1, "unsure": 0}, "fbox": [236, 104, 195, 493], "vbox": [260, 106, 170, 487], "extra": {"box_id": 2, "occ": 1}}, {"tag": "person", "hbox": [486, 119, 61, 74], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [452, 110, 169, 508], "vbox": [455, 113, 141, 501], "extra": {"box_id": 3, "occ": 1}}, {"tag": "person", "hbox": [559, 105, 53, 57], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [520, 95, 163, 381], "vbox": [553, 98, 70, 118], "extra": {"box_id": 4, "occ": 1}}, {"tag": "person", "hbox": [596, 40, 72, 83], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [546, 39, 202, 594], "vbox": [556, 39, 171, 588], "extra": {"box_id": 5, "occ": 1}}, {"tag": "person", "hbox": [731, 139, 69, 83], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [661, 132, 183, 510], "vbox": [661, 132, 183, 510], "extra": {"box_id": 6, "occ": 0}}]}'
    # obj_str = example1
    obj = '{"ID": "273271,1f249000e0cb4b12", "gtboxes": [' \
          '{"tag": "person", "hbox": [1178, 203, 203, 292], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [1004, 182, 599, 1619], "vbox": [1066, 187, 434, 814], "extra": {"box_id": 0, "occ": 1}}, ' \
          '{"tag": "person", "hbox": [812, 232, 217, 310], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [669, 209, 493, 1595], "vbox": [719, 217, 437, 784], "extra": {"box_id": 1, "occ": 1}}, ' \
          '{"tag": "person", "hbox": [540, 249, 215, 270], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [122, 193, 680, 1616], "vbox": [126, 195, 641, 800], "extra": {"box_id": 2, "occ": 1}}, ' \
          '{"tag": "person", "hbox": [100, 193, 216, 288], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [-120, 158, 552, 1596], "vbox": [1, 160, 341, 840], "extra": {"box_id": 3, "occ": 1}}, ' \
          '{"tag": "mask", "hbox": [697, 415, 78, 160], "head_attr": {}, "fbox": [697, 415, 78, 160], "vbox": [697, 415, 78, 160], "extra": {"ignore": 1}}]}'
    data = json.loads(obj)

    file_name = data["ID"]
    img = cv2.imread("E:\\Datasets\\CrowdHuman\\train\\images\\" + file_name + ".jpg")
    print(img)

    print(data["ID"])
    for obj in data["gtboxes"]:
        # print('tag',obj['tag'])
        # print('hbox',obj['hbox'])

        # print(obj['hbox'], obj['head_attr'])
        # if obj['head_attr']:
        draw_0 = cv2.rectangle(img, (obj['hbox'][0], obj['hbox'][1]), (obj['hbox'][0]+obj['hbox'][2], obj['hbox'][1]+obj['hbox'][3]), (255, 0, 0), 1)
        # print('head_attr',obj['head_attr'])
        # print('fbox',obj['fbox'])
        draw_0 = cv2.rectangle(img, (obj['fbox'][0], obj['fbox'][1]),(obj['fbox'][0] + obj['fbox'][2], obj['fbox'][1] + obj['fbox'][3]), (0, 255, 0), 1)
        # print('vbox', obj['vbox'])
        # print('extra', obj['extra'])
        # a = obj['extra']
        # try:
        #     if a['box_id']:
        #         draw_0 = cv2.rectangle(img, (obj['vbox'][0], obj['vbox'][1]),
        #                                (obj['vbox'][0] + obj['vbox'][2], obj['vbox'][1] + obj['vbox'][3]), (0, 0, 255),
        #                                1)
        # except:
        #     pass

    cv2.imshow("'sdf", img)
    cv2.waitKey()


if __name__ == '__main__':
    root_path = 'E:/Datasets/CrowdHuman/'
    label_file = root_path + 'annotation_train.odgt'
    image_path = 'E:/Datasets/CrowdHuman/train/images/'
    # m
    #save_path = root_path + 'train_m/labels/'
    # train
    #transform(root_path + 'train_m/labels_train.txt', label_file, image_path, save_path)

    # val
    #transform(root_path + 'val_m/labels_val.txt', root_path + 'annotation_val.odgt', 'E:/Datasets/CrowdHuman/val/images/', root_path + 'val_m/labels/')

    # large
    save_path = root_path + 'train_l/labels/'
    # train
    transform(root_path + 'train_l/labels_train.txt', label_file, image_path, save_path)

    # val
    transform(root_path + 'val_l/labels_val.txt', root_path + 'annotation_val.odgt',
              'E:/Datasets/CrowdHuman/val/images/', root_path + 'val_l/labels/')

    #show_pic_and_odgt_labels()
    #test_transform()
    #show_pic_and_coco_labels("273271,1f249000e0cb4b12")
    #show_pic_and_coco_labels()
