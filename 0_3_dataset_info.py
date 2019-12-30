# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/10/9
@Desc  : 读取数据集的信息
'''
import os
import cv2

def get_size(path):
    max_box = [0,0,0]
    min_box = [1000,2000, 1000*2000]
    with open(path, 'r') as file:

        label_files = file.readlines()
        label_files = [fpath.replace('\n', '').replace('_m', '').replace('labels', 'images').replace('.txt', '.jpg') for fpath in label_files]
        n = len(label_files)
        mean_w = mean_h = 0
        for f in label_files:

            img = cv2.imread(f)
            w, h, _ = img.shape
            mean_w += w/n
            mean_h += h/n
            cur_area = w*h
            if cur_area>max_box[2]:
                max_box = [w, h, cur_area]
            elif cur_area<min_box[2]:
                min_box = [w, h, cur_area]

        print('max box is {} * {}'.format(*max_box) + '\n' +
              'mix box is {} * {}'.format(*min_box) + '\n' +
              'mean box is {} * {}'.format(mean_w, mean_h))




if __name__ == '__main__':
    val = 'E:/Datasets/CrowdHuman/val_m/labels_val.txt'
    train = 'E:/Datasets/CrowdHuman/train_m/labels_train.txt'
    new_val = 'E:/Datasets/CrowdHuman/val_m/labels_val_new.txt'
    new_train = 'E:/Datasets/CrowdHuman/train_m/labels_train_new.txt'
    get_size(new_train)

