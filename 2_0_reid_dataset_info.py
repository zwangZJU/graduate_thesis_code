# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/10/23
@Desc  : 查看行人重识别数据集中图片的信息
'''
import cv2
def get_img_shape(path):
    img = cv2.imread(path)
    return img.shape

if __name__ == '__main__':
    path = 'E:/Datasets/Re_ID/msmt17/MSMT17/query/0005_c5_0026.jpg'
    w, h, c = get_img_shape(path)
    print(w, h, c)