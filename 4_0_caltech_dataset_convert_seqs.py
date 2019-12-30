# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/11/27
@Desc  : 把seqs转为图片
'''
import os
import glob
import cv2


def save_img(dname, fn, i, frame):
    cv2.imwrite('{}/{}_{}_{}.png'.format(
        out_dir, os.path.basename(dname),
        os.path.basename(fn).split('.')[0], i), frame)

out_dir = 'E:/Datasets/Caltech/images'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for dname in sorted(glob.glob('E:/Datasets/Caltech/set*')):
    for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
        cap = cv2.VideoCapture(fn)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            save_img(dname, fn, i, frame)
            i += 1
        print(fn)