# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/11/8
@Desc  : 准备ImageNet验证机的label
label在val中，只有5k行的编号，要将编号与训练集里的文件夹对应
ILSVRC2012_validation_ground_truth.txt中的编号有问题
'''
import os
import shutil

def list_dir(path):
    all_dirs = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            all_dirs.append(dir)
    return all_dirs

def new_dir(root, dir_names):
    '''
    新建文件夹
    :param root:
    :param dir_names:
    :return:
    '''
    for dir in dir_names:
        path = os.path.join(root, dir)
        if not os.path.exists(path):
            os.makedirs(path)

def load_label(path):
    with open(path, 'r') as f:
        all_labels = [line.split()[1] for line in f.readlines()]
    return all_labels

def load_img_and_copy(path,new_path, all_labels):
    for i, filename in enumerate(os.listdir(path)):
        print(i)
        shutil.copyfile(os.path.join(path,filename), os.path.join(new_path[int(all_labels[i])], filename))

    #d =



if __name__ == '__main__':
    path = 'E:/Datasets/ImageNet/ILSVRC2012_img_train'
    all_dirs = list_dir(path)
    all_labels = load_label('E:/Datasets/ImageNet/val.txt')
    root = 'E:/Datasets/ImageNet/val'
    new_dir(root, all_dirs)
    d_new_path = dict(zip(list(range(0,1000)),[os.path.join(root,dir_name) for dir_name in all_dirs]))
    print(d_new_path[1])
    load_img_and_copy('E:/Datasets/ImageNet/ILSVRC2012_img_val', d_new_path, all_labels)