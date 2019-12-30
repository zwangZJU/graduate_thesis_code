# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/10/9
@Desc  : 过滤标签中的小目标后，会产生一些空的标签文件，将这些文件删除掉
'''
import os


def filter_empty_file(path, new_path):

    with open(path, 'r') as file:
        with open(new_path, 'a') as new_f:
            label_files = file.readlines()
            for fp in label_files:
                if os.path.getsize(fp.replace('\n','')) > 0:
                    new_f.write(fp)




if __name__ == '__main__':
    # for m
    # val = 'E:/Datasets/CrowdHuman/val_m/labels_val.txt'
    # train = 'E:/Datasets/CrowdHuman/train_m/labels_train.txt'
    # new_val = 'E:/Datasets/CrowdHuman/val_m/labels_val_new.txt'
    # new_train = 'E:/Datasets/CrowdHuman/train_m/labels_train_new.txt'

    # for large
    val = 'E:/Datasets/CrowdHuman/val_l/labels_val.txt'
    train = 'E:/Datasets/CrowdHuman/train_l/labels_train.txt'
    new_val = 'E:/Datasets/CrowdHuman/val_l/labels_val_new.txt'
    new_train = 'E:/Datasets/CrowdHuman/train_l/labels_train_new.txt'


    filter_empty_file(val, new_val)
    filter_empty_file(train, new_train)

