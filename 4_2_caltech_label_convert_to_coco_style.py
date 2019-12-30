# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/1
@Desc  : 把matlab处理后的txt标签转为coco格式，xyxy转为xywh,并过滤掉没有人的图片
'''
import os


def convert_xyxy_to_xywh(path, new_label_path, all_labels_txt_path):
    '''
    将xmin, ymin, xmax, ymax的左边转为中心x,y和长宽
    path: 原始label的路径
    new_label_path: 生成新的单个label存储路径
    all_labels_txt_path: 存储生成所有label路径的文件路径
    :return:
    '''
    if not os.path.exists(new_label_path):
        os.makedirs(new_label_path)
    w, h = 480, 640
    all_labels = []
    for file_name in os.listdir(path):
        if os.path.getsize(path+file_name):
            all_labels.append(new_label_path+file_name)
            with open(path + file_name, 'r') as f:
                info = []
                for line in f.readlines():
                    c, xmin, ymin, xmax, ymax = map(int, line.split())
                    c = 0
                    x = (xmin+xmax)/(2*640)
                    y = (ymin+ymax)/(2*480)
                    w = (xmax-xmin)/640
                    h = (ymax-ymin)/480

                    info.append(' '.join(list(map(str,[c,x,y,w,h]))))
            with open(new_label_path+file_name,'w') as fw:
                fw.write('\n'.join(info))
    with open(all_labels_txt_path, 'w') as fw:
        fw.write('\n'.join(all_labels))
    print(len(all_labels))




if __name__ == '__main__':
    test_path = 'E:/Datasets/Caltech/Person/all_test_labels/'
    new_test_path = 'E:/Datasets/Caltech/Person/test/labels/'
    new_test_all_label = 'E:/Datasets/Caltech/Person/test/test_labels.txt'
    convert_xyxy_to_xywh(test_path, new_test_path, new_test_all_label)

    train_path = 'E:/Datasets/Caltech/Person/all_train_labels/'
    new_train_path = 'E:/Datasets/Caltech/Person/train/labels/'
    new_train_all_label = 'E:/Datasets/Caltech/Person/train/train_labels.txt'

    convert_xyxy_to_xywh(train_path, new_train_path, new_train_all_label)