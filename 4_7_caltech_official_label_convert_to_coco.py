# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/4
@Desc  : 把results文件夹下的caltech416.txt和caltech608.txt转为标准的
'''
import os
def walk(root):
    for d1 in os.listdir(root):
        for d2 in os.listdir(root+d1):
            dir = root+d1+'/'+d2

            for d3 in os.listdir(dir):
                file = dir + '/' + d3
                print(d3)
                with open(file, 'r') as fr:
                    lines = fr.readlines()[1:]
                    if lines:
                        with open('E:/Datasets/Caltech/Person/all_official_labels/'+'_'.join([d1,d2,d3.split('.')[0],'usatest.txt']), 'w') as fw:
                            info = ''
                            for line in lines:
                                info += ' '.join(['0'] + line.split()[1:5])+'\n'
                            fw.write(info)






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
    h0, w0 = 480, 640
    all_labels = []
    for file_name in os.listdir(path):
        if os.path.getsize(path+file_name):
            all_labels.append(new_label_path+file_name)
            with open(path + file_name, 'r') as f:
                info = []
                for line in f.readlines():
                    c, x1, y1, w, h = map(int, line.split())
                    c = 0
                    x = (x1+w/2)/ w0
                    y = (y1+h/2)/h0
                    w = w/w0
                    h = h/h0

                    info.append(' '.join(list(map(str,[c,x,y,w,h]))))
            with open(new_label_path+file_name,'w') as fw:
                fw.write('\n'.join(info))
    with open(all_labels_txt_path, 'w') as fw:
        fw.write('\n'.join(all_labels))
    print(len(all_labels))




if __name__ == '__main__':
    # root = 'E:/Datasets/Caltech/Person/official/'
    # walk(root)
    test_path = 'E:/Datasets/Caltech/Person/all_official_labels/'
    new_test_path = 'E:/Datasets/Caltech/Person/test_official/labels/'
    new_test_all_label = 'E:/Datasets/Caltech/Person/test_official/test_labels.txt'
    convert_xyxy_to_xywh(test_path, new_test_path, new_test_all_label)
    #
    # train_path = 'E:/Datasets/Caltech/Person/all_train_labels/'
    # new_train_path = 'E:/Datasets/Caltech/Person/train/labels/'
    # new_train_all_label = 'E:/Datasets/Caltech/Person/train/train_labels.txt'
    #
    # convert_xyxy_to_xywh(train_path, new_train_path, new_train_all_label)