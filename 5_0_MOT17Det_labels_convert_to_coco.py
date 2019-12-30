# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/2
@Desc  : <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
'''
import os
import cv2
def get_image_size(path):
    img = cv2.imread(path)
    print(img.shape)

def convert_mot_to_coco(path, img_size, all_label_path):
    all_label_set = set()
    suffix = '/gt/gt.txt'
    img_h, img_w = img_size
    print(path)
    with open(path+suffix, 'r') as fr:
        for line in fr.readlines():
            frame, _, x, y, w, h, conf, cls, visibility = line.split(',')
            file_name = '0'*(6-len(frame))+frame
            if not os.path.exists(path+'/labels'):
                os.makedirs(path+'/labels')
            all_label_set.add(path+'/labels/'+file_name+'.txt')
            if cls == '1':
                with open(path+'/labels/'+file_name+'.txt','a') as fw:
                    cx = str(round((int(x)+int(w)/2)/img_w,4))
                    cy = str(round((int(y)+int(h)/2)/img_h,4))
                    w = str(round(int(w) / img_w, 4))
                    h = str(round(int(h) / img_h, 4))
                    fw.write(' '.join(['0'] + [cx, cy, w, h])+'\n')

def all_labels(root, dirs, save_path):
    with open(save_path, 'a') as f:
        for dir in dirs:
            print(dir)
            for file in os.listdir(root+dir+'/labels/'):
                f.write(root+dir+'/labels/'+file+'\n')





if __name__ == '__main__':
    root = 'E:/Datasets/MOT17Det/train/'
    all_label_path_train = 'E:/Datasets/MOT17Det/train/train_labels1.txt'
    train_dirs = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    train_img_size = [(1080, 1920),(1080, 1920),(480, 640),(1080, 1920),(1080, 1920),(1080, 1920),(1080, 1920)]

    for i, train_dir in enumerate(train_dirs):
        convert_mot_to_coco(root+train_dir, train_img_size[i], all_label_path_train)

    #all_labels(root, train_dirs, all_label_path_train)




    # root = 'E:/Datasets/MOT17Det/test/'
    # all_label_path_test = 'E:/Datasets/MOT17Det/test/test_labels.txt'
    # test_dirs = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-8', 'MOT17-12', 'MOT17-14']
    # test_img_size = [(1080, 1920), (1080, 1920), (480, 640), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920)]
    # # train_dirs = ['MOT17-02']
    # # train_img_size = [(1080, 1920)]
    # for i, train_dir in enumerate(test_dirs):
    #     convert_mot_to_coco(root+train_dir, test_img_size[i], all_label_path_test)
        #get_image_size(root+train_dir+'/img1/000001.jpg')