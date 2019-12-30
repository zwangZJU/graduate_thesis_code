# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/9/12
@Desc  : 
'''

import os

def convert(res_path, oup_dir, size=608):

    w0, h0 = 640, 480

    with open(res_path, 'r') as fr:
        # 所有label的path
        for line in fr.readlines():
            # path = os.path.basename(line)
            dir_name, file_name, n, _ = os.path.basename(line).split('_')
            with open(line.replace("\n", ''), 'r') as f:
                for l in f.readlines():
                    cls, cx, cy, w, h = map(float,l.split(' '))
                    x = (cx - w/2) * w0
                    y = (cy - h/2) * h0
                    w = w * w0
                    h = h * h0

                    final_dir = oup_dir + dir_name + '/'
                    final_file = final_dir + file_name+'.txt'
                    info = [str(int(n[1:])+1)] + list(map(lambda x: str(round(x,2)), [x, y, w, h])) + [str(1)]

                    if not os.path.exists(final_dir):
                        os.makedirs(final_dir)

                    with open(final_file, 'a') as fw:
                        fw.write(','.join(info)+'\n')
# def walk(root):
#     for d1 in os.listdir(root):
#         for d2 in os.listdir(root+d1):
#             dir = root+d1+'/'+d2
#
#             for d3 in os.listdir(dir):
#                 file = dir + '/' + d3
#                 print(d3)
#                 with open(file, 'r') as fr:
#                     lines = fr.readlines()[1:]
#                     if lines:
#                         with open('E:/Datasets/Caltech/Person/all_official_labels/'+'_'.join([d1,d2,d3.split('.')[0],'usatest.txt']), 'w') as fw:
#                             info = ''
#                             for line in lines:
#                                 info += ' '.join(['0'] + line.split()[1:5])+'\n'
#                             fw.write(info)
#
#
#
#
#
#
# def convert_xyxy_to_xywh(path, new_label_path, all_labels_txt_path):
#     '''
#     将xmin, ymin, xmax, ymax的左边转为中心x,y和长宽
#     path: 原始label的路径
#     new_label_path: 生成新的单个label存储路径
#     all_labels_txt_path: 存储生成所有label路径的文件路径
#     :return:
#     '''
#     if not os.path.exists(new_label_path):
#         os.makedirs(new_label_path)
#     h0, w0 = 480, 640
#     all_labels = []
#     for file_name in os.listdir(path):
#         if os.path.getsize(path+file_name):
#             all_labels.append(new_label_path+file_name)
#             with open(path + file_name, 'r') as f:
#                 info = []
#                 for line in f.readlines():
#                     c, x1, y1, w, h = map(int, line.split())
#                     c = 0
#                     x = (x1+w/2)/ w0
#                     y = (y1+h/2)/h0
#                     w = w/w0
#                     h = h/h0
#
#                     info.append(' '.join(list(map(str,[c,x,y,w,h]))))
#             with open(new_label_path+file_name,'w') as fw:
#                 fw.write('\n'.join(info))
#     with open(all_labels_txt_path, 'w') as fw:
#         fw.write('\n'.join(all_labels))
#     print(len(all_labels))




if __name__ == '__main__':
    # root = 'E:/Datasets/Caltech/Person/official/'
    # walk(root)
    res_path = 'E:/Datasets/Caltech/Person/test_official/test_labels.txt'
    oup_dir = 'E:/Datasets/Caltech/Person/test_official/res_official/'
    convert(res_path, oup_dir, size=608)
