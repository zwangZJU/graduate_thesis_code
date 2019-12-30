# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/4
@Desc  : 把results文件夹下的caltech416.txt和caltech608.txt转为标准的matlab能评估的格式
'''
import os
def convert(res_path, oup_dir, size=608):

    w0, h0 = 640, 480
    dh = (size-480*size/w0)/2
    with open(res_path, 'r') as fr:
        for line in fr.readlines():
            print(line)
            path, x1, y1, x2, y2, conf, cls, _ = line.split(',')
            # x = (float(cx)-float(w)/2)/size*w0
            # y = (float(cy)-float(h)/2)/size*h0
            x = float(x1) / size * w0
            y = (float(y1)-dh)/size*w0
            w = (float(x2)-float(x1))/size*w0
            h = (float(y2)-float(y1))/size*w0
            conf = float(conf)
            # if conf<0.5:
            #     conf += 0.32
            dir_name, file_name, n, _ = os.path.basename(path).split('_')
            final_dir = oup_dir + dir_name + '/'
            final_file = final_dir + file_name+'.txt'
            info = [str(int(n[1:])+1)] + list(map(lambda x: str(round(x,2)), [x, y, w, h])) + [str(round(conf,4))]
            # print(dir_name, file_name, n)
            # print(','.join(info))
            # print(final_file)
            if not os.path.exists(final_dir):
                os.makedirs(final_dir)
            #break
            with open(final_file, 'a') as fw:
                fw.write(','.join(info)+'\n')


if __name__ == '__main__':
    # res_path = 'results/caltech608all.txt'
    # oup_dir = 'results/Ours-608-all/'
    # convert(res_path, oup_dir)

    # res_path = 'results/caltech608.txt'
    # oup_dir = 'results/Ours-608-c/'
    # convert(res_path, oup_dir)

    res_path = 'results/caltech416.txt'
    oup_dir = 'results/Ours-416-c/'
    convert(res_path, oup_dir, size=416)