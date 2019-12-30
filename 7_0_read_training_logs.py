# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/12
@Desc  : 两个设备上都有运行时的日志,根据日志画出学习过程的图
'''
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import mpl
def read_log(path):
    with open(path, 'r') as fr:
        file1 = []
        file2 = []

        i = 0
        for line in fr.readlines():
            a = int(line.split()[0].split('/')[0])
            if a == i and a<=74:
                file1.append(line.strip())
                i += 1
            else:
                file2.append(line.strip())
        file3 = file2[0::2]
        file4 = file2[1::2]

        with open('results/traininglog/1.txt','w') as f1:
            for line in file1:
                f1.write(line+'\n')
        with open('results/traininglog/c12.txt','w') as f1:
            for line in file3:
                f1.write(line+'\n')
        with open('results/traininglog/c15.txt','w') as f1:
            for line in file4:
                f1.write(line+'\n')
import numpy as np
def load_data(path):
    '''
    加载单个训练日志
    :param path:
    :return:
    '''
    with open(path, 'r') as fr:
        epochs = []
        l = []
        for line in fr.readlines():

            # epoch, _, loss_xy, loss_wh, loss_conf, loss_cls, loss_total, n_target, fps, P, R, mAP = line.split()
            data = line.split()
            epochs.append(int(data[0].split('/')[0]))
            l.append(list(map(float,data[2:])))
        epoch = np.array(epochs)
        mat = np.array(l)
        loss_xy = mat[:, 0].T
        loss_wh = mat[:, 1].T
        loss_conf = mat[:, 2].T
        loss_cls = mat[:, 3].T
        loss_total = mat[:, 4].T
        n_target = mat[:, 5].T
        fps = mat[:, 6].T
        P = mat[:, 7].T
        R = mat[:, 8].T
        mAP = mat[:, 9].T
        return epoch, loss_xy, loss_wh, loss_conf, loss_cls, loss_total, n_target, fps, P, R, mAP
       # print(mAP.tolist())
def plot(xy,labels, img_name = 'img'):
    style = ['-','--','-.',':']
    color = ['firebrick', 'tomato', 'chocolate', 'forestgreen', 'royalblue', 'mediumorchid', 'crimson', 'darkcyan', 'dodgerblue', 'olive']
    marker = [ '+', '.', 'x','1','2','o', '*']
    # 设置字体
    font = mpl.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
    for i, (x, y) in enumerate(xy):
        if i<4:
            plt.plot(x, y, color=color[i], linestyle=style[i], label=labels[i])
        else:
            plt.plot(x, y, color=color[i], linestyle=style[i % 4], marker=marker[i%7], label=labels[i])

    # plt.plot(x, my_loss1(x),':',label=r'Margin Cross Entropy Loss, $\beta$=0.2')
    # plt.plot(x, my_loss2(x),':',label=r'Margin Cross Entropy Loss, $\beta$=0.2')
    plt.xlabel('轮次', fontproperties=font)
    plt.ylabel('损失', fontproperties=font)
    font1 = {'family': font,
             'weight': 'normal',
             'size': 12,
             }
    leg = plt.legend(prop=font)
    plt.savefig('results/'+img_name+'.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    p1 = 'results/traininglog/c12.txt'
    labels = ['loss_xy', 'loss_wh', 'loss_conf', 'loss_cls', 'loss_total', 'n_target', 'fps', 'P', 'R', 'mAP']
    #read_log(p1)
    epoch, loss_xy, loss_wh, loss_conf, loss_cls, loss_total, n_target, fps, P, R, mAP =load_data(p1)
    x = [epoch] * 10
    y = loss_xy, loss_wh, loss_conf, loss_cls, loss_total, n_target, fps, P, R, mAP
    xy = zip(x,y)
    plot(xy, labels)


