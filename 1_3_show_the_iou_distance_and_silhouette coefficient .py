# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/9/18
@Desc  : 画出iou距离和轮廓系数关于k变化的曲线，在同一幅图中用双y轴的画法
tutorial：https://matplotlib.org/examples/api/two_scales.html
'''
n_person = 352836
n_p = 262529
n_face = 334893
n = 687729
from matplotlib.font_manager import FontProperties
from pylab import mpl
def load_data(file_name, n):
    d = {}
    with open(file_name,'r') as f:
        for line in f.readlines():
            if not line:
                continue
            data = line.split()
            size =  data[1:-1]

            distance = float(data[-1])
            count = 0
            for s in size:
                if s != '0,0,':
                    count += 1
            if count not in d:
                d[count] = distance
        s = sorted(d.items())
        k, d = [], []
        for i in s:
            k.append(i[0])
            d.append(i[1]/n)

    return k, d

def del_outlier(k, d):
    '''
    删除异常值
    :param k:
    :param d:
    :return:
    '''
    n = len(k)
    i = 1
    while i<n:
        if d[i]>d[i-1]:
            d.pop(i)
            k.pop(i)
            n -= 1
        else:
            i += 1
    return k, d

def load_sc(file_name):
    data = []
    with open(file_name) as f:
        for line in f.readlines():

            data.append(list(map(lambda x: float(x[:-1]), line.split())))
    return data[0], data[1]

import matplotlib.pyplot as plt
def draw_curve(x, y):
    plt.title("k-means")
    plt.xlabel("k")
    plt.ylabel("distance")
    plt.plot(x, y)
    plt.show()
import seaborn as sns
def draw_three_curve(x, y1, y2):


    # plt.plot(x, cross_entropy(x))
    # 设置字体
    font = mpl.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
    font_en =   {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
    font_size = 12
    label_size = 14
    fig, ax1 = plt.subplots()
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax1.plot(x, y1, '-',color='royalblue')
    ax1.set_xlabel('k',font_en)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('平均距离', fontproperties = font, color='royalblue',fontsize=label_size)
    ax1.tick_params('y', colors='royalblue')

    ax2 = ax1.twinx()

    ax2.plot(x, y2, 'm-.')
    ax2.set_ylabel('轮廓系数', fontproperties = font, color='m',fontsize=label_size)

    ax2.tick_params('y', colors='m',labelsize=font_size)

    fig.tight_layout()
    plt.savefig('results/sc_face.png', dpi=600)
    plt.show()


if __name__ == '__main__':

    # k, d = del_outlier(*load_data('results/k_and_distance_new.txt', n))
    # _, sc = load_sc('results/k_and_silhouette_coefficient_new.txt')
    # k, d = del_outlier(*load_data('results/large_box_by_k_means_both_new_k_d_person.txt', n_person))
    # _, sc = load_sc('results/k_and_silhouette_coefficient_person_new.txt')
    k, d = del_outlier(*load_data('results/large_box_by_k_means_both_new_k_d_face.txt', n_person))
    _, sc = load_sc('results/k_and_silhouette_coefficient_face_new.txt')

    print(sc)
    print(k)
    print(d)
    # k2, distance2 = del_outlier(*load_data('results/1.0 result0.txt', n_person))
    # k3, distance3 = del_outlier(*load_data('results/1.0 result1.txt', n_face))
    draw_three_curve(k, d, sc)