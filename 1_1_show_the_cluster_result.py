# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/9/18
@Desc  : 找出合适的bbox数量，选择合适的bbox
'''
n_person = 352836
n_face = 334893
n = 687729

def load_data(file_name, n):
    d = {}
    with open(file_name,'r') as f:
        for line in f.readlines():
            if not line:
                continue
            data = line.split()
            size =  data[1:-1]
            print(line)
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

import matplotlib.pyplot as plt
def draw_curve(x, y):
    plt.title("k-means")
    plt.xlabel("k")
    plt.ylabel("distance")
    plt.plot(x, y)
    plt.show()

def draw_three_curve(x1, y1, x2, y2, x3, y3):
    plt.plot(x1, y1, color='black', label='both', linewidth=0.8)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
    plt.plot(x2, y2, color='black', label='person', linewidth=0.8)
    plt.plot(x3, y3, color='black', label='face', linewidth=0.8)
    plt.title("k-means")
    plt.xlabel("k")
    plt.ylabel("distance")
    plt.show()


if __name__ == '__main__':

    k1, distance1 = del_outlier(*load_data('results/1.0 result.txt', n))
    k2, distance2 = del_outlier(*load_data('results/1.0 result0.txt', n_person))
    k3, distance3 = del_outlier(*load_data('results/1.0 result1.txt', n_face))
    draw_three_curve(k1, distance1, k2, distance2, k3, distance3)