# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/11
@Desc  : 画出focal loss的曲线图
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import mpl

def cross_entropy(x):
    return -np.log(x)

def focal_loss(x, alpha=1, gamma=2):
    return -alpha*(1-x)**gamma*np.log(x)

def my_loss(x, beta=0.5):
    return - np.exp(beta - x**2) ** 2 * np.log(x)

def my_loss1(x):
    return - np.exp(1.5 - x) ** 2 * np.log(x)
def my_loss2(x):
    return - (1.5 - x) ** 2 * np.log(x)
sns.set()
sns.set_style('whitegrid')
x = np.linspace(0.2, 1, 50)
# plt.plot(x, cross_entropy(x))
# 设置字体
#font = mpl.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
plt.plot(x, focal_loss(x, gamma=0),'-.', label='Cross Entropy Loss')
plt.plot(x, focal_loss(x, gamma=2),':', label=r'Focal Loss ($\gamma=2$)')
plt.plot(x, my_loss(x),'-',label=r'MCE Loss ($\beta$=0.5)')
plt.plot(x, my_loss(x, beta=0.2),'--',label=r'MCE Loss ($\beta$=0.2)')
#plt.plot(x, my_loss1(x),':',label=r'Margin Cross Entropy Loss, $\beta$=0.2')
#plt.plot(x, my_loss2(x),':',label=r'Margin Cross Entropy Loss, $\beta$=0.2')
plt.ylabel('loss')
plt.xlabel('probability of ground truth class')
# font1 = {'family' : font,
# 'weight' : 'normal',
# 'size'   : 12,
# }
leg = plt.legend()
plt.savefig('results/loss_en.png', dpi=600)
plt.show()


