# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/9
@Desc  : 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MarginCrossEntropyWithLogits(nn.Module):
    def __init__(self, alpha=1, gamma=2, beta=0.5, size_average=True):
        super(MarginCrossEntropyWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.size_average = size_average

    def forward(self, pred, gt):
        gt = gt.view(-1, 1).long()
        prob = torch.sigmoid(pred.view(-1, 1)).clamp(min=0.0001, max=0.9999)
        prob_ = 1 - prob
        pt = torch.cat([prob_, prob], 1).gather(1, gt)
        #         pred = self.softmax(pred)
        #         prob = Variable(pred.gather(1, gt), requires_grad=True)
        #- np.exp(1.5 - x) ** 2 * np.log(x)
        # loss = -1 * torch.pow(torch.exp(1.5-pt), 2)*pt.log()
        loss = - 1 * self.alpha * torch.pow(torch.exp(self.beta - torch.pow(pt,2)), self.gamma) * pt.log()

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

if __name__ == "__main__":


    FL = MarginCrossEntropyWithLogits( gamma=2, size_average=True )
    CE = nn.CrossEntropyLoss()
    N = 2
    C = 3
    pred = Variable(torch.FloatTensor([[0,0,0,0,0.5,0,0,0,0,1,1,1]]), requires_grad=True)
    gt = torch.ByteTensor([[0,0,0,0,0,0,0,0,0,1,1,1]])

    fl_loss = FL(pred, gt)
    fl_loss.backward()
    print(fl_loss)
    print(pred.grad)
    print(torch.exp(torch.Tensor([1])))

    # C = 5
    # inputs = torch.rand(N, C)
    # targets = torch.LongTensor(N).random_(C)
    # inputs_fl = Variable(inputs.clone(), requires_grad=True)
    # targets_fl = Variable(targets.clone())
    #
    # inputs_ce = Variable(inputs.clone(), requires_grad=True)
    # targets_ce = Variable(targets.clone())
    # print('----inputs----')
    # print(inputs)
    # print('---target-----')
    # print(targets)
    #pred = torch.ByteTensor([[0,0.5,1],[1,1,1],[1,1,1]])
    BCE = nn.BCEWithLogitsLoss()
    #gt = torch.ByteTensor([[0,1,1],[1,0,1],[1,1,0]])
    # fl_loss = FL(inputs_fl, targets_fl)
    #ce_loss = BCE(pred, gt)
    # print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
    # fl_loss.backward()
    # ce_loss.backward()
    # #print(inputs_fl.grad.data)
    #print(ce_loss)