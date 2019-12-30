# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/21
@Desc  : 
'''
# encoding: utf-8


import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from torch.utils.data import DataLoader
from modeling.baseline import Baseline
from data.collate_batch import val_collate_fn
from data.datasets import init_dataset, ImageDataset
#from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from data.transforms import build_transforms

def main():




    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    #train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    model = Baseline(751, 1, '', 'bnneck', 'after', cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, '../configs/RSYNet.cfg', '../weights/rsynet.pt')
    model.load_param('../checkpoints/rsynet_model_50.pth')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset('market1501', root='../data')
    print(1, dataset.query)
    print(2, dataset.gallery)
    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    for i, (imgs, pids, cids) in enumerate(test_loader):
        print(imgs)
        print(pids)
        print(cids)
        print(imgs.shape)
        imgs = imgs.to(device)
        pred = model(imgs)
        print(pred.shape)
        a, b = torch.topk(pred, 5, 1)
        c= torch.nn.functional.normalize(pred, dim=1, p=2)
        print(a,b,c)
        break



if __name__ == '__main__':
    main()
