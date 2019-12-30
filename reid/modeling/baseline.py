# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.rsynet import RSYNet
from .backbones.mobilenetv4 import MobileNetV4_Large
from .backbones.mobilenetv5 import MobileNetV5_Large
# from .arcface import ArcCos
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg,
                 weights_path):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'rsynet':
            self.base = RSYNet(cfg)
            if weights_path == '':
                self.base.apply(weights_init_normal)
            else:
                checkpoint = torch.load(weights_path, map_location='cpu')

                pretrained_dict = checkpoint
                model_dict = self.base.state_dict()
                # for k, v in pretrained_dict.items():
                #     print(k)
                #     if int(k.split('.')[1])<28:
                #         print(k)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # filter out unnecessary keys
                model_dict.update(pretrained_dict)
                self.base.load_state_dict(model_dict)
        elif model_name == 'mobilenetv4':
            self.in_planes = 2048
            self.base = MobileNetV4_Large()
        elif model_name == 'mobilenetv5':
            self.in_planes = 2048
            self.base = MobileNetV5_Large
            # pretrained_dict = torch.load("./weights/mbv3_large.old.pth.tar", map_location='cpu')["state_dict"]
            # model_dict = self.base.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # filter out unnecessary keys
            # model_dict.update(pretrained_dict)
            # self.base.load_state_dict(model_dict)
            # for p in self.base.parameters():
            #     p.requires_grad = False

        # if pretrain_choice == 'imagenet':
        #     self.base.load_param(model_path)
        #     print('Loading pretrained ImageNet model......')


        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        local_conv_out_channels = 512
        self.local_f = nn.Sequential(
            nn.Conv2d(self.in_planes, local_conv_out_channels, 1, stride=1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)
        )


        self.global_f = nn.Sequential(
            nn.Conv2d(local_conv_out_channels, 2048, 1, stride=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )


        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            #self.classifier = ArcCos(self.in_planes, self.num_classes)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat = self.base(x)
        #branch1
        global_feat = self.gap(feat)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # branch2
        local_feat = torch.mean(feat, -1, keepdim=True) # (b, 2048, 16, 1) #shape [N, C, H, W] stride=1时
        # shape [N, H, c]

        local_feat = F.avg_pool2d(local_feat, (4,1)) # [b, 2048, 4, 1])
        #print(local_feat.shape)
        local_feat = self.local_f(local_feat) # [b, 512, 4, 1])
        #print(local_feat.shape)
        #global_feat = self.global_f(local_feat)
        #print(global_feat.shape)
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        #global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])
        #print(global_feat.shape)
        #global_feat = global_feat.view(global_feat.shape[0], -1)
        #print(global_feat.shape)






        # global_feat = self.gap(x)  # (b, 2048, 1, 1)


        #global_feat = local_feat.view(local_feat.shape[0], -1)  # flatten to (bs, 2048)

        # local feature
        # local_feat = torch.mean(feat, -1, keepdim=True)
        # local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # # shape [N, H, c]
        # local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
                #  return self.classifier(feat)
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
