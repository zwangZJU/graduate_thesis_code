# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/20
@Desc  :
'''

import os
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import numpy as np






def parse_model_cfg(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])

            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            group = int(module_def['group']) if 'group' in module_def else 1
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        groups=group,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]

            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def["type"] == "InvertedResidual":
            channels = int(module_def["filters"])
            stride = int(module_def["stride"]) if "stride" in module_def else 1
            # expand_ratio = int(module_def["expand_ratio"])
            inverted_res = InvertedResidual(output_filters[-1], channels, stride)
            modules.add_module("inverted_res_%d" % i, inverted_res)
            filters = channels

        elif module_def["type"] == "mobilenetv1":
            inp = output_filters[-1]
            oup = int(module_def["filters"])
            stride = int(module_def["stride"]) if "stride" in module_def else 1
            conv_dw = MobileNetv1(inp, oup, stride)
            modules.add_module("conv_dw_%d" % i, conv_dw)
            filters = oup

        elif module_def["type"] == "shuffleblock":
            if 'stride' in module_def:
                stride = int(module_def["stride"])
            else:
                stride = 1
            filters = int(module_def["filters"])

            if stride == 2:
                shuffle_block = ShuffleUnit(output_filters[-1], oup=filters, mid_channels=filters // 2, ksize=3, stride=2)
            else:
                #shuffle_block = ShuffleV2Block(output_filters[-1]//2, oup=filters, mid_channels=filters // 2, ksize=3, stride=1)
                shuffle_block = ShuffleUnit(output_filters[-1], oup=filters, mid_channels=filters // 2, ksize=3, stride=1)
            modules.add_module("shuffle_block_%d" % i, shuffle_block)

        elif module_def["type"] == "shufflenetv1":
            inp = output_filters[-1]
            oup = int(module_def["filters"])
            # stride = int(module_def["stride"])
            first_group = True if "first_group" in module_def else False

            stride = 2 if first_group else 1

            shufflev1block = ShuffleV1Block(inp, oup,
                           group=3, first_group=first_group,
                           mid_channels=oup // 4, ksize=3, stride=stride)
            modules.add_module("shufflev1block_%d" % i, shufflev1block)
            filters = oup

        elif module_def["type"] == "shufflenetv2":
            inp = output_filters[-1]
            oup = int(module_def["filters"])
            # stride = int(module_def["stride"])
            first_group = True if "stride" in module_def else False
            #first_group = True if "first_group" in module_def else False
            if first_group:
                shufflev2block = ShuffleV2Block(inp, oup,mid_channels=oup // 2, ksize=3, stride=2)
            else:
                shufflev2block = ShuffleV2Block(inp // 2, oup,mid_channels=oup // 2, ksize=3, stride=1)
            modules.add_module("shufflev2block_%d" % i, shufflev2block)
            filters = oup

        elif module_def["type"] == "resshuffleblock":
            if 'stride' in module_def:
                stride = int(module_def["stride"])
            else:
                stride = 1
            filters = int(module_def["filters"])

            if stride == 2:
                res_shuffle_block = ResShuffleBlock(output_filters[-1], oup=filters, mid_channels=filters // 2, ksize=3, stride=2)
            else:
                #shuffle_block = ShuffleV2Block(output_filters[-1]//2, oup=filters, mid_channels=filters // 2, ksize=3, stride=1)
                res_shuffle_block = ResShuffleBlock(output_filters[-1], oup=filters, mid_channels=filters // 2, ksize=3, stride=1)
            modules.add_module("res_shuffle_block_%d" % i, res_shuffle_block)
        elif module_def["type"] == "rsyblock":
            if 'stride' in module_def:
                stride = int(module_def["stride"])
            else:
                stride = 1
            filters = int(module_def["filters"])

            if stride == 2:
                res_shuffle_block = RSYBlock(output_filters[-1], oup=filters, mid_channels=filters // 2, ksize=3, stride=2)
            else:
                #shuffle_block = ShuffleV2Block(output_filters[-1]//2, oup=filters, mid_channels=filters // 2, ksize=3, stride=1)
                res_shuffle_block = RSYBlock(output_filters[-1], oup=filters, mid_channels=filters // 4, ksize=3, stride=1)
            modules.add_module("res_shuffle_block_%d" % i, res_shuffle_block)



        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)




'''
@Author: wzlab
@Date  : 2019/9/24
@Desc  : 
'''
import torch
import torch.nn as nn

class SELayer(nn.Module):

	def __init__(self, inplanes, isTensor=True):
		super(SELayer, self).__init__()
		if isTensor:
			# if the input is (N, C, H, W)
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
			)
		else:
			# if the input is (N, C)
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Linear(inplanes, inplanes // 4, bias=False),
				nn.BatchNorm1d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Linear(inplanes // 4, inplanes, bias=False),
			)

	def forward(self, x):
		atten = self.SE_opr(x)
		atten = torch.clamp(atten + 3, 0, 6) / 6
		return x * atten

class ShuffleUnit(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        #self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:

            branch_main = [
                # pw
                nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
            ]

            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:

            self.conv1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, mid_channels * 2, 1, 1, 0, groups=2, bias=False),
                nn.BatchNorm2d(mid_channels * 2),
                nn.ReLU(inplace=True)
            )

            branch_main = [

                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, oup-mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup-mid_channels),
                nn.ReLU(inplace=True),
            ]

            self.branch_proj = None

        self.branch_main = nn.Sequential(*branch_main)

    def forward(self, old_x):
        if self.stride==1:
           # x_proj, x = self.channel_clip(old_x)
           # return torch.cat((x_proj, self.branch_main(x)), 1)
           # x = self.conv1(old_x)

            old_x = self.conv1(old_x)

            x_proj, x = self.channel_clip(old_x)
            return self.channel_shuffle(torch.cat((x_proj, self.branch_main(x)), 1))
        elif self.stride==2:
            x_proj = old_x
            x = old_x

            return self.channel_shuffle(torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1))

    def channel_clip(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def channel_shuffle(self, x, group=4):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % group == 0
        group_channels = num_channels // group

        x = x.reshape(batchsize, group_channels, group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x


class ResShuffleBlock(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ResShuffleBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        #self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:

            branch_main = [
                # pw
                nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
            ]

            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:

            self.conv1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, mid_channels * 2, 1, 1, 0, groups=2, bias=False),
                nn.BatchNorm2d(mid_channels * 2),
                nn.ReLU(inplace=True)
            )

            branch_main = [

                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, oup-mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup-mid_channels),
                nn.ReLU(inplace=True),
            ]

            self.branch_proj = None

        self.branch_main = nn.Sequential(*branch_main)

    def forward(self, old_x):
        if self.stride==1:
           # x_proj, x = self.channel_clip(old_x)
           # return torch.cat((x_proj, self.branch_main(x)), 1)
           # x = self.conv1(old_x)
            identity = old_x
            old_x = self.conv1(old_x)

            x_proj, x = self.channel_clip(old_x)
            x_out = self.channel_shuffle(torch.cat((x_proj, self.branch_main(x)), 1))
            if identity.shape[1] == x_out.shape:
                return  x_out+ identity
            else:
                return x_out
        elif self.stride==2:
            x_proj = old_x
            x = old_x

            return self.channel_shuffle(torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1))

    def channel_clip(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def channel_shuffle(self, x, group=4):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % group == 0
        group_channels = num_channels // group

        x = x.reshape(batchsize, group_channels, group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x

class RSYBlock(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(RSYBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        #self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:

            branch_main = [
                # pw
                nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
            ]

            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:

            self.conv1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, mid_channels * 2, 1, 1, 0, groups=4, bias=False),
                nn.BatchNorm2d(mid_channels * 2),
                nn.ReLU(inplace=True)
            )

            branch_main = [

                # dw
                nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                # pw-linear
                nn.Conv2d(mid_channels, oup-mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup-mid_channels),
                nn.ReLU(inplace=True),
            ]

            self.branch_proj = None

        self.branch_main = nn.Sequential(*branch_main)

    def forward(self, old_x):
        if self.stride==1:
           # x_proj, x = self.channel_clip(old_x)
           # return torch.cat((x_proj, self.branch_main(x)), 1)
           # x = self.conv1(old_x)
            identity = old_x
            old_x = self.conv1(old_x)

            x_proj, x = self.channel_clip(old_x)
            x_out = self.channel_shuffle(torch.cat((x_proj, self.branch_main(x)), 1))
            if identity.shape[1] == x_out.shape:
                return  x_out+ identity
            else:
                return x_out
        elif self.stride==2:
            x_proj = old_x
            x = old_x

            return self.channel_shuffle(torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1))

    def channel_clip(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def channel_shuffle(self, x, group=4):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % group == 0
        group_channels = num_channels // group

        x = x.reshape(batchsize, group_channels, group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x

class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        branch_main_1 = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        ]
        branch_main_2 = [
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.Sequential(*branch_main_1)
        self.branch_main_2 = nn.Sequential(*branch_main_2)

        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, old_x):
        x = old_x
        x_proj = old_x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            return F.relu(x + x_proj)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class MobileNetv1(nn.Module):
    def __init__(self, inp, oup, stride):
        super(MobileNetv1, self).__init__()
        # self.inp = inp
        # self.oup = oup
        # self.stride = stride
        self.conv_dw = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_dw(x)




class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class RSYNet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(RSYNet, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)


    def forward(self, x):

        layer_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']

            if mtype in ['convolutional', 'upsample', 'maxpool', "InvertedResidual","shuffleblock", "resshuffleblock","rsyblock","mobilenetv1","shufflenetv1","shufflenetv2"]:

                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            layer_outputs.append(x)

        return x


def create_grids(self, img_size, nG):
    self.stride = img_size / nG

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
