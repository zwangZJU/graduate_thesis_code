# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/9/30
@Desc  : 
'''
from models import *
from thop import profile

input = torch.randn(1, 3, 416, 416)
# model = Darknet('cfg/yolov3-tiny.cfg', 416)
# model.load_state_dict(torch.load('weights/yolov3-tiny.pt')['model'])
#
# flops, params = profile(model, inputs=(input,))
# print('yolov3-tiny:')
# print(flops, params)
#
# model = Darknet('cfg/yolov3.cfg', 416)
# model.load_state_dict(torch.load('weights/yolov3.pt')['model'])
# flops, params = profile(model, inputs=(input,))
# print('yolov3:')
# print(flops, params)



# model = Darknet('cfg/yolov3-more-bbox.cfg', 416)
# model.load_state_dict(torch.load('weights/yolov3-more-bbox.pt')['model'])
# flops, params = profile(model, inputs=(input,))
# print('yolov3-more-bbox:')
# print(flops, params)

# model = Darknet('cfg/yolov3-better-bbox.cfg', 416)
# model.load_state_dict(torch.load('weights/yolov3-better-bbox.pt')['model'])
# flops, params = profile(model, inputs=(input,))
# print('yolov3-better-bbox:')
# print(flops, params)
#
#model = Darknet('cfg/yolov3-shufflenet.cfg', 416)
#model.load_state_dict(torch.load('weights/yolov3-shufflenet.pt')['model'])

# model = Darknet('cfg/yolov3-shufflenet.cfg', 416)
# flops, params = profile(model, inputs=(input,))
# print('yolov3-shufflenet:')
# print(flops, params)

# model = Darknet('cfg/yolov3-spp-9.cfg', 416)
# flops, params = profile(model, inputs=(input,))
# print('yolov3-spp-9:')
# print(flops, params)


# model = Darknet('cfg/slim-yolov3-spp-50.cfg', 416)
# # flops, params = profile(model, inputs=(input,))
# # print('prune_0.5:')
# # print(flops, params)

model = Darknet('cfg/yolov3-res-shuffleunit-9-tiny.cfg', 416)
flops, params = profile(model, inputs=(input,))
print('prune_0.5:')
print(flops, params)


