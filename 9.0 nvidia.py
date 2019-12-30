# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/9/23
@Desc  :  使用nvidia-ml-py库查看显存的占用情况
'''

import pynvml
pynvml.nvmlInit()
print("Driver: ", pynvml.nvmlSystemGetDriverVersion())  #显示驱动信息
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory Total: ", meminfo.total)
print("Memory Used: ", meminfo.used)
print("Memory Free: ", meminfo.free)



