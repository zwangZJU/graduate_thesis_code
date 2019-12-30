# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/1
@Desc  : 
'''
import numpy as np

a = np.array([27,51, 30,95, 33,158, 64,127, 47,224, 71,271, 111,281, 187,326])
b = a*320/416
print(b.astype(np.int32))