# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2019/12/12
@Desc  : 在图片上画出网格，用于解释YOLOv3的算法思想
'''
import cv2

def draw(n):
    img = cv2.imread('sample/eg.png')
    grid = [x for x in range(0, 651, 650//n)]
    print(grid)
    print(img.shape)
    for g in grid:
        cv2.line(img,(g,0),(g,650),(0,0,0),2)
        cv2.line(img, (0, g), (650, g), (0, 0, 0), 1)
    cv2.imwrite('sample/eg_{}.png'.format(n), img)

if __name__ == '__main__':
    draw(13)
    draw(26)
    draw(52)


