#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : common.py
# @Time : 2025/6/14 18:09

import cv2
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm

## 余弦距离
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

## 计算欧式距离
def cal_distance(coord1,coord2):
    return math.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2)

## 得到文字的长和宽
def cal_width_height(bbox):
    width = cal_distance((bbox[0],bbox[1]),(bbox[2],bbox[3]))
    height = cal_distance((bbox[2],bbox[3]),(bbox[4],bbox[5]))
    return int(width),int(height)
	
## 透视变换截取图片
def get_perspective_image(image, bbox):
    width,height = cal_width_height(bbox)
    if height>width:
        pts1 = np.float32([[0,0],[height,0],[height,width],[0,width]])
        pts2 = np.float32(np.array([bbox[2],bbox[3],bbox[4],bbox[5],bbox[6],bbox[7],bbox[0],bbox[1]]).reshape(4,2))
        width,height = height,width
    else:
        pts1 = np.float32([[0,0],[width,0],[width,height],[0,height]])
        pts2 = np.float32(bbox.reshape(4,2))
    M = cv2.getPerspectiveTransform(pts2,pts1)
    dst = cv2.warpPerspective(image,M,(width,height))
    return dst