#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : test_seal_text_det.py
# @Time : 2025/6/14 20:31

import cv2
import time
import numpy as np
from modules import SealDetOnnxInfer, SealDetOpenvinoInfer

onnx_model_path = "../weights/ocrv4_seal/mobile/mobile_v4_seal.onnx"
openvino_model_path = "../weights/ocrv4_seal/mobile/openvino/mobile_v4_seal.xml"

onnx_model_path = "../weights/ocrv4_seal/server/server_v4_seal.onnx"
openvino_model_path = "../weights/ocrv4_seal/server/openvino/server_v4_seal.xml"

test_time = 1
img = cv2.imread(r"../doc/imgs/seal_det.jpg")

onnx_det_obj = SealDetOnnxInfer(onnx_model_path, use_gpu = False)
onnx_time = 0
for i in range(test_time):
    t = time.time()
    bbox_batch, score_batch = onnx_det_obj.det_img(img, short_size = 640)
    onnx_time += time.time() - t

openvino_det_obj = SealDetOpenvinoInfer(openvino_model_path)
openvino_time = 0
for i in range(test_time):
    t = time.time()
    bbox_batch, score_batch = openvino_det_obj.det_img(img, short_size = 640)
    openvino_time += time.time() - t

for bbox in bbox_batch[0]:
    img = cv2.drawContours(img,[bbox.astype(np.int32)],-1,(0,255,255),2)
cv2.imwrite("./show.jpg", img)

print(bbox_batch, score_batch)
print("test {} , onnx cost time :{}s, openvino cost time:{}s".format(test_time, onnx_time/test_time, openvino_time/test_time))
