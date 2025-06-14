#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : test_ocr_det.py
# @Time : 2025/6/14 16:50

import cv2
import time
from modules import DetOnnxInfer, DetOpenvinoInfer

onnx_model_path = "../weights/ocrv5_mobile/dets/mobile_det.onnx"
openvino_model_path = "../weights/ocrv5_mobile/dets/openvino/mobile_det.xml"

test_time = 10
img = cv2.imread(r"../doc/imgs/det_img.jpg")

onnx_det_obj = DetOnnxInfer(onnx_model_path, use_gpu = False)
onnx_time = 0
for i in range(test_time):
    t = time.time()
    bbox_batch, score_batch = onnx_det_obj.det_img(img, short_size = 736)
    onnx_time += time.time() - t

openvino_det_obj = DetOpenvinoInfer(openvino_model_path)
openvino_time = 0
for i in range(test_time):
    t = time.time()
    bbox_batch, score_batch = openvino_det_obj.det_img(img, short_size = 736)
    openvino_time += time.time() - t

print(bbox_batch, score_batch)
print("test {} , onnx cost time :{}s, openvino cost time:{}s".format(test_time, onnx_time/test_time, openvino_time/test_time))
