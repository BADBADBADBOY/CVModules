#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : test_ocr_recog.py
# @Time : 2025/6/14 17:53

import cv2
import time
from modules import RecogOnnxInfer, RecogOpenvinoInfer

recog_dict_file = "../weights/ocrv5_mobile/recog/v5_dict.txt"
onnx_model_path = "../weights/ocrv5_mobile/recog/mobile_rec.onnx"
openvino_model_path = "../weights/ocrv5_mobile/recog/openvino/mobile_rec.xml"

test_time = 10
img = cv2.imread(r"../doc/imgs/recog_img.jpg")

onnx_recog_obj = RecogOnnxInfer(onnx_model_path, recog_dict_file)
onnx_time = 0
for i in range(test_time):
    t = time.time()
    recog_texts = onnx_recog_obj.recog_img(img)
    onnx_time += time.time() - t

openvino_recog_obj = RecogOpenvinoInfer(openvino_model_path, recog_dict_file)
openvino_time = 0
for i in range(test_time):
    t = time.time()
    recog_texts = openvino_recog_obj.recog_img(img)
    openvino_time += time.time() - t

print(recog_texts)
print("test {} , onnx cost time :{}s, openvino cost time:{}s".format(test_time, onnx_time/test_time, openvino_time/test_time))