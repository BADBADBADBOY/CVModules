#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : test_face_feature.py
# @Time : 2025/6/15 12:33

import cv2
import time
from itertools import combinations
from modules.common import cosine_similarity
from modules import FeatureExtractOnnxInfer, FeatureExtractOpenvinoInfer

# onnx_model_path = "../weights/facenet/mobile/mobile_face.onnx"
# openvino_model_path = "../weights/facenet/mobile/openvino/mobile_face.xml"

onnx_model_path = "../weights/facenet/server/resnet50_face.onnx"
openvino_model_path = "../weights/facenet/server/openvino/resnet50_face.xml"

test_time = 10

img1 = cv2.imread(r"../doc/imgs/face/lyf1.png")
img2 = cv2.imread(r"../doc/imgs/face/lyf2.png")
img3 = cv2.imread(r"../doc/imgs/face/bl1.png")
img4 = cv2.imread(r"../doc/imgs/face/bl2.png")
img5 = cv2.imread('../doc/imgs/face/lyt1.png')
img6 = cv2.imread('../doc/imgs/face/lyt2.png')
face_img = {"lyf1":img1,"lyf2":img2,"bl1":img3,"bl2":img4,"lyt1":img5,"lyt2":img6}

onnx_obj = FeatureExtractOnnxInfer(onnx_model_path, use_gpu = False)
onnx_time = 0
for i in range(test_time):
    t = time.time()
    # 提取所有人脸特征向量
    embeddings = {name: onnx_obj.feature_img(img) for name, img in face_img.items()}
    onnx_time += time.time() - t
    # 对所有人脸图像进行两两组合比较
    for (name1, embedding1), (name2, embedding2) in combinations(embeddings.items(), 2):
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Similarity between {name1} and {name2}: {similarity:.4f}")

openvino_obj = FeatureExtractOpenvinoInfer(openvino_model_path)
openvino_time = 0
for i in range(test_time):
    t = time.time()
    # 提取所有人脸特征向量
    embeddings = {name: openvino_obj.feature_img(img) for name, img in face_img.items()}
    openvino_time += time.time() - t
    # 对所有人脸图像进行两两组合比较
    for (name1, embedding1), (name2, embedding2) in combinations(embeddings.items(), 2):
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Similarity between {name1} and {name2}: {similarity:.4f}")

print("test {} , onnx cost time :{}s, openvino cost time:{}s".format(test_time, onnx_time/(6*test_time), openvino_time/(6*test_time)))
