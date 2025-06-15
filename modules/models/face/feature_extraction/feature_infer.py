#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : feature_infer.py
# @Time : 2025/6/15 11:48

import numpy as np
from .. import OnnxInfer, OpenvinoInfer
from .feature_utils import normalize_img

class FeatureExtractOnnxInfer(OnnxInfer):
    def __init__(self, model_path:str, use_gpu:bool = False):
        """
        :param model_path: 模型的地址
        :param use_gpu: 是否使用GPU
        """
        super().__init__(model_path, use_gpu)
    def feature_img(self, img:np.array, target_size = (112, 112)):
        infer_img = normalize_img(img, target_size)
        onnx_inputs = {}
        onnx_inputs[self.input_names[0]] = infer_img
        out = self.infer(onnx_inputs)[0][0]
        return out

class FeatureExtractOpenvinoInfer(OpenvinoInfer):
    def __init__(self, model_path:str):
        """
        :param model_path: 模型的地址
        :param det_params: 检测的一些参数设置
        """
        super().__init__(model_path)

    def feature_img(self, img:np.array, target_size:tuple = (112, 112)):
        infer_img = normalize_img(img, target_size)
        openvino_inputs = {}
        openvino_inputs[self.inputs[0].any_name] = infer_img
        out = self.async_infer(openvino_inputs)[0][0]
        return out