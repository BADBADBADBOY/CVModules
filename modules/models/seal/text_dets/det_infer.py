#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : det_infer.py
# @Time : 2025/6/14 20:28

import numpy as np
from typing import List,Dict
from .det_config import base_det_params
from .. import OnnxInfer, OpenvinoInfer
from .det_utils import normalize_img, DBPostProcess

class DetOnnxInfer(OnnxInfer):
    def __init__(self, model_path:str, use_gpu:bool = False, det_params:Dict = None):
        """
        :param model_path: 模型的地址
        :param det_params: 检测的一些参数设置
        """
        super().__init__(model_path, use_gpu)
        if det_params is None:
            det_params = base_det_params
        self.db_post_process = DBPostProcess(det_params)

    def det_img(self, img:np.array, short_size:int = 736) -> (List[np.array],List[List[float]]):
        """
        :param img: 待检测的图片
        :param short_size:  图片最短边设置成short_size，长边同比resize
        :return bbox_batch: 检测出的文本框的坐标
        :return score_batch: 检测的文本框的置信度
        """
        infer_img = normalize_img(img, short_size)
        onnx_inputs = {}
        onnx_inputs[self.input_names[0]] = infer_img
        out = self.infer(onnx_inputs)[0]
        scale = (img.shape[1] * 1.0 / out.shape[3], img.shape[0] * 1.0 / out.shape[2])
        bbox_batch, score_batch = self.db_post_process(out, [scale])
        return bbox_batch, score_batch

class DetOpenvinoInfer(OpenvinoInfer):
    def __init__(self, model_path:str, det_params:Dict = None):
        """
        :param model_path: 模型的地址
        :param det_params: 检测的一些参数设置
        """
        super().__init__(model_path)
        if det_params is None:
            det_params = base_det_params
        self.db_post_process = DBPostProcess(det_params)

    def det_img(self, img: np.array, short_size: int = 736) -> (List[np.array], List[List[float]]):
        """
        :param img: 待检测的图片
        :param short_size:  图片最短边设置成short_size，长边同比resize
        :return bbox_batch: 检测出的文本框的坐标
        :return score_batch: 检测的文本框的置信度
        """
        infer_img = normalize_img(img, short_size)
        openvino_inputs = {}
        openvino_inputs[self.inputs[0].any_name] = infer_img
        out = self.async_infer(openvino_inputs)[0]
        scale = (img.shape[1] * 1.0 / out.shape[3], img.shape[0] * 1.0 / out.shape[2])
        bbox_batch, score_batch = self.db_post_process(out, [scale])
        return bbox_batch, score_batch