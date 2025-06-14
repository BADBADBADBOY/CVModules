#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : recog_infer.py
# @Time : 2025/6/14 14:57

from .. import OnnxInfer, OpenvinoInfer
from .recog_utils import resize_norm_img, CTCLabelDecode

class RecogOnnxInfer(OnnxInfer):
    def __init__(self, model_path, recog_dict_file, use_gpu = False):
        super().__init__(model_path, use_gpu)
        self.decode_ctc = CTCLabelDecode(recog_dict_file, use_space_char = True)

    def recog_img(self, img):
        w_h_ratio = img.shape[1] / img.shape[0]
        infer_img = resize_norm_img(img, w_h_ratio)
        onnx_inputs = {}
        onnx_inputs[self.input_names[0]] = infer_img
        out = self.infer(onnx_inputs)[0]
        recog_texts = self.decode_ctc(out)[0]
        return recog_texts


class RecogOpenvinoInfer(OpenvinoInfer):
    def __init__(self, model_path, recog_dict_file):
        super().__init__(model_path)
        self.decode_ctc = CTCLabelDecode(recog_dict_file, use_space_char = True)

    def recog_img(self, img):
        w_h_ratio = img.shape[1] / img.shape[0]
        infer_img = resize_norm_img(img, w_h_ratio)
        openvino_inputs = {}
        openvino_inputs[self.inputs[0].any_name] = infer_img
        out = self.async_infer(openvino_inputs)[0]
        recog_texts = self.decode_ctc(out)[0]
        return recog_texts