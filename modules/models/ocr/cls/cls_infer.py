#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : cls_infer.py
# @Time : 2025/6/14 14:57

from .. import OnnxInfer, OpenvinoInfer

class ClsOnnxInfer(OnnxInfer):
    def __init__(self, model_path, use_gpu = False):
        super().__init__(model_path, use_gpu)

    def cls_img(self, img):
        return None


class ClsOpenvinoInfer(OpenvinoInfer):
    def __init__(self, model_path):
        super().__init__(model_path)

    def cls_img(self, img):
        return None