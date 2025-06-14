#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : __init__.py.py
# @Time : 2025/6/14 13:37

from .models.ocr.dets.det_infer import DetOnnxInfer, DetOpenvinoInfer
from .models.ocr.recogs.recog_infer import RecogOnnxInfer, RecogOpenvinoInfer