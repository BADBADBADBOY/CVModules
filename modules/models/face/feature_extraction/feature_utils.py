#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : feature_utils.py
# @Time : 2025/6/15 11:48

import cv2
import numpy as np
def normalize_img(img: np.array, target_size:tuple = (112,112)):
    """
    对图像进行 resize、归一化等预处理操作。

    Args:
        img: 原始图像 (H, W, C)，np.uint8 类型，BGR 或 RGB 格式
        target_size: 目标尺寸 (height, width)

    Returns:
        preprocessed_img: float32 类型，形状为 (1, C, H, W) 的 batch 输入
    """
    if target_size:
        img = cv2.resize(img, target_size)  # resize 到目标大小

    img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    # 如果是 BGR 格式，先转成 RGB 再归一化（根据模型要求）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 如需转换颜色空间

    # 转换为 CHW 格式
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

    # 添加 batch 维度
    img = np.expand_dims(img, axis=0)  # CHW -> BCHW

    return img.astype(np.float32)
