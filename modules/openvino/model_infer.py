#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : model_infer.py
# @Time : 2025/6/14 13:56

import logging
from openvino.runtime import Core, Tensor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenvinoInfer():
    def __init__(self, model_path):

        core = Core()
        model = core.read_model(model = model_path)
        # 获取模型输入的信息
        self.inputs = model.inputs
        self.net = core.compile_model(model = model, device_name = "CPU")
    def async_infer(self, inputs):
        """
        进行异步推理。

        参数:
            inputs: 一个字典，键是输入名称（字符串），值是要传入的数据（numpy数组）。
        返回:
            推理结果。
        """
        infer_request = self.net.create_infer_request()
        # 设置多个输入数据
        for input_name in inputs.keys():
            if input_name in [input.any_name for input in self.inputs]:
                infer_request.set_tensor(input_name, Tensor(inputs[input_name]))

        # 启动异步推理
        infer_request.start_async()
        # 等待推理完成
        infer_request.wait()
        # 获取所有输出结果
        results = []
        for output in self.net.outputs:
            results.append(infer_request.get_tensor(output.any_name).data)

        return results