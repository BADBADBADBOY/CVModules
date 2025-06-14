#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : CVModules
# @File : model_infer.py
# @Time : 2025/6/14 14:00

import logging
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OnnxInfer(object):
    def __init__(self, model_path, use_gpu):
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2
        sess_options.log_severity_level = 0 ### 输出所有日志
        if use_gpu and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise """1.Make sure they have installed the GPU version of ONNX Runtime (onnxruntime-gpu) instead of the CPU-only version (onnxruntime).
                     2.Ensure that the version of ONNX Runtime is compatible with the installed CUDA version (e.g., some versions of onnxruntime-gpu are built against specific CUDA/cuDNN versions).
                  """
        if use_gpu:
            session = ort.InferenceSession(model_path, options=sess_options, providers=["CUDAExecutionProvider"])
            logging.info('Load GPU Model Success !!!')
        else:
            session = ort.InferenceSession(model_path, options=sess_options, providers=["CPUExecutionProvider"])

        self.session = session
        self.input_names = [i.name for i in session.get_inputs()]
        self.output_names = [o.name for o in session.get_outputs()]

    def infer(self, inputs):
        outputs = self.session.run(self.output_names, inputs)
        return outputs